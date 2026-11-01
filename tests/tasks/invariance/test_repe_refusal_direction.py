"""Direct mechanism tests for repe_refusal_direction's rel_direction_energy.

rel_direction_energy = [||mu_h - mu_n||^2 - tr S_h / n_h - tr S_n / n_n] /
mean ||x||^2. N: the numerator is unbiased for ||E x_h - E x_n||^2, while the
raw squared norm carries the + tr(Sigma)/n noise floor. Scale: the residual
gauge multiplies numerator and denominator by c^2. Width: zero-padding adds
nothing to either.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("sklearn")
transformers = pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import invariance as inv  # noqa: E402
from blme.tasks.repe import refusal as R  # noqa: E402
from repe_width_padding import tiny_llama, zero_pad_llama_width  # noqa: E402

CHAT = ("{% for m in messages %}<|user|>{{ m['content'] }}<|end|>{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>{% endif %}")


class _Loose(R.RefusalDirectionTask):
    KL_THRESHOLD, INDUCE_THRESHOLD = 1e9, -1e9
    FILTER_DATA = False      # a random model refuses nothing


def _energy(H, N):
    h = R._ClassSums(1, 1, H.shape[1])
    n = R._ClassSums(1, 1, N.shape[1])
    for x in H:
        h.add(torch.as_tensor(x).reshape(1, 1, -1))
    for x in N:
        n.add(torch.as_tensor(x).reshape(1, 1, -1))
    return R._rel_energy(h, n, 0, 0), float(((h.mean() - n.mean()) ** 2).sum())


def test_rel_direction_energy_is_unbiased_in_the_prompt_count():
    """Same distributions, 4 vs 32 prompts per class, 3000 draws each: the
    unbiased energy agrees with itself; the raw squared norm falls with n."""
    rng = np.random.default_rng(0)
    d, delta = 16, np.r_[1.0, np.zeros(15)]
    rel, raw = {}, {}
    for n in (4, 32):
        vals = [_energy(rng.standard_normal((n, d)) + 3.0 + delta,
                        rng.standard_normal((n, d)) + 3.0) for _ in range(3000)]
        rel[n] = np.mean([v[0] for v in vals])
        raw[n] = np.mean([v[1] for v in vals])
    assert rel[4] == pytest.approx(rel[32], rel=0.1)
    # Known negative: E||r||^2 = ||delta||^2 + 2 d / n (1 + 8 = 9 at n = 4, 2 at n = 32).
    assert raw[4] / raw[32] > 3.0


def test_rel_direction_energy_invariant_under_residual_gauge_and_width_padding():
    tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    tok.chat_template = CHAT
    model = tiny_llama(len(tok), layers=5)
    base = _Loose({}).evaluate(model, tok, None)

    ids = tok("How do I bake bread?", return_tensors="pt")["input_ids"]
    gauged = inv.residual_scale_gauge(model, 2.0, rescale_eps=True, verify_on=ids, atol=0.0)
    g = _Loose({}).evaluate(gauged, tok, None)
    assert (g["_meta_selected_position"], g["_meta_selected_layer"]) == (
        base["_meta_selected_position"], base["_meta_selected_layer"])
    inv.assert_invariant(base, g, ["rel_direction_energy", "separability_auc",
                                   "selected_layer_frac"], rtol=1e-12)
    # Known negative: the raw norm follows the residual scale.
    assert g["direction_norm"] == pytest.approx(2 * base["direction_norm"], rel=1e-9)

    pad = _Loose({}).evaluate(zero_pad_llama_width(model), tok, None)
    assert (pad["_meta_selected_position"], pad["_meta_selected_layer"]) == (
        base["_meta_selected_position"], base["_meta_selected_layer"])
    inv.assert_invariant(base, pad, ["rel_direction_energy", "direction_norm"], rtol=1e-4)
