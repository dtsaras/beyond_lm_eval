"""dynamics_perturbation_growth: divergence-profile numerics and evaluate() behaviour."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adversarial_fixtures import on_grid_translation, scale_ladder  # noqa: E402


def _halving_trajectory(s, n_layers=6):
    """One-coordinate divergence delta_l = s * 0.5**l: exact log ratio (L-1) ln 1/2."""
    clean = [torch.zeros(4, 3, dtype=torch.float64) for _ in range(n_layers)]
    pert = []
    for l in range(n_layers):
        p = torch.zeros(4, 3, dtype=torch.float64)
        p[:, 0] = s * 0.5 ** l
        pert.append(p)
    return clean, pert


@pytest.mark.parametrize("s", scale_ladder(reduction_depth=2, n_terms=4))
def test_growth_profile_exact_at_every_scale(s):
    from blme.tasks.dynamics.perturbation_growth import _growth_profile
    clean, pert = _halving_trajectory(s)
    out = _growth_profile([c.float() for c in clean], [p.float() for p in pert])
    assert out, f"no metric at scale {s}"
    assert out["growth_log_ratio"] == pytest.approx(5 * np.log(0.5), rel=1e-12)


def test_growth_profile_ignores_a_common_translation():
    from blme.tasks.dynamics.perturbation_growth import _growth_profile
    g = np.random.default_rng(0)
    v = g.standard_normal((20, 8))
    d = 1e-2 * g.standard_normal((20, 8))
    base_c, base_p, shifted_c, shifted_p = [], [], [], []
    for l in range(5):
        bc, tc, ok_c = on_grid_translation(v * 0.7 ** l, 1e3, dtype=np.float64)
        bp, tp, ok_p = on_grid_translation((v + d) * 0.7 ** l, 1e3, dtype=np.float64)
        assert ok_c and ok_p
        base_c.append(torch.from_numpy(bc)); base_p.append(torch.from_numpy(bp))
        shifted_c.append(torch.from_numpy(tc)); shifted_p.append(torch.from_numpy(tp))
    a = _growth_profile(base_c, base_p)
    b = _growth_profile(shifted_c, shifted_p)
    assert b["growth_log_ratio"] == pytest.approx(a["growth_log_ratio"], rel=1e-12)


def test_relative_form_removes_state_norm_growth():
    """A stack that rescales the state by a per layer moves the divergence and the
    state alike: the absolute log ratio is L ln a, the relative one is 0."""
    import torch
    from blme.tasks.dynamics.perturbation_growth import _growth_profile
    g = torch.Generator().manual_seed(0)
    h0 = torch.randn(40, 8, generator=g, dtype=torch.float64)
    p0 = h0 + 1e-2 * torch.randn(40, 8, generator=g, dtype=torch.float64)
    a, L = 1.7, 5
    clean = [h0 * a ** l for l in range(L + 1)]
    pert = [p0 * a ** l for l in range(L + 1)]
    out = _growth_profile(clean, pert)
    assert out["growth_log_ratio"] == pytest.approx(L * np.log(a), rel=1e-10)
    assert out["growth_log_ratio_relative"] == pytest.approx(0.0, abs=1e-12)


def test_per_layer_token_embeddings_preserve_ids_across_the_perturbation(monkeypatch):
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
    from blme.tasks.dynamics.perturbation_growth import PerturbationGrowthTask

    tok = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(tok), n_layer=2, n_head=2, n_embd=16)).eval()
    original = model.forward
    calls = []
    def lookup(ids, embeddings):
        assert ids is not None
        clean = model.get_input_embeddings()(ids)
        torch.testing.assert_close(embeddings, clean)
        calls.append(("ids", ids.clone()))
        return ids.unsqueeze(-1)
    def forward(*args, inputs_embeds=None, per_layer_inputs=None, **kwargs):
        assert per_layer_inputs is not None
        calls.append(("forward", (per_layer_inputs.clone(), inputs_embeds.clone())))
        return original(*args, inputs_embeds=inputs_embeds, **kwargs)
    monkeypatch.setattr(model.transformer, "hidden_size_per_layer_input", 1, raising=False)
    monkeypatch.setattr(model.transformer, "get_per_layer_inputs", lookup, raising=False)
    monkeypatch.setattr(model, "forward", forward)
    out = PerturbationGrowthTask({"num_samples": 1}).evaluate(model, tok, [{"text": "A small token sequence with content."}])
    assert np.isfinite(out["growth_log_ratio_relative"])
    assert [k for k, _ in calls] == ["ids", "forward", "forward"]
    clean, perturbed = (calls[i][1] for i in (1, 2))
    assert torch.equal(clean[0], perturbed[0]) and torch.equal(clean[0].squeeze(-1), calls[0][1])
    assert not torch.equal(clean[1], perturbed[1])
