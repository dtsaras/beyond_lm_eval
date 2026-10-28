"""topology_zigzag_persistence follows Gardinazzi et al. 2025 / ZigZagLLMs@2d10c07.

The reference extraction is re-typed below from
``src/representation/rep-extraction``: forward hooks on the modules
``get_target_layers(option="norm2")`` names (``*.layers.{i}.post_attention_layernorm``
plus the final norm, helpers.py:5-38), the last token of each prompt
(extract_activations.py:68-72) and prompts tokenised with
``add_special_tokens=False`` (dataset_utils.py:14-22).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "invariance"))
import _family_d as fd  # noqa: E402

pytest.importorskip("ripser")
pytest.importorskip("dionysus")

from blme.tasks.topology.zigzag import ZigzagPersistenceTask, _paper_clouds  # noqa: E402


def _reference_extract(model, tok, texts, names):
    mods = dict(model.named_modules())
    tmp, reps = {}, {n: [] for n in names}
    hooks = [mods[n].register_forward_hook(
        lambda m, i, o, n=n: tmp.__setitem__(n, o)) for n in names]
    try:
        with torch.no_grad():
            for t in texts:
                enc = tok(t.strip(), add_special_tokens=False, return_tensors="pt",
                          max_length=2048, truncation=True)
                model(**enc)
                T = int(enc["input_ids"].shape[1])
                for n in names:
                    reps[n].append(tmp[n][0, T - 1].double())
    finally:
        for h in hooks:
            h.remove()
    return [torch.stack(reps[n]).numpy() for n in names]


def _arch(name):
    if name == "llama":
        m = fd.tiny_llama(n_layers=3)
        names = [f"model.layers.{i}.post_attention_layernorm" for i in range(3)] + ["model.norm"]
    elif name.startswith("neox"):
        m = fd.tiny_neox(n_layers=3, parallel=(name == "neox_parallel"))
        names = ([f"gpt_neox.layers.{i}.post_attention_layernorm" for i in range(3)]
                 + ["gpt_neox.final_layer_norm"])
    else:
        m = fd.tiny_gpt2()
        L = len(m.transformer.h)
        names = [f"transformer.h.{i}.ln_2" for i in range(L)] + ["transformer.ln_f"]
    return m, names


@pytest.mark.parametrize("arch", ["llama", "neox_sequential", "neox_parallel", "gpt2"])
def test_clouds_are_the_reference_last_token_norm2_rows(arch):
    model, names = _arch(arch)
    tok = fd.tokenizer()
    clouds, _, info = _paper_clouds(model, tok, fd.TEXTS, window=2048)
    ref = _reference_extract(model, tok, fd.TEXTS, names)
    assert len(clouds) == len(names) == info["n_sites"]
    for a, b in zip(clouds, ref):
        np.testing.assert_array_equal(a, b)
    assert info["sites_are_norm_outputs"] is True


def test_task_reads_the_paper_sites_not_block_outputs():
    """Before the fix the task mean-pooled the L block outputs (with BOS kept);
    the paper reads L + 1 clouds: each post-attention norm and the final norm."""
    model = fd.tiny_llama(n_layers=3)
    tok = fd.tokenizer()
    res = ZigzagPersistenceTask({"num_samples": len(fd.TEXTS), "n_neighbors": 4}).evaluate(
        model, tok, fd.dataset())
    assert "error" not in res, res
    assert res["_meta_zz_n_clouds"] == 3 + 1
    assert res["_meta_zz_sites_are_norm_outputs"] is True
    assert res["_meta_zz_site_norm_module"] == "post_attention_layernorm"
    assert res["_meta_zz_n_docs_starting_with_bos"] == 0
    # The reference's other pooling is kept for comparison only.
    assert "_meta_meanpool_zz_true_adjacent_mean" in res


def test_site_check_flags_a_block_whose_mlp_reads_the_raw_residual():
    """OLMo-2-style blocks normalise sublayer OUTPUTS: the MLP input is no norm
    output, and the derived flag must say so instead of assuming it."""
    model = fd.tiny_llama(n_layers=2)
    for blk in model.model.layers:
        blk.post_attention_layernorm = torch.nn.Identity()
    _, _, info = _paper_clouds(model, fd.tokenizer(), fd.TEXTS[:4], window=2048)
    assert info["sites_are_norm_outputs"] is False
    assert info["site_norm_module"] is None
