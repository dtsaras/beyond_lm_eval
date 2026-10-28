"""Parity: repe_refusal_direction vs the RELEASED andyrdt/refusal_direction code.

Runs ``pipeline.submodules.generate_directions.get_mean_diff`` and
``pipeline.submodules.select_direction.select_direction`` (@9d852fa) on a tiny
Llama with a chat template, batch_size=1 (no padding), with the same prompts,
template, post-instruction tokens and refusal tokens BLME uses. The candidate
directions, the selected (position, layer) and the selection scores must
agree. ``jaxtyping`` (annotations only) is stubbed when absent. Skips when the
clone is absent (fails under BLME_REQUIRE_REFS=1).
"""
import json
import os
import sys
import types

import numpy as np
import pytest
import torch

pytest.importorskip("sklearn")
pytest.importorskip("matplotlib")
pytest.importorskip("einops")
transformers = pytest.importorskip("transformers")

_REPO = os.path.join(os.environ.get("BLME_REFREPOS", os.path.expanduser("~/.cache/blme-refrepos")),
                     "refusal_direction")
CHAT = ("{% for m in messages %}<|user|>{{ m['content'] }}<|end|>{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>{% endif %}")


@pytest.fixture(scope="module")
def ref():
    if not os.path.isdir(os.path.join(_REPO, "pipeline")):
        if os.environ.get("BLME_REQUIRE_REFS") == "1":
            pytest.fail(f"reference not available: {_REPO}")
        pytest.skip(f"reference not available: {_REPO}")
    if "jaxtyping" not in sys.modules:
        try:
            import jaxtyping  # noqa: F401
        except ImportError:
            class _Ann:
                def __getitem__(self, item):
                    return object
            stub = types.ModuleType("jaxtyping")
            stub.Float, stub.Int = _Ann(), _Ann()
            sys.modules["jaxtyping"] = stub
    import matplotlib
    matplotlib.use("Agg")
    sys.path.insert(0, _REPO)
    try:
        from pipeline.submodules.generate_directions import get_mean_diff
        from pipeline.submodules.select_direction import select_direction
    finally:
        sys.path.remove(_REPO)
    return get_mean_diff, select_direction


@pytest.fixture(scope="module")
def setup():
    tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    tok.chat_template = CHAT
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=5,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
        attn_implementation="eager")
    return transformers.LlamaForCausalLM(cfg).eval(), tok


def test_selection_matches_released_pipeline(ref, setup, tmp_path):
    from blme.tasks.common import get_layers
    from blme.tasks.repe import refusal as R

    get_mean_diff, select_direction = ref
    model, tok = setup
    h_tr, n_tr, h_va, n_va, _ = R._prompt_sets(None, {})
    layers = get_layers(model)
    eoi = R._eoi_ids(tok)

    def tokenize_fn(instructions):
        assert len(instructions) == 1
        ids = torch.tensor([tok.encode(R._format(tok, instructions[0]), add_special_tokens=False)])
        return types.SimpleNamespace(input_ids=ids, attention_mask=torch.ones_like(ids))

    # select_direction uses the tokenizer only to label its plots, one token per
    # position (batch_decode of a flat id list no longer does that).
    labels_tok = types.SimpleNamespace(batch_decode=lambda ids: [tok.decode([i]) for i in ids])
    base = types.SimpleNamespace(
        model=model, tokenizer=labels_tok, tokenize_instructions_fn=tokenize_fn, eoi_toks=eoi,
        refusal_toks=R._refusal_token_ids(tok), model_block_modules=layers,
        model_attn_modules=[b.self_attn for b in layers], model_mlp_modules=[b.mlp for b in layers])

    with torch.no_grad():
        mean_diffs = get_mean_diff(model, tok, h_tr, n_tr, tokenize_fn, layers,
                                   batch_size=1, positions=list(range(-len(eoi), 0)))
    # Loose thresholds so a random model keeps candidates; the rule is the same.
    kl_t, ind_t = 1e9, -1e9
    pos, layer, direction = select_direction(base, h_va, n_va, mean_diffs, str(tmp_path),
                                             kl_threshold=kl_t, induce_refusal_threshold=ind_t,
                                             prune_layer_percentage=0.2, batch_size=1)
    scores = {(d["position"], d["layer"]): d for d in json.load(
        open(tmp_path / "direction_evaluations.json"))}

    class Loose(R.RefusalDirectionTask):
        KL_THRESHOLD, INDUCE_THRESHOLD = kl_t, ind_t

    out = Loose({}).evaluate(model, tok, None)
    assert (out["_meta_selected_position"], out["_meta_selected_layer"]) == (pos, layer)
    assert out["direction_norm"] == pytest.approx(float(direction.norm()), rel=1e-9)
    s = scores[(pos, layer)]
    assert out["refusal_score_bypass"] == pytest.approx(s["refusal_score"], rel=1e-5)
    assert out["refusal_score_induce"] == pytest.approx(s["steering_score"], rel=1e-5)
    assert out["_meta_kl_ablation"] == pytest.approx(s["kl_div_score"], rel=1e-4, abs=1e-9)
    # Layers at or past 0.8 L were never candidates.
    assert layer < int(0.8 * len(layers))
    # The selection is discriminating on this fixture (scores differ).
    assert len({round(d["refusal_score"], 6) for d in scores.values()}) > 3
