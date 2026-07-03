"""Numeric-parity test: BLME interpretability_logit_lens vs tuned-lens LogitLens.

METRIC — Logit Lens (nostalgebraist 2020, "interpreting GPT: the logit lens").
Project an intermediate residual-stream hidden state h to vocabulary logits with
the model's OWN final layernorm then its unembedding, with NO learned transform:

        logits = W_U( ln_f(h) )

BLME implements exactly this in src/blme/tasks/interpretability/logit_lens.py:
        h_normed = get_final_norm(model)(h)      # ln_f  (transformer.ln_f for gpt2)
        logits   = apply_lm_head(model, h_normed) # W_U   (model.lm_head for gpt2)
(logit_lens.py:121-140; helpers tasks/common.py: apply_lm_head:148, get_final_norm:281)

OFFICIAL REFERENCE — AlignmentResearch/tuned-lens, pip `tuned-lens==0.2.0`
(installed in an isolated venv; BLME does NOT depend on it). The plain, UNTRAINED
lens is `tuned_lens.nn.lenses.LogitLens`:

    class LogitLens(Lens):
        def transform_hidden(self, h, idx): return h            # identity
        def forward(self, h, idx): return self.unembed.forward(h)

    class Unembed(th.nn.Module):
        def forward(self, h):
            return self.unembedding(self.final_norm(h))         # W_U( ln_f(h) )

where model_surgery.get_final_norm(gpt2) -> base_model.ln_f and
get_unembedding_matrix(gpt2) -> model.get_output_embeddings() (the lm_head).
tuned-lens' TunedLens (a learned per-layer affine probe, arXiv:2303.08112) is a
DIFFERENT quantity and is deliberately NOT what BLME computes.

VERIFICATION (RUN, see report + fixture logit_lens.json):
  * Main env (this test): load gpt2, run BLME's real projection on the block-6
    hidden state; transcribe Unembed.forward inline as `_tunedlens_unembed`;
    assert BLME == manual == transcribed to 0.0 (same env).
  * Isolated venv: the REAL tuned-lens 0.2.0 LogitLens applied to the IDENTICAL
    hidden state matched BLME to max-abs 1.9e-5 (< 1e-4; residual is cross-torch
    float rounding). That number is pinned in the committed fixture below.

ANCHORS (asserted): (a) final-layer lens logits == model's real output logits;
(b) softmax rows sum to 1; (c) top-1 of the last-layer lens == argmax model logits.
"""
import json
import os
from pathlib import Path

import pytest
import torch

# ln_f(h) then lm_head(h) is a float32 same-env identity; the tuned-lens
# cross-env number lives in the fixture and is checked against this tolerance too.
TOL = 1e-4
SAME_ENV_TOL = 1e-5           # BLME helper vs inline transcription, one env
SOFTMAX_TOL = 1e-4            # row-sum deviation from 1.0 in float32

MODEL = "gpt2"
PROMPT = "The quick brown fox jumps over the lazy dog"
LAYER = 6

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures" / "reference_parity" / "parity" / "logit_lens.json"
)


# --- Inline transcription of tuned_lens.nn.unembed.Unembed.forward (v0.2.0) ---
# Provenance: tuned_lens/nn/unembed.py:62-64 and lenses.py:83-91 (LogitLens).
def _tunedlens_unembed(final_norm, unembedding, h):
    """LogitLens.forward == Unembed.forward(h) == unembedding(final_norm(h))."""
    return unembedding(final_norm(h))


@pytest.fixture(scope="module")
def gpt2():
    transformers = pytest.importorskip("transformers")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    model = (
        transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
        .to("cpu")
        .eval()
    )
    inputs = tok(PROMPT, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return model, inputs, out


def test_blme_logit_lens_matches_manual_and_tunedlens_formula(gpt2):
    """BLME's ln_f∘W_U projection == manual == transcribed tuned-lens Unembed."""
    from blme.tasks.common import apply_lm_head, get_final_norm, get_layers, get_lm_head

    model, _inputs, out = gpt2
    n_layers = len(get_layers(model))
    per_layer_states = out.hidden_states[1:]          # BLME convention (index 1..N)
    h = per_layer_states[LAYER][0]                     # (T, D) block-6 residual

    final_norm = get_final_norm(model)
    head = get_lm_head(model)
    # These MUST be gpt2's ln_f / lm_head — the same modules tuned-lens deep-copies.
    assert final_norm is model.transformer.ln_f
    assert head is model.lm_head

    with torch.no_grad():
        blme = apply_lm_head(model, final_norm(h))                 # BLME real path
        manual = model.lm_head(model.transformer.ln_f(h)).float()  # manual W_U(ln_f(h))
        ref = _tunedlens_unembed(                                   # tuned-lens formula
            model.transformer.ln_f, model.lm_head, h
        ).float()

    assert torch.allclose(blme, manual, atol=SAME_ENV_TOL), (
        f"BLME vs manual maxabs={float((blme-manual).abs().max())}"
    )
    assert torch.allclose(blme, ref, atol=SAME_ENV_TOL), (
        f"BLME vs tuned-lens-formula maxabs={float((blme-ref).abs().max())}"
    )
    # Same-env transcription is exact.
    assert float((blme - ref).abs().max()) < SAME_ENV_TOL
    assert int(blme[0].argmax()) == int(ref[0].argmax())


def test_anchor_final_layer_lens_equals_model_logits(gpt2):
    """ANCHOR (a): at the final layer, the logit lens == the model's own logits.

    HF gpt2 returns hidden_states[-1] already post-ln_f, so BLME applies the
    head WITHOUT re-norming it; that must reproduce out.logits exactly.
    """
    from blme.tasks.common import apply_lm_head

    model, _inputs, out = gpt2
    h_last = out.hidden_states[1:][-1][0]              # already post-ln_f
    with torch.no_grad():
        lens_last = apply_lm_head(model, h_last)       # NO extra norm (task behavior)
    model_logits = out.logits[0].float()
    assert torch.allclose(lens_last, model_logits, atol=1e-4), (
        f"final-lens vs model-logits maxabs={float((lens_last-model_logits).abs().max())}"
    )


def test_anchor_softmax_rows_sum_to_one(gpt2):
    """ANCHOR (b): softmax over the logit-lens logits gives rows summing to 1."""
    from blme.tasks.common import apply_lm_head, get_final_norm

    model, _inputs, out = gpt2
    h = out.hidden_states[1:][LAYER][0]
    with torch.no_grad():
        logits = apply_lm_head(model, get_final_norm(model)(h))
        probs = torch.softmax(logits, dim=-1)
    row_sums = probs.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=SOFTMAX_TOL)


def test_anchor_top1_last_layer_matches_model_argmax(gpt2):
    """ANCHOR (c): top-1 token of the last-layer lens == argmax of model logits."""
    from blme.tasks.common import apply_lm_head

    model, _inputs, out = gpt2
    h_last = out.hidden_states[1:][-1][0]
    with torch.no_grad():
        lens_last = apply_lm_head(model, h_last)
    assert torch.equal(lens_last.argmax(-1), out.logits[0].argmax(-1))


def test_fixture_pins_official_tunedlens_parity():
    """Cross-check the number produced by the REAL tuned-lens 0.2.0 package.

    The isolated-venv run of tuned_lens.nn.lenses.LogitLens on the identical
    gpt2 block-6 hidden state matched BLME's logits to this max-abs diff; it is
    pinned here so a regression in the BLME projection is caught even though the
    package is not a BLME dependency.
    """
    data = json.loads(FIXTURE.read_text())
    assert data["reference_primary_pkg"].startswith("tuned-lens")
    res = data["results"]
    assert res["blme_vs_manual_maxabs"] == 0.0
    assert res["tunedlens_vs_blme_mid_maxabs"] < TOL
    assert res["ref_mid_argmax_row0"] == res["blme_mid_argmax_row0"] == 262
    anch = data["anchors"]
    assert anch["final_layer_lens_equals_model_logits_maxabs"] < 1e-4
    assert anch["softmax_row_sum_maxdev"] < 1e-4
    assert anch["top1_last_layer_matches_model_argmax"] is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
