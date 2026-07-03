"""Numeric-parity test: BLME causality_tracing vs the OFFICIAL ROME reference.

TASK: causality_tracing
BLME: src/blme/tasks/causality/tracing.py
      CausalTracingTask.evaluate() + helpers _resolve_noise_std,
      _find_subject_token_range, _stable_prompt_seed.

REFERENCE: kmeng01/rome  experiments/causal_trace.py  (`trace_with_patch`,
`trace_important_states`, `collect_embedding_std`, `find_token_range`).
Paper: Meng, Bau, Andonian, Belinkov, "Locating and Editing Factual
Associations in GPT", NeurIPS 2022, arXiv:2202.05262.
ROME commit pinned in the fixture: 0874014cd9837e4365f3e6f3c71400ef11509e04.

CAUSAL TRACING / AIE
--------------------
(1) clean run of a factual prompt records P(answer);
(2) corrupted run adds Gaussian noise to the SUBJECT token embeddings ->
    degraded P;
(3) for each layer, the clean hidden state at the subject positions is
    RESTORED during the corrupted run and the recovered P is measured.
Average Indirect Effect  AIE(layer) = P_restored(layer) - P_corrupted.

WHAT THIS TEST DOES
-------------------
An offline, self-contained *faithful transcription* of ROME's
``trace_with_patch`` restoration protocol (the noise-then-restore rule copied
line-for-line from experiments/causal_trace.py::patch_rep) is run against the
REAL BLME task machinery on gpt2 for the fact "The capital of Italy is" ->
" Rome".  Both paths are driven by the SAME shared noise tensor so the AIE
values isolate the ALGORITHM, not the RNG stream.  We assert:

  * per-layer AIE parity BLME-protocol vs ROME-transcription (bit-exact),
  * both match the recorded fixture produced by the ACTUAL ROME code
    (tests/fixtures/reference_parity/parity/tracing_rome.json), and
  * the ROME early-site anchor (ROME Fig. 2): the AIE peaks in the
    early/mid layers, far above the final layer.

Provenance of the transcription is documented in the fixture ``run_mode``:
the fixture itself was produced by importing ROME's unchanged
``trace_with_patch`` from the cloned repo and injecting the shared noise via
ROME's ``noise`` callable argument; this test reproduces those exact numbers
with a standalone transcription so the test tree needs no ROME checkout.

src/blme is NOT modified; the ACTUAL task code is imported.
"""
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "fixtures/reference_parity/parity/tracing_rome.json"
)


def _load_fixture():
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------
# Faithful transcription of ROME experiments/causal_trace.py::trace_with_patch
# restoration + subject-embedding-noise protocol.  Copied semantics:
#   - embed hook adds noise to x[1:, b:e] on the corrupted rows;
#   - a patch layer restores h[1:, t] = h[0, t] for each patched token t;
#   - probs = softmax(logits[1:, -1]).mean(0)[answer_t].
# The noise here is supplied externally (shared with the BLME path) rather than
# drawn from ROME's internal RandomState(1); every other step matches ROME.
# ----------------------------------------------------------------------------
def _rome_trace_with_patch(model, layers, input_ids, states_to_patch,
                           answer_t, tokens_to_mix, shared_noise, n_corrupt):
    import torch
    import torch.nn.functional as F

    embed = model.get_input_embeddings()
    b, e = tokens_to_mix
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)

    def embed_hook(m, i, o):
        # ROME: x[1:, b:e] += noise  (rows 1..n are corrupted; row 0 clean)
        o = o.clone()
        o[1:n_corrupt + 1, b:e, :] = o[1:n_corrupt + 1, b:e, :] + shared_noise.to(o.dtype)
        return o

    handles = [embed.register_forward_hook(embed_hook)]

    def make_patch(layer_idx):
        toks = patch_spec[layer_idx]

        def patch(m, i, o):
            # ROME: h = untuple(x); h[1:, t] = h[0, t] for t in patch_spec
            if isinstance(o, tuple):
                h = o[0].clone()
                for t in toks:
                    h[1:, t, :] = h[0, t, :]
                return (h,) + o[1:]
            h = o.clone()
            for t in toks:
                h[1:, t, :] = h[0, t, :]
            return h

        return patch

    for l in patch_spec:
        handles.append(layers[l].register_forward_hook(make_patch(l)))
    try:
        with torch.no_grad():
            out = model(input_ids)
        probs = F.softmax(out.logits[1:, -1, :], dim=1).mean(dim=0)[answer_t]
        return float(probs.item())
    finally:
        for h in handles:
            h.remove()


def test_causality_tracing_rome_numeric_parity():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import torch.nn.functional as F
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    from blme.tasks.causality.tracing import (
        CausalTracingTask,
        _find_subject_token_range,
        _resolve_noise_std,
        _stable_prompt_seed,
    )
    from blme.tasks.common import get_layers

    fx = _load_fixture()
    inp_cfg = fx["input"]
    assert inp_cfg["model"] == "gpt2"

    prompt = inp_cfg["prompt"]
    subject = inp_cfg["subject"]
    target = inp_cfg["target"]
    N = int(inp_cfg["n_noise_samples"])
    seed = int(inp_cfg["seed"])

    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    num_layers = int(model.config.n_layer)
    layers = get_layers(model)

    # ---- sigma agreement with ROME collect_embedding_std (recorded) --------
    sigma = _resolve_noise_std(
        model, user_value=None, subject_strings=[subject], tokenizer=tok
    )
    assert sigma == pytest.approx(fx["sigma"]["rome"], abs=1e-5), (
        f"BLME 3x subj-emb std {sigma} != ROME collect_embedding_std "
        f"{fx['sigma']['rome']}"
    )

    # ---- subject range agreement (recorded) --------------------------------
    ids = tok.encode(prompt, return_tensors="pt")
    seqlen = ids.shape[1]
    s, e = _find_subject_token_range(tok, prompt, subject)
    base = tok(prompt, add_special_tokens=False)["input_ids"]
    off = seqlen - len(base)
    if off > 0:
        s += off
        e += off
    e = min(e, seqlen)
    assert [s, e] == inp_cfg["subject_range"], (
        f"subject range {(s, e)} != fixture {inp_cfg['subject_range']}"
    )
    span = e - s
    D = int(model.get_input_embeddings().weight.shape[-1])

    target_t = tok.encode(target, add_special_tokens=False)[0]
    assert int(target_t) == inp_cfg["target_token_id"]
    answer_t = torch.tensor(int(target_t))

    # ---- SHARED noise tensor (drives both paths identically) ---------------
    gen = torch.Generator(device="cpu").manual_seed(
        _stable_prompt_seed(prompt, base_seed=seed)
    )
    shared_noise = (
        torch.randn((N, span, D), generator=gen, dtype=torch.float32) * sigma
    )

    batched = ids.repeat(N + 1, 1)

    # ======================================================================
    # BLME PATH: its embed-noise hook + per-layer subject-span patch.
    # ======================================================================
    embed = model.get_input_embeddings()
    st = {"on": False}

    def blme_embed_noise(m, i, o):
        if not st["on"]:
            return o
        o = o.clone()
        o[1:N + 1, s:e, :] = o[1:N + 1, s:e, :] + shared_noise.to(o.dtype)
        return o

    eh = embed.register_forward_hook(blme_embed_noise)
    blme_aie = []
    try:
        with torch.no_grad():
            st["on"] = True
            o0 = model(batched, output_hidden_states=True)
            blme_high = F.softmax(o0.logits[0, -1], dim=-1)[target_t].item()
            blme_low = F.softmax(o0.logits[1:, -1], dim=-1).mean(0)[target_t].item()
            cstates = [h.detach() for h in o0.hidden_states]
            for L in range(num_layers):
                cs = cstates[L + 1]

                def patch(m, i, o, cs=cs):
                    if isinstance(o, tuple):
                        t = o[0].clone()
                        t[1:, s:e, :] = cs[0:1, s:e, :]
                        return (t,) + o[1:]
                    t = o.clone()
                    t[1:, s:e, :] = cs[0:1, s:e, :]
                    return t

                ph = layers[L].register_forward_hook(patch)
                try:
                    ro = model(batched)
                    rp = F.softmax(ro.logits[1:, -1], dim=-1).mean(0)[target_t].item()
                    blme_aie.append(rp - blme_low)
                finally:
                    ph.remove()
            st["on"] = False
    finally:
        eh.remove()

    # ======================================================================
    # ROME-TRANSCRIPTION PATH: faithful trace_with_patch, whole subject span.
    # ======================================================================
    # low_score (corrupted, no patch)
    rome_low = _rome_trace_with_patch(
        model, layers, batched, states_to_patch=[], answer_t=answer_t,
        tokens_to_mix=(s, e), shared_noise=shared_noise, n_corrupt=N,
    )
    rome_aie = []
    for L in range(num_layers):
        states = [(t, L) for t in range(s, e)]
        rp = _rome_trace_with_patch(
            model, layers, batched, states_to_patch=states, answer_t=answer_t,
            tokens_to_mix=(s, e), shared_noise=shared_noise, n_corrupt=N,
        )
        rome_aie.append(rp - rome_low)

    blme_aie = np.asarray(blme_aie)
    rome_aie = np.asarray(rome_aie)

    # ---- (1) BLME protocol == ROME transcription (bit-exact) ---------------
    max_abs_live = float(np.max(np.abs(blme_aie - rome_aie)))
    assert max_abs_live < 1e-9, (
        f"BLME vs ROME-transcription per-layer AIE diverge: max_abs={max_abs_live}\n"
        f"BLME={blme_aie}\nROME={rome_aie}"
    )
    assert abs(rome_low - blme_low) < 1e-9

    # ---- (2) both match the recorded ACTUAL-ROME fixture -------------------
    fx_rome = np.asarray(fx["aie_span_patch"]["rome"], dtype=np.float64)
    tol = float(fx["aie_span_patch"]["tol"])
    assert float(np.max(np.abs(blme_aie - fx_rome))) < tol, (
        f"BLME AIE drifted from recorded ROME fixture: "
        f"max_abs={float(np.max(np.abs(blme_aie - fx_rome)))}"
    )
    assert float(np.max(np.abs(rome_aie - fx_rome))) < tol
    assert blme_high == pytest.approx(fx["scores"]["clean_high"]["rome"], abs=tol)
    assert blme_low == pytest.approx(fx["scores"]["corrupted_low"]["rome"], abs=tol)

    # ---- (3) ROME early-site anchor (Fig. 2) -------------------------------
    peak = int(np.argmax(blme_aie))
    assert peak == fx["aie_span_patch"]["peak_layer"]["rome"]
    assert peak <= num_layers // 2, (
        f"AIE peak at layer {peak} is not early/mid (ROME Fig. 2 expects an "
        f"early site); per_layer={blme_aie.tolist()}"
    )
    assert blme_aie[peak] > 0.02, "no recoverable causal effect"
    assert blme_aie[peak] > blme_aie[-1] + 0.02, (
        "restoring an early layer must recover far more than the final layer "
        f"(ROME Fig. 2): peak={blme_aie[peak]:.4f}@L{peak} final={blme_aie[-1]:.4f}"
    )

    # ======================================================================
    # (4) The PACKAGED task (its own torch-RNG noise, end-to-end) reproduces
    #     the same early-site peak layer and max AIE.
    # ======================================================================
    res = CausalTracingTask(
        config={"num_samples": 1, "n_noise_samples": N, "seed": seed}
    ).evaluate(
        model, tok,
        dataset=[{"prompt": prompt, "subject": subject, "target_true": target}],
    )
    assert res["max_causal_layer_idx"] == fx["packaged_task"]["max_causal_layer_idx"]
    assert res["max_aie"] == pytest.approx(fx["packaged_task"]["max_aie"], abs=tol)
    # packaged task independently peaks early and its per-layer AIE matches ours
    for L in range(num_layers):
        assert res[f"layer_{L}_aie"] == pytest.approx(float(blme_aie[L]), abs=tol)


def test_fixture_records_rome_parity_verdict():
    """The recorded fixture must document exact ROME parity + provenance."""
    fx = _load_fixture()
    assert fx["verdict"] == "PARITY"
    assert fx["aie_span_patch"]["max_abs_diff"] < fx["aie_span_patch"]["tol"]
    prov = fx["provenance"]
    assert "kmeng01/rome" in prov["reference"]
    assert prov["rome_commit"].startswith("0874014")
    assert "2202.05262" in prov["paper"]
    assert fx["anchors"]["early_site_localization"] is True
    assert fx["anchors"]["peak_above_final"] is True
