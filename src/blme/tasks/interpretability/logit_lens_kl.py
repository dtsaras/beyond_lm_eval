"""``interpretability_logit_lens_kl``: when, through depth, the next-token
distribution settles.

Every block output h_l (l = 0..L-1, the last one hooked before the final norm)
is decoded with the model's own final norm and unembedding, final logit softcap
included: p_l = softmax(W_U norm(h_l)), so p_{L-1} is the model's output
distribution. Per token position, then averaged over the positions that are not
attention sinks:

* ``kl_to_final_profile[l]`` = KL(p_l || p_{L-1}) and its normalized-depth AUC
  ``kl_to_final_auc`` = (1/L) sum_l KL(p_l || p_{L-1});
* ``kl_churn`` = sum_l KL(p_l || p_{l+1}), the distribution change summed along
  depth;
* ``decision_depth`` = (s + 1) / L, s the first layer from which the final top-1
  stays the lens's top-1, and ``first_agreement_depth`` = (f + 1) / L, f the
  first layer at which it is;
* ``entropy_profile`` and ``entropy_auc`` = (1/L) sum_l H(p_l).

Layers, decode and the AUC / time-to-decision conventions follow
jennhu/model-human-processing@69e405c (Hu, Lepori & Franke, arXiv:2504.14107):
``src/model.py`` decodes the L block outputs (no embedding output) as
``lm_head(norm(h)).log_softmax(-1)``; ``analysis/notebooks/0_process_lm_data.ipynb``
sums a per-layer measure over layers (``get_auc``) and sets time to decision to
``(layer_stay_pos(first_logprob_diff) + 1) / n_layers``, the layer from which
log p(correct) - log p(incorrect) stays >= 0. On free text the final top-1 is
the "correct" answer and the strongest other token of each layer the
"incorrect" one.
"""

import contextlib
import logging
import math
from typing import List

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from .._precision import floating_dtype_summary, fp32_compute
from .._tokens import (middle_depth_layers, select_rows, sink_keep_masks, sink_meta,
                       strip_trailing_special)
from ..common import (apply_lm_head, get_embeddings, get_final_norm, get_layers, get_lm_head,
                      get_max_positions)

logger = logging.getLogger("blme")

DEFAULT_NUM_SAMPLES = 100
DEFAULT_MAX_LENGTH = 512
# Positions x vocabulary elements decoded at once: 2**22 float64 = 32 MiB per
# array, about eight alive (262k vocabulary -> 16 positions per chunk).
DEFAULT_CHUNK_ELEMENTS = 2 ** 22
_INV = "tests/tasks/invariance/test_interpretability_logit_lens_kl.py::"


def _block_outputs(model, layers, enc):
    """Raw output of every block in one forward (the last block's before the
    final norm), and the model's own logits."""
    outs: List = [None] * len(layers)

    def _capture(i):
        def hook(module, inputs, output):
            outs[i] = (output[0] if isinstance(output, tuple) else output).detach()
        return hook

    handles = [layer.register_forward_hook(_capture(i)) for i, layer in enumerate(layers)]
    try:
        out = model(**enc)
    finally:
        for h in handles:
            h.remove()
    return outs, getattr(out, "logits", None)


def _lens_logits(model, final_norm, h):
    """float32 lens logits of block outputs ``h`` (n, D): final norm, then
    ``apply_lm_head`` (unembedding and the model's final logit softcap)."""
    if final_norm is not None:
        p = next(final_norm.parameters(), None)
        h = final_norm(h.to(p.device if p is not None else h.device, torch.float32))
    return apply_lm_head(model, h.float())


def _plogp_sum(p, a):
    """sum_v p_v a_v with 0 * a = 0, so 0 log 0 = 0 and -inf logits are exact."""
    return torch.where(p > 0, p * a, torch.zeros((), dtype=p.dtype, device=p.device)).sum(-1)


def _margin(lp, top):
    """log p[y*] - max_{v != y*} log p[v]; >= 0 exactly when the final top-1 y*
    is a top-1 of this layer (a tie counts, as the release's ``>= 0`` does)."""
    v, i = lp.topk(2, dim=-1)
    own = lp.gather(-1, top).squeeze(-1)
    return torch.where(i[:, 0] == top.squeeze(-1), v[:, 0] - v[:, 1], own - v[:, 0])


def _doc_rows(model, final_norm, states, model_logits, chunk):
    """Statistics of positions 1..T-1 of one document, float64 (T-1, 5L-1):
    KL(p_l || p_final) | KL(p_final || p_l) | H(p_l) | margin_l (L columns
    each) | KL(p_l || p_{l+1}) (L-1 columns). Also the largest |lens - model|
    final-layer logit and the dtypes the decode ran in."""
    L = len(states)
    T = int(states[0].shape[0])
    rows = np.empty((T - 1, 5 * L - 1), dtype=np.float64)
    worst = 0.0
    dtypes = (None, None)
    for a in range(1, T, chunk):
        b = min(T, a + chunk)
        r = slice(a - 1, b - 1)
        z_fin = _lens_logits(model, final_norm, states[-1][a:b])
        if model_logits is not None and model_logits.shape[-1] == z_fin.shape[-1]:
            d = (z_fin - model_logits[a:b].to(z_fin.device, torch.float32)).abs().max()
            worst = max(worst, float(d))
        lp_fin = torch.log_softmax(z_fin.double(), dim=-1)
        dtypes = (z_fin.dtype, lp_fin.dtype)
        p_fin = lp_fin.exp()
        top = lp_fin.argmax(-1, keepdim=True)
        lp_prev = p_prev = None
        for l in range(L):
            lp = lp_fin if l == L - 1 else torch.log_softmax(
                _lens_logits(model, final_norm, states[l][a:b]).double(), dim=-1)
            p = lp.exp()
            rows[r, l] = _plogp_sum(p, lp - lp_fin).cpu().numpy()
            rows[r, L + l] = _plogp_sum(p_fin, lp_fin - lp).cpu().numpy()
            rows[r, 2 * L + l] = (-_plogp_sum(p, lp)).cpu().numpy()
            rows[r, 3 * L + l] = _margin(lp, top).cpu().numpy()
            if l:
                rows[r, 4 * L + l - 1] = _plogp_sum(p_prev, lp_prev - lp).cpu().numpy()
            lp_prev, p_prev = lp, p
    return rows, worst, dtypes


def _col_means(X, cols):
    """Exactly rounded column means: a column's mean depends on its values only."""
    n = X.shape[0]
    return [math.fsum(X[:, j].tolist()) / n for j in cols]


def _depths(M):
    """Per-row (stay, first) layer indices of a margin matrix (N, L).

    stay is ``layer_stay_pos`` of the release: one past the last layer whose
    margin is < 0, capped at L-1, or 0 when none is; first is the first layer
    whose margin is >= 0.
    """
    L = M.shape[1]
    neg = M < 0
    last_neg = L - 1 - np.argmax(neg[:, ::-1], axis=1)
    stay = np.where(neg.any(axis=1), np.minimum(last_neg + 1, L - 1), 0)
    first = np.argmax(M >= 0, axis=1)
    return stay, first


@register_task("interpretability_logit_lens_kl")
class LogitLensKLTask(DiagnosticTask):
    """Logit-lens KL to the output, KL churn and decision depth on normalized depth.

    Headline keys and why:

    * ``kl_churn`` -- how much the decoded distribution moves in total on its
      way through the stack. An identity block adds a KL term of exactly 0, so
      it is unchanged by identity blocks inserted after any block.
    * ``kl_to_final_auc`` -- how far, averaged over normalized depth, the
      intermediate decodes sit from the output (Hu et al.'s AUC divided by L).
      It moves under identity insertion (appended blocks scale it by L0 / L).
    * ``decision_depth`` -- the normalized depth from which the final top-1 is
      locked in (Hu et al.'s time to decision); it moves under identity
      insertion like the AUC.
    """

    TOKEN_POLICY = "exclude_sinks"
    SOURCES = {
        "paper": ("arXiv:2504.14107 (Hu, Lepori & Franke, Signatures of human-like processing "
                  "in Transformer forward passes: logit-lens AUC and time to decision); "
                  "arXiv:2502.16570 (Entropy-Lens: depth-normalized logit-lens profiles)"),
        "code": "jennhu/model-human-processing@69e405c153ae630c9f4eaad51d1c353fe643de26",
        "llm_precedent": ("arXiv:2504.14107 (logit lens on GPT-2, Llama-2, Llama-3.1, Gemma-2, "
                          "OLMo-2 and Falcon-3, 124M-405B); arXiv:2303.08112 (tuned lens: "
                          "KL between a layer's decode and the output distribution)"),
    }
    ADAPTATION_NOTE = (
        "Hu et al. decode the L block outputs (no embedding output; the last block's output "
        "through the final norm once) as lm_head(norm(h)), sum a per-layer measure over layers "
        "(AUC) and set time to decision to (s + 1) / L, s the layer from which log p(correct) - "
        "log p(incorrect) stays >= 0 (layer_stay_pos); they need correct/incorrect answer pairs. "
        "On free text BLME takes the final top-1 as the correct answer and the strongest other "
        "token of each layer as the incorrect one, so decision_depth is the depth from which the "
        "final top-1 stays the lens's top-1 and first_agreement_depth the depth where it first "
        "is, and applies the AUC to KL(p_l || p_L) and to the lens entropy, divided by L (the "
        "released sums are _meta_). The release computes no KL: kl_churn = sum_l KL(p_l || "
        "p_l+1) is the C6 adaptation, with the layer's own distribution first, and the reverse "
        "direction, which the tuned lens trains on, is _meta_. Departures: the final logit "
        "softcap is applied (the release's lm_head(norm(h)) omits Gemma-2's), the norm and "
        "unembedding run in float32 and the log-softmax in float64 at every size (the forward "
        "keeps the model's dtype), and the positions are every token of the first num_samples "
        "(100) corpus passages, at most 512 tokens each, except the sinks. An identity "
        "block adds a zero KL term, so kl_churn is unchanged by identity blocks after any block, "
        "while the AUC and both depths move (appended blocks scale them by L0 / L)."
    )
    HEADLINE_KEYS = ("kl_churn", "kl_to_final_auc", "decision_depth")
    SIZE_INVARIANT = {
        "kl_churn": (("insert_identity_blocks",),
                     _INV + "test_churn_is_exact_under_identity_blocks"),
    }

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Logit-Lens KL / decision depth...")
        num_samples = int(self.config.get("num_samples", DEFAULT_NUM_SAMPLES))
        max_length = int(self.config.get("max_length", DEFAULT_MAX_LENGTH))
        max_length = min(max_length, get_max_positions(model, max_length))
        chunk_elements = int(self.config.get("chunk_elements", DEFAULT_CHUNK_ELEMENTS))
        if dataset is None:
            from ...cache import load_default_corpus
            dataset = load_default_corpus(num_samples)

        layers = get_layers(model)
        if layers is None or len(layers) == 0:
            return {"error": "Could not detect model layers"}
        L = len(layers)
        final_norm = get_final_norm(model)
        head = get_lm_head(model)
        if head is not None:
            vocab = int(head.out_features)
        else:
            E = get_embeddings(model)
            if E is None:
                return {"error": "No unembedding to decode with"}
            vocab = int(E.shape[0])
        chunk = max(1, chunk_elements // max(1, vocab))
        lens_modules = [m for m in (final_norm, head) if m is not None]
        device = next(model.parameters()).device
        model_dtype = floating_dtype_summary(model)

        mid = middle_depth_layers(dict.fromkeys(range(L)))
        norms = {k: [] for k in mid}
        per_doc, positions = [], []
        n_docs = n_stripped = n_promoted = 0
        worst = 0.0
        dtypes = (None, None)
        width = 5 * L - 1
        with torch.no_grad():
            for i, sample in enumerate(dataset):
                if i >= num_samples:
                    break
                text = sample["text"] if isinstance(sample, dict) else str(sample)
                n_docs += 1
                enc = tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=max_length)
                enc, k = strip_trailing_special(enc, tokenizer)
                n_stripped += k
                enc = {kk: v.to(device) for kk, v in enc.items() if isinstance(v, torch.Tensor)}
                T = int(enc["input_ids"].shape[1])
                positions.append(torch.arange(1, max(T, 1), dtype=torch.int64))
                if T == 0:
                    for kk in mid:
                        norms[kk].append(torch.zeros((0, 1), dtype=torch.float64))
                    per_doc.append(torch.zeros((0, width), dtype=torch.float64))
                    continue
                # post-final-norm: not read. Every block output is hooked, the
                # last one before the final norm, which the lens applies once.
                states, logits = _block_outputs(model, layers, enc)
                hs = [s[0] for s in states]
                # Sink masks need the corpus-wide median norm, so the middle-depth
                # row norms are kept and the masks built after the last document.
                for kk in mid:
                    norms[kk].append(torch.linalg.vector_norm(
                        hs[kk].to("cpu", torch.float64), dim=1).unsqueeze(1))
                if T < 2:
                    per_doc.append(torch.zeros((0, width), dtype=torch.float64))
                    continue
                with contextlib.ExitStack() as stack:
                    # fp32 decode at every size; only the norm and the head are
                    # promoted, and only while decoding (a tied head shares the
                    # embedding, which the next forward must see in its dtype).
                    prec = [stack.enter_context(fp32_compute(m)) for m in lens_modules]
                    n_promoted += sum(p["n_params_promoted"] for p in prec)
                    rows, w, dt = _doc_rows(model, final_norm, hs,
                                            None if logits is None else logits[0], chunk)
                per_doc.append(torch.from_numpy(rows))
                worst = max(worst, w)
                dtypes = dt

        if not per_doc:
            return {"error": "No passages to decode"}
        masks, sink_info = sink_keep_masks([norms[k] for k in mid])
        X, row_meta = select_rows(per_doc, self.TOKEN_POLICY, positions=positions,
                                  keep=[m[1:] for m in masks])
        X = X.numpy()
        N = X.shape[0]
        if N == 0:
            return {"error": "No token positions left after excluding the sinks"}

        kl_final = _col_means(X, range(L))
        kl_rev = _col_means(X, range(L, 2 * L))
        entropy = _col_means(X, range(2 * L, 3 * L))
        kl_next = _col_means(X, range(4 * L, 5 * L - 1))
        M = X[:, 3 * L:4 * L]
        stay, first = _depths(M)
        agree = (M >= 0).sum(axis=0)
        out = {
            "kl_churn": math.fsum(kl_next),
            "kl_to_final_auc": math.fsum(kl_final) / L,
            "decision_depth": math.fsum(((stay + 1) / L).tolist()) / N,
            "first_agreement_depth": math.fsum(((first + 1) / L).tolist()) / N,
            "entropy_auc": math.fsum(entropy) / L,
            "kl_to_final_profile": kl_final,
            "kl_next_profile": kl_next,
            "entropy_profile": entropy,
            "top1_agreement_profile": [float(c) / N for c in agree],
            "_meta_kl_to_final_auc_sum": math.fsum(kl_final),
            "_meta_entropy_auc_sum": math.fsum(entropy),
            "_meta_kl_final_to_layer_profile": kl_rev,
            "_meta_kl_final_to_layer_auc": math.fsum(kl_rev) / L,
            "_meta_layer_depth": [(l + 1) / L for l in range(L)],
            "_meta_n_layers": L,
            "_meta_n_docs": n_docs,
            "_meta_n_tokens": N,
            "_meta_n_trailing_special_stripped": n_stripped,
            "_meta_model_dtype": model_dtype,
            "_meta_decode_dtype": str(dtypes[0]).replace("torch.", ""),
            "_meta_log_softmax_dtype": str(dtypes[1]).replace("torch.", ""),
            "_meta_n_lens_params_promoted": int(n_promoted),
            "_meta_final_lens_max_abs_logit_diff": worst,
            "_meta_positions_per_chunk": chunk,
            "_meta_max_length": max_length,
        }
        out.update(row_meta.as_meta())
        out.update(sink_meta(sink_info))
        return out
