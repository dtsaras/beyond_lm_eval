"""``geometry_prompt_id``: the intrinsic dimension of each prompt's token cloud
across depth (Viswanathan et al., "The Intrinsic Dimension of Prompts in
Internal Representations of Large Language Models", TMLR 2026,
arXiv:2501.10573).

The release (RitAreaSciencePark/token_geometry@0214897, ``src/extract_id.py``)
tokenizes each prompt without special tokens, keeps its first 1024 tokens and
reads HF's hidden states 1..L. Per prompt and layer it takes the token cloud's
``torch.cdist`` distance matrix, drops repeated rows, and runs DADApy's
``Data(distances=..., maxk=100).return_id_scaling_gride(range_max=64)``; the
paper reads the estimate at range scaling n2 = 4 (n1 = 2, ``[..., 0, 1]``).
The ID profile is averaged over prompts, and per layer the Pearson correlation
across prompts between log ID and the prompt's mean surprisal (the HF loss) is
the paper's main result. GRIDE is BLME's DADApy port in ``gride.py``.
"""
import logging
import math
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from ..common import get_max_positions, relative_depth
from .gride import _gride_scaling, _unique_rows

logger = logging.getLogger("blme")

_INV = "tests/tasks/invariance/test_geometry_prompt_id.py::"
# extract_id.py: return_id_scaling_gride(range_max=64), read at n2 = 4.
RANGE_MAX, SCALE_N2 = 64, 4
# 1024 tokens at under 8 bytes per token (at most 5.8 measured on 10 tokenizers).
_MIN_BYTES = 8192


def prompt_scaling(X, range_max=RANGE_MAX):
    """GRIDE ids of one token cloud at n2 = 2, 4, ... up to range_max, and the ranks."""
    return _gride_scaling(X, range_max)


def token_surprisal(logits, ids):
    """Per-token -log p(ids[t] | ids[:t]) for t >= 1, from float32 logits (the HF loss terms)."""
    return F.cross_entropy(logits[:-1].float(), ids[1:], reduction="none").double()


def log_id_pearson(ids, surprisal):
    """Per-layer Pearson r and p-value across prompts of log ID vs mean surprisal
    (``stats.pearsonr(np.log(ids[:, l]), loss)``, plots_for_paper.ipynb)."""
    from scipy import stats
    ids, y = np.asarray(ids, dtype=np.float64), np.asarray(surprisal, dtype=np.float64)
    r, p = [], []
    for col in ids.T:
        x = np.log(col)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
            r.append(float("nan"))
            p.append(float("nan"))
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = stats.pearsonr(x[ok], y[ok])
        r.append(float(res.statistic))
        p.append(float(res.pvalue))
    return r, p


def _at(ids, ranks, n2):
    hit = np.nonzero(ranks == n2)[0]
    return float(ids[hit[0]]) if hit.size else float("nan")


def _scaling_mean(scaling, width):
    """Mean over prompts of each layer's scaling profile; a cloud with too few
    unique tokens for the largest scales contributes NaN there."""
    S = np.full((len(scaling), len(scaling[0]), width), np.nan)
    for p, row in enumerate(scaling):
        for l, sc in enumerate(row):
            S[p, l, :min(width, len(sc))] = sc[:width]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(S, axis=0).tolist()


def _peak(profile):
    arr = np.asarray(profile, dtype=np.float64)
    if not np.isfinite(arr).any():
        return float("nan"), -1
    i = int(np.nanargmax(arr))
    return float(arr[i]), i


@register_task("geometry_prompt_id")
class PromptIntrinsicDimTask(DiagnosticTask):
    """GRIDE intrinsic dimension of each prompt's token cloud, per layer (Viswanathan et al.)."""

    TOKEN_POLICY = ("paper_protocol:Viswanathan2025 every token of the first 1024 of each prompt, "
                    "no special tokens")
    SOURCES = {
        "paper": ("arXiv:2501.10573 (Viswanathan et al., TMLR 2026, The Intrinsic Dimension of "
                  "Prompts in Internal Representations of Large Language Models); "
                  "doi:10.1038/s41598-022-20991-1 (Denti et al. 2022, GRIDE)"),
        "code": ("RitAreaSciencePark/token_geometry@02148974cd39ff7f186e734dd1e9113ed42ee614 "
                 "(src/extract_id.py); "
                 "sissa-data-science/DADApy@d729b8835e771001ec7ba6ca68938b737b96edae"),
        "llm_precedent": ("arXiv:2501.10573 (Llama-3-8B, Mistral-7B, Pythia, OPT-6.7B and GPT-2: "
                          "every token of 2244 Pile-10K prompts of 1024 tokens, HF hidden states "
                          "1..L, GRIDE at n2 = 4)"),
    }
    ADAPTATION_NOTE = (
        "The release's readout and estimator: the first 1024 tokens of each prompt, no special "
        "tokens, every HF hidden state after the embeddings (the last after the final norm), "
        "GRIDE per prompt and layer on the deduplicated token cloud read at n2 = 4, averaged "
        "over prompts, and per layer the Pearson correlation across prompts of log ID with the "
        "prompt's mean surprisal. Prompts are the first 256 WikiText-103 train articles of at "
        "least 8192 bytes, so every tokenizer reaches 1024 tokens on the same articles, where "
        "the release uses 2244 Pile-10K prompts (class 2); 256 keeps the CPU GRIDE of a 33-layer "
        "8B model near an hour. Distances and the mean surprisal are float64 on float32-cast "
        "states and logits, where the release takes torch.cdist in the model's dtype and HF's "
        "float32 loss (1e-7 apart). Position 0 stays in the cloud, as in the "
        "release; the profile without it is _meta_. The release's ESS (k = 10) and TLE (k = 20) "
        "cross-checks are not ported. The ID of a cloud pinned at 1024 points is a distance-ratio "
        "statistic, so the peak and the last-layer value are unchanged by identity blocks "
        "between blocks, zero-padding and rescaling (a block before block 0 adds the embedding "
        "state, one after the last block the pre-norm state, and either can move the peak); "
        "the peak's depth moves when blocks are added and is _meta_."
    )
    HEADLINE_KEYS = ("prompt_id_peak", "prompt_id_last_layer")
    SIZE_INVARIANT = {
        k: (ends + ("zero_pad", "rescale"), _INV + "test_headline_invariant")
        for k, ends in (
            ("prompt_id_peak", ("identity_blocks_interior",)),
            ("prompt_id_last_layer",
             ("identity_blocks_interior", "identity_blocks_leading", "identity_blocks_trailing")))}

    def _texts(self, dataset, num_samples):
        min_bytes = int(self.config.get("min_bytes", _MIN_BYTES))
        if self.config.get("use_dataset", False):
            pool = list(dataset or [])
        else:
            from ...cache import load_long_documents
            pool = load_long_documents(4 * num_samples, split=self.config.get("split", "train"))
        texts = [(d["text"] if isinstance(d, dict) else str(d)).strip() for d in pool]
        return [t for t in texts if len(t.encode("utf-8")) >= min_bytes][:num_samples]

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running prompt-level GRIDE intrinsic dimension...")
        num_samples = int(self.config.get("num_samples", 256))
        max_length = int(self.config.get("max_length", 1024))
        max_length = min(max_length, get_max_positions(model, max_length))
        n2 = int(self.config.get("scale_n2", SCALE_N2))
        range_max = int(self.config.get("range_max", RANGE_MAX))
        texts = self._texts(dataset, num_samples)
        if not texts:
            return {"error": "No prompt of the required byte length"}
        device = next(model.parameters()).device

        ids_all, ids_rest, scaling, surprisal = [], [], [], []
        ranks_ref, n_short, n_unique_min = None, 0, None
        with torch.no_grad():
            for text in texts:
                ids = tokenizer(text, add_special_tokens=False, truncation=True,
                                max_length=max_length)["input_ids"]
                if len(ids) < max_length:
                    n_short += 1
                    continue
                inp = torch.tensor([ids], device=device)
                out = model(input_ids=inp, output_hidden_states=True, use_cache=False)
                surprisal.append(math.fsum(token_surprisal(out.logits[0], inp[0]).cpu().tolist())
                                 / (len(ids) - 1))
                # post-final-norm: the release reads HF's hidden_states[1:], whose
                # last entry is the post-norm output state.
                hs = out.hidden_states[1:]
                row, row_rest, row_scaling = [], [], []
                for h in hs:
                    X = h[0].float().cpu().numpy()
                    sc, ranks = prompt_scaling(X, range_max)
                    ranks_ref = ranks if ranks_ref is None else ranks_ref
                    row.append(_at(sc, ranks, n2))
                    row_scaling.append([float(v) for v in sc])
                    sc_rest, ranks_rest = prompt_scaling(X[1:], range_max)
                    row_rest.append(_at(sc_rest, ranks_rest, n2))
                    nu = int(_unique_rows(X).shape[0])
                    n_unique_min = nu if n_unique_min is None else min(n_unique_min, nu)
                ids_all.append(row)
                ids_rest.append(row_rest)
                scaling.append(row_scaling)
                del out, hs
        if not ids_all:
            return {"error": f"No prompt reaches {max_length} tokens"}

        A = np.asarray(ids_all, dtype=np.float64)          # prompts x layers
        R = np.asarray(ids_rest, dtype=np.float64)
        profile, rest = A.mean(axis=0), R.mean(axis=0)
        L = A.shape[1]
        peak, i_peak = _peak(profile)
        peak_rest, _ = _peak(rest)
        r, pval = log_id_pearson(A, surprisal)
        r_rest, _ = log_id_pearson(R, surprisal)
        med = np.median(A, axis=1, keepdims=True)
        return {
            "prompt_id_peak": peak,
            "prompt_id_last_layer": float(profile[-1]),
            "prompt_id_profile": [float(v) for v in profile],
            "prompt_log_id_surprisal_pearson": r,
            "_meta_prompt_log_id_surprisal_pvalue": pval,
            "_meta_prompt_log_id_surprisal_pearson_at_peak": r[i_peak] if i_peak >= 0 else float("nan"),
            "_meta_prompt_log_id_surprisal_pearson_last_layer": r[-1],
            "_meta_prompt_id_peak_layer": i_peak + 1,
            "_meta_prompt_id_peak_depth": relative_depth(i_peak + 1 if i_peak >= 0 else -1, L),
            "_meta_prompt_id_std_profile": [float(v) for v in A.std(axis=0)],
            "_meta_prompt_id_median_profile": [float(v) for v in np.median(A, axis=0)],
            "_meta_prompt_id_median_normalised_profile": [float(v) for v in (A / med).mean(axis=0)],
            "_meta_prompt_id_scaling_mean": _scaling_mean(scaling, len(ranks_ref)),
            "_meta_prompt_id_scaling_ranks": [int(v) for v in ranks_ref],
            "_meta_prompt_id_profile_without_first_token": [float(v) for v in rest],
            "_meta_prompt_id_peak_without_first_token": peak_rest,
            "_meta_prompt_id_last_layer_without_first_token": float(rest[-1]),
            "_meta_prompt_log_id_surprisal_pearson_without_first_token": r_rest,
            "_meta_mean_surprisal": math.fsum(surprisal) / len(surprisal),
            "_meta_prompt_surprisal": surprisal,
            "_meta_prompt_id_scale_n2": n2,
            "_meta_prompt_tokens": max_length,
            "_meta_n_prompts": len(ids_all),
            "_meta_n_prompts_too_short": n_short,
            "_meta_n_unique_tokens_min": n_unique_min,
            "_meta_n_layers": L,
            "_meta_tokenizer_adds_bos": (len(tokenizer("a")["input_ids"])
                                         > len(tokenizer.encode("a", add_special_tokens=False))),
        }
