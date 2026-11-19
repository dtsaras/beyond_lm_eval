"""FFN activation spectral utilisation: how evenly the feed-forward network
spreads the variance of its post-activation over its D latent directions
(Jha & Reagen, EMNLP 2025).

Reference: "Spectral Scaling Laws in Language Models: How Effectively Do
Feed-Forward Networks Use Their Latent Space?", arXiv:2510.00537. No code was
released; this follows the paper's formulas.

Per layer, X (N x D) is the FFN post-activation, the input of the down
projection: sigma(W_gate x) * (W_up x) for gated MLPs, act(W_in x) for GPT-2
and GPT-NeoX. Sec. 3.1, Eq. 1: Sigma = (X - mu)^T (X - mu) / (N - 1) over all
N tokens, eigenvalues lambda_1 >= ... >= lambda_D >= 0, p_i = lambda_i / sum
lambda. Table 1:

* hard rank PR = (sum lambda)^2 / sum lambda^2, PR~ = (PR - 1) / (D - 1);
* soft rank eR = exp(-sum p log p), eR~ = (eR - 1) / (D - 1);
* spectral concentration SC = (2 / D) sum_k (sum_{i<=k} p_i - k / D);
* SUI = 2 PR~ eR~ / (PR~ + eR~), eDim = 1 + (D - 1) SUI.

D cancels in eDim = 1 + 2 (PR - 1)(eR - 1) / (PR + eR - 2), so eDim, PR and eR
are unchanged by zero-padding the FFN width (dead neurons add zero
eigenvalues), while PR~, eR~, SUI and SC divide by D and move.

Memory: the sufficient statistics of a layer are float64 sum x (D) and sum
x x^T (D x D), 1.6 GB at D = 14336. Layers are processed in consecutive groups
whose accumulators fit ``memory_budget_gb`` (one D x D copy is reserved for
eigvalsh); each group re-runs the forward over the same tokens.
"""
import logging
from typing import Dict, List, Optional

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from .._tokens import strip_trailing_special
from ..common import get_layers, get_max_positions
from .schatten import _mlp_down_projections
from .utils import eigvalsh_f64

logger = logging.getLogger("blme")

_GIB = 1024 ** 3
_KEYS = ("hard_rank", "soft_rank", "hard_rank_normalized", "soft_rank_normalized",
         "spectral_concentration", "sui", "edim", "spectral_centroid")


def spectral_utilization(eigenvalues) -> Dict[str, float]:
    """Table 1 of Jha & Reagen on one covariance spectrum, in float64.

    D is the number of eigenvalues, zeros included; rounding negatives are set
    to 0. ``spectral_centroid`` = sum_i i p_i (1-based), so SC = 1 - (2
    centroid - 1) / D; unlike SC it is unchanged by appended zero eigenvalues.
    """
    lam = np.clip(np.asarray(eigenvalues, dtype=np.float64).ravel(), 0.0, None)
    lam = np.sort(lam)[::-1]
    D = lam.size
    total = float(lam.sum())
    if D < 2 or not (np.isfinite(total) and total > 0):
        return dict.fromkeys(_KEYS, float("nan"))
    p = lam / total
    pr = 1.0 / float(np.sum(p * p))
    q = p[p > 0]
    er = float(np.exp(-np.sum(q * np.log(q))))
    pr_n = (pr - 1.0) / (D - 1)
    er_n = (er - 1.0) / (D - 1)
    k = np.arange(1, D + 1, dtype=np.float64)
    sc = 2.0 / D * float(np.sum(np.cumsum(p) - k / D))
    s = pr_n + er_n
    # Rank one: PR~ = eR~ = 0, and the harmonic mean's limit is 0.
    sui = 2.0 * pr_n * er_n / s if s > 0 else 0.0
    return {"hard_rank": pr, "soft_rank": er, "hard_rank_normalized": pr_n,
            "soft_rank_normalized": er_n, "spectral_concentration": sc, "sui": sui,
            "edim": 1.0 + (D - 1) * sui, "spectral_centroid": float(np.sum(k * p))}


def n_nonzero(eigenvalues) -> int:
    """Eigenvalues above the numerical-rank tolerance lambda_max * D * eps64."""
    lam = np.asarray(eigenvalues, dtype=np.float64)
    if lam.size == 0 or not lam.max() > 0:
        return 0
    tol = float(lam.max()) * lam.size * np.finfo(np.float64).eps
    return int(np.sum(lam > tol))


class CovarianceAccumulator:
    """Streamed float64 sufficient statistics of one layer's activation rows.

    Rows are shifted by the first batch's mean before summing: the covariance
    is shift-invariant, and the shift keeps sum x x^T - N mu mu^T from
    cancelling. Each batch's first row (position 0) is kept, so the spectrum
    without those rows is a low-rank downdate of the same sums.
    """

    def __init__(self, width: int, device="cpu"):
        self.device = torch.device(device)
        self.n = 0
        self.shift = None
        self.s1 = torch.zeros(width, dtype=torch.float64, device=self.device)
        self.s2 = torch.zeros(width, width, dtype=torch.float64, device=self.device)
        self.first: List[torch.Tensor] = []

    def add(self, rows: torch.Tensor) -> None:
        X = rows.detach().reshape(-1, rows.shape[-1]).to(self.device, torch.float64)
        if X.shape[0] == 0:
            return
        if self.shift is None:
            self.shift = X.mean(dim=0)
        X = X - self.shift
        self.s1 += X.sum(dim=0)
        self.s2.addmm_(X.T, X)
        self.n += X.shape[0]
        self.first.append(X[0].clone())

    def spectra(self, without_position0: bool = True) -> Dict[str, object]:
        """Eigenvalues of Sigma over all rows (``all``) and without the
        position-0 rows (``without_position0``), and the share of the centred
        sum of squares those rows hold. Consumes the accumulator."""
        out = {"all": None, "without_position0": None, "position0_share": float("nan"),
               "n": self.n, "n_without_position0": self.n - len(self.first)}
        n, M = self.n, self.s2
        if n < 2:
            return out
        m = self.s1 / n
        M.addr_(m, m, alpha=-float(n))
        tss = float(torch.diagonal(M).sum())
        w, solver = eigvalsh_f64(M)
        out["solvers"] = [solver]
        out["all"] = w.div_(n - 1).cpu().numpy()
        P = torch.stack(self.first)
        if tss > 0:
            out["position0_share"] = float(((P - m) ** 2).sum()) / tss
        n2 = n - P.shape[0]
        if without_position0 and n2 >= 2:
            m2 = (self.s1 - P.sum(dim=0)) / n2
            M.addr_(m, m, alpha=float(n))
            M.addmm_(P.T, P, alpha=-1.0)
            M.addr_(m2, m2, alpha=-float(n2))
            w, solver = eigvalsh_f64(M)
            out["solvers"].append(solver)
            out["without_position0"] = w.div_(n2 - 1).cpu().numpy()
        self.s2 = self.s1 = None
        self.first = []
        return out


# The named down projections of interpretability.sparsity._find_down_proj.
_DOWN_NAMES = ("down_proj", "c_proj", "dense_4h_to_h", "ff_out", "fc2", "out_proj", "proj")


def dense_down_projections(model):
    """geometry_schatten's sites without those found only by its fallback (the
    block's last projection): in a routed MoE block that is a gate, whose
    input is the residual stream. Returns (sites, n_skipped)."""
    layers = get_layers(model)
    sites, n_skipped = [], 0
    for i, dp in _mlp_down_projections(model):
        mlp = getattr(layers[i], "mlp", None) or getattr(layers[i], "feed_forward", None)
        if any(getattr(mlp, a, None) is dp for a in _DOWN_NAMES):
            sites.append((i, dp))
        else:
            n_skipped += 1
    return sites, n_skipped


def _in_width(module) -> Optional[int]:
    """Input width of an nn.Linear (weight out x in) or HF Conv1D (in x out)."""
    if isinstance(module, torch.nn.Linear):
        return int(module.in_features)
    w = getattr(module, "weight", None)
    return int(w.shape[0]) if w is not None and w.ndim == 2 else None


def layer_groups(widths, n_first_rows: int, max_length: int, budget_bytes: float):
    """Consecutive layer groups whose float64 accumulators (D x D, two D
    vectors, the position-0 rows) plus one forward's activations fit the
    budget, with one D_max x D_max copy reserved for eigvalsh. A group holds at
    least one layer."""
    reserve = 8 * max(widths) ** 2
    groups, cur, used = [], [], 0
    for j, D in enumerate(widths):
        need = 8 * (D * D + D * (2 + n_first_rows)) + 12 * D * max_length
        if cur and used + need + reserve > budget_bytes:
            groups.append(cur)
            cur, used = [], 0
        cur.append(j)
        used += need
    if cur:
        groups.append(cur)
    return groups


def token_batches(tokenizer, texts, n_tokens: int, max_length: int):
    """The texts in order, each with the tokenizer's defaults, cut at
    ``max_length`` tokens and without a trailing EOS/SEP, until ``n_tokens``
    tokens are reached; the last one is cut so the total is exact. Texts under
    two tokens are skipped. Returns (batches, n_trailing_special_stripped)."""
    out, total, n_strip = [], 0, 0
    for t in texts:
        if total >= n_tokens:
            break
        enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length)
        enc, k = strip_trailing_special(enc, tokenizer)
        ids = enc["input_ids"][:, :min(max_length, n_tokens - total)]
        if ids.shape[-1] < 2:
            continue
        out.append(ids)
        total += int(ids.shape[-1])
        n_strip += k
    return out, n_strip


def _median(v):
    v = [x for x in v if np.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


@register_task("geometry_ffn_spectral_utilization")
class FFNSpectralUtilizationTask(DiagnosticTask):
    """Hard rank, soft rank, spectral concentration and SUI of the FFN
    post-activation covariance, per layer."""

    TOKEN_POLICY = ("paper_protocol:Jha2025 all N tokens of the batch (Sec. 3.1, Eq. 1), "
                    "position 0 included")
    SOURCES = {
        "paper": ("arXiv:2510.00537 (Jha & Reagen, EMNLP 2025, Sec. 3.1-3.2, Eq. 1, "
                  "Table 1)"),
        "code": "",
        "llm_precedent": ("arXiv:2510.00537 (LLaMA 70M-250M trained on C4, GPT-2 and nGPT; "
                          "every FFN layer)"),
    }
    ADAPTATION_NOTE = (
        "X is the input of the down projection of each dense MLP (routed MoE blocks are "
        "skipped), the paper's FFN formula; its PostAct line puts sigma around the product, "
        "read as a typo. The paper trains its own models on C4 and leaves N unstated; BLME "
        "pins N = 32768 tokens (all of them, position 0 included, as Eq. 1), the first "
        "passages of the library corpus, at most 512 tokens each, with the last passage cut "
        "so N is exact (class 2). Sums and eigvalsh are float64, and layers run in groups "
        "under a memory budget with one forward per group over the same tokens. Headline: "
        "the layer medians the figures plot, of SUI, eDim and the raw soft and hard ranks; "
        "the paper fits power laws across widths and reports no model scalar. The raw ranks "
        "and eDim (D cancels in it) are unchanged by zero-padding the FFN width, while the "
        "normalised ranks, SC and SUI divide by D. An inserted identity block's MLP reads a "
        "new input, so it adds a new layer to every profile and nothing is invariant under "
        "insertion. Finite N biases every rank (the covariance has rank at most N - 1) and "
        "the spectrum without position 0 is _meta_."
    )
    HEADLINE_KEYS = ("ffn_sui_median", "ffn_edim_median", "ffn_soft_rank_median",
                     "ffn_hard_rank_median")
    # Identity blocks add layers whose MLP reads a different input, so every
    # median moves (tests/tasks/invariance/test_geometry_ffn_spectral_utilization.py).
    SIZE_INVARIANT = {}

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running FFN spectral utilisation...")
        n_tokens = int(self.config.get("n_tokens", 32768))
        max_length = int(self.config.get("max_length", 512))
        max_length = min(max_length, get_max_positions(model, max_length))
        cap = self.config.get("num_samples")
        budget = float(self.config.get("memory_budget_gb", 8.0))
        without_pos0 = bool(self.config.get("without_position0", True))

        if self.config.get("use_dataset", False) and dataset is not None:
            texts = [d["text"] if isinstance(d, dict) else str(d) for d in dataset]
        else:
            from ...cache import default_corpus_size, load_default_corpus
            texts = [d["text"] for d in load_default_corpus(default_corpus_size())]
        if cap is not None:
            texts = texts[:int(cap)]
        batches, n_strip = token_batches(tokenizer, texts, n_tokens, max_length)
        if not batches:
            return {"error": "No passage has two tokens"}
        n_total = sum(int(b.shape[-1]) for b in batches)

        sites, n_not_dense = dense_down_projections(model)
        widths = [_in_width(dp) for _, dp in sites]
        if not sites or any(w is None for w in widths):
            return {"error": "Could not find a dense MLP down-projection (MoE blocks are "
                             "not supported)"}
        dev = self.config.get("accumulator_device", "auto")
        param_dev = next(model.parameters()).device
        acc_dev = (param_dev if param_dev.type == "cuda" else torch.device("cpu")) \
            if dev == "auto" else torch.device(dev)
        groups = layer_groups(widths, len(batches), max_length, budget * _GIB)

        spectra: Dict[int, dict] = {}
        n_mismatch = 0
        for group in groups:
            accs = {j: CovarianceAccumulator(widths[j], acc_dev) for j in group}
            state, bad = {"T": 0}, set()

            def _make_hook(j):
                def _hook(module, args):
                    X = args[0] if args and isinstance(args[0], torch.Tensor) else None
                    if X is None or X.numel() // X.shape[-1] != state["T"]:
                        # A routed expert sees a subset of the tokens.
                        bad.add(j)
                        return
                    accs[j].add(X)
                return _hook

            hooks = []
            try:
                for j in group:
                    hooks.append(sites[j][1].register_forward_pre_hook(_make_hook(j)))
                with torch.no_grad():
                    for ids in batches:
                        state["T"] = int(ids.shape[-1])
                        model(input_ids=ids.to(param_dev), use_cache=False)
            finally:
                for h in hooks:
                    h.remove()
            for j in group:
                spectra[j] = None if j in bad else accs[j].spectra(without_pos0)
                n_mismatch += int(j in bad)
                del accs[j]

        prof = {k: [] for k in _KEYS}
        prof_ex = {k: [] for k in ("hard_rank", "soft_rank", "sui")}
        nonzero, share = [], []
        for j in range(len(sites)):
            sp = spectra.get(j) or {}
            lam = sp.get("all")
            m = spectral_utilization(lam) if lam is not None else dict.fromkeys(_KEYS, float("nan"))
            for k in _KEYS:
                prof[k].append(m[k])
            lam_ex = sp.get("without_position0")
            m_ex = spectral_utilization(lam_ex) if lam_ex is not None else {}
            for k in prof_ex:
                prof_ex[k].append(m_ex.get(k, float("nan")))
            nonzero.append(n_nonzero(lam) if lam is not None else 0)
            share.append(sp.get("position0_share", float("nan")))

        out = {}
        for k in ("sui", "soft_rank", "hard_rank", "soft_rank_normalized",
                  "hard_rank_normalized", "spectral_concentration", "edim"):
            out[f"ffn_{k}_median"] = _median(prof[k])
            out[f"ffn_{k}_profile"] = prof[k]
        out["ffn_sui_mean"] = (float(np.nanmean(prof["sui"]))
                               if np.isfinite(prof["sui"]).any() else float("nan"))
        out["_meta_ffn_spectral_centroid_profile"] = prof["spectral_centroid"]
        for k, v in prof_ex.items():
            out[f"_meta_ffn_{k}_median_without_position0"] = _median(v)
            out[f"_meta_ffn_{k}_profile_without_position0"] = v
        out["_meta_ffn_position0_variance_share_profile"] = share
        out["_meta_ffn_width_profile"] = widths
        out["_meta_ffn_block_indices"] = [i for i, _ in sites]
        names = {id(m): n for n, m in model.named_modules()}
        out["_meta_ffn_site_first_layer"] = names.get(id(sites[0][1]), "")
        out["_meta_n_nonzero_eigenvalues_profile"] = nonzero
        out["_meta_n_tokens"] = n_total
        out["_meta_n_tokens_without_position0"] = n_total - len(batches)
        out["_meta_tokens_per_dimension"] = n_total / max(widths)
        out["_meta_n_passages"] = len(batches)
        out["_meta_max_length"] = max_length
        out["_meta_n_trailing_special_stripped"] = n_strip
        out["_meta_n_layer_groups"] = len(groups)
        out["_meta_memory_budget_gb"] = budget
        out["_meta_accumulator_device"] = str(acc_dev)
        out["_meta_n_layers_token_mismatch"] = n_mismatch
        out["_meta_n_layers_not_dense"] = n_not_dense
        # Which float64 eigensolver produced the spectra (geometry.utils.eigvalsh_f64).
        solvers = [s for sp in spectra.values() if sp for s in sp.get("solvers", ())]
        out["_meta_ffn_eigensolver_counts"] = {s: solvers.count(s) for s in sorted(set(solvers))}
        # Derived from the spectra, not the hook count: a site whose hook never
        # fired has no spectrum.
        out["_meta_n_layers_measured"] = sum(
            1 for sp in spectra.values() if sp and sp.get("all") is not None)
        return out
