from ...tasks.base import DiagnosticTask
from ...registry import register_task
from ..common import get_embeddings
import torch
import numpy as np
from scipy.stats import skew
from tqdm import tqdm
import logging
logger = logging.getLogger("blme")


def _gini_from_counts(counts: np.ndarray) -> float:
    counts = np.sort(np.asarray(counts, dtype=np.float64))
    n = len(counts)
    total = counts.sum()
    if n == 0 or total <= 0:
        return 0.0
    ranks = np.arange(1, n + 1, dtype=np.float64)
    return float((2 * np.sum(ranks * counts)) / (n * total) - (n + 1) / n)


def _hubness_stats_from_occurrences(n_occ, k: int) -> dict:
    """Hubness skew/max/top-1%/Gini from k-NN occurrence counts."""
    n_occ_cpu = np.asarray(n_occ, dtype=np.int64)
    if n_occ_cpu.size == 0:
        hub_skew = float("nan")
    elif n_occ_cpu.std() == 0:
        # NaN, not 0.0: a constant occurrence vector makes S_Nk 0/0, as
        # scipy.stats.skew reports (test_hubness_parity.py).
        hub_skew = float("nan")
    else:
        hub_skew = float(skew(n_occ_cpu))
    hub_max = int(n_occ_cpu.max()) if n_occ_cpu.size else 0
    denom = n_occ_cpu.sum()
    # Mass of the top 1% of points: exactly the m = ceil(0.01 n) largest
    # counts. A >= percentile(99) cut over-selects on tied integer counts.
    n_pts = int(n_occ_cpu.size)
    if denom > 0 and n_pts > 0:
        m = int(np.ceil(0.01 * n_pts))
        m = max(1, min(m, n_pts))
        # np.partition puts the m largest in the tail; no full sort needed.
        top_m = np.partition(n_occ_cpu, n_pts - m)[n_pts - m:]
        top_1pct_mass = float(top_m.sum() / denom)
    else:
        top_1pct_mass = 0.0
    return {
        f"hubness_k{k}_skew": float(hub_skew),
        f"hubness_k{k}_max": hub_max,
        f"hubness_k{k}_top1pct": float(top_1pct_mass),
        f"hubness_k{k}_gini": _gini_from_counts(n_occ_cpu),
    }


def _unit_rows(E, device):
    """float64 unit rows on ``device``; exactly-zero rows stay zero.

    Each row is first divided by its largest-magnitude coordinate so the norm
    cannot underflow, then by its norm (clamped at the smallest positive
    value, never offset by an epsilon). Returns ``(U, n_zero, n_nonfinite,
    has_direction)``; rows without a direction (zero or non-finite) must not be
    ranked -- see :func:`_k_occurrences`.
    """
    U = E.detach().to(device=device, dtype=torch.float64)
    _rmax = U.abs().amax(dim=1, keepdim=True)
    U = U / _rmax.clamp_min(torch.finfo(U.dtype).tiny)
    _n = U.norm(dim=1, keepdim=True)
    U = U / _n.clamp_min(torch.finfo(U.dtype).tiny)
    # Counted on the row max: a zero row and a non-finite row are different
    # conditions (a NaN row fails `> 0` too and is zeroed below).
    _rm = _rmax.squeeze(-1)
    n_zero = int((_rm == 0).sum().item())
    n_nonfinite = int((~torch.isfinite(_rm)).sum().item())
    if n_zero or n_nonfinite:
        logger.warning(
            "hubness: %d zero-norm and %d non-finite embedding rows; "
            "they have no direction and their neighbour ranks are not "
            "meaningful", n_zero, n_nonfinite)
    has_direction = ((_rm > 0) & torch.isfinite(_rm)).cpu()
    U = torch.where(_rmax > 0, U, torch.zeros_like(U))
    return U, n_zero, n_nonfinite, has_direction


def _k_occurrences(U, k_values, batch_size):
    """k-occurrence counts N_k of every row of ``U`` (self excluded).

    Neighbours are ranked by DIRECT chord distance on unit rows (the
    dot-product ranking cancels two ~1 sums on anisotropic vocabularies), in
    float64, with a stable index tie-break so the result does not depend on
    the batch or on which other k were requested (test_hubness_parity.py).
    """
    device = U.device
    n_vocab = U.shape[0]
    max_k = max(k_values)
    # Cap the batch so the float64 (B, V) slice stays inside 512 MB; the
    # stable tie rule makes the result batch-invariant.
    _bytes = U.element_size() * max(1, n_vocab)
    batch_size = max(1, min(int(batch_size), (512 << 20) // _bytes))
    n_occ = {k: torch.zeros(n_vocab, dtype=torch.int64, device=device) for k in k_values}
    with torch.no_grad():
        for i in tqdm(range(0, n_vocab, batch_size), desc="Hubness"):
            end = min(i + batch_size, n_vocab)
            sims = torch.cdist(U[i:end], U, compute_mode="donot_use_mm_for_euclid_dist")
            rows = torch.arange(end - i, device=device)
            cols = torch.arange(i, end, device=device)
            sims[rows, cols] = float("inf")          # self is never a neighbour
            top_idx_sorted = torch.argsort(sims, dim=1, descending=False, stable=True)[:, :max_k]
            for k in k_values:
                flat = top_idx_sorted[:, :k].reshape(-1)
                n_occ[k].scatter_add_(0, flat, torch.ones_like(flat))
    return {k: v.cpu().numpy().astype(np.int64) for k, v in n_occ.items()}


@register_task("geometry_hubness")
class GlobalHubnessTask(DiagnosticTask):
    """
    Analyzes the "hubness" of the embedding space: the skewness of the distribution
    of k-nearest neighbor occurrences.

    References:
      * Radovanović, Nanopoulos, Ivanović 2010 — "Hubs in Space: Popular
        Nearest Neighbors in High-Dimensional Data", JMLR 11. The
        originator paper for k-NN hubness S_Nk.
      * Tomašev, Radovanović, Mladenić, Ivanović 2014 — "The Role of
        Hubness in Clustering High-Dimensional Data", IEEE TKDE 26(3):739-751.

    ``hubness_k*_gini`` is a BLME add-on (neither paper defines it): the
    standard Gini coefficient of the same self-excluded N_k vector.
    The k-NN search runs on GPU when one is available.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Global Hubness Analysis...")
        # De-duplicate k in order: a repeated k would add its neighbours twice.
        _k_raw = [int(k) for k in self.config.get("k_values", [10, 50, 100])]
        k_values = list(dict.fromkeys(_k_raw))
        if not k_values or min(k_values) < 1:
            return {"error": f"hubness: k_values must be positive integers, got {_k_raw}"}
        batch_size = self.config.get("batch_size", 2000)

        E = get_embeddings(model)
        if E is None:
            return {"error": "Could not extract embeddings"}

        if E.is_cuda:
            device = E.device
        elif torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        E_norm, _n_zero, _n_nonfinite, _dir = _unit_rows(E, device)
        # Rows without a direction are not ranked: a zero vector sits at chord
        # distance exactly 1 from every unit row (cosine 0.5), nearer than most
        # true neighbours in a near-isotropic table, so it became a
        # near-universal hub (test_hubness_protocol.py::test_zero_rows_are_not_ranked).
        E_norm = E_norm[_dir.to(E_norm.device)]
        occ = _k_occurrences(E_norm, k_values, batch_size)

        results = {
            "_meta_n_zero_rows": _n_zero,
            "_meta_n_nonfinite_rows": _n_nonfinite,
            "_meta_n_rows_ranked": int(E_norm.shape[0]),
            "_meta_k_values_duplicates_dropped": len(_k_raw) - len(k_values),
        }
        for k in k_values:
            results.update(_hubness_stats_from_occurrences(occ[k], k))

        return results
