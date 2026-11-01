"""``geometry_gride`` and ``geometry_information_imbalance``: the GRIDE
intrinsic-dimension profile and the information imbalance between layers,
on one point per sequence, as Cheng et al. (ICLR 2025, arXiv:2405.15471)
apply them to LLMs with DADApy.

GRIDE (Denti et al. 2022) estimates the intrinsic dimension at neighbour
scale ``n1 -> 2 n1`` by maximum likelihood on the ratios
``mu = r_{2 n1} / r_{n1}`` of each point's neighbour distances. The
information imbalance (Glielmo et al. 2022) from space A to space B is the
mean rank, in B, of each point's nearest neighbour in A, divided by N/2: 0
when A's neighbourhoods predict B's, about 1 when they are unrelated.

Both follow the released code step for step (sissa-data-science/DADApy@d729b88:
``Data.remove_identical_points``, ``return_id_scaling_gride`` with
``_filter_mus`` / ``_argmax_loglik`` / ``_fisher_info_scaling``, and
``_utils.metric_comparisons._return_imbalance`` on full-rank neighbour lists)
and the scripts of chengemily1/id-llm-abstraction@4a170a2 (``dadapy_id.py``,
``precompute_distances.py``, ``information_imbalance.py``).
"""
import logging
import math

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from ._sequence_points import DEFAULT_N_SEQUENCES, readout_for, sequence_clouds

logger = logging.getLogger("blme")

_EPS64 = np.finfo(np.float64).eps
_INV = "tests/tasks/invariance/test_geometry_gride.py::"
# return_id_scaling_gride defaults.
GRIDE_D0, GRIDE_D1, GRIDE_TOL = 0.001, 1000.0, 1e-7


# --------------------------------------------------------------------- GRIDE
def _unique_rows(X):
    """``Data.remove_identical_points``: np.unique over rows (reorders them)."""
    return np.unique(X, axis=0)


def _sorted_neighbour_distances(X, max_rank):
    """Euclidean distances of every point to its 0..max_rank nearest
    neighbours (self first), as ``_mus_scaling_reduce_func`` builds them:
    squared distances, sorted, square root, zeros below eps set to eps."""
    X = np.asarray(X, dtype=np.float64)
    sq = np.sum(X * X, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(d2, 0.0, out=d2)
    np.fill_diagonal(d2, 0.0)
    idx = np.argsort(d2, axis=1, kind="stable")[:, : max_rank + 1]
    dist = np.sqrt(np.take_along_axis(d2, idx, axis=1))
    tail = dist[:, 1:]
    tail[tail < _EPS64] = _EPS64
    return dist


def _filter_mus(mus):
    mus = mus.copy()
    mus[mus == 1] += 10 * _EPS64
    q3, q2 = np.percentile(mus, [95, 50])
    return mus[mus < 20 * (q3 - q2) + q2]


def _neg_dloglik_did(d, mus, n1, n2, N, eps):
    one_m_mus_d = 1.0 - mus ** (-d)
    one_m_mus_d[one_m_mus_d < 2 * eps] = 2 * eps
    summation = np.sum(((1 - n2 + n1) / one_m_mus_d + n2 - 1.0) * np.log(mus))
    return summation - (N - 1) / d


def _argmax_loglik(mus, n1, n2, d0=GRIDE_D0, d1=GRIDE_D1, eps=GRIDE_TOL):
    """Bisection on the derivative of the GRIDE log-likelihood."""
    mus = _filter_mus(mus)
    N = len(mus)
    l1 = _neg_dloglik_did(d1, mus, n1, n2, N, eps)
    while abs(d0 - d1) > eps:
        d2 = (d0 + d1) / 2.0
        l2 = _neg_dloglik_did(d2, mus, n1, n2, N, eps)
        if l2 * l1 > 0:
            d1 = d2
        else:
            d0 = d2
    return (d0 + d1) / 2.0


def _gride_scaling(X, range_max):
    """``return_id_scaling_gride(range_max)`` after ``remove_identical_points``:
    ``(ids, n2 ranks)`` for n1 = 1, 2, 4, ... up to the largest power of two
    not above ``min(N, range_max)``."""
    X = _unique_rows(np.asarray(X, dtype=np.float64))
    N = X.shape[0]
    max_rank = min(N, int(range_max))
    if max_rank < 2:
        return np.zeros(0), np.zeros(0, dtype=int)
    max_step = int(math.log(max_rank, 2))
    steps = np.array([2 ** i for i in range(max_step + 1)])
    if steps[-1] > N - 1:          # the release partitions up to rank steps[-1]
        steps = steps[:-1]
    dist = _sorted_neighbour_distances(X, int(steps[-1]))
    mus = dist[:, steps[1:]] / dist[:, steps[:-1]]
    ids = np.array([_argmax_loglik(mus[:, i].copy(), 2 ** i, 2 ** (i + 1))
                    for i in range(mus.shape[1])])
    return ids, steps[1:]


# ----------------------------------------------------- information imbalance
def _full_rank_indices(X):
    """Neighbour indices of every point, nearest first (self at column 0),
    over all N - 1 others: ``Data.compute_distances(maxk=N-1).dist_indices``."""
    X = np.asarray(X, dtype=np.float64)
    sq = np.sum(X * X, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(d2, 0.0, out=d2)
    np.fill_diagonal(d2, -1.0)          # self first, as the release's search does
    return np.argsort(d2, axis=1, kind="stable")


def _neighbourhood_overlap(idx_a, idx_b, k):
    """``NeighborhoodOverlap.return_data_overlap(k)``: the mean fraction of each
    point's k nearest neighbours (self excluded) shared by the two spaces."""
    a, b = idx_a[:, 1:k + 1], idx_b[:, 1:k + 1]
    shared = [np.intersect1d(a[i], b[i]).size for i in range(a.shape[0])]
    return float(np.mean(shared) / k)


def _information_imbalance(idx_a, idx_b):
    """``_return_imbalance(idx_a, idx_b, k=1)`` on full-rank neighbour lists:
    every nearest neighbour has a rank in B, so the release's random rank for
    a missing one never occurs."""
    N = idx_a.shape[0]
    rank_b = np.empty_like(idx_b)
    rank_b[np.arange(N)[:, None], idx_b] = np.arange(idx_b.shape[1])[None, :]
    ranks = rank_b[np.arange(N), idx_a[:, 1]]
    return float(np.mean(ranks) / (N / 2.0))


def _layer_clouds(task, model, tokenizer, dataset, cache):
    num_samples = int(task.config.get("num_samples", DEFAULT_N_SEQUENCES))
    if dataset is None:
        from ...cache import load_default_corpus
        dataset = load_default_corpus(num_samples)
    # post-final-norm: the released reference's layers are HF's hidden_states
    # 1..L (extract_final_representations.py), whose deepest entry is the
    # post-norm output state, so the profile keeps it.
    sc = sequence_clouds(
        model, tokenizer, dataset, cache, n=num_samples, layer="all",
        replace_final_norm=False, use_cache=task.config.get("use_cache", True),
        token_docs=0, readout=readout_for(task.TOKEN_POLICY))
    keys = sorted(sc.last)
    return sc, [sc.last[k].double().cpu().numpy() if isinstance(sc.last[k], torch.Tensor)
                else np.asarray(sc.last[k], dtype=np.float64) for k in keys]


@register_task("geometry_gride")
class GrideProfileTask(DiagnosticTask):
    """GRIDE intrinsic-dimension profile of the last-token cloud (Cheng et al. 2025)."""

    TOKEN_POLICY = "last_per_sequence"
    SOURCES = {
        "paper": ("arXiv:2405.15471 (Cheng et al. 2025, high-dimensional abstraction phase); "
                  "doi:10.1038/s41598-022-20991-1 (Denti et al. 2022, GRIDE)"),
        "code": "sissa-data-science/DADApy@d729b8835e771001ec7ba6ca68938b737b96edae",
        "llm_precedent": ("chengemily1/id-llm-abstraction@4a170a2d8195729a3337a54392f054780b385225 "
                          "(dadapy_id.py: last token of each sequence, every layer, GRIDE scaling)"),
    }
    ADAPTATION_NOTE = (
        "GRIDE estimates the intrinsic dimension at neighbour scale n1 -> 2 n1 from the "
        "ratios of each point's neighbour distances. Cheng et al. apply it with DADApy to "
        "the last token of each of 10000 sequences at every layer and read the "
        "high-dimensional peak of the profile; BLME uses the last content token of each of "
        "1000 documents at every HF hidden state after the embeddings, as their scripts do. "
        "They choose the scale per model by eye (most near n2 = 32 at N = 10000); BLME fixes "
        "n2 = 4 at N = 1000, the power of two nearest the same fraction of the cloud "
        "(0.004 vs 0.0032), and keeps the whole scaling profile as _meta_. The peak height "
        "is a distance statistic, unchanged by zero-padding, isometric embedding and "
        "rescaling; the peak's layer index moves when blocks are added and is _meta_."
    )
    HEADLINE_KEYS = ("gride_id_peak", "gride_id_last_layer")
    SIZE_INVARIANT = dict.fromkeys(
        HEADLINE_KEYS,
        (("zero_pad", "embed_isometric", "rescale", "insert_identity_blocks"),
         _INV + "test_headline_invariant[geometry_gride-gride_keys]"))

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running GRIDE intrinsic-dimension profile...")
        n2 = int(self.config.get("scale_n2", 4))
        range_max = int(self.config.get("range_max", 2 ** 13))
        sc, clouds = _layer_clouds(self, model, tokenizer, dataset, cache)
        if not clouds:
            return {"error": "No hidden states available for GRIDE"}
        profile, scaling, n_unique = [], [], []
        for X in clouds:
            ids, ranks = _gride_scaling(X, range_max)
            scaling.append([float(v) for v in ids])
            hit = np.nonzero(ranks == n2)[0]
            profile.append(float(ids[hit[0]]) if hit.size else float("nan"))
            n_unique.append(int(_unique_rows(X).shape[0]))
        arr = np.asarray(profile, dtype=np.float64)
        finite = np.isfinite(arr)
        out = {
            "gride_id_profile": [float(v) for v in arr],
            "gride_id_peak": float(np.nanmax(arr)) if finite.any() else float("nan"),
            "gride_id_last_layer": float(arr[-1]),
            "_meta_gride_peak_layer_frac": (float(np.nanargmax(arr) / max(1, len(arr) - 1))
                                            if finite.any() else float("nan")),
            "_meta_gride_scaling_per_layer": scaling,
            "_meta_gride_scale_n2": n2,
            "_meta_n_points": int(sc.n_points),
            "_meta_n_unique_points_min": int(min(n_unique)),
            "_meta_n_layers": len(clouds),
        }
        return out


@register_task("geometry_information_imbalance")
class InformationImbalanceTask(DiagnosticTask):
    """Information imbalance between the first, middle and last layers."""

    TOKEN_POLICY = "last_per_sequence"
    SOURCES = {
        "paper": ("arXiv:2104.15079 (Glielmo et al. 2022, PNAS Nexus, information imbalance); "
                  "arXiv:2405.15471 (Cheng et al. 2025)"),
        "code": "sissa-data-science/DADApy@d729b8835e771001ec7ba6ca68938b737b96edae",
        "llm_precedent": ("chengemily1/id-llm-abstraction@4a170a2d8195729a3337a54392f054780b385225 "
                          "(information_imbalance.py: k = 1 on full-rank Euclidean neighbour "
                          "lists of the last-token representations, between layers)"),
    }
    ADAPTATION_NOTE = (
        "The information imbalance from layer A to layer B is the mean rank, in B, of each "
        "point's nearest neighbour in A, over N/2 (k = 1, Euclidean, full-rank neighbour lists, "
        "as in Cheng et al.'s scripts). BLME scores the last content token of each of 1000 "
        "documents at every HF hidden state after the embeddings, and reports the two "
        "directions between the first and last layer, which identity blocks leave unchanged, "
        "and their difference. The full profiles to and from the last layer are kept; values "
        "for an index-selected middle layer move when blocks are added and are _meta_. Ranks "
        "are unchanged by zero-padding, isometric embedding and rescaling at fixed N."
    )
    HEADLINE_KEYS = ("ii_first_to_last", "ii_last_to_first", "ii_asymmetry_first_last")
    SIZE_INVARIANT = dict.fromkeys(
        HEADLINE_KEYS,
        (("zero_pad", "embed_isometric", "rescale", "insert_identity_blocks"),
         _INV + "test_headline_invariant[geometry_information_imbalance-ii_keys]"))

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running information imbalance between layers...")
        sc, clouds = _layer_clouds(self, model, tokenizer, dataset, cache)
        if len(clouds) < 2:
            return {"error": "Information imbalance needs at least two layers"}
        idx = [_full_rank_indices(X) for X in clouds]
        last = idx[-1]
        to_last = [_information_imbalance(i, last) for i in idx]
        from_last = [_information_imbalance(last, i) for i in idx]
        mid = len(idx) // 2
        return {
            "ii_first_to_last": to_last[0],
            "ii_last_to_first": from_last[0],
            "ii_asymmetry_first_last": from_last[0] - to_last[0],
            "ii_layer_to_last": to_last,
            "ii_last_to_layer": from_last,
            "_meta_ii_mid_to_last": to_last[mid],
            "_meta_ii_last_to_mid": from_last[mid],
            "_meta_n_points": int(sc.n_points),
            "_meta_n_layers": len(clouds),
        }


@register_task("geometry_neighborhood_overlap")
class NeighborhoodOverlapTask(DiagnosticTask):
    """Neighbourhood overlap chi^k between layers (Doimo et al. 2020; DADApy)."""

    TOKEN_POLICY = "last_per_sequence"
    SOURCES = {
        "paper": ("arXiv:2007.03506 (Doimo et al. 2020, NeurIPS, hierarchical nucleation); "
                  "arXiv:2302.00294 (Valeriani et al. 2023, geometry of hidden representations "
                  "of large transformer models)"),
        "code": "sissa-data-science/DADApy@d729b8835e771001ec7ba6ca68938b737b96edae",
        "llm_precedent": ("arXiv:2302.00294 (neighbourhood overlap with k = 30 between the "
                          "layers of transformer language and protein models)"),
    }
    ADAPTATION_NOTE = (
        "chi^k(A, B) is the mean fraction of each point's k = 30 nearest Euclidean neighbours "
        "in A that are also among its k nearest in B (DADApy return_data_overlap, the default "
        "k). BLME scores the clouds of geometry_gride and geometry_information_imbalance (the "
        "last content token of each of 1000 documents at every HF hidden state after the "
        "embeddings) and reports the overlap between the first and the last layer, and the "
        "smallest overlap between adjacent layers, the largest single reorganisation of the "
        "neighbourhoods. Identity blocks add adjacent pairs with overlap 1 and leave both "
        "unchanged; the mean adjacent overlap rises with them and is _meta_. Neighbour ranks "
        "are unchanged by zero-padding, isometric embedding and rescaling at fixed N."
    )
    HEADLINE_KEYS = ("nn_overlap_first_last", "nn_overlap_adjacent_min")
    SIZE_INVARIANT = dict.fromkeys(
        HEADLINE_KEYS,
        (("zero_pad", "embed_isometric", "rescale", "insert_identity_blocks"),
         _INV + "test_headline_invariant[geometry_neighborhood_overlap-no_keys]"))

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running neighbourhood overlap between layers...")
        k = int(self.config.get("k", 30))
        sc, clouds = _layer_clouds(self, model, tokenizer, dataset, cache)
        if len(clouds) < 2:
            return {"error": "Neighbourhood overlap needs at least two layers"}
        if clouds[0].shape[0] <= k:
            return {"error": f"Neighbourhood overlap needs more than k = {k} points"}
        idx = [_full_rank_indices(X) for X in clouds]
        to_last = [_neighbourhood_overlap(i, idx[-1], k) for i in idx]
        adjacent = [_neighbourhood_overlap(a, b, k) for a, b in zip(idx, idx[1:])]
        return {
            "nn_overlap_first_last": to_last[0],
            "nn_overlap_adjacent_min": float(min(adjacent)),
            "nn_overlap_layer_to_last": to_last,
            "nn_overlap_adjacent": adjacent,
            "_meta_nn_overlap_adjacent_mean": float(np.mean(adjacent)),
            "_meta_nn_overlap_k": k,
            "_meta_n_points": int(sc.n_points),
            "_meta_n_layers": len(clouds),
        }
