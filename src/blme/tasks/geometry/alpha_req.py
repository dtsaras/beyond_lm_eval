"""``geometry_alpha_req``: the eigenspectrum decay (alpha-ReQ) and RankMe of one
point per sequence.

alpha-ReQ (Agrawal et al., NeurIPS 2022) fits a power law
``lambda_i ∝ i^-alpha`` to the eigenvalues of the centred feature covariance:
a small alpha is a flat spectrum (variance spread over many directions), a
large alpha variance concentrated in few. RankMe (Garrido et al. 2023) is
``exp`` of the entropy of the normalised eigenvalues. Li et al. 2025
(arXiv:2509.23024) report both on "the LLM's last layer representation of the
last token" of each sequence, with ``p_i = sigma_i / sum_j sigma_j`` over the
eigenvalues sigma_i of the centred covariance.

The estimator follows the released ``reptrix`` (BARL-SSL/reptrix@dbf53c5)
step for step:

* ``utils.get_eigenspectrum``: PCA explained-variance ratios of the centred
  features, ``min(2048, N, D)`` of them;
* ``alpha.get_powerlaw``: weighted least squares of ``log lambda`` on
  ``-log rank`` over ranks 11-100 (indices 10..99), weights ``1 / rank``
  computed in float32;
* ``rankme.get_rank``: ``p = lambda / sum|lambda| + 1e-7``, ``exp(-sum p log p)``.

Here the spectrum comes from an SVD of the float64-centred cloud, which equals
scikit-learn's ``PCA(svd_solver='full')`` ratios to rounding
(tests/tasks/parity/test_alpha_req_parity.py).
"""
import logging

import numpy as np
import scipy.linalg
from sklearn.metrics import r2_score

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from ._sequence_points import DEFAULT_N_SEQUENCES, readout_for, sequence_clouds

logger = logging.getLogger("blme")

# reptrix defaults: get_eigenspectrum(max_eigenvals=2048), get_alpha(fit_range=
# np.arange(10, 100)), get_rank(eps=1e-7).
MAX_EIGENVALS = 2048
FIT_RANGE = np.arange(10, 100)
RANKME_EPS = 1e-7


def _eigenspectrum(X, max_eigenvals: int = MAX_EIGENVALS) -> np.ndarray:
    """Explained-variance ratios of the centred rows of ``X``, largest first."""
    X = np.asarray(X, dtype=np.float64)
    X = X.reshape(X.shape[0], -1)
    Xc = X - X.mean(axis=0)
    s = scipy.linalg.svd(Xc, compute_uv=False, lapack_driver="gesdd")
    ev = s ** 2
    total = ev.sum()
    if not np.isfinite(total) or total <= 0:
        return np.zeros(0)
    # PCA(n_components=k).explained_variance_ratio_: the first k of the ratios
    # over the whole spectrum.
    k = min(int(max_eigenvals), Xc.shape[0], Xc.shape[1])
    return (ev / total)[:k]


def _powerlaw_alpha(eigen: np.ndarray, fit_range: np.ndarray = FIT_RANGE):
    """``(alpha, fit_R2_100)`` of ``reptrix.alpha.get_powerlaw``; NaNs when the
    spectrum is shorter than the fit window or zero inside it."""
    eigen = np.asarray(eigen, dtype=np.float64)
    trange = np.array(fit_range, dtype=int)
    if eigen.size <= int(trange.max()) or np.any(eigen[trange] <= 0):
        return float("nan"), float("nan")
    logss = np.log(np.abs(eigen))
    y = logss[trange][:, np.newaxis]
    trange = trange + 1
    nt = trange.size
    x = np.concatenate((-np.log(trange)[:, np.newaxis], np.ones((nt, 1))), axis=1)
    w = 1.0 / trange.astype(np.float32)[:, np.newaxis]
    b = np.linalg.solve(x.T @ (x * w), (w * x).T @ y).flatten()
    allrange = np.arange(0, eigen.size).astype(int) + 1
    xa = np.concatenate((-np.log(allrange)[:, np.newaxis], np.ones((eigen.size, 1))), axis=1)
    ypred = np.exp((xa * b).sum(axis=1))
    with np.errstate(divide="ignore"):
        fit_r2_100 = r2_score(y_true=logss[trange[0]:100],
                              y_pred=np.log(np.abs(ypred))[trange[0]:100])
    return float(b[0]), float(fit_r2_100)


def _rankme(eigen: np.ndarray, eps: float = RANKME_EPS) -> float:
    """``reptrix.rankme.get_rank`` on an eigenvalue spectrum."""
    eigen = np.asarray(eigen, dtype=np.float64)
    l1 = np.sum(np.abs(eigen))
    if eigen.size == 0 or not np.isfinite(l1) or l1 <= 0:
        return float("nan")
    p = eigen / l1 + eps
    return float(np.exp(-np.sum(p * np.log(p))))


def _spectral_stats(X):
    eig = _eigenspectrum(X)
    alpha, r2 = _powerlaw_alpha(eig)
    return {"alpha": alpha, "r2": r2, "rankme": _rankme(eig), "n_eig": int(eig.size)}


@register_task("geometry_alpha_req")
class AlphaReQTask(DiagnosticTask):
    """alpha-ReQ and RankMe of the last-token cloud (Li et al. 2025)."""

    TOKEN_POLICY = "last_per_sequence"
    SOURCES = {
        "paper": ("Agrawal et al., NeurIPS 2022 (alpha-ReQ, proceedings "
                  "70596d70542c51c8d9b4e423f4bf2736); arXiv:2210.02885 (Garrido et al. "
                  "2023, RankMe)"),
        "code": "BARL-SSL/reptrix@dbf53c52696282c9aeb0338fd35c88f9f74da8ae",
        "llm_precedent": ("arXiv:2509.23024 (Li et al. 2025: last layer, last token of each "
                          "of 15k FineWeb sequences of up to 512 tokens)"),
    }
    ADAPTATION_NOTE = (
        "alpha-ReQ is the decay exponent of a power law fitted to the eigenvalues of the "
        "centred covariance of one representation per sample, and RankMe the exponential of "
        "their entropy; both follow the released reptrix estimator. Li et al. (2025) apply "
        "them to LLMs at the last layer's representation of the last token of each sequence, "
        "which BLME reads as the model's output state (HF's final hidden state, after the "
        "final norm) of the last content token of each of 1000 documents. Li et al. use 15000 "
        "sequences and report the spectral metrics robust to the sample count; BLME pins N at "
        "1000 for every model. The fit window, ranks 11-100, is fixed, so the exponent ignores "
        "the zero eigenvalues that zero-padding or an isometric embedding adds; RankMe adds "
        "the estimator's 1e-7 to each of them (at most 1.6e-6 nats per added eigenvalue). "
        "An optional per-layer profile (profile: true) uses block outputs."
    )
    HEADLINE_KEYS = ("alpha_req", "rankme")
    _INV = "tests/tasks/invariance/test_geometry_alpha_req.py::"
    SIZE_INVARIANT = dict.fromkeys(
        HEADLINE_KEYS,
        (("zero_pad", "embed_isometric", "rescale", "insert_identity_blocks"),
         _INV + "test_headline_invariant"))

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running alpha-ReQ / RankMe Analysis...")
        num_samples = int(self.config.get("num_samples", DEFAULT_N_SEQUENCES))
        use_cache = self.config.get("use_cache", True)
        if dataset is None:
            from ...cache import load_default_corpus
            dataset = load_default_corpus(num_samples)

        # post-final-norm: the headline is the model's output representation
        # (Li et al.'s "last layer"), HF's final hidden state.
        sc = sequence_clouds(
            model, tokenizer, dataset, cache, n=num_samples, layer=-1,
            replace_final_norm=False, use_cache=use_cache, token_docs=0,
            readout=readout_for(self.TOKEN_POLICY))
        if not sc.last:
            return {"error": "No hidden states available for alpha-ReQ"}
        X = next(iter(sc.last.values()))
        st = _spectral_stats(X)
        out = {
            "alpha_req": st["alpha"],
            "rankme": st["rankme"],
            "_meta_alpha_req_fit_r2_100": st["r2"],
            "_meta_n_points": int(sc.n_points),
            "_meta_n_eigenvalues": st["n_eig"],
            "_meta_fit_ranks": [int(FIT_RANGE[0]) + 1, int(FIT_RANGE[-1]) + 1],
        }
        if st["n_eig"] <= int(FIT_RANGE[-1]):
            out["_meta_alpha_req_skipped_reason"] = (
                f"{st['n_eig']} eigenvalues; the fit needs ranks 11-100")

        if self.config.get("profile", False):
            prof = sequence_clouds(
                model, tokenizer, dataset, cache, n=num_samples, layer="all",
                replace_final_norm=True, use_cache=use_cache, token_docs=0,
                readout=readout_for(self.TOKEN_POLICY))
            per = [_spectral_stats(prof.last[k]) for k in sorted(prof.last)]
            out["alpha_req_per_layer"] = [p["alpha"] for p in per]
            out["rankme_per_layer"] = [p["rankme"] for p in per]
            out["_meta_profile_last_entry_is_block_output"] = bool(
                prof.info.get("last_is_block", False))
        return out
