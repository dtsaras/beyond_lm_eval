"""DiME: the augmentation invariance of each layer, as a difference of
matrix-based entropies between two augmented views of every prompt
(Skean et al., ICML 2025).

References: "Layer by Layer: Uncovering Hidden Representations in Language
Models", arXiv:2502.02013, code OFSkean/information_flow
(``metric_functions.compute_dime``, ``metric_calling.compute_on_concatenated_passes``,
``text_dataloader.text_augmentation`` / ``multiview_collate``); the estimator
"DiME: Maximizing Mutual Information by a Difference of Matrix-Based
Entropies", arXiv:2301.08164, code uk-cliplab/representation-itl
(``difference_of_entropies.doe``, ``matrix_itl.vonNeumannEntropy``).

Each prompt is augmented twice by nlpaug's SplitAug -> RandomCharAug ->
KeyboardAug chain. Each view is mean-pooled over its tokens at every HF hidden
state; per layer and view the N pooled rows are centred and L2-normalised, and
with K_A, K_B their N x N Gram matrices

    DiME = E_P[ H(K_A o P K_B P^T) ] - H(K_A o K_B),

H the von Neumann entropy of the eigenvalues normalised to sum 1 (alpha = 1),
o the Hadamard product and P a uniform permutation of the prompts (10 draws).
It is near 0 when the views of a prompt are no closer than those of different
prompts and at most log N.
"""
import contextlib
import logging
import math
import random

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from ..common import get_max_positions
from .utils import safe_unit_rows_torch

logger = logging.getLogger("blme")

N_VIEWS = 2
NLPAUG_MISSING = ("geometry_dime needs nlpaug, the augmentation library of the released code "
                  "(pip install nlpaug==1.1.11)")


# -------------------------------------------------------------- augmentation
def augmenter():
    """``text_augmentation``'s chain at nlpaug's defaults (aug_p 0.3, at most
    10 words per augmenter). Raises ImportError when nlpaug is absent."""
    import nlpaug.augmenter.char as nac
    import nlpaug.augmenter.word as naw
    import nlpaug.flow as naf
    return naf.Sequential([naw.SplitAug(), nac.RandomCharAug(), nac.KeyboardAug()])


def view_seeds(seed, index, view):
    """(random, numpy) seeds for view ``view`` of passage ``index``: a view
    does not depend on the other passages, so every tokenizer sees the same text."""
    s = np.random.SeedSequence([int(seed), int(index), int(view)]).generate_state(2)
    return int(s[0]), int(s[1])


@contextlib.contextmanager
def seeded_global_rngs(py_seed, np_seed):
    """nlpaug draws from the global ``random`` and numpy streams and takes no
    generator: seed both for the block, then restore what the caller had."""
    py_state, np_state = random.getstate(), np.random.get_state()
    random.seed(py_seed)
    np.random.seed(np_seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def augment_view(aug, text, seed, index, view):
    """One augmented view. nlpaug 1.1.11 returns a one-element list; the
    release passes its str() on, BLME the text itself."""
    with seeded_global_rngs(*view_seeds(seed, index, view)):
        out = aug.augment(text, n=1)
    return out[0] if isinstance(out, list) else out


def select_prompts(texts, tokenizer, min_tokens):
    """Indices of the passages with at least ``min_tokens`` content tokens; a
    dropped passage is not replaced, so the kept set does not grow with the corpus."""
    return [i for i, t in enumerate(texts)
            if len(tokenizer(t, add_special_tokens=False)["input_ids"]) >= min_tokens]


# ----------------------------------------------------------------- estimator
def unit_rows(x):
    """``metric_functions.normalize`` over all N rows: centre, then unit L2 rows."""
    x = x.double()
    return safe_unit_rows_torch(x - x.mean(dim=0, keepdim=True))


def von_neumann_entropy(K):
    """``matrix_itl.vonNeumannEntropy``: the positive eigenvalues, normalised to sum 1."""
    ek = torch.linalg.eigvalsh(K)
    mek = ek[ek > 0]
    mek = mek / mek.sum()
    return -torch.sum(mek * torch.log(mek))


def permutations(n, n_iters, seed):
    """The ``torch.randperm`` draws ``doe`` makes after ``torch.manual_seed(seed)``."""
    g = torch.Generator().manual_seed(int(seed))
    return [torch.randperm(n, generator=g) for _ in range(n_iters)]


def doe(Kx, Ky, perms):
    """``difference_of_entropies.doe`` at alpha = 1 on given permutations:
    E_P[H(Kx o P Ky P^T)] - H(Kx o Ky). NaN where the eigensolver fails, as
    ``compute_dime`` records it."""
    try:
        H = von_neumann_entropy(Kx * Ky)
        H_perm_avg = 0
        for idx in perms:
            H_perm_avg = H_perm_avg + von_neumann_entropy(Kx * Ky[idx, :][:, idx]) / len(perms)
    except RuntimeError:
        return float("nan")
    return float(H_perm_avg - H)


def dime_profiles(views_a, views_b, n_iters=10, seed=0):
    """DiME per layer from the pooled (uncentred) views, (N, D) per layer.

    Returns the N x N Gram profile (raw nats), the release's profile and the
    release's branch: ``compute_dime`` takes the D x D covariance when N > D,
    and ``doe`` then permutes features. The same permutations serve every layer.
    """
    n, d = views_a[0].shape
    perms_n = permutations(n, n_iters, seed)
    perms_d = permutations(d, n_iters, seed) if n > d else None
    gram, release = [], []
    for a, b in zip(views_a, views_b):
        a, b = unit_rows(a), unit_rows(b)
        g = doe(a @ a.T, b @ b.T, perms_n)
        gram.append(g)
        release.append(doe(a.T @ a, b.T @ b, perms_d) if n > d else g)
    return gram, release, ("covariance" if n > d else "gram")


def max_entropy(n, d):
    """``entropy_normalization(..., 'maxEntropy', N, D)``'s divisor."""
    return min(math.log(n), math.log(d))


def _context_limit(model, max_length):
    return min(int(max_length), get_max_positions(model, int(max_length)))


@register_task("geometry_dime")
class DiMETask(DiagnosticTask):
    """DiME profile of two augmented views per prompt (Skean et al. 2025)."""

    TOKEN_POLICY = ("paper_protocol:Skean2025 the mean over every token of each augmented "
                    "view, the tokenizer's special tokens kept")
    SOURCES = {
        "paper": ("arXiv:2502.02013 (Skean et al., ICML 2025, Layer by Layer); arXiv:2301.08164 "
                  "(Skean et al. 2023, DiME); arXiv:1211.2459 (Sanchez Giraldo et al., "
                  "matrix-based entropy)"),
        "code": ("OFSkean/information_flow@dde387a70254e21c95a6fdda2ab7db93fb5c2a00 "
                 "(metric_functions.compute_dime, text_dataloader.text_augmentation); "
                 "uk-cliplab/representation-itl@e5d8a125ca5adb76b1d148a567ceefbb81372c28 "
                 "(difference_of_entropies.doe)"),
        "llm_precedent": ("arXiv:2502.02013 (Pythia, Llama-3-8B, LLM2Vec, Mamba: mean-pooled "
                          "augmented views of WikiText prompts at every layer)"),
    }
    ADAPTATION_NOTE = (
        "Follows the release: two views per prompt from nlpaug's SplitAug, RandomCharAug and "
        "KeyboardAug chained at their defaults (p = 0.3, at most 10 words each), the mean over "
        "every token of a view at every HF hidden state (embeddings to the post-final-norm "
        "output), rows centred and L2-normalised, and doe at alpha = 1 with 10 permutations in "
        "float64. DiME is raw nats, as the release's graphs-for-paper notebook plots it, on the "
        "N x N Gram for every model: when N > D the release passes the D x D covariance, whose "
        "permutations shuffle features instead of prompts, and that value over "
        "min(log N, log D) (compute_dime's default) is _meta_. Departures (class 2): the release centres each batch of two "
        "prompts, which maps every pair to +u and -u and lifts unrelated views towards log 2 "
        "(0.59 nats on a full-rank 64-D cloud at N = 400), so BLME centres over all N; it pads "
        "views on the right with token 0 and averages the last T positions, so BLME runs each "
        "view unpadded; it feeds the str() of nlpaug's one-element list, BLME the text. "
        "Augmentation is seeded per passage and view (random and numpy restored after) and "
        "the same 10 seeded permutations serve every layer. Prompts are the first 1000 corpus "
        "passages with at least 30 tokens, not refilled, so N moves slightly with the "
        "tokenizer and DiME <= log N (the release's WikiText run keeps 472 of the first 1000 "
        "train rows). On a model with a BOS the all-token mean carries the BOS state / T, a "
        "length code two views share (Llama-3.2-1B: a length-only control scores 0.32 against "
        "0.43), so _meta_dime_profile_without_pos0 is the sink-free twin and models with and "
        "without a BOS compare only through it. dime_peak and dime_last_layer are unchanged by "
        "identity blocks and width padding; dime_peak_depth (argmax / L) moves by design."
    )
    HEADLINE_KEYS = ("dime_peak", "dime_last_layer", "dime_peak_depth")
    SIZE_INVARIANT = dict.fromkeys(
        ("dime_peak", "dime_last_layer"),
        (("insert_identity_blocks", "zero_pad_width"),
         "tests/tasks/invariance/test_geometry_dime.py::test_headline_invariant"))

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running DiME (augmentation invariance)...")
        try:
            aug = augmenter()
        except ImportError as e:
            logger.info(f"{NLPAUG_MISSING}: {e}")
            return {"error": NLPAUG_MISSING}
        import nlpaug

        num_samples = int(self.config.get("num_samples", 1000))
        min_tokens = int(self.config.get("min_tokens", 30))
        n_iters = int(self.config.get("n_iters", 10))
        offset = int(self.config.get("seed_offset", 0))
        aug_seed = int(self.config.get("augment_seed", 0)) + offset
        perm_seed = int(self.config.get("perm_seed", 0)) + offset
        limit = _context_limit(model, self.config.get("max_length", 2048))
        if dataset is None:
            from ...cache import load_default_corpus
            dataset = load_default_corpus(num_samples)
        texts = [s["text"] if isinstance(s, dict) else str(s) for s in list(dataset)[:num_samples]]
        keep = select_prompts(texts, tokenizer, min_tokens)
        if len(keep) < 2:
            return {"error": f"Fewer than 2 passages of >= {min_tokens} tokens"}

        device = next(model.parameters()).device
        bos = getattr(tokenizer, "bos_token_id", None)
        views = [None] * N_VIEWS
        # The same pool without position 0: under causal attention its state
        # is identical for every prompt with a BOS, so in the all-token mean it
        # adds c / T, a code for the view's length that two views share.
        views_np = [None] * N_VIEWS
        n_changed = n_truncated = n_bos = 0
        with torch.no_grad():
            for i in keep:
                for v in range(N_VIEWS):
                    text = augment_view(aug, texts[i], aug_seed, i, v)
                    n_changed += text != texts[i]
                    ids = tokenizer(text)["input_ids"]
                    if len(ids) > limit:
                        n_truncated += 1
                        ids = tokenizer(text, truncation=True, max_length=limit)["input_ids"]
                    n_bos += bos is not None and ids[0] == bos
                    out = model(input_ids=torch.tensor([ids], device=device),
                                output_hidden_states=True, use_cache=False)
                    # post-final-norm: the release's profile is every HF hidden
                    # state, the embeddings through the last, which is after the final norm.
                    hs = out.hidden_states
                    if views[v] is None:
                        views[v] = [[] for _ in hs]
                        views_np[v] = [[] for _ in hs]
                    for l, h in enumerate(hs):
                        views[v][l].append(h[0].double().mean(dim=0).cpu())
                        views_np[v][l].append(h[0, 1:].double().mean(dim=0).cpu())
        views_a = [torch.stack(rows) for rows in views[0]]
        views_b = [torch.stack(rows) for rows in views[1]]
        n, d = views_a[0].shape

        profile, release, branch = dime_profiles(views_a, views_b, n_iters, perm_seed)
        profile_np, _, _ = dime_profiles([torch.stack(r) for r in views_np[0]],
                                         [torch.stack(r) for r in views_np[1]], n_iters, perm_seed)
        if not np.isfinite(profile).any():
            return {"error": "DiME eigendecomposition failed at every layer"}
        L = len(profile) - 1
        peak = int(np.nanargmax(profile))
        norm = max_entropy(n, d)
        return {
            "dime_profile": profile,
            "dime_peak": float(profile[peak]),
            "dime_last_layer": float(profile[-1]),
            "dime_peak_depth": peak / L if L else 0.0,
            "_meta_dime_peak_layer": peak,
            "_meta_dime_profile_without_pos0": profile_np,
            "_meta_dime_peak_without_pos0": float(np.nanmax(profile_np)),
            "_meta_dime_last_layer_without_pos0": float(profile_np[-1]),
            "_meta_dime_profile_max_entropy": [x / norm for x in profile],
            "_meta_dime_release_profile": [x / norm for x in release],
            "_meta_dime_release_branch": branch,
            "_meta_max_entropy": norm,
            "_meta_log_n": math.log(n),
            "_meta_n_prompts": n,
            "_meta_n_too_short": len(texts) - n,
            "_meta_hidden_size": d,
            "_meta_n_views_augmented": int(n_changed),
            "_meta_n_views_truncated": int(n_truncated),
            "_meta_n_views_with_bos": int(n_bos),
            "_meta_context_limit": limit,
            "_meta_augment_seed": aug_seed,
            "_meta_perm_seed": perm_seed,
            "_meta_n_iters": n_iters,
            "_meta_nlpaug_version": getattr(nlpaug, "__version__", "unknown"),
        }
