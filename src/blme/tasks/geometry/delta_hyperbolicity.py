"""Relative Gromov delta-hyperbolicity of a prompt's token cloud: how
tree-like the representation of one text is (Khrulkov et al. 2020, as
HypLoRA applies it to LLMs).

References: Yang et al., "Hyperbolic Fine-Tuning for Large Language Models",
arXiv:2410.04010 (NeurIPS 2025); code marlin-codes/HypLoRA
(``utils/delta_hyperbolicity.py``: ``delta_hyp``, ``get_delta``, derived from
Khrulkov et al.'s hyptorch/delta.py). The hidden-state profile follows the
per-layer reading of arXiv:2609.07053 (Yang et al., "A Hyperbolicity Atlas of
Large Language Model Hidden States").

For a point cloud with Euclidean distances d and base point p = the first
point, the Gromov products are (x|y)_p = (d(x,p) + d(y,p) - d(x,y)) / 2 and

    delta = max_{x,y} [ max_z min((x|z)_p, (z|y)_p) - (x|y)_p ],

reported relative to the cloud's diameter as 2 delta / diam (0 for a tree
metric). It also falls with dimension, since distances concentrate (64 uniform
points on S^1: 0.58, S^767: 0.11, S^4095: 0.05; an equilateral cloud: 0), so
a low value across widths is not by itself tree-likeness.
"""
import logging

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from ..common import get_max_positions

logger = logging.getLogger("blme")


def _distances(X):
    """``scipy.spatial.distance_matrix``, as the release computes them."""
    from scipy.spatial import distance_matrix
    X = np.asarray(X, dtype=np.float64)
    return distance_matrix(X, X)


def delta_hyp(dismat, chunk=64):
    """``delta_hyp(dismat)`` with base point 0, the max-min product taken over
    row chunks (the release materialises an N x N x N array)."""
    row = dismat[0, :][None, :]
    col = dismat[:, 0][:, None]
    XY = 0.5 * (row + col - dismat)
    best = -np.inf
    for s in range(0, XY.shape[0], chunk):
        maxmin = np.max(np.minimum(XY[s:s + chunk, :, None], XY[None, :, :]), axis=1)
        best = max(best, float(np.max(maxmin - XY[s:s + chunk])))
    return best


def relative_delta(X):
    """``get_delta`` on all points: 2 delta / diam (0 for a zero diameter)."""
    D = _distances(X)
    diam = float(D.max())
    return 2.0 * delta_hyp(D) / diam if diam > 0 else 0.0


@register_task("geometry_delta_hyperbolicity")
class DeltaHyperbolicityTask(DiagnosticTask):
    """Relative delta-hyperbolicity per prompt: input embeddings and every layer."""

    TOKEN_POLICY = ("paper_protocol:Yang2025 the first 64 tokens of each prompt; hidden-state "
                    "layers without position 0")
    SOURCES = {
        "paper": ("arXiv:2410.04010 (Yang et al., NeurIPS 2025, Hyperbolic Fine-Tuning for LLMs); "
                  "arXiv:1904.02239 (Khrulkov et al. 2020, delta estimator); arXiv:2609.07053 "
                  "(per-layer hidden states)"),
        "code": "marlin-codes/HypLoRA@c888e163c462b7202ef57f44e62e49526e3add5b",
        "llm_precedent": ("arXiv:2410.04010 (token input embeddings of each prompt, mean 2 delta / "
                          "diam over prompts, LLaMA / Gemma / Qwen)"),
    }
    ADAPTATION_NOTE = (
        "delta_rel_input_embedding follows the release: the input-embedding lookup of each "
        "prompt's tokens (special tokens as the model adds them), Euclidean distances, "
        "Khrulkov's delta with the first point as base point, 2 delta / diam, the mean over "
        "prompts. delta grows with the number of points, so every cloud is the first 64 tokens "
        "of a passage of at least 1000 bytes (the same passages for every tokenizer), where the "
        "release takes whole prompts of any length (class 2). The per-layer profile applies the "
        "same estimator to HF's hidden states 1..L (the last after the final norm) without "
        "position 0, whose sink row would otherwise set the diameter (Atlas pitfall; class 2). "
        "Ratios of distances are unchanged by rescaling, rotation and zero-padding, and the "
        "embedding value and the last layer are unchanged by inserted identity blocks."
    )
    HEADLINE_KEYS = ("delta_rel_input_embedding", "delta_rel_last_layer", "delta_rel_min")
    SIZE_INVARIANT = dict.fromkeys(
        HEADLINE_KEYS,
        (("insert_identity_blocks",),
         "tests/tasks/invariance/test_geometry_delta_hyperbolicity.py::test_headline_invariant"))

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running delta-hyperbolicity...")
        num_samples = int(self.config.get("num_samples", 200))
        n_points = int(self.config.get("n_points", 64))
        n_points = min(n_points, get_max_positions(model, n_points))
        min_bytes = int(self.config.get("min_bytes", 1000))
        # The library's whole corpus, so the prompts do not depend on how large
        # a corpus the co-running tasks asked for.
        if self.config.get("use_dataset", False):
            pool = list(dataset)
        else:
            from ...cache import default_corpus_size, load_default_corpus
            pool = load_default_corpus(default_corpus_size())
        texts = [d["text"] if isinstance(d, dict) else str(d) for d in pool]
        texts = [t for t in texts if len(t.encode("utf-8")) >= min_bytes][:num_samples]
        device = next(model.parameters()).device
        emb = model.get_input_embeddings()

        emb_deltas, layer_deltas, n_short = [], None, 0
        with torch.no_grad():
            for text in texts:
                ids = tokenizer(text, truncation=True, max_length=n_points)["input_ids"]
                if len(ids) < n_points:
                    n_short += 1
                    continue
                ids_t = torch.tensor([ids], device=device)
                emb_deltas.append(relative_delta(emb(ids_t)[0].float().cpu().numpy()))
                out = model(input_ids=ids_t, output_hidden_states=True, use_cache=False)
                # post-final-norm: the profile reads HF's hidden_states 1..L,
                # the last after the final norm.
                hs = out.hidden_states[1:]
                if layer_deltas is None:
                    layer_deltas = [[] for _ in hs]
                for l, h in enumerate(hs):
                    layer_deltas[l].append(relative_delta(h[0, 1:].float().cpu().numpy()))
        if not emb_deltas:
            return {"error": f"No passage reaches {n_points} tokens"}
        profile = [float(np.mean(v)) for v in layer_deltas]
        return {
            "delta_rel_input_embedding": float(np.mean(emb_deltas)),
            "delta_rel_last_layer": profile[-1],
            "delta_rel_min": float(min(profile)),
            "delta_rel_profile": profile,
            "_meta_delta_rel_input_embedding_std": float(np.std(emb_deltas)),
            "_meta_n_prompts": len(emb_deltas),
            "_meta_n_prompts_too_short": n_short,
            "_meta_n_points": n_points,
        }
