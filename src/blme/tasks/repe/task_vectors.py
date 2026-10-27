"""``repe_task_vectors``: geometry of the RepE ClusterMean reading vector."""
import math

import numpy as np
import torch
import torch.nn.functional as F

from ..base import DiagnosticTask
from ...registry import register_task
from ..common import get_layers
from ._common import logger

# Bundled true/false contrastive pairs, used when the caller's dataset carries
# no {text_pos, text_neg} items (the default study corpus has only {text}).
_FALLBACK_PAIRS = [
    {"text_pos": "The earth revolves around the sun.",
     "text_neg": "The sun revolves around the earth."},
    {"text_pos": "Water boils at 100 degrees Celsius.",
     "text_neg": "Water boils at 0 degrees Celsius."},
    {"text_pos": "A triangle has three sides.",
     "text_neg": "A triangle has four sides."},
    {"text_pos": "Humans typically have two arms.",
     "text_neg": "Humans typically have three arms."},
    {"text_pos": "The Pacific is the largest ocean.",
     "text_neg": "The Atlantic is the largest ocean."},
]


def _contrastive_samples(dataset, num_samples):
    """The caller's contrastive pairs, else the bundled set cycled to ``num_samples``.

    The dataset is materialised once, so a generator is not exhausted by a
    preview (tests/tasks/test_repe.py).
    """
    need_fallback = False
    if dataset is None:
        samples = []
        need_fallback = True
    else:
        samples = list(dataset)[:num_samples]
        if not samples or not all(
            isinstance(s, dict) and "text_pos" in s and "text_neg" in s
            for s in samples
        ):
            need_fallback = True
    if need_fallback:
        samples = (_FALLBACK_PAIRS * ((num_samples // len(_FALLBACK_PAIRS)) + 1))[:num_samples]
    return samples


def _distinct_indices(samples):
    """Indices of distinct pairs: a cycled fallback repeats pairs, and a
    duplicate enters the pairwise cosine as an exact 1.0."""
    seen, uniq = set(), []
    for i, s in enumerate(samples):
        sig = (s.get("text_pos"), s.get("text_neg")) if isinstance(s, dict) else str(s)
        if sig not in seen:
            seen.add(sig)
            uniq.append(i)
    return uniq


@register_task("repe_task_vectors")
class TaskVectorGeometryTask(DiagnosticTask):
    """Geometry of the RepE reading vector v = mean(h+) - mean(h-) per layer.

    The vector is the released ``ClusterMeanRepReader`` direction (Zou et al.
    2023, arXiv:2310.01405) at the last token of each true/false pair, on the
    raw ``hidden_states`` tuple as ``rep_reading_pipeline.py`` reads it. What
    is reported about it (norms, per-pair direction agreement, argmax layer)
    is BLME's.
    """

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Task Vector Geometry (RepE)...")
        num_samples = self.config.get("num_samples", 5)

        device = next(model.parameters()).device
        layers = get_layers(model)
        num_layers = max(1, len(layers))

        samples = _contrastive_samples(dataset, num_samples)
        uniq_idx = _distinct_indices(samples)
        if len(samples) < 1:
            return {"error": "Need at least 1 sample with 'text_pos' and 'text_neg' keys"}

        pos_activations, neg_activations = self._collect(
            model, tokenizer, samples, num_layers, device)
        return self._summarise(pos_activations, neg_activations, uniq_idx, num_layers)

    @staticmethod
    def _collect(model, tokenizer, samples, num_layers, device):
        """Last-token state of every pair at hs[1..L] of the raw tuple."""
        pos_activations = {l: [] for l in range(num_layers)}
        neg_activations = {l: [] for l in range(num_layers)}
        # post-final-norm=False, DELIBERATELY: the released reader reads the raw
        # tuple (rep_reading_pipeline.py:23-24), so the deepest entry is post-ln_f.
        with torch.no_grad():
            for s in samples:
                ids_pos = tokenizer.encode(s["text_pos"], return_tensors="pt", truncation=True, max_length=128).to(device)
                hs_pos = list(model(ids_pos, output_hidden_states=True).hidden_states)
                ids_neg = tokenizer.encode(s["text_neg"], return_tensors="pt", truncation=True, max_length=128).to(device)
                hs_neg = list(model(ids_neg, output_hidden_states=True).hidden_states)
                for l in range(num_layers):
                    pos_activations[l].append(hs_pos[l + 1][0, -1].cpu().float())
                    neg_activations[l].append(hs_neg[l + 1][0, -1].cpu().float())
        return pos_activations, neg_activations

    @staticmethod
    def _summarise(pos_activations, neg_activations, uniq_idx, num_layers):
        results = {}
        task_vector_norms = []
        centroid_cosines = []
        pair_dir_cos_mean = []
        pair_dir_cos_min = []

        for l in range(num_layers):
            A_pos = torch.stack(pos_activations[l])
            A_neg = torch.stack(neg_activations[l])
            # float32 means, as the released reader computes them: the vector is
            # bit-pinned to ClusterMeanRepReader (tests/test_reference_parity_guard.py).
            mean_pos = A_pos.mean(dim=0)
            mean_neg = A_neg.mean(dim=0)
            v = mean_pos - mean_neg

            # Norm in float32; double only where float32 under/overflowed.
            v_norm = float(torch.linalg.vector_norm(v.float(), ord=2).item())
            if not math.isfinite(v_norm) or (v_norm == 0.0 and bool(v.any())):
                v_norm = float(torch.linalg.vector_norm(
                    A_pos.double().mean(dim=0) - A_neg.double().mean(dim=0),
                    ord=2).item())
            task_vector_norms.append(v_norm)

            # cos(mean_pos, mean_neg) is dominated by the shared centroid offset,
            # so it is provenance only.
            centroid_cosines.append(
                F.cosine_similarity(mean_pos.unsqueeze(0), mean_neg.unsqueeze(0)).item())

            # Agreement of the per-pair directions d_i = h_i+ - h_i-.
            _d = A_pos - A_neg
            if len(uniq_idx) < _d.shape[0]:
                _d = _d[torch.as_tensor(uniq_idx, dtype=torch.long)]
            if _d.shape[0] >= 2:
                _dn = F.normalize(_d, dim=1)
                _G = _dn @ _dn.t()
                _iu = torch.triu_indices(_G.shape[0], _G.shape[0], offset=1)
                _pw = _G[_iu[0], _iu[1]]
                pair_dir_cos_mean.append(float(_pw.mean()))
                pair_dir_cos_min.append(float(_pw.min()))
            else:
                pair_dir_cos_mean.append(float("nan"))
                pair_dir_cos_min.append(float("nan"))

        results["layer_task_vector_norms"] = task_vector_norms
        results["_meta_centroid_cosine"] = centroid_cosines
        results["layer_pair_direction_cos_mean"] = pair_dir_cos_mean
        results["layer_pair_direction_cos_min"] = pair_dir_cos_min
        _finite = [c for c in pair_dir_cos_mean if c == c]
        results["pair_direction_cos_mean"] = float(np.mean(_finite)) if _finite else float("nan")

        if task_vector_norms:
            results["_meta_max_norm_layer"] = int(np.argmax(task_vector_norms))
            results["max_norm_layer_fraction"] = (
                float(np.argmax(task_vector_norms))
                / max(1.0, float(len(task_vector_norms) - 1))
                if len(task_vector_norms) > 1 else float("nan"))
            results["_meta_max_norm_layer_is_deepest"] = bool(
                int(np.argmax(task_vector_norms)) == len(task_vector_norms) - 1)
            results["_meta_mean_vector_norm"] = float(np.mean(task_vector_norms))
            d_model = pos_activations[0][0].numel()
            scale = float(np.sqrt(d_model))
            results["layer_task_vector_norms_normalized"] = [
                n / scale for n in task_vector_norms
            ]
            results["mean_vector_norm_normalized"] = (
                results["_meta_mean_vector_norm"] / scale
            )

        return results
