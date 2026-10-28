"""``repe_task_vectors``: geometry of the RepE ClusterMean reading vector."""
import math

import numpy as np
import torch
import torch.nn.functional as F

from ..base import DiagnosticTask
from ...registry import register_task
from ..common import get_layers
from ._common import logger
from .concept_separability import _rep_token

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


def _consensus_energy(D, H):
    """U-statistic estimate of ||E d||^2 over the mean squared residual norm.

    ``D`` (N, d) holds the per-pair differences d_i = h_i+ - h_i-, ``H`` the
    states they came from. mean_{i != j} <d_i, d_j> is unbiased for ||E d||^2
    (the plain ||mean d||^2 carries + tr(Sigma)/N), and dividing by mean ||h||^2
    removes the residual scale. Signed: <= 0 means no consensus. NaN for N < 2.
    """
    D = D.double()
    n = D.shape[0]
    if n < 2:
        return float("nan")
    G = D @ D.t()
    u = (G.sum() - torch.trace(G)) / (n * (n - 1))
    h2 = (H.double() ** 2).sum(dim=1).mean()
    return float(u / h2) if h2 > 0 else float("nan")


@register_task("repe_task_vectors")
class TaskVectorGeometryTask(DiagnosticTask):
    """Geometry of the RepE reading vector v = mean(h+) - mean(h-) per layer.

    The vector is the released ``ClusterMeanRepReader`` direction (Zou et al.
    2023, arXiv:2310.01405) at the last token of each true/false pair, on the
    raw ``hidden_states`` tuple as ``rep_reading_pipeline.py`` reads it. What
    is reported about it (norms, per-pair direction agreement, argmax layer)
    is BLME's.
    """

    TOKEN_POLICY = "paper_protocol:Zou2023 rep_token=-1"
    SOURCES = {
        "paper": "arXiv:2310.01405",
        "code": "andyzoujm/representation-engineering@5455d8a375d5fb1cb191f9ebcd089b7c21e9a31e",
        "llm_precedent": ("Turner et al. 2023 arXiv:2308.10248 App. F (residual-norm "
                          "denominator); Hendel et al. 2023 arXiv:2310.15916 and Todd et al. "
                          "2024 arXiv:2310.15213 are different 'task vector' objects"),
    }
    ADAPTATION_NOTE = (
        "The vector is the RepE ClusterMean reading vector (mean last-token state of "
        "true statements minus that of their false minimal pairs, at every entry of the "
        "raw hidden-state tuple), exactly as the released reader computes it. What is "
        "reported about it is BLME's: whether the per-pair contrasts agree "
        "(pair_direction_cos_mean, the mean pairwise cosine of d_i = h_i+ - h_i-) and "
        "how large the consensus is (consensus_energy_rel, the U-statistic "
        "mean_{i != j} <d_i, d_j> over the mean squared residual norm, averaged over "
        "layers). Hendel-style ICL task vectors and Todd-style function vectors were "
        "not used: they need ICL task suites and causal head search. A raw or "
        "sqrt(d)-divided norm was rejected because it moves under isometric embedding, "
        "under a function-preserving residual rescale, and with the number of pairs."
    )
    HEADLINE_KEYS = ("consensus_energy_rel", "pair_direction_cos_mean", "max_norm_layer_fraction")
    # max_norm_layer_fraction is an argmax-layer fraction: it moves under
    # identity-block insertion, so it has no entry here. consensus_energy_rel is
    # also flat in the pair count:
    # test_repe_task_vectors.py::test_consensus_energy_is_unbiased_in_the_pair_count.
    SIZE_INVARIANT = dict.fromkeys(
        ("consensus_energy_rel", "pair_direction_cos_mean"),
        (("residual_scale_gauge", "zero_pad_width"),
         "tests/tasks/invariance/test_repe_task_vectors.py::"
         "test_consensus_energy_invariant_under_residual_gauge_and_width_padding"))

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
            model, tokenizer, samples, num_layers, device, _rep_token(self.TOKEN_POLICY))
        out = self._summarise(pos_activations, neg_activations, uniq_idx, num_layers)
        # The pair count the estimates actually rest on.
        out["_meta_n_pairs"] = len(samples)
        out["_meta_n_distinct_pairs"] = len(uniq_idx)
        return out

    @staticmethod
    def _collect(model, tokenizer, samples, num_layers, device, rep_token):
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
                    pos_activations[l].append(hs_pos[l + 1][0, rep_token].cpu().float())
                    neg_activations[l].append(hs_neg[l + 1][0, rep_token].cpu().float())
        return pos_activations, neg_activations

    @staticmethod
    def _summarise(pos_activations, neg_activations, uniq_idx, num_layers):
        results = {}
        task_vector_norms = []
        consensus = []
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
            _h = torch.cat([A_pos, A_neg])
            if len(uniq_idx) < _d.shape[0]:
                _u = torch.as_tensor(uniq_idx, dtype=torch.long)
                _d = _d[_u]
                _h = torch.cat([A_pos[_u], A_neg[_u]])
            consensus.append(_consensus_energy(_d, _h))
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
        # Size-robust consensus magnitude (tests/tasks/invariance/test_repe_task_vectors.py).
        results["layer_consensus_energy_rel"] = consensus
        _fc = [c for c in consensus if c == c]
        results["consensus_energy_rel"] = float(np.mean(_fc)) if _fc else float("nan")

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
            # /sqrt(d) moves x 1/sqrt(k) under isometric embedding and with the
            # residual scale, and falls as N^-1/2: _meta_ only.
            results["_meta_layer_task_vector_norms_normalized"] = [
                n / scale for n in task_vector_norms
            ]
            results["_meta_mean_vector_norm_normalized"] = (
                results["_meta_mean_vector_norm"] / scale
            )

        return results
