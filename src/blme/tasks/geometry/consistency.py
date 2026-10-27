from ...tasks.base import DiagnosticTask
from ...registry import register_task
from ..common import get_embeddings, get_lm_head
from .utils import collect_prediction_stats, centre_and_rescale
import torch
import torch.nn.functional as F
import numpy as np
import logging
from .utils import safe_unit_rows_torch as _safe_unit
logger = logging.getLogger("blme")


def _get_output_projection_weight(model) -> torch.Tensor:
    """Return the weight matrix actually used to project the final
    hidden state into logits, preferring ``lm_head.weight`` over the
    input embedding table.

    For tied LM heads (GPT-2, Pythia, Llama 1/2, Qwen 2) these are the
    same tensor, so the choice doesn't matter. For **untied** LM heads
    (Gemma 3/4, some recent Llama checkpoints) they differ and using
    the input embedding would measure alignment with the token's
    *input* representation, not with the projection that actually
    produces the next-token logit — a different geometric object. This
    task is advertised as "prediction alignment", so the output
    projection is the correct tensor.
    """
    head = get_lm_head(model)
    if head is not None and hasattr(head, "weight"):
        return head.weight.detach()
    # Fallback to input embeddings (the old behaviour).
    return get_embeddings(model)


@register_task("geometry_prediction_alignment")
class PredictionAlignmentTask(DiagnosticTask):
    """
    Measures how well the final hidden state aligns with the target token
    row of the output projection, ANISOTROPY-CORRECTED.

    The raw cosine is NOT "essentially a normalized logit", as this docstring
    previously claimed. Measured over 236 gpt2 positions, Spearman(cosine over
    the vocabulary, logit over the vocabulary) is -0.2246 (min -0.626), the two
    argmaxes agree 3.4% of the time, and the cosine-argmax is the true next
    token 3.8% of the time against 30.5% for the logit. Nor does a high raw
    value indicate target alignment: replacing every target with a uniformly
    random vocabulary row reproduces it to within 0.1-1.1% of its entire
    cross-model range, because the embedding space is strongly anisotropic.

    The headline `prediction_alignment_mean` is therefore the MARGIN over a
    frequency-matched control (a derangement of the true labels, which holds
    the token-frequency marginal fixed while breaking the position-target
    pairing). The uncorrected cosine is retained under `_meta_`.

    SCOPE: the control is deranged WITHIN each document, not across the corpus.
    That holds the per-document token distribution fixed, which is the stricter
    control, but it is a choice and it is worth 13.0-17.2%: gpt2 reads
    +0.011443 within-document against +0.012927 with a corpus-global
    derangement. Residual label collisions (the control token happening to
    equal the true one) are **0.000%** after the repair pass -- the emitted
    margin equals a strictly-non-target margin to 8 decimals on gpt2,
    pythia-70m and distilgpt2. (An earlier version of this note cited ~1.4%;
    that figure was measured against the OLD, broken derangement and understated
    the fix.)

    Uses the output projection (``lm_head.weight``) rather than the input
    embedding table, so the metric retains its "prediction alignment"
    interpretation for untied-head architectures (Gemma 3/4, etc.). Falls
    back to input embeddings if ``lm_head`` is unavailable.
    """
    TOKEN_POLICY = "exclude_pos0"

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Prediction Alignment Analysis...")
        if dataset is None:
            dataset = [{"text": "The quick brown fox jumps over the lazy dog."} for _ in range(50)]

        num_samples = self.config.get("num_samples", 100)
        use_cache = self.config.get("use_cache", True)

        # Accept the cache on content (hidden states and labels), not on the
        # populated flag (tests/tasks/test_prediction_alignment_storage.py).
        stats = None
        if cache is not None and cache.is_populated and use_cache:
            stats, _ = cache.get_prediction_stats(num_samples=num_samples)
            if not stats or any(stats.get(k) is None for k in ("hidden", "labels")):
                logger.info(
                    "  prediction_alignment: shared cache is populated but is "
                    "missing hidden states or labels; self-collecting.")
                stats = None
        if stats is None:
            stats, _ = collect_prediction_stats(model, tokenizer, dataset, num_samples=num_samples)

        # Resolve the projection weight the model actually uses for
        # logits (lm_head.weight). Ignore the cache's ``embeddings``
        # return value, which is the input-embedding table.
        embeddings = _get_output_projection_weight(model)
        if embeddings is None:
            return {"error": "Could not access output projection"}

        embeddings = embeddings.cpu()
        # Centre the unembedding rows: adding v to every row adds h.v to every
        # logit, a softmax gauge no alignment measure may see. Promote, centre,
        # then rescale (`centre_and_rescale`) so no step over/underflows.
        embeddings, _ = centre_and_rescale(embeddings)
        cosine_sims = []
        _residual_collisions = [0]
        _control_positions = [0]
        # Anisotropy controls: a frequency-matched one (a derangement of the
        # document's own labels) and a uniform-row one.
        baseline_freq = []
        baseline_unif = []
        margin_tokens = []
        _coll_hits, _coll_tot = [0], [0]
        _g = torch.Generator().manual_seed(20260903)
        _n_p0_dropped = [0]
        _vocab = embeddings.shape[0]

        for h, labels in zip(stats["hidden"], stats["labels"]):
            # Normalize shapes: h → (N, D), labels → (N,). The cache
            # historically stores hidden flattened but labels with a batch
            # dim, so we need to reconcile them before F.embedding.
            if hasattr(h, "dim") and h.dim() == 3:
                B, T, D = h.shape
                h = h.reshape(-1, D)
            if hasattr(labels, "dim") and labels.dim() >= 2:
                labels = labels.reshape(-1)
            # Row r is position r predicting token r+1. The position-0 state
            # is the context-free sink, so its pair is dropped (TOKEN_POLICY).
            if self.TOKEN_POLICY == "exclude_pos0" and labels.shape[0] > 0:
                h, labels = h[1:], labels[1:]
                _n_p0_dropped[0] += 1
            if labels.shape[0] == 0:
                continue

            target_embs = F.embedding(labels, embeddings)
            # At this point both should be 2D of shape (N, D).

            # Scale-exact unit rows (F.normalize underflows).
            h_norm = _safe_unit(h.float())
            e_norm = _safe_unit(target_embs.float())

            cos = (h_norm * e_norm).sum(dim=-1)
            # Ensure 1D before extending to avoid nested lists.
            if cos.dim() > 1:
                cos = cos.reshape(-1)
            cosine_sims.extend(cos.tolist())

            _lab_flat = labels.reshape(-1)
            n_lab = _lab_flat.shape[0]
            if n_lab > 1:
                # Control: positions shuffled by Sattolo's algorithm (a single
                # cycle), then grouped by label and rotated by the largest
                # group's size, so no position keeps its own token whenever
                # that is arithmetically possible (minimum collisions otherwise).
                _idx = torch.arange(n_lab)
                for _i in range(n_lab - 1, 0, -1):
                    _j = int(torch.randint(0, _i, (1,), generator=_g).item())
                    _idx[_i], _idx[_j] = _idx[_j].clone(), _idx[_i].clone()
                _shuf = _idx.tolist()

                _lab_list = _lab_flat.tolist()
                _by_label = {}
                for _i in _shuf:                      # randomised order
                    _by_label.setdefault(_lab_list[_i], []).append(_i)
                _grouped = [i for _l, g in sorted(
                    _by_label.items(), key=lambda kv: (-len(kv[1]), kv[0]))
                    for i in g]
                _m = max(len(g) for g in _by_label.values())
                _order = [0] * n_lab
                for _pos, _i in enumerate(_grouped):
                    _order[_i] = _grouped[(_pos + _m) % n_lab]
                _n_collision = sum(
                    1 for _i in range(n_lab)
                    if _lab_list[_order[_i]] == _lab_list[_i])
                _residual_collisions[0] += _n_collision
                _control_positions[0] += n_lab
                _rolled = torch.tensor(_order, dtype=torch.long)
                cos_f = (h_norm * _safe_unit(
                    F.embedding(_lab_flat[_rolled], embeddings).float()
                )).sum(dim=-1)
                baseline_freq.extend(cos_f.reshape(-1).tolist())
                # Per-token margin in float64, so mean(margin) is exactly
                # mean(raw) - mean(control) and the std matches the mean.
                margin_tokens.extend(
                    (cos.reshape(-1).double()
                     - cos_f.reshape(-1).double()).tolist())
                # Collision rate: a corpus property (100% on repeated-token text).
                _coll_hits[0] += int((_lab_flat[_rolled] == _lab_flat).sum())
                _coll_tot[0] += int(n_lab)
            if _vocab > 0 and n_lab > 0:
                rnd = torch.randint(0, _vocab, (n_lab,), generator=_g)
                cos_u = (h_norm * _safe_unit(
                    F.embedding(rnd, embeddings).float())).sum(dim=-1)
                baseline_unif.extend(cos_u.reshape(-1).tolist())

        _mean = float(np.mean(cosine_sims))
        out = {
            # The raw cosine is reproduced by uniformly random rows to within
            # 0.1-1.1% of its cross-model range: _meta_.
            "_meta_prediction_alignment_raw_mean": _mean,
            "_meta_prediction_alignment_raw_std": float(np.std(cosine_sims)),
        }
        if _coll_tot[0]:
            out["_meta_control_collision_rate"] = float(_coll_hits[0] / _coll_tot[0])
        if baseline_freq:
            _bf = float(np.mean(baseline_freq))
            out["_meta_prediction_alignment_freq_baseline"] = _bf
            # Headline: mean of the per-token margins over the matched control.
            out["prediction_alignment_mean"] = (
                float(np.mean(margin_tokens)) if margin_tokens else _mean - _bf)
            out["prediction_alignment_std"] = (
                float(np.std(margin_tokens)) if margin_tokens
                else float(np.std(cosine_sims)))
        if "prediction_alignment_mean" not in out:
            # No control could be formed: NaN, with the raw cosine under its
            # own name (never the raw value under the corrected key).
            out["prediction_alignment_mean"] = float("nan")
            out["prediction_alignment_std"] = float("nan")
            out["prediction_alignment_raw_cosine_mean"] = _mean
            out["prediction_alignment_raw_cosine_std"] = float(np.std(cosine_sims))
            out["_meta_prediction_alignment_uncorrected"] = True

        if _control_positions[0]:
            out["_meta_prediction_alignment_residual_collision_frac"] = float(
                _residual_collisions[0] / _control_positions[0])
        out["_meta_n_position0_pairs_dropped"] = int(_n_p0_dropped[0])
        if baseline_unif:
            _bu = float(np.mean(baseline_unif))
            out["_meta_prediction_alignment_uniform_baseline"] = _bu
            out["prediction_alignment_margin_uniform"] = _mean - _bu
        return out
