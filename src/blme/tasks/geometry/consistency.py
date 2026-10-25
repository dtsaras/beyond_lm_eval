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
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Prediction Alignment Analysis...")
        if dataset is None:
            dataset = [{"text": "The quick brown fox jumps over the lazy dog."} for _ in range(50)]

        num_samples = self.config.get("num_samples", 100)
        use_cache = self.config.get("use_cache", True)

        # A POPULATED CACHE IS NOT NECESSARILY A COMPLETE ONE.
        #
        # `is_populated` says the cache HAS been filled, not that it was filled
        # with the observations THIS task needs. A cache populated for a
        # consumer that does not request hidden states returns a stats dict
        # without the "hidden" key, and the loop below then raised
        # KeyError('hidden') -- destroying the task rather than degrading it,
        # and losing the analytic 4/3 simplex result and the real gpt2 /
        # pythia-70m / distilgpt2 headlines 0.00476083671008902 /
        # 0.00797083420412881 / 0.003904443819608007 in all three incomplete
        # cache feature combinations (codex round 13, 2026-09-08).
        #
        # This is the same class as geometry_vendi_score, geometry_magnitude,
        # geometry_isoscore and geometry_svd: accept the cache on CONTENT, not
        # on a populated flag. Checked by asking for the keys this task
        # dereferences rather than by trusting the flag.
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
        # CENTRE THE UNEMBEDDING ROWS. Adding one fixed vector v to every row
        # of W adds h.v to every logit, which cancels exactly in the softmax --
        # so it is a gauge freedom of the model's function, and no faithful
        # alignment measure may move under it. Cosine against RAW rows does
        # move: adding 100*1 to every row (max softmax probability change 0.0)
        # collapsed both corrected margins from 1.000000 to 0.004987508, and
        # subtracting a random-row cosine does not remove it because the
        # control rows carry the same offset (adversarial review, 2026-09-04).
        # Centring removes exactly that direction and nothing else.
        # `centre_and_rescale`, not `.float()` then centre.
        #
        # The margin is a COSINE, so it is scale-free and the returned exponent
        # is deliberately ignored. The declared target-minus-control margin on
        # a centred four-row simplex is exactly 4/3; this returned
        # 1.3333332538604736 at 1e-30 through 1e25 and **NaN at the finite
        # scale 2.406159650500376e38** (gpt-5.6-sol round 9, 2026-09-07),
        # because the float32 cast came before the centring and the
        # subtraction then overflowed.
        #
        # This is the fourth site found with that exact ordering defect on one
        # day, after `matrix_entropy`, `collapse` and `contextualization`, so
        # it uses the shared helper rather than a fourth hand-written copy --
        # see `centre_and_rescale` for why the promote-centre-rescale order is
        # the only one that satisfies all three failure modes at once.
        embeddings, _ = centre_and_rescale(embeddings)
        cosine_sims = []
        _residual_collisions = [0]
        _control_positions = [0]
        # ANISOTROPY BASELINES. A raw cosine to the target row is inflated by
        # the embedding space's own anisotropy: rows share a dominant
        # direction, so ANY token scores high. Without a baseline the value is
        # not evidence of target alignment at all -- measured against uniformly
        # random vocabulary rows, gpt2's margin is NEGATIVE (-0.0077) and
        # pythia-70m's is +0.00014, i.e. two models show no measurable target
        # alignment whatever their raw score (adversarial review, 2026-09-03).
        #
        # Two controls:
        #   freq-matched -- a PERMUTATION of the actual labels, which preserves
        #     the empirical token-frequency marginal exactly, so the control
        #     differs from the target only in *which* position each label is
        #     paired with.
        #   uniform -- an arbitrary vocabulary row, ignoring frequency.
        baseline_freq = []
        baseline_unif = []
        margin_tokens = []
        _coll_hits, _coll_tot = [0], [0]
        _g = torch.Generator().manual_seed(20260903)
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

            target_embs = F.embedding(labels, embeddings)
            # At this point both should be 2D of shape (N, D).

            # SCALE-EXACT normalisation. `F.normalize` squares before summing,
            # so it underflows: on a centred simplex the true margin is
            # 1.3333333 at every scale, but the naive path returned 0.009999999
            # at 1e-13 (adversarial review round 3, 2026-09-04).
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
                # DERANGEMENT, not a plain permutation. A plain shuffle leaves
                # ~1/e of positions mapped to a token that happens to equal the
                # true one (measured 2.29% here after accounting for repeats),
                # and each such collision contributes a TARGET cosine to the
                # control, biasing the margin down -- by 3.2% on the headline
                # (+0.011371 emitted against +0.011738 corrected). Rotating by
                # one guarantees position i never keeps its own label, then a
                # shuffle is applied on top; any residual collision is a
                # genuine repeat of the same token, which is not a leak.
                # SATTOLO's algorithm: a single cycle, hence a genuine
                # permutation with NO fixed point.
                #
                # The previous "derangement" was not a permutation at all:
                # `torch.where(self_hit, torch.roll(rolled, 1), rolled)` writes
                # rolled[i-1] into position i while i-1 still holds it, so the
                # output had a duplicate index and a missing one. Measured on
                # real data it corrupted 60/100 documents (gpt2) and 61/100
                # (pythia-70m), which means some positions were scored twice in
                # the control and others not at all (adversarial review,
                # 2026-09-04).
                # THE SATTOLO PERMUTATION SEEDS THE GROUPING BELOW.
                #
                # It used to be computed and then never read: the round-2
                # constructive repair replaced `_idx` with a grouping built
                # from each label's positions in NATURAL ORDER, which made the
                # control fully deterministic and tie-broken by numeric token
                # id. A pure vocabulary RELABELLING -- which leaves every raw
                # cosine bit-identical -- then moved the headline by 0.34-1.72%
                # (independent Opus review, 2026-09-04). Shuffling the
                # positions before grouping removes that dependence, and the
                # zero-collision guarantee is untouched because the rotation
                # argument depends only on the GROUP SIZES.
                _idx = torch.arange(n_lab)
                for _i in range(n_lab - 1, 0, -1):
                    _j = int(torch.randint(0, _i, (1,), generator=_g).item())
                    _idx[_i], _idx[_j] = _idx[_j].clone(), _idx[_i].clone()
                _shuf = _idx.tolist()

                # Index-derangement alone removes only trivially self-paired
                # positions, not repeats of the SAME TOKEN: residual label
                # collisions (control token == true token) remained 1.438% on
                # gpt2. Repair them by swapping with a partner for which the
                # exchange is collision-free in both directions.
                # CONSTRUCTIVE repair, not a greedy single-pass swap.
                #
                # The greedy pass only fixed a collision when it found one
                # partner whose exchange was clean in both directions, and gave
                # up otherwise: on [A,A,B,B] it left 2 of 4 collisions and
                # emitted 0.500000 where the exact non-target margin is
                # 1.000000, and on 64 labels split 32/32 it left 30/64 =
                # 46.875% and emitted 0.531250 (adversarial review,
                # 2026-09-04).
                #
                # Group the positions by label, largest group first, and rotate
                # the grouped order by the size of the LARGEST group. Every
                # position then lands on one at least `m` places away in the
                # grouped order, which is outside its own block, so NO
                # collision survives whenever the largest label occupies at
                # most half the positions -- and when it occupies more than
                # half, some collisions are arithmetically unavoidable and this
                # attains the minimum possible number of them.
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
                # Per-token margin, so the reported std matches the reported
                # mean. The previous std was the RAW cosine's (0.048174) beside
                # a mean that is a margin (0.011371), making mean +/- std
                # incoherent.
                # Subtract in float64. In float32 the difference of two
                # cosines of order 5e-2 carries ~6e-9 of rounding, which broke
                # the exact identity
                # `mean(margin) == mean(raw) - mean(control)` by 1.07e-9 --
                # small, but the identity is either exact or it is not.
                margin_tokens.extend(
                    (cos.reshape(-1).double()
                     - cos_f.reshape(-1).double()).tolist())
                # Collision rate: how often the control token IS the true one.
                #
                # This is a CORPUS property, not an estimator property, and it
                # was measured nowhere. On a corpus of repeated-token documents
                # it is 100% and `prediction_alignment_mean` collapses to
                # exactly 0.0 -- a number indistinguishable from "no target
                # alignment" -- with nothing in the output to say why.
                _coll_hits[0] += int((_lab_flat[_rolled] == _lab_flat).sum())
                _coll_tot[0] += int(n_lab)
            if _vocab > 0 and n_lab > 0:
                rnd = torch.randint(0, _vocab, (n_lab,), generator=_g)
                cos_u = (h_norm * _safe_unit(
                    F.embedding(rnd, embeddings).float())).sum(dim=-1)
                baseline_unif.extend(cos_u.reshape(-1).tolist())

        _mean = float(np.mean(cosine_sims))
        out = {
            # DEMOTED. The raw cosine carries essentially no target
            # information: replacing every target with a UNIFORMLY RANDOM
            # vocabulary row reproduces the reported value to within 0.1-1.1%
            # of its entire 1.043 cross-model range. Independently, over 236
            # gpt2 positions Spearman(cosine over vocab, logit over vocab) is
            # -0.2246 (min -0.626), the argmax agrees 3.4% of the time, and the
            # cosine-argmax is the true next token 3.8% of the time against
            # 30.5% for the logit -- so it is not a "normalized logit" either
            # (adversarial review, 2026-09-04).
            "_meta_prediction_alignment_raw_mean": _mean,
            "_meta_prediction_alignment_raw_std": float(np.std(cosine_sims)),
        }
        if _coll_tot[0]:
            out["_meta_control_collision_rate"] = float(_coll_hits[0] / _coll_tot[0])
        if baseline_freq:
            _bf = float(np.mean(baseline_freq))
            out["_meta_prediction_alignment_freq_baseline"] = _bf
            # PROMOTED to the headline: the anisotropy-corrected margin is the
            # only part that carries evidence of target alignment. Emitted
            # ONCE -- `prediction_alignment_margin` was a bit-identical
            # duplicate column, which doubles this task's weight in any
            # regularized regression.
            # Mean of the PER-TOKEN margins, not (mean raw - mean control).
            #
            # The two differ whenever the row sets differ: the control is only
            # formed for documents with >1 label, so a corpus containing
            # single-token documents leaves the baseline bit-identical while
            # the difference-of-means drifts. Counterexample: 50 two-token
            # documents move the headline +0.01111735 -> +0.01088389 (-2.10%)
            # with the baseline unchanged. `margin_tokens` is already computed
            # and is bit-identical on WikiText (adversarial review, 2026-09-04).
            out["prediction_alignment_mean"] = (
                float(np.mean(margin_tokens)) if margin_tokens else _mean - _bf)
            out["prediction_alignment_std"] = (
                float(np.std(margin_tokens)) if margin_tokens
                else float(np.std(cosine_sims)))
        if "prediction_alignment_mean" not in out:
            # No document had >1 label, so no frequency-matched control could
            # be formed. The key means "cosine MINUS a matched control"; filling
            # it with the RAW cosine silently changes the estimand under the
            # same name -- eight one-label documents returned 1.000000 there
            # with zero controls formed (adversarial review, 2026-09-04). Emit
            # NaN for the corrected quantity and put the raw cosine under its
            # own name.
            out["prediction_alignment_mean"] = float("nan")
            out["prediction_alignment_std"] = float("nan")
            out["prediction_alignment_raw_cosine_mean"] = _mean
            out["prediction_alignment_raw_cosine_std"] = float(np.std(cosine_sims))
            out["_meta_prediction_alignment_uncorrected"] = True

        if _control_positions[0]:
            out["_meta_prediction_alignment_residual_collision_frac"] = float(
                _residual_collisions[0] / _control_positions[0])
        if baseline_unif:
            _bu = float(np.mean(baseline_unif))
            out["_meta_prediction_alignment_uniform_baseline"] = _bu
            out["prediction_alignment_margin_uniform"] = _mean - _bu
        return out
