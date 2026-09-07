from ...tasks.base import DiagnosticTask
from ...registry import register_task
from ..common import get_embeddings, apply_lm_head
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import logging
logger = logging.getLogger("blme")


def _canonical_alpha(alpha: float) -> float:
    """Round alpha to avoid string-key collisions in convexity lookups."""
    return round(float(alpha), 6)


def _alpha_label(alpha: float) -> str:
    """Human-readable alpha label for exported metric keys."""
    a = _canonical_alpha(alpha)
    if a == 0.0:
        return "0.0"
    if a == 1.0:
        return "1.0"
    if a == 0.5:
        return "0.5"
    return f"{a:.6g}"


def _slerp(h1, h2, alpha):
    """Spherical linear interpolation between two vectors."""
    # SCALE-EXACT unit vectors, float32-first.
    #
    # SLERP and the task's radial projection are positively homogeneous, so the
    # transform h -> c*h with W -> W/c (which preserves the endpoint logits to
    # 1.073e-7) must leave the measured gap unchanged. It did not: the required
    # gap 0.131697065 became **0.0 at c=1e-25**, 5.960e-8 at c=1e-13, and NaN
    # at c=1e25 (gpt-5.6-sol round 9, 2026-09-07).
    #
    # `F.normalize` divides by `max(||x||, 1e-12)` and `||x||` SQUARES before
    # summing, so at 1e-25 the float32 norm underflows to 0, the 1e-12 floor
    # takes over, and the "unit" vectors are ~1e-13 in length -- their dot
    # product is then nothing like a cosine and `omega` is meaningless.
    #
    # `F.normalize` stays authoritative wherever its norm is finite and
    # nonzero, so ordinary data is bit-for-bit unchanged; the max-factoring
    # path is used only where it has already failed. Same structure as the
    # `dynamics_coe` repair.
    def _unit(v):
        n = v.norm(dim=-1, keepdim=True)
        if bool(torch.isfinite(n).all()) and float(n.min()) > 0.0:
            return v / n
        m = v.abs().amax(dim=-1, keepdim=True)
        if not bool(torch.isfinite(m).all()) or float(m.max()) == 0.0:
            return F.normalize(v, dim=-1)
        vs = v / m.clamp_min(torch.finfo(v.dtype).tiny)
        return vs / vs.norm(dim=-1, keepdim=True).clamp_min(
            torch.finfo(vs.dtype).tiny)

    h1_norm = _unit(h1)
    h2_norm = _unit(h2)
    dot = torch.clamp((h1_norm * h2_norm).sum(), -1.0, 1.0)
    omega = torch.acos(dot)
    if omega.abs() < 1e-6:
        # Vectors are nearly parallel, fall back to lerp
        return (1 - alpha) * h1 + alpha * h2
    sin_omega = torch.sin(omega)
    return (torch.sin((1 - alpha) * omega) / sin_omega) * h1 + \
           (torch.sin(alpha * omega) / sin_omega) * h2


@register_task("dynamics_interpolation")
class LatentInterpolationTask(DiagnosticTask):
    """
    Interpolates between two hidden states in latent space.
    Measures entropy of decoded predictions along the path (convexity check).
    Uses norm-corrected linear interpolation and slerp for comparison.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Latent Interpolation...")
        num_pairs = self.config.get("num_pairs", 50)
        steps = self.config.get("steps", 10)
        num_samples = self.config.get("num_samples", 10)

        device = next(model.parameters()).device

        if dataset is None:
            from ...cache import load_default_corpus
            dataset = load_default_corpus(num_samples)

        # THE POOL IS `num_samples`, NOT "whatever the shared corpus happens to
        # be". `core.py` sizes the shared corpus to max(num_samples) over ALL
        # requested tasks, and this task drew its pairs from the WHOLE list, so
        # the answer depended on which other tasks you happened to run
        # alongside it. Same model, same corpus source, same seed (red-team
        # review, 2026-09-05):
        #
        #   evaluate(["dynamics_interpolation"])                     -> convexity_gap 1.4415232
        #   evaluate(["dynamics_interpolation","dynamics_stability"]) -> convexity_gap 1.8230699  (+26.5%)
        #
        # Every one of the 22 emitted floats moved, and the deltas were exactly
        # 0.0 on rerun, so this was the co-runner set and not noise:
        # `dynamics_stability`'s num_samples 5000 dragged the pool 200 -> 1747
        # documents. The sweep pinned `dataset=load_corpus(500)` so it was at
        # least consistent WITHIN the sweep, but the shipped `blme` CLI passes
        # no dataset at all, so a package user's number depended on their task
        # list. Truncating here makes the declared knob the actual pool.
        samples = list(dataset)[:num_samples] if num_samples else list(dataset)
        if len(samples) < 2: return {"error": "Need at least 2 samples"}

        entropies = defaultdict(list)
        slerp_entropies = defaultdict(list)
        base_alphas = np.linspace(0, 1, steps)
        alphas = np.array(sorted(set(np.concatenate([
            base_alphas,
            np.array([0.0, 0.5, 1.0]),
        ]))))

        # Seeded RNG for reproducible pair sampling across runs. `seed` is
        # HONOURED: it was hard-coded to 0, so the study's `seed: 42` -- and
        # every other seed a user set -- was silently ignored, which made the
        # knob look like a control it was not (red-team review, 2026-09-05).
        # The default stays 0, so an unconfigured run is unchanged.
        import random as _random
        _rng = _random.Random(int(self.config.get("seed", 0)))
        count = 0
        attempts = 0
        # Cap attempts so unusable samples (rows lacking text/input_ids, or
        # whose forward yields <2 hidden states) can't spin the loop forever —
        # the old `while count < num_pairs` never progressed on such rows.
        max_attempts = max(20, num_pairs * 20)
        with torch.no_grad():
            while count < num_pairs and attempts < max_attempts:
                attempts += 1
                s1, s2 = _rng.sample(samples, 2)

                h_states = []
                for s in [s1, s2]:
                    if isinstance(s, str):
                        inp = tokenizer(s, return_tensors="pt").to(device)
                    elif 'text' in s:
                        inp = tokenizer(s['text'][:128], return_tensors="pt").to(device)
                    elif 'input_ids' in s:
                        inp = {'input_ids': torch.tensor(s['input_ids']).long().unsqueeze(0).to(device)}
                    else: continue

                    out = model(**inp, output_hidden_states=True)
                    # post-final-norm: DELIBERATELY post-norm, and REQUIRED here. The
                    # interpolated vector is fed back through the LM head, and the head
                    # consumes the post-`ln_f` vector -- interpolating raw block outputs
                    # and then decoding them would decode vectors the model never sees.
                    h_states.append(out.hidden_states[-1][0, -1].float())

                if len(h_states) < 2: continue

                h1, h2 = h_states

                for alpha in alphas:
                    alpha_key = _canonical_alpha(alpha)
                    # Norm-corrected linear interpolation
                    h_interp = (1 - alpha) * h1 + alpha * h2
                    target_norm = (1 - alpha) * h1.norm() + alpha * h2.norm()
                    # `clamp_min(tiny)`, not `+ 1e-10`. An ABSOLUTE epsilon inside a
                    # scale-free rescale hard-codes a magnitude into an
                    # operation that has none; it biases every small vector
                    # toward the origin instead of guarding only the zero
                    # one. Two absolute epsilons caused real NaNs elsewhere
                    # today (`stability`'s clamp_min(1e-10) broke the
                    # noise_std=0 axiom, `coe`'s z_mag<=1e-12 nulled a whole
                    # metric family at 1e-30), so this file's were fixed too
                    # rather than left as "probably benign".
                    h_interp = h_interp * (target_norm / h_interp.norm().clamp_min(
                        torch.finfo(h_interp.dtype).tiny))

                    try:
                        logits = apply_lm_head(model, h_interp.unsqueeze(0))
                    except RuntimeError:
                        E = get_embeddings(model).to(device)
                        logits = h_interp.unsqueeze(0) @ E.float().T

                    probs = F.softmax(logits, dim=-1)
                    entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).item()
                    entropies[alpha_key].append(entropy)

                    # Slerp interpolation for comparison
                    h_slerp = _slerp(h1, h2, alpha)
                    # Restore magnitude via same norm target
                    # NOTE: renormalizing slerp to the SAME target_norm as the
                    # lerp path makes the two arms IDENTICAL at exactly the
                    # alphas the summaries read.
                    #
                    # Slerp's coefficients are sin((1-a)w)/sin w and
                    # sin(a w)/sin w. At a = 0.5 they are EQUAL, so h_slerp is
                    # proportional to h1 + h2, i.e. to h_lerp; forcing both to
                    # the same norm makes them the same vector. At a in {0, 1}
                    # both collapse to an endpoint. Those are the three alphas
                    # `convexity_gap` uses, which is why `convexity_gap` and
                    # `slerp_convexity_gap` agree to 1.3e-12 in float64 and
                    # <= 4.2e-6 end-to-end on real models -- one degree of
                    # freedom shipped as two features (adversarial review,
                    # 2026-09-03).
                    #
                    # The renormalization is kept because it is what makes the
                    # two arms comparable at all; the DUPLICATE SUMMARY is
                    # dropped below instead, which is the honest fix.
                    h_slerp = h_slerp * (target_norm / h_slerp.norm().clamp_min(
                        torch.finfo(h_slerp.dtype).tiny))

                    try:
                        logits_s = apply_lm_head(model, h_slerp.unsqueeze(0))
                    except RuntimeError:
                        E = get_embeddings(model).to(device)
                        logits_s = h_slerp.unsqueeze(0) @ E.float().T

                    probs_s = F.softmax(logits_s, dim=-1)
                    entropy_s = -(probs_s * (probs_s + 1e-10).log()).sum(dim=-1).item()
                    slerp_entropies[alpha_key].append(entropy_s)

                count += 1

        if count == 0:
            return {"error": "No usable interpolation pairs (samples lack text/input_ids)"}

        results = {}
        # `_meta_` on the WHOLE per-alpha family, not just the endpoints.
        #
        # `LEAKY_PREFIXES` excludes `interp_entropy_0.0` and `_1.0` because they
        # are the model's OWN next-token entropy -- a direct capability measure.
        # But every other alpha is a point on a path between those two, so the
        # leak simply walks in through the next one along: `interp_entropy_0.111111`
        # is spearman **+0.9394** with the excluded endpoint entropy
        # (independent Opus review, 2026-09-05). Excluding two members of a
        # family whose remaining members are near-copies of them is not an
        # exclusion.
        #
        # The task's actual estimand is `convexity_gap` -- how much MORE
        # uncertain the model is midway along the path than at its ends -- and
        # that is what it now contributes. The per-alpha profile stays emitted
        # as a diagnostic.
        for alpha_key, vals in entropies.items():
            results[f"_meta_interp_entropy_{_alpha_label(alpha_key)}"] = float(np.mean(vals))
        # The slerp arm is a COMPARISON diagnostic, not a second measurement,
        # so all of it is `_meta_`. Measured over 6 models (red-team review,
        # 2026-09-05):
        #
        #   interp_entropy_0.0 == slerp_entropy_0.0   BIT-IDENTICAL, 6/6
        #   interp_entropy_1.0 == slerp_entropy_1.0   BIT-IDENTICAL, 6/6
        #   interp_entropy_0.5 vs slerp_entropy_0.5   pearson 0.99999999,
        #                                             max rel 5.7e-5
        #
        # The endpoints are identical BY DEFINITION -- slerp and lerp coincide
        # at alpha in {0, 1} -- and `LEAKY_PREFIXES` caught those two pairs. It
        # did NOT catch the alpha=0.5 pair, which `is_excluded_predictor`
        # returned KEPT for on both sides: one degree of freedom, two columns,
        # and a sparse selection splitting a coefficient between them.
        for alpha_key, vals in slerp_entropies.items():
            results[f"_meta_slerp_entropy_{_alpha_label(alpha_key)}"] = float(np.mean(vals))

        mid = float(np.mean(entropies[_canonical_alpha(0.5)]))
        end = (
            float(np.mean(entropies[_canonical_alpha(0.0)]))
            + float(np.mean(entropies[_canonical_alpha(1.0)]))
        ) / 2
        results["convexity_gap"] = mid - end
        # DISCLOSURE, not a fix. `convexity_gap` is EXACTLY
        # `interp_entropy_0.5 - (interp_entropy_0.0 + interp_entropy_1.0)/2`
        # (max deviation 0.0 over 6 models), and both endpoint terms are in
        # `LEAKY_PREFIXES` precisely because they are the model's OWN
        # next-token entropy -- a direct capability measure. So a KEPT column
        # contains an excluded quantity at coefficient -1 (pearson +0.2365 /
        # spearman +0.5645 with it over 18 models -- the +0.398 first
        # recorded here does not reproduce -- but +0.8887 with
        # `interp_entropy_0.5`, so the leak is reduced, not removed). That is not large enough to call the gap a
        # relabelled capability measure, and the harness cannot decide it; the
        # decomposition is emitted so the analysis can (red-team review,
        # 2026-09-05; recorded as a paper action in RERUN_LEDGER.md).
        results["_meta_convexity_gap_endpoint_entropy_mean"] = end

        mid_s = float(np.mean(slerp_entropies[_canonical_alpha(0.5)]))
        end_s = (
            float(np.mean(slerp_entropies[_canonical_alpha(0.0)]))
            + float(np.mean(slerp_entropies[_canonical_alpha(1.0)]))
        ) / 2
        # _meta_: identical to `convexity_gap` by construction (see the
        # renormalization note above) -- 1.3e-12 in float64, <= 4.2e-6
        # end-to-end. Emitting both puts one degree of freedom into the feature
        # matrix twice, which a sparse selection then splits a coefficient
        # between.
        results["_meta_slerp_convexity_gap"] = mid_s - end_s

        return results
