"""``repe_steering_effectiveness``: does adding the concept vector move the output
along the concept, more than a norm-matched random direction does?"""
import copy
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from ..base import DiagnosticTask
from ...registry import register_task
from ..common import get_layers
from ._common import find_final_norm, logger

# Random-control draws per layer. The 32-draw median ratio is biased high and
# drifts with the draw count; the count is disclosed via the config.
_N_RAND_DRAWS_DEFAULT = 32

# Distinct (positive, negative, neutral) probe triples. The whole set is used
# when it is the probe set: any 3 of the 8 swung the headline 12.9x.
_FALLBACK_TRIPLES = [
    {"text_pos": "This is absolutely true and correct.",
     "text_neg": "This is completely false and wrong.",
     "neutral": "The weather today is"},
    {"text_pos": "The statement is accurate and verified.",
     "text_neg": "The statement is inaccurate and unverified.",
     "neutral": "In the morning the city"},
    {"text_pos": "She answered honestly and openly.",
     "text_neg": "She answered dishonestly and evasively.",
     "neutral": "The train arrived at"},
    {"text_pos": "The report was factual and reliable.",
     "text_neg": "The report was fabricated and unreliable.",
     "neutral": "Yesterday the committee"},
    {"text_pos": "He described the events truthfully.",
     "text_neg": "He described the events deceptively.",
     "neutral": "Across the river there"},
    {"text_pos": "The measurement was precise and sound.",
     "text_neg": "The measurement was sloppy and unsound.",
     "neutral": "During the winter months"},
    {"text_pos": "Their account was consistent with the record.",
     "text_neg": "Their account contradicted the record.",
     "neutral": "The library opens at"},
    {"text_pos": "The summary faithfully reflects the source.",
     "text_neg": "The summary distorts the source.",
     "neutral": "Next to the harbour"},
]


def _probe_samples(dataset, num_samples):
    """``(samples, using_fallback, error)``.

    A dataset with no contrastive item (the default text corpus) falls back to
    the bundled triples, used whole; a mix of contrastive and plain items is an
    error. Samples are copies, so the caller's dicts are never mutated.
    """
    preview = ([s for s in list(dataset)[:num_samples]]
               if dataset is not None else [])
    has_keys = [isinstance(s, dict) and "text_pos" in s and "text_neg" in s
                for s in preview]
    using_fallback = False
    if not preview or not any(has_keys):
        reps = max(1, (num_samples + len(_FALLBACK_TRIPLES) - 1) // len(_FALLBACK_TRIPLES))
        dataset = (_FALLBACK_TRIPLES * reps)[:max(num_samples, len(_FALLBACK_TRIPLES))]
        using_fallback = True
    elif not all(has_keys):
        return None, False, {"error": "Dataset mixes items with and without "
                                      "'text_pos'/'text_neg' keys."}
    n_take = max(num_samples, len(_FALLBACK_TRIPLES)) if using_fallback else num_samples
    samples = [dict(s) for s in list(dataset)[:n_take]]
    if not samples:
        return None, using_fallback, {"error": "Need at least 1 sample."}
    for s in samples:
        if "neutral" not in s:
            s["neutral"] = "The weather today is"
    return samples, using_fallback, None


def _capture_last_block(layers, sink):
    """Hook ``layers[-1]`` to store its raw output in ``sink["h"]``."""
    def _cap(module, inputs, output):
        sink["h"] = (output[0] if isinstance(output, tuple) else output).detach()
    return layers[-1].register_forward_hook(_cap)


def _extract_task_vectors(model, tokenizer, layers, samples, num_layers, device):
    """Per-layer mean(h+) - mean(h-) at the last token of each block output.

    Returns ``(task_vectors, h_ref)`` where ``h_ref`` is the last captured raw
    deepest-block output (the provisional linearisation point).
    post-final-norm: SUBSTITUTED. The vector is injected into the pre-norm
    residual stream, so the deepest entry must be the raw block output.
    """
    sink = {}
    hook = _capture_last_block(layers, sink) if layers is not None and len(layers) > 0 else None

    def _states(out, cap):
        # post-final-norm: SUBSTITUTED by the hooked raw deepest-block output.
        hs = list(out.hidden_states)
        h = cap.get("h")
        if h is not None and tuple(h.shape) == tuple(hs[-1].shape):
            hs[-1] = h
        return hs

    task_vectors = {}
    try:
        with torch.no_grad():
            pos_acts = {l: [] for l in range(num_layers)}
            neg_acts = {l: [] for l in range(num_layers)}
            for s in samples:
                ids_pos = tokenizer.encode(s["text_pos"], return_tensors="pt",
                                           truncation=True, max_length=128).to(device)
                out_pos = model(ids_pos, output_hidden_states=True)
                cap_p = dict(sink)
                ids_neg = tokenizer.encode(s["text_neg"], return_tensors="pt",
                                           truncation=True, max_length=128).to(device)
                out_neg = model(ids_neg, output_hidden_states=True)
                cap_n = dict(sink)
                hp, hn = _states(out_pos, cap_p), _states(out_neg, cap_n)
                for l in range(num_layers):
                    pos_acts[l].append(hp[l + 1][0, -1].cpu().float())
                    neg_acts[l].append(hn[l + 1][0, -1].cpu().float())
            for l in range(num_layers):
                # Means in float64 and the contrast narrowed AFTER the
                # subtraction: float32 cannot hold a large common offset.
                mean_pos = torch.stack(pos_acts[l]).double().mean(dim=0)
                mean_neg = torch.stack(neg_acts[l]).double().mean(dim=0)
                task_vectors[l] = (mean_pos - mean_neg).to(pos_acts[l][0].dtype)
    finally:
        if hook is not None:
            hook.remove()
    return task_vectors, sink.get("h")


def _logit_dir_factory(model):
    """``make(h0)`` -> ``logit_dir(vec)``: residual vector -> unit direction in
    logit space, u = W_out J_ln(h0) vec, with the final-norm Jacobian taken by
    JVP at the steered prompt's own final state ``h0``. None without a head.

    The norm is a float32 deepcopy, so the shared model is never cast in place.
    """
    emb = model.get_output_embeddings()
    W = emb.weight if emb is not None else None
    ln = find_final_norm(model)
    if W is None:
        return None
    W32 = W.to(torch.float32)
    ln32 = copy.deepcopy(ln).float() if ln is not None else None

    def make(h0):
        lnd = ln32.to(h0.device) if (ln32 is not None and h0 is not None) else ln32

        def logit_dir(vec):
            v32 = vec.to(torch.float32)
            if lnd is not None and h0 is not None:
                # The tangent goes on the primal's device (CPU vectors, GPU h0).
                with torch.enable_grad():
                    _, jv = torch.autograd.functional.jvp(
                        lambda z: lnd(z), (h0,), (v32.to(h0.device),))
                v32 = jv.detach()
            uu = (W32 @ v32.to(W32.device)).detach().float()
            nn_ = float(torch.linalg.vector_norm(uu))
            return (uu / nn_).detach().cpu() if nn_ > 1e-12 else None
        return logit_dir
    return make


def _test_layers(num_layers):
    if num_layers > 10:
        return [0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1]
    return list(range(num_layers))


def _anchor_ids(tokenizer, smp):
    """Token ids only in text_pos / only in text_neg: a data-defined yardstick."""
    try:
        pos = set(tokenizer.encode(" " + smp["text_pos"], add_special_tokens=False))
        neg = set(tokenizer.encode(" " + smp["text_neg"], add_special_tokens=False))
    except Exception:
        return None, None
    p_only, n_only = sorted(pos - neg), sorted(neg - pos)
    if not p_only or not n_only:
        return None, None
    return (torch.tensor(p_only, dtype=torch.long),
            torch.tensor(n_only, dtype=torch.long))


def _steering_hook(vec, alpha):
    """Add ``alpha * vec`` to the block output at the last position, in the
    hidden state's own dtype (an fp32 add would promote a bf16 residual)."""
    def hook(module, input, output):
        if isinstance(output, tuple):
            out_t = output[0].clone()
            out_t[:, -1, :] += (alpha * vec).to(out_t.dtype)
            return (out_t,) + output[1:]
        else:
            out_t = output.clone()
            out_t[:, -1, :] += (alpha * vec).to(out_t.dtype)
            return out_t
    return hook


def _random_controls(tv, l_idx, n_draws, ctrl_seed):
    """Norm-matched isotropic Gaussian directions, seeded on (seed, layer) only,
    so every sample at a layer draws the same set."""
    g = torch.Generator(device="cpu")
    g.manual_seed(1234 + 7919 * l_idx + 104729 * ctrl_seed)
    tv_norm = torch.linalg.vector_norm(tv.float().cpu())
    vecs = []
    for _ in range(n_draws):
        _rv = torch.randn(tv.shape, generator=g, dtype=torch.float32)
        _rv = _rv * (tv_norm / torch.linalg.vector_norm(_rv).clamp_min(1e-12))
        vecs.append(_rv.to(tv.device).to(tv.dtype))
    return vecs


def _mean_by_layer(pairs):
    d = defaultdict(list)
    for l_idx, v in pairs:
        d[l_idx].append(v)
    return {l: float(np.mean(vs)) for l, vs in d.items()}


def _matched_denominator(triples):
    """median_r |mean_s <dlogits, u_r>|: the numerator's functional (signed mean
    over samples) applied per draw, then the median magnitude over draws."""
    by = defaultdict(list)
    for l_idx, draw_id, v in triples:
        by[(l_idx, draw_id)].append(v)
    per_draw = defaultdict(list)
    for (l_idx, draw_id), vs in by.items():
        per_draw[l_idx].append(abs(float(np.mean(vs))))
    return {l: float(np.median(vs)) for l, vs in per_draw.items()}


def _asym(a, b):
    num = a - b
    den = abs(a) + abs(b)
    return float(num / den) if den > 1e-12 else float("nan")


class _Records:
    """Per-(layer, sample) measurements collected by the injection loop."""

    def __init__(self):
        self.kl_pos = []       # (layer, KL) at +alpha
        self.kl_neg = []       # (layer, KL) at -alpha
        self.dir_pos = []      # (layer, <dlogits, u(tv_l)>) at +alpha
        self.dir_neg = []      # ... at -alpha
        self.dir_rand = []     # (layer, draw, <dlogits, u(r)>)
        self.fixed_pos = []    # (layer, <dlogits, u_hat>) on the FIXED final-layer u
        self.fixed_neg = []
        self.sem_pos = []      # (layer, anchor effect)
        self.sem_neg = []


def _measure(model, tokenizer, layers, samples, task_vectors, test_layers,
             num_layers, device, steering_alpha, n_rand_draws, ctrl_seed,
             make_logit_dir, logit_dir_fn, u_hat):
    """Inject +v, -v and the random draws at every test layer of every neutral
    prompt and record the logit-space effects.

    Two yardsticks: a per-layer u(tv_l) for the concept-vs-random ratio (both
    arms measured along their own direction) and the fixed final-layer u_hat for
    the sign family (against which a layer can point the wrong way).
    post-final-norm: SUBSTITUTED. The linearisation point is the raw deepest
    block output of the prompt being steered, captured by a hook.
    """
    rec = _Records()
    sink = {}
    hook_m = _capture_last_block(layers, sink) if layers is not None and len(layers) > 0 else None
    try:
        with torch.no_grad():
            for s in samples:
                apos, aneg = _anchor_ids(tokenizer, s)
                neutral_ids = tokenizer.encode(s["neutral"], return_tensors="pt",
                                               truncation=True, max_length=128).to(device)
                base_out = model(neutral_ids)
                base_logits = base_out.logits[0, -1]
                base_probs = F.softmax(base_logits, dim=-1)

                # Rebuild the logit-space map at THIS prompt's own final state.
                h_pt = sink.get("h")
                if h_pt is not None and make_logit_dir is not None:
                    logit_dir_fn = make_logit_dir(h_pt[0, -1].detach().to(torch.float32))
                    if logit_dir_fn is not None:
                        u_hat = logit_dir_fn(task_vectors[num_layers - 1])

                for l_idx in test_layers:
                    tv = task_vectors[l_idx].to(device)
                    rand_vecs = _random_controls(tv, l_idx, n_rand_draws, ctrl_seed)
                    runs = [(tv, +1.0, rec.kl_pos, rec.dir_pos, -1),
                            (tv, -1.0, rec.kl_neg, rec.dir_neg, -1)]
                    runs += [(rv, +1.0, None, rec.dir_rand, i) for i, rv in enumerate(rand_vecs)]
                    for vec, sgn, kl_sink, dir_sink, draw_id in runs:
                        handle = layers[l_idx].register_forward_hook(
                            _steering_hook(vec, sgn * steering_alpha))
                        try:
                            steered_logits = model(neutral_ids).logits[0, -1]
                            steered_log_probs = F.log_softmax(steered_logits, dim=-1)
                            if kl_sink is not None:
                                kl = F.kl_div(steered_log_probs, base_probs,
                                              reduction='sum', log_target=False).item()
                                kl_sink.append((l_idx, max(0.0, kl)))
                            if u_hat is None:
                                continue
                            dl = steered_logits.float().cpu() - base_logits.float().cpu()
                            if dir_sink is rec.dir_rand:
                                # A random draw is scored along its OWN logit direction.
                                ur = logit_dir_fn(vec) if logit_dir_fn is not None else None
                                if ur is not None:
                                    dir_sink.append((l_idx, draw_id, float(torch.dot(dl, ur))))
                                continue
                            uc = logit_dir_fn(tv) if logit_dir_fn is not None else u_hat
                            if uc is not None:
                                dir_sink.append((l_idx, float(torch.dot(dl, uc))))
                            (rec.fixed_pos if sgn > 0 else rec.fixed_neg).append(
                                (l_idx, float(torch.dot(dl, u_hat))))
                            if apos is not None:
                                sem = float(dl[apos].mean()) - float(dl[aneg].mean())
                                (rec.sem_pos if sgn > 0 else rec.sem_neg).append((l_idx, sem))
                        finally:
                            handle.remove()
    finally:
        if hook_m is not None:
            hook_m.remove()
    return rec


def _summarise(rec, threshold):
    """Headline and provenance keys from the recorded effects."""
    kl_by_layer = defaultdict(list)
    for l_idx, kl in rec.kl_pos:
        kl_by_layer[l_idx].append(kl)
    layer_mean_kl = {l: float(np.mean(kls)) for l, kls in kl_by_layer.items()}
    kl_values = list(layer_mean_kl.values())
    kl_layers = list(layer_mean_kl.keys())
    best_idx = int(np.argmax(kl_values))
    best_layer = kl_layers[best_idx]
    success_rate = sum(1 for v in kl_values if v > threshold) / len(kl_values)

    dir_pos = _mean_by_layer(rec.dir_pos)
    dir_rand = _matched_denominator(rec.dir_rand)
    kl_neg = _mean_by_layer(rec.kl_neg)

    # KL is quadratic in the perturbation (sign-blind), so every KL key is _meta_.
    out = {
        "_meta_steering_kl_divergence": layer_mean_kl,
        "_meta_best_steering_kl": float(kl_values[best_idx]),
        "_meta_steering_kl_divergence_reversed": kl_neg,
        "_meta_best_steering_layer": int(best_layer),
        "_meta_steering_success_rate": float(success_rate),
    }
    if not dir_pos:
        return out

    layers_ = sorted(dir_pos)
    out["layer_steering_directional_effect"] = dir_pos
    # Signed argmax: a layer moving the output the wrong way cannot win.
    best_dir = max(layers_, key=lambda l: dir_pos[l])
    out["best_directional_layer"] = int(best_dir)
    out["best_directional_effect"] = float(dir_pos[best_dir])
    out["mean_directional_effect"] = float(np.mean([dir_pos[l] for l in layers_]))

    fx_pos = _mean_by_layer(rec.fixed_pos)
    fx_neg = _mean_by_layer(rec.fixed_neg)
    da = [_asym(fx_pos[l], fx_neg.get(l, float("nan")))
          for l in sorted(fx_pos) if l in fx_neg]
    ka = [_asym(layer_mean_kl[l], kl_neg.get(l, float("nan")))
          for l in layers_ if l in kl_neg and l in layer_mean_kl]
    da = [x for x in da if x == x]
    ka = [x for x in ka if x == x]

    if dir_rand:
        usable = [l for l in layers_ if l in dir_rand and abs(dir_rand[l]) > 1e-12]
        # Signed numerator: a layer steering against its concept scores 0.
        ratios = [max(0.0, dir_pos[l]) / abs(dir_rand[l]) for l in usable]
        out["layer_steering_directional_effect_random"] = dir_rand
        out["steering_concept_over_random"] = (
            float(np.median(ratios)) if ratios else float("nan"))
        out["steering_beats_random_frac"] = (
            float(np.mean([1.0 if (dir_pos[l] > 0.0 and dir_pos[l] > abs(dir_rand[l])) else 0.0
                           for l in usable]))
            if usable else float("nan"))

    # Data-defined anchors: not positive by construction, unlike the projections.
    sp = _mean_by_layer(rec.sem_pos)
    sn = _mean_by_layer(rec.sem_neg)
    if sp:
        sl = sorted(sp)
        out["steering_semantic_effect"] = float(np.mean([sp[l] for l in sl]))
        out["steering_semantic_success_rate"] = float(np.mean(
            [1.0 if (sp[l] > 0 and sn.get(l, 0.0) < 0) else 0.0
             for l in sl if l in sn])) if sn else float("nan")
        out["_meta_steering_semantic_effect_neg"] = float(np.mean(
            [sn[l] for l in sorted(sn)])) if sn else float("nan")

    # mean|asym| ~ 1 is the control that the estimand is directional (KL's ~ 0);
    # the signed mean says whether +v points the right way.
    out["steering_sign_reversal"] = float(np.mean(np.abs(da))) if da else float("nan")
    out["_meta_kl_sign_reversal"] = float(np.mean(np.abs(ka))) if ka else float("nan")
    out["directional_consistency"] = float(np.mean(da)) if da else float("nan")
    out["_meta_kl_sign_asymmetry"] = float(np.mean(ka)) if ka else float("nan")
    out["directional_success_rate"] = float(np.mean(
        [1.0 if (fx_pos[l] > 0 and fx_neg.get(l, 0.0) < 0) else 0.0
         for l in sorted(fx_pos) if l in fx_neg])) if fx_neg else float("nan")
    return out


@register_task("repe_steering_effectiveness")
class SteeringEffectivenessTask(DiagnosticTask):
    """Effect of adding the contrastive task vector to the residual stream.

    The vector (last-token mean difference) is added at the last position of a
    neutral prompt at each test layer; the signed change of next-token logits
    along the vector's own logit-space direction is the directional effect, with
    a sign-reversal control (-v), a norm-matched random control and a
    data-defined token-anchor effect. KL is sign-blind and is ``_meta_`` only.
    """

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Steering Vector Effectiveness...")
        num_samples = self.config.get("num_samples", 3)
        steering_alpha = self.config.get("steering_alpha", 1.0)
        n_rand_draws = int(self.config.get("n_random_draws", _N_RAND_DRAWS_DEFAULT))

        device = next(model.parameters()).device
        layers = get_layers(model)
        if layers is None:
            return {"error": "Could not detect model layers."}
        num_layers = len(layers)

        # fp32 compute below 2e9 parameters: the directional columns flip sign
        # in bf16 arithmetic.
        lowp = {torch.float16, torch.bfloat16}
        saved_dtypes = None
        if (getattr(model, "dtype", None) in lowp
                and sum(p_.numel() for p_ in model.parameters()) <= 2_000_000_000):
            try:
                saved_dtypes = [(p_, p_.dtype) for p_ in model.parameters()]
                model.float()
            except Exception as e:  # pragma: no cover - defensive
                logger.info(f"steering: fp32 promotion failed ({e})")
                saved_dtypes = None

        def restore_dtypes():
            if saved_dtypes is not None:
                for p_, dt in saved_dtypes:
                    p_.data = p_.data.to(dt)

        # The shared model is restored on every exit, including an exception.
        try:
            return self._run(model, tokenizer, dataset, layers, num_layers, device,
                             num_samples, steering_alpha, n_rand_draws)
        finally:
            restore_dtypes()

    def _run(self, model, tokenizer, dataset, layers, num_layers, device,
             num_samples, steering_alpha, n_rand_draws):
        samples, using_fallback, err = _probe_samples(dataset, num_samples)
        if err is not None:
            return err

        task_vectors, h_ref = _extract_task_vectors(
            model, tokenizer, layers, samples, num_layers, device)

        # Provisional logit-space map at the last captured state; replaced per
        # prompt inside `_measure`.
        u_hat = None
        logit_dir_fn = None
        make_logit_dir = None
        try:
            make_logit_dir = _logit_dir_factory(model)
            if make_logit_dir is not None:
                logit_dir_fn = make_logit_dir(
                    h_ref[0, -1].detach().to(torch.float32) if h_ref is not None else None)
                u_hat = logit_dir_fn(task_vectors[num_layers - 1])
        except Exception as e:  # pragma: no cover - defensive
            logger.info(f"steering: could not build logit-space concept direction ({e})")

        rec = _measure(model, tokenizer, layers, samples, task_vectors,
                       _test_layers(num_layers), num_layers, device, steering_alpha,
                       n_rand_draws, int(self.config.get("seed", 42)),
                       make_logit_dir, logit_dir_fn, u_hat)
        if not rec.kl_pos:
            return {"error": "No steering results computed."}

        out = _summarise(rec, self.config.get("steering_threshold", 0.01))
        out["_meta_n_probe_triples"] = int(len(samples))
        out["_meta_probe_set_is_bundled"] = bool(using_fallback)
        return out
