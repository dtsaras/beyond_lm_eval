"""Geometry utility functions — universal HuggingFace compatibility.

Uses `output_hidden_states=True` instead of architecture-specific hooks
so that all functions work with any AutoModelForCausalLM.
"""

import torch
import numpy as np
from tqdm import tqdm
from ..common import get_embeddings
import os
import tempfile
import atexit
import logging
import math

# `logger` was already referenced by the perplexity-window warning below
# without ever being defined, so that branch raised NameError instead of
# warning. Define it once here.
logger = logging.getLogger(__name__)


def effective_rank(S: np.ndarray) -> float:
    """Roy & Vetterli 2007 effective rank, computed on raw singular
    values using the exponential entropy formula:

        erank = exp(− Σ pᵢ log pᵢ),   where pᵢ = σᵢ / Σ σⱼ

    All BLME geometry tasks should funnel through this helper so the
    "effective rank" feature is comparable across tasks. Metrics that
    intentionally use covariance/eigenvalue probabilities should be
    named as such rather than reported as Roy-Vetterli effective rank.
    """
    # NaN, not 0.0, for a spectrum with no positive singular value.
    #
    # Roy & Vetterli Property 1 is erank >= 1 for ANY nonzero matrix, so 0.0 is
    # outside the function's range and cannot be a legal answer. Returning it
    # silently drove `collapse_ratio` to 0.0 while `_meta_n_failed_layers`
    # stayed 0, so a degenerate layer was invisible in the output (independent
    # Opus review, 2026-09-04). A rank-one matrix still gives exactly 1.0 by
    # the formula itself.
    S = np.asarray(S, dtype=np.float64)
    if S.size == 0:
        return float("nan")
    S = S[np.isfinite(S) & (S > 0)]
    if S.size == 0:
        return float("nan")
    total = float(S.sum())
    if total <= 0 or not np.isfinite(total):
        return float("nan")
    p = S / total
    p = p[p > 0]
    if p.size == 0:
        return float("nan")
    entropy_sv = float(-np.sum(p * np.log(p)))
    return float(np.exp(entropy_sv))

_OFFLOAD_FILES = []

def _cleanup_offload_files():
    for f in _OFFLOAD_FILES:
        try:
            if os.path.exists(f):
                os.remove(f)
        except Exception:
            pass

atexit.register(_cleanup_offload_files)

def _get_temp_dat_path():
    fd, path = tempfile.mkstemp(prefix="blme_offload_", suffix=".dat")
    os.close(fd)
    _OFFLOAD_FILES.append(path)
    return path


def block_output_count(model=None, n_hidden_entries=None) -> int:
    """Number of hidden_states entries that are genuine BLOCK OUTPUTS.

    HuggingFace returns `n_layers + 1` entries: [0] is the embedding output and
    [i+1] is block i's output -- EXCEPT that for most architectures the LAST
    entry has already passed through the final norm (`ln_f` / `final_layer_norm`
    / `norm`). So `hidden_states[-1]` is not a block output, and any task that
    treats the last adjacent PAIR as a layer transformation is measuring a
    LayerNorm.

    This is not a corner case. Confirmed across five tasks:
      * geometry_lipschitz   -- the reported max IS the final norm
                                (pythia-70m per-pair: [.78, 1.00, .81, .96, 28.6])
      * dynamics_coe         -- the last step is 24-40% of total path length and
                                sets Z_Mag; dropping it flips gpt2-large's
                                coe_r from +1.04 to -0.073
      * dynamics_perturbation_growth -- mean |42%| change in growth_log_ratio,
                                and spearman(as-reported, corrected) = +0.571,
                                so 43% of the cross-model ordering is a LayerNorm
      * geometry_rsa         -- rsa_adjacent_min is the final norm's transition
                                in 16/16 gpt2 windows (0.5042 vs 0.7255 true)
      * geometry_procrustes_linearity -- _min == _last == the final norm

    Returns how many entries to keep from the RAW ``out.hidden_states`` tuple
    (i.e. embedding + blocks 0..L-2). Tasks reading the model's actual output
    representation should keep using the last entry explicitly.

    NOTE the two different indexings, which is easy to get wrong:
      * raw ``out.hidden_states``  -> L+1 entries, [emb, b0, ..., b_{L-1}];
        b_{L-1} is post-norm. Keep ``block_output_count`` = L of them.
      * ``collect_hidden_states(layer_idx="all")`` and the cache already DROP
        the embedding, so their keys are 0..L-1 mapping to b0..b_{L-1}, and the
        post-norm entry is the LAST KEY. Use ``collected_layer_count`` there.
    Applying the raw count to collected keys is a silent no-op -- it was, on
    four tasks, until a verification run caught it.
    """
    n = n_hidden_entries
    if n is None and model is not None:
        cfg = getattr(model, "config", None)
        n = (getattr(cfg, "num_hidden_layers", None)
             or getattr(cfg, "n_layer", None))
        n = (n + 1) if n else None
    if not n or n < 3:
        return max(0, (n or 1) - 1)
    # entries: [emb, block_0, ..., block_{L-1}=post-norm]. Genuine block
    # outputs available for transition analysis: emb + blocks 0..L-2.
    return n - 1


def collected_layer_count(model=None, n_collected=None) -> int:
    """How many keys to keep from `collect_hidden_states(layer_idx="all")`.

    Those keys already exclude the embedding, so key L-1 IS the post-final-norm
    entry and must be dropped for transition analysis: keep L-1 keys.
    """
    n = n_collected
    if n is None and model is not None:
        cfg = getattr(model, "config", None)
        n = (getattr(cfg, "num_hidden_layers", None)
             or getattr(cfg, "n_layer", None))
    # Dropping the post-norm entry needs at least 3 collected layers to leave a
    # usable pair. Below that, keep everything: a 2-layer test fixture has no
    # spare layer, and trimming it turns the task into
    # "Need at least 2 layers", which is worse than the bias we are removing.
    if not n or n < 3:
        return int(n or 2)
    return n - 1


def pow2_rescale(X, gmax=None):
    """Divide ``X`` by a power of two near its largest magnitude, EXACTLY.

    Shifting exponents introduces no rounding, which is why the magnitude
    guards in this repo rescale by a power of two rather than by the actual
    maximum -- it keeps bit-exact parity with released reference code on
    ordinary data while making extreme magnitudes representable.

    The obvious spelling, ``X / 2.0 ** round(log2(gmax))``, is WRONG AT BOTH
    ENDS and shipped in three places:

      * ``round`` reaches 2**128 for any ``gmax >= 2^127.5 = 2.4062e38`` --
        the top 29% of the float32 range -- and 2**128 overflows float32 to
        ``inf``, so the divisor became inf and the data became 0. Measured:
        MNN/T = 0.5773502588 at 2.406159650500376e38 and **0.0 at the very
        next float32**, a silently finite 100% error.
      * ``floor`` underflows at the other end: at ``gmax = 1e-45`` it asks for
        2**-150, which is 0 in float32, so the divisor became 0.

    ``ldexp`` scales by an exponent directly and never materialises the power,
    and the exponent is clamped to the dtype's normal range so the result is
    always representable (independent Opus review, 2026-09-05).
    """
    import torch as _t
    if gmax is None:
        gmax = float(X.detach().abs().max()) if X.numel() else 0.0
    gmax = float(gmax)
    if not (gmax > 0.0) or not math.isfinite(gmax):
        return X
    _fi = _t.finfo(X.dtype)
    _emin = int(math.floor(math.log2(float(_fi.tiny))))
    _emax = int(math.floor(math.log2(float(_fi.max))))
    e = int(max(_emin, min(_emax, round(math.log2(gmax)))))
    return _t.ldexp(X, _t.tensor(-e, dtype=_t.int32, device=X.device))


def safe_norm(x) -> float:
    """L2 norm of a tensor that is exact at ANY representable scale.

    ``x.norm()`` squares before summing, so it underflows to 0.0 (and, at the
    other end, overflows to inf) long before the data does. Measured in this
    campaign: a gradient whose 8 coordinates were all the representable
    1.250000039e-23 reported norm 0.0 against a true 3.535534017e-23, and two
    nonzero gradients of -5.000000097707407e-25 reported norm 0.0 and turned
    SAM sharpness into NaN (adversarial review round 3, 2026-09-04).

    Factoring out the largest magnitude first only shifts exponents, so it is
    exact, and it brings every coordinate into [-1, 1] before the squaring.
    """
    import torch as _t
    if not isinstance(x, _t.Tensor):
        x = _t.as_tensor(x)
    m = x.detach().abs().max()
    if not bool(_t.isfinite(m)) or float(m) == 0.0:
        return 0.0 if float(m) == 0.0 else float("nan")
    return float(m) * float((x.detach() / m).norm())


def safe_unit_rows(X, eps_zero: bool = True):
    """Row-wise L2 normalisation that is exact at ANY representable scale.

    A plain ``X / norm(X)`` is not: the norm SQUARES each coordinate, so it
    underflows (and, at the other end, overflows) long before the data does.
    Measured across this campaign's tasks: rows of magnitude 3e-23 whose
    coordinates are all nonzero produced a norm of exactly 0.0 in float32,
    which turned a true mean cosine of 0.288888889 into 0.145793468, moved a
    hubness skew 0.393048896 -> 7.301942076, and made an entropy read
    0.482747648 instead of its scale-free value (adversarial review rounds 1-2,
    2026-09-04).

    Dividing each row by its own largest-magnitude coordinate first is exact in
    binary floating point -- it only shifts exponents -- and brings every row
    into [-1, 1] before the squaring happens. The unit vector is identical.

    Rows that are EXACTLY zero have no direction and come back as zero rows.
    """
    X = np.asarray(X)
    rmax = np.max(np.abs(X), axis=1, keepdims=True)
    tiny = np.finfo(X.dtype).tiny if np.issubdtype(X.dtype, np.floating) else 0.0
    Xs = X / np.maximum(rmax, tiny)
    n = np.linalg.norm(Xs, axis=1, keepdims=True)
    U = Xs / np.maximum(n, tiny)
    if eps_zero:
        U = np.where(rmax > 0, U, 0.0)
    return U


def safe_unit_rows_torch(X):
    """``safe_unit_rows`` for torch tensors; same rationale, same guarantees.

    ONE full-size temporary, not four. The first version wrote

        Xs = X / rmax.clamp_min(tiny)                       # copy 1
        U  = Xs / Xs.norm(...).clamp_min(tiny)              # copy 2
        return torch.where(rmax > 0, U, torch.zeros_like(U))  # copies 3 and 4

    which peaks at roughly 4x the input. Measured on a real 2.18 GB unembedding
    head that is **8.52 GB** (independent Opus review, 2026-09-05) -- and the
    same review measured the float64 chunked Gram it feeds at 0.00 GB extra, so
    the normaliser was the entire memory cost of the "hardened" path, and the
    "~300 MB" figure recorded for it was wrong.

    The arithmetic below is identical -- divide by the row max, then by the norm
    of the result, then zero any all-zero row -- performed in place. Value
    equality with the old form is pinned in `tests/test_scale_invariance.py`.
    """
    rmax = X.abs().amax(dim=-1, keepdim=True)
    tiny = torch.finfo(X.dtype).tiny
    U = X / rmax.clamp_min(tiny)
    U.div_(U.norm(dim=-1, keepdim=True).clamp_min(tiny))
    zero_rows = (rmax <= 0).squeeze(-1)
    if bool(zero_rows.any()):
        U[zero_rows] = 0
    return U


def as_numpy(t):
    """Tensor -> numpy without changing any value.

    numpy has no bfloat16, so ``.numpy()`` raises on a bf16 tensor; bf16 widens
    EXACTLY to float32. Every other dtype is kept as is (in particular float64
    is never narrowed).
    """
    t = t.detach().cpu()
    if t.dtype == torch.bfloat16:
        t = t.float()
    return t.numpy()


def _offload_bytes(t):
    """Raw bytes of ``t`` plus a dtype tag the memmap reader understands.

    bfloat16 has no numpy dtype, so it is stored as its 16-bit pattern and
    viewed back as bfloat16 on read: the round trip is bit-exact and keeps the
    dtype the in-memory route returns.
    """
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().tobytes(), "bfloat16"
    a = t.numpy()
    return a.tobytes(), a.dtype.str


def _offload_view(path, tag, shape):
    if tag == "bfloat16":
        return torch.from_numpy(np.memmap(path, dtype=np.int16, mode="r",
                                          shape=shape)).view(torch.bfloat16)
    return torch.from_numpy(np.memmap(path, dtype=np.dtype(tag), mode="r", shape=shape))


def safe_svdvals(M):
    """Singular values that do not overflow at the top of the float32 range.

    `np.linalg.svd` / `torch.linalg.svdvals` compute internally and then CAST
    the singular values back to the input dtype. For float32 input whose
    largest entry is near 3.4e38 that cast overflows to `inf` -- numpy even says
    so ("overflow encountered in cast") -- and every downstream consumer then
    filters the non-finite values and returns NaN. Measured on a (120, 16)
    Gaussian: `svdvals[0]` is 3.8730795 at scale 1, 3.8730798e+25 at 1e25, and
    **inf** at 2.406e38, which is what made `effective_rank` NaN there
    (property sweep, 2026-09-06).

    Factoring out the largest magnitude first only shifts exponents, so the
    returned values are exact; scale-free consumers see the factor cancel and
    scale-covariant ones get it back unchanged.
    """
    import numpy as _np
    _is_torch = isinstance(M, torch.Tensor)
    A = M if _is_torch else torch.as_tensor(_np.asarray(M))
    # CAST BEFORE SCALING. `(A / g).double()` performs the division in the
    # INPUT dtype, so a float32 matrix is rounded once more before it is
    # promoted -- the promotion then preserves an error it could have avoided.
    # Doing `A.double()` first makes the division exact in double and the
    # rescale a pure multiply-back. On a 20x20 seed-140 matrix translated by
    # +1e6 the old order gave official alpha 1.249006646 against a 100-digit
    # SVD reference's 1.163063736 (+7.39%) (gpt-5.6-sol round 9, 2026-09-07).
    Ad = A.double()
    gd = Ad.abs().amax()
    gf = float(gd)
    if not _np.isfinite(gf) or gf == 0.0:
        S = torch.linalg.svdvals(Ad)
        return S if _is_torch else S.cpu().numpy()
    S = torch.linalg.svdvals(Ad / gd) * gd
    return S if _is_torch else S.cpu().numpy()


def collect_hidden_states(model, tokenizer, dataset, num_samples=100, layer_idx=-1,
                          use_disk_offload=False, replace_final_norm=False,
                          status=None, max_tokens_per_doc=10,
                          include_embedding=False):
    """Collect hidden states from a specific layer or all layers.

    Uses `output_hidden_states=True` for universal compatibility.

    Args:
        model: HuggingFace causal LM
        tokenizer: corresponding tokenizer
        dataset: list of dicts with 'text' key or strings
        num_samples: max samples to process
        layer_idx: int for specific layer, "all" for all layers.
                   -1 means the final layer.
        use_disk_offload: bool, if True streams tensors to memory-mapped disk files.
        status: optional dict, filled in with ``last_is_block`` -- True only if
            the final-block hook actually fired AND its shape matched the slot
            it replaced. Callers MUST NOT assume the substitution happened:
            a model whose blocks are not discoverable by ``get_layers`` silently
            keeps the post-norm entry, and a caller that assumed otherwise then
            reported a LayerNorm output as a block output (adversarial review,
            2026-09-04).
        replace_final_norm: bool. HuggingFace's last ``hidden_states`` entry is
            ``ln_f(block L-1)``, NOT block L-1's raw output -- so the deepest
            block is not directly observable in the tuple. When True, a forward
            hook on ``layers[-1]`` captures that block's raw (pre-norm) output
            and substitutes it for the final entry, giving a profile over
            blocks 0..L-1 with no LayerNorm artifact and no deleted layer.

            This is the correct fix for BOTH failure modes: including the
            post-norm entry contaminates transition metrics (on gpt2 ~90% of
            the last transition's block-influence was the ln_f artifact),
            while TRUNCATING it away silently deletes the deepest block
            (gpt2 collapse_ratio 0.5200 trimmed vs 0.4626 hook-recovered,
            +12.4%; max_erank -11.0%).

    Returns:
        If layer_idx != "all": single tensor (TotalTokens, D)
        If layer_idx == "all": dict {layer_idx: (TotalTokens, D)}
    """
    use_offload = use_disk_offload or (os.environ.get("BLME_DISK_OFFLOAD", "0") == "1")
    all_hidden = {}  # layer_index -> list of (N_tokens, D) tensors

    # Optional recovery of the deepest block's RAW output (see docstring).
    _last_block_out = {}
    _hook = None
    # COUNTS, not a run-level boolean: `last_is_block` must mean "the swap
    # fired on EVERY forward", not "on at least one".
    _swapped = [0]
    _n_forwards = [0]
    if replace_final_norm:
        try:
            from ..common import get_layers
            _blocks = get_layers(model)
            if _blocks is not None and len(_blocks) > 0:
                def _capture_last(module, inputs, output):
                    out_t = output[0] if isinstance(output, tuple) else output
                    _last_block_out["h"] = out_t.detach()
                _hook = _blocks[-1].register_forward_hook(_capture_last)
        except Exception as e:  # pragma: no cover - defensive
            logger.info(f"replace_final_norm: could not hook last block ({e})")

    try:
        with torch.no_grad():
            for i, sample in enumerate(tqdm(dataset, desc="Collecting states")):
                if i >= num_samples:
                    break

                # Truncate at 512 to match the shared cache's tokenization
                # (audit 2026-07-16: no truncation crashed on >n_positions
                # docs AND made the cache/non-cache paths inconsistent).
                if isinstance(sample, str):
                    text = sample
                else:
                    text = sample.get("text", "")
                inputs = tokenizer(text, return_tensors="pt", truncation=True,
                                   max_length=512).to(model.device)

                # Skip empty / single-token docs: an empty forward pass crashes
                # ("cannot reshape tensor of 0 elements").
                if inputs["input_ids"].shape[1] < 1:
                    continue

                # CLEAR the hook buffer before every forward. A model whose
                # last block is conditional may not fire the hook on some
                # inputs, and a stale entry of the right shape was then
                # substituted into the NEXT sample's states -- max tensor error
                # 5.0 in a two-forward test (adversarial review round 2,
                # 2026-09-04).
                _last_block_out.pop("h", None)
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states  # tuple of (1, T, D)

                if replace_final_norm and "h" in _last_block_out:
                    _h = _last_block_out["h"]
                    # Only substitute when the hook's shape matches the slot it
                    # replaces; a mismatch means the hook fired on something
                    # other than the final block and must not be trusted.
                    # post-final-norm: THIS IS THE SUBSTITUTION -- the mechanism
                    # every other site opts into. Reading the last entry here
                    # is the point.
                    if tuple(_h.shape) == tuple(hidden_states[-1].shape):
                        hidden_states = tuple(hidden_states[:-1]) + (_h,)
                        _swapped[0] += 1
                _n_forwards[0] += 1

                # hidden_states[0] = embedding output
                # hidden_states[i+1] = output of layer i
                n_layers = len(hidden_states) - 1  # subtract embedding layer

                if layer_idx == "all":
                    # `include_embedding=True` puts the embedding OUTPUT at key
                    # 0 and shifts block i to key i+1, matching the cache's
                    # convention. Tasks that score residual UPDATES need it:
                    # starting from block 0's output scores only L-1 of the L
                    # updates, and the first is systematically the largest
                    # (adversarial review round 2, 2026-09-04).
                    _first = 0 if include_embedding else 1
                    for _slot, li in enumerate(range(_first, n_layers + 1)):
                        # post-final-norm: reads the tuple AFTER the swap
                        # above, so entry -1 is already the raw block output
                        # when the caller asked for it.
                        h = hidden_states[li]
                        h_flat = h.reshape(-1, h.shape[-1]).detach().cpu()
                        # PRESERVE the incoming float dtype. `.float()` is a
                        # no-op for the float32 every real checkpoint emits
                        # and a NARROWING for float64, so it silently
                        # discarded precision the estimators were repaired
                        # to keep -- geometry_intrinsic_dim's collection
                        # routes returned 3.552080162654 against the
                        # released 4.606333343805 (-22.89%) for this reason
                        # (flagged round 12, fixed round 14, 2026-09-09).
                        # Promoting outright would double the memory of
                        # every real run for no gain, so only NON-float
                        # tensors are converted.
                        if not h_flat.is_floating_point():
                            h_flat = h_flat.float()
                        _k = _slot
                        if use_offload:
                            if _k not in all_hidden:
                                path = _get_temp_dat_path()
                                f = open(path, "wb")
                                all_hidden[_k] = {"path": path, "handle": f, "count": 0, "D": h_flat.shape[-1],
                                    # RECORD THE DTYPE. The bytes below are
                                    # written raw and the memmap must read them
                                    # back in the SAME format; hard-coding
                                    # float32 there while the writer preserves
                                    # the incoming dtype silently reinterprets
                                    # 8-byte floats as 4-byte ones, which turned
                                    # an exact [4, 2, 1] profile into
                                    # [1.994847434735, 2, 1] and GPT-2 CKA into
                                    # NaN (regression introduced 2026-09-09 by
                                    # the dtype-preserving collection fix, caught
                                    # by codex round 15 the same day).
                                    "dtype_tag": None}
                            f = all_hidden[_k]["handle"]
                            _bytes, _tag = _offload_bytes(h_flat)
                            all_hidden[_k]["dtype_tag"] = _tag
                            f.write(_bytes)
                            all_hidden[_k]["count"] += h_flat.shape[0]
                        else:
                            if _k not in all_hidden:
                                all_hidden[_k] = []
                            all_hidden[_k].append(h_flat)
                else:
                    # Resolve negative index
                    actual_idx = layer_idx if layer_idx >= 0 else n_layers + layer_idx
                    # Clamp
                    actual_idx = max(0, min(actual_idx, n_layers - 1))

                    # post-final-norm: same -- post-swap read.
                    h = hidden_states[actual_idx + 1]  # +1 to skip embedding

                    # Subsample tokens for single-layer mode
                    # `max_tokens_per_doc=None` keeps every token, which is
                    # what the CACHE route does. Hard-coding 10 here made the
                    # two routes different protocols: on the same 50 GPT-2
                    # texts they used 500 vs 1456 points and LID read
                    # 1.787210365259 vs 4.608033135762, +157.8% (adversarial
                    # review round 2, 2026-09-04).
                    T = h.shape[1]
                    _cap = max_tokens_per_doc
                    if _cap is not None and T > _cap:
                        rng = torch.Generator(device=h.device)
                        rng.manual_seed(42 + i * 1009 + actual_idx)
                        indices = torch.randperm(T, generator=rng,
                                                 device=h.device)[:_cap]
                        h = h[:, indices, :]

                    h_flat = h.reshape(-1, h.shape[-1]).detach().cpu()
                    # PRESERVE the incoming float dtype. `.float()` is a
                    # no-op for the float32 every real checkpoint emits
                    # and a NARROWING for float64, so it silently
                    # discarded precision the estimators were repaired
                    # to keep -- geometry_intrinsic_dim's collection
                    # routes returned 3.552080162654 against the
                    # released 4.606333343805 (-22.89%) for this reason
                    # (flagged round 12, fixed round 14, 2026-09-09).
                    # Promoting outright would double the memory of
                    # every real run for no gain, so only NON-float
                    # tensors are converted.
                    if not h_flat.is_floating_point():
                        h_flat = h_flat.float()
                    if use_offload:
                        if layer_idx not in all_hidden:
                            path = _get_temp_dat_path()
                            f = open(path, "wb")
                            all_hidden[layer_idx] = {"path": path, "handle": f, "count": 0, "D": h_flat.shape[-1],
                                    # RECORD THE DTYPE. The bytes below are
                                    # written raw and the memmap must read them
                                    # back in the SAME format; hard-coding
                                    # float32 there while the writer preserves
                                    # the incoming dtype silently reinterprets
                                    # 8-byte floats as 4-byte ones, which turned
                                    # an exact [4, 2, 1] profile into
                                    # [1.994847434735, 2, 1] and GPT-2 CKA into
                                    # NaN (regression introduced 2026-09-09 by
                                    # the dtype-preserving collection fix, caught
                                    # by codex round 15 the same day).
                                    "dtype_tag": None}
                        f = all_hidden[layer_idx]["handle"]
                        _bytes, _tag = _offload_bytes(h_flat)
                        all_hidden[layer_idx]["dtype_tag"] = _tag
                        f.write(_bytes)
                        all_hidden[layer_idx]["count"] += h_flat.shape[0]
                    else:
                        if layer_idx not in all_hidden:
                            all_hidden[layer_idx] = []
                        all_hidden[layer_idx].append(h_flat)
    finally:
        if _hook is not None:
            _hook.remove()
        if status is not None:
            status["last_is_block"] = bool(
                _n_forwards[0] > 0 and _swapped[0] == _n_forwards[0])
            status["n_forwards"] = int(_n_forwards[0])
            status["n_swapped"] = int(_swapped[0])
        if use_offload:
            for meta in all_hidden.values():
                if isinstance(meta, dict) and "handle" in meta and not meta["handle"].closed:
                    meta["handle"].close()

    # Concatenate or MemMap
    results = {}
    if use_offload:
        for idx, meta in all_hidden.items():
            meta["handle"].close()
            if meta["count"] > 0:
                results[idx] = _offload_view(meta["path"], meta["dtype_tag"],
                                             (meta["count"], meta["D"]))
            else:
                results[idx] = torch.empty(0)
    else:
        for idx, tensors in all_hidden.items():
            results[idx] = torch.cat(tensors, dim=0) if tensors else torch.empty(0)

    if layer_idx != "all":
        return list(results.values())[0] if results else torch.empty(0)
    else:
        return results


def collect_prediction_stats(model, tokenizer, dataset, num_samples=100, use_disk_offload=False):
    """Collect logits, labels, and hidden states for consistency analysis.

    Uses `output_hidden_states=True` for universal compatibility.

    Returns:
        (stats, embeddings) where:
        - stats["logits"]: list of (N_tokens, V) tensors
        - stats["labels"]: list of (N_tokens,) tensors
        - stats["hidden"]: list of (N_tokens, D) tensors
        - stats["token_counts"]: numpy array of shape (vocab_size,)
        - embeddings: (V, D) tensor or None
    """
    use_offload = use_disk_offload or (os.environ.get("BLME_DISK_OFFLOAD", "0") == "1")
    stats = {
        # Llama-3 and friends: ``tokenizer.vocab_size`` is the number of
        # *base* BPE merges (128 000) but the added-special-tokens push
        # the id range up to 128 256. ``len(tokenizer)`` is the safe
        # upper bound that ``input_ids`` can contain; falling back to
        # the model's config covers multimodal wrappers.
        "token_counts": np.zeros(
            max(
                len(tokenizer) if hasattr(tokenizer, "__len__") else 0,
                int(getattr(tokenizer, "vocab_size", 0) or 0),
            ) or 50257,
        ),
    }

    if use_offload:
        p_logits, p_labels, p_hidden = _get_temp_dat_path(), _get_temp_dat_path(), _get_temp_dat_path()
        f_logits, f_labels, f_hidden = open(p_logits, "wb"), open(p_labels, "wb"), open(p_hidden, "wb")
        c_logits = c_labels = c_hidden = 0
        # PER-DOCUMENT LENGTHS. The offloaded arrays are one flat stream, but
        # the consumer treats each list element as ONE DOCUMENT, so the
        # segmentation has to survive the round-trip (see the return below).
        _doc_lens = []
        D_logits = D_hidden = 0
    else:
        stats["logits"] = []
        stats["labels"] = []
        stats["hidden"] = []

    embeddings = get_embeddings(model)

    try:
        with torch.no_grad():
            for i, sample in enumerate(tqdm(dataset, desc="Collecting prediction stats")):
                if i >= num_samples:
                    break

                # Truncate at 512 to match the shared cache's tokenization
                # (audit 2026-07-16: no truncation crashed on >n_positions
                # docs AND made the cache/non-cache paths inconsistent).
                if isinstance(sample, str):
                    text = sample
                else:
                    text = sample.get("text", "")
                inputs = tokenizer(text, return_tensors="pt", truncation=True,
                                   max_length=512).to(model.device)

                # Skip empty / single-token docs: they yield no (T-1) labels to
                # score and an empty forward pass crashes ("cannot reshape tensor
                # of 0 elements"). Skipping keeps this scoring window aligned with
                # the BPC recount window in geometry_perplexity.
                if inputs["input_ids"].shape[1] < 2:
                    continue

                outputs = model(**inputs, output_hidden_states=True)
                logits = outputs.logits.float().detach().cpu()

                input_ids = inputs["input_ids"].cpu()
                labels = input_ids[:, 1:]
                logits = logits[:, :-1, :]

                # post-final-norm: DELIBERATELY post-norm. This helper pairs
                # `h` with `outputs.logits`, and `lm_head(hidden_states[-1])`
                # reproduces those logits exactly -- the raw block output is
                # NOT the vector the head consumed, so substituting it here
                # would break the pairing the analysis depends on.
                h = outputs.hidden_states[-1].float().detach().cpu()
                h_pred = h[:, :-1, :]

                l_flat = logits.reshape(-1, logits.shape[-1]).numpy()
                y_flat = labels.reshape(-1).numpy()
                h_flat = h_pred.reshape(-1, h_pred.shape[-1]).numpy()

                if use_offload:
                    f_logits.write(l_flat.tobytes())
                    f_labels.write(y_flat.tobytes())
                    f_hidden.write(h_flat.tobytes())
                    c_logits += l_flat.shape[0]
                    c_labels += y_flat.shape[0]
                    c_hidden += h_flat.shape[0]
                    _doc_lens.append(int(y_flat.shape[0]))
                    D_logits, D_hidden = l_flat.shape[-1], h_flat.shape[-1]
                else:
                    stats["logits"].append(torch.from_numpy(l_flat))
                    stats["labels"].append(torch.from_numpy(y_flat))
                    stats["hidden"].append(torch.from_numpy(h_flat))

                ids_flat = input_ids.view(-1).numpy()
                np.add.at(stats["token_counts"], ids_flat, 1)
    finally:
        if use_offload:
            for fh in (f_logits, f_labels, f_hidden):
                if not fh.closed:
                    fh.close()

    if use_offload:
        f_logits.close()
        f_labels.close()
        f_hidden.close()
        if c_logits > 0:
            m_logits = np.memmap(p_logits, dtype=np.float32, mode='r', shape=(c_logits, D_logits))
            m_labels = np.memmap(p_labels, dtype=np.int64, mode='r', shape=(c_labels,))
            m_hidden = np.memmap(p_hidden, dtype=np.float32, mode='r', shape=(c_hidden, D_hidden))
            # SPLIT BACK INTO DOCUMENTS. Returning one tensor for the whole
            # corpus silently redefined the statistic: `consistency.py` builds
            # WITHIN-document controls and treats every list element as one
            # document, so a single element made every control corpus-wide.
            #
            # With W = [[1,0],[-1,0]] and documents h = [[1,0],[1,0]],
            # labels [0,0] and h = [[-1,0],[-1,0]], labels [1,1], every legal
            # within-document control equals its target and the exact margin is
            # 0.0: the in-memory and ModelOutputCache routes emit 0.0 while the
            # offloaded route emitted 2.0, with the collision rate going
            # 1.0 -> 0.0. A collision-free anchor's exact margin 1.0 became
            # 1.4999999105930328. The concatenated tensors are bit-identical
            # between routes -- ONLY the lengths [2,2] -> [4] differed. On real
            # models the headline moved gpt2 0.010629930335388156 ->
            # 0.011383407172830932 (+7.09%), pythia-70m +9.00% and distilgpt2
            # +10.15% (codex round 12, 2026-09-08).
            _t_logits = torch.from_numpy(m_logits)
            _t_labels = torch.from_numpy(m_labels)
            _t_hidden = torch.from_numpy(m_hidden)
            if _doc_lens and sum(_doc_lens) == c_labels:
                _off = 0
                stats["logits"], stats["labels"], stats["hidden"] = [], [], []
                for _n in _doc_lens:
                    stats["logits"].append(_t_logits[_off:_off + _n])
                    stats["labels"].append(_t_labels[_off:_off + _n])
                    stats["hidden"].append(_t_hidden[_off:_off + _n])
                    _off += _n
            else:
                stats["logits"] = [_t_logits]
                stats["labels"] = [_t_labels]
                stats["hidden"] = [_t_hidden]
        else:
            stats["logits"], stats["labels"], stats["hidden"] = [], [], []

    return stats, embeddings


_DEFAULT_PPL_WINDOW = 1024


def _bytes_to_unicode_map():
    """The GPT-2 byte-level BPE alphabet (Radford et al. 2019), inverted."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\u00a1"), ord("\u00ac") + 1))
          + list(range(ord("\u00ae"), ord("\u00ff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_U2B = _bytes_to_unicode_map()


def _tokenizer_preserves_byte_count(tokenizer, ids, corpus) -> bool:
    """True when re-decoding the ids reproduces the source's BYTE COUNT.

    A normalising tokenizer (NFC composition, accent stripping, case folding
    that changes width) scores a different byte sequence from the source, so
    attributing SOURCE bytes to scored targets is wrong even when no offset
    span is shared. This is the cheapest exact test of that condition.

        SPECIAL TOKENS ARE SKIPPED, or this test fires for the wrong reason: a
    BOS-adding tokenizer renders its BOS as the literal '<|endoftext|>' (13
    characters that are nowhere in the corpus), so a plain decode never matches
    the source length and BPC was withheld as NaN on every BOS tokenizer --
    a false positive that broke `test_perplexity_bpc_correct_with_bos_tokenizer`
    (BPC nan against the textbook 1.05579). Skipping special tokens isolates the
    condition this guard is actually for, which is NORMALISATION of the source
    text.
    """
    try:
        # STRING EQUALITY, not byte-count equality.
        #
        # The first version compared only LENGTHS, which passes any
        # BYTE-PRESERVING normalisation. Pythia-70m maps 'a' + U+0323
        # (combining dot below) onto the composed 'a-with-dot': both encode to
        # 3 bytes, so the length test saw nothing, while the attribution still
        # charged 3 bytes and 2 characters to a single scored token that owns
        # 1 byte and 1 character -- BPB 3.579988796568 against 10.739966389705
        # (-66.6667%) and BPC 5.369983194853 against 10.739966389705 (-50%),
        # with Pythia-160m independently at 1.737205126188 against
        # 5.211615378563 (codex round 14, 2026-09-09).
        return tokenizer.decode(list(ids), skip_special_tokens=True) == corpus
    except Exception:
        return False


def _exact_token_bytes(tokenizer, ids, corpus):
    """Raw UTF-8 bytes owned by each token, or ``None`` if not establishable.

    BYTE OWNERSHIP IS A PROPERTY OF THE TOKEN PIECES, NOT OF CHARACTER OFFSETS.
    A byte-level BPE splits a single character across several tokens, and the
    character-offset map reports the SAME span for every one of them -- so no
    arithmetic on offsets can recover which token owns which byte. Dividing the
    shared span evenly invents an answer: real GPT-2 on the emoji '😀b'
    has true piece ownership [3, 1, 1] and the even split assigns [2, 2].

    Returns None unless the reconstructed pieces concatenate to exactly the
    source UTF-8, so a normalizing tokenizer (lower-casing, accent stripping)
    cannot silently pass a wrong partition off as exact.
    """
    try:
        toks = tokenizer.convert_ids_to_tokens(list(ids))
    except Exception:
        return None
    if not toks or any(t is None for t in toks):
        return None
    pieces = []
    for t in toks:
        if t and all(ch in _U2B for ch in t):
            pieces.append(bytes(_U2B[ch] for ch in t))          # byte-level BPE
        elif len(t) == 6 and t.startswith("<0x") and t.endswith(">"):
            try:
                pieces.append(bytes([int(t[3:5], 16)]))          # byte fallback
            except ValueError:
                return None
        else:
            pieces.append(t.replace("\u2581", " ").encode("utf-8"))
    if b"".join(pieces) != corpus.encode("utf-8"):
        return None
    return [len(p) for p in pieces]


def strided_perplexity(model, tokenizer, dataset, num_samples=100,
                       max_length=None, stride=None, doc_sep="\n\n"):
    """Corpus perplexity by the standard strided sliding-window protocol.

    This is the HuggingFace reference procedure
    (https://huggingface.co/docs/transformers/perplexity), which is what
    every published WikiText-2 perplexity number is computed with:

      1. Concatenate the corpus into ONE stream, documents joined by
         ``doc_sep``.
      2. Slide a ``max_length`` window over it in steps of ``stride``.
      3. In each window score ONLY the ``stride`` newly-revealed tokens;
         the overlapping prefix is masked to -100 so it serves purely as
         context and is never double-counted.
      4. Return exp(sum NLL / number of scored tokens).

    Why this replaced per-document scoring (adversarial review, 2026-09-02):
    the previous path tokenized each document independently with
    ``truncation=True, max_length=512``, which

      * DISCARDED every token past 512 in long documents -- on WikiText-2
        that silently drops most of the corpus, and the discarded fraction
        differs per tokenizer, so models were not scored on the same text;
      * RESET the context at every document boundary, so each document's
        first tokens were predicted from nothing. With short rows this is
        most of the corpus and it inflates the mean NLL.

    Verified against the HuggingFace reference on the FULL WikiText-2-raw test
    split: HF 25.17041 vs this implementation 25.17048 (rel 2.8e-6) at
    max_length=1024 / stride=512; exact agreement at stride=1024; rel 3.9e-7 at
    512/256.

    NOTE ON THE NUMBER: 25.17 is the WikiText-2-raw test value under this
    protocol. An earlier version of this docstring claimed "21.80, the published
    GPT-2 small number" -- that was WRONG on both counts. 21.80 is what BLME's
    own default corpus yields, not WikiText-2; and the GPT-2 paper reports 29.41
    for WikiText-2 under its own (inverted-preprocessing, zero-shot) protocol.
    No published GPT-2-small figure is 21.8. The claim is retracted
    (adversarial review, 2026-09-03).

    Returns ``(ppl, mean_nll_nats, n_tokens_scored, n_bytes, n_chars)``,
    where the byte/char counts describe the SAME scored text span so a
    caller can form bits-per-byte without a second, differently-truncated
    tokenization pass.
    """
    texts = []
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break
        text = sample if isinstance(sample, str) else sample.get("text", "")
        # Empty rows are KEPT. WikiText and friends carry many blank lines,
        # and the canonical protocol joins the split verbatim; dropping them
        # silently changes the corpus. Measured on gpt2/WikiText-2-raw test,
        # filtering blanks moved ppl 25.17 -> 24.36 (3.2%) and the scored-token
        # count 287643 -> 286176, i.e. it produced a number that no longer
        # matched any published figure.
        texts.append(text if text else "")
    if not any(texts):
        return float("nan"), float("nan"), 0, 0, 0, 0, 0, "none"

    corpus = doc_sep.join(texts)

    # Full-stream tokenization: no truncation, no per-document reset.
    # Offset mapping when the tokenizer can produce it: it is the ONLY way to
    # attribute source bytes to the tokens that were actually scored.
    _offsets = None
    try:
        _enc_off = tokenizer(corpus, return_tensors="pt", truncation=False,
                             return_offsets_mapping=True)
        _om = _enc_off.get("offset_mapping")
        if _om is not None:
            _offsets = _om[0].tolist()
    except Exception:
        _offsets = None
    enc = tokenizer(corpus, return_tensors="pt", truncation=False)
    input_ids_all = enc["input_ids"]
    n_total = input_ids_all.shape[1]
    if n_total < 2:
        return float("nan"), float("nan"), 0, 0, 0, 0, 0, "none"

    if max_length is None:
        # PINNED window, identical for every model.
        #
        # Deriving it from each model's own context made the metric
        # incomparable, which is what the strided rewrite existed to fix:
        # perplexity depends strongly on the window (pythia-70m on the same
        # text: 49.09 at 512, 43.15 at 1024, 39.80 at 2048 -- a 23% swing),
        # and a config-derived window assigned gemma-2-2b 8192 while giving
        # Llama-3.2-1B 1024, purely because 131072 tripped a sentinel guard.
        # A fixed 1024 is within every modern model's context and is the
        # window the published GPT-2-era numbers use.
        max_length = _DEFAULT_PPL_WINDOW
    # CLAMP to the model's own context. Pinning the window without this is a
    # CRASH: forcing a window past `n_positions` on gpt2 raises a CUDA
    # device-side assert that poisons the CUDA context for the entire process,
    # taking every later task in the run with it (adversarial review,
    # 2026-09-03). The pin makes models comparable where they CAN be compared;
    # it must never ask a model to attend beyond its own limit.
    _ctx = (getattr(getattr(model, "config", None), "max_position_embeddings", None)
            or getattr(getattr(model, "config", None), "n_positions", None))
    if isinstance(_ctx, int) and 0 < _ctx < max_length:
        logger.info(f"  ppl window {max_length} exceeds model context {_ctx}; "
                    f"clamping. This model is NOT comparable at the pinned "
                    f"window -- see _meta_ppl_window.")
        max_length = _ctx
    max_length = int(min(max_length, n_total))
    if stride is None:
        stride = max(1, max_length // 2)
    # A stride LARGER than the window silently skips tokens: target[:, :-trg_len]
    # masks nothing when trg_len exceeds the window, so whole spans are never
    # scored (measured: max_length=512 with stride=2048 scored 25,912 of 102,762
    # tokens -- 75% dropped -- while the byte denominator still covered the whole
    # stream, understating bits_per_byte ~4x).
    stride = int(max(1, min(int(stride), max_length)))

    device = getattr(model, "device", None) or next(model.parameters()).device

    nll_sum = 0.0
    n_scored = 0
    prev_end = 0
    _scored_pos = set()
    for begin in range(0, n_total, stride):
        end = min(begin + max_length, n_total)
        trg_len = end - prev_end          # tokens newly revealed by this window
        if trg_len <= 0:
            continue
        ids = input_ids_all[:, begin:end].to(device)
        target = ids.clone()
        # Mask the context prefix: only the last trg_len positions are scored.
        target[:, :-trg_len] = -100

        with torch.no_grad():
            out = model(ids, labels=target)

        # HF shifts internally, so the number of positions that actually
        # contributed is (#unmasked labels - 1) when the whole window is
        # unmasked, and trg_len otherwise. Recover it from the mask to keep
        # the denominator exact rather than assuming.
        shift_labels = target[:, 1:]
        n_valid = int((shift_labels != -100).sum().item())
        # Absolute stream positions of the tokens PREDICTED by this window.
        # `shift_labels[0, j]` is stream position begin + j + 1.
        _sm = (shift_labels[0] != -100).nonzero(as_tuple=True)[0].tolist()
        for _j in _sm:
            _scored_pos.add(begin + _j + 1)
        if n_valid <= 0:
            prev_end = end
            if end == n_total:
                break
            continue
        # out.loss is the MEAN over the unmasked shifted labels.
        nll_sum += float(out.loss.detach().float().item()) * n_valid
        n_scored += n_valid

        prev_end = end
        if end == n_total:
            break

    if n_scored == 0:
        return float("nan"), float("nan"), 0, 0, 0, 0, 0, "none"

    mean_nll = nll_sum / n_scored
    # Byte/char counts come from the corpus STRING, not from decoding the
    # ids back. Two reasons: a round-trip through decode() is lossy for some
    # tokenizers (byte-fallback pieces, normalization) so it would not return
    # the text that was actually scored; and it requires a `.decode` method,
    # which minimal tokenizer wrappers need not provide -- that raised inside
    # the try/except in perplexity.py and silently produced bits_per_char=NaN.
    #
    # This is also the standard bits-per-character reduction: bits per token
    # times tokens per character, with the character count taken over the
    # whole evaluation text. The first token of the stream is never predicted,
    # so its characters sit in the denominator without contributing to the
    # numerator; that is inherent to the definition and is one token in the
    # entire corpus.
    # Byte/char span: the SOURCE text minus the first token, which is never
    # predicted. Counting decoded bytes instead would reintroduce a
    # tokenizer-family bias into a statistic whose whole purpose is tokenizer
    # independence -- decode round-trips lose 1.70% of the bytes on
    # Llama-3.2-1B and OLMo-1B while being exact on gpt2/pythia/Qwen.
    #
    # ATTRIBUTE bytes to the tokens ACTUALLY SCORED, via the offset mapping.
    #
    # Subtracting `len(decode(first_id))` was wrong twice over. It uses a
    # DECODED length where a source-offset length is meant -- a zero-byte BOS
    # that prints as '<BOS>' subtracted 5 bytes that were never there, turning
    # a true BPB of 1.000000003 into 2.000000005 -- and it assumed exactly one
    # unscored token in the whole stream. When the stride reaches the window,
    # EVERY later chunk's position-0 target is unscoreable too: on a 30-byte
    # anchor with max_length = stride = 7, 25 targets were scored while 29
    # bytes were counted (adversarial review, 2026-09-04).
    #
    # With offsets, the denominator is exactly the source span of the scored
    # targets. Without them (slow tokenizers), fall back to scaling the corpus
    # by the scored fraction and SAY SO via `_meta_ppl_byte_attribution`.
    corpus_bytes = corpus.encode("utf-8")
    _byte_exact = False
    _byte_src = "offsets"
    if _offsets is not None and len(_offsets) == n_total:
        # PARTITION the corpus, do not sum offset widths.
        #
        # Offset spans are neither a cover nor disjoint. GPT-NeoX/Pythia leaves
        # whitespace outside every span, so summing widths counted 48,130 of
        # 59,281 actually-scored bytes and inflated BPB by +23.17%; GPT-2 maps
        # BOTH pieces of an emoji to (1, 2), so summing double-counts
        # (adversarial review round 2, 2026-09-04).
        #
        # Instead attribute to token i the half-open source span
        # (end_{i-1}, end_i], with end_{-1} = 0 and ends made non-decreasing.
        # That partitions the WHOLE corpus exactly once: no gap, no overlap. A
        # zero-width BOS gets an empty span, the second piece of a shared-span
        # pair gets an empty span, and leading whitespace is charged to the
        # token that consumes it -- which is the token being predicted.
        _ends = []
        _prev_end = 0
        for _a, _bnd in _offsets:
            _prev_end = max(_prev_end, int(_bnd))
            _ends.append(_prev_end)
        # SHARED SPANS ARE SPLIT, not given wholly to the first piece.
        #
        # A byte-fallback tokenizer emits several pieces for ONE character:
        # GPT-2 maps the two pieces of an emoji to offsets [(0,1), (0,1)], so
        # the cumulative-ends partition gave piece 0 the entire character and
        # piece 1 exactly zero (adversarial review round 4, 2026-09-04). The
        # total stays exact either way, but a token that genuinely predicts
        # part of a character should not be credited with none of it, and at a
        # chunk boundary the difference lands in the denominator. Each group of
        # tokens sharing a span divides that span's bytes and characters
        # evenly.
        _grp_start = 0
        _wb = [0.0] * len(_ends)
        _wc = [0.0] * len(_ends)
        _shared = False       # did ANY span have to be divided?
        for _i in range(len(_ends)):
            if _i + 1 < len(_ends) and _ends[_i + 1] == _ends[_i]:
                continue
            _lo = _ends[_grp_start - 1] if _grp_start > 0 else 0
            _hi = _ends[_i]
            _n_share = _i - _grp_start + 1
            if _n_share > 1 and _hi > _lo:
                _shared = True
            if _hi > _lo:
                _bts = len(corpus[_lo:_hi].encode("utf-8")) / _n_share
                _chs = (_hi - _lo) / _n_share
                for _j in range(_grp_start, _i + 1):
                    _wb[_j] = _bts
                    _wc[_j] = _chs
            _grp_start = _i + 1

        # EXACT PIECE OWNERSHIP FIRST; the even split is not a partition.
        #
        # Dividing a shared span evenly and then rounding INVENTS byte
        # ownership and used to label it exact. Real GPT-2 on
        # '😀 This is an ordinary sentence about the weather.' counted 50
        # scored bytes against the true 49 and emitted bits_per_byte
        # 1.443730926792 instead of 1.473194823257 (-2.000%); on '😀b',
        # whose true ownership is [3, 1, 1], it counted 3 scored bytes instead
        # of 2 and emitted 8.143592235461 instead of 12.215388353192
        # (-33.333%), and Pythia-70m 4.623930194629 instead of 6.165240259505.
        # On GPT-2/WikiText-2 exactly one position differs (token ID 564, raw
        # pieces [32,226,128]/[145] assigned [2,2] instead of [3,1]), for
        # 1295024 counted against 1295023 true bytes (codex round 12,
        # 2026-09-08).
        _own = _exact_token_bytes(tokenizer, input_ids_all[0].tolist(), corpus)
        if _own is not None and len(_own) == len(_ends):
            # A character belongs to the token that owns its LAST byte.
            _cum, _acc = [], 0
            for _n in _own:
                _acc += _n
                _cum.append(_acc)
            _wb = [float(_n) for _n in _own]
            _wc = [0.0] * len(_own)
            _ci, _bpos = 0, 0
            for _ch in corpus:
                _bpos += len(_ch.encode("utf-8"))
                while _ci < len(_cum) - 1 and _cum[_ci] < _bpos:
                    _ci += 1
                _wc[_ci] += 1.0
            _nb = sum(_wb[_i] for _i in _scored_pos if _i < len(_wb))
            _nc = sum(_wc[_i] for _i in _scored_pos if _i < len(_wc))
            _nb, _nc = int(_nb), int(_nc)
            if _nb > 0:
                n_bytes, n_chars, _byte_exact = max(1, _nb), max(1, _nc), True
                _byte_src = "token_pieces"
        elif not _shared and _tokenizer_preserves_byte_count(
                tokenizer, input_ids_all[0].tolist(), corpus):
            # No span was shared, so the offset partition IS the exact one --
            # BUT ONLY IF THE TOKENIZER DID NOT NORMALISE THE TEXT.
            #
            # A normalising tokenizer scores a DIFFERENT byte sequence from the
            # source, so counting source bytes gives a denominator the model
            # never saw. Pythia-70m maps decomposed 'e' + combining acute + 'b'
            # (4 source bytes) onto the SAME ids [860, 67] as the composed form
            # (3 bytes); charging the extra byte to the scored targets gave
            # bits_per_byte 3.292712874841 against 9.878138624522 (-66.6667%),
            # and 50 instead of 48 scored bytes on the Pythia sentence
            # (-4.000%) (codex round 13, 2026-09-08).
            #
            # The guard is a byte-count round-trip, which is exactly the
            # condition that matters and costs one decode.
            _nb = sum(_wb[_i] for _i in _scored_pos if _i < len(_wb))
            _nc = sum(_wc[_i] for _i in _scored_pos if _i < len(_wc))
            _nb, _nc = int(round(_nb)), int(round(_nc))
            if _nb > 0:
                n_bytes, n_chars, _byte_exact = max(1, _nb), max(1, _nc), True
        # Otherwise ownership cannot be established and BPB is WITHHELD below
        # rather than fabricated -- the same rule the no-offsets path follows.
    _byte_mode = (_byte_src if _byte_exact else "none")
    if not _byte_exact:
        # NO FABRICATED DENOMINATOR. Scaling total bytes by a token-count
        # fraction has no exact byte interpretation: a slow-tokenizer anchor
        # whose 6 scored targets cover 6 ASCII bytes was assigned 15 of 22,
        # giving BPB 0.400000 where the true value is 1.000000 (adversarial
        # review round 2, 2026-09-04).
        #
        # There is exactly one case that is still well defined without offsets:
        # when EVERY token but the stream's first is scored, the covered span
        # is the whole corpus minus that one token. Its width is unknown
        # without offsets, so the full corpus is an UPPER BOUND on the
        # denominator and BPB/BPC are correspondingly a LOWER bound -- by at
        # most one token's worth, ~0.02% on a real corpus. That is disclosed as
        # `full_corpus_upper_bound`, not passed off as exact. Any other scored
        # set (a stride that reaches the window, so later chunks lose their
        # position-0 target) is genuinely unknown and is withheld.
        if _scored_pos == set(range(1, n_total)):
            n_bytes = max(1, len(corpus_bytes))
            n_chars = max(1, len(corpus))
            _byte_mode = "full_corpus_upper_bound"
        else:
            n_bytes, n_chars = 0, 0
    # `max_length` is the window ACTUALLY used after the context clamp. The
    # clamp's own warning told the reader to "see _meta_ppl_window", and that
    # key did not exist anywhere in src/ -- so a clamped model's ppl_overall
    # was silently on a different protocol, worth ~23% by the code's own
    # numbers, with no way to tell from the output (adversarial review,
    # 2026-09-04).
    return (float(np.exp(mean_nll)), float(mean_nll), n_scored,
            n_bytes, n_chars, int(max_length), int(stride),
            _byte_mode)


def centre_and_rescale(X, centre: bool = True, want_torch: bool | None = None):
    """Promote -> centre -> power-of-two rescale, in the ONE order that is safe.

    Returns ``(Z, exp2)`` where ``Z`` is the prepared array and ``exp2`` is the
    binary exponent that was factored out, so a caller reporting an
    ABSOLUTE-scale quantity can undo it with ``ldexp(value, exp2)``. A caller
    reporting a scale-FREE quantity (an entropy, an effective rank, a cosine)
    ignores it.

    Why this exists, and why the order is not negotiable
    ----------------------------------------------------
    Four tasks were found with the same defect on 2026-09-07 -- `matrix_entropy`,
    `collapse`, `contextualization`, `prediction_alignment` -- plus three more
    in the same family. Every one of them had been "fixed" for a scale bug at
    least once already. The three failure modes interact, and fixing any one of
    them in isolation reintroduces another:

    1. **Cast, then centre** loses a common offset to quantisation. In float32
       the payload of ``1e6 + 0.125`` is already gone, so subtracting the mean
       afterwards cannot recover it. `matrix_entropy` moved **-80.17%** under a
       +1e6 offset and returned NaN at +1e8, for a statistic Wei et al. define
       to be translation-invariant.
    2. **Centre in float32** overflows in the SUBTRACTION itself. A unit-max
       cloud scaled to 2.4e38 has centred entries beyond 3.4e38, so ``X - mean``
       returns inf. In `collapse` this survived wiring `safe_svdvals`, because
       the overflow was one line ABOVE the SVD -- the helper was right, the call
       site was right, and the answer was still NaN.
    3. **Rescale before centring** -- which is what round 4 changed
       `matrix_entropy` TO, correctly, because centring first lost the data at
       denormal scales. That fix created failure 1. Ordering fixes trade against
       each other unless the promotion comes first.

    Promoting to float64 first makes all three go away together: the offset
    survives (2), the subtraction cannot overflow (1), and the rescale then has
    centred data to bring into range (3).

    The rescale is a POWER OF TWO via ``ldexp``, not a division by the maximum.
    Dividing introduces rounding, which made *whether the guard fired* visible
    in the result -- the same fixture returned different values depending on
    which scale route reached it (codex round 5, 2026-09-05). ``ldexp`` shifts
    exponents only, so it is exact, and it is skipped entirely inside
    1e-6..1e6 so ordinary data takes no path change at all.
    """
    import numpy as _np
    import torch as _t

    is_torch = isinstance(X, _t.Tensor) if want_torch is None else want_torch
    if is_torch:
        Z = X.detach().double()
        if centre:
            Z = Z - Z.mean(dim=0, keepdim=True)
        gmax = float(Z.abs().max()) if Z.numel() else 0.0
    else:
        Z = _np.asarray(X, dtype=_np.float64)
        if centre:
            Z = Z - Z.mean(axis=0, keepdims=True)
        gmax = float(_np.max(_np.abs(Z))) if Z.size else 0.0

    exp2 = 0
    if gmax > 0 and _np.isfinite(gmax) and not (1e-6 < gmax < 1e6):
        import math as _math
        exp2 = int(max(-1022, min(1023, round(_math.log2(gmax)))))
        Z = _t.ldexp(Z, _t.tensor(-exp2)) if is_torch else _np.ldexp(Z, -exp2)
    return Z, exp2
