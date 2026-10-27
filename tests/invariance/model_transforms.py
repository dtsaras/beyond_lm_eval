"""Function-preserving (and deliberately non-preserving) model transforms.

Supported families: GPT-2 (``transformer.h``), GPT-NeoX/Pythia
(``gpt_neox.layers``) and the Llama family (``model.layers``: llama, mistral,
qwen2, qwen3). Every transform deep-copies; the input model is never modified.

Exactness, max |dlogit| on single-threaded fp32 CPU forwards (tiny random
gpt2/neox/llama/qwen3, gpt2, pythia-160m, TinyLlama-1.1B):

  insert_identity_blocks        exact, 0.0: an inserted block adds exact zeros
                                to the residual stream.
  redecompose                   exact, 0.0, for sequential blocks (GPT-2,
                                Llama, GPT-NeoX with use_parallel_residual=
                                False). Impossible for parallel-residual blocks
                                (Pythia): the MLP reads the block input, which
                                a second block cannot see.
  residual_scale_gauge          approximate: norms are scale invariant only up
                                to eps. c=2 on text: gpt2 2.8e-3 (logits up to
                                198), pythia-160m 0.34 (logits up to 844); tiny
                                random models 5e-4..2e-2. With rescale_eps=True:
                                exact 0.0 for power-of-two c, float rounding
                                otherwise (gpt2 c=3: 1.2e-4).
  final_norm_unembedding_gauge  exact 0.0 for power-of-two D; otherwise float32
                                rounding, <= 2.4 eps32 * max|logit| measured.
  pad_vocabulary(kind="masked") exact 0.0 on the original logits; the added
                                logits are -inf, so the softmax is unchanged.
                                "zero"/"small_random" are NOT function-
                                preserving: the added tokens take probability
                                mass (gpt2 log-softmax shifts by up to 155,
                                because its logits are large and negative).
  zero_pad_weight_matrices      weight-only, not runnable.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np
import torch

from ._threads import limit_threads

_LLAMA_FAMILY = ("llama", "mistral", "qwen2", "qwen3")


@dataclass(frozen=True)
class _Arch:
    family: str
    get_blocks: Callable
    set_blocks: Callable
    attn_out: Callable          # block -> residual-writing attention projection
    mlp_out: Callable           # block -> residual-writing MLP projection
    input_adds: Callable        # model -> modules whose outputs form the initial residual
    final_norm: Callable
    unembed: Callable
    input_embed: Callable
    depth_keys: Tuple[str, ...]
    is_sequential: Callable     # block -> bool
    block_norms: Callable       # block -> norms that read the residual stream


def _arch(model) -> _Arch:
    t = getattr(model.config, "model_type", None)
    if t == "gpt2":
        return _Arch(
            family="gpt2",
            get_blocks=lambda m: m.transformer.h,
            set_blocks=lambda m, b: setattr(m.transformer, "h", b),
            attn_out=lambda b: b.attn.c_proj,
            mlp_out=lambda b: b.mlp.c_proj,
            input_adds=lambda m: [m.transformer.wte, m.transformer.wpe],
            final_norm=lambda m: m.transformer.ln_f,
            unembed=lambda m: m.lm_head,
            input_embed=lambda m: m.transformer.wte,
            depth_keys=("n_layer",),
            is_sequential=lambda b: True,
            block_norms=lambda b: [b.ln_1, b.ln_2],
        )
    if t == "gpt_neox":
        return _Arch(
            family="gpt_neox",
            get_blocks=lambda m: m.gpt_neox.layers,
            set_blocks=lambda m, b: setattr(m.gpt_neox, "layers", b),
            attn_out=lambda b: b.attention.dense,
            mlp_out=lambda b: b.mlp.dense_4h_to_h,
            input_adds=lambda m: [m.gpt_neox.embed_in],
            final_norm=lambda m: m.gpt_neox.final_layer_norm,
            unembed=lambda m: m.embed_out,
            input_embed=lambda m: m.gpt_neox.embed_in,
            depth_keys=("num_hidden_layers",),
            is_sequential=lambda b: not bool(getattr(b, "use_parallel_residual", True)),
            block_norms=lambda b: [b.input_layernorm, b.post_attention_layernorm],
        )
    if t in _LLAMA_FAMILY:
        return _Arch(
            family=t,
            get_blocks=lambda m: m.model.layers,
            set_blocks=lambda m, b: setattr(m.model, "layers", b),
            attn_out=lambda b: b.self_attn.o_proj,
            mlp_out=lambda b: b.mlp.down_proj,
            input_adds=lambda m: [m.model.embed_tokens],
            final_norm=lambda m: m.model.norm,
            unembed=lambda m: m.lm_head,
            input_embed=lambda m: m.model.embed_tokens,
            depth_keys=("num_hidden_layers",),
            is_sequential=lambda b: True,
            block_norms=lambda b: [b.input_layernorm, b.post_attention_layernorm],
        )
    raise NotImplementedError(
        f"model_type {t!r} is not supported (gpt2, gpt_neox, {', '.join(_LLAMA_FAMILY)})")


def supported(model) -> bool:
    try:
        _arch(model)
        return True
    except NotImplementedError:
        return False


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def _zero_(module: torch.nn.Module) -> None:
    with torch.no_grad():
        module.weight.zero_()
        if getattr(module, "bias", None) is not None:
            module.bias.zero_()


def _scale_(module: torch.nn.Module, c) -> None:
    with torch.no_grad():
        module.weight.mul_(c)
        if getattr(module, "bias", None) is not None:
            module.bias.mul_(c)


def is_tied(model) -> bool:
    a = _arch(model)
    return a.unembed(model).weight is a.input_embed(model).weight


def untie_embeddings(model):
    """Give the unembedding its own copy of a tied input-embedding weight.

    In place on ``model``; returns it. A no-op on untied models.
    """
    a = _arch(model)
    head = a.unembed(model)
    if head.weight is a.input_embed(model).weight:
        head.weight = torch.nn.Parameter(head.weight.detach().clone(),
                                         requires_grad=head.weight.requires_grad)
    for cfg in {id(model.config): model.config,
                id(model.config.get_text_config()): model.config.get_text_config()}.values():
        if hasattr(cfg, "tie_word_embeddings"):
            cfg.tie_word_embeddings = False
    return model


def _renumber(model, blocks: List[torch.nn.Module], layer_types: Optional[list]) -> None:
    a = _arch(model)
    a.set_blocks(model, torch.nn.ModuleList(blocks))
    for i, b in enumerate(blocks):
        for mod in b.modules():
            if hasattr(mod, "layer_idx"):
                mod.layer_idx = i
    for key in a.depth_keys + ("num_hidden_layers",):
        if hasattr(model.config, key):
            setattr(model.config, key, len(blocks))
    if layer_types is not None:
        model.config.layer_types = layer_types


def _layer_types(model) -> Optional[list]:
    lt = getattr(model.config, "layer_types", None)
    return list(lt) if lt is not None else None


# --------------------------------------------------------------------------
# Depth transforms
# --------------------------------------------------------------------------

def identity_layout(n_layers: int, after_indices: Iterable[int]) -> List[Tuple[str, int]]:
    """Block order produced by ``insert_identity_blocks``.

    Entry j is ``("orig", i)`` for original block i or ``("identity", i)`` for
    an identity copy of block i (``-1`` -> copy of block 0, placed first). The
    output of an identity entry equals the output of the entry before it (or
    the embedding output, for a leading identity).
    """
    after = list(after_indices)
    for i in after:
        if not -1 <= i < n_layers:
            raise ValueError(f"after index {i} out of range for {n_layers} blocks")
    out = [("identity", 0)] * after.count(-1)
    for i in range(n_layers):
        out.append(("orig", i))
        out.extend([("identity", i)] * after.count(i))
    return out


def insert_identity_blocks(model, after_indices: Iterable[int], verify_on=None):
    """Insert function-preserving identity blocks.

    Each inserted block is a deep copy of a real block with its attention
    output projection and MLP down-projection (weights and biases) zeroed, so
    block(x) = x + 0 + 0 = x exactly. Its attention and MLP still run. Placement
    follows ``identity_layout``; repeated indices insert several blocks. Exact:
    max |dlogit| = 0.0 (asserted when ``verify_on`` inputs are given).
    """
    new = copy.deepcopy(model).eval()
    a = _arch(new)
    old_blocks = list(a.get_blocks(new))
    old_types = _layer_types(new)
    layout = identity_layout(len(old_blocks), after_indices)
    blocks, types = [], []
    for kind, i in layout:
        if kind == "orig":
            blocks.append(old_blocks[i])
        else:
            b = copy.deepcopy(old_blocks[i])
            _zero_(a.attn_out(b))
            _zero_(a.mlp_out(b))
            blocks.append(b)
        if old_types is not None:
            types.append(old_types[i])
    _renumber(new, blocks, types if old_types is not None else None)
    if verify_on is not None:
        _verify(model, new, verify_on, atol=0.0, what="insert_identity_blocks")
    return new


def redecompose(model, block_idx: int, verify_on=None):
    """Split block ``block_idx`` into an attention-only block followed by an
    MLP-only block, keeping the function.

    For a sequential pre-LN block, h = x + attn(LN1 x), out = h + mlp(LN2 h).
    Copy A (MLP output zeroed) maps x -> h; copy B (attention output zeroed)
    maps h -> h + mlp(LN2 h) = out. Exact. The depth grows by one and the
    profile gains the mid-block state h. Raises ``NotImplementedError`` for a
    parallel-residual block, where out = x + attn(LN1 x) + mlp(LN2 x) and no
    second block can recover LN2 x.
    """
    new = copy.deepcopy(model).eval()
    a = _arch(new)
    old_blocks = list(a.get_blocks(new))
    if not 0 <= block_idx < len(old_blocks):
        raise ValueError(f"block {block_idx} out of range for {len(old_blocks)} blocks")
    src = old_blocks[block_idx]
    if not a.is_sequential(src):
        raise NotImplementedError(
            f"{a.family} block {block_idx} uses a parallel residual "
            f"(x + attn(LN1 x) + mlp(LN2 x)); it cannot be split into two "
            f"sequential blocks with the same function")
    attn_only = copy.deepcopy(src)
    _zero_(a.mlp_out(attn_only))
    mlp_only = copy.deepcopy(src)
    _zero_(a.attn_out(mlp_only))
    blocks = old_blocks[:block_idx] + [attn_only, mlp_only] + old_blocks[block_idx + 1:]
    old_types = _layer_types(new)
    types = None
    if old_types is not None:
        types = old_types[:block_idx] + [old_types[block_idx]] * 2 + old_types[block_idx + 1:]
    _renumber(new, blocks, types)
    if verify_on is not None:
        _verify(model, new, verify_on, atol=0.0, what="redecompose")
    return new


# --------------------------------------------------------------------------
# Gauges
# --------------------------------------------------------------------------

def _set_eps_scale_(norm: torch.nn.Module, factor: float) -> None:
    for name in ("eps", "variance_epsilon"):
        if hasattr(norm, name):
            setattr(norm, name, getattr(norm, name) * factor)
            return
    raise NotImplementedError(f"no eps attribute on {type(norm).__name__}")


def residual_scale_gauge(model, c: float, rescale_eps: bool = False, verify_on=None,
                         atol: Optional[float] = None):
    """Scale the residual stream by c without changing the function.

    Token (and position) embeddings and every block's residual-writing output
    projection (weights and biases) are multiplied by c, so every residual
    state is c times the original and every LayerNorm/RMSNorm sees c*x. Norms
    are scale invariant up to eps, so logits are unchanged up to eps effects,
    which grow as eps/var(x): large on random-init models (residual var ~ 4e-4),
    small on trained ones. ``rescale_eps=True`` also multiplies the eps of every
    residual-reading norm by c^2, which makes the gauge exact in exact
    arithmetic (and bit-exact for c a power of two). A tied unembedding is
    untied first (otherwise it would scale the logits).
    """
    if c <= 0:
        raise ValueError("c must be positive (a norm is invariant to c > 0 only)")
    new = untie_embeddings(copy.deepcopy(model).eval())
    a = _arch(new)
    for m in a.input_adds(new):
        _scale_(m, c)
    for b in a.get_blocks(new):
        _scale_(a.attn_out(b), c)
        _scale_(a.mlp_out(b), c)
        if rescale_eps:
            for n in a.block_norms(b):
                _set_eps_scale_(n, c * c)
    if rescale_eps:
        _set_eps_scale_(a.final_norm(new), c * c)
    if verify_on is not None:
        _verify(model, new, verify_on, atol=atol, what="residual_scale_gauge")
    return new


def _diag(D, d: int, seed: int) -> torch.Tensor:
    if D is None:
        g = np.random.default_rng(seed)
        return torch.from_numpy(np.exp(0.5 * g.standard_normal(d)))
    if np.isscalar(D):
        return torch.full((d,), float(D), dtype=torch.float64)
    D = torch.as_tensor(np.asarray(D), dtype=torch.float64).reshape(-1)
    if D.numel() != d:
        raise ValueError(f"D has {D.numel()} entries, expected {d}")
    return D


def final_norm_unembedding_gauge(model, D=None, seed: int = 0, verify_on=None,
                                 atol: Optional[float] = None):
    """gamma -> D gamma, beta -> D beta (final norm), W_U -> W_U D^-1.

    ``D`` is a scalar, a length-d vector, or None for a seeded log-normal
    (sigma 0.5) diagonal. logits = W_U (gamma*n + beta) is unchanged; the final
    hidden state h = gamma*n + beta is not, which is what this gauge probes.
    Untied first. Gemma-style (1 + w) norms are not supported.
    """
    new = untie_embeddings(copy.deepcopy(model).eval())
    a = _arch(new)
    norm = a.final_norm(new)
    head = a.unembed(new)
    Dv = _diag(D, norm.weight.shape[0], seed)
    if torch.any(Dv == 0):
        raise ValueError("D must be nonzero")
    with torch.no_grad():
        norm.weight.copy_((norm.weight.double() * Dv).to(norm.weight.dtype))
        if getattr(norm, "bias", None) is not None:
            norm.bias.copy_((norm.bias.double() * Dv).to(norm.bias.dtype))
        head.weight.copy_((head.weight.double() / Dv[None, :]).to(head.weight.dtype))
    if verify_on is not None:
        _verify(model, new, verify_on, atol=atol, what="final_norm_unembedding_gauge")
    return new


# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------

class MaskedPaddedLinear(torch.nn.Linear):
    """Unembedding with ``extra`` appended rows whose logits are -inf.

    The original logits come from ``F.linear`` on the original rows (a view),
    so they are bit-identical; the appended logits are exactly -inf, so their
    softmax mass is exactly 0. ``weight`` exposes all V + extra rows (the
    appended rows are zero) for weight-reading tasks.
    """

    def __init__(self, base: torch.nn.Linear, extra: int):
        super().__init__(base.in_features, base.out_features + extra,
                         bias=base.bias is not None,
                         device=base.weight.device, dtype=base.weight.dtype)
        self.n_orig = base.out_features
        with torch.no_grad():
            self.weight.zero_()
            self.weight[: self.n_orig].copy_(base.weight)
            if self.bias is not None:
                self.bias.zero_()
                self.bias[: self.n_orig].copy_(base.bias)

    def forward(self, x):
        b = None if self.bias is None else self.bias[: self.n_orig]
        y = torch.nn.functional.linear(x, self.weight[: self.n_orig], b)
        pad = y.new_full(y.shape[:-1] + (self.out_features - self.n_orig,), float("-inf"))
        return torch.cat([y, pad], dim=-1)


def pad_vocabulary(model, extra_rows: int, kind: str = "zero", seed: int = 0,
                   rel_scale: float = 1e-2):
    """Append ``extra_rows`` never-emitted tokens to the vocabulary.

    kind="zero"          zero input and output rows. The added logits are 0,
                         so they take softmax mass: NOT function-preserving
                         for any softmax/log-prob metric.
    kind="small_random"  rows ~ N(0, (rel_scale * row-RMS)^2). Same caveat.
    kind="masked"        zero rows plus -inf output logits (``MaskedPaddedLinear``):
                         original logits bit-identical, softmax unchanged.
                         Only this kind is function-preserving. Caveat: a
                         metric that averages raw logits, or computes
                         p * log p without xlogy, gets -inf/NaN from it.

    Input and output rows stay tied for tied models (except "masked").
    """
    if kind not in ("zero", "small_random", "masked"):
        raise ValueError(f"unknown kind {kind!r}")
    if extra_rows < 1:
        raise ValueError("extra_rows must be >= 1")
    new = copy.deepcopy(model).eval()
    a = _arch(new)
    tied = is_tied(new)
    emb = a.input_embed(new)
    V, d = emb.weight.shape
    g = torch.Generator().manual_seed(seed)

    def new_rows(W):
        if kind == "small_random":
            rms = W.detach().double().pow(2).mean().sqrt()
            r = torch.randn(extra_rows, W.shape[1], generator=g, dtype=torch.float64)
            return (r * rel_scale * rms).to(W.dtype)
        return W.new_zeros(extra_rows, W.shape[1])

    new_emb = torch.nn.Embedding(V + extra_rows, d, device=emb.weight.device,
                                 dtype=emb.weight.dtype,
                                 padding_idx=getattr(emb, "padding_idx", None))
    with torch.no_grad():
        new_emb.weight.copy_(torch.cat([emb.weight, new_rows(emb.weight)], 0))
    _set_input_embed(new, a, new_emb)

    head = a.unembed(new)
    if kind == "masked":
        new_head = MaskedPaddedLinear(head, extra_rows)
        _set_unembed(new, a, new_head)
        untie_embeddings(new)
    elif tied:
        head.weight = new_emb.weight
        head.out_features = V + extra_rows
    else:
        Vo = head.out_features
        new_head = torch.nn.Linear(d, Vo + extra_rows, bias=head.bias is not None,
                                   device=head.weight.device, dtype=head.weight.dtype)
        with torch.no_grad():
            new_head.weight.copy_(torch.cat([head.weight, new_rows(head.weight)], 0))
            if head.bias is not None:
                new_head.bias.copy_(torch.cat([head.bias, head.bias.new_zeros(extra_rows)]))
        _set_unembed(new, a, new_head)
    new.config.vocab_size = int(a.unembed(new).out_features)
    return new


def _set_input_embed(model, a: _Arch, module) -> None:
    if a.family == "gpt2":
        model.transformer.wte = module
    elif a.family == "gpt_neox":
        model.gpt_neox.embed_in = module
    else:
        model.model.embed_tokens = module


def _set_unembed(model, a: _Arch, module) -> None:
    if a.family == "gpt_neox":
        model.embed_out = module
    else:
        model.lm_head = module


# --------------------------------------------------------------------------
# Weight-only transform
# --------------------------------------------------------------------------

def zero_pad_weight_matrices(model, k: int = 2):
    """Zero-pad every >=2-D parameter inside the blocks to (k*m, k*n).

    Leaves every stable rank, Frobenius and spectral norm unchanged and
    multiplies min(m, n) by k. Weight-reading tasks only: the result cannot
    run a forward pass (calling it raises).
    """
    if int(k) != k or k < 1:
        raise ValueError("k must be a positive integer")
    new = copy.deepcopy(model)
    a = _arch(new)
    with torch.no_grad():
        for b in a.get_blocks(new):
            for p in b.parameters():
                if p.ndim < 2:
                    continue
                big = p.new_zeros(p.shape[:-2] + (k * p.shape[-2], k * p.shape[-1]))
                big[..., : p.shape[-2], : p.shape[-1]] = p
                p.data = big

    def _no_forward(*args, **kwargs):
        raise RuntimeError("zero_pad_weight_matrices: weight-only copy, not runnable")

    new.forward = _no_forward
    return new


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------

def _as_batches(inputs) -> List[torch.Tensor]:
    if isinstance(inputs, torch.Tensor):
        inputs = [inputs]
    out = []
    for x in inputs:
        x = torch.as_tensor(x)
        out.append(x[None] if x.ndim == 1 else x)
    return out


@torch.no_grad()
def logits(model, inputs, threads: Optional[int] = 1) -> List[torch.Tensor]:
    """fp64 copies of the logits for each input batch (eval mode, no KV cache).

    Single-threaded by default, so "exactly 0.0" is checkable (see ``_threads``).
    """
    model.eval()
    dev = next(model.parameters()).device
    with limit_threads(threads):
        return [model(input_ids=x.to(dev), use_cache=False).logits.detach().double().cpu()
                for x in _as_batches(inputs)]


def max_logit_delta(model_a, model_b, inputs, threads: Optional[int] = 1) -> float:
    """max |logit_a - logit_b| over the inputs, on model_a's vocabulary."""
    worst = 0.0
    for la, lb in zip(logits(model_a, inputs, threads), logits(model_b, inputs, threads)):
        V = la.shape[-1]
        worst = max(worst, float((la - lb[..., :V]).abs().max()))
    return worst


def _verify(base, new, inputs, atol, what):
    if atol is None:
        return
    delta = max_logit_delta(base, new, inputs)
    if not delta <= atol:
        raise AssertionError(f"{what}: max |dlogit| = {delta:.3e} > {atol:.3e}")
