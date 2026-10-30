"""Drive a task's real ``evaluate()`` from injected hidden states.

``InjectedStateCache`` subclasses the library's ``ModelOutputCache`` and fills
its stores directly, so slicing, per-document splitting, ``replace_final_norm``
and ``include_embedding`` all run the library's own code. ``StubModel`` reports
the depth and width of the injected states in its config (``num_hidden_layers``,
``n_layer``, ``hidden_size``, ``n_embd``): a config without them makes
``collected_layer_count(model)`` fall back to 2 for any task that still asks the
model rather than the collected layers. Any forward pass, tokenizer call or cache re-population raises
``TaskForwardAttempted``; ``run_task_on_states`` converts that, even when the
task swallowed it, into ``TaskNotDrivable``.

Coverage was established by running every registered task through the driver
(``tests/test_invariance_harness.py`` pins it): the tasks in
``STATE_DRIVABLE_TASKS`` compute only from cached hidden states and reproduce a
real populated cache bit for bit. Everything else is listed in
``NOT_STATE_DRIVABLE`` with the reason.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from blme.cache import ModelOutputCache

from ._threads import limit_threads
from .states import States, as_states


class TaskForwardAttempted(RuntimeError):
    """A stub was asked for a forward pass, a tokenisation or a re-population."""


class TaskNotDrivable(RuntimeError):
    """The task does not compute from cached hidden states alone."""


class TaskReturnedError(RuntimeError):
    """The task returned ``{"error": ...}``."""


# Tasks that compute only from cached hidden states, each with the config it
# needs to take the cache path. Every entry reproduces a real populated cache
# bit for bit (pinned in tests/test_invariance_harness.py). lid and isoscore's
# single-layer reads take HF's post-final-norm entry, served from
# ``States.final_norm``.
STATE_DRIVABLE_TASKS: Dict[str, Dict[str, Any]] = {
    "geometry_cka": {},
    "geometry_cknna": {},
    "geometry_collapse": {},
    "geometry_hsic": {},
    # The default (non-layerwise) mode measures the embedding matrix.
    "geometry_intrinsic_dim": {"layerwise": True},
    "geometry_isoscore": {},
    "geometry_lid": {},
    "geometry_lipschitz": {},
    "geometry_magnitude": {},
    "geometry_matrix_entropy": {},
    "geometry_mp_bulk_deviation": {},
    "geometry_procrustes_linearity": {},
    "geometry_rsa": {},
    "geometry_svd": {},
    "geometry_trajectory_curvature": {},
    "geometry_vendi_score": {},
    "topology_betti_curve": {},
    "topology_homology": {},
    "topology_persistence_entropy": {},
    "topology_persistence_landscape": {},
    "interpretability_activation_kurtosis": {},
}

# Tasks the stub cache cannot drive, grouped by reason.
NOT_STATE_DRIVABLE: Dict[str, str] = {
    **{t: "runs its own forward pass to collect hidden states" for t in (
        "geometry_contextualization", "geometry_correlation_dimension",
        "geometry_effective_depth", "geometry_mahalanobis",
        "geometry_neural_collapse", "geometry_representation_sensitivity",
        "repe_concept_separability", "repe_refusal_direction",
        "repe_steering_effectiveness", "repe_task_vectors",
    )},
    "topology_zigzag_persistence": (
        "hooks the post-attention-norm sites and the final norm in its own forward"),
    "geometry_schatten": (
        "runs its own forward pass: Yusupov et al.'s Schatten norm reads the "
        "MLP-internal activations, which the cache does not hold"),
    "geometry_phd_dimension": (
        "only the pooled _meta_ value reads the cache; the per-text headline "
        "runs its own forward pass per document"),
    **{t: "needs cached logits/labels and the model's unembedding, not hidden states" for t in (
        "geometry_perplexity", "geometry_prediction_alignment",
        "consistency_calibration",
    )},
    **{t: "needs attention weights or runs attention forwards" for t in (
        "geometry_positional_decay", "interpretability_attention_entropy",
    )},
    **{t: "reads weights only; drive it with a real model" for t in (
        "geometry_weight_norms", "geometry_spectral", "geometry_unembedding",
        "geometry_hubness", "geometry_superposition_strength",
        "geometry_categories",
    )},
    **{t: "measures forward/backward dynamics or generation" for t in (
        "dynamics_coe", "dynamics_fractal", "dynamics_generation_diversity",
        "dynamics_gradient_flow", "dynamics_interpolation",
        "dynamics_perturbation_growth", "dynamics_sharpness",
        "dynamics_stability",
    )},
    "geometry_tokenizer_efficiency": "measures the tokenizer, not hidden states",
}
# Every other registered task (consistency_*, causality_*, the remaining
# interpretability_*) tokenises prompts and runs the model itself.


class _Counter:
    def __init__(self):
        self.n = 0


class _Trap(torch.nn.Module):
    """A module whose forward counts the attempt and raises."""

    def __init__(self, counter: _Counter, what: str):
        super().__init__()
        self._counter = counter
        self._what = what

    def forward(self, *args, **kwargs):
        self._counter.n += 1
        raise TaskForwardAttempted(f"{self._what}: forward pass on a stub model")


class StubConfig:
    """Attribute bag standing in for a HF config."""

    def __init__(self, **attrs):
        self.__dict__.update(attrs)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    def get_text_config(self, *args, **kwargs):
        return self

    def __repr__(self):
        return f"StubConfig({self.__dict__})"


class StubModel(torch.nn.Module):
    """Model stand-in: correct config and block count, no computation.

    ``get_layers(model)`` finds ``model.model.layers`` (n_layers trap modules),
    so tasks that count blocks see the injected depth. One frozen float32
    parameter makes ``next(model.parameters())`` resolve the device and dtype.
    """

    def __init__(self, n_layers: int, d_model: int, config_attrs: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.forward_calls = _Counter()
        attrs = dict(
            model_type="blme_invariance_stub",
            num_hidden_layers=n_layers, n_layer=n_layers,
            hidden_size=d_model, n_embd=d_model,
            name_or_path="blme-invariance-stub",
            _name_or_path="blme-invariance-stub",
        )
        attrs.update(config_attrs or {})
        self.config = StubConfig(**attrs)
        self.model = _Trap(self.forward_calls, "base model")
        self.model.layers = torch.nn.ModuleList(
            [_Trap(self.forward_calls, f"block {i}") for i in range(n_layers)])
        self._stub_anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    @property
    def device(self):
        return self._stub_anchor.device

    @property
    def dtype(self):
        return self._stub_anchor.dtype

    def forward(self, *args, **kwargs):
        self.forward_calls.n += 1
        raise TaskForwardAttempted("forward pass on a stub model")

    def generate(self, *args, **kwargs):
        self.forward_calls.n += 1
        raise TaskForwardAttempted("generate() on a stub model")


class StubTokenizer:
    """Tokenizer stand-in. Any call raises: a task that tokenises on the
    cache path is preparing a computation the injected states do not cover."""

    def __init__(self, **attrs):
        self.calls = 0
        self.bos_token_id = None
        self.eos_token_id = None
        self.pad_token_id = None
        self.bos_token = None
        self.eos_token = None
        self.pad_token = None
        self.name_or_path = "blme-invariance-stub"
        self.__dict__.update(attrs)

    def _trap(self, *args, **kwargs):
        self.calls += 1
        raise TaskForwardAttempted("tokenizer called on the stub")

    __call__ = _trap
    encode = _trap
    decode = _trap
    batch_decode = _trap
    convert_ids_to_tokens = _trap
    apply_chat_template = _trap

    def __len__(self):
        return 0


def _as_tensor(x, dtype) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().to(device="cpu", dtype=dtype)
    return torch.as_tensor(x).to(dtype=dtype)


class InjectedStateCache(ModelOutputCache):
    """A populated ``ModelOutputCache`` whose stores hold given states.

    Layout matches ``ModelOutputCache._run_forward_pass``: ``_hidden_states[l]``
    is block l for l < L-1 and HF's post-final-norm entry for l = L-1;
    ``_last_block_raw`` is the raw deepest block; ``_embedding_states`` is
    hidden_states[0]. Without ``final_norm`` the raw deepest block fills the
    post-norm slot too. Stored as float32 by default, as the real cache does.
    """

    def __init__(self, states: States, model=None, tokenizer=None,
                 store_dtype: torch.dtype = torch.float32):
        super().__init__(model, tokenizer, dataset=None, num_samples=states.n_docs)
        self.forward_attempts = 0

        def cat(docs):
            return torch.cat([_as_tensor(x, store_dtype) for x in docs], dim=0)

        L = states.n_layers
        hs = {li: cat(states.blocks[li]) for li in range(L)}
        self._last_block_raw = hs[L - 1]
        if states.final_norm is not None:
            hs[L - 1] = cat(states.final_norm)
        self._hidden_states = hs
        self._embedding_states = (cat(states.embedding)
                                  if states.embedding is not None else None)
        self._sample_lengths = list(states.lengths)
        self._need_hidden = True
        self._populated = True

    def _run_forward_pass(self) -> None:
        self.forward_attempts += 1
        raise TaskForwardAttempted("InjectedStateCache cannot run a forward pass")


def run_task_on_states(
    task_name: str,
    per_sample_states_by_layer,
    config: Optional[Dict[str, Any]] = None,
    model_stub_attrs: Optional[Dict[str, Any]] = None,
    *,
    final_norm=None,
    embedding=None,
    tokenizer=None,
    store_dtype: torch.dtype = torch.float32,
    use_default_config: bool = True,
    strict: bool = True,
    force: bool = False,
    seed: int = 42,
    threads: Optional[int] = 1,
) -> Dict[str, Any]:
    """Run ``task_name``'s real ``evaluate()`` on injected states.

    ``per_sample_states_by_layer`` is a ``States`` or ``{block: [(T_i, d)]}``;
    ``final_norm`` / ``embedding`` add those entries to a dict input. The config
    is resolved against the packaged defaults as ``blme.core.evaluate`` does,
    with the task's ``STATE_DRIVABLE_TASKS`` entry applied under ``config``;
    ``num_samples`` defaults to the number of injected documents unless
    ``config`` sets it. ``evaluate()`` runs with ``threads`` torch/BLAS
    threads (default 1, bit-reproducible; None leaves the process setting).

    Tasks outside ``STATE_DRIVABLE_TASKS`` raise ``TaskNotDrivable`` unless
    ``force``: a weight-reading task run on the stub sees no weights and can
    return zeros rather than fail. Raises ``TaskNotDrivable`` if the task
    attempted a forward pass, and (``strict``) ``TaskReturnedError`` on an
    error result.
    """
    from blme.core import _register_all_tasks
    from blme.registry import get_task
    from blme.tasks.config_loader import resolve_task_config
    from blme.utils import set_global_seed

    states = as_states(per_sample_states_by_layer, final_norm=final_norm, embedding=embedding)
    _register_all_tasks()
    task_cls = get_task(task_name)
    if task_cls is None:
        raise KeyError(f"unknown task {task_name!r}")
    if task_name not in STATE_DRIVABLE_TASKS and not force:
        why = NOT_STATE_DRIVABLE.get(task_name, "not verified against a real cache")
        raise TaskNotDrivable(f"{task_name}: {why}")

    user = {**STATE_DRIVABLE_TASKS.get(task_name, {}), **(config or {})}
    cfg = resolve_task_config(task_name, user) if use_default_config else dict(user)
    cfg.setdefault("seed", seed)
    if "num_samples" not in user:
        cfg["num_samples"] = states.n_docs
    cfg["use_cache"] = True

    model = StubModel(states.n_layers, states.d, model_stub_attrs)
    tok = tokenizer if tokenizer is not None else StubTokenizer()
    cache = InjectedStateCache(states, model, tok, store_dtype=store_dtype)
    dataset = [{"text": f"<injected document {i}>"} for i in range(states.n_docs)]

    set_global_seed(seed)
    try:
        with limit_threads(threads):
            result = task_cls(config=cfg).evaluate(model, tok, dataset=dataset, cache=cache)
    except TaskForwardAttempted as e:
        raise TaskNotDrivable(f"{task_name}: {e}") from e
    attempts = (model.forward_calls.n + cache.forward_attempts
                + int(getattr(tok, "calls", 0)))
    if attempts:
        raise TaskNotDrivable(
            f"{task_name}: attempted {attempts} forward pass(es) or tokenisation(s); "
            f"its result is not computed from the injected states")
    if strict and isinstance(result, dict) and "error" in result:
        raise TaskReturnedError(f"{task_name}: {result['error']}")
    return result


def run_task_on_model(task_name: str, model, tokenizer, texts, config=None,
                      *, seed: int = 42, use_cache: bool = True,
                      threads: Optional[int] = 1) -> Dict[str, Any]:
    """Reference path: the same task on a real model, with a real populated
    cache built from ``texts`` (mirrors ``blme.core.evaluate``). Applies the
    same ``STATE_DRIVABLE_TASKS`` config and thread count as
    ``run_task_on_states``."""
    from blme.core import _register_all_tasks
    from blme.registry import get_task
    from blme.tasks.config_loader import resolve_task_config
    from blme.utils import set_global_seed

    _register_all_tasks()
    user = {**STATE_DRIVABLE_TASKS.get(task_name, {}), **(config or {})}
    cfg = resolve_task_config(task_name, user)
    cfg.setdefault("seed", seed)
    if "num_samples" not in user:
        cfg["num_samples"] = len(texts)
    dataset = [{"text": t} for t in texts]
    cache = None
    with limit_threads(threads):
        if use_cache:
            cache = ModelOutputCache(model, tokenizer, dataset=dataset, num_samples=len(texts))
            cache.populate(need_hidden=True, need_attentions=False, need_logits=False)
        set_global_seed(seed)
        return get_task(task_name)(config=cfg).evaluate(
            model, tokenizer, dataset=dataset, cache=cache)
