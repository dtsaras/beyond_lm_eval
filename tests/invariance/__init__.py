"""Direct mechanism tests for size-normalised metrics.

A normalised version of a metric is accepted only if it survives a direct test
(see ``src/blme/normalization.py``): it must stay put under a transform that
changes width, depth, sample count or vocabulary without changing what the
model computes, while the raw value it corrects is shown to move.

  clouds            isometric embedding, zero-padding, row subsampling and
                    rescaling of point clouds, per array or per-document list
  driver            run a task's real ``evaluate()`` on injected states through
                    a stub of ``ModelOutputCache``
  model_transforms  identity-block insertion, block re-decomposition,
                    residual-scale and final-norm/unembedding gauges,
                    vocabulary padding, weight zero-padding
  precision         bf16 storage vs bf16 arithmetic
  compare           invariance / movement assertions on result dicts

Forwards and evaluations run single-threaded by default (``limit_threads``):
a multi-threaded BLAS reduction is not bit-reproducible across thread counts.
"""
from ._threads import limit_threads
from .clouds import (
    embed_isometric, embed_isometric_docs, pool_docs, random_isometry, rescale,
    rescale_docs, subsample_docs, subsample_rows, zero_pad, zero_pad_docs,
)
from .compare import assert_invariant, assert_moves, get_key, rel_change, uniform_layer_mean
from .driver import (
    NOT_STATE_DRIVABLE, STATE_DRIVABLE_TASKS, InjectedStateCache, StubModel,
    StubTokenizer, TaskForwardAttempted, TaskNotDrivable, TaskReturnedError,
    run_task_on_model, run_task_on_states,
)
from .model_transforms import (
    MaskedPaddedLinear, final_norm_unembedding_gauge, identity_layout,
    insert_identity_blocks, is_tied, logits, max_logit_delta, pad_vocabulary,
    redecompose, residual_scale_gauge, untie_embeddings, zero_pad_weight_matrices,
)
from .precision import bf16_weights_fp32_math, precision_error_split, run_with_dtype
from .states import States, as_states, states_from_model
