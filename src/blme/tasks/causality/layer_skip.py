"""Layer-skip effect profile: how much later computation and the output
depend on each block (Csordas, Manning & Potts 2025).

Reference: "Do Language Models Use Their Depth Efficiently?", arXiv:2505.13898
(NeurIPS 2025); code robertcsordas/llm_effective_depth,
``analysis/analyze_future_effects.py::test_effect`` (part="layer") and
``analysis/skip_max_depth.py::effect_layer_score``.

For each prompt, up to 4 split points t are drawn from range(8, T-4, 8). For
each t and each block s, block s is skipped (its output set to its input) on
tokens 1..t-1 and the forward is rerun. On tokens >= t this records:

* for every block l, the relative change of its contribution,
  ||c_l - c'_l|| / ||c_l|| with c_l = block output - block input,
  maximised over the tokens;
* the L2 change of the output distribution, ||p - p'||_2, maximised over the
  tokens.

Both are maximised over split points and prompts. A block's importance e_s is
its mean effect on all later blocks (skip_max_depth's reduce), and the depth
score is d = sum_s s * e_s / sum_s e_s with s = 1..L (Sec. 3.3).
"""

import logging
import math
import random

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from ..common import get_layers

logger = logging.getLogger("blme")

# analyze_future_effects.py: positions = range(8, len(tokens)-4, 8), shuffled,
# the first N_CHUNKS = 4 kept; no_skip_front=1.
_STRIDE = 8
_TAIL = 4
_NO_SKIP_FRONT = 1
# The release floors the contribution norm at 1e-6. The float32 tiny floor
# gives the same ratio for every norm above 1e-6 (and 0 for a block that
# contributes nothing, as the release does) without setting a scale.
_EPS = torch.finfo(torch.float32).tiny


def split_positions(n_tokens, n_positions, rng):
    """The release's split points for a prompt of ``n_tokens`` tokens."""
    positions = list(range(_STRIDE, n_tokens - _TAIL, _STRIDE))
    rng.shuffle(positions)
    return positions[:n_positions]


def _block_input(args, kwargs):
    return args[0] if args and torch.is_tensor(args[0]) else kwargs["hidden_states"]


def prompt_effects(model, layers, input_ids, positions):
    """``test_effect(part="layer")`` for one prompt.

    Returns the (L, L) matrix of the maximal relative change of block l's
    contribution when block s is skipped (row s, column l) and the (L,) maximal
    output-distribution change, both maximised over ``positions``.
    """
    L = len(layers)
    state = {"skip": None, "t": None}
    base, effect = [None] * L, [None] * L

    def hook(i):
        def fn(module, args, kwargs, output):
            h_in = _block_input(args, kwargs)
            out = output[0] if isinstance(output, tuple) else output
            t = state["t"]
            replaced = None
            if state["skip"] == i:
                out = torch.cat([out[:, :_NO_SKIP_FRONT], h_in[:, _NO_SKIP_FRONT:t].to(out.dtype),
                                 out[:, t:]], dim=1)
                replaced = (out,) + tuple(output[1:]) if isinstance(output, tuple) else out
            # The release casts both to float32 before subtracting.
            c = out[0].float() - h_in[0].float()
            if state["skip"] is None:
                base[i] = c
            else:
                ref = base[i][t:]
                rel = (ref - c[t:]).norm(dim=-1) / ref.norm(dim=-1).clamp(min=_EPS)
                effect[i] = rel.max()
            return replaced
        return fn

    handles = [b.register_forward_hook(hook(i), with_kwargs=True) for i, b in enumerate(layers)]
    try:
        with torch.no_grad():
            probs = model(input_ids=input_ids, use_cache=False).logits[0].float().softmax(dim=-1)
            mat = torch.zeros(L, L, dtype=torch.float32)
            out_change = torch.zeros(L, dtype=torch.float32)
            for t in positions:
                for s in range(L):
                    state["skip"], state["t"] = s, t
                    logits = model(input_ids=input_ids, use_cache=False).logits[0, t:]
                    d_out = (logits.float().softmax(dim=-1) - probs[t:]).norm(dim=-1).max()
                    row = torch.stack(effect).cpu()
                    mat[s] = torch.maximum(mat[s], row)
                    out_change[s] = torch.maximum(out_change[s], d_out.cpu())
                state["skip"], state["t"] = None, None
    finally:
        for h in handles:
            h.remove()
    return mat, out_change


def importance(mat):
    """skip_max_depth.py: a block's summed effect over the later blocks,
    divided by how many there are."""
    L = mat.shape[0]
    return mat.sum(-1) / torch.arange(L - 1, -1, -1).clamp(min=1)


def depth_score(e):
    """Sec. 3.3: d = sum_s s * e_s / sum_s e_s over s = 1..L. The released
    ``effect_layer_score`` indexes from 0 and so reads d - 1."""
    e = np.asarray(e, dtype=np.float64)
    total = math.fsum(e)
    if not total > 0:
        return float("nan")
    return math.fsum(np.arange(1, e.size + 1) * e) / total


@register_task("causality_layer_skip")
class LayerSkipEffectTask(DiagnosticTask):
    """Layer-skip effect profile and depth score (Csordas et al. 2025)."""

    TOKEN_POLICY = ("paper_protocol:Csordas2025 every token; token 0 is never skipped and "
                    "effects are read on the tokens after each split point")
    SOURCES = {
        "paper": "arXiv:2505.13898 (Csordas, Manning & Potts, NeurIPS 2025)",
        "code": "robertcsordas/llm_effective_depth@393e5c740e2a46d8b06d60129b1a3fb4b581a01a",
        "llm_precedent": "arXiv:2505.13898 (Llama 3.1 8B-405B, Qwen 3 8B-32B, OLMo 2)",
    }
    ADAPTATION_NOTE = (
        "The intervention, the split points (4 per prompt from range(8, T-4, 8), Python "
        "random seeded 123), the relative contribution change, the L2 "
        "output-probability change, the maxima over tokens, split points and prompts, and the "
        "importance (mean effect on the later blocks) follow the release. The depth score uses "
        "the paper's 1-based layer index; the release's effect_layer_score is 0-based and "
        "reads one less. Prompts are the first 10 corpus passages with a split point, up to "
        "512 tokens, where the release uses 10 GSM8K problems with their solutions (class 2). "
        "The *_ratio keys divide the depth score by L (class 2), so models of different depth "
        "compare on one scale. They are not invariant to identity blocks: an appended one "
        "lowers them by L0 / L, and the release's mean over later blocks counts inserted "
        "blocks. skip_output_effect_total and skip_output_effect_max (sum and maximum over "
        "blocks of the output change) are: skipping an identity block changes nothing, and "
        "skipping a real block gives the same output as before."
    )
    HEADLINE_KEYS = ("skip_depth_score_future_ratio", "skip_depth_score_output_ratio",
                     "skip_output_effect_total", "skip_output_effect_max")
    _INV = "tests/tasks/invariance/test_causality_layer_skip.py::"
    SIZE_INVARIANT = {
        "skip_output_effect_total": (("insert_identity_blocks",),
                                     _INV + "test_output_effects_under_identity_blocks"),
        "skip_output_effect_max": (("insert_identity_blocks",),
                                   _INV + "test_output_effects_under_identity_blocks"),
    }

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Layer-Skip Effect Profile...")
        num_samples = int(self.config.get("num_samples", 10))
        n_positions = int(self.config.get("n_positions", 4))
        max_length = int(self.config.get("max_length", 512))
        rng = random.Random(int(self.config.get("position_seed", 123)))

        layers = get_layers(model)
        if layers is None:
            return {"error": "Could not detect model layers"}
        if dataset is None:
            from ...cache import load_default_corpus
            dataset = load_default_corpus(num_samples)
        device = next(model.parameters()).device

        # The first num_samples passages, short ones dropped rather than
        # replaced, so the prompts do not depend on how large a corpus the
        # co-running tasks asked for.
        mats, outs, n_short, n_split = [], [], 0, 0
        for s in list(dataset)[:num_samples]:
            text = s["text"] if isinstance(s, dict) else str(s)
            ids = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=max_length)["input_ids"].to(device)
            positions = split_positions(ids.shape[1], n_positions, rng)
            if not positions:
                n_short += 1
                continue
            mat, out = prompt_effects(model, layers, ids, positions)
            mats.append(mat)
            outs.append(out)
            n_split += len(positions)
        if not mats:
            return {"error": "No prompt is long enough for a split point (needs 13 tokens)"}

        mat_max = torch.stack(mats).amax(dim=0)
        out_max = torch.stack(outs).amax(dim=0)
        e_max = importance(mat_max)
        e_mean = importance(torch.stack(mats).mean(dim=0))
        out_mean = torch.stack(outs).mean(dim=0)
        L = len(layers)

        d_future, d_output = depth_score(e_max), depth_score(out_max)
        out_list = [float(v) for v in out_max]
        return {
            "skip_future_effect_per_layer": [float(v) for v in e_max],
            "skip_output_effect_per_layer": out_list,
            "skip_depth_score_future": d_future,
            "skip_depth_score_output": d_output,
            "skip_depth_score_future_ratio": d_future / L,
            "skip_depth_score_output_ratio": d_output / L,
            "skip_output_effect_total": math.fsum(out_list),
            "skip_output_effect_max": max(out_list),
            "_meta_skip_future_effect_matrix": mat_max.tolist(),
            # The same statistics with the mean over prompts in place of the
            # maximum (skip_max_depth.py reports both).
            "_meta_skip_depth_score_future_ratio_mean": depth_score(e_mean) / L,
            "_meta_skip_depth_score_output_ratio_mean": depth_score(out_mean) / L,
            "_meta_skip_output_effect_total_mean": math.fsum(float(v) for v in out_mean),
            "_meta_n_prompts": len(mats),
            "_meta_n_prompts_too_short": n_short,
            "_meta_n_split_points": n_split,
            "_meta_n_layers": L,
        }
