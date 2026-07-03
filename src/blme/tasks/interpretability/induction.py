
from ...tasks.base import DiagnosticTask
from ...registry import register_task
from ..common import get_layers
import torch
import numpy as np
import random
from tqdm import tqdm
import logging
logger = logging.getLogger("blme")


def _find_attn_out_proj(layer):
    """Locate the attention output projection for a transformer block.

    Returns the nn.Module (Linear or Conv1D) that maps the concatenated
    per-head attention outputs back to the residual stream, or None if
    not found.
    """
    # Llama / Mistral / Qwen / Gemma
    sa = getattr(layer, "self_attn", None)
    if sa is not None:
        for name in ("o_proj", "out_proj"):
            if hasattr(sa, name):
                return getattr(sa, name)
    # GPT-2 / GPT-Neo / CodeGen
    attn = getattr(layer, "attn", None)
    if attn is not None:
        for name in ("c_proj", "out_proj"):
            if hasattr(attn, name):
                return getattr(attn, name)
    # Pythia / GPT-NeoX
    attention = getattr(layer, "attention", None)
    if attention is not None:
        for name in ("dense", "out_proj"):
            if hasattr(attention, name):
                return getattr(attention, name)
    return None


def _make_head_ablation_pre_hook(head_indices, head_dim):
    """Forward pre-hook that zeroes out the slice(s) of the o_proj input
    corresponding to the given head indices."""
    def pre_hook(module, args):
        if not args:
            return args
        x = args[0]
        x = x.clone()
        for h in head_indices:
            x[..., h * head_dim : (h + 1) * head_dim] = 0.0
        return (x,) + tuple(args[1:])
    return pre_hook


def _induction_score_per_head(att, N):
    """Per-head induction (prefix-matching) score for a repeated random
    sequence ``[r_0..r_{N-1} r_0..r_{N-1}]`` of length ``2N``.

    The induction "stripe" is the attention diagonal at offset ``1 - N``:
    query position ``k`` attending to key ``(k - N) + 1`` — the token that
    followed the previous occurrence of the current token. This is exactly
    the official **TransformerLens** ``induction_score`` kernel (Olsson et al.
    2022): ``pattern.diagonal(offset=1-seq_len).mean()`` over the full
    diagonal, i.e. query rows ``k in [N-1, 2N-1]`` (``N+1`` entries).

    Args:
        att: (H, T, T) attention pattern for one layer/sample (torch or numpy).
        N:   the repeat-block length (``seq_len``); the sequence length is 2N.

    Returns:
        list[float] of length H — the per-head induction score in [0, 1].

    Note (2026-07 fix): previously averaged only ``k in [N, 2N-2]`` (``N-1``
    entries), which dropped the two endpoint stripe entries and read ~0.03
    below the published TransformerLens number. Now averages the full
    diagonal for exact parity with the official metric.
    """
    if hasattr(att, "detach"):
        att = att.detach().float().cpu().numpy()
    att = np.asarray(att, dtype=np.float64)
    H = att.shape[0]
    offset = -(N - 1)
    return [float(np.diagonal(att[h], offset=offset).mean()) for h in range(H)]


@register_task("interpretability_induction_heads")
class InductionHeadTask(DiagnosticTask):
    """
    Identifies induction heads by measuring their ability to copy the token
    that followed a previous occurrence of the current token.
    Ref: Olsson et al., "In-context Learning and Induction Heads" (2022)

    Two complementary signals are reported:
      1. **Prefix-matching score** (attention-based): for the repeated
         sequence "A B C ... A B C ...", we check whether each head at
         position k attends to position (k - N) + 1 — the token after the
         previous occurrence of the current token.
      2. **Causal validation score** (OV-side): we ablate the top-k heads
         identified by the prefix-matching score and measure how much the
         model's next-token accuracy on the second half of the repeated
         sequence drops, relative to ablating an equal number of random
         heads. Heads that pass both checks are genuine induction heads —
         this addresses the Jain & Wallace 2019 critique that attention
         weights alone don't establish causal use.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Induction Head Analysis...")
        
        # We need a synthetic dataset of repeated random tokens to isolate induction behavior
        # "A B ... A B" pattern.
        # Construct random sequences of tokens.
        
        vocab_size = tokenizer.vocab_size
        seq_len = self.config.get("seq_len", 30) # Short sequence
        num_samples = self.config.get("num_samples", 20)
        
        scores = [] # (L, H)
        
        with torch.no_grad():
            for _ in tqdm(range(num_samples), desc="Analyzing Heads"):
                # Generate random sequence
                rand_tokens = torch.randint(0, vocab_size, (1, seq_len))
                
                # Repeat it: [A B C ... A B C ...]
                input_ids = torch.cat([rand_tokens, rand_tokens], dim=1).to(model.device)
                
                # Forward pass
                outputs = model(input_ids, output_attentions=True)
                attentions = outputs.attentions # (L, B, H, T, T)
                
                if not attentions or attentions is None or len(attentions) == 0 or attentions[0] is None:
                    return {"error": "Model does not return attention weights. Reload with attn_implementation='eager'."}
                
                # We analyze the second half of the sequence (the repetition)
                # For a token at pos `i` (in 2nd half), we check if it attends to `i - seq_len + 1`?
                # No. Induction head: content at `i` matches content at `j`. 
                # Head at `i+1` (next token prediction) should attend to `j+1`.
                # Here we are looking at attention *at* token `i`.
                # If current token is X (at pos i), and previous X was at pos j.
                # Induction head at `i` should attend to `j+1`.
                
                # In our repeated sequence:
                # Sequence 1: 0 to N-1
                # Sequence 2: N to 2N-1
                # Token at `k` (where k >= N) corresponds to token at `k-N`.
                # Previous occurrence of token `input_ids[k]` is at `k-N`.
                # We want to predict `input_ids[k+1]`.
                # So head at `k` should attend to `(k-N) + 1`.
                
                # Wait, standard definition:
                # Induction head attends to the token *after* the previous copy of the current token.
                # Current token is `input_ids[k]`. Previous copy is `input_ids[j]`.
                # Head at `k` should attend to `j+1`.
                # In our setup: `k` is in [N, 2N-2].
                # `input_ids[k] == input_ids[k-N]`.
                # We want head at `k` to attend to `(k-N) + 1`.
                
                T_total = input_ids.shape[1]
                N = seq_len
                
                sample_scores = []
                
                for layer_idx, layer_att in enumerate(attentions):
                    # layer_att: (B, H, T, T)
                    # Squeeze batch
                    att = layer_att[0] # (H, T, T)
                    
                    # We only care about queries in the second half
                    # From N to 2N-2 (last token 2N-1 has no next token in this tensor usually, or it does?)
                    # Attention matrix is TxT.
                    
                    # Per-head induction score over the FULL induction diagonal
                    # (offset 1-N) — the official TransformerLens kernel. See
                    # _induction_score_per_head (2026-07: was [N, 2N-2], which
                    # dropped 2 endpoint entries and read ~0.03 below TL).
                    sample_scores.append(_induction_score_per_head(att, N))
                
                scores.append(np.array(sample_scores)) # (L, H)
                
        # Average over samples
        if not scores:
            return {"error": "No scores computed"}

        avg_scores = np.mean(np.stack(scores), axis=0)  # (L, H)
        num_layers, num_heads_per_layer = avg_scores.shape

        num_top = min(5, avg_scores.size)
        top_flat_idx = np.argsort(avg_scores, axis=None)[::-1][:num_top]
        top_heads_indices = np.unravel_index(top_flat_idx, avg_scores.shape)
        top_heads = []
        top_head_pairs = []  # list of (layer_idx, head_idx)
        for i in range(num_top):
            l = int(top_heads_indices[0][i])
            h = int(top_heads_indices[1][i])
            top_heads.append(f"L{l}H{h}: {avg_scores[l, h]:.4f}")
            top_head_pairs.append((l, h))

        result = {
            "max_induction_score": float(np.max(avg_scores)),
            "avg_induction_score": float(np.mean(avg_scores)),
            "prefix_match_score_max": float(np.max(avg_scores)),
            "prefix_match_score_mean": float(np.mean(avg_scores)),
            "top_induction_heads": top_heads,
        }

        # ── Causal validation (OV-side check) ─────────────────────────────
        # Ablate the top-k prefix-matching heads and measure how much the
        # next-token accuracy drops on the second half of the repeated
        # sequences. Compare to the drop from ablating an equal number of
        # random heads. Heads that pass both checks are validated.
        try:
            layers = get_layers(model)
            if layers is None:
                logger.info("  Skipping causal validation: could not detect layers")
                result["causal_validation_score"] = None
                return result

            # Resolve head dimension. Most architectures expose
            # num_attention_heads + hidden_size on the config.
            cfg = getattr(model, "config", None)
            n_heads_cfg = getattr(cfg, "num_attention_heads", None) if cfg else None
            hidden_size = getattr(cfg, "hidden_size", None) if cfg else None
            if n_heads_cfg is None or hidden_size is None or n_heads_cfg == 0:
                logger.info("  Skipping causal validation: missing num_attention_heads/hidden_size")
                result["causal_validation_score"] = None
                return result
            # Gemma 3+ and some other models set head_dim explicitly
            # (it may differ from hidden_size // num_heads).
            head_dim = getattr(cfg, "head_dim", None) or (hidden_size // n_heads_cfg)

            # Group head pairs by layer for hook bookkeeping. Note that the
            # number of heads in `avg_scores` may exceed cfg num_heads on
            # GQA models — clip to be safe.
            def group_by_layer(pairs):
                grouped = {}
                for (l, h) in pairs:
                    if h >= n_heads_cfg:
                        continue
                    grouped.setdefault(l, []).append(h)
                return grouped

            top_grouped = group_by_layer(top_head_pairs)

            # Pick K random heads as a control baseline (excluding the top-k).
            top_set = set(top_head_pairs)
            all_heads = [(l, h) for l in range(num_layers) for h in range(n_heads_cfg)]
            available = [p for p in all_heads if p not in top_set]
            rng = random.Random(0)
            rand_pairs = rng.sample(available, k=min(len(top_head_pairs), len(available)))
            rand_grouped = group_by_layer(rand_pairs)

            def measure_induction_accuracy(grouped_ablation):
                """Run the same synthetic repeated sequences with the
                specified set of heads ablated and return mean next-token
                accuracy on the second half."""
                handles = []
                try:
                    for l_idx, head_list in grouped_ablation.items():
                        if l_idx >= len(layers):
                            continue
                        proj = _find_attn_out_proj(layers[l_idx])
                        if proj is None:
                            continue
                        handles.append(
                            proj.register_forward_pre_hook(
                                _make_head_ablation_pre_hook(head_list, head_dim)
                            )
                        )

                    correct, total = 0, 0
                    rng_local = random.Random(123)
                    with torch.no_grad():
                        for _ in range(num_samples):
                            rand_tokens = torch.randint(
                                0, vocab_size, (1, seq_len),
                                generator=torch.Generator().manual_seed(rng_local.randint(0, 2**31)),
                            )
                            input_ids_local = torch.cat([rand_tokens, rand_tokens], dim=1).to(model.device)
                            out = model(input_ids_local)
                            preds = out.logits[0, :-1].argmax(dim=-1)
                            targets = input_ids_local[0, 1:]
                            # Score only the second half (positions where
                            # induction would help — i.e., k in [N-1, 2N-2]
                            # predicts token at k+1 in [N, 2N-1])
                            mask_start = seq_len - 1
                            correct += (preds[mask_start:] == targets[mask_start:]).sum().item()
                            total += (len(preds) - mask_start)
                    return correct / max(1, total)
                finally:
                    for h in handles:
                        h.remove()

            baseline_acc = measure_induction_accuracy({})  # no ablation
            top_ablated_acc = measure_induction_accuracy(top_grouped)
            rand_ablated_acc = measure_induction_accuracy(rand_grouped)

            # Causal validation: how much more does ablating top-k heads
            # hurt induction accuracy compared to ablating random heads?
            # Positive = top-k matters more than random. Reported as a
            # *raw* accuracy difference (not normalised by baseline) so
            # that cross-model comparisons aren't distorted by the
            # model-dependent baseline level (pythia-70m has baseline
            # ≈ 0.5 while llama3-8b has ≈ 0.98; dividing by those
            # numbers artificially inflates small-model scores). The
            # accuracy drop itself is already bounded in [-1, 1].
            top_drop = baseline_acc - top_ablated_acc
            rand_drop = baseline_acc - rand_ablated_acc
            causal_validation = top_drop - rand_drop

            result.update({
                "induction_baseline_acc": float(baseline_acc),
                "induction_acc_after_top_ablation": float(top_ablated_acc),
                "induction_acc_after_random_ablation": float(rand_ablated_acc),
                "causal_validation_score": float(causal_validation),
                "validated_top_heads": [
                    s for s, (l, h) in zip(top_heads, top_head_pairs)
                    if causal_validation > 0  # if globally validated
                ] if causal_validation > 0 else [],
            })
        except Exception as e:
            logger.info(f"  Causal validation failed: {type(e).__name__}: {e}")
            result["causal_validation_score"] = None

        return result
