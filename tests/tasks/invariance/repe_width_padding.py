"""A function-preserving width transform for runnable Llama-family models.

``zero_pad_llama_width`` maps a model of width d to one of width k*d whose
extra residual coordinates are dead: residual-reading projections get zero
input columns, residual-writing ones zero output rows, the embedding and the
unembedding zero columns. RMSNorm over k*d coordinates sees mean(x^2)/k, so its
eps is divided by k and its live gains by sqrt(k); k = 4 keeps both exact
powers of two. Logits agree to float rounding of the padded reductions.
"""
import copy
import math

import torch
import transformers


def tiny_llama(vocab, seed=0, hidden=32, layers=3):
    torch.manual_seed(seed)
    cfg = transformers.LlamaConfig(
        vocab_size=vocab, hidden_size=hidden, intermediate_size=2 * hidden,
        num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, attn_implementation="eager")
    return transformers.LlamaForCausalLM(cfg).eval()


def zero_pad_llama_width(model, k=4):
    cfg = copy.deepcopy(model.config)
    d = cfg.hidden_size
    cfg.head_dim = getattr(cfg, "head_dim", None) or d // cfg.num_attention_heads
    cfg.hidden_size = k * d
    cfg.rms_norm_eps = cfg.rms_norm_eps / k
    new = type(model)(cfg).eval()
    src = model.state_dict()
    with torch.no_grad():
        for name, p in new.state_dict().items():
            s = src[name]
            p.zero_()
            if name.endswith("norm.weight"):
                p[:d] = s / math.sqrt(k)
            elif p.ndim == 2:
                p[:s.shape[0], :s.shape[1]] = s
            else:
                p[:s.shape[0]] = s
    return new
