import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from scipy.stats import kurtosis

from ...tasks.base import DiagnosticTask
from ...registry import register_task
from ..common import get_layers
import logging
logger = logging.getLogger("blme")


try:
    from transformers.pytorch_utils import Conv1D as _HFConv1D
except Exception:
    _HFConv1D = None


def _is_projection_module(m):
    """Projection can be nn.Linear or HuggingFace Conv1D (GPT-2 / CodeGen).
    Both expose a 2-D `weight` and accept an (..., in) input tensor."""
    if isinstance(m, torch.nn.Linear):
        return True
    if _HFConv1D is not None and isinstance(m, _HFConv1D):
        return True
    return False


def _find_down_proj(mlp):
    """Find the 'down projection' at the end of an MLP block.

    Names by architecture family:
      Llama/Qwen/Gemma/Mistral/Mixtral:  mlp.down_proj    (nn.Linear)
      GPT-2 / GPT-Neo:                   mlp.c_proj       (Conv1D)
      Pythia/GPT-NeoX:                   mlp.dense_4h_to_h (nn.Linear)
      OLMo:                              mlp.ff_out        (nn.Linear)
      Phi-2:                             mlp.fc2           (nn.Linear)
      Generic fallback:                  last Linear/Conv1D submodule
    """
    for attr in ("down_proj", "c_proj", "dense_4h_to_h", "ff_out", "fc2",
                 "out_proj", "proj"):
        if hasattr(mlp, attr):
            m = getattr(mlp, attr)
            if _is_projection_module(m):
                return m
    # Fallback: last projection-like submodule
    last = None
    for sub in mlp.modules():
        if _is_projection_module(sub):
            last = sub
    return last


@register_task("interpretability_sparsity")
class ActivationSparsityTask(DiagnosticTask):
    """
    Measures MLP intermediate activation sparsity (L0) and kurtosis.

    **Fix (2026-04-15 audit):** previous implementation hooked the MLP's
    full output (post-residual addition), which captures the dense
    residual stream, not the sparse MLP intermediate. That made L0 > 0.999
    for 88% of models (non-discriminative). The new implementation uses a
    pre-forward hook on down_proj to capture its INPUT — i.e. the tensor
    of shape (batch, seq, intermediate_size) containing the post-activation
    MLP neuron values. The threshold is also raised from 1e-5 to 1e-2 so
    bf16/fp16 numerical noise doesn't count as "active".
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Activation Sparsity (MLP intermediate)...")
        num_samples = self.config.get("num_samples", 5)
        threshold = float(self.config.get("l0_threshold", 1e-2))

        device = next(model.parameters()).device

        if dataset is None:
            from ...cache import load_default_corpus
            dataset = load_default_corpus(num_samples)

        samples = list(dataset)[:num_samples]
        if len(samples) < 1:
            return {"error": "Need at least 1 sample"}

        layers = get_layers(model)
        num_layers = len(layers)

        activation_stats = defaultdict(lambda: {"l0_rates": [], "kurtosis_vals": []})
        hooks = []

        # Pre-forward hook: captures the tensor FED INTO down_proj (i.e. the
        # MLP intermediate, post-activation). For a gated MLP this is
        # silu(gate_proj(x)) * up_proj(x); for a vanilla MLP it's act(up_proj(x)).
        # Either way, this is the "neuron" axis that interpretability cares about.
        def get_pre_hook(layer_idx):
            def pre_hook(module, args, kwargs=None):
                # args[0] is the input tensor of shape (B, T, D_intermediate)
                if not args:
                    return
                x = args[0]
                if not isinstance(x, torch.Tensor):
                    return
                # L0 at threshold — relative threshold would be ideal but
                # absolute 1e-2 is conservative and architecture-independent
                active = (x.abs() > threshold).float()
                l0_rate = active.mean().item()
                # Kurtosis across the entire intermediate tensor
                flat = x.detach().float().cpu().numpy().flatten()
                # Subsample if too large (kurtosis on > 1M values is slow)
                if flat.size > 500_000:
                    idx = np.random.default_rng(0).choice(flat.size, 500_000, replace=False)
                    flat = flat[idx]
                try:
                    k = float(kurtosis(flat, fisher=True))
                except Exception:
                    k = float("nan")
                activation_stats[layer_idx]["l0_rates"].append(l0_rate)
                activation_stats[layer_idx]["kurtosis_vals"].append(k)
            return pre_hook

        # Attach hooks to each layer's down_proj
        for i, layer in enumerate(layers):
            mlp = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
            if mlp is None:
                # Some architectures may structure differently; skip layer.
                continue
            down_proj = _find_down_proj(mlp)
            if down_proj is None:
                continue
            h = down_proj.register_forward_pre_hook(get_pre_hook(i))
            hooks.append(h)

        if not hooks:
            return {"error": "Could not find MLP down_proj in any layer"}

        try:
            with torch.no_grad():
                for s in samples:
                    text = s["text"] if isinstance(s, dict) and "text" in s else str(s)
                    inputs = tokenizer(text, return_tensors="pt",
                                        truncation=True, max_length=512).to(device)
                    model(**inputs)
        finally:
            for h in hooks:
                h.remove()

        results = {}
        mean_l0_rates = []
        mean_kurtosis = []
        for i in range(num_layers):
            if i in activation_stats:
                layer_l0 = np.mean(activation_stats[i]["l0_rates"])
                layer_kurt = np.mean(activation_stats[i]["kurtosis_vals"])
                mean_l0_rates.append(float(layer_l0))
                mean_kurtosis.append(float(layer_kurt))
            else:
                mean_l0_rates.append(float("nan"))
                mean_kurtosis.append(float("nan"))

        results["layer_l0_rates"] = mean_l0_rates
        results["layer_kurtosis"] = mean_kurtosis
        results["l0_threshold"] = threshold
        results["hook_target"] = "down_proj_input"  # documents where we measured
        valid_l0 = [v for v in mean_l0_rates if not np.isnan(v)]
        valid_kurt = [v for v in mean_kurtosis if not np.isnan(v)]
        results["global_mean_l0"] = float(np.mean(valid_l0)) if valid_l0 else 0.0
        results["global_mean_kurtosis"] = float(np.mean(valid_kurt)) if valid_kurt else 0.0

        return results
