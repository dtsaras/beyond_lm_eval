"""dynamics_gradient_flow: the headline end-to-end log ratio against autograd.

gradient_flow_log_ratio_in_out = log(mean ||dL/dh_0||) - log(mean ||dL/dh_L||),
h_0 the input of block 0 and h_L the input of the final norm, rows at
position 0 excluded, means over passages. The reference registers its own
pre-hooks and takes the gradients with torch.autograd.grad (as
test_comprehensive_parity::test_dynamics_gradient_flow does for the
per-layer profile, which is pinned there; the headline key is not).
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "invariance"))
import _family_d as fd  # noqa: E402
from blme.tasks.dynamics.gradient_flow import GradientFlowTask  # noqa: E402

TEXTS = fd.TEXTS[:3]


def _reference_endpoint_norms(model, tok, text):
    for p in model.parameters():
        p.requires_grad_(False)
    seen = {}

    def grab(key):
        def hook(module, args):
            args[0].requires_grad_(True)
            seen[key] = args[0]
        return hook

    hs = [model.transformer.h[0].register_forward_pre_hook(grab("in")),
          model.transformer.ln_f.register_forward_pre_hook(grab("out"))]
    try:
        ids = tok(text, return_tensors="pt", truncation=True, max_length=128)["input_ids"]
        logits = model(ids).logits
    finally:
        for h in hs:
            h.remove()
    loss = F.cross_entropy(logits[0, :-1].float(), ids[0, 1:])
    g_in, g_out = torch.autograd.grad(loss, [seen["in"], seen["out"]])
    return (float(g_in.double()[0, 1:].norm()), float(g_out.double()[0, 1:].norm()))


def test_log_ratio_matches_autograd_endpoints():
    model, tok = fd.tiny_gpt2(), fd.tokenizer()
    res = GradientFlowTask({"num_samples": len(TEXTS)}).evaluate(
        model, tok, dataset=[{"text": t} for t in TEXTS])
    assert "error" not in res, res
    assert res["_meta_sink_n_massive_dropped"] == 0
    ref = np.array([_reference_endpoint_norms(model, tok, t) for t in TEXTS])
    n_in, n_out = ref.mean(axis=0)
    want = math.log(n_in) - math.log(n_out)
    assert abs(want) > 1e-2
    assert res["gradient_flow_log_ratio_in_out"] == pytest.approx(want, rel=1e-5, abs=1e-7)
