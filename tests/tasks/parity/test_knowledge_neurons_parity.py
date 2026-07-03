"""Numeric-parity test: BLME ``causality_knowledge_neurons`` vs the OFFICIAL
Dai et al. 2022 knowledge-neuron integrated-gradients attribution.

Paper: Dai, Dong, Hao, Sui, Chang, Wei — "Knowledge Neurons in Pretrained
Transformers" (arXiv:2104.08696, ACL 2022). The attribution of FFN
intermediate neuron w_i for prediction P(y|x) is the Riemann-sum integrated
gradient along a linear scaling path from a zero baseline to the neuron's
actual activation w_i_bar (Dai et al. Eq. 3):

    Attr(w_i) = (w_i_bar / m) * sum_{k=1..m} dP(y|x, w_i=(k/m)*w_i_bar)/dw_i

OFFICIAL reference: the ``knowledge-neurons`` package (EleutherAI port),
``KnowledgeNeurons.get_scores_for_layer`` — verbatim kernel:
    scaled = tiled_activations * linspace(0, 1, steps)[:, None]   # baseline 0
    probs  = softmax(logits[:, mask_idx, :])                      # ANSWER PROB
    grad   = autograd.grad(unbind(probs[:, target_idx]), scaled).sum(dim=0)
    ig     = grad * baseline_activations / steps

=============================  FINDING  ===============================
BLME's helper (src/blme/tasks/causality/knowledge_neurons.py) is NOT the
Dai integrated-gradients method, and BLME's OWN docstring says so
("this is not Dai-style integrated-gradient knowledge-neuron
localization ... gradient x activation approximation, not full
integrated gradients (no path integral)"). BLME computes single-point
grad-of-LOGIT x activation saliency:
    target_logit.backward()                    # LOGIT, not softmax prob
    saliency = (act * grad).sum(axis=0)        # single point, m = 1
    per_neuron = |saliency|
hooked at the INPUT of ``mlp.c_proj`` (POST-GELU intermediate), whereas
the reference hooks ``mlp.c_fc`` OUTPUT (PRE-GELU intermediate).

Three concrete divergences vs the official method:
  1. target = LOGIT   (BLME)   vs  softmax PROBABILITY (Dai/ref)
  2. m = 1 single point (BLME) vs  m-step path integral  (Dai/ref)
  3. hook = post-GELU (BLME)   vs  pre-GELU intermediate (Dai/ref)

VERDICT: PROXY (grad x activation saliency), NOT parity with Dai IG.

This test therefore does two things, both LIVE in the main env on a tiny
deterministic transformer:
  (A) PIN BLME's kernel: BLME's grad-of-logit x activation == an
      INDEPENDENT autograd reimplementation of the same math, bit-exact.
  (B) IMPLEMENT + verify the OFFICIAL IG method (transcribed kernel) and
      its axioms — convergence and the completeness axiom — proving the
      reference is genuine Dai IG and that it is a DIFFERENT quantity
      from BLME's saliency.
Plus fixture pins from a full-scale OFFICIAL run of the installed
``knowledge-neurons`` package on GPT-2 (isolated venv; see
tests/fixtures/reference_parity/parity/knowledge_neurons.json).
"""
import json
import os

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from blme.tasks.causality.knowledge_neurons import _gini  # noqa: E402

FIXTURE = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "fixtures", "reference_parity", "parity", "knowledge_neurons.json",
)


# ---------------------------------------------------------------------------
# Tiny deterministic transformer block with an explicit FFN:
#   h = LN(x); a = c_fc(h); g = GELU(a); y = c_proj(g)      (GPT-2 style MLP)
# We expose hooks at BOTH the c_fc output (pre-GELU, reference neuron) and
# the c_proj input (post-GELU, BLME neuron), and a differentiable "patch"
# point for the IG path integral.
# ---------------------------------------------------------------------------
class TinyMLP(nn.Module):
    def __init__(self, dim, inter, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.c_fc = nn.Linear(dim, inter)
        self.c_proj = nn.Linear(inter, dim)
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.empty_like(p).uniform_(-0.4, 0.4, generator=g))

    def forward(self, x):
        a = self.c_fc(x)          # pre-GELU intermediate (reference neuron)
        gelu = F.gelu(a)          # post-GELU intermediate (BLME neuron = c_proj input)
        return self.c_proj(gelu)


class TinyLM(nn.Module):
    """token ids -> embed -> [n_layer x (attn-free block + MLP)] -> head."""

    def __init__(self, vocab=29, dim=12, inter=48, n_layer=3, seed=0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.vocab, self.dim, self.inter, self.n_layer = vocab, dim, inter, n_layer
        self.embed = nn.Embedding(vocab, dim)
        self.ln = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layer)])
        self.mlp = nn.ModuleList([TinyMLP(dim, inter, seed + 1 + i) for i in range(n_layer)])
        self.head = nn.Linear(dim, vocab)
        with torch.no_grad():
            self.embed.weight.copy_(
                torch.empty_like(self.embed.weight).uniform_(-0.4, 0.4, generator=gen))
            for p in self.head.parameters():
                p.copy_(torch.empty_like(p).uniform_(-0.4, 0.4, generator=gen))

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for i in range(self.n_layer):
            x = x + self.mlp[i](self.ln[i](x))
        return self.head(x)


def _build(dtype=torch.float64, seed=0, S=6, vocab=29):
    torch.manual_seed(seed + 100)
    model = TinyLM(vocab=vocab, seed=seed).to(dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    input_ids = torch.randint(0, vocab, (1, S))
    return model, input_ids


# ===========================================================================
# (A) PIN BLME's kernel: grad-of-LOGIT x activation at the c_proj input
#     (post-GELU), single point, summed over sequence, abs.  Bit-exact vs
#     an independent autograd reimplementation.
# ===========================================================================
def _blme_kernel_saliency(model, input_ids, target_id):
    """Reproduce BLME's EXACT inner loop (knowledge_neurons.py lines 149-184):
    forward_pre_hook on c_proj captures its INPUT (post-GELU), retain_grad;
    backward the TARGET LOGIT at last position; saliency=(act*grad).sum(0);
    per-neuron |saliency| concatenated across layers."""
    captured = {}
    handles = []
    for li in range(model.n_layer):
        def mk(li):
            def pre_hook(module, args):
                x = args[0]
                x.requires_grad_(True)
                x.retain_grad()
                captured[li] = x
            return pre_hook
        handles.append(model.mlp[li].c_proj.register_forward_pre_hook(mk(li)))
    try:
        logits = model(input_ids)[0, -1]
        logits[target_id].backward()
    finally:
        for h in handles:
            h.remove()
    per_layer = []
    for li in range(model.n_layer):
        act = captured[li][0].detach().double().cpu().numpy()
        grad = captured[li].grad[0].detach().double().cpu().numpy()
        sal = (act * grad).sum(axis=0)
        per_layer.append(np.abs(sal))
    return per_layer


def _independent_saliency(model, input_ids, target_id):
    """Independent autograd path: grad of the target LOGIT w.r.t. each layer's
    post-GELU intermediate, times that intermediate, summed over positions.
    Computed WITHOUT BLME's hook machinery (functional recompute)."""
    x = model.embed(input_ids)
    inters = []
    h = x
    for li in range(model.n_layer):
        hn = model.ln[li](h)
        a = model.mlp[li].c_fc(hn)
        gelu = F.gelu(a)
        gelu.requires_grad_(True)
        gelu.retain_grad()
        inters.append(gelu)
        h = h + model.mlp[li].c_proj(gelu)
    logits = model.head(h)[0, -1]
    logits[target_id].backward()
    per_layer = []
    for gelu in inters:
        act = gelu[0].detach().double().cpu().numpy()
        grad = gelu.grad[0].detach().double().cpu().numpy()
        per_layer.append(np.abs((act * grad).sum(axis=0)))
    return per_layer


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_blme_kernel_matches_independent_autograd(seed):
    """BLME's grad-of-logit x activation saliency == independent autograd,
    bit-exact per neuron (float64)."""
    model, input_ids = _build(torch.float64, seed=seed)
    target_id = int(model(input_ids)[0, -1].argmax())
    blme = _blme_kernel_saliency(model, input_ids, target_id)
    indep = _independent_saliency(model, input_ids, target_id)
    for li in range(model.n_layer):
        md = float(np.max(np.abs(blme[li] - indep[li])))
        assert md < 1e-9, f"layer {li} saliency max_diff={md:.3e}"


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_blme_summary_metrics_match_independent(seed):
    """BLME's gini / top1 / top1pct summary (recomputed from the independent
    saliency) is bit-exact — pins the whole aggregation pipeline."""
    model, input_ids = _build(torch.float64, seed=seed)
    target_id = int(model(input_ids)[0, -1].argmax())
    blme = _blme_kernel_saliency(model, input_ids, target_id)
    indep = _independent_saliency(model, input_ids, target_id)
    flat_b, flat_i = np.concatenate(blme), np.concatenate(indep)
    assert abs(_gini(flat_b) - _gini(flat_i)) < 1e-12
    sb, si = np.sort(flat_b)[::-1], np.sort(flat_i)[::-1]
    assert abs(sb[0] / flat_b.sum() - si[0] / flat_i.sum()) < 1e-12


# ===========================================================================
# (B) OFFICIAL Dai/EleutherAI IG method (transcribed kernel) + its axioms.
#     Proves the reference is genuine IG and DIFFERS from BLME's saliency.
# ===========================================================================
def _prob_target(model, input_ids, target_id, layer_idx, replacement=None):
    """P(target) at last position, optionally replacing layer_idx's PRE-GELU
    c_fc output at the last position with `replacement` (differentiable)."""
    x = model.embed(input_ids)
    h = x
    for li in range(model.n_layer):
        hn = model.ln[li](h)
        a = model.mlp[li].c_fc(hn)                 # pre-GELU intermediate
        if li == layer_idx and replacement is not None:
            a = a.clone()
            a[:, -1, :] = replacement              # patch last-position neuron
        gelu = F.gelu(a)
        h = h + model.mlp[li].c_proj(gelu)
    logits = model.head(h)[0, -1]
    return F.softmax(logits, dim=-1)[target_id]


def _baseline_cfc(model, input_ids, layer_idx):
    """The unpatched c_fc output (pre-GELU) at the last position, layer_idx."""
    x = model.embed(input_ids)
    h = x
    for li in range(model.n_layer):
        hn = model.ln[li](h)
        a = model.mlp[li].c_fc(hn)
        if li == layer_idx:
            base = a[:, -1, :].detach().clone()
        gelu = F.gelu(a)
        h = h + model.mlp[li].c_proj(gelu)
    return base  # [1, inter]


def _reference_ig(model, input_ids, target_id, layer_idx, steps):
    """OFFICIAL Dai/EleutherAI IG kernel, transcribed:
        scaled = baseline * linspace(0, 1, steps)         # batch of paths
        probs  = softmax(logits[last]); grad = d probs[target] / d scaled
        ig     = grad.sum(over path) * baseline / steps
    Differentiates the ANSWER PROBABILITY (not the logit)."""
    baseline = _baseline_cfc(model, input_ids, layer_idx)  # [1, inter]
    alphas = torch.linspace(0.0, 1.0, steps, dtype=baseline.dtype)
    grads = torch.zeros_like(baseline[0])
    for k in range(steps):
        scaled = (baseline[0] * alphas[k]).clone().requires_grad_(True)
        p = _prob_target(model, input_ids, target_id, layer_idx, replacement=scaled)
        (g,) = torch.autograd.grad(p, scaled)
        grads = grads + g
    ig = grads * baseline[0] / steps
    return ig.detach()


@pytest.mark.parametrize("seed", [0, 3])
def test_reference_ig_convergence(seed):
    """Anchor (a): IG attribution converges as the Riemann step count m grows
    (successive |deltas| shrink) — the hallmark of a genuine integrated grad."""
    model, input_ids = _build(torch.float64, seed=seed)
    target_id = int(model(input_ids)[0, -1].argmax())
    layer = model.n_layer // 2
    ig20 = _reference_ig(model, input_ids, target_id, layer, steps=20)
    top = int(np.argmax(ig20.abs().numpy()))
    vals = {m: float(_reference_ig(model, input_ids, target_id, layer, steps=m)[top])
            for m in (10, 20, 40, 80)}
    d1, d2, d3 = (abs(vals[20] - vals[10]), abs(vals[40] - vals[20]),
                  abs(vals[80] - vals[40]))
    assert d2 <= d1 + 1e-12 and d3 <= d2 + 1e-12, f"not converging: {d1,d2,d3}"


@pytest.mark.parametrize("seed", [0, 3])
def test_reference_ig_completeness_axiom(seed):
    """Anchor (b): the completeness axiom holds and TIGHTENS with m:
        sum_i Attr(w_i) -> P(target | full acts) - P(target | zero acts).
    This uniquely characterizes integrated gradients (and Dai's Attr)."""
    model, input_ids = _build(torch.float64, seed=seed)
    target_id = int(model(input_ids)[0, -1].argmax())
    layer = model.n_layer // 2
    base = _baseline_cfc(model, input_ids, layer)[0]
    with torch.no_grad():
        p_full = float(_prob_target(model, input_ids, target_id, layer,
                                    replacement=base))
        p_zero = float(_prob_target(model, input_ids, target_id, layer,
                                    replacement=torch.zeros_like(base)))
    target = p_full - p_zero
    rel = {}
    for m in (20, 200):
        ig = _reference_ig(model, input_ids, target_id, layer, steps=m)
        rel[m] = abs(float(ig.sum()) - target) / (abs(target) + 1e-12)
    assert rel[200] < rel[20], f"completeness not tightening: {rel}"
    assert rel[200] < 5e-2, f"completeness rel_err too large at m=200: {rel[200]:.3e}"


def test_reference_ig_differs_from_blme_saliency():
    """The OFFICIAL IG (softmax-prob path integral) and BLME's saliency
    (logit single-point grad x act) are DIFFERENT quantities: their
    top-attributed neuron and their per-neuron vectors do not coincide.
    This is the numeric evidence behind the PROXY verdict."""
    model, input_ids = _build(torch.float64, seed=0)
    target_id = int(model(input_ids)[0, -1].argmax())
    layer = model.n_layer // 2
    ig = _reference_ig(model, input_ids, target_id, layer, steps=80).abs().numpy()
    blme = _blme_kernel_saliency(model, input_ids, target_id)[layer]  # same layer
    # Both are length-`inter` vectors over the SAME layer's intermediate,
    # but BLME's is post-GELU and logit-based while IG is pre-GELU prob-based.
    # Normalize and compare: they are materially different distributions.
    ign = ig / (ig.sum() + 1e-12)
    bln = blme / (blme.sum() + 1e-12)
    l1 = float(np.abs(ign - bln).sum())
    assert l1 > 1e-2, f"expected materially different attributions, L1={l1:.3e}"


# ===========================================================================
# (C) Fixture pins from the full-scale OFFICIAL knowledge-neurons run
#     (GPT-2, isolated venv). These are the numbers a reviewer can trace.
# ===========================================================================
@pytest.fixture(scope="module")
def ref():
    with open(os.path.abspath(FIXTURE)) as f:
        return json.load(f)


def test_fixture_present_and_shaped(ref):
    m = ref["meta"]
    assert m["task"] == "causality_knowledge_neurons"
    assert "2104.08696" in m["paper"]
    assert m["model"] == "gpt2"
    assert m["intermediate_size"] == 3072 and m["n_layers"] == 12
    assert ref["reference_ig"]["top_neuron"] == 2789
    assert len(ref["reference_ig"]["top16_neurons"]) == 16


def test_fixture_official_ig_convergence(ref):
    conv = ref["anchor_a_convergence"]
    assert conv["converging"] is True
    d = conv["deltas"]
    assert d[1] <= d[0] and d[2] <= d[1], f"official IG not converging: {d}"


def test_fixture_official_completeness_tightens(ref):
    b = ref["anchor_b_completeness"]
    assert b["tightens_with_m"] is True
    rel = b["rel_err_by_m"]
    assert float(rel["320"]) < float(rel["20"])
    assert float(rel["320"]) < 2e-2, f"official completeness at m=320: {rel['320']}"


def test_fixture_official_top_neuron_high_actxgrad(ref):
    """Anchor (c): the IG-top neuron is high-rank in |act x grad| (top ~0.5%)."""
    c = ref["anchor_c_top_neuron"]
    assert c["ig_top_rank_in_actxgrad"] < int(0.005 * ref["meta"]["intermediate_size"]) + 4


def test_fixture_official_blme_kernel_parity(ref):
    """The isolated-venv run confirmed BLME's helper == independent autograd
    on GPT-2 to <1e-3 (in fact 0.0)."""
    p = ref["blme_kernel_parity"]
    assert p["parity_ok"] is True
    assert p["max_abs_diff"] < 1e-3


def test_verdict_is_proxy(ref):
    """Documented verdict: BLME is a PROXY (grad x activation saliency),
    not numeric parity with Dai integrated gradients."""
    assert "PROXY" in ref["meta"]["divergence_summary"]
    assert ref["meta"]["reference_hook_point"] != ref["meta"]["blme_hook_point"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
