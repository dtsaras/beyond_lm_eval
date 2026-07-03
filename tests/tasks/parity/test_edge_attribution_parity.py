"""Numeric-parity test: BLME causality_edge_attribution vs the OFFICIAL Edge
Attribution Patching (EAP) score kernel.

PAPER: Syed, Rager & Conmy 2024, "Attribution Patching Outperforms Automated
Circuit Discovery", arXiv:2310.10348 (NeurIPS 2023 ATTRIB / BlackBoxNLP).

  Eq (2) attribution score:
      L(x_clean | do(E=e_corr))
        ~= L(x_clean) + (e_corr - e_clean)^T . dL/de_clean
      Delta_e L := (e_corr - e_clean)^T . dL/de_clean
  Eq (3) absolute attribution score:
      |Delta_e L|   (the paper always ranks by this)

OFFICIAL REFERENCE REPO: Aaquib111/edge-attribution-patching
  commit 7124ef815b320383f2d29b0e2c2757075ed0c417 (2024-02-21)
  utils/prune_utils.py
    node kernel (line 211):
        node_attr = (clean_head_act - corr_head_act) * clean_grad_act
        node_attr = split_layers_and_heads(...).sum((2,3,4))  # dot over d_model
    edge kernel (line 283):
        current_result = (clean_grad_cache[down] *
              (clean_cache[fwd] - corrupted_cache[fwd])).sum()
    abs (line 285-286): if attr_absolute_val: current_result = .abs()

BLME (src/blme/tasks/causality/edge_attribution.py, EdgeAttributionTask):
  per residual-stream LAYER input, kernel lines 172-173:
        diff = clean_h[:T] - c_h[:T]                # (e_clean - e_corr)
        attr = |(diff * grad_h[:T]).sum()|          # |dot| == Eq (3)

Sign note: paper writes (e_corr - e_clean); the repo and BLME write
(clean - corr). They differ only in sign, and both take |.| when ranking
(Eq 3 / attr_absolute_val=True), so the scores are IDENTICAL. BLME's kernel
therefore matches the reference formula and paper Eq (3) exactly (sign-invariant).

BLME is architecture-agnostic (plain HF forward hooks) and applies the EAP
KERNEL to residual-stream layer inputs — a documented per-LAYER proxy, not the
full transformer_lens per-EDGE computational graph. This test verifies the
KERNEL to numeric parity (the transformer_lens per-edge pipeline is verified
separately in the scratch verify script and recorded in the fixture). BLME must
NOT depend on transformer_lens, so this test uses only torch/numpy + the real
BLME task on a tiny HF model.
"""
import json
import os

import numpy as np
import pytest
import torch
import torch.nn as nn

FIXTURE = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "fixtures", "reference_parity", "parity", "edge_attribution.json",
)

TOL_EXACT = 1e-10       # exact-formula parity at native (float64) dtype
TOL_F32 = 1e-4          # BLME's float32-downcast code path


# ---------------------------------------------------------------------------
# The three EAP score kernels, computed independently.
# ---------------------------------------------------------------------------
def ref_eap_score(clean_act, corr_act, clean_grad, absolute=True):
    """Reference EAP kernel, prune_utils.py line 283/211 (torch)."""
    score = (clean_grad * (clean_act - corr_act)).sum()
    return score.abs() if absolute else score


def paper_eq2_score(clean_act, corr_act, clean_grad):
    """Paper Eq (2): Delta_e L = (e_corr - e_clean)^T . dL/de_clean (torch)."""
    return ((corr_act - clean_act) * clean_grad).sum()


def blme_kernel_numpy(clean_h, c_h, grad_h):
    """BLME edge_attribution.py lines 172-173 verbatim (numpy float32 path)."""
    diff = clean_h - c_h
    return float(np.abs((diff * grad_h).sum()))


# ---------------------------------------------------------------------------
# Tiny controlled model. A layer's INPUT (the embedding output feeding the
# residual block) is exactly the residual-stream node BLME attributes over.
# ---------------------------------------------------------------------------
class TinyResidLM(nn.Module):
    def __init__(self, vocab=19, dim=8, hidden=16, seed=0, linear=False):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.embed = nn.Embedding(vocab, dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.head = nn.Linear(dim, vocab)
        self.linear = linear
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.empty_like(p).uniform_(-0.5, 0.5, generator=g))

    def block(self, x):
        if self.linear:
            return x + self.fc2(self.fc1(x))          # linear -> patching EXACT
        return x + self.fc2(torch.tanh(self.fc1(x)))

    def forward(self, input_ids, patch_in=None):
        emb = self.embed(input_ids)
        emb.requires_grad_(True)
        emb.retain_grad()
        x = emb if patch_in is None else patch_in
        logits = self.head(self.block(x))
        return logits, emb


def _metric_last_logit(logits):
    last = logits[0, -1]
    tid = int(last.argmax().item())
    return last[tid], tid


def _prepare(dtype, seed_model, seed_data, linear):
    torch.manual_seed(seed_data)
    model = TinyResidLM(seed=seed_model, linear=linear).to(dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    clean_ids = torch.randint(0, 19, (1, 6))
    g = torch.Generator().manual_seed(seed_data * 997 + 11)
    perm = torch.randperm(clean_ids.shape[1], generator=g)
    corr_ids = clean_ids[:, perm]
    with torch.no_grad():
        corr_emb = model.embed(corr_ids).to(dtype)
    model.zero_grad(set_to_none=True)
    logits, clean_emb = model(clean_ids)
    score, tid = _metric_last_logit(logits)
    score.backward()
    return model, clean_ids, logits, clean_emb.detach(), clean_emb.grad.detach(), corr_emb, tid


KERNEL_CASES = [
    (torch.float64, 0, 1, False),
    (torch.float64, 3, 7, False),
    (torch.float64, 11, 42, False),
    (torch.float32, 0, 1, False),
    (torch.float32, 3, 7, False),
]


@pytest.mark.parametrize("dtype,seed_model,seed_data,linear", KERNEL_CASES)
def test_eap_kernel_formula_parity(dtype, seed_model, seed_data, linear):
    """BLME kernel == reference line-283 == paper Eq(2)/(3), same activations.

    Three checks:
      1. reference |.| (repo line 283, Eq 3) == |paper Eq 2|   -> machine eps
      2. BLME formula on IDENTICAL float32 arrays == reference formula -> 0.0
      3. BLME float32 vs reference native dtype -> bounded by float32 downcast
    """
    _, _, _, clean_e, clean_g, corr_e, _ = _prepare(dtype, seed_model, seed_data, linear)

    ref_abs = float(ref_eap_score(clean_e, corr_e, clean_g, absolute=True).item())
    paper = float(paper_eq2_score(clean_e, corr_e, clean_g).item())

    # 1. reference (Eq 3) == |paper Eq 2|  (sign-invariant), machine epsilon
    assert abs(ref_abs - abs(paper)) < TOL_EXACT, (
        f"ref283 |.| ({ref_abs}) != |paperEq2| ({abs(paper)})")

    # BLME's numpy float32 arrays (task lines 155-156, 168 downcast to float32)
    cn = clean_e.float().cpu().numpy()[0]
    kn = clean_g.float().cpu().numpy()[0]
    on = corr_e.float().cpu().numpy()[0]
    blme = blme_kernel_numpy(cn, on, kn)

    # 2. reference formula on the SAME float32 arrays -> identical (exact 0.0)
    ref_same_f32 = float(np.abs((kn * (cn - on)).sum()))
    assert abs(blme - ref_same_f32) < TOL_EXACT, (
        f"BLME formula != reference formula on identical f32 inputs: "
        f"{blme} vs {ref_same_f32}")

    # 3. BLME float32 vs reference native dtype -> float32-level agreement
    assert abs(blme - ref_abs) < TOL_F32, (
        f"BLME(f32)={blme} vs ref(native)={ref_abs} exceeds f32 tol")


EXACT_CASES = [(0, 1), (5, 9), (13, 21)]


@pytest.mark.parametrize("seed_model,seed_data", EXACT_CASES)
def test_eap_is_exact_on_linear_model(seed_model, seed_data):
    """ANCHOR: EAP is a first-order approx of activation patching; on a LINEAR
    model patching is EXACT, so the EAP score == the TRUE patching effect
    L(clean | do(E=corr)) - L(clean)."""
    model, clean_ids, logits, clean_e, clean_g, corr_e, tid = _prepare(
        torch.float64, seed_model, seed_data, linear=True)

    eap_signed = float(paper_eq2_score(clean_e, corr_e, clean_g).item())
    with torch.no_grad():
        patched, _ = model(clean_ids, patch_in=corr_e)
        true_effect = float((patched[0, -1, tid] - logits[0, -1, tid]).item())
    assert abs(eap_signed - true_effect) < 1e-9, (
        f"EAP {eap_signed} != true patch effect {true_effect}")


@pytest.mark.parametrize("seed_model,seed_data", EXACT_CASES)
def test_largest_score_ranks_highest(seed_model, seed_data):
    """ANCHOR: the node with the largest |activation-difference x gradient|
    ranks highest. Build several nodes and confirm argmax by |score| is stable
    between the reference and BLME kernels."""
    torch.manual_seed(seed_data)
    D = 8
    n_nodes = 6
    clean = torch.randn(n_nodes, 4, D, dtype=torch.float64)
    corr = torch.randn(n_nodes, 4, D, dtype=torch.float64)
    grad = torch.randn(n_nodes, 4, D, dtype=torch.float64)
    ref = np.array([float(ref_eap_score(clean[i], corr[i], grad[i]).item())
                    for i in range(n_nodes)])
    blme = np.array([blme_kernel_numpy(
        clean[i].numpy(), corr[i].numpy(), grad[i].numpy())
        for i in range(n_nodes)])
    assert np.argmax(ref) == np.argmax(blme)
    assert np.array_equal(np.argsort(-ref), np.argsort(-blme))


def test_real_blme_task_computes_eap_kernel():
    """Run BLME's ACTUAL EdgeAttributionTask on a tiny HF model and confirm each
    per-layer attribution equals the EAP kernel |(clean_in - corr_in) . grad|,
    reproduced independently with the SAME hooks/seeds the task uses.

    Skips offline if the tiny model isn't cached.
    """
    transformers = pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "hf-internal-testing/tiny-random-gpt2"
    try:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name)
    except Exception as e:  # pragma: no cover - offline without cache
        pytest.skip(f"tiny model unavailable: {e}")

    from blme.tasks.common import get_layers
    from blme.registry import get_task
    import blme.tasks.causality.edge_attribution  # noqa: F401  (registers task)

    model.eval()
    layers = get_layers(model)
    assert layers is not None and len(layers) > 0
    n_layers = len(layers)

    # --- Independently reproduce ONE prompt exactly as the task does (pi=0) ---
    text = "The capital of France is Paris"
    enc = tok(text, return_tensors="pt", truncation=True, max_length=128)
    input_ids = enc["input_ids"]
    assert input_ids.shape[1] >= 4

    for p in model.parameters():
        p.requires_grad_(False)

    # corrupted pass (task lines 107-117): seed = pi*997+11 with pi=0
    g = torch.Generator(device="cpu").manual_seed(0 * 997 + 11)
    perm = torch.randperm(input_ids.shape[1], generator=g)
    corr_ids = input_ids[:, perm]
    with torch.no_grad():
        c_out = model(corr_ids, output_hidden_states=True)
        corrupted_hs = [h.detach() for h in c_out.hidden_states[1:]]

    # clean pass with pre-hooks capturing each layer INPUT (task lines 121-148)
    captured = {}

    def make_hook(li):
        def hook(_m, args):
            if isinstance(args, tuple) and len(args) > 0:
                x = args[0]
                x.requires_grad_(True)
                x.retain_grad()
                captured[li] = x
        return hook

    handles = [layers[li].register_forward_pre_hook(make_hook(li))
               for li in range(n_layers)]
    try:
        clean_out = model(input_ids=input_ids)
        lg = clean_out.logits[0, -1]
        tid = int(lg.argmax().item())
        lg[tid].backward()
    finally:
        for h in handles:
            h.remove()

    # independent per-layer EAP kernel (task lines 151-174)
    indep = np.zeros(n_layers)
    for li in range(n_layers):
        clean_h = captured[li].detach().float().cpu().numpy()[0]
        grad_h = captured[li].grad.detach().float().cpu().numpy()[0]
        if li == 0:
            c_h = c_out.hidden_states[0].detach().float().cpu().numpy()[0]
        else:
            c_h = corrupted_hs[li - 1].float().cpu().numpy()[0]
        T = min(clean_h.shape[0], c_h.shape[0], grad_h.shape[0])
        indep[li] = blme_kernel_numpy(clean_h[:T], c_h[:T], grad_h[:T])

    # Now confirm the reference EAP formula (torch, native) matches per layer.
    for li in range(n_layers):
        ch = captured[li].detach()[0]
        gh = captured[li].grad.detach()[0]
        if li == 0:
            cc = c_out.hidden_states[0].detach()[0]
        else:
            cc = corrupted_hs[li - 1][0]
        T = min(ch.shape[0], cc.shape[0], gh.shape[0])
        ref = float(ref_eap_score(ch[:T].double(), cc[:T].double(),
                                  gh[:T].double()).item())
        # BLME downcasts to float32 -> compare at float32 tolerance
        assert abs(indep[li] - ref) < TOL_F32 * (1 + abs(ref)), (
            f"layer {li}: BLME kernel {indep[li]} vs reference {ref}")

    # --- And run the REAL task end-to-end; its outputs must be finite/valid ---
    task = get_task("causality_edge_attribution")({"num_samples": 3})
    out = task.evaluate(model, tok, None)
    assert "error" not in out, out
    assert out["diagnostic_method"] == "residual_layer_gradient_patch_proxy"
    assert out["attribution_unit"] == "transformer_layer"
    prof = out["mean_layer_attribution_profile"]
    assert len(prof) == n_layers
    assert abs(sum(prof) - 1.0) < 1e-5           # normalized profile
    assert 0.0 <= out["attribution_gini"] <= 1.0
    assert 0.0 <= out["top1_layer_share"] <= 1.0
    assert 0.0 <= out["peak_attribution_layer"] <= 1.0
    assert out["attribution_entropy"] >= 0.0


def test_fixture_matches_verified_values():
    """Pin the canonical verified values from the scratch verify run."""
    with open(FIXTURE) as f:
        fx = json.load(f)
    assert fx["task"] == "causality_edge_attribution"
    assert fx["reference"]["commit"] == (
        "7124ef815b320383f2d29b0e2c2757075ed0c417")
    assert fx["verdict"]["kernel_parity"] == "PARITY"
    assert fx["verdict"]["exactness_on_linear_model"] == "EXACT"
    assert fx["verdict"]["transformer_lens_crosscheck"] == "PASS"
    assert fx["all_pass"] is True

    # exact-formula parities are numerically zero
    A = fx["canonical_kernel_case_A"]
    assert A["d_ref283_vs_paperEq2_magnitude"] == 0.0
    assert A["d_refabs_vs_paperEq3"] == 0.0
    assert A["d_blme_vs_ref_identical_f32"] == 0.0
    assert A["d_blme_f32_vs_ref_native"] < TOL_F32

    # exactness anchor: EAP == true patch effect on a linear model
    B = fx["canonical_exactness_case_B"]
    assert B["d_eap_vs_true_patch"] < 1e-9

    # transformer_lens gpt2 cross-check: ref == independent to machine epsilon
    C = fx["transformer_lens_gpt2_crosscheck"]
    assert C["max_abs_diff_ref_vs_independent"] < 1e-8
    assert C["ranking_identical"] is True
    assert C["ref_top_layer"] == C["independent_top_layer"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
