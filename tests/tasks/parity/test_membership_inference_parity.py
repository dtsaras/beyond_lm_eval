"""Numeric-parity test: BLME consistency_membership_inference
vs. Yeom et al. 2018 (CSF) loss-threshold membership inference.

BLME (src/blme/tasks/consistency/membership_inference.py,
MembershipInferenceTask.evaluate, lines ~178-189) scores each example by the
NEGATED per-example NLL (higher score == more member-like) and reports:
    separability_auroc / mia_auroc = roc_auc_score(labels, -losses)
    loss_gap                       = mean(nonmember_loss) - mean(member_loss)

OFFICIAL reference (sam-yeom/ml-privacy-csf18, Yeom et al. 2018 CSF
"Privacy Risk in Machine Learning", arXiv:1709.01604), transcribed below:
    code/inclusion.py:27-50  sklearn_decide / sklearn_inclusion
        membership adversary predicts TRAIN (member) when the per-example
        statistic crosses a threshold (one-error case: |error| < r_emp).
        Generalised here to: predict member iff score >= threshold, with
        score = -loss, matching BLME's "higher == more member-like" sign.
    code/main.py:23-30,177-195
        train_TRAIN = fraction of true members guessed member   = TPR
        test_TRAIN  = fraction of non-members guessed member     = FPR
        membership advantage = train_TRAIN - test_TRAIN          = TPR - FPR

This test recomputes, INDEPENDENTLY of BLME:
  * Yeom membership advantage = max_threshold (TPR - FPR) for the loss attack,
  * AUROC of the loss score three ways (sklearn, Mann-Whitney rank stat,
    trapezoidal swept-ROC area) -- all must agree,
  * loss_gap,
and drives BLME's REAL evaluate() on the SAME losses through a stub
model/tokenizer whose float64 logits reproduce the prescribed cross-entropy.
All shared metrics must match to < 1e-9.
"""
import math

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Transcribed Yeom reference (inclusion.py / main.py).
# ---------------------------------------------------------------------------
TRAIN = 100  # inclusion.py:3 -- adversary guess "in training set" (member)
TEST = 200   # inclusion.py:4 -- adversary guess "in test set"     (non-member)


def _yeom_decide(scores, threshold):
    """inclusion.sklearn_decide analogue: member iff score >= threshold."""
    return np.where(scores >= threshold, TRAIN, TEST)


def _yeom_tpr_fpr(member_scores, nonmember_scores, threshold):
    tpr = np.count_nonzero(_yeom_decide(member_scores, threshold) == TRAIN) / len(member_scores)
    fpr = np.count_nonzero(_yeom_decide(nonmember_scores, threshold) == TRAIN) / len(nonmember_scores)
    return tpr, fpr


def yeom_membership_advantage(member_losses, nonmember_losses):
    """Yeom advantage = max_threshold (TPR - FPR), score = -loss."""
    ms = -np.asarray(member_losses, float)
    nms = -np.asarray(nonmember_losses, float)
    alls = np.concatenate([ms, nms])
    cands = np.concatenate([np.unique(alls), [alls.max() + 1.0]])
    return float(max(_yeom_tpr_fpr(ms, nms, t)[0] - _yeom_tpr_fpr(ms, nms, t)[1] for t in cands))


def auc_mannwhitney(member_losses, nonmember_losses):
    """AUROC = P(member_loss < nonmember_loss) + 0.5 P(tie); no sklearn."""
    m = np.asarray(member_losses, float)
    nm = np.asarray(nonmember_losses, float)
    concordant = np.sum(m[:, None] < nm[None, :])
    ties = 0.5 * np.sum(m[:, None] == nm[None, :])
    return float((concordant + ties) / (len(m) * len(nm)))


def auc_trapezoid(member_losses, nonmember_losses):
    """AUROC via trapezoidal area under the swept ROC curve, score = -loss."""
    ms = -np.asarray(member_losses, float)
    nms = -np.asarray(nonmember_losses, float)
    alls = np.concatenate([ms, nms])
    cands = np.concatenate([[alls.max() + 1.0], np.unique(alls)[::-1]])
    pts = sorted(_yeom_tpr_fpr(ms, nms, t)[::-1] for t in cands)  # (fpr, tpr)
    pts = np.array(pts)
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trap(pts[:, 1], pts[:, 0]))


# ---------------------------------------------------------------------------
# Synthetic per-example NLL arrays (must match the embedded fixture exactly).
# ---------------------------------------------------------------------------
SEED = 20240624
N_MEMBER = 40
N_NONMEMBER = 45


def make_synthetic_losses():
    rng = np.random.default_rng(SEED)
    m = np.clip(rng.normal(2.0, 0.8, N_MEMBER), 0.05, None).astype(float)
    nm = np.clip(rng.normal(3.2, 0.9, N_NONMEMBER), 0.05, None).astype(float)
    return m, nm


# Embedded OFFICIAL expected values (from the wave1 harness / fixture).
OFFICIAL_ADVANTAGE = 0.5027777777777778
OFFICIAL_AUROC = 0.8027777777777777
OFFICIAL_LOSS_GAP = 1.0204059464541317
TOL = 1e-9


# ---------------------------------------------------------------------------
# Stub model/tokenizer so BLME's REAL evaluate() runs on our prescribed losses.
# _compute_nll computes F.cross_entropy(logits, labels). For a 2-class problem
# with true label 0, CE = log(1+exp(-g)) where g is the logit gap; set
# g = -log(exp(L)-1) to make the mean CE equal the prescribed loss L exactly
# (float64 logits keep it bit-accurate).
# ---------------------------------------------------------------------------
def _run_blme(member_losses, nonmember_losses):
    import torch
    from blme.tasks.consistency.membership_inference import MembershipInferenceTask

    dataset, loss_by_text = [], {}
    for i, l in enumerate(member_losses):
        t = f"MEMBER_TEXT_{i} aa bb cc dd"
        dataset.append({"text": t, "label": 1})
        loss_by_text[t] = float(l)
    for i, l in enumerate(nonmember_losses):
        t = f"NONMEMBER_TEXT_{i} aa bb cc dd"
        dataset.append({"text": t, "label": 0})
        loss_by_text[t] = float(l)

    class _Enc(dict):
        def to(self, device):
            return self

    class StubTokenizer:
        def __call__(self, text, return_tensors=None, truncation=None, max_length=None):
            return _Enc(input_ids=torch.zeros((1, 5), dtype=torch.long), _text=text)

    class _Out:
        def __init__(self, logits):
            self.logits = logits

    class StubModel:
        def __init__(self):
            self._p = [torch.nn.Parameter(torch.zeros(1))]

        def parameters(self):
            return iter(self._p)

        def __call__(self, **enc):
            L = loss_by_text.get(enc["_text"], 1.234567)  # dummy for CF/shuffled texts
            seq = enc["input_ids"].shape[1]
            g = -math.log(math.expm1(L))
            logits = torch.zeros((1, seq, 2), dtype=torch.float64)
            logits[:, :, 0] = g  # true label is class 0 everywhere -> mean CE == L
            return _Out(logits)

    return MembershipInferenceTask(config={}).evaluate(StubModel(), StubTokenizer(), dataset)


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------
def test_reference_auroc_three_ways_agree():
    """sklearn AUROC == Mann-Whitney == trapezoid swept-ROC (all independent)."""
    m, nm = make_synthetic_losses()
    auc_mw = auc_mannwhitney(m, nm)
    auc_trap = auc_trapezoid(m, nm)
    assert abs(auc_mw - auc_trap) < TOL
    assert abs(auc_mw - OFFICIAL_AUROC) < TOL
    try:
        from sklearn.metrics import roc_auc_score
        labels = [1] * len(m) + [0] * len(nm)
        scores = list(-m) + list(-nm)
        assert abs(float(roc_auc_score(labels, scores)) - auc_mw) < TOL
    except ImportError:
        pytest.skip("sklearn not available")


def test_yeom_membership_advantage_matches_fixture():
    """Yeom membership advantage = max TPR - FPR matches the recorded value."""
    m, nm = make_synthetic_losses()
    adv = yeom_membership_advantage(m, nm)
    assert abs(adv - OFFICIAL_ADVANTAGE) < TOL, f"adv={adv}"
    # Advantage and AUROC are mutually consistent for a threshold-on-score family.
    assert 0.0 <= adv <= 2.0 * OFFICIAL_AUROC - 1.0 + 1e-12


def test_blme_auroc_parity():
    """BLME separability_auroc/mia_auroc == independent Mann-Whitney AUROC."""
    m, nm = make_synthetic_losses()
    ref_auc = auc_mannwhitney(m, nm)
    res = _run_blme(m, nm)
    assert res["n_members"] == N_MEMBER
    assert res["n_nonmembers"] == N_NONMEMBER
    assert abs(res["separability_auroc"] - ref_auc) < TOL, res["separability_auroc"]
    assert abs(res["mia_auroc"] - ref_auc) < TOL
    assert abs(res["separability_auroc"] - OFFICIAL_AUROC) < TOL


def test_blme_loss_gap_parity():
    """BLME loss_gap == mean(nonmember) - mean(member) (Yeom-aligned sign)."""
    m, nm = make_synthetic_losses()
    ref_gap = float(np.mean(nm) - np.mean(m))
    res = _run_blme(m, nm)
    assert abs(res["loss_gap"] - ref_gap) < TOL, res["loss_gap"]
    assert abs(res["loss_gap"] - OFFICIAL_LOSS_GAP) < TOL
    # mean losses round-trip exactly through the stub.
    assert abs(res["mean_loss_member"] - float(np.mean(m))) < TOL
    assert abs(res["mean_loss_nonmember"] - float(np.mean(nm))) < TOL


def test_score_direction_lower_loss_is_member():
    """Members have lower loss -> AUROC must be > 0.5 (correct score sign)."""
    m, nm = make_synthetic_losses()
    res = _run_blme(m, nm)
    assert res["loss_gap"] > 0  # model "does better" on members
    assert res["separability_auroc"] > 0.5


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
