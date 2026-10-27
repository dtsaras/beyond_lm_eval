"""Whole-CLASS regression guards for defects fixed in this campaign.

Every guard here exists because the same defect was found at one site, fixed,
and then found again at a sibling — repeatedly. Counting the population is a
habit that failed four times on 2026-09-08 alone; these are the same rule
expressed as assertions, which is the only form that has actually held.

Each guard names its population explicitly. When a legitimate new site appears,
add it to the allowlist WITH a reason — that forces the question "is this site
different?" to be answered in writing rather than assumed.
"""
from __future__ import annotations

import ast
import pathlib
import re

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "blme"


def _py_files():
    return sorted(SRC.rglob("*.py"))


def test_no_token_pooling_in_the_model_dtype():
    """`h.mean(dim=0)` over tokens must accumulate in double.

    Found at 13 sites across 3 sweeps on 2026-09-08. The first sweep matched
    `mean(dim=0).float().cpu().numpy()` and found 6; the second, order-
    insensitive pass found 3 reordered variants; a semantic sweep then found 6
    more with an entirely different shape (`torch.stack(...).mean(dim=0)`).
    float32's ULP at 1e6 is 0.0625, which turned an exactly-represented
    side-0.25 square into mean H0 0.0625.
    """
    # (file, needle) -> why this site is CORRECT in the model dtype
    ALLOW = {
        ("task_vectors.py", "mean_pos = A_pos.mean(dim=0)"):
            "repe_task_vectors is bit-pinned to the released ClusterMeanRepReader "
            "at max abs error 0.0; promoting here moved it to 1.907e-06 and was "
            "reverted. Only the reported NORM is taken in double.",
        ("task_vectors.py", "mean_neg = A_neg.mean(dim=0)"):
            "see mean_pos above — same reference-parity constraint.",
    }
    bad = []
    for f in _py_files():
        for i, line in enumerate(f.read_text().splitlines(), 1):
            s = line.strip()
            # An explicit float64 promotion in either spelling is compliant. The
            # guard used to accept only torch's `.double()`, so it rejected the
            # correct numpy form and ACCEPTED `ch.double().mean(0)` on a numpy
            # array, which raised AttributeError on every call (phd, 2026-10-26).
            # A text match rewards a spelling, not a behaviour; the AST rewrite
            # is queued (plan, Phase 3).
            if s.startswith("#") or "double()" in s or "float64" in s:
                continue
            if any(f.name == fn and needle in s for (fn, needle) in ALLOW):
                continue
            # pooling shapes: `<x>.mean(dim=0)` / `.mean(0)` feeding a rep list
            if re.search(r"\.(mean\(dim=0\)|mean\(0\))", s) and re.search(
                    r"append|rep\s*=|reps|vecs|collected|mean_pos|mean_neg|"
                    r"sentence_rep|pooled|layer_reps", s):
                bad.append(f"{f.relative_to(SRC.parent.parent)}:{i}  {s[:76]}")
    assert not bad, (
        "token-mean pooling in the model dtype (must be `.double().mean(...)`; "
        "the accumulator overflows or quantises):\n  " + "\n  ".join(bad))


def test_no_absolute_epsilon_floors_on_a_magnitude():
    """A scale-free quantity may not be gated by an absolute constant.

    Four instances on 2026-09-07: `stability`'s clamp_min(1e-10) broke the
    noise_std=0 self-identity axiom; `coe`'s `z_mag <= 1e-12` nulled an entire
    metric family at 1e-30; `trajectories` twice; `trajectory_curvature`'s
    _ZERO_NORM_EPS discarded a genuine step. The correct floor is either 0, a
    relative fraction of the data's own scale, or `finfo.tiny`.
    """
    ALLOW = {
        # (file, needle) -> reason this constant is legitimately absolute.
        # Each was checked against the data the site actually receives; a
        # population is a list of places to LOOK, not to change.
        ("prediction_entropy.py", "1e-10"):
            "log guard on probabilities, dimensionless and in [0,1]",
        ("trajectories.py", "1e-10"):
            "log guard on softmax probabilities, not a magnitude",
        ("stability.py", "raw.norm"):
            "`raw` is a fresh torch.randn, O(1) by construction — not data",
        ("refusal.py", "direction.norm(dim=-1, keepdim=True) + 1e-8"):
            "the released refusal_direction ablation hook's own normalisation "
            "(hook_utils.py), kept for parity with select_direction",
        ("steering.py", "vector_norm(_rv)"):
            "`_rv` is a fresh torch.randn control vector, O(1) by construction",
        ("cknna.py", "clamp_min(1e-12)"):
            "float32-first `_l2_normalize`: unsafe rows are detected per-row "
            "with finfo-derived bounds and routed to safe_unit_rows_torch, so "
            "this clamp only ever runs where the norm is provably healthy",
        ("information_geometry.py", "clamp_min(1e-9)"):
            "denominator of a RELATIVE error diagnostic, not a data magnitude",
        ("information_geometry.py", "clamp_min(1e-30)"):
            "denominator of the linearity ratio; guards 0/0 on a zero probe",
    }
    pat = re.compile(
        r"(clamp_min\(\s*1e-\d+\s*\)|norm\([^)]*\)\s*\+\s*1e-\d+|"
        r"(?:_mag|scale|norm|magnitude)\w*\s*<=?\s*1e-\d+)", re.I)
    bad = []
    for f in _py_files():
        for i, line in enumerate(f.read_text().splitlines(), 1):
            s = line.strip()
            if s.startswith("#") or "finfo" in s or "eps" in s.lower():
                continue
            m = pat.search(s)
            if not m:
                continue
            if any(f.name == fn and needle in s for (fn, needle) in ALLOW):
                continue
            bad.append(f"{f.relative_to(SRC.parent.parent)}:{i}  {s[:76]}")
    assert not bad, (
        "absolute epsilon floor on a magnitude in a scale-free operation "
        "(use 0, a relative fraction, or finfo.tiny):\n  " + "\n  ".join(bad))


def test_linalg_handlers_catch_what_safe_svdvals_actually_raises():
    """`safe_svdvals` raises `torch._C._LinAlgError`, a RuntimeError.

    `np.linalg.LinAlgError` derives from ValueError, so a handler naming only
    the numpy type cannot catch it and its fallback is dead code. Found in
    `isotropy` by both round-11 reviewers, after `safe_svdvals` was wired into
    28+ sites without anyone re-checking the except clauses.
    """
    bad = []
    for f in _py_files():
        src = f.read_text()
        if "safe_svdvals" not in src:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            guards_svd = "safe_svdvals" in ast.dump(node)
            if not guards_svd:
                continue
            for h in node.handlers:
                if h.type is None:
                    continue
                names = ast.dump(h.type)
                if "LinAlgError" in names and "_LinAlgError" not in names \
                        and "RuntimeError" not in names and "Exception" not in names:
                    bad.append(
                        f"{f.relative_to(SRC.parent.parent)}:{h.lineno}  "
                        f"catches only the numpy LinAlgError")
    assert not bad, (
        "handler cannot catch what safe_svdvals raises:\n  " + "\n  ".join(bad))
