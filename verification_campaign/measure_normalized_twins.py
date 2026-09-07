"""Measure whether a candidate normalized twin actually removes the size term.

The schatten case showed a divisor can INTRODUCE the dependence it claims to
remove, so every candidate is scored against the raw column on the same models.
"""
import json, glob, math, os, numpy as np

# Run from ~/projects/beyond_benchmarks (the study repo holds results/).
os.chdir(os.path.expanduser("~/projects/beyond_benchmarks"))
from scipy.stats import spearmanr

rows = []
for fp in sorted(glob.glob("results/blme_verified_a/*.json")):
    d = json.load(open(fp)); r = d.get("results", {})
    ss = r.get("geometry_superposition_strength") or {}
    dm = ss.get("hidden_dim")
    nl = None
    for t in ("geometry_cka", "geometry_vendi_score", "geometry_weight_norms",
              "geometry_procrustes_linearity", "geometry_cknna"):
        v = (r.get(t) or {}).get("_meta_n_layers")
        if isinstance(v, (int, float)) and v > 0: nl = float(v); break
    if isinstance(dm, (int, float)) and dm > 0 and nl:
        rows.append((float(dm), nl, r))
print(f"models with width AND depth: {len(rows)}")

DIV = {
    "d_model":       lambda d, L: d,
    "log_d_model":   lambda d, L: math.log(d),
    "sqrt_d":        lambda d, L: math.sqrt(d),
    "n_layers":      lambda d, L: L,
    "sqrt_n_layers": lambda d, L: math.sqrt(L),
    "x_sqrt_L":      lambda d, L: 1.0 / math.sqrt(L),   # multiply by sqrt(L)
}

def col(task, key):
    out = []
    for dm, nl, r in rows:
        v = (r.get(task) or {}).get(key)
        if isinstance(v, (int, float)) and np.isfinite(v):
            out.append((float(v), dm, nl))
    return out

def report(task, key):
    c = col(task, key)
    if len(c) < 12:
        return None
    v = np.array([x[0] for x in c]); dm = np.array([x[1] for x in c]); nl = np.array([x[2] for x in c])
    raw = (spearmanr(v, dm).statistic, spearmanr(v, nl).statistic)
    best = None
    for name, f in DIV.items():
        t = v / np.array([f(a, b) for a, b in zip(dm, nl)])
        s = (spearmanr(t, dm).statistic, spearmanr(t, nl).statistic)
        score = max(abs(s[0]), abs(s[1]))
        if best is None or score < best[1]:
            best = (name, score, s)
    return len(c), raw, best

TARGETS = [
    ("geometry_vendi_score", "vendi_mean_last_layer"),
    ("geometry_vendi_score", "vendi_min"),
    ("geometry_hubness", "hubness_k10_gini"),
    ("geometry_rsa", "rsa_adjacent_mean"),
    ("geometry_intrinsic_dim", "intrinsic_dimension"),
    ("geometry_lid", "lid_mean"),
    ("geometry_matrix_entropy", "matrix_entropy"),
    ("geometry_svd", "effective_rank"),
]
print(f"\n{'task.key':52s} {'n':>3} {'raw rho(d/L)':>16} -> {'best divisor':>14} {'twin rho(d/L)':>16}")
for t, k in TARGETS:
    res = report(t, k)
    if res is None:
        print(f"{t+'.'+k:52s}  -- insufficient archive coverage"); continue
    n, raw, (name, score, s) = res
    print(f"{t+'.'+k:52s} {n:>3} {raw[0]:+.3f}/{raw[1]:+.3f}      {name:>14} {s[0]:+.3f}/{s[1]:+.3f}"
          + ("   <-- improves" if score < max(abs(raw[0]), abs(raw[1])) - 0.15 else ""))
