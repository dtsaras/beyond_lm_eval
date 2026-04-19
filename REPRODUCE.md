# Reproducing the BLME Study

End-to-end reproduction guide for the 32-model × 731-feature × 68-benchmark
study behind the paper. The commands below reproduce the exact numbers in
`docs/TOP_PREDICTORS.md`, `results/study_v2/analysis/findings_report.md`,
and `results/study_v2/analysis/bootstrap_ci.json`.

## 0. Hardware & software

| Thing | Minimum | Used for reference numbers |
|---|---|---|
| GPUs | 1× 24 GB (A10G/RTX 3090) | 8× A10G (24 GB each) on `eez130.ece.ust.hk` |
| Python | 3.11 | 3.11.x via miniconda |
| Torch | 2.3+ | 2.5.x (CUDA 12.4) |
| Transformers | 5.x | see `pyproject.toml` |
| scikit-learn | **1.8.0** | `pip install 'scikit-learn==1.8.0'` |
| numpy | **1.26.x** | `pip install 'numpy<2.0'` |

**Version pinning matters**: sklearn 1.7.2 vs 1.8.0 yields LassoCV point
estimates that differ by up to 0.04 in held-out R² on this problem (see
the commit log for 2026-04-20). Pin to 1.8.0 to reproduce
`LOO R² = 0.772` and `LOFO R² = 0.266`.

## 1. Environment setup

```bash
conda create -n blme python=3.11 -y
conda activate blme

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[all]"
pip install 'scikit-learn==1.8.0' 'numpy<2.0'

# lm-eval for the benchmark side
pip install 'lm-eval[api]'
```

`scripts/setup_server.sh` does this plus Qwen-3.5 flash-linear-attention
dependencies.

## 2. Run the 32-model evaluation

The 32-model zoo is in `scripts/model_zoo.py`. Each model's HuggingFace
ID, dtype, GPU count, and attention implementation are fixed there.

```bash
python scripts/run_study.py \
    --output-dir results/study_v2 \
    --task-group all

# On a single 24 GB GPU this takes ~6 hours per small model
# and up to 24 h for qwen3.5-27b-it / gemma4-31b with
# device_map=auto across 4 GPUs.
```

Pythia-6.9b and Pythia-12b must be loaded in **fp32** for
dynamics/perplexity/entropy tasks (fp16 overflow produces NaN
logits). The zoo already has `dtype=float32` for Pythia-6.9b and
Pythia-12b in the fp32-sensitive tasks — if you get NaN outputs,
check that `dtype` in the HF kwargs is fp32 for these two models and
that `attn_implementation="eager"` is set for `dynamics_sharpness`
(needed for double backward).

## 3. Run the 68-benchmark lm-eval side

```bash
python scripts/run_comprehensive_benchmarks.py \
    --output-dir results/study_v2/lm_eval \
    --models all
```

Produces per-(model, benchmark) `results_*.json` files under
`results/study_v2/lm_eval/<model>/<hfid>/`. The 68 benchmarks are
tracked by the run script; swap in / out tasks with `--tasks`.

## 4. Aggregate intrinsic × benchmark matrix

```bash
python scripts/aggregate_results.py --input-dir results/study_v2
```

Produces:
- `results/study_v2/aggregated.csv` — 32 × 812 (intrinsic +
  benchmark + metadata)
- `results/study_v2/feature_metadata.csv`
- `results/study_v2/metadata.csv`

Summary line expected:
```
Wrote aggregated features: results/study_v2/aggregated.csv (32 models x 812 columns)
  Models: 32 | Features: 787 | Benchmarks: 67 | Composite benchmark coverage: 32/32
```

## 5. Correlation analyses

```bash
python scripts/analyze_correlations.py --input-dir results/study_v2
```

Expected headline output (verbatim, on scikit-learn 1.8.0):

```
LASSO on composite_benchmark:
  Held-out LOO R² = 0.772  ← honest within-family generalization
  Held-out LOFO R² = 0.266  ← cross-family generalization (strict)
  Baseline log_n_params: train R² = 0.498, LOO R² = 0.429
  Selected 26 features out of 731
```

Produces:
- `results/study_v2/analysis/univariate.csv` — 49,708 Spearman tests,
  FDR-corrected.
- `results/study_v2/analysis/partial.csv` — same, controlling for
  `log(N_params)`.
- `results/study_v2/analysis/lasso_features.csv` — 26 selected
  features with coefficients.
- `results/study_v2/analysis/pca_coords.csv`
- `results/study_v2/analysis/base_vs_instruct.csv`

## 6. Bootstrap confidence intervals

```bash
python scripts/bootstrap_lasso_r2.py \
    --input-dir results/study_v2 \
    --n-bootstrap 200 \
    --seed 42
```

Expected output (sklearn 1.8.0):

```
── Point estimates ──
  lasso_loo             +0.772
  lasso_lofo            +0.266
  baseline_loo          +0.429
  gain                  +0.343

── 95 % OOB-bootstrap CIs ──
  lasso_r2              CI=[+0.07, +0.90]  median=+0.66
  baseline_r2           CI=[-0.51, +0.71]  median=+0.36
  gain                  CI=[-0.30, +1.03]  median=+0.30
```

Writes `results/study_v2/analysis/bootstrap_ci.json`.

Runtime: ~3 minutes for B=200 on a 24-core CPU.

## 7. Dedupe top predictors (paper-ready tables)

```bash
python /tmp/dedupe_top_predictors.py results/study_v2
```

The dedupe script is at `/tmp/dedupe_top_predictors.py` in the repo
root after you copy it out of the commit history (see the
`3eb4347` commit message). It strips the aggregator-summary suffix
(`.mean`, `.std`, `.min`, `.max`, `.q25/q50/q75`, `.slope`) to
report one row per feature family.

## 8. Human-readable findings report

```bash
python scripts/analyze_findings.py --input-dir results/study_v2
```

Produces `results/study_v2/analysis/findings_report.md` with the
Q1–Q8 narrative.

## 9. Paper tables + figures

```bash
bash scripts/build_paper_artifacts.sh
```

Runs `make_tables.py` + `make_figures.py` + `make_insight_plots.py`;
writes LaTeX tables to `paper/tables/` and PDFs to `paper/figures/`.

## Where the numbers live

- **Headline LOO / LOFO / gain**: `results/study_v2/analysis/lasso_features.csv`
  + `scripts/analyze_correlations.py` stdout.
- **Top-25 univariate / partial tables**: `docs/TOP_PREDICTORS.md` §1–2.
- **Paper-ready verbatim claim**: `docs/TOP_PREDICTORS.md` §4.
- **Bootstrap CIs**: `results/study_v2/analysis/bootstrap_ci.json`.
- **Per-benchmark findings**: `results/study_v2/analysis/findings_report.md`.
- **Base-vs-instruct shifts**: `results/study_v2/analysis/base_vs_instruct.csv`.

## Troubleshooting

- **NaN in forward pass**: load the model in fp32 (not fp16). Applies
  to pythia-6.9b, pythia-12b, and any model on exotic attention.
- **CUDA OOM on sharpness**: use `device_map=auto` with ≥ 4 GPUs and
  `attn_implementation=eager`.
- **Sklearn number drift**: check `sklearn.__version__`; if not
  1.8.0, numbers will differ by up to ±0.04.
- **Missing per-model results.json**: run `scripts/patch_failed_tasks.py`
  to selectively re-run only the failed tasks.
