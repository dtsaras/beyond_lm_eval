#!/bin/bash
# End-to-end pipeline: results JSONs → aggregated CSV → correlations → figures + LaTeX tables
#
# The tables/figures are written directly into paper/tables/ and paper/figures/
# so that paper/main.tex can \input{tables/...} and \includegraphics{figures/...}
# without any path gymnastics. They are also copied into the input dir for
# archival purposes.
#
# Usage:
#     bash scripts/build_paper_artifacts.sh [input_dir]
# Default input_dir: results/study_v1

set -euo pipefail

INPUT_DIR="${1:-results/study_v2}"
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

PAPER_TABLES="paper/tables"
PAPER_FIGURES="paper/figures"

echo "═════════════════════════════════════════════════════════"
echo "Building paper artifacts from $INPUT_DIR"
echo "═════════════════════════════════════════════════════════"

if [ ! -d "$INPUT_DIR/blme" ]; then
    echo "ERROR: $INPUT_DIR/blme does not exist. Run the experiments first."
    exit 1
fi

mkdir -p "$PAPER_TABLES" "$PAPER_FIGURES"

echo ""
echo "─── Step 1: Aggregate BLME + lm_eval results ───"
python scripts/aggregate_results.py --input-dir "$INPUT_DIR"

echo ""
echo "─── Step 2: Correlation analysis ───"
python scripts/analyze_correlations.py --input-dir "$INPUT_DIR"

echo ""
echo "─── Step 3: Bootstrap confidence intervals ───"
python scripts/bootstrap_lasso_r2.py --input-dir "$INPUT_DIR" \
    --n-bootstrap 200 --seed 42

echo ""
echo "─── Step 4: Human-readable findings report ───"
python scripts/analyze_findings.py --input-dir "$INPUT_DIR"

echo ""
echo "─── Step 5: Generate figures (→ paper/figures) ───"
python scripts/make_figures.py --input-dir "$INPUT_DIR" --output-dir "$PAPER_FIGURES"

echo ""
echo "─── Step 6: Generate LaTeX tables (→ paper/tables) ───"
python scripts/make_tables.py --input-dir "$INPUT_DIR" --output-dir "$PAPER_TABLES"

echo ""
echo "═════════════════════════════════════════════════════════"
echo "Done. Paper artifacts are in:"
echo "  $INPUT_DIR/aggregated.csv                 (features x models matrix)"
echo "  $INPUT_DIR/analysis/univariate.csv        (Spearman + FDR)"
echo "  $INPUT_DIR/analysis/partial.csv           (controlling for log N)"
echo "  $INPUT_DIR/analysis/lasso_features.csv    (LASSO selection)"
echo "  $INPUT_DIR/analysis/base_vs_instruct.csv  (paired shifts)"
echo "  $INPUT_DIR/analysis/bootstrap_ci.json     (OOB 95 % CIs)"
echo "  $INPUT_DIR/analysis/findings_report.md    (Q1–Q8 narrative)"
echo "  $PAPER_FIGURES/*.pdf                      (NeurIPS figures)"
echo "  $PAPER_TABLES/*.tex                       (LaTeX fragments)"
echo ""
echo "Compile the paper:"
echo "  cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main"
echo "═════════════════════════════════════════════════════════"
