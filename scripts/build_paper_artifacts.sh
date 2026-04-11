#!/bin/bash
# End-to-end pipeline: results JSONs → aggregated CSV → correlations → figures + LaTeX tables
#
# Usage:
#     bash scripts/build_paper_artifacts.sh [input_dir]
# Default input_dir: results/study_v1

set -euo pipefail

INPUT_DIR="${1:-results/study_v1}"
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

echo "═════════════════════════════════════════════════════════"
echo "Building paper artifacts from $INPUT_DIR"
echo "═════════════════════════════════════════════════════════"

if [ ! -d "$INPUT_DIR/blme" ]; then
    echo "ERROR: $INPUT_DIR/blme does not exist. Run the experiments first."
    exit 1
fi

echo ""
echo "─── Step 1: Aggregate BLME + lm_eval results ───"
python scripts/aggregate_results.py --input-dir "$INPUT_DIR"

echo ""
echo "─── Step 2: Correlation analysis ───"
python scripts/analyze_correlations.py --input-dir "$INPUT_DIR"

echo ""
echo "─── Step 3: Generate figures ───"
python scripts/make_figures.py --input-dir "$INPUT_DIR"

echo ""
echo "─── Step 4: Generate LaTeX tables ───"
python scripts/make_tables.py --input-dir "$INPUT_DIR"

echo ""
echo "═════════════════════════════════════════════════════════"
echo "Done. Paper artifacts are in:"
echo "  $INPUT_DIR/aggregated.csv        (features x models matrix)"
echo "  $INPUT_DIR/analysis/             (correlation outputs)"
echo "  $INPUT_DIR/figures/*.pdf         (7 NeurIPS figures)"
echo "  $INPUT_DIR/tables/*.tex          (6 LaTeX table fragments)"
echo "═════════════════════════════════════════════════════════"
