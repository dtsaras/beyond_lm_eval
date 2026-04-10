#!/bin/bash
# Server setup script for BLME benchmark study.
# Run on eez130.ece.ust.hk after cloning the repo.
#
# Usage: bash scripts/setup_server.sh

set -euo pipefail

echo "=== BLME Server Setup ==="
echo "Host: $(hostname)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"

# ── 1. Create conda environment ─────────────────────────────────────
echo ""
echo "=== Creating conda environment 'blme' ==="
conda create -n blme python=3.11 -y
eval "$(conda shell.bash hook)"
conda activate blme

# ── 2. Install PyTorch (CUDA 12.x) ──────────────────────────────────
echo ""
echo "=== Installing PyTorch ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# ── 3. Install BLME and dependencies ────────────────────────────────
echo ""
echo "=== Installing BLME ==="
pip install -e ".[all]"

# ── 4. Install lm_eval ──────────────────────────────────────────────
echo ""
echo "=== Installing lm-eval ==="
pip install lm-eval[api]

# ── 5. Install Qwen 3.5 dependencies (hybrid attention) ─────────────
echo ""
echo "=== Installing flash-linear-attention for Qwen 3.5 ==="
pip install flash-linear-attention causal-conv1d 2>/dev/null || echo "Warning: flash-linear-attention install may need manual setup"

# ── 6. Verify GPU access ────────────────────────────────────────────
echo ""
echo "=== GPU Verification ==="
python -c "
import torch
n = torch.cuda.device_count()
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {n}')
for i in range(n):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name} ({props.total_mem / 1024**3:.1f} GB)')
"

# ── 7. Verify BLME installation ─────────────────────────────────────
echo ""
echo "=== BLME Verification ==="
python -c "
from blme.core import _register_all_tasks
from blme.registry import list_tasks
_register_all_tasks()
tasks = list_tasks()
print(f'BLME tasks registered: {len(tasks)}')
"

# ── 8. Verify model zoo ─────────────────────────────────────────────
echo ""
echo "=== Model Zoo ==="
python scripts/model_zoo.py

echo ""
echo "=== Setup Complete ==="
echo "To start the study:"
echo "  conda activate blme"
echo "  python scripts/run_study.py --dry-run --output-dir results/study_v1"
echo "  python scripts/run_study.py --phase all --output-dir results/study_v1"
