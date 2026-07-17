#!/bin/bash
set -euo pipefail

# Self-locate the repo root (dir with pyproject.toml) so relative --data / --outdir
# paths resolve the same way under sbatch and on a workstation.
SCRIPT_DIR="$(cd "$(dirname "${SLURM_JOB_SCRIPT:-${BASH_SOURCE[0]}}")" && pwd)"
REPO_DIR="${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}"
while [[ ! -f "$REPO_DIR/pyproject.toml" && "$REPO_DIR" != / ]]; do
    REPO_DIR="$(dirname "$REPO_DIR")"
done
cd "$REPO_DIR"

conda activate edm2
export TORCH_CUDA_ARCH_LIST="9.0"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 1024x1024 training. Submit e.g. sbatch --account=<proj> --partition=rocky --gpus=2 sh/train_1024.sh
edm2-train \
    --outdir=./runs/edm2-img1024-s \
    --cfg=edm2-img1024-s \
    --data=./datasets/wc_co_1024x1024.zip \
    --gpus=2 \
    --batch-gpu=16 \
    --tick=128 --snap=64 \
    --eval-sampler=dpm++ --eval-sampling-steps=25
