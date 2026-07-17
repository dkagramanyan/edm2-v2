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

# 1024x1024 per-class generation into the wc_cv angle-pipeline HDF5 layout.
# --gpus spawns per-GPU workers internally (no torchrun). Point --network at the
# newest inference snapshot in the run directory.
edm2-gen-images \
    --network=./runs/edm2-img1024-s/edm2-snapshot-latest-inference.pt \
    --outdir=./generated/1024 \
    --classes=0,1,2 --samples-per-class=1000 \
    --gpus=2 --batch-gpu=32 \
    --save-mode=hdf5 \
    --sampler=dpm++ --steps=25
