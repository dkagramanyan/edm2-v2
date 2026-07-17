#!/bin/bash
# 256x256 training launch. Runs unmodified on a workstation (`bash sh/train_256.sh`)
# or under SLURM, with cluster specifics supplied at submission time:
#   sbatch --account=<proj> --partition=rocky --gpus=2 --cpus-per-task=16 \
#       --time=3-0:0 --nodes=1 sh/train_256.sh
set -euo pipefail

# Self-locate the repo root (dir with pyproject.toml) so relative --data / --outdir
# paths resolve the same way under sbatch and on a workstation.
SCRIPT_DIR="$(cd "$(dirname "${SLURM_JOB_SCRIPT:-${BASH_SOURCE[0]}}")" && pwd)"
REPO_DIR="${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}"
while [[ ! -f "$REPO_DIR/pyproject.toml" && "$REPO_DIR" != / ]]; do
    REPO_DIR="$(dirname "$REPO_DIR")"
done
cd "$REPO_DIR"

# Environment (env name = repo name). EDM2 is pure PyTorch (no custom CUDA-op
# compilation), so no system CUDA module is required; --gpus spawns the workers
# internally (no torchrun).
conda activate edm2
export TORCH_CUDA_ARCH_LIST="9.0"

# Offline-cluster contract: backbones (SD-VAE, and combra's InceptionV3 / CLIP /
# DINOv2) are prefetched once on a login node via `edm2-download-models`; force HF
# offline so combra precompute reads the cache instead of hanging on the network.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

edm2-train \
    --outdir=./runs/edm2-img256-s \
    --cfg=edm2-img256-s \
    --data=./datasets/wc_co_256x256.zip \
    --gpus=2 \
    --batch-gpu=64 \
    --tick=128 --snap=64 \
    --eval-sampler=dpm++ --eval-sampling-steps=25
