#!/bin/bash
#SBATCH --job-name=edm2-1024
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --time=4-0:0
#SBATCH --nodes=1
#SBATCH --partition=rocky

# Self-contained: resolve the repo root (dir with pyproject.toml) and cd in so the
# relative --data / --outdir paths resolve consistently.
SCRIPT_DIR="$(cd "$(dirname "${SLURM_JOB_SCRIPT:-${BASH_SOURCE[0]}}")" && pwd)"
REPO_DIR="${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}"
while [[ ! -f "$REPO_DIR/pyproject.toml" && "$REPO_DIR" != / ]]; do
    REPO_DIR="$(dirname "$REPO_DIR")"
done
cd "$REPO_DIR" || exit 1

conda activate edm2

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_VISIBLE_DEVICES=0,1

nvidia-smi

# 1024^2 -> 128x128 latents, but the inline VAE still ingests full 1024^2 RGB:
# keep --batch-gpu small; EDM2 reaches the global batch via gradient
# accumulation automatically. --gpus spawns the workers internally (no torchrun).
edm2-train \
    --outdir=./runs/edm2-img1024-s \
    --cfg=edm2-img1024-s \
    --data=./datasets/imagenet_9to4_1024x1024_1024x1024.zip \
    --gpus=2 \
    --batch-gpu=8 \
    --combra-metrics True \
    --save-inference-only True \
    --snap 100 \
    --sampler=dpm++ \
    --sampling-steps=25
