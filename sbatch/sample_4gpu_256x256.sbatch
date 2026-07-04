#!/bin/bash
#SBATCH --job-name=edm2-smp-256
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --time=1-0:0
#SBATCH --nodes=1
#SBATCH --partition=rocky

SCRIPT_DIR="$(cd "$(dirname "${SLURM_JOB_SCRIPT:-${BASH_SOURCE[0]}}")" && pwd)"
REPO_DIR="${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}"
while [[ ! -f "$REPO_DIR/pyproject.toml" && "$REPO_DIR" != / ]]; do
    REPO_DIR="$(dirname "$REPO_DIR")"
done
cd "$REPO_DIR" || exit 1

conda activate edm2
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_VISIBLE_DEVICES=0,1,2,3
nvidia-smi

MODEL=${MODEL:-./runs/edm2-img256-s/network-snapshot-final.pkl}

# 50k samples as a single .npz for FID evaluation (edm2-eval / calculate_metrics.py).
torchrun --standalone --nproc_per_node=4 sample_images.py \
    --net="$MODEL" \
    --outdir=./samples/256 \
    --num-samples=50000 \
    --batch=32 \
    --sampler=dpm++ \
    --steps=25
