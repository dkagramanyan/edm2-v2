#!/bin/bash
#SBATCH --job-name=edm2-cmp-256
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
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
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_VISIBLE_DEVICES=0
nvidia-smi

MODEL=${MODEL:-./training-runs/00000-edm2-img256-s/network-snapshot-final.pkl}

# Find the optimal number of sampling steps per sampler with combra
# (metric-vs-steps curve + plateau). Needs `pip install -e '.[combra]'`.
python compare_samplers.py \
    --net="$MODEL" \
    --data=./datasets/imagenet_256x256.zip \
    --num-samples=512 \
    --batch=32 \
    --samplers=edm,euler,ddim,dpm++ \
    --k-values=5,10,20,50,100,250 \
    --outdir=./sampler-comparison/256
