#!/usr/bin/env bash
# edm2 -- train at 512x512.
#
# Workstation:  bash sh/train_512.sh    (DATA=<zip> GPUS=<n> ... to override)
# SLURM:        sbatch --account=<proj> --partition=<part> --gpus=2 --cpus-per-task=16 --time=3-0:0 sh/train_512.sh
#
# Every knob is an env var with a default; anything after the script name is appended to
# the command (e.g. `... --kimg 200 --snap 2` for a smoke run). No user homes, --nodelist
# or account IDs live here -- SLURM specifics come from the sbatch line (spec §9).
set -euo pipefail

# --- Environment -------------------------------------------------------------
# Repo root: under SLURM the script runs from a spool copy, so walk up from the submit
# dir there and from this file's own location on a workstation.
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
while [[ ! -f "$REPO_DIR/pyproject.toml" && "$REPO_DIR" != / ]]; do REPO_DIR="$(dirname "$REPO_DIR")"; done
[[ -f "$REPO_DIR/pyproject.toml" ]] || { echo "cannot find the repo root -- submit from inside the repo" >&2; exit 1; }
cd "$REPO_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-edm2-v2}"   # env name = repo name
# Pure PyTorch: no custom CUDA ops, so no toolkit or arch list is needed.
# Offline-cluster contract: backbones are prefetched once on a login node
# (edm2-download-models); compute nodes never reach the network.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# --- One console-command call ------------------------------------------------
edm2-train \
    --outdir "${OUTDIR:-./runs}" \
    --cfg "${CFG:-edm2-img512-s}" \
    --data "${DATA:-./datasets/imagenet_9to4_1024x1024_512x512.zip}" \
    --gpus "${GPUS:-2}" \
    --batch-gpu "${BATCH_GPU:-32}" \
    --cond True --mirror False \
    --tick "${TICK:-128}" --snap "${SNAP:-64}" --snapshot-keep-last "${KEEP_LAST:-3}" \
    --combra-metrics True --num-fid-samples "${NUM_FID_SAMPLES:-10000}" \
    --eval-sampler "${EVAL_SAMPLER:-dpm++}" --eval-sampling-steps "${EVAL_STEPS:-25}" \
    --seed "${SEED:-0}" \
    "$@"
