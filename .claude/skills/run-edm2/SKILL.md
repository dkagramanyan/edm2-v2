---
name: run-edm2
description: Build, run, and drive the edm2 diffusion codebase — prepare data, train, generate images (HDF5), sample, and run tests. Use when asked to run, launch, train, smoke-test, generate samples from, or screenshot edm2. Covers the offline GPU path and the edm2-train / edm2-gen-images / edm2-prepare-data console CLIs.
---

# Run edm2

edm2 is a **CLI-driven** class-conditional latent-diffusion codebase (an EDM2
refresh). There is no GUI/server — you drive it through console entry points
(`edm2-train`, `edm2-gen-images`, `edm2-prepare-data`, `edm2-eval`,
`edm2-compare-samplers`) installed by `pip install -e .`. The primary agent path
is the smoke driver below, which runs the whole pipeline on **one GPU, fully
offline** and checks every artifact.

**Paths below are relative to the `edm2/` repo root.** The driver lives at
`.claude/skills/run-edm2/smoke.sh`.

## Run (agent path) — the driver

```bash
bash .claude/skills/run-edm2/smoke.sh
```

~95 s on an RTX 3090. It synthesizes a tiny 2-class dataset (no private data
needed), then runs and asserts each stage: **prepare-data → train `--dry-run` →
train 1 kimg (RGB `edm2-img64-s`) → gen-images → HDF5 → `pytest`**. Prints
`ALL SMOKE STEPS PASSED` and leaves artifacts under a `mktemp` dir it reports.

Useful overrides:

```bash
# drive real WC-Co images instead of synthetic (folder of class subdirs)
bash .claude/skills/run-edm2/smoke.sh --source ../datasets/original/o

# pin the env / workdir
EDM2_ENV=edm2 WORKDIR=/tmp/edm2-smoke bash .claude/skills/run-edm2/smoke.sh
```

The visual output to eyeball is in the run dir the driver prints:
`reals.png` (real image grid) and `fakes000001.png` (model samples — pure noise
after only 1 kimg, which is correct).

## Prerequisites

- Linux, one NVIDIA GPU (verified on RTX 3090, driver 610, CUDA 13). CPU-only
  runs the tests but not training/generation.
- A conda env named **`edm2`** with PyTorch + edm2's deps (see Build). The
  driver auto-finds `"$(conda info --base)"/envs/edm2/bin/python`; override with
  `EDM2_PY=/path/to/python`.
- No OS packages beyond the conda env were needed.

## Build

This container has **no network**, so the from-scratch install in `README.md`
(`conda create -n edm2 … ; pip install torch --index-url … ; pip install -e
'.[combra]'`) can't fetch wheels. The working offline path reuses an existing
torch env and installs edm2 in place:

```bash
# clone any env that already has torch+diffusers (here: diffit) into `edm2`
conda create --name edm2 --clone diffit -y

# install edm2's console entry points (deps already satisfied by the clone).
# --no-build-isolation is REQUIRED offline: pip's default isolation tries to
# fetch setuptools from PyPI and fails with no network.
conda activate edm2
pip install -e . --no-deps --no-build-isolation
```

On a networked machine, follow `README.md` instead. `combra` (inline training
metrics) is a **private** repo and is not installed here — training runs without
it via `--combra-metrics=False`.

## Direct invocation (single entry point)

Each stage is a console command; run them directly to reproduce or debug one
step (activate the env first: `conda activate edm2`, and
`export HF_HUB_OFFLINE=1`):

```bash
# 1) build an RGB training zip (grayscale SEM -> RGB, labels from folder order)
edm2-prepare-data convert --source=<img_folder> --dest=data_64.zip \
    --resolution=64x64 --transform=center-crop-dhariwal

# 2) resolve+print the full training config without computing
edm2-train --outdir=runs --cfg=edm2-img64-s --data=data_64.zip --gpus=1 -n

# 3) train (RGB, offline: no combra, no VAE); writes snapshots + fakes/reals grids
edm2-train --outdir=runs --cfg=edm2-img64-s --data=data_64.zip \
    --gpus=1 --batch-gpu=8 --tick=1 --snap=1 --kimg=1 \
    --combra-metrics=False --num-fid-samples=0

# 4) generate per-class images into the wc_cv HDF5 layout
edm2-gen-images --network=runs/00000-*/edm2-snapshot-000001-0.100-inference.pt \
    --outdir=gen --classes=0 --samples-per-class=4 --gpus=1 --batch-gpu=4 \
    --save-mode=hdf5 --sampler=dpm++ --steps=8
```

## Test

```bash
conda activate edm2
python -m pytest tests -q      # 14 CPU tests: sampler contracts + Precond forward
```

## Gotchas

- **Latent presets need network; only RGB runs offline.** `edm2-img256/512/1024-*`
  encode through the Stability VAE (`stabilityai/sd-vae-ft-mse`). The HF cache
  here is **incomplete** (blobs but no `snapshots/` / `config.json`), so offline
  they die with *"does not appear to have a file named config.json"*. Use
  `edm2-img64-*` (StandardRGBEncoder) offline; run `edm2-download-models` **with
  network** once to enable latent training.
- **`--max-images N` yields a single-class dataset.** It takes images in
  class-folder-alphabetical order, so a small cap grabs only the first class. The
  model is then 1-class and `edm2-gen-images --classes=1,2…` errors *"index N out
  of range for a 1-class model"*. Use the driver's synthetic data (2 balanced
  classes) or don't cap.
- **`combra` is optional and absent here.** Training-time metrics and
  `edm2-compare-samplers` need it. Pass `--combra-metrics=False --num-fid-samples=0`
  to train without it. The imports are guarded, so the model math is unaffected.
- **Benign shutdown noise.** `destroy_process_group() was not called … leak
  resources` and `Guessing device ID based on global rank` print at exit of every
  single-GPU run. Harmless — not a failure.
- **Runs are not resumable by design.** Every launch makes a fresh
  `runs/<id>-…` dir; a crash/kill cannot be continued. Size `--kimg` to fit.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `pip install -e .` → `Could not find a version that satisfies setuptools>=61` | Add `--no-build-isolation` (offline; setuptools is already in the env). |
| Train dies at *"Setting up encoder…"* with `config.json` not found | You used a latent preset offline. Switch to `--cfg=edm2-img64-*`, or fetch the VAE online via `edm2-download-models`. |
| `edm2-*: command not found` | `conda activate edm2` (entry points live in that env's `bin/`). |
| gen-images: *"index N out of range for a 1-class model"* | Dataset had one class (see `--max-images` gotcha). Retrain on multi-class data or request only `--classes=0`. |
