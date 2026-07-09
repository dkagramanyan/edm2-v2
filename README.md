# EDM2: Analyzing and Improving the Training Dynamics of Diffusion Models (v2 refresh)

Official EDM2 codebase, refreshed to match the **DiffiT-v2** workflow: a
multi-format training logger, inline **combra** generative-quality metrics
computed **across all GPU ranks**, a packaged `click` + console-entry-point API,
class-conditional ImageNet latent-diffusion training at **256 / 512 / 1024**,
and multiple reverse-diffusion samplers (**EDM Heun / Euler / DDIM / DPM-Solver++**).

Based on:

**Analyzing and Improving the Training Dynamics of Diffusion Models** (CVPR 2024 oral)<br>
Tero Karras, Miika Aittala, Jaakko Lehtinen, Janne Hellsten, Timo Aila, Samuli Laine<br>
https://arxiv.org/abs/2312.02696

**Guiding a Diffusion Model with a Bad Version of Itself** (NeurIPS 2024 oral)<br>
Tero Karras, Miika Aittala, Tuomas Kynkäänniemi, Jaakko Lehtinen, Timo Aila, Samuli Laine<br>
https://arxiv.org/abs/2406.02507

> **The training math is unchanged.** The EDM2 loss, learning-rate schedule,
> optimizer step, magnitude-preserving network, preconditioning and Power-Function
> EMA (the "training dynamics and update layers") are preserved exactly. Everything
> new below runs *around* that core — at logging, inference and eval time only.

## Differences from upstream NVlabs/edm2

| Area | Upstream edm2 | This refresh |
|---|---|---|
| **Logging** | single `Status:` line + `stats.jsonl` | DiffiT-style logger: `log.txt`, `progress.csv`, `progress.json`, `stats.jsonl`, TensorBoard (`events.out.tfevents.*`), a `tick … kimg … sec/tick …` console line, plus `reals.png` / `fakes_init.png` / `fakes<kimg>.png` grids. **Every scalar** (losses, LR, timing, resources, combra metrics, tick) **and the image grids are mirrored to TensorBoard** |
| **Latent encoding** | dataset pre-encoded to an 8-channel latent zip offline | **inline VAE encode** (DiffiT-style): train latent diffusion straight from a raw-RGB zip, the frozen Stability VAE runs each step — no pre-encode pass. Offline 8-channel latent zips still work and are auto-detected |
| **Checkpointing** | full resumable `training-state-*.pt` | self-contained inference `network-snapshot-*.pkl` (EMA + encoder) **plus** the resumable `.pt`; `--checkpoint=0` writes **inference-only** (DiffiT `--save-inference-only`) |
| **Metrics** | offline FID / FD-DINOv2 only (`calculate_metrics.py`) | inline **combra** metrics every snapshot tick, **sharded and gathered across all GPU ranks**: `combra_fid10k`, `combra_cmmd10k`, `combra_fd_dinov2_10k` + angle-density metrics |
| **Samplers** | EDM 2nd-order Heun only | `dpm++` (DPM-Solver++ 2M, **default**, 25 steps), `edm` (Heun), `euler`, `ddim`, σ-space, **one implementation shared by training-eval, generation and bulk sampling** |
| **Step-count analysis** | — | `edm2-compare-samplers`: sweep samplers × step counts, score with combra, find the optimal number of sampling steps (metric-vs-steps plateau) |
| **Packaging / API** | `python train_edm2.py …` | `pip install -e '.[combra]'` + console entry points (`edm2-train`, `edm2-gen-images`, `edm2-sample`, `edm2-eval`, `edm2-compare-samplers`, `edm2-prepare-data`, `edm2-download-models`) |
| **Resolutions** | img64, img512 presets | added `edm2-img256-*` and `edm2-img1024-*` presets + sbatch for 256/512/1024 |

## Installation

64-bit **Python 3.10+** (3.12 recommended). Install the latest **PyTorch** from
the CUDA 13.x wheels, then the package:

```bash
conda create -n edm2 python=3.12 -y
conda activate edm2
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132
pip install -e '.[combra]'      # omit [combra] to train without inline combra metrics
```

`combra` (the WC-Co computer-vision metrics library) is optional; the import is
guarded, so training runs unchanged without it. Its base dependencies cover every
image metric — FID (pytorch-fid), CMMD (open-clip-torch), FD-DINOv2 (torch.hub
DINOv2). It lives in a **private** repo, so the `[combra]` extra clones it over
`git+https` and only succeeds when you are authenticated to GitHub — sign in once
with the GitHub CLI and `pip` inherits its credential helper:

```bash
gh auth login        # github.com → HTTPS
pip install -e '.[combra]'
```

Pre-fetch the VAE and metric backbones for offline nodes:

```bash
edm2-download-models
```

## Class conditioning — how the model is made conditional

EDM2 is a **class-conditional** diffusion model. Conditioning is expressed with a
few composable methods, all available here:

- **One-hot class labels.** The network is built with `label_dim = <num classes>`
  (inferred from the dataset). The label embedding is combined with the noise
  embedding inside the U-Net (`label_balance`), so the denoiser
  `net(x, sigma, class_labels)` is conditioned on the class. Train conditional with
  `--cond=True` (default); a dataset with no labels + `--cond=False` gives an
  unconditional model (`label_dim = 0`).
- **Null (unconditional) label.** Passing `class_labels=None` (or an all-zero
  one-hot) evaluates the model unconditionally — this is what the guiding network
  uses, and what omitting `--class` combined with a null network produces.
- **Classifier-free / auto-guidance.** At sampling time you can steer generation
  with a second *guiding* network via `--gnet` and `--guidance` (strength `> 1`):
  the denoiser output is extrapolated away from the guiding network's,
  `D = lerp(D_guide, D_main, guidance)`. Use the model's own weaker/earlier
  checkpoint (autoguidance) or an unconditional model as `--gnet`. `--guidance 1`
  (default) disables guidance. The same `--guidance` is honored during training-time
  combra eval and by every sampler (`edm/euler/ddim/dpm++`).
- **Specific class.** `edm2-gen-images --class=<idx>` (and `edm2-sample --class=<idx>`)
  fixes the class label for all generated images; without it, a class is sampled
  at random per image.

## Data preparation (ImageNet)

Convert an ImageNet-like dataset to a training zip in the VAE latent space (labels
are stored in `dataset.json`), using ADM/Dhariwal center-cropping:

```bash
edm2-prepare-data --source=/data/imagenet --dest=datasets/imagenet_512x512.zip \
    --resolution=512x512 --transform=center-crop-dhariwal
```

Produce one zip per target resolution (`256x256`, `512x512`, `1024x1024`). The
model resolution follows the dataset's latent resolution (32² / 64² / 128²).

## Training

Pick a preset with `--preset`; any CLI option overrides the preset. Launch with
`torchrun` for multi-GPU (the global batch is reached via gradient accumulation
automatically):

```bash
# 256²
torchrun --standalone --nproc_per_node=2 train_edm2.py \
    --outdir=training-runs --preset=edm2-img256-s \
    --data=datasets/imagenet_256x256.zip --batch-gpu=96

# 512²
torchrun --standalone --nproc_per_node=2 train_edm2.py \
    --outdir=training-runs --preset=edm2-img512-s \
    --data=datasets/imagenet_512x512.zip --batch-gpu=32

# 1024²
torchrun --standalone --nproc_per_node=2 train_edm2.py \
    --outdir=training-runs --preset=edm2-img1024-s \
    --data=datasets/imagenet_1024x1024.zip --batch-gpu=8
```

To **resume**, run the exact same command again — training auto-resumes from the
latest `training-state-*.pt` in `--outdir`.

Ready-made SLURM scripts (H200) live in `sbatch/`:

```bash
sbatch sbatch/train_2h200_256x256.sbatch
sbatch sbatch/train_2h200_512x512.sbatch
sbatch sbatch/train_2h200_1024x1024.sbatch
```

### Key training options

| Option | Default | Description |
|--------|---------|-------------|
| `--outdir` | required | Output directory for the run |
| `--data` | required | Dataset zip/dir (`edm2-prepare-data` output) |
| `--preset` | `edm2-img512-s` | Config preset (`edm2-img256/512/1024-*`, `edm2-img64-*`) |
| `--cond` | `True` | Train class-conditional model |
| `--batch-gpu` | auto | Per-GPU batch size (global batch reached via accumulation) |
| `--fp16` | `True` | Mixed-precision training |
| `--status` / `--snapshot` / `--checkpoint` | 128Ki / 8Mi / 128Mi | Status / snapshot / checkpoint intervals (images) |
| `--combra-metrics / --no-combra-metrics` | on | Inline combra metrics each snapshot tick (all ranks) |
| `--num-fid-samples` | 10000 | Fakes generated (all ranks) per combra eval; 0 disables |
| `--combra-ref-count` | 0 (whole set) | Real reference images for combra |
| `--eval-sampler` | `dpm++` | Eval-time / snapshot sampler (`edm/euler/ddim/dpm++`) |
| `--eval-sampling-steps` | 25 | Eval-time sampling steps |
| `--guidance` | 1 | Eval-time classifier-free guidance strength |
| `-n, --dry-run` | off | Print resolved config and exit |

### Hydra entry point

`train_hydra.py` is a thin wrapper around the same launch path (DiffiT-v2 / san-v2
style). The click CLI in `train_edm2.py` stays the single source of truth for every
option and default: the Hydra path introspects it, overlays `configs/config.yaml`
plus any command-line overrides, and calls the same
`train_edm2.launch_from_opts(opts)` the click entry point uses — so both paths
produce identical run configs.

```bash
python train_hydra.py outdir=./training-runs preset=edm2-img256-s \
    data=./datasets/imagenet_256x256.zip gpus=2 batch_gpu=64
```

Override any option by its **Python name** (dashes become underscores; `--cfg` /
`--preset` is named `preset`):

```bash
python train_hydra.py outdir=./training-runs preset=edm2-img256-s data=... \
    gpus=2 batch_gpu=64 combra_metrics=false save_inference_only=true \
    eval_sampler=edm eval_sampling_steps=32 snap=100
```

Every option is listed in `configs/config.yaml` so plain `key=value` overrides work
without Hydra's `+` prefix. A `null` there means "not provided" and leaves the click
default in place, so the YAML never duplicates — or drifts from — those defaults.

### Training output

```
training-runs/00000-edm2-img256-s-gpus2-batch2048/
├── training_options.json   # all hyperparameters
├── log.txt                 # human-readable log (rank 0)
├── log-rank001.txt …       # per-rank logs
├── progress.csv            # CSV metrics
├── progress.json           # JSON-lines metrics
├── stats.jsonl             # SAN-v2-style stats + combra metrics
├── events.out.tfevents.*   # TensorBoard
├── reals.png               # real image grid
├── fakes_init.png          # pre-training samples
├── fakes000200.png …       # samples per snapshot tick
├── network-snapshot-*.pkl  # EMA weights (inference artifact)
└── training-state-*.pt     # full resumable state
```

Each run gets its own directory under `--outdir`, named
`<id:05d>-<preset>-gpus<N>-batch<B>` (DiffiT-style). **Re-running the same command
reuses the matching directory** rather than numbering a new one — that is how edm2
resumes, since it picks up the latest `training-state-*.pt` from the run directory.
Changing the preset, GPU count or batch size yields a different name and therefore a
fresh run.

Monitor with `tensorboard --logdir training-runs`.

## Quality metrics (combra, all ranks)

With `--combra-metrics` on (default), every snapshot tick generates
`--num-fid-samples` (10k) fakes **on all ranks** with the eval sampler, scored
against the **whole training set** as the real reference. Feature and angle
extraction is sharded per rank and gathered to rank 0, which computes the
distances — so the metrics are computed on all GPU ranks, matching DiffiT-v2.
Logged under `Metrics/` in TensorBoard and to `stats.jsonl`:

- `combra_fid10k` (InceptionV3 FID), `combra_cmmd10k` (CLIP-MMD), `combra_fd_dinov2_10k` (DINOv2 Fréchet)
- angle-density metrics: `combra_w1`, `combra_w2`, `combra_circular_w1/w2`, `combra_mu1/mu2`, `combra_sigma1/sigma2`, `combra_amp1/amp2`

The offline `calculate_metrics.py` (`edm2-eval`) FID / FD-DINOv2 evaluator is kept
unchanged for standalone reference-stats evaluation.

## Generating samples

Individual PNGs (per seed):

```bash
edm2-gen-images --net=training-runs/00000-.../network-snapshot-final.pkl \
    --outdir=generated/512 --seeds=0-63 --sampler=dpm++ --steps=25 --class=207
```

Bulk `.npz` for FID (distributed):

```bash
torchrun --standalone --nproc_per_node=4 sample_images.py \
    --net=…/network-snapshot-final.pkl --outdir=samples/512 \
    --num-samples=50000 --batch=16 --sampler=dpm++ --steps=25
```

SLURM: `sbatch sbatch/generate_1gpu_{256,512,1024}.sbatch`,
`sbatch sbatch/sample_4gpu_{256,512,1024}.sbatch`.

## Samplers

All samplers run in EDM σ-space on the same `net(x, σ, labels)` denoiser and honor
`--guidance`/`--gnet`:

- **`dpm++`** — DPM-Solver++(2M) in log-σ space. **The default**, at 25 steps. 2nd-order
  accurate at one denoiser evaluation per step (Heun needs two), so it is the cheapest
  route to near-converged quality: 25 evaluations vs 63 for the old `edm`-at-32 default.
- **`edm`** — 2nd-order Heun (EDM paper); the only sampler supporting stochasticity, via `S_churn`.
- **`euler`** — 1st-order deterministic Euler.
- **`ddim`** — deterministic DDIM (η=0), which is the first-order EDM step (≡ `euler`).

## Finding the optimal number of sampling steps

`edm2-compare-samplers` sweeps samplers × step counts, scores each batch against a
real reference with `combra.metrics.compare_samplers`, and writes a table + plot;
the metric-vs-steps curve plateaus at the optimal step count per sampler:

```bash
python compare_samplers.py \
    --net=…/network-snapshot-final.pkl --data=datasets/imagenet_256x256.zip \
    --samplers=edm,euler,ddim,dpm++ --k-values=5,10,20,50,100,250 \
    --num-samples=512 --outdir=sampler-comparison/256
# -> sampler_comparison.parquet + sampler_comparison.png
```

SLURM: `sbatch sbatch/compare_samplers_256x256.sbatch`.

## Project structure

```
edm2-v2/
├── train_edm2.py            # training entry point (edm2-train)
├── generate_images.py       # per-seed PNG generation (edm2-gen-images)
├── sample_images.py         # bulk .npz sampling (edm2-sample)
├── compare_samplers.py      # optimal-steps analysis (edm2-compare-samplers)
├── calculate_metrics.py     # offline FID / FD-DINOv2 (edm2-eval)
├── dataset_tool.py          # dataset preparation (edm2-prepare-data)
├── download_models.py       # prefetch VAE + combra backbones (edm2-download-models)
├── training/
│   ├── training_loop.py     # main loop (frozen loss/optimizer/EMA update)
│   ├── networks_edm2.py     # Precond + magnitude-preserving U-Net (frozen)
│   ├── phema.py             # Power-Function / Traditional EMA (frozen)
│   ├── encoders.py          # RGB / Stability VAE latent encode-decode
│   ├── dataset.py           # ImageFolderDataset
│   ├── logger.py            # DiffiT-style multi-format logger
│   ├── metrics.py           # inline combra metrics, sharded across ranks
│   └── samplers.py          # edm / euler / ddim / dpm++
├── sbatch/                  # SLURM scripts (train/generate/sample/compare, 3 res)
├── tests/test_smoke.py      # CPU smoke tests
└── pyproject.toml           # packaging + console entry points
```

## Tests

```bash
pip install pytest
pytest tests/ -q
```

## License

This codebase inherits the upstream EDM2 license (Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International — see `LICENSE.txt`).
