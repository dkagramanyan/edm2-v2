# EDM2: Analyzing and Improving the Training Dynamics of Diffusion Models (v2 refresh)

Official EDM2 codebase, refreshed to the **v2 model-API convention** (see the wc_cv
`models_api_proposal`): a unified `click` + console-entry-point CLI, inline **combra**
generative-quality metrics computed **across all GPU ranks**, EMA-only `.pt`
inference snapshots, per-class **HDF5** generation for the WC-Co angle pipeline,
class-conditional latent-diffusion training at **256 / 512 / 1024**, and multiple
reverse-diffusion samplers (**EDM Heun / Euler / DDIM / DPM-Solver++**).

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
| **Logging** | single `Status:` line + `stats.jsonl` | rank-0 `<run>.log`, scalar-only `stats.jsonl`, TensorBoard (`events.out.tfevents.*` with the run name as `filename_suffix`), a `tick … kimg … sec/tick …` console line, plus `reals.png` / `fakes_init.png` / `fakes<kimg>.png` grids |
| **Latent encoding** | dataset pre-encoded to an 8-channel latent zip offline | **inline VAE encode** (DiffiT-style): train latent diffusion straight from a raw-RGB zip, the frozen Stability VAE runs each step — no pre-encode pass. Offline 8-channel latent zips still work and are auto-detected |
| **Checkpointing** | full resumable `training-state-*.pt` (one per tick) | **EMA-only `.pt` state-dict inference snapshots** `edm2-snapshot-<kimg>[-<std>]-inference.pt`, written atomically each snapshot tick **and always at the last tick**, pruned to `--snapshot-keep-last` (default 3). Every snapshot carries `{n_classes, resolution, class_names, cur_nimg}`. **No resume, no best-model, no rolling latest** — the newest snapshot is the final model |
| **Metrics** | offline FID / FD-DINOv2 only (`calculate_metrics.py`) | inline **combra** metrics every snapshot tick, **sharded and gathered across all GPU ranks**, reference from **raw dataset pixels**: `combra_fid10k`, `combra_cmmd10k`, `combra_fd_dinov2_10k` + angle-density metrics |
| **Generation** | flat `<seed>.png` | per-class HDF5 (`edm2-gen-images --classes … --samples-per-class …`) in the wc_cv angle-pipeline `RankH5Writer` layout, self-spawning `--gpus` (no torchrun) |
| **Samplers** | EDM 2nd-order Heun only | `dpm++` (DPM-Solver++ 2M, **default**, 25 steps), `edm` (Heun), `euler`, `ddim`, σ-space, **one implementation shared by training-eval and generation** |
| **Packaging / API** | `python train_edm2.py …` | `pip install -e '.[combra]'` + console entry points; pyproject is the only dependency declaration (no `requirements.txt`, no Hydra) |
| **Resolutions** | img64, img512 presets | added `edm2-img256-*` and `edm2-img1024-*` presets + `sh/` launch scripts for 256/512/1024 |

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
  uses.
- **Classifier-free / auto-guidance.** At sampling time you can steer generation
  with a second *guiding* network via `--gnet` and `--guidance` (strength `> 1`):
  the denoiser output is extrapolated away from the guiding network's,
  `D = lerp(D_guide, D_main, guidance)`. Use the model's own weaker/earlier
  checkpoint (autoguidance) or an unconditional model as `--gnet`. `--guidance 1`
  (default) disables guidance. The same `--guidance` is honored during training-time
  combra eval and by every sampler (`edm/euler/ddim/dpm++`).
- **Specific classes.** `edm2-gen-images --classes=<spec> --samples-per-class=N`
  selects which classes to generate — `<spec>` is indices, ranges, or class names
  (`0,1,4-6` or `Ultra_Co11`). The legacy `--seeds` mode takes a single `--class`.

## Data preparation

`edm2-prepare-data` is a click group. `convert` center-crops a folder of images into
an **RGB** training zip, deriving integer labels from the **alphabetical
class-folder order** and writing index-aligned `class_names` into `dataset.json`
(grayscale SEM images are converted to RGB at build time):

```bash
edm2-prepare-data convert --source=/data/wc_co --dest=datasets/wc_co_512x512.zip \
    --resolution=512x512 --transform=center-crop-dhariwal
```

Produce one zip per target resolution (`256x256`, `512x512`, `1024x1024`). Training
runs latent diffusion straight from the RGB zip — the frozen Stability VAE encodes
inline each step, no pre-encode pass. An 8-channel pre-encoded latent zip
(`edm2-prepare-data encode`, legacy) is auto-detected too.

## Training

Pick a preset with `--cfg`; any CLI option overrides the preset. `--gpus N`
self-spawns one worker per GPU — **no `torchrun` for training**. The global batch is
`--batch-gpu × --gpus × --grad-accum`:

```bash
# 256²
edm2-train --outdir=runs --cfg=edm2-img256-s \
    --data=datasets/wc_co_256x256.zip --gpus=2 --batch-gpu=64 --tick=128 --snap=64

# 512²
edm2-train --outdir=runs --cfg=edm2-img512-s \
    --data=datasets/wc_co_512x512.zip --gpus=2 --batch-gpu=32 --tick=128 --snap=64

# 1024²
edm2-train --outdir=runs --cfg=edm2-img1024-s \
    --data=datasets/wc_co_1024x1024.zip --gpus=2 --batch-gpu=16 --tick=128 --snap=64
```

**Runs are not resumable by design** (§3): a crash or SLURM walltime kill cannot be
continued, and every launch allocates a fresh run id. Size `--kimg` (or split
resolution stages) so a run fits its job's time limit.

Ready-made launch scripts live in `sh/` — self-locating and offline-cluster ready
(`HF_HUB_OFFLINE=1`); SLURM specifics are supplied at submission time:

```bash
bash sh/train_256.sh                                   # workstation
sbatch --account=<proj> --partition=rocky --gpus=2 sh/train_256.sh   # cluster
```

### Key training options

| Option | Default | Description |
|--------|---------|-------------|
| `--outdir` | required | Output directory for the run |
| `--data` | required | Dataset zip/dir (`edm2-prepare-data` output) |
| `--cfg` | `edm2-img512-s` | Config preset (`edm2-img256/512/1024-*`, `edm2-img64-*`) |
| `--gpus` | 1 | GPUs to self-spawn (no torchrun) |
| `--cond` | `True` | Train class-conditional model |
| `--batch-gpu` | 32 | Per-GPU batch size |
| `--grad-accum` | 1 | Gradient accumulation rounds (total batch = batch-gpu × gpus × grad-accum) |
| `--precision` | `fp16` | Training precision (`fp32`/`fp16`/`bf16`) |
| `--tf32` | `True` | Enable TF32 on cuDNN / matmul |
| `--tick` / `--snap` | 128 / 64 | Status tick interval (kimg) / snapshot every N ticks |
| `--kimg` | preset | Total training length in kimg |
| `--mirror` | `False` | Stochastic horizontal flip in the training loader only |
| `--workers` | 3 | DataLoader worker processes |
| `--snapshot-keep-last` | 3 | Newest inference snapshots kept (0 = keep all) |
| `--desc` | — | String appended to the run directory name |
| `--combra-metrics` | `True` | Inline combra metrics each snapshot tick (all ranks) |
| `--num-fid-samples` | 10000 | Fakes generated (all ranks) per combra eval; 0 disables |
| `--combra-ref-count` | 0 (whole set) | Real reference images for combra (seeded random subset) |
| `--eval-sampler` | `dpm++` | Eval-time / snapshot sampler (`edm/euler/ddim/dpm++`) |
| `--eval-sampling-steps` | 25 | Eval-time sampling steps |
| `--guidance` | 1 | Eval-time classifier-free guidance strength |
| `-n, --dry-run` | off | Print resolved config and exit |

### Training output

```
runs/00000-edm2-img256-s-gpus2-batch128/
├── training_options.json                       # all hyperparameters
├── 00000-edm2-img256-s-gpus2-batch128.log      # rank-0 console transcript
├── stats.jsonl                                 # machine-readable scalars + combra metrics (scalar rows only)
├── events.out.tfevents.*.00000-…-batch128      # TensorBoard (run name as filename_suffix)
├── reals.png                                   # real image grid (raw dataset pixels, class-sorted)
├── fakes_init.png                              # pre-training samples
├── fakes000200.png …                           # samples per snapshot tick
└── edm2-snapshot-000200-0.100-inference.pt …   # EMA-only .pt state dicts (one per EMA std), pruned
```

Each run gets a **fresh** directory under `--outdir`, named
`<id:05d>-<cfg>-gpus<N>-batch<B>[-desc]`. The newest snapshot is always the final
model (a snapshot is written at the last tick regardless of cadence).

Monitor with `tensorboard --logdir runs`.

## Quality metrics (combra, all ranks)

With `--combra-metrics` on (default), every snapshot tick generates
`--num-fid-samples` (10k) fakes **on all ranks** with the eval sampler, scored
against the training set as the real reference — **raw dataset pixels**, never VAE
round-tripped (`--combra-ref-count` caps it to a seeded random subset). Feature and
angle extraction is sharded per rank and gathered to rank 0, which computes the
distances — so the metrics are computed on all GPU ranks, matching DiffiT-v2.
Logged under `Metrics/` in TensorBoard and to `stats.jsonl`:

- `combra_fid10k` (InceptionV3 FID), `combra_cmmd10k` (CLIP-MMD), `combra_fd_dinov2_10k` (DINOv2 Fréchet)
- angle-density metrics: `combra_w1`, `combra_w2`, `combra_circular_w1/w2`, `combra_mu1/mu2`, `combra_sigma1/sigma2`, `combra_amp1/amp2`

The offline `calculate_metrics.py` (`edm2-eval`) FID / FD-DINOv2 evaluator is kept
unchanged for standalone reference-stats evaluation.

## Generating samples

Per-class HDF5 in the wc_cv angle-pipeline layout (`--gpus` self-spawns workers, no
torchrun). `--classes` accepts indices, ranges, or class names:

```bash
edm2-gen-images \
    --network=runs/00000-.../edm2-snapshot-000200-0.100-inference.pt \
    --outdir=generated/512 --classes=0,1,2 --samples-per-class=1000 \
    --gpus=2 --batch-gpu=32 --save-mode=hdf5 --sampler=dpm++ --steps=25
```

This writes per-rank `shards/rank_NNN.h5` (`class_<c>/images|seeds`, uint8 NHWC),
merged into `<desc>.h5` with `format="generated_images_shard"`, `schema_version=1`
and `class_names`; the merge hard-fails if any shard is incomplete. `--save-mode=dir`
writes `class_<c>/idx_<i:06d>_seed_<s>.png` + a `classes.json` manifest instead.

SLURM: `sbatch --gpus=2 sh/generate_{256,512,1024}.sh`. The legacy per-seed
(`--seeds`) mode and the bulk `.npz` `sample_images.py` remain for the upstream FID
protocol but carry no contract guarantees.

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
    --net=runs/00000-.../edm2-snapshot-000200-0.100-inference.pt --data=datasets/wc_co_256x256.zip \
    --samplers=edm,euler,ddim,dpm++ --k-values=5,10,20,50,100,250 \
    --num-samples=512 --outdir=sampler-comparison/256
# -> sampler_comparison.parquet + sampler_comparison.png
```

## Project structure

```
edm2-v2/
├── train_edm2.py            # training entry point (edm2-train)
├── generate_images.py       # per-class HDF5 generation (edm2-gen-images)
├── sample_images.py         # [legacy] bulk .npz sampling (edm2-sample)
├── compare_samplers.py      # optimal-steps analysis (edm2-compare-samplers)
├── calculate_metrics.py     # offline FID / FD-DINOv2 (edm2-eval)
├── dataset_tool.py          # dataset preparation (edm2-prepare-data)
├── download_models.py       # prefetch VAE + combra backbones (edm2-download-models)
├── training/
│   ├── training_loop.py     # main loop (frozen loss/optimizer/EMA update)
│   ├── networks_edm2.py     # Precond + magnitude-preserving U-Net (frozen)
│   ├── phema.py             # Power-Function / Traditional EMA (frozen)
│   ├── encoders.py          # RGB / Stability VAE latent encode-decode
│   ├── dataset.py           # ImageFolderDataset (writes/reads class_names)
│   ├── checkpoint.py        # .pt EMA-only inference snapshot save/load
│   ├── h5_writer.py         # RankH5Writer + shard merge
│   ├── logger.py            # rank-0 text/TensorBoard logger
│   ├── metrics.py           # inline combra metrics, sharded across ranks
│   └── samplers.py          # edm / euler / ddim / dpm++
├── sh/                      # launch scripts (train/generate, 3 res)
├── tests/                   # CPU smoke + §13 conformance tests
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
