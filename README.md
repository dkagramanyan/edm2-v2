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

> **The training math is unchanged.** The EDM2 loss, optimizer step,
> magnitude-preserving network, preconditioning and Power-Function EMA (the "training
> dynamics and update layers") are preserved exactly; only the learning-rate rampup
> length now follows the batch size (see [Learning-rate schedule](#learning-rate-schedule)).
> Everything else new below runs *around* that core — at logging, inference and eval
> time only.

## Differences from upstream NVlabs/edm2

Every difference from [NVlabs/edm2](https://github.com/NVlabs/edm2), marked by kind:
**improvement** (a deliberate change to how training or sampling behaves),
**contract** (the v2 model-API convention shared by the four model repos) or
**adaptation** (needed for this data, hardware or a current software stack).

| Area | Kind | Upstream edm2 | This repo |
|---|---|---|---|
| **Precision** | improvement | FP16 mixed precision (`--fp16`) | `--precision fp32/fp16/bf16`; **fp16 stays the default**, bf16 is an option (`Precond(mixed_precision_dtype=…)`) |
| **TF32** | improvement | TF32 disabled on cuDNN and matmul | **TF32 on by default** (`--tf32 True`, since e661427); `--tf32 False` restores upstream |
| **Latent encoding** | improvement | dataset pre-encoded to an 8-channel latent zip offline | **on-the-fly StabilityVAE encode** (`StabilityVAEOnTheFlyEncoder`, DiffiT-style): latent diffusion straight from a raw-RGB zip, the frozen VAE runs each step under `no_grad`. Offline 8-channel latent zips still work and are auto-detected. A failed VAE load names the cache dir and the fix |
| **LR rampup** | improvement | 10 Mimg at any batch (~4.9k iterations at batch 2048) | counted in iterations: 10 Mimg × batch / 2048, so it stays ~4.9k iterations at any batch; `--rampup` overrides (see [Learning-rate schedule](#learning-rate-schedule)) |
| **Samplers** | improvement | EDM 2nd-order Heun only | `dpm++` (DPM-Solver++ 2M, **default**, 25 steps), `edm` (Heun), `euler`, `ddim`, σ-space, **one implementation shared by training-eval and generation** |
| **Augmentation** | adaptation | none: `Dataset(xflip=…)` option off and not exposed by `train_edm2.py` | **`--augment True`** (default): each training item gets a uniformly random dihedral transform (rot90 × h-flip, 8 in all) on the fly, for a small dataset (1080 crops) whose microstructure has no preferred orientation. The combra reference is expanded to the same 8 orientations. `--augment False` trains without augmentation, as upstream. The x-flip dataset option itself is gone (see [Augmentation](#augmentation)) |
| **Batch size** | adaptation | preset batch 2048 (`--batch`) | from the CLI: `--batch-gpu × --gpus × --grad-accum`; the `sh/` scripts use **128 / 64 / 32** at 256 / 512 / 1024 px (2 GPUs). α_ref and t_ref stay the paper's |
| **Resolutions / presets** | adaptation | img64, img512 presets | added `edm2-img256-*` and `edm2-img1024-*` presets with the paper's (Table 6) img512 values for the same model size, + `sh/` launch scripts for 256/512/1024 |
| **Classes** | adaptation | ImageNet, `label_dim = 1000` | `label_dim` is still inferred from the dataset; the WC-Co zips have 3 classes, so **`label_dim = 3`** |
| **Loader workers** | adaptation | 2 DataLoader workers | **3** (`--workers`), sized for 8 CPUs / 2 ranks |
| **InfiniteSampler** | adaptation | `super().__init__(dataset)` + a warning filter | `super().__init__()`: current PyTorch dropped the sampler's `data_source` argument |
| **Dataset checks** | contract | any channel count; no class names | images must be **3-channel RGB** (asserted on load; `edm2-prepare-data convert` converts grayscale at build time); `dataset.json` carries index-aligned **`class_names`**, and a conditional run on a zip without them is refused. `--max-images` is stratified across classes |
| **Checkpointing** | contract | resumable `training-state-*.pt` + pickled `network-snapshot-*.pkl` per EMA std | **EMA-only `.pt` state-dict inference snapshots** `edm2-snapshot-<kimg>[-<std>]-inference.pt`, written atomically each snapshot tick **and always at the last tick**, carrying `{n_classes, resolution, class_names, cur_nimg}`. **No resume**: every launch gets a fresh run dir. Retention: `--snapshot-keep-last N` (default 1) newest **plus the best by each of `combra_fid`, `combra_fd_dinov2`, `combra_cmmd`** (never pruned); `0` keeps all |
| **Metrics** | contract | offline FID / FD-DINOv2 only (`calculate_metrics.py`) | inline **combra** metrics every snapshot tick, **sharded and gathered across all GPU ranks**, reference from **raw dataset pixels**: `combra_fid`, `combra_cmmd`, `combra_fd_dinov2` (DINOv2 ViT-L/14) + angle-density metrics. The offline evaluator is kept |
| **Training-time guidance** | contract | — (no eval in the loop) | the loop has no guiding network, so `edm2-train --guidance` other than 1 is refused; guidance is applied at generation time (`--gnet --guidance`) |
| **Logging** | contract | single `Status:` line + `stats.jsonl` | the §7 spec: rank-0 `<run>.log` (one timestamp per line), scalar-only `stats.jsonl` (one row per tick, metrics in the eval tick's row), TensorBoard (`events.out.tfevents.*` with the run name as `filename_suffix`), a `tick … kimg … sec/tick …` console line, `reals.png` / `fakes_init.png` / `fakes<kimg>.png` grids |
| **Generation** | contract | flat `<seed>.png` from `.pkl` networks | per-class HDF5 (`edm2-gen-images --classes … --samples-per-class …`) in the wc_cv angle-pipeline `RankH5Writer` layout from `.pt` snapshots; `.pkl` loading removed; `reconstruct_phema.py` kept for upstream `.pkl` files only |
| **Packaging / launch** | contract | `torchrun python train_edm2.py …`, `--preset/--batch/--duration` | `pip install -e '.[combra]'` + console entry points; `--gpus N` self-spawns (no torchrun); `--cfg`/`--kimg`/`--tick`/`--snap`; pyproject is the only dependency declaration |
| **Small fixes** | adaptation | — | `scipy.linalg.sqrtm` without the removed `disp` argument (`calculate_metrics.py`); the process group is destroyed at exit |

## Installation

64-bit **Python 3.10+** (3.12 recommended). Install the latest **PyTorch** from
the CUDA 13.x wheels, then the package:

```bash
conda create -n edm2-v2 python=3.12 -y
conda activate edm2-v2
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132
pip install -e '.[combra]'      # omit [combra] to train without inline combra metrics
```

`combra` (the WC-Co computer-vision metrics library) is optional; the import is
guarded, so training runs unchanged without it. The image metrics — FID
(pytorch-fid), CMMD (open-clip-torch), FD-DINOv2 (torch.hub DINOv2) — need combra's
**`[metrics]` extra**, which the `[combra]` extra here requests for you: combra 0.5.0
moved that torch stack out of its base dependencies, so a bare `combra` install
leaves all three returning `nan`. combra also floors Python at **3.12**, which is why
this package does too. It lives in a **private** repo, so the `[combra]` extra clones it over
`git+https` and only succeeds when you are authenticated to GitHub — sign in once
with the GitHub CLI and `pip` inherits its credential helper:

```bash
gh auth login        # github.com → HTTPS
pip install -e '.[combra]'
```

On an air-gapped node `pip install -e .` fails with *"Could not find a version
that satisfies setuptools>=61"* — build isolation tries to fetch its own build
backend. Reuse the env's instead:

```bash
pip install -e . --no-build-isolation
```

Pre-fetch the VAE and metric backbones for offline nodes (run on a node with internet):

```bash
bash download_models.sh                                  # caches under ~/.cache
MODEL_CACHE=/shared/team/caches bash download_models.sh  # or somewhere else
```

This fetches, with wget/curl + git (no GPU), InceptionV3 (FID) and **DINOv2 ViT-L/14**
(FD-DINOv2, combra's default since 0.18.0; about 1.2 GB) into `torch/hub`, the CLIP
ViT-L-14-336 `openai` weights (CMMD) into the HuggingFace hub cache, and — through the
`hf` / `huggingface-cli` CLI — the Stability VAE `stabilityai/sd-vae-ft-mse`. Without the
CLI it prints the fallback, `python download_models.py --no-combra` (also installed as
`edm2-download-models`, which warms the same caches through combra's feature extractors
and `load_stability_vae`). With `MODEL_CACHE` set, point the jobs at it:
`TORCH_HOME=$MODEL_CACHE/torch HF_HOME=$MODEL_CACHE/huggingface DNNLIB_CACHE_DIR=$MODEL_CACHE/dnnlib`.

The VAE is cached under `~/.cache/dnnlib/diffusers` (`$DNNLIB_CACHE_DIR/diffusers` when
that is set), **not** the standard `~/.cache/huggingface` — `load_stability_vae`
passes that dir as `cache_dir`, so a copy another tool cached is not visible here. Without it,
latent presets (`edm2-img256/512/1024-*`) cannot run; the RGB `edm2-img64-*` presets
need no VAE at all.

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
  (default) disables guidance. Every sampler (`edm/euler/ddim/dpm++`) honors it.
  Training-time combra eval has no guiding network, so `edm2-train` refuses
  `--guidance` other than 1.
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

The WC-Co zips used by the `sh/` scripts are
`imagenet_9to4_1024x1024_<r>x<r>.zip` (r = 256 / 512 / 1024): the **1080 original crops**,
360 per class, `class_names` `['Ultra_Co25', 'Ultra_Co11', 'Ultra_Co6_2']`. They
replace the `imagenet_9to4_1024x1024_<r>x<r>.zip` zips, which stored each crop in all
8 dihedral orientations (8640 images); the orientations are now drawn on the fly
(`--augment`, see [Augmentation](#augmentation)).

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

### Learning-rate schedule

The schedule is the paper's Eq. 67 with a linear warm-up:
α(t) = α_ref · min(t / t_rampup, 1) / √max(t / t_ref, 1), with t in iterations. The
paper (Table 6) and upstream tune it at batch 2048: α_ref per model size, t_ref = 70k
iterations (35k for img64) and a 10 Mimg rampup, i.e. ~4.9k iterations. t_ref is
already counted in iterations; upstream counts the rampup in images, which at this
repo's batch 128 / 64 / 32 (256 / 512 / 1024 px on 2 GPUs) would stretch it to
78k / 156k / 312k iterations — past the decay knee — so the peak would never reach
α_ref (0.95 / 0.67 / 0.47 × α_ref). Here the rampup is 10 Mimg × batch / 2048, which
keeps both the rampup (~4.9k iterations) and the knee (70k iterations) where the paper
puts them, and the peak at α_ref. α_ref itself is left at the paper's value: neither
the paper nor upstream gives a rule for rescaling it with the batch size.

Ready-made launch scripts live in `sh/` — self-locating and offline-cluster ready
(`HF_HUB_OFFLINE=1`); SLURM specifics are supplied at submission time:

```bash
bash sh/train_256.sh                                   # workstation
sbatch --account=<proj> --partition=rocky --gpus=2 sh/train_256.sh   # cluster
```

### Augmentation

`--augment True` (default) applies a random element of the dihedral group to each
training item: rot90 by k ∈ {0, 1, 2, 3} and a horizontal flip with probability 0.5,
all 8 transforms equally likely. It acts on the raw uint8 image in the training batch,
before the VAE encode, so it needs square RGB images (a pre-encoded latent zip needs
`--augment False`). The draw comes from the per-iteration seed
(`seed, rank, cur_nimg`), so a run is reproducible. Only training batches are
augmented: the combra eval fakes, the `reals.png` / `fakes*.png` grids and generation
never are. The combra reference instead covers all 8 orientations of every reference
image (`precompute_reference(..., dihedral=True)`, combra ≥ 0.19.0), so it matches the
distribution training sees. `--augment False` trains on the images as stored and uses
the plain reference.

An epoch is one pass over the dataset: **1080 images** with the current zips (it was
8640 with the old 8-orientation zips). kimg counts training images seen, so `--kimg`,
`--tick` and the eval / snapshot cadence mean the same amount of training as before;
1 kimg is now ~0.93 epochs instead of ~0.12.

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
| `--lr` / `--decay` / `--rampup` | preset / preset / 10 × batch / 2048 | Learning rate max α_ref / decay knee t_ref (iterations) / rampup (Mimg) |
| `--tick` / `--snap` | 128 / 64 | Status tick interval (kimg) / snapshot every N ticks |
| `--kimg` | preset | Total training length in kimg |
| `--augment` | `True` | Random dihedral transform (rot90 × h-flip) per training item; `False` = none (see [Augmentation](#augmentation)) |
| `--workers` | 3 | DataLoader worker processes |
| `--snapshot-keep-last` | 1 | Newest inference snapshots kept, plus the best by each of `combra_fid` / `combra_fd_dinov2` / `combra_cmmd` (0 = keep all) |
| `--desc` | — | String appended to the run directory name |
| `--combra-metrics` | `True` | Inline combra metrics each snapshot tick (all ranks) |
| `--num-fid-samples` | 10000 | Fakes generated (all ranks) per combra eval; 0 disables |
| `--combra-ref-count` | 0 (whole set) | Real reference images for combra (seeded random subset) |
| `--eval-sampler` | `dpm++` | Eval-time / snapshot sampler (`edm/euler/ddim/dpm++`) |
| `--eval-sampling-steps` | 25 | Eval-time sampling steps |
| `--guidance` | 1 | Eval-time guidance strength; only 1 accepted (training has no `--gnet`) |
| `-n, --dry-run` | off | Print resolved config and exit |

### Training output

```
runs/00000-edm2-img256-s-gpus2-batch128/
├── training_options.json                       # all hyperparameters
├── 00000-edm2-img256-s-gpus2-batch128.log      # rank-0 console transcript
├── stats.jsonl                                 # one row per tick: scalars (+ combra metrics on eval ticks)
├── events.out.tfevents.*.00000-…-batch128      # TensorBoard (run name as filename_suffix)
├── reals.png                                   # real image grid (raw dataset pixels, class-sorted)
├── fakes_init.png                              # pre-training samples
├── fakes000200.png …                           # samples per snapshot tick
└── edm2-snapshot-000200-0.100-inference.pt …   # EMA-only .pt state dicts (one per EMA std), newest + best kept
```

Each run gets a **fresh** directory under `--outdir`, named
`<id:05d>-<cfg>-gpus<N>-batch<B>[-desc]`. The newest snapshot is always the final
model (a snapshot is written at the last tick regardless of cadence).

Snapshot retention: after each snapshot tick's combra eval has scored the new
snapshot, `--snapshot-keep-last N` (default 1) newest snapshots are kept plus the best
one by each of `combra_fid`, `combra_fd_dinov2` and `combra_cmmd` (lower is better;
nan or missing values are skipped; a tie keeps the earlier one). A best snapshot is
never pruned by later ticks, one snapshot may be best for several metrics, and all
per-EMA-std files of one kimg count as one snapshot, so at most N + 3 kimgs remain.
`0` keeps every snapshot. Each snapshot tick logs one line:
`Best snapshots: combra_fid <v> <file>  combra_fd_dinov2 <v> <file>  combra_cmmd <v> <file>`.

Monitor with `tensorboard --logdir runs`.

## Quality metrics (combra, all ranks)

With `--combra-metrics` on (default), every snapshot tick generates
`--num-fid-samples` (10k) fakes **on all ranks** with the eval sampler, scored
against the training set as the real reference — **raw dataset pixels**, never VAE
round-tripped (`--combra-ref-count` caps it to a seeded random subset); with
`--augment True` each reference image enters in all 8 dihedral orientations. Feature and
angle extraction is sharded per rank and gathered to rank 0, which computes the
distances — so the metrics are computed on all GPU ranks, matching DiffiT-v2.
Logged under `Metrics/` in TensorBoard and to `stats.jsonl`:

- `combra_fid` (InceptionV3 FID), `combra_cmmd` (CLIP-MMD), `combra_fd_dinov2` (DINOv2 ViT-L/14 Fréchet), `combra_fid_best` (running best), `combra_num_fid_samples` (the count the run used)
- angle-density metrics: `combra_w1`, `combra_w2`, `combra_circular_w1/w2`, `combra_mu1/mu2`, `combra_sigma1/sigma2`, `combra_pi`

`stats.jsonl` holds one row per tick; on an eval tick the `Metrics/*` keys (and
`Timing/eval_sec`) sit in that tick's row alongside `Progress/kimg`, so
`combra.metrics.load_fid_by_kimg` reads a run's FID history directly. Values are
full-precision JSON, with non-finite values written as `null`. (The
keys used to carry a literal `10k` suffix that never tracked `--num-fid-samples`, and
sat in a row without `Progress/kimg`; neither is true any more.)

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
├── download_models.sh       # prefetch VAE + combra backbones (bash download_models.sh)
├── download_models.py       # Python fallback for the VAE (edm2-download-models)
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
