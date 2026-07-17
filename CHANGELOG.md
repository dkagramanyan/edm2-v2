# Changelog

All notable changes to this fork (`edm2`) are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [3.0.0] — 2026-07-17

Conformance with the **v2 model-API convention** documented in `wc_cv`
(`docs/examples/models_api_proposal.md`). This is a breaking release: interrupted
runs can no longer be resumed, the checkpoint format changed from pickled modules to
`.pt` state dicts, and several CLI flags were renamed or removed.

### Changed
- **Unified training CLI (§2).** Progress is counted in **kimg and ticks**:
  `--duration/--status/--snapshot` (and the `Ki/Mi` suffix parsing) are replaced by
  `--kimg`/`--tick`/`--snap`. The total-batch flag `--batch` is gone; the batch is
  `--batch-gpu × --gpus × --grad-accum` with `--grad-accum` explicit (default 1).
  `--fp16` becomes `--precision {fp32,fp16,bf16}`; `--tf32 True/False` (default
  `True`, previously hardcoded off) and `--bench True/False` control the cuDNN/matmul
  paths. `--latent/--pixel` becomes `--latent True/False`. Added `--desc`,
  `--workers` (default 3), and `--mirror True/False` (loader-level stochastic
  horizontal flip in the **training** loader only; eval and the combra reference
  never flip).
- **Checkpoint contract (§3).** Snapshots are now EMA-only `.pt` **state dicts**
  named `edm2-snapshot-<kimg:06d>[-<ema_std>]-inference.pt`, written atomically
  (temp + `os.replace`) every snapshot tick **and always at the last tick**, pruned
  to `--snapshot-keep-last` (default 3, `0` = keep all). Every snapshot carries
  `{n_classes, resolution, class_names, cur_nimg}` metadata; loading rebuilds the
  model from current code.
- **Generation contract (§4).** `edm2-gen-images` gains a class-batch mode
  (`--classes 0,1,4-6` or names + `--samples-per-class N`) and `--save-mode
  {hdf5,dir}`. HDF5 output is the `RankH5Writer` layout (`class_<c>/images|seeds`,
  uint8 NHWC) written as per-rank `shards/rank_NNN.h5` and merged into `<desc>.h5`
  with `format="generated_images_shard"`, `schema_version=1` and `class_names`; the
  merge **hard-fails** on incomplete shards (`missing_count`). Generation
  self-spawns per-GPU workers via `--gpus` (no torchrun) and uses `--batch-gpu`. The
  per-image seed is `base + class·samples_per_class + idx`. `--network` is an alias
  for `--net`.
- **combra evaluation (§6).** The reference is now extracted from **raw dataset
  pixels** (never VAE round-tripped), and `--combra-ref-count` takes a **seeded
  random** subset instead of the first N.
- **Logging (§7).** `stats.jsonl` is scalar-rows-only; the vendored
  OpenAI-baselines `progress.csv` / `progress.json` are gone. The console log is
  rank-0-only and named after the run directory; the tfevents file carries the run
  name as a `filename_suffix`.
- **Dataset/label contract (§5).** `edm2-prepare-data convert` writes index-aligned
  `class_names` (alphabetical folder order) into `dataset.json`; RGB conversion
  happens at build time with runtime 3-channel asserts.

### Removed
- **Resume / best-model machinery**: `--resume`-style auto-resume, the rolling
  `network-snapshot-latest.pt`, `best_model.pt`, `--save-inference-only`, and the
  desc-matching run-dir reuse. Every launch allocates a **fresh** run id.
- **Pickled-module snapshots** (`.pkl`) and their loaders; the last pickle-capable
  commit is tagged `legacy-pkl`.
- **Hydra** (`train_hydra.py`, `configs/`, the `hydra-core` dependency) and
  `requirements.txt` (pyproject is the only dependency declaration). The committed
  `.hydra/` dir and `train_hydra.log` were untracked.
- Dead `should_stop` / `should_suspend` / `update_progress` stubs; the stale
  `docs/*-help.txt` dumps; the `sbatch/` collection.

### Added
- `sh/` launch scripts (`train_{256,512,1024}.sh`, `generate_{256,512,1024}.sh`) —
  self-locating, offline-cluster-ready (`HF_HUB_OFFLINE=1`), SLURM specifics
  supplied at submission time.
- `h5py` dependency; `--precision`, `--tf32`, `--grad-accum`, `--desc`,
  `--workers`, `--mirror` training flags; conformance smoke tests (§13).

## [2.1.0] — 2026-07-09

### Fixed
- **Rank-0 training crashed at startup.** `launch_training` installs
  `dnnlib.util.Logger` as `sys.stdout`, and `training_loop` then passed that tee to
  `HumanOutputFormat`, which asserted the stream had `.read`. The tee is write-only,
  so every rank-0 run died with an `AssertionError` before the first tick. The
  assertion now checks for `.write`, the only method used.
  (`training/logger.py`)
- **Training could not start on torch >= 2.11.** `InfiniteSampler.__init__` called
  `super().__init__(dataset)`, but `torch.utils.data.Sampler.__init__` is now plain
  `object.__init__` and rejects the argument (`TypeError: object.__init__() takes
  exactly one argument`). Dropped the argument, matching `san-v2` and `StyleSwin`.
  (`torch_utils/misc.py`)
- **Resume was impossible on torch >= 2.6.** `CheckpointIO.load` called
  `torch.load(...)` without `weights_only=False`, and the safe unpickler rejects the
  `dnnlib.EasyDict` state the checkpoint holds (`UnpicklingError: Unsupported global`).
  Loading a `training-state-*.pt` therefore always failed. (`torch_utils/distributed.py`)
- **Log lines carried two or three stacked timestamps.** `dnnlib.util.Logger`,
  `logger._do_log` and the hand-built tick line each prefixed their own. The stamp is
  now applied once: `dnnlib.util.Logger` skips lines that already carry one.
  (`dnnlib/util.py`, `training/training_loop.py`)

### Added
- **System time on every logged event.** `progress.csv` / `progress.json` rows now
  carry a `datetime` column (human-readable) and a `wall_time` column (Unix epoch
  seconds), so scalar rows can be aligned with the text log and with each other. The
  `stats.jsonl` text mirror gained `datetime`. (`training/logger.py`)
- **Hydra entry point** (`train_hydra.py` + `configs/config.yaml`), mirroring
  DiffiT-v2 and san-v2. The click CLI in `train_edm2.py` remains the single source of
  truth for every option and default; the Hydra path introspects it, overlays the
  YAML/CLI overrides, and calls the same `train_edm2.launch_from_opts(opts)` the click
  entry point uses, so both paths produce identical runs. `hydra-core` is a core
  dependency.
- **`train_edm2.launch_from_opts(opts)`** — the body of the click `main()`, extracted
  so the click and Hydra entry points share one launch path.
- **CI workflow** (`.github/workflows/ci.yml`) running ruff lint + CPU smoke tests, and
  a `requirements.txt` mirroring the pyproject dependencies.

### Changed
- **`dpm++` (DPM-Solver++(2M)) is now the default sampler everywhere**, at **25 steps**
  — training-time eval, `edm2-gen-images` and `sample_images.py`. It is 2nd-order
  accurate at one denoiser evaluation per step where the previous `edm` (Heun) default
  needed two, so the default eval costs 25 network evaluations instead of 63.
  Verified against the analytic Gaussian probability-flow ODE solution: empirical
  convergence order 2.09 (Heun: 2.06; euler/ddim: 0.99).
- **Per-run output directories**, DiffiT-style: training now writes to
  `<outdir>/<id:05d>-<preset>-gpus<N>-batch<B>` instead of straight into `--outdir`
  (which is what the README already claimed). Re-running the same command reuses the
  matching directory, preserving edm2's implicit "run it again to resume" behaviour;
  a different preset / GPU count / batch size gets a fresh number. The directory is
  resolved once in the parent process so spawned ranks cannot race to number one
  each. (`train_edm2.py`)
- **`--sampler` / `--sampling-steps` renamed to `--eval-sampler` /
  `--eval-sampling-steps`** in `train_edm2.py`, matching DiffiT-v2's training CLI. The
  generation scripts (`generate_images.py`, `sample_images.py`) keep `--sampler` /
  `--steps`. The training sbatch scripts were updated.
