# Changelog

All notable changes to this fork (`edm2`) are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Fixed
- **`stats.jsonl` rows are built by a testable function**, and a new
  `tests/test_stats_contract.py` feeds a real row to `combra.metrics.load_fid_by_kimg`.
  The reader was only ever tested against a synthetic flat row, so nothing checked the
  producer.
- **The §7 logging contract is now asserted** (`tests/test_logging_contract.py`).
  Thirteen scalar keys had drifted across the four repos; nothing failed because
  nothing checked. See below for this repo's share.

### Changed
- **The sharded eval harness moved into combra** (`combra.metrics.distributed`). This
  repo kept only what is model-specific: producing a shard of generated images and the
  float->uint8 denormalisation. The four private copies had drifted three ways --
  `all_gather` vs `gather`, a failure flag or none, and a different
  `precompute_reference` signature in each.
- **The combra startup check is `self_test(image_metrics=True, strict=True, images=...)`.**
  A missing CLIP download previously surfaced only as a whole run logging `nan`.
- **Hyperparameters reach TensorBoard.** The resolved config is read back from
  `training_options.json` at the end of training and written to the HPARAMS tab with
  the run's final `Metrics/combra_fid_best`, so runs are comparable by configuration
  and not only by curve shape. Nothing logged them before.
- **§7 keys:** the TensorBoard global step was kimg; it is now `cur_nimg`.
  `Loss/learning_rate` moved to `LearningRate/lr`, the image tags `reals`/`fakes`
  are now `Reals`/`Fakes` to match the other repos, and `Timing/eval_sec` is logged.

- **The two combra smoke fixtures were too small to fit a bimodal Gaussian.**
  Four 96px/10-polygon synthetic images yield only ~70 vertex angles, and the
  second mode then fits as a ~200 deg-wide pedestal, which combra reports as
  `nan`. `test_combra_angle_metrics_run_offline` failed on it and
  `test_combra_smoke_when_available` silently *skipped*, mistaking it for an
  offline-backend failure. Both now use 256px/80-polygon images (~740 angles).
  Real reference images were never affected -- a single 768px micrograph
  already yields ~300 angles and fits cleanly.
- **The combra contract test fed a unimodal sample to a bimodal-fit metric.**
  `test_angle_metrics_run_on_pooled_angles` drew two near-identical normals
  (mu 120 and 126), so the second Gaussian had no mode to sit on. combra now
  reports that as `nan` rather than dividing by the phantom, which turned the
  assertion red. The fixture is now genuinely bimodal (a 70/30 mixture at
  100 deg and 240 deg), which is what a WC-Co vertex-angle distribution
  actually looks like.
- **`scipy.linalg.sqrtm(..., disp=False)` raises under SciPy >= 1.18**, which
  removed the `disp` parameter. Fixed in `calculate_metrics.py`. Calling `sqrtm(X)` without `disp` returns
  the matrix alone on every SciPy version, so the fix is version-agnostic. This
  surfaced when the environment moved to SciPy 1.18 (see below); before that the
  call would have failed at runtime the moment anyone upgraded.

- `REQUIRED` in the contract test listed `self_test`, which this repo never calls.

### Changed
- **The conda environment is now `edm2-v2`** (Python 3.12, torch 2.13+cu130,
  numpy 2.5, SciPy 1.18), rebuilt alongside the previous `edm2` env rather
  than replacing it. `requires-python` has said `>=3.12` since the v2 convention
  landed, but the working env was still 3.11 — so `pip install -e .` could not
  succeed, which is why the console scripts were missing and combra was absent.
  README and `sh/` launch scripts point at the new name.
- **CI installs combra and arms the contract test.** `tests/test_combra_contract.py`
  is entirely `skipif(not combra_installed)`, and no CI job installed combra, so the
  file could go green by doing nothing. CI now installs combra when a `COMBRA_TOKEN`
  secret is present and sets `COMBRA_REQUIRED=1`; a new always-on test fails if
  combra is missing under that flag.

## [3.1.0] — 2026-08-18

Repairs the combra integration and makes a run's metric history machine-readable.

### Fixed
- **combra metrics were silently disabled.** `training/metrics.py` imported
  `angle_density_metrics_from_pooled`, `fid_from_features` and
  `fd_dinov2_from_features`, all removed in combra 0.5.0. The module-level
  `except ImportError` set `HAS_COMBRA = False` and training then printed
  *"combra is not installed; skipping"* — a false diagnosis that sent anyone
  debugging it to reinstall a package that was already present. Now imports
  `frechet_from_features` (one helper for both Fréchet metrics); combra >= 0.7.0
  restores `angle_density_metrics_from_pooled`.
- **The startup warning now tells the truth**, and an incompatible-but-present combra
  is fatal: training refuses to start rather than burning a run that will log no
  metrics. A genuinely absent combra still warns and continues.
- **`[combra]` installed a combra with no metric backends.** The extra pulled bare
  `combra`; since combra 0.5.0 the torch / `pytorch-fid` / `open-clip-torch` stack is
  behind `combra[metrics]`, so FID / CMMD / FD-DINOv2 would have returned `nan` even
  after the import fix. Now `combra[metrics] @ git+…`.
- **`stats.jsonl` metric rows were unreadable.** The combra metrics were written to
  their own row with unprefixed keys and a bare `kimg`, while `Progress/kimg` lived in
  the status rows. `combra.metrics.load_fid_by_kimg` needs `Metrics/combra_fid` and
  `Progress/kimg` on the *same* line, so it matched nothing and returned an empty dict
  for every edm2 run — silently, since it shape-filters rather than raising. The
  metrics row now carries both, plus `Progress/tick`, `wall_time` and `datetime`.
- **Angle extraction ran single-threaded.** `images_to_pooled_angles` was called
  without `workers`, leaving the most expensive part of an eval tick on one core.
  It now uses `cpu_count // gpus` (capped at 32).
- **The combra smoke test used random noise as its fixture.** Noise has no grain
  contours, so the angle pipeline extracted no vertices and the check failed with
  `attempt to get argmin of an empty sequence` — a message about nothing. The test now
  builds synthetic grain images, and combra >= 0.7.0 names the empty-density case.

### Changed
- **Metric keys lost the literal `10k`.** `combra_fid10k` was emitted whatever
  `--num-fid-samples` said, so any chart built from it was mislabelled. Keys are now
  bare — `combra_fid`, `combra_cmmd`, `combra_fd_dinov2` — and the count is logged
  once as `combra_num_fid_samples`.
- **The status rows in `stats.jsonl` carry `wall_time` and `datetime`**, as the
  logging contract requires.
- `requires-python` raised to **3.12** to match combra.

### Added
- `Metrics/combra_fid_best`, the running best FID.
- `tests/test_combra_contract.py` — asserts every combra symbol this repo imports
  actually exists. CPU-only, no GPU/dataset/network, so it runs in every CI job.
- `tests/test_smoke.py::test_combra_angle_metrics_run_offline` — the angle half needs
  no backbone, so it is exercised even with no network.
- `test_combra_import_guard` now asserts `HAS_COMBRA is True` whenever combra is
  importable. It previously asserted only that the flag was a `bool`, which passed
  either way — which is why the breakage above survived a whole combra release.

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
