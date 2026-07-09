# Changelog

All notable changes to this fork (`edm2`) are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

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
- **`--sampler` / `--sampling-steps` renamed to `--eval-sampler` /
  `--eval-sampling-steps`** in `train_edm2.py`, matching DiffiT-v2's training CLI. The
  generation scripts (`generate_images.py`, `sample_images.py`) keep `--sampler` /
  `--steps`. The training sbatch scripts were updated.
