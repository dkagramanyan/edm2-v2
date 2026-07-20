# TODO

Problems surfaced while building the `run-edm2` skill (offline, single RTX 3090).

## Codebase issues

- [ ] **`--max-images N` produces a single-class dataset.** `edm2-prepare-data
  convert` takes images in class-folder-alphabetical order, so a small cap grabs
  only the first class. The resulting model is 1-class and `edm2-gen-images
  --classes=1,2…` then errors *"index N out of range for a 1-class model"*.
  Fix: sample `--max-images` proportionally across classes (or at least warn when
  the capped set covers fewer classes than the source).

- [ ] **Cryptic error when a latent preset can't reach the VAE.** Offline (or with
  an incomplete HF cache), `edm2-img256/512/1024-*` die deep in
  `training/encoders.py:load_stability_vae` with *"does not appear to have a file
  named config.json"* / `LocalEntryNotFoundError`. Fix: catch this and emit an
  actionable message ("VAE not cached — run `edm2-download-models` with network,
  or use an RGB `edm2-img64-*` preset").

- [ ] **NCCL cleanup warning at exit.** Every single-GPU run prints
  *"destroy_process_group() was not called before program exit … leak resources"*
  (+ *"Guessing device ID based on global rank"*). Harmless but noisy — call
  `torch.distributed.destroy_process_group()` on shutdown.

## Environment / docs (not code bugs)

- [ ] **`pip install -e .` fails offline** with *"Could not find a version that
  satisfies setuptools>=61"* because build isolation tries to fetch setuptools.
  Workaround: `--no-build-isolation`. Consider documenting in README for
  air-gapped/cluster installs.

- [ ] **Incomplete VAE cache on this box.** `~/.cache/huggingface/hub/
  models--stabilityai--sd-vae-ft-mse` has blobs but no `snapshots/`/`config.json`,
  so latent training can't run offline. Re-fetch once online with
  `edm2-download-models`.
