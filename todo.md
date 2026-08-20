# TODO

Problems surfaced while building the `run-edm2` skill (offline, single RTX 3090).

## Codebase issues

- [x] **`--max-images N` produces a single-class dataset.** Resolved 2026-08-20:
  `dataset_tool.stratified_subset` keys on the label each image will be written
  with and picks round-robin, so a cap of N over C classes gives each class N/C;
  it warns when the cap is smaller than the class count and leaves a single-class
  source alone. Applied to the folder and zip openers. Reproduced first: a 3-class
  source capped at 6 gave `class_names ['a','b','c']` but labels `[0,0,0,0,0,0]`;
  now `[0,0,1,1,2,2]`. **The same bug was in StyleSwin** and was fixed there too.
  `tests/test_dataset_tool.py` pins it.

  Original report: `edm2-prepare-data`
  `convert` takes images in class-folder-alphabetical order, so a small cap grabs
  only the first class. The resulting model is 1-class and `edm2-gen-images
  --classes=1,2…` then errors *"index N out of range for a 1-class model"*.

- [x] **Cryptic error when a latent preset can't reach the VAE.** Resolved
  2026-08-20: `load_stability_vae` catches the diffusers failure and re-raises with
  the cache path it actually used, the note that this is **not**
  `~/.cache/huggingface`, and the two ways forward (`edm2-download-models`, or an RGB
  `edm2-img64-*` preset). The bare `except:` it hid behind is now `except Exception`.

  Original report: Offline (or with
  an incomplete HF cache), `edm2-img256/512/1024-*` die deep in
  `training/encoders.py:load_stability_vae` with *"does not appear to have a file
  named config.json"* / `LocalEntryNotFoundError`.

- [x] **NCCL cleanup warning at exit.** Resolved 2026-08-20:
  `torch_utils/distributed.init()` registers an `atexit` handler that destroys the
  group — but only when that call created it, so a caller that set one up itself
  keeps ownership. Reproduced the warning on a single GPU and confirmed it gone.

## Environment / docs (not code bugs)

- [x] **`pip install -e .` fails offline.** Resolved 2026-08-20: documented in the
  README install section alongside `edm2-download-models` — use
  `pip install -e . --no-build-isolation` on an air-gapped node.

- [x] **VAE cache.** Resolved 2026-08-20, and the diagnosis in the original report
  was looking at the wrong directory. `~/.cache/huggingface/.../sd-vae-ft-mse` is in
  fact **complete** (config.json + a 320 MB safetensors) — but edm2 never reads it:
  `load_stability_vae` sets `HF_HOME` to `dnnlib.make_cache_dir_path('diffusers')`,
  i.e. `~/.cache/dnnlib/diffusers`, which was empty. Ran `edm2-download-models` to
  populate it and verified an offline `load_stability_vae()` succeeds
  (`HF_HUB_OFFLINE=1`, 83.6 M params). The cache location is now in the README,
  because "the VAE is cached" and "edm2 can see the VAE" are not the same thing.

  Original report: `~/.cache/huggingface/hub/
  models--stabilityai--sd-vae-ft-mse` has blobs but no `snapshots/`/`config.json`,
  so latent training can't run offline.
