#!/usr/bin/env bash
# smoke.sh -- drive the edm2 CLI end-to-end on ONE GPU, fully offline.
#
# Exercises the real pipeline PRs touch: prepare-data -> train (dry-run) ->
# train (real, RGB img64) -> gen-images (HDF5) -> pytest. Each step checks an
# exit code AND an on-disk artifact, so a silent regression fails loudly.
#
# RGB img64 is used on purpose: it is the ONLY preset that runs with no network.
# The 256/512/1024 latent presets need the Stability VAE, whose HF cache here is
# incomplete (see Gotchas in SKILL.md) -- they require a network fetch first.
#
# Usage:
#   bash smoke.sh                       # synthetic 2-class dataset (portable)
#   bash smoke.sh --source /path/to/imgs   # real image folder (class subdirs)
#   EDM2_ENV=edm2 WORKDIR=/tmp/edm2-smoke bash smoke.sh
#
# Env overrides:
#   EDM2_ENV   conda env name with edm2 installed (default: edm2)
#   EDM2_PY    explicit python interpreter (overrides EDM2_ENV discovery)
#   WORKDIR    scratch dir for the run (default: a fresh mktemp dir)
#   SOURCE     image folder to convert (default: synthetic; --source sets it too)
set -euo pipefail

# repo root = three levels up from this script (edm2/.claude/skills/run-edm2/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# --- locate the edm2 python -------------------------------------------------
EDM2_ENV="${EDM2_ENV:-edm2}"
if [ -z "${EDM2_PY:-}" ]; then
  CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")"
  EDM2_PY="$CONDA_BASE/envs/$EDM2_ENV/bin/python"
fi
[ -x "$EDM2_PY" ] || { echo "FAIL: python not found at $EDM2_PY (set EDM2_PY or EDM2_ENV)"; exit 1; }
BIN="$(dirname "$EDM2_PY")"
export PATH="$BIN:$PATH"
export HF_HUB_OFFLINE=1          # never reach out to the network in the smoke

# --- args -------------------------------------------------------------------
SOURCE="${SOURCE:-}"
while [ $# -gt 0 ]; do
  case "$1" in
    --source) SOURCE="$2"; shift 2;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done

WORKDIR="${WORKDIR:-$(mktemp -d /tmp/edm2-smoke.XXXXXX)}"
mkdir -p "$WORKDIR"
echo "== edm2 smoke =="
echo "   python : $EDM2_PY"
echo "   workdir: $WORKDIR"
"$EDM2_PY" -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('   torch  :', torch.__version__, 'cuda', torch.version.cuda)"

pass() { echo "PASS: $1"; }
have() { [ -e "$1" ] || { echo "FAIL: missing artifact $1"; exit 1; }; }

# --- 0. synthesize a dataset if none given ----------------------------------
# Two class folders of small noise PNGs -> portable, no private data needed.
if [ -z "$SOURCE" ]; then
  SOURCE="$WORKDIR/src"
  "$EDM2_PY" - "$SOURCE" <<'PY'
import sys, os, numpy as np
from PIL import Image
root = sys.argv[1]
for c in ("class_a", "class_b"):
    d = os.path.join(root, c); os.makedirs(d, exist_ok=True)
    for i in range(16):
        arr = (np.random.rand(80, 80, 3) * 255).astype("uint8")
        Image.fromarray(arr).save(os.path.join(d, f"{i:03d}.png"))
print("synthesized", root)
PY
  pass "synthetic dataset created ($SOURCE)"
fi

# --- 1. prepare-data convert -> RGB training zip ----------------------------
ZIP="$WORKDIR/data_64.zip"
edm2-prepare-data convert --source="$SOURCE" --dest="$ZIP" \
  --resolution=64x64 --transform=center-crop-dhariwal >/dev/null
have "$ZIP"
"$EDM2_PY" - "$ZIP" <<'PY'
import sys, zipfile, json
z = zipfile.ZipFile(sys.argv[1]); d = json.loads(z.read("dataset.json"))
assert d["labels"], "no labels in dataset.json"
print("   zip:", len(d["labels"]), "images, class_names:", d.get("class_names"))
PY
pass "prepare-data convert"

# --- 2. train dry-run (resolve config, no compute) --------------------------
edm2-train --outdir="$WORKDIR/dry" --cfg=edm2-img64-s --data="$ZIP" \
  --gpus=1 --batch-gpu=8 --kimg=1 -n >/dev/null
pass "train --dry-run (config resolves)"

# --- 3. real training, 1 kimg, RGB img64, no combra/VAE ---------------------
edm2-train --outdir="$WORKDIR/runs" --cfg=edm2-img64-s --data="$ZIP" \
  --gpus=1 --batch-gpu=8 --tick=1 --snap=1 --kimg=1 \
  --combra-metrics=False --num-fid-samples=0 --workers=2 >"$WORKDIR/train.log" 2>&1 \
  || { echo "FAIL: training errored"; tail -30 "$WORKDIR/train.log"; exit 1; }
RUN="$(ls -d "$WORKDIR"/runs/*/ | head -1)"
have "$RUN/reals.png"
have "$RUN/fakes000001.png"
SNAP="$(ls "$RUN"/edm2-snapshot-*-inference.pt | tail -1)"
have "$SNAP"
have "$RUN/stats.jsonl"
pass "train 1 kimg -> snapshot + fakes/reals grids ($RUN)"

# --- 4. generate images from the snapshot -> HDF5 ---------------------------
edm2-gen-images --network="$SNAP" --outdir="$WORKDIR/gen" --classes=0 \
  --samples-per-class=4 --gpus=1 --batch-gpu=4 --save-mode=hdf5 \
  --sampler=dpm++ --steps=8 >"$WORKDIR/gen.log" 2>&1 \
  || { echo "FAIL: generation errored"; tail -30 "$WORKDIR/gen.log"; exit 1; }
H5="$(ls "$WORKDIR"/gen/*.h5 | head -1)"
have "$H5"
"$EDM2_PY" - "$H5" <<'PY'
import sys, h5py
h = h5py.File(sys.argv[1])
assert h.attrs["format"] == "generated_images_shard"
assert "class_0" in h and h["class_0"]["images"].shape[1:] == (64, 64, 3)
print("   h5:", {k: h[k]["images"].shape for k in h})
PY
pass "gen-images -> HDF5 (documented schema)"

# --- 5. test suite (CPU) ----------------------------------------------------
( cd "$REPO" && "$EDM2_PY" -m pytest tests -q ) \
  || { echo "FAIL: pytest"; exit 1; }
pass "pytest tests/"

echo
echo "ALL SMOKE STEPS PASSED"
echo "artifacts under: $WORKDIR"
