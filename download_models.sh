#!/bin/bash
# Pure-shell prefetch of every pretrained weight EDM2 training + the combra
# metrics need, straight into the caches the libraries look in -- no GPU
# required. Run on a node WITH internet (e.g. a login node); the weights cache
# under $HOME, shared with the offline compute nodes, so the training jobs then
# need no network.
#
# Three cache families are populated:
#   * torch.hub cache       -- via plain wget/curl/git (combra InceptionV3 +
#     DINOv2 backbones).
#   * HuggingFace hub cache -- the combra CLIP weights, laid out by hand.
#   * dnnlib cache          -- the Stability VAE for the latent presets, via
#     `hf` / huggingface-cli (falls back to `python download_models.py` if the
#     CLI is absent). load_stability_vae reads it from
#     dnnlib.make_cache_dir_path('diffusers') = $DNNLIB_CACHE_DIR/diffusers, else
#     $HOME/.cache/dnnlib/diffusers -- NOT the standard HuggingFace cache.
#
# URLs and on-disk filenames are pinned to what the installed libraries expect
# (pytorch-fid, open_clip 'openai' via the HuggingFace hub, torch.hub dinov2,
# diffusers AutoencoderKL).
#
# Usage:
#   bash download_models.sh                      # caches under $HOME/.cache (defaults)
#   MODEL_CACHE=/shared/team/caches bash download_models.sh
# With MODEL_CACHE set, point the jobs at it:
#   export TORCH_HOME=$MODEL_CACHE/torch HF_HOME=$MODEL_CACHE/huggingface \
#          DNNLIB_CACHE_DIR=$MODEL_CACHE/dnnlib
set -u

MODEL_CACHE="${MODEL_CACHE:-$HOME/.cache}"
HUB_CKPT="${MODEL_CACHE}/torch/hub/checkpoints"   # pytorch-fid + dinov2 weights
HUB_DIR="${MODEL_CACHE}/torch/hub"                # torch.hub repo code (dinov2)
HF_HUB="${MODEL_CACHE}/huggingface/hub"           # HuggingFace hub cache (open_clip weights)
VAE_DIR="${MODEL_CACHE}/dnnlib/diffusers"         # dnnlib.make_cache_dir_path('diffusers')
mkdir -p "$HUB_CKPT" "$HUB_DIR" "$HF_HUB" "$VAE_DIR"

if ! command -v wget >/dev/null 2>&1 && ! command -v curl >/dev/null 2>&1; then
    echo "ERROR: need wget or curl on PATH." >&2
    exit 1
fi

status=0
fetch() {  # fetch <url> <dest>
    local url="$1" dest="$2"
    if [[ -s "$dest" ]]; then
        echo "  exists: ${dest##*/}"
        return 0
    fi
    echo "  downloading: ${url##*/}"
    if command -v wget >/dev/null 2>&1; then
        wget -c -O "$dest" "$url" || { echo "  FAILED: $url"; rm -f "$dest"; status=1; return 1; }
    else
        curl -fL -o "$dest" "$url" || { echo "  FAILED: $url"; rm -f "$dest"; status=1; return 1; }
    fi
}

echo "Caching pretrained models under: $MODEL_CACHE"

echo; echo "[1/3] InceptionV3 FID weights (combra fid) -> $HUB_CKPT"
fetch "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth" "$HUB_CKPT/pt_inception-2015-12-05-6726825d.pth"

echo; echo "[2/3] CLIP ViT-L-14-336 'openai' (combra cmmd) -> $HF_HUB"
# open_clip loads the 'openai' tag from the HF repo timm/vit_large_patch14_clip_336.openai
# (huggingface_hub is one of its dependencies), so lay the file out as the HF hub cache
# does: blobs/<sha256>, snapshots/<rev>/<file> -> blob, refs/main = <rev>.
CLIP_REV="81e38efc4637de5023b10e75a7f9bd1c6fa6b010"
CLIP_SHA="fbc415c3d0d7b79faed8f5ccfb740c32b7c4f5ffe7283b851f89c6231c01a8e0"
CLIP_REPO_DIR="$HF_HUB/models--timm--vit_large_patch14_clip_336.openai"
mkdir -p "$CLIP_REPO_DIR/blobs" "$CLIP_REPO_DIR/refs" "$CLIP_REPO_DIR/snapshots/$CLIP_REV"
fetch "https://huggingface.co/timm/vit_large_patch14_clip_336.openai/resolve/$CLIP_REV/open_clip_model.safetensors" "$CLIP_REPO_DIR/blobs/$CLIP_SHA" \
    && ln -sfn "../../blobs/$CLIP_SHA" "$CLIP_REPO_DIR/snapshots/$CLIP_REV/open_clip_model.safetensors" \
    && printf '%s' "$CLIP_REV" > "$CLIP_REPO_DIR/refs/main"

echo; echo "[3/3] DINOv2 dinov2_vitl14 (combra fd_dinov2) -> $HUB_CKPT"
fetch "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth" "$HUB_CKPT/dinov2_vitl14_pretrain.pth"
# torch.hub also needs the dinov2 model code (it normally fetches the repo itself).
if [[ -d "$HUB_DIR/facebookresearch_dinov2_main" ]]; then
    echo "  exists: facebookresearch_dinov2_main/"
elif command -v git >/dev/null 2>&1; then
    echo "  cloning facebookresearch/dinov2"
    git clone --depth 1 https://github.com/facebookresearch/dinov2 "$HUB_DIR/facebookresearch_dinov2_main" \
        || { echo "  FAILED: git clone dinov2"; status=1; }
else
    echo "  SKIPPED dinov2 repo: git not on PATH"; status=1
fi

# --- Stability VAE (dnnlib cache) ------------------------------------------
# load_stability_vae calls from_pretrained(..., cache_dir=<dnnlib>/diffusers), so the
# HF snapshots/blobs layout goes there, not into the HuggingFace cache. Only the
# config and the safetensors weights are fetched (diffusers prefers safetensors).
# Newer huggingface_hub ships `hf` and turns `huggingface-cli` into a deprecation
# stub that exits non-zero, so prefer `hf`.
echo; echo "[VAE] stabilityai/sd-vae-ft-mse (latent presets) -> $VAE_DIR"
if command -v hf >/dev/null 2>&1; then
    hf_cmd="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
    hf_cmd="huggingface-cli"
else
    hf_cmd=""
fi
if [[ -n "$hf_cmd" ]]; then
    echo "  downloading: stabilityai/sd-vae-ft-mse"
    "$hf_cmd" download stabilityai/sd-vae-ft-mse config.json diffusion_pytorch_model.safetensors \
        --cache-dir "$VAE_DIR" >/dev/null \
        || { echo "  FAILED: stabilityai/sd-vae-ft-mse"; status=1; }
else
    echo "  hf / huggingface-cli not found -- fetch the VAE with:"
    echo "      DNNLIB_CACHE_DIR=$MODEL_CACHE/dnnlib python download_models.py --no-combra"
fi

echo
if [[ $status -eq 0 ]]; then
    echo "Done. All weights cached under $MODEL_CACHE."
else
    echo "Some downloads failed (see above)."
fi
exit $status
