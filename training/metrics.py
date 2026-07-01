# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Inline combra generative-quality metrics for EDM2 training.

Ported from DiffiT-v2/diffit/metrics.py: the reference/generated feature and
angle extraction is sharded across all GPU ranks and gathered to rank 0, which
computes the final distances -- so the combra metrics are computed on every rank
(each rank scores its own shard). combra is an optional dependency; the import is
guarded so training runs unchanged when it is not installed.
"""

import numpy as np
import torch
import torch.distributed as dist

# Optional combra integration. Guarded so training runs without it.
try:
    from combra.metrics import (
        angle_density_metrics_from_pooled as _combra_angle_metrics_from_pooled,
        cmmd_features as _combra_cmmd_features,
        cmmd_from_features as _combra_cmmd_from_features,
        compute_all_metrics as _combra_compute_all_metrics,
        fd_dinov2_features as _combra_fd_dinov2_features,
        fd_dinov2_from_features as _combra_fd_dinov2_from_features,
        fid_features as _combra_fid_features,
        fid_from_features as _combra_fid_from_features,
        images_to_pooled_angles as _combra_images_to_pooled_angles,
    )

    HAS_COMBRA = True
except ImportError:
    _combra_angle_metrics_from_pooled = _combra_images_to_pooled_angles = None
    _combra_fid_features = _combra_cmmd_features = _combra_fd_dinov2_features = None
    _combra_fid_from_features = _combra_cmmd_from_features = _combra_fd_dinov2_from_features = None
    _combra_compute_all_metrics = None
    HAS_COMBRA = False

# combra image-feature metrics carry their generated-sample count in the key,
# matching the SAN-v2 / DiffiT-v2 reference dashboards. Angle-density metrics keep
# their bare names.
_COMBRA_IMAGE_METRICS = ("fid", "cmmd", "fd_dinov2")
_COMBRA_IMAGE_RENAME = {"fid": "fid10k", "cmmd": "cmmd10k", "fd_dinov2": "fd_dinov2_10k"}

#----------------------------------------------------------------------------
# Feature extraction / distance dispatch (combra defaults, so the distributed
# result matches a single-GPU compute_all_metrics(image_metrics=True)).

def _combra_extract_features(name, images, device):
    if name == "fid":
        return _combra_fid_features(images, device=device).astype(np.float32)
    if name == "cmmd":
        return _combra_cmmd_features(images, device=device).astype(np.float32)
    return _combra_fd_dinov2_features(images, device=device).astype(np.float32)

def _combra_distance(name, ref_features, gen_features):
    if name == "fid":
        return _combra_fid_from_features(ref_features, gen_features)
    if name == "cmmd":
        return _combra_cmmd_from_features(ref_features, gen_features)
    return _combra_fd_dinov2_from_features(ref_features, gen_features)

#----------------------------------------------------------------------------
# Cross-rank gathering.

def _gather_feature_rows(local, device, rank, world_size):
    """Gather per-rank feature rows ``[n_i, D]`` to rank 0, concatenated in rank
    order (None on other ranks). Ranks may hold different ``n_i``, so each block is
    padded to the max before the collective gather and trimmed on rank 0."""
    if world_size == 1:
        return local

    t = torch.from_numpy(np.ascontiguousarray(local)).to(device)
    count = torch.tensor([t.shape[0]], device=device, dtype=torch.long)
    all_counts = [torch.zeros_like(count) for _ in range(world_size)]
    dist.all_gather(all_counts, count)

    max_count = max(c.item() for c in all_counts)
    if t.shape[0] < max_count:
        pad = torch.zeros(max_count - t.shape[0], *t.shape[1:], device=device, dtype=t.dtype)
        t = torch.cat([t, pad], 0)

    if rank == 0:
        gathered = [torch.zeros_like(t) for _ in range(world_size)]
        dist.gather(t, gathered, dst=0)
        rows = [g[:all_counts[i].item()].cpu().numpy() for i, g in enumerate(gathered)]
        return np.concatenate(rows, axis=0)
    dist.gather(t, dst=0)
    return None

def _gather_combra_gen_features(local_images, device, rank, world_size):
    """Each rank extracts the three image-feature sets from its own generated shard;
    rows are gathered to rank 0. ``{metric: [N, D]}`` on rank 0, ``{metric: None}``
    elsewhere (every rank still runs the collective for each metric, same order)."""
    return {
        name: _gather_feature_rows(
            _combra_extract_features(name, local_images, device), device, rank, world_size
        )
        for name in _COMBRA_IMAGE_METRICS
    }

def _gather_pooled_angles(local_images, device, rank, world_size):
    """Each rank extracts its shard's pooled vertex angles; the 1-D arrays are
    gathered to rank 0 (concatenated). Pooled angles from disjoint shards
    concatenate directly, so the rank-0 histogram matches a single-GPU
    ``images_to_pooled_angles`` over the full set."""
    local = np.asarray(_combra_images_to_pooled_angles(local_images), np.float32).reshape(-1, 1)
    gathered = _gather_feature_rows(local, device, rank, world_size)
    return gathered.reshape(-1) if gathered is not None else None

def precompute_combra_reference(local_ref_images, device, rank, world_size):
    """All-ranks: extract pooled angles + three feature sets from this rank's
    reference shard and gather to rank 0. ``{"angles": [M], "feat": {name: [N, D]}}``
    on rank 0 (None elsewhere). Called once before the training loop; the cached
    result is reused every eval tick."""
    angles = _gather_pooled_angles(local_ref_images, device, rank, world_size)
    feat = {
        name: _gather_feature_rows(
            _combra_extract_features(name, local_ref_images, device), device, rank, world_size
        )
        for name in _COMBRA_IMAGE_METRICS
    }
    if rank == 0:
        return {"angles": angles, "feat": feat}
    return None

def _combra_distributed_metrics(combra_ref, gen_angles, gen_feats):
    """Rank-0 combra metrics from already-gathered inputs: angle-density /
    Gaussian-fit metrics from pooled angles + image-feature distances from gathered
    features. Equivalent to ``compute_all_metrics(image_metrics=True)`` but sharded."""
    metrics = dict(_combra_angle_metrics_from_pooled(combra_ref["angles"], gen_angles))
    for name in _COMBRA_IMAGE_METRICS:
        metrics[name] = _combra_distance(name, combra_ref["feat"][name], gen_feats[name])
    return metrics

#----------------------------------------------------------------------------
# EDM-specific helpers: turn dataset/model outputs into RGB (NHWC uint8) batches.

def _decode_to_nhwc_uint8(encoder, latents):
    """final latents -> RGB uint8 NHWC numpy (combra's expected image layout)."""
    px = encoder.decode(latents)  # uint8 NCHW
    return px.permute(0, 2, 3, 1).contiguous().cpu().numpy()

@torch.inference_mode()
def load_reference_shard(dataset_obj, encoder, count, batch, device, rank, world_size):
    """Load this rank's shard of the real reference set as RGB uint8 NHWC.

    The first ``min(count, len)`` dataset items are split round-robin across ranks
    (``idx % world_size == rank``). Each raw item is passed through the encoder
    (``encode_latents`` then ``decode``) so the reference lives in the same
    VAE-decoded pixel space as the generated samples."""
    encoder.init(device)
    n_total = min(int(count), len(dataset_obj))
    my_idx = [i for i in range(n_total) if i % world_size == rank]
    chunks, buf = [], []
    for i in my_idx:
        img, _ = dataset_obj[i]
        buf.append(torch.as_tensor(img))
        if len(buf) == batch:
            chunks.append(_decode_to_nhwc_uint8(encoder, encoder.encode_latents(torch.stack(buf).to(device))))
            buf = []
    if buf:
        chunks.append(_decode_to_nhwc_uint8(encoder, encoder.encode_latents(torch.stack(buf).to(device))))
    if chunks:
        return np.concatenate(chunks, 0)
    return np.zeros((0, 1, 1, 3), dtype=np.uint8)

@torch.inference_mode()
def generate_fake_shard(net, encoder, gnet, num_samples, batch, device, rank, world_size,
                        *, sampler, num_steps, guidance, seed):
    """Generate this rank's shard of fakes and return them as RGB uint8 NHWC.

    ``num_samples`` is the global count; each rank produces its ``1/world_size``
    slice with a distinct seed so the union is deterministic and non-overlapping."""
    from training.samplers import sample as sampler_sample

    encoder.init(device)
    n_local = (int(num_samples) + world_size - 1 - rank) // world_size  # ceil-split
    g = torch.Generator(device=device).manual_seed(int(seed) + rank)
    C, R = net.img_channels, net.img_resolution
    chunks, got = [], 0
    while got < n_local:
        b = min(batch, n_local - got)
        noise = torch.randn(b, C, R, R, device=device, generator=g)
        labels = None
        if net.label_dim > 0:
            idx = torch.randint(net.label_dim, (b,), device=device, generator=g)
            labels = torch.eye(net.label_dim, device=device)[idx]
        latents = sampler_sample(net, noise, labels=labels, gnet=gnet,
                                 sampler=sampler, num_steps=num_steps, guidance=guidance)
        chunks.append(_decode_to_nhwc_uint8(encoder, latents))
        got += b
    if chunks:
        return np.concatenate(chunks, 0)
    return np.zeros((0, 1, 1, 3), dtype=np.uint8)

@torch.inference_mode()
def compute_combra_metrics(net, encoder, combra_ref, num_samples, batch, device, rank, world_size,
                           *, sampler, num_steps, guidance=1, gnet=None, seed=0, log_fn=print):
    """Generate fakes on every rank, extract+gather combra features, and compute the
    ``combra_*`` metrics on rank 0. Returns a metric dict on rank 0, ``None`` else."""
    local_fakes = generate_fake_shard(net, encoder, gnet, num_samples, batch, device,
                                       rank, world_size, sampler=sampler,
                                       num_steps=num_steps, guidance=guidance, seed=seed)
    gen_feats = _gather_combra_gen_features(local_fakes, device, rank, world_size)
    gen_angles = _gather_pooled_angles(local_fakes, device, rank, world_size)
    if rank != 0:
        return None
    metrics = {}
    try:
        raw = _combra_distributed_metrics(combra_ref, gen_angles, gen_feats)
        for k, v in raw.items():
            key = _COMBRA_IMAGE_RENAME.get(k, k)
            metrics[f"combra_{key}"] = float(v)
    except Exception as e:  # noqa: BLE001 -- never let metrics crash training
        log_fn(f"  combra metrics failed: {e}")
    return metrics

#----------------------------------------------------------------------------

def combra_smoke_test(ref_images, device, log_fn=print):
    """Verify combra metrics actually compute (not just import) before training.
    Runs the real pipeline once on a tiny reference slice; raises on failure."""
    sample = ref_images[: min(4, len(ref_images))]
    try:
        metrics = _combra_compute_all_metrics(
            sample, sample, device=device, image_metrics=True, reference_cache={},
        )
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"combra metrics smoke test failed to run: {e}") from e
    bad = sorted(k for k, v in metrics.items() if not np.isfinite(v))
    if bad:
        raise RuntimeError(
            f"combra metrics smoke test produced non-finite values for {bad} -- a "
            "metric backend or optional dependency is missing/broken. Fix the install "
            "or pass --no-combra-metrics."
        )
    log_fn("combra metrics smoke test passed")

#----------------------------------------------------------------------------
