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

# Optional combra integration. Guarded so training runs without it.
#
# Only what this module actually calls is imported: the sharded harness itself now
# lives in `combra.metrics.distributed`, so the feature extractors and distance
# helpers it needs are combra's own dependency, covered by combra's tests rather
# than re-listed here.
try:
    from combra.metrics import self_test as _combra_self_test
    from combra.metrics.distributed import (
        distributed_metrics as _combra_distributed_metrics_impl,
        gather_generated as _combra_gather_generated,
        precompute_reference as _combra_precompute_reference,
    )

    HAS_COMBRA = True
    COMBRA_IMPORT_ERROR = None
except ImportError as _combra_exc:
    _combra_self_test = _combra_precompute_reference = None
    _combra_gather_generated = _combra_gather_pooled_angles = None
    _combra_distributed_metrics_impl = None
    HAS_COMBRA = False
    # Keep the reason. "combra is not installed" is the wrong diagnosis when combra
    # IS installed but has moved a symbol -- that misdirection is exactly how this
    # integration stayed broken for a whole combra release.
    COMBRA_IMPORT_ERROR = _combra_exc

# combra metric keys are bare: combra_fid, combra_cmmd, combra_fd_dinov2. They used
# to carry a "10k" suffix that stayed 10k whatever --num-fid-samples said, so every
# chart built from them was mislabelled. The sample count is logged once instead, as
# Metrics/combra_num_fid_samples.
#----------------------------------------------------------------------------
# The shard -> extract -> gather -> distance harness now lives in combra
# (`combra.metrics.distributed`), so all four model repos share one implementation
# instead of four copies that had drifted apart. What stays here is the part that is
# genuinely EDM2-specific: turning latents into uint8 RGB, and generating a shard.

precompute_combra_reference = _combra_precompute_reference

#----------------------------------------------------------------------------

# EDM-specific helpers: turn dataset/model outputs into RGB (NHWC uint8) batches.

def _decode_to_nhwc_uint8(encoder, latents):
    """final latents -> RGB uint8 NHWC numpy (combra's expected image layout)."""
    px = encoder.decode(latents)  # uint8 NCHW
    return px.permute(0, 2, 3, 1).contiguous().cpu().numpy()

@torch.inference_mode()
def load_reference_shard(dataset_obj, count, batch, device, rank, world_size, *, seed=0):
    """Load this rank's shard of the real reference set as **raw** RGB uint8 NHWC.

    Reference images are the raw dataset pixels -- never flip-augmented and never VAE
    round-tripped (§6), so a VAE quality gap shows up in the metric instead of being
    hidden. When ``count`` caps the reference below the dataset size the subset is a
    **seeded random** draw (never the first N -- dataset zips are class-sorted, so a
    first-N slice is class-biased). The chosen indices are split round-robin across
    ranks (``pos % world_size == rank``)."""
    n = len(dataset_obj)
    n_total = min(int(count), n)
    if n_total < n:
        idx = np.sort(np.random.RandomState(int(seed) & 0x7fffffff).choice(n, n_total, replace=False))
    else:
        idx = np.arange(n)
    my_idx = [int(i) for pos, i in enumerate(idx) if pos % world_size == rank]
    chunks = []
    for i in my_idx:
        img, _ = dataset_obj[i]  # uint8 CHW
        chunks.append(np.asarray(img)[np.newaxis].transpose(0, 2, 3, 1))
    if chunks:
        return np.concatenate(chunks, 0).astype(np.uint8)
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
    gen_feats, gen_angles = _combra_gather_generated(local_fakes, device, rank, world_size)
    if rank != 0:
        return None
    metrics = {}
    try:
        # device= so the CMMD reduction runs where the features were extracted; left
        # unset it resolves independently and a CPU extraction reduces on CUDA.
        raw = _combra_distributed_metrics_impl(combra_ref, gen_angles, gen_feats, device=device)
        for k, v in raw.items():
            metrics[f"combra_{k}"] = float(v)
        metrics["combra_num_fid_samples"] = float(num_samples)
    except Exception as e:  # noqa: BLE001 -- never let metrics crash training
        log_fn(f"  combra metrics failed: {e}")
    return metrics

#----------------------------------------------------------------------------

def combra_smoke_test(ref_images, device, log_fn=print):
    """Verify the combra metrics actually *compute* before training, not just import.

    Delegates to ``combra.metrics.self_test``, which since combra 0.8 takes
    ``strict=True`` (require the image-feature metrics to be finite, rather than
    tolerating ``nan``) and ``images=`` (score a real reference slice against itself).
    That is what this repo's private copy used to do by hand; models-API §6 always
    specified one shared implementation, and now there is one.
    """
    sample = ref_images[: min(4, len(ref_images))]
    try:
        metrics = _combra_self_test(
            images=sample, device=device, image_metrics=True, strict=True
        )
    except RuntimeError:
        raise
    except Exception as e:  # noqa: BLE001 -- surface any backend failure the same way
        raise RuntimeError(f"combra metrics smoke test failed to run: {e}") from e
    log_fn(f"combra metrics smoke test passed ({len(metrics)} metrics computed)")

#----------------------------------------------------------------------------
