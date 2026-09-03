# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.

"""CPU smoke tests: model forward contract, samplers, and the combra import
guard. No GPU, dataset, or external weights required."""

import numpy as np
import pytest
import torch

from training import samplers
from training.networks_edm2 import Precond


class _StubDenoiser(torch.nn.Module):
    """Minimal denoiser with the net(x, sigma, labels) contract the samplers use."""
    img_channels = 3
    img_resolution = 8
    label_dim = 4

    def forward(self, x, sigma, labels=None, **kw):
        return x * 0.5  # arbitrary but well-defined


def _noise(n=2):
    return torch.randn(n, _StubDenoiser.img_channels, _StubDenoiser.img_resolution, _StubDenoiser.img_resolution)


def test_sampler_names():
    assert set(samplers.SAMPLER_NAMES) == {"edm", "euler", "ddim", "dpm++"}


def test_each_sampler_runs_and_preserves_shape():
    net = _StubDenoiser()
    noise = _noise()
    labels = torch.eye(net.label_dim)[torch.randint(net.label_dim, (noise.shape[0],))]
    for name in samplers.SAMPLER_NAMES:
        out = samplers.sample(net, noise, labels=labels, sampler=name, num_steps=4)
        assert out.shape == noise.shape
        assert torch.isfinite(out).all()


def test_ddim_is_first_order_euler():
    net = _StubDenoiser()
    noise = _noise()
    a = samplers.sample(net, noise, sampler="ddim", num_steps=5)
    b = samplers.sample(net, noise, sampler="euler", num_steps=5)
    assert torch.allclose(a, b)


def test_precond_forward_contract():
    net = Precond(img_resolution=16, img_channels=4, label_dim=10, use_fp16=False,
                  model_channels=8, channel_mult=[1, 2], num_blocks=1, attn_resolutions=[8])
    x = torch.randn(2, 4, 16, 16)
    sigma = torch.rand(2) + 0.1
    labels = torch.eye(10)[torch.randint(10, (2,))]
    denoised = net(x, sigma, labels)
    assert denoised.shape == x.shape
    denoised, logvar = net(x, sigma, labels, return_logvar=True)
    assert denoised.shape == x.shape
    assert logvar.shape == (2, 1, 1, 1)


def test_combra_import_guard():
    # Not "is it a bool" -- that passed whichever way it went, which is precisely how
    # a real breakage (combra 0.5.0 moving three symbols) survived a whole release.
    # If combra is importable at all, the integration must be live.
    import importlib.util

    from training import metrics

    if importlib.util.find_spec("combra") is None:
        assert metrics.HAS_COMBRA is False
        return
    assert metrics.HAS_COMBRA is True, (
        f"combra is installed but this repo cannot use it: {metrics.COMBRA_IMPORT_ERROR}"
    )


def _grain_image(seed, size=96, n=10):
    """A small synthetic microstructure: filled polygons on a light ground.

    combra's angle pipeline measures vertex angles on grain contours, so its input
    has to *have* contours. Random noise has none — it yields an empty angle density
    and the smoke test fails for a reason that has nothing to do with the install.
    """
    import cv2

    rng = np.random.default_rng(seed)
    img = np.full((size, size), 255, np.uint8)
    for _ in range(n):
        centre = rng.integers(20, size - 20, size=2)
        pts = centre + rng.integers(-15, 15, size=(int(rng.integers(3, 7)), 2))
        cv2.fillPoly(img, [pts.astype(np.int32)], int(rng.integers(0, 120)))
    return np.stack([img] * 3, axis=-1)


def test_combra_smoke_when_available():
    from training import metrics
    if not metrics.HAS_COMBRA:
        return  # combra optional; nothing to check
    # A 6-parameter bimodal fit needs enough vertices to constrain it. Four
    # 96px/10-polygon images give only ~70 vertex angles, and the second mode then
    # fits as a ~200 deg-wide pedestal -- combra reports that as nan rather than
    # dividing by a phantom. 256px x 80 polygons gives ~740 angles, stable across
    # seeds. (Real reference images are not affected: one 768px micrograph already
    # yields ~300 angles and fits cleanly.)
    imgs = np.stack([_grain_image(i, size=256, n=80) for i in range(4)])
    try:
        metrics.combra_smoke_test(imgs, torch.device("cpu"), log_fn=lambda *a: None)
    except RuntimeError as e:
        # The smoke test is *meant* to fail loudly when an image-feature backend is
        # unusable -- that is its whole job before a training run. On a CI box with no
        # network the CLIP / DINOv2 weights cannot be fetched, which is an environment
        # limitation rather than a defect, so skip on exactly that. Any other failure
        # (including an empty angle density) is a real one and still fails the test.
        if "non-finite" not in str(e):
            raise
        pytest.skip(f"image-feature backends unavailable offline: {e}")


def test_combra_angle_metrics_run_offline():
    """The angle-density half needs no backbone, so it must work with no network."""
    from training import metrics
    if not metrics.HAS_COMBRA:
        return
    from combra.metrics import angle_density_metrics_from_pooled, images_to_pooled_angles

    # A 6-parameter bimodal fit needs enough vertices to constrain it. Four
    # 96px/10-polygon images give only ~70 vertex angles, and the second mode then
    # fits as a ~200 deg-wide pedestal -- combra reports that as nan rather than
    # dividing by a phantom. 256px x 80 polygons gives ~740 angles, stable across
    # seeds. (Real reference images are not affected: one 768px micrograph already
    # yields ~300 angles and fits cleanly.)
    ref = np.stack([_grain_image(i, size=256, n=80) for i in range(4)])
    gen = np.stack([_grain_image(100 + i, size=256, n=80) for i in range(4)])
    out = angle_density_metrics_from_pooled(
        images_to_pooled_angles(ref), images_to_pooled_angles(gen)
    )
    for key in ("w1", "w2", "circular_w1", "circular_w2", "mu1", "sigma1", "pi"):
        assert np.isfinite(out[key]), f"{key} is not finite"
