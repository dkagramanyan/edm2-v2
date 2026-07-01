# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.

"""CPU smoke tests: model forward contract, samplers, and the combra import
guard. No GPU, dataset, or external weights required."""

import numpy as np
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
    from training import metrics
    assert isinstance(metrics.HAS_COMBRA, bool)


def test_combra_smoke_when_available():
    from training import metrics
    if not metrics.HAS_COMBRA:
        return  # combra optional; nothing to check
    imgs = np.random.randint(0, 256, size=(4, 16, 16, 3), dtype=np.uint8)
    metrics.combra_smoke_test(imgs, torch.device("cpu"), log_fn=lambda *a: None)
