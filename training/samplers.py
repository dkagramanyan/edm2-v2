# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Reverse-diffusion samplers for EDM2, all operating in the native EDM
sigma-space on the `net(x, sigma, labels)` denoiser.

The default `edm` sampler is the 2nd-order Heun sampler from the EDM paper
(identical to the one in `generate_images.edm_sampler`). `euler`, `ddim` and
`dpm++` are added so the model can be sampled with a few different samplers like
DiffiT-v2 -- they are inference-only and do not touch the training/update logic.

DDIM: for the EDM probability-flow ODE the deterministic first-order update is
exactly the (eta=0) DDIM step, so `ddim` is the first-order deterministic
sampler. `dpm++` is DPM-Solver++(2M) in log-sigma space (Lu et al., 2022),
following the standard k-diffusion formulation.
"""

import numpy as np
import torch

#----------------------------------------------------------------------------

SAMPLER_NAMES = ['edm', 'euler', 'ddim', 'dpm++']

#----------------------------------------------------------------------------
# Shared helpers.

def _make_denoise(net, labels, gnet, guidance, dtype):
    # Guided denoiser closure, matching generate_images.edm_sampler.
    def denoise(x, t):
        Dx = net(x, t, labels).to(dtype)
        if guidance == 1 or gnet is None or gnet is net:
            return Dx
        ref_Dx = gnet(x, t, labels).to(dtype)
        return ref_Dx.lerp(Dx, guidance)
    return denoise

def _sigma_schedule(num_steps, sigma_min, sigma_max, rho, dtype, device):
    # Karras rho time-step discretization with a trailing t_N = 0.
    step_indices = torch.arange(num_steps, dtype=dtype, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) *
               (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0

#----------------------------------------------------------------------------
# EDM 2nd-order Heun sampler (default), extended with classifier-free guidance.
# Kept identical to generate_images.edm_sampler.

def edm_sampler(
    net, noise, labels=None, gnet=None,
    num_steps=32, sigma_min=0.002, sigma_max=80, rho=7, guidance=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    dtype=torch.float32, randn_like=torch.randn_like,
):
    denoise = _make_denoise(net, labels, gnet, guidance, dtype)
    t_steps = _sigma_schedule(num_steps, sigma_min, sigma_max, rho, dtype, noise.device)

    x_next = noise.to(dtype) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        # Increase noise temporarily.
        if S_churn > 0 and S_min <= t_cur <= S_max:
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1)
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        else:
            t_hat = t_cur
            x_hat = x_cur
        # Euler step.
        d_cur = (x_hat - denoise(x_hat, t_hat)) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur
        # Apply 2nd order correction.
        if i < num_steps - 1:
            d_prime = (x_next - denoise(x_next, t_next)) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next

#----------------------------------------------------------------------------
# 1st-order deterministic Euler / DDIM sampler.

def euler_sampler(
    net, noise, labels=None, gnet=None,
    num_steps=32, sigma_min=0.002, sigma_max=80, rho=7, guidance=1,
    dtype=torch.float32, randn_like=torch.randn_like, **kwargs,
):
    denoise = _make_denoise(net, labels, gnet, guidance, dtype)
    t_steps = _sigma_schedule(num_steps, sigma_min, sigma_max, rho, dtype, noise.device)
    x_next = noise.to(dtype) * t_steps[0]
    for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
        d_cur = (x_next - denoise(x_next, t_cur)) / t_cur
        x_next = x_next + (t_next - t_cur) * d_cur
    return x_next

# For the EDM probability-flow ODE the eta=0 DDIM update is the first-order
# deterministic step, i.e. the Euler sampler above.
ddim_sampler = euler_sampler

#----------------------------------------------------------------------------
# DPM-Solver++(2M) in log-sigma space (Lu et al., 2022), k-diffusion style.

def dpmpp_2m_sampler(
    net, noise, labels=None, gnet=None,
    num_steps=32, sigma_min=0.002, sigma_max=80, rho=7, guidance=1,
    dtype=torch.float32, randn_like=torch.randn_like, **kwargs,
):
    denoise = _make_denoise(net, labels, gnet, guidance, dtype)
    sigmas = _sigma_schedule(num_steps, sigma_min, sigma_max, rho, dtype, noise.device)

    def t_fn(sigma):
        return -sigma.log()

    x = noise.to(dtype) * sigmas[0]
    old_denoised = None
    for i in range(num_steps):
        s_cur, s_next = sigmas[i], sigmas[i + 1]
        denoised = denoise(x, s_cur)
        t, t_next = t_fn(s_cur), t_fn(s_next)
        h = t_next - t
        if old_denoised is None or s_next == 0:
            x = (s_next / s_cur) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (s_next / s_cur) * x - (-h).expm1() * denoised_d
        old_denoised = denoised
    return x

#----------------------------------------------------------------------------

_SAMPLERS = {
    'edm': edm_sampler,
    'euler': euler_sampler,
    'ddim': ddim_sampler,
    'dpm++': dpmpp_2m_sampler,
}

def sample(net, noise, labels=None, gnet=None, randn_like=torch.randn_like, *,
           sampler='edm', **sampler_kwargs):
    """Dispatch to the chosen sampler. `sampler` in SAMPLER_NAMES."""
    if sampler not in _SAMPLERS:
        raise ValueError(f'Unknown sampler {sampler!r}; choose from {SAMPLER_NAMES}')
    return _SAMPLERS[sampler](net, noise, labels=labels, gnet=gnet,
                              randn_like=randn_like, **sampler_kwargs)

#----------------------------------------------------------------------------
