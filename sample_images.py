# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Bulk-sample a trained EDM2 model into a single ``.npz`` of uint8 NHWC images
for FID-style evaluation, distributed across ranks with ``torchrun``.

Example:
    torchrun --standalone --nproc_per_node=4 sample_images.py \\
        --net=training-runs/00000-.../network-snapshot-....pkl \\
        --outdir=samples/512 --num-samples=50000 --batch=16 --sampler=dpm++ --steps=25
"""

import os

import click
import numpy as np
import torch
import torch.distributed
import tqdm

from generate_images import generate_images, parse_int_list
from torch_utils import distributed as dist

#----------------------------------------------------------------------------

def _gather_images(local_nhwc, device):
    """All-gather variable-length uint8 NHWC image blocks to rank 0."""
    if dist.get_world_size() == 1:
        return local_nhwc
    gathered = [None for _ in range(dist.get_world_size())]
    torch.distributed.all_gather_object(gathered, local_nhwc)
    if dist.get_rank() == 0:
        blocks = [b for b in gathered if b is not None and len(b) > 0]
        return np.concatenate(blocks, axis=0) if blocks else local_nhwc
    return None

#----------------------------------------------------------------------------

@click.command()
@click.option('--net',          help='Network pickle filename', metavar='PATH|URL',     type=str, required=True)
@click.option('--gnet',         help='Guiding network pickle filename', metavar='PATH|URL', type=str, default=None)
@click.option('--outdir',       help='Where to save the output .npz', metavar='DIR',    type=str, required=True)
@click.option('--num-samples',  help='Number of images to generate', metavar='INT',     type=click.IntRange(min=1), default=50000, show_default=True)
@click.option('--batch', 'max_batch_size', help='Max batch size per GPU', metavar='INT', type=click.IntRange(min=1), default=32, show_default=True)
@click.option('--class', 'class_idx', help='Class label  [default: random]', metavar='INT', type=click.IntRange(min=0), default=None)
@click.option('--seed',         help='Base random seed', metavar='INT',                 type=int, default=0, show_default=True)
@click.option('--sampler',      help='Reverse-diffusion sampler', type=click.Choice(['edm', 'euler', 'ddim', 'dpm++']), default='dpm++', show_default=True)
@click.option('--steps', 'num_steps', help='Number of sampling steps', metavar='INT',   type=click.IntRange(min=1), default=25, show_default=True)
@click.option('--sigma_min',    help='Lowest noise level', metavar='FLOAT',             type=click.FloatRange(min=0, min_open=True), default=0.002, show_default=True)
@click.option('--sigma_max',    help='Highest noise level', metavar='FLOAT',            type=click.FloatRange(min=0, min_open=True), default=80, show_default=True)
@click.option('--rho',          help='Time step exponent', metavar='FLOAT',             type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--guidance',     help='Guidance strength  [default: 1; no guidance]', metavar='FLOAT', type=float, default=None)
def cmdline(net, gnet, outdir, num_samples, max_batch_size, class_idx, seed, sampler,
            num_steps, sigma_min, sigma_max, rho, guidance):
    """Generate ``--num-samples`` images and save them to ``outdir/samples.npz``."""
    dist.init()
    device = torch.device('cuda')
    if guidance is None or guidance == 1:
        guidance, gnet = 1, None
    elif gnet is None:
        raise click.ClickException('Please specify --gnet when using guidance')

    seeds = parse_int_list(f'{seed}-{seed + num_samples - 1}')
    image_iter = generate_images(
        net=net, gnet=gnet, outdir=None, seeds=seeds, class_idx=class_idx,
        max_batch_size=max_batch_size, device=device, sampler=sampler,
        num_steps=num_steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho, guidance=guidance,
    )
    local = []
    for r in tqdm.tqdm(image_iter, unit='batch', disable=(dist.get_rank() != 0)):
        if r.images is not None:
            local.append(r.images.permute(0, 2, 3, 1).contiguous().cpu().numpy())
    local = np.concatenate(local, axis=0) if local else np.zeros((0, 1, 1, 3), np.uint8)

    images = _gather_images(local, device)
    if dist.get_rank() == 0:
        os.makedirs(outdir, exist_ok=True)
        images = images[:num_samples]
        out = os.path.join(outdir, f'samples_{len(images)}x{images.shape[1]}x{images.shape[2]}x{images.shape[3]}.npz')
        np.savez(out, images)
        dist.print0(f'Saved {len(images)} images to {out}')
    torch.distributed.barrier()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------
