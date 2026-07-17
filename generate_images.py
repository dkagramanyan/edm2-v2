# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate images per class in the wc_cv angle-pipeline HDF5 layout (§4).

Class-batch mode (``--classes`` + ``--samples-per-class``) writes per-rank
``shards/rank_NNN.h5`` in the RankH5Writer layout, merged into ``<desc>.h5``.
``--gpus N`` self-spawns per-GPU workers (no torchrun). The legacy ``--seeds`` mode
(one class per run) is kept for ad-hoc sampling.
"""

import json
import os
import pickle
import re
import socket
import warnings

import click
import numpy as np
import PIL.Image
import torch
import tqdm

import dnnlib
from torch_utils import distributed as dist
from training import checkpoint as ckpt
from training.h5_writer import RankH5Writer, merge_shards

warnings.filterwarnings('ignore', '`resume_download` is deprecated')
warnings.filterwarnings('ignore', 'You are using `torch.load` with `weights_only=False`')
warnings.filterwarnings('ignore', '1Torch was not compiled with flash attention')

#----------------------------------------------------------------------------
# Published NVIDIA reference checkpoints (external pickled modules). The WC-Co
# workflow uses local .pt inference snapshots instead; these presets are kept for
# reproducing the upstream ImageNet numbers.

model_root = 'https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions'

config_presets = {
    'edm2-img512-s-fid':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.130.pkl'),   # fid = 2.56
    'edm2-img512-m-fid':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.100.pkl'),   # fid = 2.25
    'edm2-img512-xxl-fid':      dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.070.pkl'), # fid = 1.91
    'edm2-img64-s-fid':         dnnlib.EasyDict(net=f'{model_root}/edm2-img64-s-1073741-0.075.pkl'),    # fid = 1.58
}

#----------------------------------------------------------------------------
# Samplers live in training/samplers.py so training-time eval and generation share
# one implementation. `edm_sampler` is kept importable for backward compatibility.

from training.samplers import edm_sampler, sample as sampler_dispatch  # noqa: E402, F401

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows a different random seed per sample.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------

def parse_int_list(s):
    """Parse '1,2,5-10' -> [1,2,5,6,7,8,9,10]."""
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in str(s).split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------
# Network loading. Local `.pt` inference snapshots use the v2 state-dict format
# (§3); external / legacy `.pkl` checkpoints keep the pickled-module loader.

def load_network(path, device, verbose=True):
    if isinstance(path, str) and path.endswith('.pt'):
        net, encoder, meta = ckpt.load_inference_snapshot(path, device, verbose=verbose)
        return net, encoder, meta
    with dnnlib.util.open_url(path, verbose=(verbose and dist.get_rank() == 0)) as f:
        data = pickle.load(f)
    net = data['ema'].to(device)
    encoder = data.get('encoder', None)
    if encoder is None:
        encoder = dnnlib.util.construct_class_by_name(class_name='training.encoders.StandardRGBEncoder')
    meta = dict(n_classes=int(net.label_dim), class_names=None)
    return net, encoder, meta

#----------------------------------------------------------------------------
# Resolve a --classes spec (indices, ranges, and/or names) against the checkpoint's
# class metadata. Names require the checkpoint to carry class_names (§5).

def resolve_classes(spec, n_classes, class_names):
    name_to_idx = {name: i for i, name in enumerate(class_names)} if class_names else {}
    out = []
    for part in str(spec).split(','):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r'(\d+)-(\d+)', part)
        if m:
            out.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        elif part.isdigit():
            out.append(int(part))
        elif part in name_to_idx:
            out.append(name_to_idx[part])
        else:
            raise click.ClickException(
                f'--classes: {part!r} is neither an index nor a known class name '
                f'({sorted(name_to_idx) or "checkpoint carries no class_names"})')
    for c in out:
        if not (0 <= c < max(n_classes, 1)):
            raise click.ClickException(f'--classes: index {c} out of range for a {n_classes}-class model')
    return sorted(dict.fromkeys(out))

#----------------------------------------------------------------------------
# Class-batch generation into HDF5 shards / PNG dirs.

@torch.inference_mode()
def run_class_generation(net, encoder, gnet, *, classes, samples_per_class, base_seed, outdir,
                         desc, save_mode, batch_gpu, class_names, device, sampler, sampler_kwargs,
                         verbose=True):
    encoder.init(device)
    rank, world_size = dist.get_rank(), dist.get_world_size()

    # Deterministic work list: seed = base + class*samples_per_class + idx (§4).
    items = [(c, i, base_seed + c * samples_per_class + i) for c in classes for i in range(samples_per_class)]
    rank_items = items[rank::world_size]
    by_class = {}
    for c, i, s in rank_items:
        by_class.setdefault(c, []).append((i, s))

    resolution = None
    writer = None
    dir_manifest = {}
    if save_mode == 'hdf5':
        # Probe the output resolution with a 1-sample decode so the shard can be
        # preallocated before the main loop.
        probe_labels = torch.eye(net.label_dim, device=device)[[classes[0]]] if net.label_dim > 0 else None
        probe = encoder.decode(sampler_dispatch(
            net=net, noise=torch.randn(1, net.img_channels, net.img_resolution, net.img_resolution, device=device),
            labels=probe_labels, gnet=gnet, sampler=sampler, **sampler_kwargs))
        resolution = int(probe.shape[-1])
        os.makedirs(os.path.join(outdir, 'shards'), exist_ok=True)
        writer = RankH5Writer(
            os.path.join(outdir, 'shards', f'rank_{rank:03d}.h5'), rank,
            {c: len(v) for c, v in by_class.items()}, resolution, channels=int(probe.shape[1]),
            class_names=class_names)
    else:
        os.makedirs(outdir, exist_ok=True)

    for c, work in by_class.items():
        for start in range(0, len(work), batch_gpu):
            chunk = work[start:start + batch_gpu]
            idxs = [i for i, _s in chunk]
            seeds = [s for _i, s in chunk]
            rnd = StackedRandomGenerator(device, seeds)
            noise = rnd.randn([len(chunk), net.img_channels, net.img_resolution, net.img_resolution], device=device)
            labels = None
            if net.label_dim > 0:
                labels = torch.eye(net.label_dim, device=device)[[c] * len(chunk)]
            latents = sampler_dispatch(net=net, noise=noise, labels=labels, gnet=gnet,
                                       randn_like=rnd.randn_like, sampler=sampler, **sampler_kwargs)
            images = encoder.decode(latents).permute(0, 2, 3, 1).contiguous().cpu().numpy()  # NHWC uint8
            if save_mode == 'hdf5':
                writer.write(c, images, seeds, idxs)
            else:
                cdir = os.path.join(outdir, f'class_{c}')
                os.makedirs(cdir, exist_ok=True)
                for img, i, s in zip(images, idxs, seeds):
                    PIL.Image.fromarray(img, 'RGB').save(os.path.join(cdir, f'idx_{i:06d}_seed_{s}.png'))
                dir_manifest[f'class_{c}'] = class_names[c] if class_names else str(c)

    if save_mode == 'hdf5':
        writer.close()

    torch.distributed.barrier()
    if rank == 0 and save_mode == 'hdf5':
        shard_paths = sorted(
            os.path.join(outdir, 'shards', p) for p in os.listdir(os.path.join(outdir, 'shards'))
            if re.fullmatch(r'rank_\d+\.h5', p))
        out_path = os.path.join(outdir, f'{desc}.h5')
        counts = merge_shards(shard_paths, out_path, class_names=class_names)
        if verbose:
            dist.print0(f'Merged {sum(counts.values())} images into {out_path} ({counts})')
    if rank == 0 and save_mode == 'dir':
        with open(os.path.join(outdir, 'classes.json'), 'w') as f:
            json.dump(dir_manifest, f, indent=2)
    torch.distributed.barrier()

#----------------------------------------------------------------------------
# Legacy per-seed generation (one class per run). Yields batches of images; kept for
# sample_images.py and ad-hoc PNG dumps.

def generate_images(net, gnet=None, encoder=None, outdir=None, seeds=range(16, 24), class_idx=None,
                    max_batch_size=32, encoder_batch_size=4, verbose=True, device=torch.device('cuda'),
                    sampler='dpm++', **sampler_kwargs):
    if dist.get_rank() != 0:
        torch.distributed.barrier()
    if isinstance(net, str):
        if verbose:
            dist.print0(f'Loading main network from {net} ...')
        net, enc, _meta = load_network(net, device, verbose=verbose)
        if encoder is None:
            encoder = enc
    assert net is not None
    if isinstance(gnet, str):
        gnet, _e, _m = load_network(gnet, device, verbose=verbose)
    if gnet is None:
        gnet = net
    assert encoder is not None
    encoder.init(device)
    if encoder_batch_size is not None and hasattr(encoder, 'batch_size'):
        encoder.batch_size = encoder_batch_size
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    num_batches = max((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1, 1) * dist.get_world_size()
    rank_batches = np.array_split(np.arange(len(seeds)), num_batches)[dist.get_rank() :: dist.get_world_size()]
    if verbose:
        dist.print0(f'Generating {len(seeds)} images...')

    class ImageIterable:
        def __len__(self):
            return len(rank_batches)

        def __iter__(self):
            for batch_idx, indices in enumerate(rank_batches):
                r = dnnlib.EasyDict(images=None, labels=None, noise=None, batch_idx=batch_idx, num_batches=len(rank_batches), indices=indices)
                r.seeds = [seeds[idx] for idx in indices]
                if len(r.seeds) > 0:
                    rnd = StackedRandomGenerator(device, r.seeds)
                    r.noise = rnd.randn([len(r.seeds), net.img_channels, net.img_resolution, net.img_resolution], device=device)
                    r.labels = None
                    if net.label_dim > 0:
                        r.labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[len(r.seeds)], device=device)]
                        if class_idx is not None:
                            r.labels[:, :] = 0
                            r.labels[:, class_idx] = 1
                    latents = sampler_dispatch(net=net, noise=r.noise, labels=r.labels, gnet=gnet,
                        randn_like=rnd.randn_like, sampler=sampler, **sampler_kwargs)
                    r.images = encoder.decode(latents)
                    if outdir is not None:
                        for seed, image in zip(r.seeds, r.images.permute(0, 2, 3, 1).cpu().numpy()):
                            os.makedirs(outdir, exist_ok=True)
                            PIL.Image.fromarray(image, 'RGB').save(os.path.join(outdir, f'{seed:06d}.png'))
                torch.distributed.barrier()
                yield r

    return ImageIterable()

#----------------------------------------------------------------------------
# Per-rank entry point spawned by --gpus (no torchrun needed).

def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    port = s.getsockname()[1]
    s.close()
    return port

def _gen_subprocess_fn(rank, opts, num_gpus, master_port):
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(num_gpus)
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = str(master_port)
    dist.init()
    _run(opts)

def _run(opts):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net, encoder, meta = load_network(opts.net, device, verbose=opts.verbose)
    gnet = None
    if opts.guidance not in (None, 1) and opts.gnet is not None:
        gnet, _e, _m = load_network(opts.gnet, device, verbose=opts.verbose)
    n_classes = int(meta.get('n_classes', net.label_dim))
    class_names = meta.get('class_names')
    sampler_kwargs = dict(num_steps=opts.num_steps, sigma_min=opts.sigma_min, sigma_max=opts.sigma_max,
                          rho=opts.rho, guidance=(opts.guidance or 1), S_churn=opts.S_churn,
                          S_min=opts.S_min, S_max=opts.S_max, S_noise=opts.S_noise)
    classes = resolve_classes(opts.classes, n_classes, class_names) if opts.classes is not None else [opts.class_idx or 0]
    run_class_generation(net, encoder, gnet, classes=classes, samples_per_class=opts.samples_per_class,
                         base_seed=opts.seed, outdir=opts.outdir, desc=opts.desc, save_mode=opts.save_mode,
                         batch_gpu=opts.batch_gpu, class_names=class_names, device=device,
                         sampler=opts.sampler, sampler_kwargs=sampler_kwargs, verbose=opts.verbose)

#----------------------------------------------------------------------------
# Command line interface.

@click.command()
@click.option('--net', '--network', 'net',  help='Network checkpoint (.pt inference snapshot or legacy .pkl)', metavar='PATH|URL', type=str, default=None)
@click.option('--gnet',                     help='Guiding network', metavar='PATH|URL',                 type=str, default=None)
@click.option('--preset',                   help='External reference preset', metavar='STR',            type=str, default=None)
@click.option('--outdir',                   help='Where to save the output', metavar='DIR',             type=str, required=True)
@click.option('--desc',                     help='Merged HDF5 basename (<desc>.h5)', metavar='STR',     type=str, default=None)
@click.option('--classes',                  help='Classes to generate: indices/ranges or names (e.g. 0,1,4-6 or Ultra_Co11)', metavar='LIST', type=str, default=None)
@click.option('--samples-per-class',        help='Samples per class', metavar='INT',                    type=click.IntRange(min=1), default=1000, show_default=True)
@click.option('--save-mode',                help='Output layout', type=click.Choice(['hdf5', 'dir']), default='hdf5', show_default=True)
@click.option('--gpus',                     help='Number of GPUs to spawn (no torchrun needed)', metavar='INT', type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--batch-gpu',                help='Per-GPU batch size', metavar='INT',                   type=click.IntRange(min=1), default=32, show_default=True)
@click.option('--seed',                     help='Base random seed', metavar='INT',                     type=int, default=0, show_default=True)
# Legacy per-seed mode.
@click.option('--seeds',                    help='[legacy] explicit seed list (one class per run)', metavar='LIST', type=parse_int_list, default=None)
@click.option('--class', 'class_idx',       help='[legacy] class index for --seeds mode', metavar='INT', type=click.IntRange(min=0), default=None)

@click.option('--sampler',                  help='Reverse-diffusion sampler', type=click.Choice(['edm', 'euler', 'ddim', 'dpm++']), default='dpm++', show_default=True)
@click.option('--steps', 'num_steps',       help='Number of sampling steps', metavar='INT',             type=click.IntRange(min=1), default=25, show_default=True)
@click.option('--sigma_min',                help='Lowest noise level', metavar='FLOAT',                 type=click.FloatRange(min=0, min_open=True), default=0.002, show_default=True)
@click.option('--sigma_max',                help='Highest noise level', metavar='FLOAT',                type=click.FloatRange(min=0, min_open=True), default=80, show_default=True)
@click.option('--rho',                      help='Time step exponent', metavar='FLOAT',                 type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--guidance',                 help='Guidance strength  [default: 1; no guidance]', metavar='FLOAT', type=float, default=None)
@click.option('--S_churn', 'S_churn',       help='Stochasticity strength', metavar='FLOAT',             type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',           help='Stoch. min noise level', metavar='FLOAT',             type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',           help='Stoch. max noise level', metavar='FLOAT',             type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',       help='Stoch. noise inflation', metavar='FLOAT',             type=float, default=1, show_default=True)

def cmdline(preset, **opts):
    """Generate images per class into the wc_cv angle-pipeline HDF5 layout.

    \b
    # 1000 samples each of classes 0,1,2 across 2 GPUs into <desc>.h5
    edm2-gen-images --network=run/edm2-snapshot-002000-0.100-inference.pt \\
        --outdir=out --classes=0,1,2 --samples-per-class=1000 \\
        --gpus=2 --batch-gpu=32 --save-mode=hdf5
    """
    opts = dnnlib.EasyDict(opts)
    if preset is not None:
        if preset not in config_presets:
            raise click.ClickException(f'Invalid configuration preset "{preset}"')
        for key, value in config_presets[preset].items():
            if opts.get(key, None) is None:
                opts[key] = value
    if opts.net is None:
        raise click.ClickException('Please specify either --preset or --net/--network')
    if opts.guidance in (None, 1):
        opts.guidance, opts.gnet = 1, None
    elif opts.gnet is None:
        raise click.ClickException('Please specify --gnet when using guidance')
    opts.desc = opts.desc or os.path.splitext(os.path.basename(opts.net))[0]
    opts.verbose = True

    # Legacy per-seed mode: one class per run, PNG dump.
    if opts.seeds is not None and opts.classes is None:
        dist.init()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        image_iter = generate_images(
            net=opts.net, gnet=opts.gnet, outdir=opts.outdir, seeds=opts.seeds, class_idx=opts.class_idx,
            max_batch_size=opts.batch_gpu, device=device, sampler=opts.sampler, num_steps=opts.num_steps,
            sigma_min=opts.sigma_min, sigma_max=opts.sigma_max, rho=opts.rho, guidance=opts.guidance,
            S_churn=opts.S_churn, S_min=opts.S_min, S_max=opts.S_max, S_noise=opts.S_noise)
        for _r in tqdm.tqdm(image_iter, unit='batch', disable=(dist.get_rank() != 0)):
            pass
        return

    if opts.classes is None:
        opts.classes = '0'  # default: single class 0

    # Class-batch mode: self-spawn per-GPU workers.
    gpus = opts.pop('gpus')
    torch.multiprocessing.set_start_method('spawn', force=True)
    master_port = _free_port()
    if gpus == 1:
        _gen_subprocess_fn(rank=0, opts=opts, num_gpus=1, master_port=master_port)
    else:
        torch.multiprocessing.spawn(_gen_subprocess_fn, args=(opts, gpus, master_port), nprocs=gpus)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------
