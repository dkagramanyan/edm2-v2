# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Train diffusion models according to the EDM2 recipe from the paper
"Analyzing and Improving the Training Dynamics of Diffusion Models"."""

import json
import os
import socket
import warnings

import click
import torch

import dnnlib
import training.training_loop
from torch_utils import distributed as dist

warnings.filterwarnings('ignore', 'You are using `torch.load` with `weights_only=False`')

#----------------------------------------------------------------------------
# Configuration presets.

config_presets = {
    'edm2-img256-xs':   dnnlib.EasyDict(duration=2048<<20, batch=2048, channels=128, lr=0.0120, decay=70000, dropout=0.00, P_mean=-0.4, P_std=1.0),
    'edm2-img256-s':    dnnlib.EasyDict(duration=2048<<20, batch=2048, channels=192, lr=0.0100, decay=70000, dropout=0.00, P_mean=-0.4, P_std=1.0),
    'edm2-img256-m':    dnnlib.EasyDict(duration=2048<<20, batch=2048, channels=256, lr=0.0090, decay=70000, dropout=0.10, P_mean=-0.4, P_std=1.0),
    'edm2-img512-xxs':  dnnlib.EasyDict(duration=2048<<20, batch=2048, channels=64,  lr=0.0170, decay=70000, dropout=0.00, P_mean=-0.4, P_std=1.0),
    'edm2-img512-xs':   dnnlib.EasyDict(duration=2048<<20, batch=2048, channels=128, lr=0.0120, decay=70000, dropout=0.00, P_mean=-0.4, P_std=1.0),
    'edm2-img512-s':    dnnlib.EasyDict(duration=2048<<20, batch=2048, channels=192, lr=0.0100, decay=70000, dropout=0.00, P_mean=-0.4, P_std=1.0),
    'edm2-img512-m':    dnnlib.EasyDict(duration=2048<<20, batch=2048, channels=256, lr=0.0090, decay=70000, dropout=0.10, P_mean=-0.4, P_std=1.0),
    'edm2-img512-l':    dnnlib.EasyDict(duration=1792<<20, batch=2048, channels=320, lr=0.0080, decay=70000, dropout=0.10, P_mean=-0.4, P_std=1.0),
    'edm2-img512-xl':   dnnlib.EasyDict(duration=1280<<20, batch=2048, channels=384, lr=0.0070, decay=70000, dropout=0.10, P_mean=-0.4, P_std=1.0),
    'edm2-img512-xxl':  dnnlib.EasyDict(duration=896<<20,  batch=2048, channels=448, lr=0.0065, decay=70000, dropout=0.10, P_mean=-0.4, P_std=1.0),
    'edm2-img64-xs':    dnnlib.EasyDict(duration=1024<<20, batch=2048, channels=128, lr=0.0120, decay=35000, dropout=0.00, P_mean=-0.8, P_std=1.6),
    'edm2-img64-s':     dnnlib.EasyDict(duration=1024<<20, batch=2048, channels=192, lr=0.0100, decay=35000, dropout=0.00, P_mean=-0.8, P_std=1.6),
    'edm2-img64-m':     dnnlib.EasyDict(duration=2048<<20, batch=2048, channels=256, lr=0.0090, decay=35000, dropout=0.10, P_mean=-0.8, P_std=1.6),
    'edm2-img64-l':     dnnlib.EasyDict(duration=1024<<20, batch=2048, channels=320, lr=0.0080, decay=35000, dropout=0.10, P_mean=-0.8, P_std=1.6),
    'edm2-img64-xl':    dnnlib.EasyDict(duration=640<<20,  batch=2048, channels=384, lr=0.0070, decay=35000, dropout=0.10, P_mean=-0.8, P_std=1.6),
    'edm2-img1024-s':   dnnlib.EasyDict(duration=1024<<20, batch=1024, channels=192, lr=0.0080, decay=70000, dropout=0.10, P_mean=-0.4, P_std=1.0),
    'edm2-img1024-m':   dnnlib.EasyDict(duration=1024<<20, batch=1024, channels=256, lr=0.0070, decay=70000, dropout=0.10, P_mean=-0.4, P_std=1.0),
}

#----------------------------------------------------------------------------
# Setup arguments for training.training_loop.training_loop().

def setup_training_config(preset='edm2-img512-s', **opts):
    opts = dnnlib.EasyDict(opts)
    c = dnnlib.EasyDict()

    # Preset.
    if preset not in config_presets:
        raise click.ClickException(f'Invalid configuration preset "{preset}"')
    for key, value in config_presets[preset].items():
        if opts.get(key, None) is None:
            opts[key] = value

    # Dataset.
    c.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=opts.data, use_labels=opts.get('cond', True))
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
        dataset_channels = dataset_obj.num_channels
        if c.dataset_kwargs.use_labels and not dataset_obj.has_labels:
            raise click.ClickException('--cond=True, but no labels found in the dataset')
        del dataset_obj # conserve memory
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

    # Encoder. A raw 3-channel RGB dataset can train either in pixel space
    # (StandardRGBEncoder) or, DiffiT-style, in VAE latent space with the encode
    # done inline every step (StabilityVAEOnTheFlyEncoder) -- no dataset pre-
    # encoding needed. An 8-channel dataset is already VAE-encoded offline.
    # Default: latent for every preset except the pixel-space edm2-img64 family.
    latent = opts.get('latent', None)
    if latent is None:
        latent = not preset.startswith('edm2-img64')
    if dataset_channels == 3:
        cls = 'StabilityVAEOnTheFlyEncoder' if latent else 'StandardRGBEncoder'
        c.encoder_kwargs = dnnlib.EasyDict(class_name=f'training.encoders.{cls}')
    elif dataset_channels == 8:
        c.encoder_kwargs = dnnlib.EasyDict(class_name='training.encoders.StabilityVAEEncoder')
    else:
        raise click.ClickException(f'--data: Unsupported channel count {dataset_channels}')

    # Hyperparameters.
    c.update(total_nimg=opts.duration, batch_size=opts.batch)
    c.network_kwargs = dnnlib.EasyDict(class_name='training.networks_edm2.Precond', model_channels=opts.channels, dropout=opts.dropout)
    c.loss_kwargs = dnnlib.EasyDict(class_name='training.training_loop.EDM2Loss', P_mean=opts.P_mean, P_std=opts.P_std)
    c.lr_kwargs = dnnlib.EasyDict(func_name='training.training_loop.learning_rate_schedule', ref_lr=opts.lr, ref_batches=opts.decay)

    # Performance-related options.
    c.batch_gpu = opts.get('batch_gpu', 0) or None
    c.network_kwargs.use_fp16 = opts.get('fp16', True)
    c.loss_scaling = opts.get('ls', 1)
    c.cudnn_benchmark = opts.get('bench', True)

    # I/O-related options.
    c.status_nimg = opts.get('status', 0) or None
    c.snapshot_nimg = opts.get('snapshot', 0) or None
    c.checkpoint_nimg = opts.get('checkpoint', 0) or None
    c.seed = opts.get('seed', 0)

    # Inline evaluation (combra metrics + eval-time sampler), DiffiT-v2 style.
    c.combra_metrics = opts.get('combra_metrics', True)
    c.num_fid_samples = opts.get('num_fid_samples', 10000)
    c.combra_ref_count = opts.get('combra_ref_count', 0) or None
    c.eval_sampler = opts.get('eval_sampler', 'dpm++')
    c.eval_num_steps = opts.get('eval_sampling_steps', 25)
    c.eval_guidance = opts.get('guidance', 1.0)
    return c

#----------------------------------------------------------------------------
# Print training configuration.

def print_training_config(run_dir, c, num_gpus):
    dist.print0()
    dist.print0('Training config:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {run_dir}')
    dist.print0(f'Dataset path:            {c.dataset_kwargs.path}')
    dist.print0(f'Class-conditional:       {c.dataset_kwargs.use_labels}')
    dist.print0(f'Encoder:                 {c.encoder_kwargs.class_name.rsplit(".", 1)[-1]}')
    dist.print0(f'Number of GPUs:          {num_gpus}')
    dist.print0(f'Total batch size:        {c.batch_size}')
    dist.print0(f'Mixed-precision:         {c.network_kwargs.use_fp16}')
    dist.print0()

#----------------------------------------------------------------------------
# Launch training.

def launch_training(run_dir, c):
    if dist.get_rank() == 0 and not os.path.isdir(run_dir):
        dist.print0('Creating output directory...')
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)

    torch.distributed.barrier()
    dnnlib.util.Logger(file_name=os.path.join(run_dir, 'log.txt'), file_mode='a', should_flush=True)
    training.training_loop.training_loop(run_dir=run_dir, **c)

#----------------------------------------------------------------------------
# Per-rank entry point spawned by --gpus (DiffiT-style; no torchrun needed).
# Sets the env vars torch_utils.distributed.init() reads, then trains.

def subprocess_fn(rank, c, run_dir, num_gpus, master_port):
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(num_gpus)
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = str(master_port)
    dist.init()
    launch_training(run_dir=run_dir, c=c)

def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    port = s.getsockname()[1]
    s.close()
    return port

#----------------------------------------------------------------------------
# Parse an integer with optional power-of-two suffix:
# 'Ki' = kibi = 2^10
# 'Mi' = mebi = 2^20
# 'Gi' = gibi = 2^30

def parse_nimg(s):
    if isinstance(s, int):
        return s
    if s.endswith('Ki'):
        return int(s[:-2]) << 10
    if s.endswith('Mi'):
        return int(s[:-2]) << 20
    if s.endswith('Gi'):
        return int(s[:-2]) << 30
    return int(s)

#----------------------------------------------------------------------------
# Command line interface.

@click.command()

# Main options.
@click.option('--outdir',           help='Where to save the results', metavar='DIR',            type=str, required=True)
@click.option('--data',             help='Path to the dataset', metavar='ZIP|DIR',              type=str, required=True)
@click.option('--gpus',             help='Number of GPUs to spawn (no torchrun needed)', metavar='INT', type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--cond',             help='Train class-conditional model', metavar='BOOL',       type=bool, default=True, show_default=True)
@click.option('--cfg', '--preset', 'preset', help='Configuration preset', metavar='STR',        type=str, default='edm2-img512-s', show_default=True)
@click.option('--latent/--pixel',   'latent', help='VAE latent-space (inline encode) vs raw pixel space; default: latent for all presets except edm2-img64', default=None)

# Hyperparameters.
@click.option('--duration',         help='Training duration', metavar='NIMG',                   type=parse_nimg, default=None)
@click.option('--batch',            help='Total batch size', metavar='NIMG',                    type=parse_nimg, default=None)
@click.option('--channels',         help='Channel multiplier', metavar='INT',                   type=click.IntRange(min=64), default=None)
@click.option('--dropout',          help='Dropout probability', metavar='FLOAT',                type=click.FloatRange(min=0, max=1), default=None)
@click.option('--P_mean', 'P_mean', help='Noise level mean', metavar='FLOAT',                   type=float, default=None)
@click.option('--P_std', 'P_std',   help='Noise level standard deviation', metavar='FLOAT',     type=click.FloatRange(min=0, min_open=True), default=None)
@click.option('--lr',               help='Learning rate max. (alpha_ref)', metavar='FLOAT',     type=click.FloatRange(min=0, min_open=True), default=None)
@click.option('--decay',            help='Learning rate decay (t_ref)', metavar='BATCHES',      type=click.FloatRange(min=0), default=None)

# Performance-related options.
@click.option('--batch-gpu',        help='Limit batch size per GPU', metavar='NIMG',            type=parse_nimg, default=0, show_default=True)
@click.option('--fp16',             help='Enable mixed-precision training', metavar='BOOL',     type=bool, default=True, show_default=True)
@click.option('--ls',               help='Loss scaling', metavar='FLOAT',                       type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--bench',            help='Enable cuDNN benchmarking', metavar='BOOL',           type=bool, default=True, show_default=True)

# I/O-related options.
@click.option('--status',           help='Interval of status prints', metavar='NIMG',           type=parse_nimg, default='128Ki', show_default=True)
@click.option('--snapshot',         help='Interval of network snapshots', metavar='NIMG',       type=parse_nimg, default='8Mi', show_default=True)
@click.option('--snap',             help='Snapshots every N status ticks (overrides --snapshot)', metavar='TICKS', type=click.IntRange(min=1), default=None)
@click.option('--checkpoint',       help='Interval of training checkpoints', metavar='NIMG',    type=parse_nimg, default='128Mi', show_default=True)
@click.option('--save-inference-only', 'save_inference_only', help='Save ONLY the inference snapshot (.pkl); skip the resumable training-state (.pt)', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--seed',             help='Random seed', metavar='INT',                          type=int, default=0, show_default=True)
@click.option('-n', '--dry-run',    help='Print training options and exit',                     is_flag=True)

# Inline evaluation options (combra metrics + eval-time sampler), DiffiT-v2 style.
@click.option('--combra-metrics',   'combra_metrics', help='Compute combra generative-quality metrics each snapshot tick', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--num-fid-samples',  help='Fakes generated (all ranks) per combra eval; 0=disable', metavar='INT', type=int, default=10000, show_default=True)
@click.option('--combra-ref-count', help='Real reference images for combra; 0=whole dataset', metavar='INT', type=int, default=0, show_default=True)
@click.option('--eval-sampler',     help='Eval-time / snapshot sampler', type=click.Choice(['edm', 'euler', 'ddim', 'dpm++']), default='dpm++', show_default=True)
@click.option('--eval-sampling-steps', help='Eval-time sampling steps', metavar='INT',          type=click.IntRange(min=1), default=25, show_default=True)
@click.option('--guidance',         help='Eval-time classifier-free guidance strength', metavar='FLOAT', type=float, default=1.0, show_default=True)

def main(**opts):
    """Train diffusion models according to the EDM2 recipe from the paper
    "Analyzing and Improving the Training Dynamics of Diffusion Models".

    Examples:

    \b
    # Train an S-sized ImageNet-256 latent model on 2 GPUs (no torchrun)
    edm2-train --outdir=./runs/edm2-img256-s \\
        --cfg=edm2-img256-s \\
        --data=./datasets/imagenet_256x256.zip \\
        --gpus=2 --batch-gpu=64 \\
        --combra-metrics True --save-inference-only True --snap 100

    \b
    # To resume training (only if --save-inference-only was False), run again.
    """
    launch_from_opts(opts)

#----------------------------------------------------------------------------

def launch_from_opts(opts):
    """Build the run config from a CLI-style opts dict and launch training.

    Shared by the click entry point (``main``) and the Hydra entry point
    (``train_hydra.py``) so both paths produce identical runs. ``opts`` is a
    dict keyed by the click option Python names (the same keys click passes to
    ``main`` as ``**kwargs``).
    """
    opts = dict(opts)
    outdir = opts.pop('outdir')
    gpus = opts.pop('gpus')
    snap = opts.pop('snap')
    save_inference_only = opts.pop('save_inference_only')
    dry_run = opts.pop('dry_run')

    # DiffiT-style knobs mapped onto edm2 intervals.
    if save_inference_only:      # skip the resumable .pt; keep only the .pkl
        opts['checkpoint'] = 0
    if snap is not None:         # snapshot every N status ticks
        opts['snapshot'] = snap * opts['status']

    print('Setting up training config...')
    c = setup_training_config(**opts)
    print_training_config(run_dir=outdir, c=c, num_gpus=gpus)
    if dry_run:
        print('Dry run; exiting.')
        return

    torch.multiprocessing.set_start_method('spawn', force=True)
    master_port = _free_port()
    if gpus == 1:
        subprocess_fn(rank=0, c=c, run_dir=outdir, num_gpus=1, master_port=master_port)
    else:
        torch.multiprocessing.spawn(subprocess_fn, args=(c, outdir, gpus, master_port), nprocs=gpus)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
