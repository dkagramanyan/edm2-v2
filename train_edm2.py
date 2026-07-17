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
import re
import socket
import warnings

import click
import torch

import dnnlib
import training.training_loop
from torch_utils import distributed as dist

warnings.filterwarnings('ignore', 'You are using `torch.load` with `weights_only=False`')

#----------------------------------------------------------------------------
# Configuration presets. `duration` is the total training length in images (the
# --kimg default); the batch is derived from --batch-gpu x --gpus x --grad-accum.

config_presets = {
    'edm2-img256-xs':   dnnlib.EasyDict(duration=2048<<20, channels=128, lr=0.0120, decay=70000, dropout=0.00, P_mean=-0.4, P_std=1.0),
    'edm2-img256-s':    dnnlib.EasyDict(duration=2048<<20, channels=192, lr=0.0100, decay=70000, dropout=0.00, P_mean=-0.4, P_std=1.0),
    'edm2-img256-m':    dnnlib.EasyDict(duration=2048<<20, channels=256, lr=0.0090, decay=70000, dropout=0.10, P_mean=-0.4, P_std=1.0),
    'edm2-img512-xxs':  dnnlib.EasyDict(duration=2048<<20, channels=64,  lr=0.0170, decay=70000, dropout=0.00, P_mean=-0.4, P_std=1.0),
    'edm2-img512-xs':   dnnlib.EasyDict(duration=2048<<20, channels=128, lr=0.0120, decay=70000, dropout=0.00, P_mean=-0.4, P_std=1.0),
    'edm2-img512-s':    dnnlib.EasyDict(duration=2048<<20, channels=192, lr=0.0100, decay=70000, dropout=0.00, P_mean=-0.4, P_std=1.0),
    'edm2-img512-m':    dnnlib.EasyDict(duration=2048<<20, channels=256, lr=0.0090, decay=70000, dropout=0.10, P_mean=-0.4, P_std=1.0),
    'edm2-img512-l':    dnnlib.EasyDict(duration=1792<<20, channels=320, lr=0.0080, decay=70000, dropout=0.10, P_mean=-0.4, P_std=1.0),
    'edm2-img512-xl':   dnnlib.EasyDict(duration=1280<<20, channels=384, lr=0.0070, decay=70000, dropout=0.10, P_mean=-0.4, P_std=1.0),
    'edm2-img512-xxl':  dnnlib.EasyDict(duration=896<<20,  channels=448, lr=0.0065, decay=70000, dropout=0.10, P_mean=-0.4, P_std=1.0),
    'edm2-img64-xs':    dnnlib.EasyDict(duration=1024<<20, channels=128, lr=0.0120, decay=35000, dropout=0.00, P_mean=-0.8, P_std=1.6),
    'edm2-img64-s':     dnnlib.EasyDict(duration=1024<<20, channels=192, lr=0.0100, decay=35000, dropout=0.00, P_mean=-0.8, P_std=1.6),
    'edm2-img64-m':     dnnlib.EasyDict(duration=2048<<20, channels=256, lr=0.0090, decay=35000, dropout=0.10, P_mean=-0.8, P_std=1.6),
    'edm2-img64-l':     dnnlib.EasyDict(duration=1024<<20, channels=320, lr=0.0080, decay=35000, dropout=0.10, P_mean=-0.8, P_std=1.6),
    'edm2-img64-xl':    dnnlib.EasyDict(duration=640<<20,  channels=384, lr=0.0070, decay=35000, dropout=0.10, P_mean=-0.8, P_std=1.6),
    'edm2-img1024-s':   dnnlib.EasyDict(duration=1024<<20, channels=192, lr=0.0080, decay=70000, dropout=0.10, P_mean=-0.4, P_std=1.0),
    'edm2-img1024-m':   dnnlib.EasyDict(duration=1024<<20, channels=256, lr=0.0070, decay=70000, dropout=0.10, P_mean=-0.4, P_std=1.0),
}

#----------------------------------------------------------------------------
# Round an image count to a whole number of batches (>= one batch), so the tick /
# snapshot / total cadences all land on batch boundaries whatever --batch-gpu x
# --gpus x --grad-accum works out to.

def _round_to_batch(nimg, batch_size):
    return max(batch_size, int(round(nimg / batch_size)) * batch_size)

#----------------------------------------------------------------------------
# Setup arguments for training.training_loop.training_loop().

def setup_training_config(cfg='edm2-img512-s', gpus=1, **opts):
    opts = dnnlib.EasyDict(opts)
    c = dnnlib.EasyDict()

    # Preset.
    if cfg not in config_presets:
        raise click.ClickException(f'Invalid configuration preset "{cfg}"')
    for key, value in config_presets[cfg].items():
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
    # done inline every step (StabilityVAEOnTheFlyEncoder). An 8-channel dataset is
    # already VAE-encoded offline. Default: latent for every preset except the
    # pixel-space edm2-img64 family.
    latent = opts.get('latent', None)
    if latent is None:
        latent = not cfg.startswith('edm2-img64')
    if dataset_channels == 3:
        cls = 'StabilityVAEOnTheFlyEncoder' if latent else 'StandardRGBEncoder'
        c.encoder_kwargs = dnnlib.EasyDict(class_name=f'training.encoders.{cls}')
    elif dataset_channels == 8:
        c.encoder_kwargs = dnnlib.EasyDict(class_name='training.encoders.StabilityVAEEncoder')
    else:
        raise click.ClickException(f'--data: Unsupported channel count {dataset_channels}')

    # Batch formula: total = batch_gpu x gpus x grad_accum (§2). Ticks and snapshots
    # are counted in kimg and rounded to whole batches.
    batch_gpu = opts.batch_gpu
    grad_accum = opts.get('grad_accum', 1)
    batch_size = batch_gpu * gpus * grad_accum
    c.batch_size = batch_size
    c.batch_gpu = batch_gpu

    total_nimg = (opts.kimg * 1000) if opts.get('kimg', None) is not None else opts.duration
    c.total_nimg = _round_to_batch(total_nimg, batch_size)
    c.status_nimg = _round_to_batch(opts.tick * 1000, batch_size)
    c.snapshot_nimg = opts.snap * c.status_nimg

    # Network / loss / lr.
    precision = opts.get('precision', 'fp16')
    c.network_kwargs = dnnlib.EasyDict(class_name='training.networks_edm2.Precond', model_channels=opts.channels, dropout=opts.dropout,
                                       use_fp16=(precision != 'fp32'), mixed_precision_dtype=precision)
    c.loss_kwargs = dnnlib.EasyDict(class_name='training.training_loop.EDM2Loss', P_mean=opts.P_mean, P_std=opts.P_std)
    c.lr_kwargs = dnnlib.EasyDict(func_name='training.training_loop.learning_rate_schedule', ref_lr=opts.lr, ref_batches=opts.decay)

    # Performance-related options.
    c.loss_scaling = opts.get('ls', 1)
    c.cudnn_benchmark = opts.get('bench', True)
    c.allow_tf32 = opts.get('tf32', True)
    c.mirror = opts.get('mirror', False)
    c.data_loader_kwargs = dnnlib.EasyDict(class_name='torch.utils.data.DataLoader', pin_memory=True,
                                           num_workers=opts.get('workers', 3), prefetch_factor=2)

    # I/O-related options.
    c.snapshot_keep_last = opts.get('snapshot_keep_last', 3)
    c.seed = opts.get('seed', 0)

    # Inline evaluation (combra metrics + eval-time sampler).
    c.combra_metrics = opts.get('combra_metrics', True)
    c.num_fid_samples = opts.get('num_fid_samples', 10000)
    c.combra_ref_count = opts.get('combra_ref_count', 0) or None
    c.eval_sampler = opts.get('eval_sampler', 'dpm++')
    c.eval_num_steps = opts.get('eval_sampling_steps', 25)
    c.eval_guidance = opts.get('guidance', 1.0)
    return c

#----------------------------------------------------------------------------
# Print training configuration.

def print_training_config(run_dir, c, num_gpus, precision):
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
    dist.print0(f'Precision:               {precision}')
    dist.print0()

#----------------------------------------------------------------------------
# Run directory naming: <outdir>/<id:05d>-<cfg>-gpus<N>-batch<B>[-desc]. A fresh id
# is always allocated -- runs are never resumed or reused (§3).

def make_run_desc(cfg, num_gpus, batch_size, desc=None):
    name = f'{cfg}-gpus{num_gpus}-batch{batch_size}'
    return f'{name}-{desc}' if desc else name

def make_run_dir(outdir, desc):
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    return os.path.join(outdir, f'{cur_run_id:05d}-{desc}')

#----------------------------------------------------------------------------
# Launch training.

def launch_training(run_dir, c):
    if dist.get_rank() == 0 and not os.path.isdir(run_dir):
        dist.print0('Creating output directory...')
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)

    torch.distributed.barrier()
    # Rank-0-only run log, named after the run directory (§7).
    if dist.get_rank() == 0:
        run_name = os.path.basename(os.path.normpath(run_dir))
        dnnlib.util.Logger(file_name=os.path.join(run_dir, f'{run_name}.log'), file_mode='a', should_flush=True)
    training.training_loop.training_loop(run_dir=run_dir, **c)

#----------------------------------------------------------------------------
# Per-rank entry point spawned by --gpus (no torchrun needed). Sets the env vars
# torch_utils.distributed.init() reads, then trains.

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
# Command line interface.

@click.command()

# Main options.
@click.option('--outdir',           help='Where to save the results', metavar='DIR',            type=str, required=True)
@click.option('--data',             help='Path to the dataset', metavar='ZIP|DIR',              type=str, required=True)
@click.option('--gpus',             help='Number of GPUs to spawn (no torchrun needed)', metavar='INT', type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--cond',             help='Train class-conditional model', metavar='BOOL',       type=bool, default=True, show_default=True)
@click.option('--cfg',              help='Configuration preset', metavar='STR',                 type=str, default='edm2-img512-s', show_default=True)
@click.option('--latent',           help='Train in VAE latent space (else raw pixel space); default: latent for all presets except edm2-img64', metavar='BOOL', type=bool, default=None)
@click.option('--desc',             help='String to append to the run directory name', metavar='STR', type=str, default=None)

# Hyperparameters.
@click.option('--kimg',             help='Training duration in kimg', metavar='INT',            type=click.IntRange(min=1), default=None)
@click.option('--channels',         help='Channel multiplier', metavar='INT',                   type=click.IntRange(min=64), default=None)
@click.option('--dropout',          help='Dropout probability', metavar='FLOAT',                type=click.FloatRange(min=0, max=1), default=None)
@click.option('--P_mean', 'P_mean', help='Noise level mean', metavar='FLOAT',                   type=float, default=None)
@click.option('--P_std', 'P_std',   help='Noise level standard deviation', metavar='FLOAT',     type=click.FloatRange(min=0, min_open=True), default=None)
@click.option('--lr',               help='Learning rate max. (alpha_ref)', metavar='FLOAT',     type=click.FloatRange(min=0, min_open=True), default=None)
@click.option('--decay',            help='Learning rate decay (t_ref)', metavar='BATCHES',      type=click.FloatRange(min=0), default=None)

# Batch / precision.
@click.option('--batch-gpu',        help='Per-GPU batch size', metavar='INT',                   type=click.IntRange(min=1), default=32, show_default=True)
@click.option('--grad-accum',       help='Gradient accumulation rounds', metavar='INT',         type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--precision',        help='Training precision', type=click.Choice(['fp32', 'fp16', 'bf16']), default='fp16', show_default=True)
@click.option('--tf32',             help='Enable TF32 on cuDNN / matmul', metavar='BOOL',       type=bool, default=True, show_default=True)
@click.option('--bench',            help='Enable cuDNN benchmarking', metavar='BOOL',           type=bool, default=True, show_default=True)
@click.option('--ls',               help='Loss scaling', metavar='FLOAT',                       type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)

# Data.
@click.option('--mirror',           help='Stochastic horizontal flip in the training loader', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--workers',          help='DataLoader worker processes', metavar='INT',          type=click.IntRange(min=1), default=3, show_default=True)

# I/O-related options.
@click.option('--tick',             help='Status/eval tick interval in kimg', metavar='INT',    type=click.IntRange(min=1), default=128, show_default=True)
@click.option('--snap',             help='Snapshot every N ticks', metavar='INT',               type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--snapshot-keep-last', 'snapshot_keep_last', help='Keep only the N newest inference snapshots (0 = keep all)', metavar='INT', type=click.IntRange(min=0), default=3, show_default=True)
@click.option('--seed',             help='Random seed', metavar='INT',                          type=int, default=0, show_default=True)
@click.option('-n', '--dry-run',    help='Print training options and exit',                     is_flag=True)

# Inline evaluation options (combra metrics + eval-time sampler).
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
    # Train an S-sized 256px latent model on 2 GPUs (no torchrun)
    edm2-train --outdir=./runs/edm2-img256-s \\
        --cfg=edm2-img256-s \\
        --data=./datasets/wc_co_256x256.zip \\
        --gpus=2 --batch-gpu=64 \\
        --tick=128 --snap=64

    \b
    # Each snapshot tick (and the last tick) writes EMA-only .pt inference snapshots
    # edm2-snapshot-<kimg>[-<std>]-inference.pt, pruned to --snapshot-keep-last.
    # Runs are not resumable: size --kimg (or split stages) to fit the job's time
    # limit. Every launch allocates a fresh run id.
    """
    launch_from_opts(opts)

#----------------------------------------------------------------------------

def launch_from_opts(opts):
    """Build the run config from a CLI-style opts dict and launch training."""
    opts = dict(opts)
    outdir = opts.pop('outdir')
    gpus = opts.pop('gpus')
    desc = opts.pop('desc', None)
    dry_run = opts.pop('dry_run', False)
    cfg = opts['cfg']
    precision = opts.get('precision', 'fp16')

    print('Setting up training config...')
    c = setup_training_config(gpus=gpus, **opts)

    # Resolve the run directory here, in the parent, so every spawned rank is handed
    # the same path instead of racing to number one for itself.
    run_desc = make_run_desc(cfg, gpus, c.batch_size, desc)
    run_dir = make_run_dir(outdir, run_desc)
    print_training_config(run_dir=run_dir, c=c, num_gpus=gpus, precision=precision)
    if dry_run:
        print('Dry run; exiting.')
        return

    torch.multiprocessing.set_start_method('spawn', force=True)
    master_port = _free_port()
    if gpus == 1:
        subprocess_fn(rank=0, c=c, run_dir=run_dir, num_gpus=1, master_port=master_port)
    else:
        torch.multiprocessing.spawn(subprocess_fn, args=(c, run_dir, gpus, master_port), nprocs=gpus)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
