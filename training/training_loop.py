# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""

import glob
import json
import os
import re
import time

import numpy as np
import PIL.Image
import psutil
import torch

import dnnlib
from torch_utils import distributed as dist, misc, persistence, training_stats
from training import checkpoint as ckpt, logger as tblog, metrics as combra_mod, samplers

#----------------------------------------------------------------------------
# Delete all but the `keep_last` newest per-tick inference snapshots so the history
# stays disk-bounded. `keep_last <= 0` keeps everything. Prunes by distinct kimg (a
# phema run writes one suffixed .pt per EMA std per tick), matching ONLY the
# `edm2-snapshot-<kimg>[-<std>]-inference.pt` history.

def prune_inference_snapshots(run_dir, keep_last):
    if keep_last <= 0:
        return
    kimg_re = re.compile(r'edm2-snapshot-(\d+).*-inference\.pt$')
    snaps = []
    for path in glob.glob(os.path.join(run_dir, 'edm2-snapshot-*-inference.pt')):
        m = kimg_re.search(os.path.basename(path))
        if m is not None:
            snaps.append((int(m.group(1)), path))
    keep = set(sorted({kimg for kimg, _ in snaps})[-keep_last:])
    for kimg, path in snaps:
        if kimg not in keep:
            try:
                os.remove(path)
            except OSError:
                pass

#----------------------------------------------------------------------------
# Uncertainty-based loss function (Equations 14,15,16,21) proposed in the
# paper "Analyzing and Improving the Training Dynamics of Diffusion Models".

@persistence.persistent_class
class EDM2Loss:
    def __init__(self, P_mean=-0.4, P_std=1.0, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        noise = torch.randn_like(images) * sigma
        denoised, logvar = net(images + noise, sigma, labels, return_logvar=True)
        loss = (weight / logvar.exp()) * ((denoised - images) ** 2) + logvar
        return loss

#----------------------------------------------------------------------------
# Learning rate decay schedule used in the paper "Analyzing and Improving
# the Training Dynamics of Diffusion Models".

def learning_rate_schedule(cur_nimg, batch_size, ref_lr=100e-4, ref_batches=70e3, rampup_Mimg=10):
    lr = ref_lr
    if ref_batches > 0:
        lr /= np.sqrt(max(cur_nimg / (ref_batches * batch_size), 1))
    if rampup_Mimg > 0:
        lr *= min(cur_nimg / (rampup_Mimg * 1e6), 1)
    return lr

#----------------------------------------------------------------------------
# Helpers for the snapshot image grids and picking the EMA network used for
# eval-time generation. Inference-only -- never touches the update path.

def _primary_ema_net(ema, net):
    if ema is None:
        return net
    got = ema.get()
    return got[0][0] if isinstance(got, list) else got

def save_image_grid(images_nchw_uint8, fname, grid_size):
    # images_nchw_uint8: torch.uint8 [N, C, H, W]. Saves a gw x gh PNG grid.
    gw, gh = grid_size
    img = images_nchw_uint8.cpu().numpy()
    n, c, h, w = img.shape
    n = min(n, gw * gh)
    img = img[:n]
    canvas = np.zeros([c, gh * h, gw * w], dtype=np.uint8)
    for idx in range(n):
        y, x = (idx // gw) * h, (idx % gw) * w
        canvas[:, y:y + h, x:x + w] = img[idx]
    canvas = np.transpose(canvas, (1, 2, 0))
    PIL.Image.fromarray(canvas[:, :, 0] if c == 1 else canvas, {1: 'L', 3: 'RGB'}[c]).save(fname)
    return canvas # HWC uint8, for optional TensorBoard logging

def _grid_size(n):
    # Smallest square grid that holds n samples.
    side = int(np.ceil(np.sqrt(max(n, 1))))
    return (side, side)

def _class_sorted_onehot(label_dim, n, device):
    # One-hot labels grouped by class in sorted (0,1,2,...) order, so labeled grids
    # show class-sorted rows. None when the model is unconditional.
    if label_dim <= 0:
        return None
    idx = torch.arange(n, device=device) * label_dim // max(n, 1)
    idx = idx.clamp(max=label_dim - 1)
    return torch.eye(label_dim, device=device)[idx]

def _pick_reals_sorted(dataset_obj, n, device):
    # Raw dataset pixels (never VAE round-tripped), class-sorted for the grid rows.
    cap = min(len(dataset_obj), 4096)
    tagged = []
    for i in range(cap):
        lbl = dataset_obj.get_label(i)
        c = int(np.argmax(lbl)) if getattr(lbl, 'size', 0) > 0 else 0
        tagged.append((c, i))
    tagged.sort()
    sel = [i for _c, i in tagged[:n]]
    return torch.stack([torch.as_tensor(dataset_obj[i][0]) for i in sel]).to(device)

@torch.inference_mode()
def _generate_grid(eval_net, encoder, gnet, device, *, sampler, num_steps, guidance, seed, n):
    eval_net.eval()
    g = torch.Generator(device=device).manual_seed(int(seed))
    noise = torch.randn(n, eval_net.img_channels, eval_net.img_resolution, eval_net.img_resolution,
                        device=device, generator=g)
    labels = _class_sorted_onehot(eval_net.label_dim, n, device)
    latents = samplers.sample(eval_net, noise, labels=labels, gnet=gnet,
                              sampler=sampler, num_steps=num_steps, guidance=guidance)
    return encoder.decode(latents)

#----------------------------------------------------------------------------
# Main training loop.

def training_loop(
    dataset_kwargs      = dict(class_name='training.dataset.ImageFolderDataset', path=None),
    encoder_kwargs      = dict(class_name='training.encoders.StabilityVAEEncoder'),
    data_loader_kwargs  = dict(class_name='torch.utils.data.DataLoader', pin_memory=True, num_workers=3, prefetch_factor=2),
    network_kwargs      = dict(class_name='training.networks_edm2.Precond'),
    loss_kwargs         = dict(class_name='training.training_loop.EDM2Loss'),
    optimizer_kwargs    = dict(class_name='torch.optim.Adam', betas=(0.9, 0.99)),
    lr_kwargs           = dict(func_name='training.training_loop.learning_rate_schedule'),
    ema_kwargs          = dict(class_name='training.phema.PowerFunctionEMA'),

    run_dir             = '.',      # Output directory.
    seed                = 0,        # Global random seed.
    batch_size          = 2048,     # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU. None = no limit.
    mirror              = False,     # Stochastic per-item horizontal flip in the training loader only.
    total_nimg          = 8<<30,    # Train for a total of N training images.
    slice_nimg          = None,     # Train for a maximum of N training images in one invocation. None = no limit.
    status_nimg         = 128<<10,  # Report status every N training images. None = disable.
    snapshot_nimg       = 8<<20,    # Save network snapshot every N training images. None = disable.
    snapshot_keep_last  = 3,        # Keep only the N newest per-tick inference snapshots (0 = keep all).

    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    force_finite        = True,     # Get rid of NaN/Inf gradients before feeding them to the optimizer.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    allow_tf32          = True,     # Enable TF32 on the cuDNN / matmul paths?
    device              = torch.device('cuda'),

    combra_metrics      = True,     # Compute combra generative-quality metrics each snapshot tick.
    num_fid_samples     = 10000,    # Fakes generated (across all ranks) per combra eval. 0 = disable.
    combra_ref_count    = None,     # Real reference images for combra. None = whole training set.
    eval_sampler        = 'dpm++',  # Sampler used for eval-time / snapshot generation.
    eval_num_steps      = 25,       # Sampling steps for eval-time / snapshot generation.
    eval_guidance       = 1,        # Classifier-free guidance strength for eval-time generation.
):
    # Initialize.
    prev_status_time = time.time()
    misc.set_random_seed(seed, dist.get_rank())
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Validate batch size.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()
    assert total_nimg % batch_size == 0
    assert slice_nimg is None or slice_nimg % batch_size == 0
    assert status_nimg is None or status_nimg % batch_size == 0
    assert snapshot_nimg is None or snapshot_nimg % batch_size == 0

    # Setup dataset, encoder, and network.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    ref_image, ref_label = dataset_obj[0]
    class_names = dataset_obj.class_names
    dist.print0('Setting up encoder...')
    encoder = dnnlib.util.construct_class_by_name(**encoder_kwargs)
    ref_image = encoder.encode_latents(torch.as_tensor(ref_image).to(device).unsqueeze(0))
    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=ref_image.shape[-1], img_channels=ref_image.shape[1], label_dim=ref_label.shape[-1])
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs)
    net.train().requires_grad_(True).to(device)
    # Full construct kwargs (model + interface), stored in every inference snapshot so
    # loading rebuilds the model from current code (§3).
    full_network_kwargs = dict(network_kwargs, **interface_kwargs)

    # Print network summary.
    if dist.get_rank() == 0:
        misc.print_module_summary(net, [
            torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device),
            torch.ones([batch_gpu], device=device),
            torch.zeros([batch_gpu, net.label_dim], device=device),
        ], max_nesting=2)

    # Setup training state.
    dist.print0('Setting up training state...')
    state = dnnlib.EasyDict(cur_nimg=0, total_elapsed_time=0)
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device])
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs)
    ema = dnnlib.util.construct_class_by_name(net=net, **ema_kwargs) if ema_kwargs is not None else None

    # No resume: every launch trains start-to-finish from a fresh run directory (§3).
    stop_at_nimg = total_nimg
    if slice_nimg is not None:
        granularity = snapshot_nimg if snapshot_nimg is not None else batch_size
        slice_end_nimg = (state.cur_nimg + slice_nimg) // granularity * granularity # round down
        stop_at_nimg = min(stop_at_nimg, slice_end_nimg)
    assert stop_at_nimg > state.cur_nimg
    dist.print0(f'Training from {state.cur_nimg // 1000} kimg to {stop_at_nimg // 1000} kimg:')
    dist.print0()

    # Setup logging (§7): rank-0-only text log (teed into <run>.log by dnnlib.Logger),
    # scalars written straight to stats.jsonl + TensorBoard. The tfevents file carries
    # the run name as a filename_suffix so a copied-out event file self-identifies.
    rank, world_size = dist.get_rank(), dist.get_world_size()
    run_name = os.path.basename(os.path.normpath(run_dir))
    tblog.configure(rank=rank)
    sw = None
    stats_jsonl = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
        try:
            from torch.utils.tensorboard import SummaryWriter
            sw = SummaryWriter(run_dir, filename_suffix=f'.{run_name}')
            tblog.register_tb_writer(sw)
        except ImportError:
            dist.print0('TensorBoard not available; skipping tfevents.')

    # Precompute the combra reference (sharded across all ranks) once, so the cached
    # features are reused every eval tick. Reference = raw dataset pixels (§6).
    use_combra = bool(combra_metrics) and combra_mod.HAS_COMBRA and (num_fid_samples or 0) > 0
    if combra_metrics and not combra_mod.HAS_COMBRA:
        dist.print0('WARNING: --combra-metrics requested but combra is not installed; skipping.')
    combra_ref = None
    if use_combra:
        ref_count = combra_ref_count if combra_ref_count is not None else len(dataset_obj)
        dist.print0(f'Precomputing combra reference from {min(ref_count, len(dataset_obj))} raw images...')
        local_ref = combra_mod.load_reference_shard(dataset_obj, ref_count, batch_gpu, device, rank, world_size, seed=seed)
        if rank == 0:
            combra_mod.combra_smoke_test(local_ref, device, dist.print0)
        combra_ref = combra_mod.precompute_combra_reference(local_ref, device, rank, world_size)

    # Save a grid of real images (raw pixels) and an initial (pre-training) fakes grid.
    pixel_resolution = None
    if rank == 0:
        grid_n = min(64, batch_gpu * num_accumulation_rounds)
        grid_size = _grid_size(grid_n)
        reals = _pick_reals_sorted(dataset_obj, min(grid_n, len(dataset_obj)), device)
        pixel_resolution = int(reals.shape[-1])
        reals_canvas = save_image_grid(reals, os.path.join(run_dir, 'reals.png'), grid_size)
        init_fakes = _generate_grid(_primary_ema_net(ema, net), encoder, None, device,
                                    sampler=eval_sampler, num_steps=eval_num_steps, guidance=eval_guidance, seed=seed, n=grid_n)
        init_canvas = save_image_grid(init_fakes, os.path.join(run_dir, 'fakes_init.png'), grid_size)
        if sw is not None:
            sw.add_image('reals', reals_canvas, global_step=0, dataformats='HWC')
            sw.add_image('fakes', init_canvas, global_step=0, dataformats='HWC')
            sw.flush()
        net.train()

    # Main training loop.
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed, start_idx=state.cur_nimg)
    dataset_iterator = iter(dnnlib.util.construct_class_by_name(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))
    prev_status_nimg = state.cur_nimg
    cumulative_training_time = 0
    start_nimg = state.cur_nimg
    cur_tick = 0
    stats_metrics = None            # latest combra metrics dict (rank 0)
    while True:
        done = (state.cur_nimg >= stop_at_nimg)

        # Report status.
        if status_nimg is not None and (done or state.cur_nimg % status_nimg == 0) and (state.cur_nimg != start_nimg or start_nimg == 0):
            cur_time = time.time()
            state.total_elapsed_time += cur_time - prev_status_time
            cur_process = psutil.Process(os.getpid())
            cpu_memory_usage = sum(p.memory_info().rss for p in [cur_process] + cur_process.children(recursive=True))
            fields = dict(
                kimg        = training_stats.report0('Progress/kimg',                       state.cur_nimg / 1e3),
                time        = training_stats.report0('Timing/total_sec',                    state.total_elapsed_time),
                sec_per_tick= training_stats.report0('Timing/sec_per_tick',                 cur_time - prev_status_time),
                sec_per_kimg= training_stats.report0('Timing/sec_per_kimg',                 cumulative_training_time / max(state.cur_nimg - prev_status_nimg, 1) * 1e3),
                maintenance = training_stats.report0('Timing/maintenance_sec',              cur_time - prev_status_time - cumulative_training_time),
                cpumem      = training_stats.report0('Resources/cpu_mem_gb',                cpu_memory_usage / 2**30),
                gpumem      = training_stats.report0('Resources/peak_gpu_mem_gb',           torch.cuda.max_memory_allocated(device) / 2**30),
                reserved    = training_stats.report0('Resources/peak_gpu_mem_reserved_gb',  torch.cuda.max_memory_reserved(device) / 2**30),
            )
            cur_tick += 1
            # Console tick line (also teed into <run>.log, mirrored to TB text).
            tblog.log(' '.join([
                f"tick {cur_tick:<5d}",
                f"kimg {fields['kimg']:<9.1f}",
                f"time {dnnlib.util.format_time(fields['time']):<12s}",
                f"sec/tick {fields['sec_per_tick']:<8.2f}",
                f"sec/kimg {fields['sec_per_kimg']:<7.3f}",
                f"maintenance {fields['maintenance']:<7.2f}",
                f"cpumem {fields['cpumem']:<6.2f}",
                f"gpumem {fields['gpumem']:<6.2f}",
                f"reserved {fields['reserved']:<6.2f}",
            ]))
            cumulative_training_time = 0
            prev_status_nimg = state.cur_nimg
            prev_status_time = cur_time
            torch.cuda.reset_peak_memory_stats()

            # Flush training scalars to stats.jsonl (scalar rows only) and TensorBoard.
            training_stats.default_collector.update()
            if rank == 0:
                fmt = {'Progress/tick': '%.0f', 'Progress/kimg': '%.3f', 'timestamp': '%.3f'}
                collected = [(name, value.mean) for name, value in training_stats.default_collector.as_dict().items()]
                items = collected + [('Progress/tick', cur_tick), ('timestamp', time.time())]
                items = [f'"{name}": ' + (fmt.get(name, '%g') % value if np.isfinite(value) else 'NaN') for name, value in items]
                stats_jsonl.write('{' + ', '.join(items) + '}\n')
                stats_jsonl.flush()
                if sw is not None:
                    for name, value in collected:
                        if np.isfinite(value):
                            sw.add_scalar(name, float(value), global_step=int(state.cur_nimg / 1e3), walltime=state.total_elapsed_time)
                    sw.add_scalar('Progress/tick', float(cur_tick), global_step=int(state.cur_nimg / 1e3), walltime=state.total_elapsed_time)
                    sw.flush()

        # Evaluate combra metrics (all ranks generate their shard; rank 0 aggregates)
        # and save a fakes grid. Runs at snapshot cadence and always at the last tick,
        # entirely outside the loss/optimizer/EMA update path.
        at_snapshot = snapshot_nimg is not None and (
            (state.cur_nimg % snapshot_nimg == 0 and (state.cur_nimg != start_nimg or start_nimg == 0))
            or (done and state.cur_nimg != start_nimg))
        if at_snapshot and (use_combra or rank == 0):
            eval_net = _primary_ema_net(ema, net)
            eval_net.eval()
            if use_combra:
                stats_metrics = combra_mod.compute_combra_metrics(
                    eval_net, encoder, combra_ref, num_fid_samples, batch_gpu, device, rank, world_size,
                    sampler=eval_sampler, num_steps=eval_num_steps, guidance=eval_guidance,
                    seed=seed + state.cur_nimg, log_fn=dist.print0)
                if rank == 0 and stats_metrics:
                    gstep, walltime = int(state.cur_nimg / 1e3), state.total_elapsed_time
                    stats_jsonl.write(json.dumps(dict(stats_metrics, timestamp=time.time(), kimg=state.cur_nimg / 1e3)) + '\n')
                    stats_jsonl.flush()
                    if sw is not None:
                        for name, value in stats_metrics.items():
                            if np.isfinite(value):
                                sw.add_scalar(f'Metrics/{name}', value, global_step=gstep, walltime=walltime)
                        sw.flush()
                    tblog.log('Metrics: ' + '  '.join(f'{k} {v:g}' for k, v in stats_metrics.items()))
            if rank == 0:
                grid_n = min(64, batch_gpu * num_accumulation_rounds)
                grid_size = _grid_size(grid_n)
                fakes = _generate_grid(eval_net, encoder, None, device, sampler=eval_sampler,
                                       num_steps=eval_num_steps, guidance=eval_guidance, seed=seed, n=grid_n)
                fakes_canvas = save_image_grid(fakes, os.path.join(run_dir, f'fakes{state.cur_nimg // 1000:06d}.png'), grid_size)
                if sw is not None:
                    sw.add_image('fakes', fakes_canvas, global_step=int(state.cur_nimg / 1e3), dataformats='HWC')
                    sw.flush()
                net.train()

        # Save checkpoints: EMA-only .pt state-dict inference snapshots, one per EMA
        # std, written atomically and pruned to --snapshot-keep-last (§3). The newest
        # snapshot is always the final model (last-tick MUST above).
        if at_snapshot:
            misc.check_ddp_consistency(net)  # collective: must run on every rank
        if at_snapshot and rank == 0:
            ema_list = ema.get() if ema is not None else [(net, '')]
            for ema_net, ema_suffix in ema_list:
                fname = f'edm2-snapshot-{state.cur_nimg//1000:06d}{ema_suffix}-inference.pt'
                dist.print0(f'Saving {fname} ... ', end='', flush=True)
                ckpt.save_inference_snapshot(
                    os.path.join(run_dir, fname),
                    ema_net=ema_net, network_kwargs=full_network_kwargs, encoder_kwargs=encoder_kwargs,
                    class_names=class_names, cur_nimg=state.cur_nimg, resolution=pixel_resolution)
                dist.print0('done')
            prune_inference_snapshots(run_dir, snapshot_keep_last)

        # Done?
        if done:
            break

        # Evaluate loss and accumulate gradients.
        batch_start_time = time.time()
        misc.set_random_seed(seed, dist.get_rank(), state.cur_nimg)
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                images, labels = next(dataset_iterator)
                images = images.to(device)
                if mirror:
                    # Loader-level stochastic per-item horizontal flip (training only).
                    flip = torch.rand(images.shape[0], device=device) < 0.5
                    images[flip] = images[flip].flip(-1)
                images = encoder.encode_latents(images)
                loss = loss_fn(net=ddp, images=images, labels=labels.to(device))
                training_stats.report('Loss/loss', loss)
                loss.sum().mul(loss_scaling / batch_gpu_total).backward()

        # Run optimizer and update weights.
        lr = dnnlib.util.call_func_by_name(cur_nimg=state.cur_nimg, batch_size=batch_size, **lr_kwargs)
        training_stats.report('Loss/learning_rate', lr)
        for g in optimizer.param_groups:
            g['lr'] = lr
        if force_finite:
            for param in net.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
        optimizer.step()

        # Update EMA and training state.
        state.cur_nimg += batch_size
        if ema is not None:
            ema.update(cur_nimg=state.cur_nimg, batch_size=batch_size)
        cumulative_training_time += time.time() - batch_start_time

    if stats_jsonl is not None:
        stats_jsonl.close()

#----------------------------------------------------------------------------
