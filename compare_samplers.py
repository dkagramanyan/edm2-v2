# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Compare EDM2 samplers by combra metrics as a function of sampling steps, to
find how many steps ``k`` each sampler needs -- i.e. the optimal number of
sampling steps.

For each sampler (edm / euler / ddim / dpm++) and each step count ``k``, a batch
of samples is generated and scored against a fixed real reference batch with
:func:`combra.metrics.compare_samplers`; the metric-vs-``k`` curve plateaus at the
optimal step count. The generic sweep/plot lives in combra so it stays
codebase-agnostic; this script is only the EDM2-side wiring.

Example (see also sbatch/compare_samplers_256x256.sbatch):
    python compare_samplers.py --net=network-snapshot-....pkl \\
        --data=datasets/imagenet_256x256.zip --num-samples=512 \\
        --samplers=edm,euler,ddim,dpm++ --k-values=5,10,20,50,100,250 \\
        --outdir=sampler-comparison/256
"""

import os
import pickle
from pathlib import Path

import click
import torch
import dnnlib
from torch_utils import distributed as dist
from training.metrics import load_reference_shard, generate_fake_shard

try:
    from combra.metrics import compare_samplers, plot_sampler_comparison
    HAS_COMBRA = True
except ImportError:
    HAS_COMBRA = False


def _parse_csv(s):
    return [x.strip() for x in str(s).split(',') if x.strip()]


@click.command()
@click.option('--net',          help='Network pickle filename', metavar='PATH|URL',     type=str, required=True)
@click.option('--gnet',         help='Guiding network pickle filename', metavar='PATH|URL', type=str, default=None)
@click.option('--data',         help='Real reference dataset (zip or dir)', metavar='ZIP|DIR', type=str, required=True)
@click.option('--num-samples',  help='Samples (and real refs) per (sampler, k)', metavar='INT', type=click.IntRange(min=2), default=512, show_default=True)
@click.option('--batch',        help='Per-forward batch size', metavar='INT',           type=click.IntRange(min=1), default=32, show_default=True)
@click.option('--samplers', 'sampler_names', help='Comma-separated sampler names', metavar='LIST', type=str, default='edm,euler,ddim,dpm++', show_default=True)
@click.option('--k-values',     help='Comma-separated step counts to sweep', metavar='LIST', type=str, default='5,10,20,50,100,250', show_default=True)
@click.option('--guidance',     help='Guidance strength  [default: 1; no guidance]', metavar='FLOAT', type=float, default=1.0, show_default=True)
@click.option('--seed',         help='Base random seed', metavar='INT',                 type=int, default=0, show_default=True)
@click.option('--outdir',       help='Output directory for the table + plot', metavar='DIR', type=str, required=True)
def main(net, gnet, data, num_samples, batch, sampler_names, k_values, guidance, seed, outdir):
    """Sweep samplers over step counts and score each batch with combra."""
    if not HAS_COMBRA:
        raise click.UsageError('combra is not installed; `pip install combra` to run this comparison.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    names = _parse_csv(sampler_names)
    ks = [int(x) for x in _parse_csv(k_values)]

    # Load model + encoder from the snapshot pickle.
    with dnnlib.util.open_url(net) as f:
        pkl = pickle.load(f)
    model = pkl['ema'].to(device).eval()
    encoder = pkl['encoder']
    encoder.init(device)
    guide_net = None
    if guidance != 1 and gnet is not None:
        with dnnlib.util.open_url(gnet) as f:
            guide_net = pickle.load(f)['ema'].to(device).eval()

    # Fixed real reference batch (single process: rank 0 / world 1).
    dataset = dnnlib.util.construct_class_by_name(
        class_name='training.dataset.ImageFolderDataset', path=data, use_labels=False)
    reference = load_reference_shard(dataset, encoder, num_samples, batch, device, 0, 1)

    # One generator per sampler; fn(k) -> a uint8 NHWC batch generated with k steps.
    def make_fn(name):
        def fn(k):
            dist.print0(f'Generating {num_samples} samples: sampler={name} k={k} ...')
            return generate_fake_shard(model, encoder, guide_net, num_samples, batch, device, 0, 1,
                                       sampler=name, num_steps=k, guidance=guidance, seed=seed)
        return fn
    samplers_map = {name: make_fn(name) for name in names}

    df = compare_samplers(reference, samplers_map, ks, device=str(device), image_metrics=True)

    outdir_p = Path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)
    table_path = outdir_p / 'sampler_comparison.parquet'
    plot_path = outdir_p / 'sampler_comparison.png'
    df.to_parquet(table_path)
    plot_sampler_comparison(df, save_path=str(plot_path))
    print(f'Wrote {table_path}')
    print(f'Wrote {plot_path}')


if __name__ == '__main__':
    main()
