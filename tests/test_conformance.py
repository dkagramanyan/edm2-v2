# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.

"""CPU conformance checks for the v2 model-API convention (§13). No model execution,
GPU, dataset, or combra required: CLI contract, HDF5 artifact schema + merge
hard-fail, checkpoint metadata, normalization round-trip, and the class-label rules.
"""

import numpy as np
import pytest
import torch

import generate_images
import train_edm2
from training import checkpoint as ckpt
from training.encoders import StandardRGBEncoder
from training.h5_writer import RankH5Writer, merge_shards
from training.networks_edm2 import Precond


def _opts(cmd):
    return {p.name: p for p in cmd.params}


def _all_flags(cmd):
    flags = set()
    for p in cmd.params:
        flags.update(p.opts)
        flags.update(p.secondary_opts)
    return flags


# --- CLI contract (§2) -------------------------------------------------------

def test_train_cli_has_v2_flags_and_defaults():
    o = _opts(train_edm2.main)
    for name in ('cfg', 'kimg', 'tick', 'snap', 'batch_gpu', 'grad_accum',
                 'precision', 'tf32', 'bench', 'mirror', 'workers', 'desc',
                 'snapshot_keep_last', 'combra_metrics', 'num_fid_samples', 'combra_ref_count'):
        assert name in o, f'missing --{name}'
    assert o['tick'].default == 128
    assert o['snap'].default == 64
    assert o['precision'].default == 'fp16'
    assert o['tf32'].default is True
    assert o['grad_accum'].default == 1
    assert o['batch_gpu'].default == 32
    assert o['workers'].default == 3
    assert o['snapshot_keep_last'].default == 3
    assert set(o['precision'].type.choices) == {'fp32', 'fp16', 'bf16'}


def test_train_cli_drops_legacy_flags():
    flags = _all_flags(train_edm2.main)
    for dead in ('--preset', '--duration', '--batch', '--status', '--snapshot',
                 '--fp16', '--save-inference-only'):
        assert dead not in flags, f'{dead} should be removed'
    assert '--cfg' in flags


def test_gen_cli_contract():
    o = _opts(generate_images.cmdline)
    for name in ('classes', 'samples_per_class', 'save_mode', 'gpus', 'batch_gpu'):
        assert name in o, f'missing --{name}'
    assert o['save_mode'].default == 'hdf5'
    assert set(o['save_mode'].type.choices) == {'hdf5', 'dir'}
    assert '--network' in _all_flags(generate_images.cmdline)  # --net alias


# --- HDF5 artifact contract (§4) --------------------------------------------

def test_h5_writer_merge_roundtrip(tmp_path):
    shard_dir = tmp_path / 'shards'
    shard_dir.mkdir()
    names = ['Ultra_Co11', 'Ultra_Co25', 'Ultra_Co6_2']
    # Two ranks, classes 0 and 2, 2 samples each per class per rank.
    for rank in range(2):
        w = RankH5Writer(str(shard_dir / f'rank_{rank:03d}.h5'), rank,
                         {0: 2, 2: 2}, resolution=4, channels=3, class_names=names)
        for c in (0, 2):
            imgs = np.full((2, 4, 4, 3), rank, dtype=np.uint8)
            w.write(c, imgs, seeds=[rank * 10 + 0, rank * 10 + 1], indices=[rank * 2, rank * 2 + 1])
        assert w.close() == 0

    out = tmp_path / 'wc.h5'
    counts = merge_shards([str(p) for p in sorted(shard_dir.glob('rank_*.h5'))], str(out), class_names=names)
    assert counts == {0: 4, 2: 4}

    import h5py
    with h5py.File(out, 'r') as f:
        assert f.attrs['format'] == 'generated_images_shard'
        assert int(f.attrs['schema_version']) == 1
        assert int(f.attrs['missing_count']) == 0
        assert list(f.attrs['class_names']) == names
        assert set(f.keys()) == {'class_0', 'class_2'}
        g = f['class_2']
        assert g['images'].shape == (4, 4, 4, 3)
        assert g['images'].dtype == np.uint8
        assert g.attrs['class_name'] == 'Ultra_Co6_2'
        # Ordered by sample index across the two shards (rank0 idx 0,1 then rank1 idx 2,3).
        assert list(g['seeds'][:]) == [0, 1, 10, 11]


def test_h5_merge_hard_fails_on_incomplete_shard(tmp_path):
    p = tmp_path / 'rank_000.h5'
    w = RankH5Writer(str(p), 0, {0: 3}, resolution=4, channels=3, class_names=['A'])
    w.write(0, np.zeros((2, 4, 4, 3), np.uint8), seeds=[0, 1], indices=[0, 1])  # leave 1 unwritten
    assert w.close() == 1  # missing_count
    with pytest.raises(ValueError, match='incomplete shard'):
        merge_shards([str(p)], str(tmp_path / 'out.h5'))


# --- Checkpoint metadata contract (§3) --------------------------------------

def test_checkpoint_metadata_and_reload(tmp_path):
    net = Precond(img_resolution=8, img_channels=3, label_dim=3, use_fp16=False,
                  model_channels=8, channel_mult=[1, 2], num_blocks=1, attn_resolutions=[8])
    nk = dict(class_name='training.networks_edm2.Precond', model_channels=8, channel_mult=[1, 2],
              num_blocks=1, attn_resolutions=[8], use_fp16=False, mixed_precision_dtype='fp16',
              img_resolution=8, img_channels=3, label_dim=3)
    ek = dict(class_name='training.encoders.StandardRGBEncoder')
    path = str(tmp_path / 'edm2-snapshot-000042-0.100-inference.pt')
    ckpt.save_inference_snapshot(path, ema_net=net, network_kwargs=nk, encoder_kwargs=ek,
                                 class_names=['A', 'B', 'C'], cur_nimg=42000, resolution=256)
    net2, enc2, meta = ckpt.load_inference_snapshot(path, torch.device('cpu'))
    assert set(meta) == {'n_classes', 'resolution', 'class_names', 'cur_nimg'}
    assert meta['n_classes'] == 3 and meta['resolution'] == 256
    assert meta['class_names'] == ['A', 'B', 'C'] and meta['cur_nimg'] == 42000
    assert isinstance(enc2, StandardRGBEncoder)
    out = net2(torch.randn(2, 3, 8, 8), torch.rand(2) + 0.1, torch.eye(3)[[0, 1]])
    assert out.shape == (2, 3, 8, 8)


# --- Normalization contract (§5) --------------------------------------------

def test_standard_rgb_encoder_roundtrips_uint8():
    enc = StandardRGBEncoder()
    u = torch.arange(256, dtype=torch.uint8).reshape(1, 1, 16, 16).repeat(1, 3, 1, 1)
    back = enc.decode(enc.encode_latents(u))
    assert back.dtype == torch.uint8
    assert torch.equal(back, u)  # decode is the exact inverse of encode for uint8 pixels


# --- Label contract (§5) -----------------------------------------------------

def test_resolve_classes_by_index_range_and_name():
    names = ['Ultra_Co11', 'Ultra_Co25', 'Ultra_Co6_2']
    assert generate_images.resolve_classes('0,2', 3, names) == [0, 2]
    assert generate_images.resolve_classes('0-2', 3, names) == [0, 1, 2]
    assert generate_images.resolve_classes('Ultra_Co6_2,Ultra_Co11', 3, names) == [0, 2]
    import click
    with pytest.raises(click.ClickException):
        generate_images.resolve_classes('9', 3, names)
    with pytest.raises(click.ClickException):
        generate_images.resolve_classes('Nope', 3, names)
