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
                 'precision', 'tf32', 'bench', 'workers', 'desc',
                 'snapshot_keep_last', 'combra_metrics', 'num_fid_samples', 'combra_ref_count'):
        assert name in o, f'missing --{name}'
    assert o['tick'].default == 128
    assert o['snap'].default == 64
    assert o['precision'].default == 'fp16'
    assert o['tf32'].default is True
    assert o['grad_accum'].default == 1
    assert o['batch_gpu'].default == 32
    assert o['workers'].default == 3
    assert o['snapshot_keep_last'].default == 1
    assert set(o['precision'].type.choices) == {'fp32', 'fp16', 'bf16'}


def test_train_cli_drops_legacy_flags():
    flags = _all_flags(train_edm2.main)
    for dead in ('--preset', '--duration', '--batch', '--status', '--snapshot',
                 '--fp16', '--save-inference-only', '--mirror'):
        assert dead not in flags, f'{dead} should be removed'
    assert '--cfg' in flags


def _labelled_zip(path, class_names):
    import io
    import json
    import zipfile

    import PIL.Image
    meta = {'labels': [[f'{i}.png', i % 2] for i in range(4)]}
    if class_names is not None:
        meta['class_names'] = class_names
    with zipfile.ZipFile(path, 'w') as z:
        for i in range(4):
            buf = io.BytesIO()
            PIL.Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(buf, format='PNG')
            z.writestr(f'{i}.png', buf.getvalue())
        z.writestr('dataset.json', json.dumps(meta))
    return str(path)


def _train_config(data, **opts):
    return train_edm2.setup_training_config(data=data, batch_gpu=2, tick=1, snap=1, **opts)


def test_train_refuses_conditional_run_on_nameless_zip(tmp_path):
    import click
    named = _labelled_zip(tmp_path / 'named.zip', ['A', 'B'])
    assert _train_config(named).dataset_kwargs.use_labels
    nameless = _labelled_zip(tmp_path / 'nameless.zip', None)
    with pytest.raises(click.ClickException, match='class_names'):
        _train_config(nameless)
    assert not _train_config(nameless, cond=False).dataset_kwargs.use_labels


def test_train_refuses_guidance_without_guiding_network(tmp_path):
    import click
    data = _labelled_zip(tmp_path / 'named.zip', ['A', 'B'])
    assert _train_config(data, guidance=1.0).eval_guidance == 1
    with pytest.raises(click.ClickException, match='guiding network'):
        _train_config(data, guidance=2.0)


def test_lr_schedule_matches_paper_in_iterations(tmp_path):
    # Paper (Table 6): batch 2048, 10 Mimg rampup (~4.9k iterations), decay knee at t_ref = 70k
    # iterations. Both must stay in iterations at any batch, so the peak LR is alpha_ref.
    import dnnlib
    data = _labelled_zip(tmp_path / 'named.zip', ['A', 'B'])
    for gpus in (1, 16):  # batch 2 and 32
        c = _train_config(data, gpus=gpus, cfg='edm2-img512-s')
        batch = c.batch_size
        # Called exactly as the training loop calls it.
        lr = lambda it: dnnlib.util.call_func_by_name(cur_nimg=it * batch, batch_size=batch, **c.lr_kwargs)
        rampup_iters = 10e6 / 2048
        assert lr(rampup_iters / 2) == pytest.approx(0.0100 / 2)
        assert lr(rampup_iters) == pytest.approx(0.0100)
        assert lr(70000) == pytest.approx(0.0100)
        assert lr(4 * 70000) == pytest.approx(0.0100 / 2)
    assert _train_config(data, gpus=1, rampup=10).lr_kwargs.rampup_Mimg == 10


def test_img1024_presets_match_paper_values():
    # The paper (Table 6) has no 1024 px models: each img1024 preset takes the paper's
    # img512 values for the same model size (alpha_ref, dropout, duration, ...).
    for size in ('s', 'm'):
        assert train_edm2.config_presets[f'edm2-img1024-{size}'] == train_edm2.config_presets[f'edm2-img512-{size}']


def test_snapshot_retention_keeps_newest_and_best_per_metric():
    from training.training_loop import select_snapshots
    nan = float('nan')
    records = [
        (100, {'combra_fid': 50.0, 'combra_fd_dinov2': 900.0, 'combra_cmmd': 2.0}),
        (200, {'combra_fid': 30.0, 'combra_fd_dinov2': 950.0, 'combra_cmmd': nan}),
        (300, {'combra_fid': 40.0, 'combra_fd_dinov2': 800.0, 'combra_cmmd': 3.0}),
        (400, {}),                                  # eval failed / disabled
        (500, {'combra_fid': 30.0, 'combra_fd_dinov2': nan}),
    ]
    keep, best = select_snapshots(records, keep_last=1)
    # Tie on fid keeps the earlier 200; nan and missing never win; cmmd best stays at 100.
    assert {k: kimg for k, (_, kimg) in best.items()} == {'combra_fid': 200, 'combra_fd_dinov2': 300, 'combra_cmmd': 100}
    assert best['combra_fid'][0] == 30.0
    assert keep == {100, 200, 300, 500}
    assert select_snapshots(records, keep_last=2)[0] == {100, 200, 300, 400, 500}
    assert select_snapshots(records, keep_last=0)[0] == {100, 200, 300, 400, 500}
    # Bests survive later, worse ticks; one kimg may fill every role.
    keep, best = select_snapshots(records[:1] + [(k, {'combra_fid': 99.0}) for k in (600, 700)], keep_last=1)
    assert keep == {100, 700} and {kimg for _, kimg in best.values()} == {100}
    # Nothing evaluated: only the newest is kept.
    assert select_snapshots([(100, {}), (200, None)], keep_last=1) == ({200}, {})
    assert select_snapshots([], keep_last=1) == (set(), {})


def test_prune_inference_snapshots_by_kimg(tmp_path):
    from training.training_loop import prune_inference_snapshots
    names = ['edm2-snapshot-000100-0.050-inference.pt', 'edm2-snapshot-000100-0.100-inference.pt',
             'edm2-snapshot-000200-0.050-inference.pt', 'edm2-snapshot-000300-inference.pt',
             'fakes000200.png']
    for n in names:
        (tmp_path / n).touch()
    prune_inference_snapshots(str(tmp_path), {100, 300})
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(names[:2] + names[3:])


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
                         {0: 2, 2: 2}, resolution=4, samples_per_class=4, channels=3, class_names=names)
        for c in (0, 2):
            imgs = np.full((2, 4, 4, 3), rank, dtype=np.uint8)
            w.write(c, imgs, seeds=[rank * 10 + 0, rank * 10 + 1], indices=[rank * 2, rank * 2 + 1])
        assert w.close() == 0

    out = tmp_path / 'wc.h5'
    counts = merge_shards([str(p) for p in sorted(shard_dir.glob('rank_*.h5'))], str(out), class_names=names)
    assert counts == {0: 4, 2: 4}

    import h5py
    # Parity attrs (image_shape_hwc / samples_per_class / class_idx) on the shards...
    with h5py.File(shard_dir / 'rank_000.h5', 'r') as f:
        assert tuple(f.attrs['image_shape_hwc']) == (4, 4, 3)
        assert int(f.attrs['samples_per_class']) == 4
        g = f['class_2']
        assert int(g.attrs['class_idx']) == 2
        assert int(g.attrs['samples_per_class']) == 4
        assert tuple(g.attrs['image_shape_hwc']) == (4, 4, 3)
    # ...and on the merged file.
    with h5py.File(out, 'r') as f:
        assert f.attrs['format'] == 'generated_images_shard'
        assert int(f.attrs['schema_version']) == 1
        assert int(f.attrs['missing_count']) == 0
        assert list(f.attrs['class_names']) == names
        assert tuple(f.attrs['image_shape_hwc']) == (4, 4, 3)
        assert int(f.attrs['samples_per_class']) == 4
        assert set(f.keys()) == {'class_0', 'class_2'}
        g = f['class_2']
        assert g['images'].shape == (4, 4, 4, 3)
        assert g['images'].dtype == np.uint8
        assert g.attrs['class_name'] == 'Ultra_Co6_2'
        assert int(g.attrs['class_idx']) == 2
        assert int(g.attrs['samples_per_class']) == 4
        assert tuple(g.attrs['image_shape_hwc']) == (4, 4, 3)
        # Ordered by sample index across the two shards (rank0 idx 0,1 then rank1 idx 2,3).
        assert list(g['seeds'][:]) == [0, 1, 10, 11]


def test_h5_merge_hard_fails_on_incomplete_shard(tmp_path):
    p = tmp_path / 'rank_000.h5'
    w = RankH5Writer(str(p), 0, {0: 3}, resolution=4, samples_per_class=3, channels=3, class_names=['A'])
    w.write(0, np.zeros((2, 4, 4, 3), np.uint8), seeds=[0, 1], indices=[0, 1])  # leave 1 unwritten
    assert w.close() == 1  # missing_count
    with pytest.raises(ValueError, match='incomplete shard'):
        merge_shards([str(p)], str(tmp_path / 'out.h5'))


def test_h5_merge_hard_fails_on_unclosed_shard(tmp_path):
    # A rank that dies before close() leaves a shard with no missing_count attr at
    # all; the merge gate must not read that as "0 missing".
    import h5py
    p = tmp_path / 'rank_000.h5'
    with h5py.File(p, 'w') as f:
        f.attrs['format'] = 'generated_images_shard'
        f.attrs['schema_version'] = 1
    with pytest.raises(ValueError, match='never closed'):
        merge_shards([str(p)], str(tmp_path / 'out.h5'))


def test_h5_merge_recomputes_missing_from_written_mask(tmp_path):
    # The gate must not trust the missing_count attr alone: falsify it and leave a
    # hole in the written mask.
    p = tmp_path / 'rank_000.h5'
    w = RankH5Writer(str(p), 0, {0: 3}, resolution=4, samples_per_class=3, channels=3, class_names=['A'])
    w.write(0, np.zeros((2, 4, 4, 3), np.uint8), seeds=[0, 1], indices=[0, 1])  # leave 1 unwritten
    assert w.close() == 1
    import h5py
    with h5py.File(p, 'r+') as f:
        f.attrs['missing_count'] = 0
        f['class_0'].attrs['missing_count'] = 0
    with pytest.raises(ValueError, match='written mask'):
        merge_shards([str(p)], str(tmp_path / 'out.h5'))


def test_h5_writer_requires_class_names(tmp_path):
    with pytest.raises(ValueError, match='class_names'):
        RankH5Writer(str(tmp_path / 'rank_000.h5'), 0, {0: 1}, resolution=4, samples_per_class=1)


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


def test_load_network_rejects_legacy_pkl():
    with pytest.raises(ValueError, match=r'\.pt inference snapshot'):
        generate_images.load_network('edm2-img512-s.pkl', torch.device('cpu'))


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


def _shard(path, rank, indices, samples_per_class, c=0):
    w = RankH5Writer(str(path), rank, {c: len(indices)}, resolution=4, samples_per_class=samples_per_class,
                     channels=3, class_names=['A'])
    w.write(c, np.zeros((len(indices), 4, 4, 3), np.uint8), seeds=list(indices), indices=list(indices))
    assert w.close() == 0
    return str(path)


def test_h5_merge_hard_fails_on_duplicate_indices(tmp_path):
    # A stale shard from an earlier run repeats sample indices of the current one.
    a = _shard(tmp_path / 'rank_000.h5', 0, [0, 1], samples_per_class=2)
    b = _shard(tmp_path / 'rank_001.h5', 1, [0, 1], samples_per_class=2)
    with pytest.raises(ValueError, match='duplicate sample indices'):
        merge_shards([a, b], str(tmp_path / 'out.h5'))
    assert not (tmp_path / 'out.h5').exists()


def test_h5_merge_hard_fails_on_wrong_class_count(tmp_path):
    a = _shard(tmp_path / 'rank_000.h5', 0, [0, 2], samples_per_class=4)
    with pytest.raises(ValueError, match='expected samples_per_class=4'):
        merge_shards([a], str(tmp_path / 'out.h5'))


class _LabelledSet:
    # Minimal dataset: 360 images per class, stored class-sorted as in the wc zips.
    def __init__(self, per_class=360, label_dim=3):
        self.label_dim, self.has_labels = label_dim, label_dim > 0
        self._c = np.repeat(np.arange(max(label_dim, 1)), per_class)  # per_class: int or per-class list

    def __len__(self):
        return len(self._c)

    def get_label(self, i):
        return np.eye(self.label_dim, dtype=np.float32)[self._c[i]] if self.label_dim else np.zeros([0], np.float32)

    def __getitem__(self, i):
        return np.full((1, 2, 2), self._c[i], dtype=np.uint8), self.get_label(i)


@pytest.mark.parametrize('n', [64, 49, 16, 3, 1])
def test_reals_grid_is_class_balanced_like_the_fakes(n):
    from training.training_loop import _class_sorted_onehot, _pick_reals_sorted
    reals = _pick_reals_sorted(_LabelledSet(), n, torch.device('cpu'))
    real_classes = reals[:, 0, 0, 0].tolist()
    fake_classes = _class_sorted_onehot(3, n, torch.device('cpu')).argmax(1).tolist()
    assert real_classes == fake_classes


def test_reals_grid_tops_up_a_short_class_and_handles_unlabelled():
    from training.training_loop import _pick_reals_sorted
    reals = _pick_reals_sorted(_LabelledSet(per_class=[2, 10, 10]), 9, torch.device('cpu'))
    assert reals[:, 0, 0, 0].tolist() == [0, 0, 1, 1, 1, 1, 2, 2, 2]
    reals = _pick_reals_sorted(_LabelledSet(per_class=10, label_dim=0), 4, torch.device('cpu'))
    assert reals.shape[0] == 4
