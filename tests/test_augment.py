"""Training-only dihedral augmentation (--augment) and the matching combra reference.

CPU only. The loop test runs one real training iteration of a tiny pixel-space model
with combra's reference precompute mocked, so it needs no GPU, weights or combra.
"""

import io
import json
import zipfile

import numpy as np
import PIL.Image
import pytest
import torch
from scipy.stats import chisquare

import train_edm2
from torch_utils import misc
from training import training_loop as tl


def _d4(x):
    """The 8 dihedral transforms of an [..., H, W] tensor, in dihedral_augment's code order."""
    return [torch.rot90(x.flip(-1) if code >= 4 else x, code % 4, dims=(-2, -1)) for code in range(8)]


def _codes(inp, out):
    """Per item, the index of the (unique) D4 transform mapping inp[i] to out[i]."""
    codes = []
    for x, y in zip(inp, out):
        hits = [code for code, t in enumerate(_d4(x)) if torch.equal(t, y)]
        assert len(hits) == 1, f'output is not exactly one dihedral transform of its own input: {hits}'
        codes.append(hits[0])
    return np.array(codes)


def test_dihedral_augment_draws_the_8_transforms_uniformly():
    # Random 5x5 images have no symmetry, so each output identifies its transform
    # uniquely -- and matching against the item's OWN input checks that items (and
    # hence their labels, which the loop never touches) stay paired.
    n = 16000
    inp = torch.randint(0, 256, [n, 3, 5, 5], generator=torch.Generator().manual_seed(1), dtype=torch.uint8)
    misc.set_random_seed(0, 0, 0)
    out = tl.dihedral_augment(inp)
    assert out.dtype == torch.uint8 and out.shape == inp.shape
    counts = np.bincount(_codes(inp, out), minlength=8)
    assert chisquare(counts).pvalue > 1e-3, counts
    assert np.all(np.abs(counts - n / 8) < 0.05 * n / 8), counts


def test_dihedral_augment_is_deterministic_per_seed_and_iteration():
    inp = torch.randint(0, 256, [64, 3, 4, 4], generator=torch.Generator().manual_seed(2), dtype=torch.uint8)

    def draw(*seed_args):
        misc.set_random_seed(*seed_args)
        return tl.dihedral_augment(inp)

    assert torch.equal(draw(0, 0, 128), draw(0, 0, 128))
    assert not torch.equal(draw(0, 0, 128), draw(0, 0, 256))   # next iteration
    assert not torch.equal(draw(0, 0, 128), draw(1, 0, 128))   # other seed


def test_dihedral_augment_refuses_non_square():
    with pytest.raises(ValueError, match='square'):
        tl.dihedral_augment(torch.zeros([2, 3, 4, 6], dtype=torch.uint8))


def _zip(path, shape=(8, 8, 3), n=4):
    meta = {'labels': [[f'{i}.png', i % 2] for i in range(n)], 'class_names': ['A', 'B']}
    rng = np.random.default_rng(0)
    with zipfile.ZipFile(path, 'w') as z:
        for i in range(n):
            buf = io.BytesIO()
            PIL.Image.fromarray(rng.integers(0, 256, shape, dtype=np.uint8)).save(buf, format='PNG')
            z.writestr(f'{i}.png', buf.getvalue())
        z.writestr('dataset.json', json.dumps(meta))
    return str(path)


def _latent_zip(path, n=4):
    meta = {'labels': [[f'{i}.npy', i % 2] for i in range(n)], 'class_names': ['A', 'B']}
    with zipfile.ZipFile(path, 'w') as z:
        for i in range(n):
            buf = io.BytesIO()
            np.save(buf, np.zeros((8, 4, 4), np.uint8))
            z.writestr(f'{i}.npy', buf.getvalue())
        z.writestr('dataset.json', json.dumps(meta))
    return str(path)


def test_augment_cli_defaults_on_and_can_be_switched_off(tmp_path):
    import click
    opt = {p.name: p for p in train_edm2.main.params}['augment']
    assert opt.default is True and opt.type is click.BOOL and opt.opts == ['--augment']
    data = _zip(tmp_path / 'd.zip')
    cfg = lambda path, **o: train_edm2.setup_training_config(data=path, batch_gpu=2, tick=1, snap=1, **o)
    assert cfg(data).augment is True
    assert cfg(data, augment=False).augment is False
    latents = _latent_zip(tmp_path / 'latent.zip')
    with pytest.raises(click.ClickException, match='--augment'):
        cfg(latents)
    assert cfg(latents, augment=False).augment is False


class _NoDDP:
    """Stands in for DistributedDataParallel on CPU: returns the module itself."""
    def __new__(cls, module, device_ids=None):
        return module


@pytest.mark.parametrize('augment', [True, False])
def test_training_loop_augments_only_training_batches_and_matches_the_reference(tmp_path, monkeypatch, augment):
    ref_calls, aug_calls = [], []

    def fake_precompute(images_u8, device, rank, world_size, **kwargs):
        ref_calls.append(kwargs)
        return None, False  # "failed": metrics off, the loop carries on

    real_augment = tl.dihedral_augment
    def spy_augment(images):
        aug_calls.append(images.shape)
        return real_augment(images)

    monkeypatch.setattr(tl.combra_mod, 'HAS_COMBRA', True)
    monkeypatch.setattr(tl.combra_mod, 'combra_smoke_test', lambda *a, **k: None)
    monkeypatch.setattr(tl.combra_mod, 'all_ranks_ok', lambda ok, *a, **k: ok)
    monkeypatch.setattr(tl.combra_mod, 'precompute_combra_reference', fake_precompute)
    monkeypatch.setattr(tl, 'dihedral_augment', spy_augment)
    monkeypatch.setattr(torch.nn.parallel, 'DistributedDataParallel', _NoDDP)

    tl.training_loop(
        dataset_kwargs=dict(class_name='training.dataset.ImageFolderDataset', path=_zip(tmp_path / 'd.zip')),
        encoder_kwargs=dict(class_name='training.encoders.StandardRGBEncoder'),
        data_loader_kwargs=dict(class_name='torch.utils.data.DataLoader', num_workers=0),
        network_kwargs=dict(class_name='training.networks_edm2.Precond', model_channels=64, use_fp16=False),
        run_dir=str(tmp_path), batch_size=2, total_nimg=2, status_nimg=None, snapshot_nimg=None,
        device=torch.device('cpu'), eval_num_steps=2, augment=augment)

    assert ref_calls == [{'dihedral': augment}]
    # One training batch, augmented iff --augment; reals/fakes grids never are.
    assert aug_calls == ([torch.Size([2, 3, 8, 8])] if augment else [])
