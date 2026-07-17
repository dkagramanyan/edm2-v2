# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Inference-snapshot format for the v2 checkpoint contract (§3).

A snapshot is EMA-only and stored as a ``.pt`` **state dict** (weight tensors keyed
by parameter name) plus enough kwargs to rebuild the model and encoder from current
code -- no pickled modules. Every snapshot carries the self-describing metadata
``{n_classes, resolution, class_names, cur_nimg}`` so downstream code reads grain-class
*names* from the checkpoint instead of guessing integer conventions.
"""

import copy
import os

import torch

import dnnlib

SNAPSHOT_FORMAT = 'edm2_inference_snapshot'
SCHEMA_VERSION = 1


def _strip_class_name(kwargs):
    return {k: v for k, v in dict(kwargs).items() if k != 'class_name'}


def atomic_torch_save(obj, path):
    """Write ``obj`` to ``path`` atomically (temp file + ``os.replace``), so a
    snapshot that exists under its final name is always complete (§3 MUST)."""
    tmp_path = path + '.tmp'
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def save_inference_snapshot(path, *, ema_net, network_kwargs, encoder_kwargs,
                            class_names, cur_nimg, resolution):
    """Atomically write one EMA-only inference snapshot as a ``.pt`` state dict.

    ``network_kwargs`` must be the full construct kwargs (class_name + model + the
    img_resolution/img_channels/label_dim interface) so loading rebuilds the model.
    """
    net = copy.deepcopy(ema_net).cpu().eval().requires_grad_(False).to(torch.float16)
    payload = dict(
        format=SNAPSHOT_FORMAT,
        schema_version=SCHEMA_VERSION,
        net=dict(
            class_name=network_kwargs['class_name'],
            kwargs=_strip_class_name(network_kwargs),
            state_dict=net.state_dict(),
        ),
        encoder=dict(
            class_name=encoder_kwargs['class_name'],
            kwargs=_strip_class_name(encoder_kwargs),
        ),
        metadata=dict(
            n_classes=int(net.label_dim),
            resolution=int(resolution),
            class_names=list(class_names) if class_names else None,
            cur_nimg=int(cur_nimg),
        ),
    )
    atomic_torch_save(payload, path)


def load_inference_snapshot(path, device=torch.device('cpu'), *, verbose=False):
    """Load a ``.pt`` inference snapshot. Returns ``(net, encoder, metadata)`` with the
    EMA network rebuilt from current code and moved to ``device`` (eval, no grad)."""
    with dnnlib.util.open_url(path, verbose=verbose) if isinstance(path, str) and dnnlib.util.is_url(path) \
            else open(path, 'rb') as f:
        data = torch.load(f, map_location=torch.device('cpu'), weights_only=False)
    if data.get('format') != SNAPSHOT_FORMAT:
        raise ValueError(f'{path}: not an edm2 inference snapshot (format={data.get("format")!r}); '
                         'pre-3.0.0 .pkl artifacts are readable only via the `legacy-pkl` tag')
    n = data['net']
    net = dnnlib.util.construct_class_by_name(class_name=n['class_name'], **n['kwargs'])
    net.load_state_dict(n['state_dict'])
    net = net.eval().requires_grad_(False).to(device)
    e = data['encoder']
    encoder = dnnlib.util.construct_class_by_name(class_name=e['class_name'], **e['kwargs'])
    return net, encoder, data.get('metadata', {})
