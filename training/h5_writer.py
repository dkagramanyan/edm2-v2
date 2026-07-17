# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Per-rank HDF5 generation output in the RankH5Writer layout (§4).

Each rank writes a shard ``shards/rank_NNN.h5`` with one group per class
(``class_<c>/images|seeds``, images stored as **uint8 NHWC**) plus a per-sample
``written`` mask and a ``missing_count`` attribute. Rank 0 merges the shards into a
single ``<desc>.h5`` and **hard-fails** if any shard is incomplete, so a crashed
generation run never feeds zero-filled (black) slots into the downstream angle
pipeline. Every shard and merged file carries ``format="generated_images_shard"``
and ``schema_version=1`` so downstream code sniffs any model's output identically.
"""

import numpy as np

FORMAT = 'generated_images_shard'
SCHEMA_VERSION = 1


def _import_h5py():
    import h5py  # deferred so `import training.h5_writer` works without h5py installed
    return h5py


def _set_class_names(obj, class_names):
    if class_names is not None:
        obj.attrs['class_names'] = np.asarray(list(class_names), dtype=object)


class RankH5Writer:
    """Preallocated per-class shard for one rank. ``class_to_count`` maps each class
    index to the number of samples this rank will write for it."""

    def __init__(self, path, rank, class_to_count, resolution, channels=3, class_names=None):
        h5py = _import_h5py()
        self.f = h5py.File(path, 'w')
        self.f.attrs['format'] = FORMAT
        self.f.attrs['schema_version'] = SCHEMA_VERSION
        self.f.attrs['rank'] = int(rank)
        _set_class_names(self.f, class_names)
        self._pos = {}
        str_dt = h5py.string_dtype()
        for c, n in class_to_count.items():
            g = self.f.create_group(f'class_{c}')
            g.create_dataset('images', shape=(n, resolution, resolution, channels), dtype='uint8')
            g.create_dataset('seeds', shape=(n,), dtype='int64')
            g.create_dataset('indices', shape=(n,), dtype='int64')
            g.create_dataset('written', shape=(n,), dtype='bool', data=np.zeros(n, dtype=bool))
            if class_names is not None:
                g.attrs.create('class_name', class_names[c], dtype=str_dt)
            self._pos[c] = 0

    def write(self, class_idx, images_nhwc, seeds, indices):
        g = self.f[f'class_{class_idx}']
        p = self._pos[class_idx]
        b = len(images_nhwc)
        g['images'][p:p + b] = np.asarray(images_nhwc, dtype=np.uint8)
        g['seeds'][p:p + b] = np.asarray(seeds, dtype=np.int64)
        g['indices'][p:p + b] = np.asarray(indices, dtype=np.int64)
        g['written'][p:p + b] = True
        self._pos[class_idx] = p + b

    def close(self):
        total_missing = 0
        for c in self._pos:
            g = self.f[f'class_{c}']
            missing = int((~g['written'][:]).sum())
            g.attrs['missing_count'] = missing
            total_missing += missing
        self.f.attrs['missing_count'] = total_missing
        self.f.close()
        return total_missing


def merge_shards(shard_paths, out_path, class_names=None):
    """Merge per-rank shards into ``out_path``, ordered by sample index within each
    class. Raises if any shard is not a complete ``generated_images_shard`` (the §4
    merge hard-fail). Returns the merged per-class sample counts."""
    h5py = _import_h5py()
    per_class = {}
    for sp in shard_paths:
        with h5py.File(sp, 'r') as f:
            if f.attrs.get('format') != FORMAT:
                raise ValueError(f'{sp}: not a {FORMAT} shard (format={f.attrs.get("format")!r})')
            missing = int(f.attrs.get('missing_count', 0))
            if missing != 0:
                raise ValueError(f'{sp}: incomplete shard, missing_count={missing}; refusing to merge '
                                 '(a crashed generation run must not feed black images downstream)')
            for name in f:
                c = int(name.split('_')[1])
                g = f[name]
                per_class.setdefault(c, []).append((g['indices'][:], g['seeds'][:], g['images'][:]))

    counts = {}
    str_dt = h5py.string_dtype()
    with h5py.File(out_path, 'w') as out:
        out.attrs['format'] = FORMAT
        out.attrs['schema_version'] = SCHEMA_VERSION
        _set_class_names(out, class_names)
        for c in sorted(per_class):
            idxs = np.concatenate([blk[0] for blk in per_class[c]])
            seeds = np.concatenate([blk[1] for blk in per_class[c]])
            imgs = np.concatenate([blk[2] for blk in per_class[c]])
            order = np.argsort(idxs, kind='stable')
            g = out.create_group(f'class_{c}')
            g.create_dataset('images', data=imgs[order])
            g.create_dataset('seeds', data=seeds[order])
            g.attrs['missing_count'] = 0
            if class_names is not None:
                g.attrs.create('class_name', class_names[c], dtype=str_dt)
            counts[c] = len(order)
        out.attrs['missing_count'] = 0
    return counts
