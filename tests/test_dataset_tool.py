# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.

"""`--max-images` must sample across classes, not fill the first one.

The image list is sorted, so it is grouped by class folder. Truncating it to the
cap took every image from the alphabetically first class: a capped run produced a
single-class dataset from a multi-class source, and `edm2-gen-images
--classes=1,2...` then failed with "index out of range for a 1-class model".
"""

import numpy as np
import PIL.Image

from dataset_tool import open_image_folder, stratified_subset


def _make_source(root, per_class):
    for cls, n in sorted(per_class.items()):
        d = root / cls
        d.mkdir(parents=True)
        for i in range(n):
            PIL.Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(d / f'{i:03d}.png')
    return root


def test_stratified_subset_spreads_the_cap():
    files = [f'c{c}/{i:03d}.png' for c in range(4) for i in range(10)]
    picked = stratified_subset(files, lambda f: f.split('/')[0], 8)
    assert len(picked) == 8
    assert sorted({f.split('/')[0] for f in picked}) == ['c0', 'c1', 'c2', 'c3']


def test_stratified_subset_leaves_a_single_class_alone():
    files = [f'{i:03d}.png' for i in range(10)]
    assert stratified_subset(files, lambda f: None, 3) == files[:3]


def test_stratified_subset_drains_small_classes_into_large_ones():
    # 2 images in c0, 10 in c1: a cap of 8 must still return 8, not 4.
    files = ['c0/000.png', 'c0/001.png'] + [f'c1/{i:03d}.png' for i in range(10)]
    picked = stratified_subset(files, lambda f: f.split('/')[0], 8)
    assert len(picked) == 8
    assert sum(f.startswith('c0/') for f in picked) == 2


def test_open_image_folder_cap_covers_every_class(tmp_path):
    src = _make_source(tmp_path / 'src', {'a': 6, 'b': 6, 'c': 6})
    num_files, entries, class_names = open_image_folder(str(src), max_images=6)
    assert num_files == 6
    assert class_names == ['a', 'b', 'c']
    labels = [e.label for e in entries]
    assert len(labels) == 6
    # Two per class, not six from 'a'.
    assert sorted(labels) == [0, 0, 1, 1, 2, 2]


def test_open_image_folder_uncapped_is_unchanged(tmp_path):
    src = _make_source(tmp_path / 'src', {'a': 3, 'b': 3})
    num_files, entries, class_names = open_image_folder(str(src), max_images=None)
    assert num_files == 6
    assert sorted(e.label for e in entries) == [0, 0, 0, 1, 1, 1]
