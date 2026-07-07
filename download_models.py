# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Pre-download the model weights EDM2 needs for offline nodes: the Stability
VAE (for latent encode/decode) and combra's image-metric backbones (InceptionV3
for FID, CLIP for CMMD, DINOv2 for FD-DINOv2). Run once on a networked node."""

import click
import numpy as np
import torch


@click.command()
@click.option('--vae', 'vae_names', help='VAE(s) to fetch', multiple=True,
              default=['stabilityai/sd-vae-ft-mse'], show_default=True)
@click.option('--combra/--no-combra', 'do_combra', help='Also fetch combra metric backbones', default=True, show_default=True)
def main(vae_names, do_combra):
    """Download and cache the VAE and (optionally) combra metric backbones."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from training.encoders import load_stability_vae
    for name in vae_names:
        print(f'Fetching VAE {name} ...')
        load_stability_vae(name, device=torch.device('cpu'))
        print('  done')

    if do_combra:
        try:
            from combra.metrics import cmmd_features, fd_dinov2_features, fid_features
        except ImportError:
            print("combra not installed; skipping metric backbones. `pip install -e '.[combra]'` to fetch them.")
            return
        print('Fetching combra metric backbones (InceptionV3 / CLIP / DINOv2) ...')
        # Call the feature extractors directly rather than compute_all_metrics: each loads
        # and caches its backbone, which is all we want. compute_all_metrics also runs the
        # angle-density path, which needs real microstructure images and errors on blank
        # dummies (empty pooled-angle arrays -> zero-size reduction).
        dummy = np.zeros((2, 64, 64, 3), dtype=np.uint8)
        fid_features(dummy, device=device)
        cmmd_features(dummy, device=device)
        fd_dinov2_features(dummy, device=device)
        print('  done')


if __name__ == '__main__':
    main()
