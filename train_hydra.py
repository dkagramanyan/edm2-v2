"""Hydra entry point for EDM2 training.

This is a thin wrapper around ``train_edm2.py``: the click CLI there is the
single source of truth for every option and its default. We introspect those
click options to seed defaults, overlay whatever ``configs/config.yaml`` (and
command-line overrides) provide, then hand the merged dict to the same
``train_edm2.launch_from_opts`` the click entry point uses -- so the Hydra and
CLI paths produce identical runs.

Usage:
    python train_hydra.py outdir=./training-runs preset=edm2-img256-s \\
        data=./datasets/imagenet_256x256.zip gpus=2 batch_gpu=64

    # override any train_edm2.py option by its Python name (dashes become
    # underscores; --cfg/--preset is named `preset`):
    python train_hydra.py outdir=./training-runs preset=edm2-img256-s data=... \\
        gpus=2 batch_gpu=64 combra_metrics=false save_inference_only=true snap=100
"""

import click
import hydra
from omegaconf import DictConfig, OmegaConf

import train_edm2


def _resolve_opts(cfg: DictConfig) -> dict:
    """Merge the click defaults with the Hydra config, applying click's types.

    Defaults such as ``status='128Ki'`` are only meaningful once run through the
    option's click type (``parse_nimg``), so both the defaults and the overrides
    are type-cast here exactly as the click CLI would cast them.

    ``configs/config.yaml`` lists every option so Hydra's struct mode allows plain
    ``key=value`` overrides. A ``null`` there means "not provided" and leaves the
    click default in place, so the YAML never duplicates the click defaults.
    """
    ctx = click.Context(train_edm2.main)
    params = {p.name: p for p in train_edm2.main.params if isinstance(p, click.Option)}

    def cast(param, value):
        # get_default() returns the *raw* default ('128Ki', 1), which click would
        # normally cast while parsing argv. Cast it the same way, or `status` stays a
        # string and `snap * status` silently repeats it instead of multiplying.
        return None if value is None else param.type_cast_value(ctx, value)

    opts = {name: cast(p, p.get_default(ctx)) for name, p in params.items()}

    for name, value in OmegaConf.to_container(cfg, resolve=True).items():
        if name not in params:
            raise KeyError(f'Unknown option {name!r}; expected one of: {sorted(params)}')
        if value is not None:
            opts[name] = cast(params[name], value)

    for name in ('outdir', 'data'):
        if opts[name] is None:
            raise ValueError(f'{name!r} is required; set it in configs/config.yaml or as {name}=...')
    return opts


@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(cfg: DictConfig) -> None:
    train_edm2.launch_from_opts(_resolve_opts(cfg))


if __name__ == '__main__':
    main()
