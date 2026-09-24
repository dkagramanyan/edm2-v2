# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Minimal rank-0 text logger (§7).

Replaces the vendored OpenAI-baselines logger. There is no ``progress.csv`` /
``progress.json`` and no per-rank ``log-rankNNN.txt``: scalars go straight to
``stats.jsonl`` and TensorBoard from the training loop, and this module only prints
a one-line text message to stdout (teed into the run's ``.log`` and timestamped by
``dnnlib.util.Logger``). All output is rank-0-only; nothing goes to TensorBoard text.
"""

_rank = 0


def configure(rank=0):
    """Record the caller's rank so non-zero ranks stay silent."""
    global _rank
    _rank = int(rank)


def log(msg):
    """Emit one rank-0 text line to stdout (teed to the run's ``.log``).

    The ``[YYYY-MM-DD HH:MM:SS]`` prefix is added once, by ``dnnlib.util.Logger``."""
    if _rank != 0:
        return
    print(msg)
