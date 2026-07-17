# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Minimal rank-0 text logger (§7).

Replaces the vendored OpenAI-baselines logger. There is no ``progress.csv`` /
``progress.json`` and no per-rank ``log-rankNNN.txt``: scalars go straight to
``stats.jsonl`` and TensorBoard from the training loop, and this module only fans a
one-line text message out to stdout (teed into the run's ``.log`` by
``dnnlib.util.Logger``) and to TensorBoard's ``log`` text tag. All output is
rank-0-only.
"""

import datetime

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

_rank = 0
_tb_writer = None
_tb_step = 0


def timestamp():
    """Local system time, as stamped on every text event."""
    return datetime.datetime.now().strftime(TIMESTAMP_FORMAT)


def configure(rank=0):
    """Record the caller's rank so non-zero ranks stay silent."""
    global _rank, _tb_writer, _tb_step
    _rank = int(rank)
    _tb_writer = None
    _tb_step = 0


def register_tb_writer(summary_writer):
    """Attach a TensorBoard ``SummaryWriter`` so ``log()`` also mirrors to add_text."""
    global _tb_writer
    _tb_writer = summary_writer


def log(msg):
    """Emit one rank-0 text line: stdout (teed to the run's ``.log``) + TB ``log`` tag.

    ``dnnlib.util.Logger`` skips re-stamping lines that already carry a
    ``[YYYY-MM-DD HH:MM:SS]`` prefix, so the timestamp is applied exactly once here."""
    global _tb_step
    if _rank != 0:
        return
    print(f"[{timestamp()}] {msg}")
    if _tb_writer is not None:
        try:
            _tb_writer.add_text("log", str(msg), global_step=_tb_step)
        except Exception:
            pass
        _tb_step += 1
