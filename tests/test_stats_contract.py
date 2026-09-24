"""The row this repo writes to ``stats.jsonl`` must be readable by combra.

``combra.metrics.load_fid_by_kimg`` reads ``Metrics/combra_fid`` and
``Progress/kimg`` from the same JSON line, and shape-filters away any record whose
values are not plain scalars -- silently, returning ``{}``. That is exactly how
san-v2 and StyleSwin runs produced an unreadable metric history while every
combra-side test passed: the reader was tested against a synthetic *flat* row, and
nothing tested the producer.

This is the producer half. It builds the real row through the training loop's own
function, so a change to the row shape fails here instead of silently emptying the
analysis layer.
"""

import importlib.util
import json

import pytest

pytest.importorskip("torch")  # the training-loop module imports torch at module level

requires_combra = pytest.mark.skipif(
    importlib.util.find_spec("combra") is None, reason="combra is not installed"
)


def _row(scalars=None):
    from training.training_loop import build_stats_row

    # A tick row with that tick's eval folded in, the way the training loop builds it.
    row = build_stats_row(scalars or {"Loss/loss": 0.25}, 403_200, 7, 1000.0, 900.0)
    row.update({"Metrics/combra_fid": 12.5})
    return row


def _line(row):
    from training.training_loop import stats_line

    return stats_line(row)


def test_row_contains_only_json_scalars():
    for key, value in _row().items():
        assert isinstance(value, (int, float, str)), (
            f"{key} is {type(value).__name__}, not a JSON scalar -- "
            "load_fid_by_kimg will shape-filter this record away"
        )


def test_metrics_share_the_tick_row():
    # One row per tick: the eval's Metrics/* sit beside that tick's scalars and kimg.
    row = json.loads(_line(_row()))
    assert row["Metrics/combra_fid"] == 12.5
    assert row["Loss/loss"] == 0.25
    assert row["Progress/kimg"] == 403.2
    assert row["Progress/tick"] == 7


def test_non_finite_is_null_and_precision_is_full():
    row = json.loads(_line(_row({"Loss/loss": float("nan"), "Timing/sec_per_kimg": 1 / 3,
                                 "Resources/cpu_mem_gb": float("inf")})))
    assert row["Loss/loss"] is None
    assert row["Resources/cpu_mem_gb"] is None
    assert row["Timing/sec_per_kimg"] == 1 / 3


@requires_combra
def test_row_round_trips_through_load_fid_by_kimg(tmp_path):
    from combra.metrics import load_fid_by_kimg

    path = tmp_path / "stats.jsonl"
    no_eval = _row()
    del no_eval["Metrics/combra_fid"]
    path.write_text(_line(no_eval) + "\n" + _line(_row({"Loss/loss": float("nan")})) + "\n")
    assert load_fid_by_kimg(str(path)) == {"000403": 12.5}
