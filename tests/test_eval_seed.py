"""The in-training eval draws follow the seed rule (spec §2).

Sample ``i``'s noise and label come from ``seed + i`` alone, so the eval set is
the same at any ``--gpus`` and any subset reproduces in isolation. Before this
each rank seeded its own generator with ``seed + rank`` over a per-rank block,
so the set changed with the GPU count.
"""

import pytest

torch = pytest.importorskip("torch")

from training.metrics import _eval_draw  # noqa: E402


def test_eval_draw_is_a_pure_function_of_seed_and_index():
    z1, c1 = _eval_draw(42, 7, 3, 8, 3)
    z2, c2 = _eval_draw(42, 7, 3, 8, 3)
    assert torch.equal(z1, z2) and c1 == c2
    assert z1.shape == (3, 8, 8) and 0 <= c1 < 3
    assert _eval_draw(42, 7, 3, 8, 0)[1] is None  # unconditional: no label


def test_eval_draw_differs_across_indices_and_seeds():
    z_a, _ = _eval_draw(42, 7, 3, 8, 3)
    z_b, _ = _eval_draw(42, 8, 3, 8, 3)
    z_c, _ = _eval_draw(43, 7, 3, 8, 3)
    assert not torch.equal(z_a, z_b) and not torch.equal(z_a, z_c)
