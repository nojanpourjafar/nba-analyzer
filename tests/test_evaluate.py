import numpy as np
import pytest

from nba_analyzer.evaluate import accuracy, brier, calibration_table, log_loss


def test_coin_flip_reference_scores():
    p = np.full(4, 0.5)
    y = np.array([1, 0, 1, 0])
    assert brier(p, y) == pytest.approx(0.25)
    assert log_loss(p, y) == pytest.approx(np.log(2))


def test_perfect_predictions():
    y = np.array([1, 0, 1])
    p = y.astype(float)
    assert brier(p, y) == 0
    assert log_loss(p, y) < 1e-9  # clipped, not infinite
    assert accuracy(p, y) == 1


def test_confident_miss_is_finite():
    assert np.isfinite(log_loss(np.array([0.0]), np.array([1])))


def test_calibration_table_buckets():
    p = np.array([0.05, 0.15, 0.95, 1.0])
    y = np.array([0, 0, 1, 1])
    t = calibration_table(p, y)
    assert list(t.index) == ["0.0-0.1", "0.1-0.2", "0.9-1.0"]  # 1.0 lands in the top bucket
    assert t.loc["0.9-1.0", "games"] == 2
    assert t.loc["0.9-1.0", "actual"] == 1.0
