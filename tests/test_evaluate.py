import numpy as np
import pandas as pd
import pytest

from nba_analyzer.evaluate import accuracy, brier, calibration_table, compare_to_market, log_loss


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


def test_compare_to_market_uses_only_priced_games():
    rng = np.random.default_rng(0)
    n = 4000
    truth = rng.uniform(0.2, 0.8, n)
    y = (rng.uniform(size=n) < truth).astype(int)
    model = np.clip(truth + rng.normal(0, 0.15, n), 0.01, 0.99)  # noisy view of the truth
    market = truth.copy()
    market[:500] = np.nan  # some games have no odds
    results = pd.DataFrame({"p_home": model, "p_market": market, "home_win": y})
    t = compare_to_market(results)
    assert list(t.index) == ["model", "market"]
    assert t.loc["model", "games"] == t.loc["market", "games"] == n - 500
    assert t.loc["market", "brier"] < t.loc["model", "brier"]
    assert t.loc["market", "log_loss"] < t.loc["model", "log_loss"]


def test_compare_to_market_needs_prices():
    results = pd.DataFrame({"p_home": [0.6], "p_market": [np.nan], "home_win": [1]})
    with pytest.raises(ValueError):
        compare_to_market(results)
