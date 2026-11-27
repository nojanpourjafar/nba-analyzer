import numpy as np
import pandas as pd
import pytest

from nba_analyzer.market import (
    american_to_prob,
    attach_market,
    decimal_to_prob,
    fair_probs,
    load_odds,
    vig,
)


def test_american_to_prob():
    assert american_to_prob(-150) == pytest.approx(0.6)
    assert american_to_prob(130) == pytest.approx(100 / 230)
    assert american_to_prob(100) == pytest.approx(0.5)
    assert american_to_prob(-100) == pytest.approx(0.5)
    np.testing.assert_allclose(american_to_prob([-200, 200]), [2 / 3, 1 / 3])


@pytest.mark.parametrize("bad", [0, 50, -99])
def test_american_rejects_impossible_odds(bad):
    with pytest.raises(ValueError):
        american_to_prob(bad)


def test_decimal_to_prob():
    assert decimal_to_prob(2.5) == pytest.approx(0.4)
    with pytest.raises(ValueError):
        decimal_to_prob(1.0)


def test_fair_probs_remove_standard_vig():
    # A typical -110 / -110 line: each side implies 52.4%, 4.8% margin, fair price is 50/50.
    raw_h, raw_a = american_to_prob(-110), american_to_prob(-110)
    assert vig(raw_h, raw_a) == pytest.approx(0.0476, abs=1e-4)
    h, a = fair_probs(raw_h, raw_a)
    assert h == pytest.approx(0.5)
    assert h + a == pytest.approx(1)


def test_fair_probs_keep_the_favourite():
    h, a = fair_probs(american_to_prob(-250), american_to_prob(205))
    assert h + a == pytest.approx(1)
    assert 0.68 < h < 0.71


def _games():
    return pd.DataFrame(
        {
            "game_id": ["1", "2", "3"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
            "home": ["BOS", "LAL", "BOS"],
            "away": ["NYK", "GSW", "MIA"],
        }
    )


def test_attach_market_matches_on_date_and_teams(tmp_path):
    path = tmp_path / "odds.csv"
    path.write_text(
        "date,home,away,home_ml,away_ml,extra\n"
        "2024-01-01,Boston Celtics,NY,-110,-110,x\n"
        "2024-01-02,BOS,Heat,-250,205,x\n"
        "2024-01-05,BOS,MIA,-250,205,x\n"  # no such game: ignored
    )
    out = attach_market(_games(), load_odds(path))
    assert list(out["game_id"]) == ["1", "2", "3"]
    assert out.loc[0, "p_market"] == pytest.approx(0.5)
    assert np.isnan(out.loc[1, "p_market"])  # LAL-GSW has no odds
    assert 0.68 < out.loc[2, "p_market"] < 0.71
    assert out.loc[0, "vig"] == pytest.approx(0.0476, abs=1e-4)


def test_load_odds_requires_columns(tmp_path):
    path = tmp_path / "odds.csv"
    path.write_text("date,home,away,home_ml\n2024-01-01,BOS,NYK,-110\n")
    with pytest.raises(ValueError, match="away_ml"):
        load_odds(path)


def test_attach_market_rejects_duplicate_prices():
    odds = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "home": ["BOS", "BOS"],
            "away": ["NYK", "NYK"],
            "home_ml": [-110, -120],
            "away_ml": [-110, 100],
        }
    )
    with pytest.raises(ValueError, match="share a date"):
        attach_market(_games(), odds)
