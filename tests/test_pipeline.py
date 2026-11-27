"""Offline tests: reshaping logic plus an Elo sanity check on simulated seasons."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nba_analyzer.data import season_label, to_games  # noqa: E402
from nba_analyzer.elo import EloModel, run_elo  # noqa: E402
from nba_analyzer.evaluate import summarize  # noqa: E402


def test_to_games_pairs_home_and_away():
    raw = pd.DataFrame(
        {
            "GAME_ID": ["001", "001", "002", "002"],
            "GAME_DATE": ["2024-01-02", "2024-01-02", "2024-01-01", "2024-01-01"],
            "TEAM_ABBREVIATION": ["BOS", "NYK", "LAL", "GSW"],
            "MATCHUP": ["BOS vs. NYK", "NYK @ BOS", "LAL @ GSW", "GSW vs. LAL"],
            "PTS": [110, 102, 99, 120],
        }
    )
    g = to_games(raw, "2023-24")
    assert list(g["game_id"]) == ["002", "001"]  # sorted by date
    assert g.iloc[1][["home", "away", "home_pts", "away_pts"]].tolist() == ["BOS", "NYK", 110, 102]
    assert g.iloc[0][["home", "away"]].tolist() == ["GSW", "LAL"]


def test_season_label():
    assert season_label(2023) == "2023-24"
    assert season_label(1999) == "1999-00"


def test_elo_update_is_zero_sum_and_rewards_winner():
    m = EloModel()
    p = m.update("A", "B", 110, 100)
    assert 0.5 < p < 1  # home advantage makes the home team the favourite
    assert m.rating("A") > m.mean > m.rating("B")
    assert abs(m.rating("A") + m.rating("B") - 2 * m.mean) < 1e-9


def _simulate(seasons=4, teams=30, games_per_team=82, seed=0):
    """Teams with fixed true strengths play random games; margin ~ N(strength diff + 3, 12)."""
    rng = np.random.default_rng(seed)
    names = [f"T{i:02d}" for i in range(teams)]
    strength = dict(zip(names, rng.normal(0, 5, teams)))
    rows, gid, day = [], 0, pd.Timestamp("2015-10-27")
    for s in range(seasons):
        for _ in range(teams * games_per_team // 2):
            h, a = rng.choice(names, 2, replace=False)
            margin = rng.normal(strength[h] - strength[a] + 3, 12)
            margin = margin if abs(margin) >= 1 else 1.0
            hp = 105 + max(margin, 0)
            ap = 105 + max(-margin, 0)
            rows.append((f"{gid:06d}", season_label(2015 + s), day, h, a, int(round(hp)), int(round(ap))))
            gid += 1
            day += pd.Timedelta(hours=8)
    return pd.DataFrame(rows, columns=["game_id", "season", "date", "home", "away", "home_pts", "away_pts"])


def test_elo_beats_baseline_on_simulated_data():
    games = _simulate()
    results, _ = run_elo(games)
    scored = results[results["season"] != results["season"].iloc[0]]
    s = summarize(scored)
    assert s["brier"] < s["baseline_brier"]
    assert s["accuracy"] > 0.6
