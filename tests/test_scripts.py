"""Smoke test: the backtest script runs end to end on simulated games plus odds."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nba_analyzer.simulate import simulate_seasons

ROOT = Path(__file__).resolve().parents[1]


def test_backtest_with_odds(tmp_path):
    games = simulate_seasons(seasons=2, teams=6, games_per_team=20)
    games.to_csv(tmp_path / "games.csv", index=False)
    odds = games[["date", "home", "away"]].iloc[::2].assign(home_ml=-150, away_ml=130)
    odds.to_csv(tmp_path / "odds.csv", index=False)

    out = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "backtest.py"),
            "--games",
            str(tmp_path / "games.csv"),
            "--odds",
            str(tmp_path / "odds.csv"),
            "--out",
            str(tmp_path / "pred.csv"),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "Model vs market (30 of 60 scored games have odds)" in out
    assert "market" in out
    pred = pd.read_csv(tmp_path / "pred.csv")
    assert np.isclose(pred["p_market"].dropna(), 0.6 / (0.6 + 100 / 230)).all()
