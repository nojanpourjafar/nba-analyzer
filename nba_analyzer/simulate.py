"""Simulated NBA-like seasons with known team strengths, for offline tests and demos."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_analyzer.data import GAME_COLUMNS, season_label
from nba_analyzer.teams import ABBREVIATIONS


def simulate_seasons(
    seasons: int = 4,
    teams: int = 30,
    games_per_team: int = 82,
    home_edge: float | list[float] = 3.0,
    strength_sd: float = 5.0,
    drift_sd: float = 2.0,
    margin_sd: float = 12.0,
    seed: int = 0,
    start_year: int = 2015,
) -> pd.DataFrame:
    """Random schedules; home margin ~ N(strength_home - strength_away + home_edge, margin_sd).

    Team strengths (in points) drift by N(0, drift_sd) between seasons, so carrying ratings
    over matters but isn't the whole story. `home_edge` may be a list with one value per
    season, to simulate the home edge changing over time. Output has the same columns as
    data/games.csv.
    """
    edges = list(home_edge) if isinstance(home_edge, (list, tuple)) else [home_edge] * seasons
    if len(edges) != seasons:
        raise ValueError("home_edge needs one value per season")
    if not 2 <= teams <= len(ABBREVIATIONS):
        raise ValueError(f"teams must be between 2 and {len(ABBREVIATIONS)}")
    rng = np.random.default_rng(seed)
    names = ABBREVIATIONS[:teams]  # real abbreviations, so simulated data looks like the real thing
    strength = rng.normal(0, strength_sd, teams)
    rows, gid = [], 0
    for s in range(seasons):
        if s:
            strength = strength + rng.normal(0, drift_sd, teams)
        day = pd.Timestamp(f"{start_year + s}-10-25")
        for _ in range(teams * games_per_team // 2):
            h, a = rng.choice(teams, 2, replace=False)
            margin = rng.normal(strength[h] - strength[a] + edges[s], margin_sd)
            margin = round(margin) or (1 if margin >= 0 else -1)  # no ties in the NBA
            rows.append(
                (
                    f"{gid:08d}",
                    season_label(start_year + s),
                    day,
                    names[h],
                    names[a],
                    105 + max(margin, 0),
                    105 + max(-margin, 0),
                )
            )
            gid += 1
            day += pd.Timedelta(hours=4)
    return pd.DataFrame(rows, columns=GAME_COLUMNS)
