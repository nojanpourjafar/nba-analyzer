"""Simulated NBA-like seasons with known team strengths, for offline tests and demos."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_analyzer.data import GAME_COLUMNS, season_label
from nba_analyzer.features import miles_between
from nba_analyzer.teams import ABBREVIATIONS, ARENAS


def simulate_seasons(
    seasons: int = 4,
    teams: int = 30,
    games_per_team: int = 82,
    home_edge: float | list[float] = 3.0,
    strength_sd: float = 5.0,
    drift_sd: float = 2.0,
    margin_sd: float = 12.0,
    b2b_penalty: float = 0.0,
    travel_penalty: float = 0.0,
    b2b_rate: float = 0.17,
    seed: int = 0,
    start_year: int = 2015,
) -> pd.DataFrame:
    """Random schedules; home margin ~ N(strength_home - strength_away + home_edge, margin_sd).

    Team strengths (in points) drift by N(0, drift_sd) between seasons, so carrying ratings
    over matters but isn't the whole story. `home_edge` may be a list with one value per
    season, to simulate the home edge changing over time.

    Schedule: one game per team per day at most. A team that played yesterday plays again
    with probability `b2b_rate` (so about that share of games are back-to-backs, as in the
    NBA); a rested team plays with probability 1 - b2b_rate, giving a game every other day
    on average. A team on the second night of a back-to-back plays `b2b_penalty` points worse,
    and each 1,000 miles a team flew since its last game costs it `travel_penalty` points.
    Output has the same columns as data/games.csv.
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
        played_yesterday = np.zeros(teams, dtype=bool)
        last_site = list(range(teams))  # each team starts the season at home
        target, played = teams * games_per_team // 2, 0
        while played < target:
            p_play = np.where(played_yesterday, b2b_rate, 1 - b2b_rate)
            today = rng.permutation(np.flatnonzero(rng.uniform(size=teams) < p_play))
            today = today[: min(len(today) // 2 * 2, 2 * (target - played))]
            for h, a in today.reshape(-1, 2):
                tired = b2b_penalty * (int(played_yesterday[h]) - int(played_yesterday[a]))
                if travel_penalty:
                    flown = [miles_between(*ARENAS[names[last_site[t]]], *ARENAS[names[h]]) for t in (h, a)]
                    tired += travel_penalty * (flown[0] - flown[1]) / 1000
                last_site[h] = last_site[a] = h
                margin = rng.normal(strength[h] - strength[a] + edges[s] - tired, margin_sd)
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
            played += len(today) // 2
            played_yesterday = np.zeros(teams, dtype=bool)
            played_yesterday[today] = True
            day += pd.Timedelta(days=1)
    return pd.DataFrame(rows, columns=GAME_COLUMNS)


def demo_seasons() -> pd.DataFrame:
    """Eight simulated seasons with every effect the model knows about, for offline demos.

    The home edge drifts down, collapses for a fan-less season, then partly recovers; teams
    on a back-to-back play 2 points worse, and 1 point worse per 1,000 miles flown.
    """
    return simulate_seasons(
        seasons=8,
        home_edge=[3.5, 3.3, 3.1, 2.9, 2.7, 0.5, 2.0, 2.2],
        b2b_penalty=2.0,
        travel_penalty=1.0,
    )
