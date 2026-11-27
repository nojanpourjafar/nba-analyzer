"""Game features beyond Elo, built only from information known before tip-off."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_analyzer.teams import ARENAS


def add_rest_days(games: pd.DataFrame, cap: int = 4) -> pd.DataFrame:
    """Add home_rest / away_rest: days since each team's previous game in the same season.

    1 means a back-to-back (played yesterday). Values are capped at `cap`, since an extra
    day off matters little past a few days; a team's first game of a season also gets `cap`.
    Adds home_b2b / away_b2b flags (1 on the second night of a back-to-back).
    """
    games = games.sort_values(["date", "game_id"]).reset_index(drop=True)
    # One row per team per game, so each team's schedule can be walked in order.
    sides = pd.concat(
        [
            games[["game_id", "season", "date", "home"]].rename(columns={"home": "team"}).assign(side="home"),
            games[["game_id", "season", "date", "away"]].rename(columns={"away": "team"}).assign(side="away"),
        ]
    ).sort_values(["date", "game_id"], kind="stable")
    prev = sides.groupby(["season", "team"])["date"].shift()
    days = (sides["date"].dt.normalize() - prev.dt.normalize()).dt.days
    sides["rest"] = days.fillna(cap).clip(upper=cap).astype(int)
    rest = sides.pivot(index="game_id", columns="side", values="rest")

    out = games.copy()
    out["home_rest"] = out["game_id"].map(rest["home"]).astype(int)
    out["away_rest"] = out["game_id"].map(rest["away"]).astype(int)
    out["home_b2b"] = (out["home_rest"] == 1).astype(int)
    out["away_b2b"] = (out["away_rest"] == 1).astype(int)
    return out


def miles_between(lat1, lon1, lat2, lon2):
    """Great-circle distance in miles (haversine); works on scalars or arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def add_travel(games: pd.DataFrame) -> pd.DataFrame:
    """Add home_travel / away_travel: miles from each team's previous game site to this one.

    Every game is assumed to be at the home team's arena (rare neutral-site games, such as
    international or NBA Cup games, are treated as home games). A team's first game of a
    season counts as travelling from its own arena.
    """
    unknown = sorted(set(games["home"]) - set(ARENAS))
    if unknown:
        raise ValueError(f"No arena location for: {unknown}")
    games = games.sort_values(["date", "game_id"]).reset_index(drop=True)
    sides = pd.concat(
        [
            games[["game_id", "season", "date", "home"]].assign(team=games["home"], side="home"),
            games[["game_id", "season", "date", "home"]].assign(team=games["away"], side="away"),
        ]
    ).sort_values(["date", "game_id"], kind="stable")
    site = sides["home"]  # where this game is played
    prev_site = sides.groupby(["season", "team"])["home"].shift().fillna(sides["team"])
    lat, lon = site.map(lambda t: ARENAS[t][0]), site.map(lambda t: ARENAS[t][1])
    plat, plon = prev_site.map(lambda t: ARENAS[t][0]), prev_site.map(lambda t: ARENAS[t][1])
    sides["miles"] = miles_between(plat, plon, lat, lon).round()
    miles = sides.pivot(index="game_id", columns="side", values="miles")

    out = games.copy()
    out["home_travel"] = out["game_id"].map(miles["home"]).astype(int)
    out["away_travel"] = out["game_id"].map(miles["away"]).astype(int)
    return out
