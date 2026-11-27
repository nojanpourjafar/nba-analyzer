"""Game features beyond Elo, built only from information known before tip-off."""

from __future__ import annotations

import pandas as pd


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
