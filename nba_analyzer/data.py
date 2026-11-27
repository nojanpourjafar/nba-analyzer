"""Fetch NBA game results with nba_api and reshape them to one row per game."""
from __future__ import annotations

import time

import pandas as pd

GAME_COLUMNS = ["game_id", "season", "date", "home", "away", "home_pts", "away_pts"]


def fetch_season_raw(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """Raw team-game rows for one season, e.g. season="2023-24".

    nba_api returns two rows per game (one per team).
    """
    from nba_api.stats.endpoints import leaguegamefinder  # imported lazily so tests don't need network

    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00",  # NBA only
        season_type_nullable=season_type,
        timeout=60,
    )
    return finder.get_data_frames()[0]


def to_games(raw: pd.DataFrame, season: str) -> pd.DataFrame:
    """Collapse two team rows into one row per game with home/away columns.

    MATCHUP looks like "BOS vs. NYK" for the home team and "NYK @ BOS" for the away team.
    """
    cols = ["GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION", "MATCHUP", "PTS"]
    df = raw[cols].dropna(subset=["PTS"])
    home = df[df["MATCHUP"].str.contains(" vs. ", regex=False)]
    away = df[df["MATCHUP"].str.contains(" @ ", regex=False)]
    g = home.merge(away, on="GAME_ID", suffixes=("_home", "_away"))

    games = pd.DataFrame(
        {
            "game_id": g["GAME_ID"],
            "season": season,
            "date": pd.to_datetime(g["GAME_DATE_home"]),
            "home": g["TEAM_ABBREVIATION_home"],
            "away": g["TEAM_ABBREVIATION_away"],
            "home_pts": g["PTS_home"].astype(int),
            "away_pts": g["PTS_away"].astype(int),
        }
    )
    return games.sort_values(["date", "game_id"]).reset_index(drop=True)


def fetch_seasons(seasons: list[str], pause: float = 1.5) -> pd.DataFrame:
    """Fetch several seasons, pausing between requests so stats.nba.com doesn't throttle us."""
    frames = []
    for i, season in enumerate(seasons):
        if i:
            time.sleep(pause)
        frames.append(to_games(fetch_season_raw(season), season))
    return pd.concat(frames, ignore_index=True)


def season_label(start_year: int) -> str:
    """2023 -> "2023-24"."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"
