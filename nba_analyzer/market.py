"""Turn betting odds into probabilities that can be compared with the model's.

Bookmaker prices include a margin (the "vig"), so the raw implied probabilities of the two
sides add up to more than 1. `fair_probs` removes it by scaling both sides proportionally.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_analyzer.teams import normalize_teams


def american_to_prob(odds):
    """Implied probability of American (moneyline) odds, vig included.

    -150 means risk 150 to win 100 -> 0.60; +130 means risk 100 to win 130 -> 0.435.
    """
    odds = np.asarray(odds, dtype=float)
    if np.any(np.abs(odds) < 100):
        raise ValueError("American odds must be <= -100 or >= +100")
    return np.where(odds < 0, -odds, 100.0) / (np.abs(odds) + 100)


def decimal_to_prob(odds):
    """Implied probability of decimal odds (2.50 -> 0.40), vig included."""
    odds = np.asarray(odds, dtype=float)
    if np.any(odds <= 1):
        raise ValueError("Decimal odds must be greater than 1")
    return 1 / odds


def fair_probs(p_home_raw, p_away_raw):
    """Remove the bookmaker's margin. Returns (p_home, p_away), which sum to 1."""
    p_home_raw = np.asarray(p_home_raw, dtype=float)
    p_away_raw = np.asarray(p_away_raw, dtype=float)
    total = p_home_raw + p_away_raw
    return p_home_raw / total, p_away_raw / total


def vig(p_home_raw, p_away_raw):
    """Bookmaker margin: how far the raw implied probabilities sum above 1."""
    return np.asarray(p_home_raw, dtype=float) + np.asarray(p_away_raw, dtype=float) - 1


ODDS_COLUMNS = ["date", "home", "away", "home_ml", "away_ml"]


def load_odds(path) -> pd.DataFrame:
    """Read an odds CSV with columns date, home, away, home_ml, away_ml (American odds).

    Team names may be written any common way ("Golden State Warriors", "GS", ...); they are
    converted to the abbreviations data/games.csv uses.
    """
    odds = pd.read_csv(path, parse_dates=["date"])
    missing = set(ODDS_COLUMNS) - set(odds.columns)
    if missing:
        raise ValueError(f"Odds file is missing columns: {sorted(missing)}")
    odds = odds[ODDS_COLUMNS].copy()
    odds["home"] = normalize_teams(odds["home"])
    odds["away"] = normalize_teams(odds["away"])
    return odds


def attach_market(games: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Add the market's margin-free home win probability to each game, matched on date + teams.

    Adds `p_market` (NaN where no odds were found) and `vig`. Raises if the odds contain
    duplicate rows for one game, since there'd be no way to tell which price to trust.
    """
    keys = ["date", "home", "away"]
    dupes = odds.duplicated(keys, keep=False)
    if dupes.any():
        raise ValueError(f"{int(dupes.sum())} odds rows share a date + matchup with another row")
    raw_h = american_to_prob(odds["home_ml"])
    raw_a = american_to_prob(odds["away_ml"])
    priced = odds[keys].assign(p_market=fair_probs(raw_h, raw_a)[0], vig=vig(raw_h, raw_a))
    return games.merge(priced, on=keys, how="left", validate="one_to_one")
