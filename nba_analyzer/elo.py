"""Elo ratings for NBA teams (modelled on FiveThirtyEight's NBA Elo).

- Win probability: 1 / (1 + 10^(-diff/400)), where diff includes home-court advantage.
- Margin-of-victory multiplier, damped for heavy favourites so blowouts by strong teams
  don't inflate ratings.
- Between seasons, ratings regress partway back to the league mean.
- Optionally (home_k > 0), home-court advantage is learned as games are played, so it can
  follow league-wide shifts such as the 2020 bubble or the long-run decline in home edge.
- Optionally (b2b_penalty > 0), a team on the second night of a back-to-back is treated as
  that many Elo points weaker for the game. Ratings themselves are not changed by it.
- Optionally (travel_penalty > 0), each 1,000 miles a team flew more than its opponent since
  their previous games costs it that many Elo points for the game.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nba_analyzer.features import add_rest_days, add_travel

CONTEXT_COLUMNS = ["home_b2b", "away_b2b", "home_travel", "away_travel"]
ELO_PER_POINT = 28.0  # FiveThirtyEight's conversion: 28 Elo points ~ 1 point of margin


@dataclass
class EloModel:
    k: float = 20.0
    home_adv: float = 100.0
    mean: float = 1505.0
    carryover: float = 0.75  # fraction of a rating kept going into a new season
    home_k: float = 0.0  # learning rate for home_adv; 0 keeps it fixed
    b2b_penalty: float = 0.0  # Elo points subtracted from a team playing a back-to-back
    travel_penalty: float = 0.0  # Elo points per 1,000 miles flown more than the opponent
    ratings: dict[str, float] = field(default_factory=dict)

    def rating(self, team: str) -> float:
        return self.ratings.get(team, self.mean)

    def _fatigue(self, home_b2b=0, away_b2b=0, home_travel=0, away_travel=0) -> float:
        """Elo points the home side loses to schedule fatigue (negative if the away side does)."""
        b2b = self.b2b_penalty * (home_b2b - away_b2b)
        return b2b + self.travel_penalty * (home_travel - away_travel) / 1000

    def _diff(self, home: str, away: str, fatigue: float = 0.0) -> float:
        """Home rating edge for this game: ratings, home court, and schedule fatigue."""
        return self.rating(home) + self.home_adv - self.rating(away) - fatigue

    def win_prob(self, home: str, away: str, **ctx: float) -> float:
        """Probability the home team wins.

        ctx may hold home_b2b / away_b2b (0 or 1) and home_travel / away_travel (miles).
        """
        diff = self._diff(home, away, self._fatigue(**ctx))
        return 1.0 / (1.0 + 10 ** (-diff / 400.0))

    def spread(self, home: str, away: str, **ctx: float) -> float:
        """Expected home margin in points (positive = home favoured)."""
        return self._diff(home, away, self._fatigue(**ctx)) / ELO_PER_POINT

    def update(self, home: str, away: str, home_pts: int, away_pts: int, **ctx: float) -> float:
        """Apply one game result. Returns the PRE-game home win probability."""
        return self._step(home, away, home_pts, away_pts, self._fatigue(**ctx))[0]

    def _step(self, home: str, away: str, home_pts: int, away_pts: int, fatigue: float):
        """Apply one game result given its fatigue adjustment. Returns (pre-game p, diff)."""
        elo_diff = self._diff(home, away, fatigue)
        p_home = 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))
        home_won = home_pts > away_pts
        mov = abs(home_pts - away_pts)

        winner_diff = elo_diff if home_won else -elo_diff
        mov_mult = ((mov + 3) ** 0.8) / (7.5 + 0.006 * winner_diff)

        surprise = (1.0 if home_won else 0.0) - p_home
        shift = self.k * mov_mult * surprise
        self.ratings[home] = self.rating(home) + shift
        self.ratings[away] = self.rating(away) - shift
        # Home teams beating expectations league-wide nudges the home edge up, and vice versa.
        self.home_adv += self.home_k * mov_mult * surprise
        return p_home, elo_diff

    def new_season(self) -> None:
        for team, r in self.ratings.items():
            self.ratings[team] = self.carryover * r + (1 - self.carryover) * self.mean


def run_elo(games: pd.DataFrame, model: EloModel | None = None) -> tuple[pd.DataFrame, EloModel]:
    """Walk games in date order, recording each game's pre-game probability and spread.

    Only information from before each game is used, so the output is safe to backtest on.
    Back-to-back flags (and travel miles, if travel_penalty is used) are added from the
    schedule when the games don't have them yet.
    """
    model = model or EloModel()
    games = games.sort_values(["date", "game_id"]).reset_index(drop=True)
    if not {"home_b2b", "away_b2b"} <= set(games.columns):
        games = add_rest_days(games)
    if model.travel_penalty and not {"home_travel", "away_travel"} <= set(games.columns):
        games = add_travel(games)
    # Fatigue depends only on the schedule, so work it out for every game at once.
    ctx = {c: games[c].to_numpy() for c in CONTEXT_COLUMNS if c in games.columns}
    fatigue = model._fatigue(**ctx) if ctx else np.zeros(len(games))
    fatigue = np.broadcast_to(fatigue, len(games)).tolist()
    columns = [games[c].tolist() for c in ("season", "home", "away", "home_pts", "away_pts")]
    probs, spreads, current_season = [], [], None
    for season, home, away, home_pts, away_pts, tired in zip(*columns, fatigue, strict=True):
        if current_season is not None and season != current_season:
            model.new_season()
        current_season = season
        p_home, diff = model._step(home, away, home_pts, away_pts, tired)
        probs.append(p_home)
        spreads.append(diff / ELO_PER_POINT)  # from the pre-game rating gap
    out = games.copy()
    out["p_home"] = probs
    out["spread"] = spreads
    out["home_win"] = (out["home_pts"] > out["away_pts"]).astype(int)
    return out, model
