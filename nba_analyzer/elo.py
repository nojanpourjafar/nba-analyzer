"""Elo ratings for NBA teams (modelled on FiveThirtyEight's NBA Elo).

- Win probability: 1 / (1 + 10^(-diff/400)), where diff includes home-court advantage.
- Margin-of-victory multiplier, damped for heavy favourites so blowouts by strong teams
  don't inflate ratings.
- Between seasons, ratings regress partway back to the league mean.
- Optionally (home_k > 0), home-court advantage is learned as games are played, so it can
  follow league-wide shifts such as the 2020 bubble or the long-run decline in home edge.
- Optionally (b2b_penalty > 0), a team on the second night of a back-to-back is treated as
  that many Elo points weaker for the game. Ratings themselves are not changed by it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from nba_analyzer.features import add_rest_days


@dataclass
class EloModel:
    k: float = 20.0
    home_adv: float = 100.0
    mean: float = 1505.0
    carryover: float = 0.75  # fraction of a rating kept going into a new season
    home_k: float = 0.0  # learning rate for home_adv; 0 keeps it fixed
    b2b_penalty: float = 0.0  # Elo points subtracted from a team playing a back-to-back
    ratings: dict[str, float] = field(default_factory=dict)

    def rating(self, team: str) -> float:
        return self.ratings.get(team, self.mean)

    def _diff(self, home: str, away: str, home_b2b: int, away_b2b: int) -> float:
        """Home rating edge for this game, including home court and back-to-back fatigue."""
        fatigue = self.b2b_penalty * (home_b2b - away_b2b)
        return self.rating(home) + self.home_adv - self.rating(away) - fatigue

    def win_prob(self, home: str, away: str, home_b2b: int = 0, away_b2b: int = 0) -> float:
        """Probability the home team wins."""
        diff = self._diff(home, away, home_b2b, away_b2b)
        return 1.0 / (1.0 + 10 ** (-diff / 400.0))

    def update(
        self, home: str, away: str, home_pts: int, away_pts: int, home_b2b: int = 0, away_b2b: int = 0
    ) -> float:
        """Apply one game result. Returns the PRE-game home win probability."""
        p_home = self.win_prob(home, away, home_b2b, away_b2b)
        home_won = home_pts > away_pts
        mov = abs(home_pts - away_pts)

        elo_diff = self._diff(home, away, home_b2b, away_b2b)
        winner_diff = elo_diff if home_won else -elo_diff
        mov_mult = ((mov + 3) ** 0.8) / (7.5 + 0.006 * winner_diff)

        surprise = (1.0 if home_won else 0.0) - p_home
        shift = self.k * mov_mult * surprise
        self.ratings[home] = self.rating(home) + shift
        self.ratings[away] = self.rating(away) - shift
        # Home teams beating expectations league-wide nudges the home edge up, and vice versa.
        self.home_adv += self.home_k * mov_mult * surprise
        return p_home

    def new_season(self) -> None:
        for team, r in self.ratings.items():
            self.ratings[team] = self.carryover * r + (1 - self.carryover) * self.mean


def run_elo(games: pd.DataFrame, model: EloModel | None = None) -> tuple[pd.DataFrame, EloModel]:
    """Walk games in date order, recording each game's pre-game probability.

    Only information from before each game is used, so the output is safe to backtest on.
    Back-to-back flags are added from the schedule if the games don't have them yet.
    """
    model = model or EloModel()
    games = games.sort_values(["date", "game_id"]).reset_index(drop=True)
    if not {"home_b2b", "away_b2b"} <= set(games.columns):
        games = add_rest_days(games)
    probs, current_season = [], None
    for row in games.itertuples(index=False):
        if current_season is not None and row.season != current_season:
            model.new_season()
        current_season = row.season
        probs.append(model.update(row.home, row.away, row.home_pts, row.away_pts, row.home_b2b, row.away_b2b))
    out = games.copy()
    out["p_home"] = probs
    out["home_win"] = (out["home_pts"] > out["away_pts"]).astype(int)
    return out, model
