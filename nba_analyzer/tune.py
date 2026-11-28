"""Pick Elo parameters on early seasons, then check them on later seasons the search never saw.

Ratings are carried straight through from train into test seasons (that only uses past
information), but parameters are chosen using train-season scores alone.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import pandas as pd

from nba_analyzer.elo import EloModel, run_elo
from nba_analyzer.evaluate import summarize
from nba_analyzer.features import add_rest_days, add_travel

DEFAULT_GRID: dict[str, list[float]] = {
    "k": [10, 15, 20, 25],
    "home_adv": [60, 80, 100, 120],
    "carryover": [0.65, 0.8, 1.0],
    "home_k": [0.0, 0.5, 1.0],  # 0 = fixed home advantage
    "b2b_penalty": [0, 50],  # Elo points; roughly 28 Elo per point of margin
    "travel_penalty": [0, 20],  # Elo points per 1,000 extra miles flown
}


@dataclass
class TuneResult:
    best: dict[str, float]
    table: pd.DataFrame  # one row per parameter combo, sorted by train score
    train_seasons: list[str]
    test_seasons: list[str]

    def test_score(self, params: dict[str, float]) -> float:
        row = self.table
        for name, value in params.items():
            row = row[row[name] == value]
        if row.empty:
            raise KeyError(f"{params} was not part of the grid")
        return float(row["test"].iloc[0])


def split_seasons(games: pd.DataFrame, burn_in: int, test: int) -> tuple[list[str], list[str]]:
    """Return (train, test) season labels; the first `burn_in` seasons belong to neither."""
    seasons = sorted(games["season"].unique())
    if burn_in < 0 or test < 1:
        raise ValueError("burn_in must be >= 0 and test >= 1")
    if burn_in + test >= len(seasons):
        raise ValueError(f"Need more than burn_in + test = {burn_in + test} seasons, data has {len(seasons)}")
    return seasons[burn_in:-test], seasons[-test:]


def score(games: pd.DataFrame, params: dict[str, float], train: list[str], test: list[str], metric: str):
    results, _ = run_elo(games, EloModel(**params))
    return (
        summarize(results[results["season"].isin(train)])[metric],
        summarize(results[results["season"].isin(test)])[metric],
    )


def tune(
    games: pd.DataFrame,
    grid: dict[str, list[float]] | None = None,
    burn_in: int = 1,
    test: int = 2,
    metric: str = "log_loss",
) -> TuneResult:
    """Grid-search EloModel parameters, ranking combos by `metric` on train seasons only.

    `metric` is any lower-is-better key from evaluate.summarize ("log_loss" or "brier").
    """
    grid = grid or DEFAULT_GRID
    train, test_seasons = split_seasons(games, burn_in, test)
    games = add_rest_days(games)  # once, rather than inside every run
    if any(grid.get("travel_penalty", [0])):
        games = add_travel(games)
    names = list(grid)
    rows = []
    for values in itertools.product(*(grid[n] for n in names)):
        params = dict(zip(names, values, strict=True))
        tr, te = score(games, params, train, test_seasons, metric)
        rows.append({**params, "train": tr, "test": te})
    table = pd.DataFrame(rows).sort_values("train", kind="stable").reset_index(drop=True)
    best = {n: table.loc[0, n].item() for n in names}
    return TuneResult(best=best, table=table, train_seasons=train, test_seasons=test_seasons)


def walk_forward(
    games: pd.DataFrame,
    grid: dict[str, list[float]] | None = None,
    burn_in: int = 1,
    min_train: int = 2,
    metric: str = "log_loss",
    progress=None,
) -> pd.DataFrame:
    """Tune before every season using only earlier seasons, then score that season.

    This is how the model would have been used live: for each target season (after the
    burn-in and at least `min_train` training seasons), parameters are picked on the seasons
    before it and judged on it alone. Returns one row per target season with the chosen
    parameters, their score, and the default parameters' score on the same season.
    `progress`, if given, is called with each row as soon as its season is done.
    """
    seasons = sorted(games["season"].unique())
    first = burn_in + min_train
    if first >= len(seasons):
        raise ValueError(f"Need more than burn_in + min_train = {first} seasons, data has {len(seasons)}")
    defaults = EloModel()
    rows = []
    for target in seasons[first:]:
        history = games[games["season"] <= target]
        res = tune(history, grid=grid, burn_in=burn_in, test=1, metric=metric)
        names = list(res.best)
        default_params = {n: getattr(defaults, n) for n in names}
        _, default_score = score(history, default_params, res.train_seasons, [target], metric)
        tuned_score = res.test_score(res.best)
        rows.append({"season": target, **res.best, "tuned": tuned_score, "default": default_score})
        if progress:
            progress(rows[-1])
    return pd.DataFrame(rows)
