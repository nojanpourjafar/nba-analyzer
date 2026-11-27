"""Tune Elo parameters on early seasons and report how they do on held-out later seasons.

Usage:  python scripts/tune.py [--burn-in 1] [--test 2] [--metric log_loss]
        python scripts/tune.py --simulate     # offline demo on simulated seasons
"""

import argparse
import math

import pandas as pd

from nba_analyzer.elo import EloModel
from nba_analyzer.simulate import demo_seasons
from nba_analyzer.tune import DEFAULT_GRID, score, tune


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", default="data/games.csv")
    ap.add_argument("--simulate", action="store_true", help="use simulated seasons instead of --games")
    ap.add_argument("--burn-in", type=int, default=1, help="warm-up seasons, scored in neither split")
    ap.add_argument("--test", type=int, default=2, help="number of final seasons held out")
    ap.add_argument("--metric", choices=["log_loss", "brier"], default="log_loss")
    ap.add_argument("--top", type=int, default=10, help="rows of the train leaderboard to print")
    args = ap.parse_args()

    if args.simulate:
        games = demo_seasons()
    else:
        games = pd.read_csv(args.games, parse_dates=["date"], dtype={"game_id": str})

    combos = math.prod(len(v) for v in DEFAULT_GRID.values())
    print(f"Searching {combos} parameter combos ...")
    res = tune(games, burn_in=args.burn_in, test=args.test, metric=args.metric)

    print(f"Train seasons: {res.train_seasons[0]} .. {res.train_seasons[-1]}")
    print(f"Test seasons:  {res.test_seasons[0]} .. {res.test_seasons[-1]}\n")
    print(f"Top {args.top} by train {args.metric}:")
    print(res.table.drop(columns="test").head(args.top).to_string(index=False, float_format="%.4f"))

    d = EloModel()
    defaults = {name: getattr(d, name) for name in DEFAULT_GRID}
    _, default_test = score(games, defaults, res.train_seasons, res.test_seasons, args.metric)
    best_test = res.test_score(res.best)
    print(f"\nHeld-out {args.metric}:")
    print(f"  defaults {defaults}: {default_test:.4f}")
    print(f"  tuned    {res.best}: {best_test:.4f}  ({best_test - default_test:+.4f})")
    flags = " ".join(f"--{k.replace('_', '-')} {v:g}" for k, v in res.best.items())
    print(f"\nTo use them:  python scripts/backtest.py {flags}")


if __name__ == "__main__":
    main()
