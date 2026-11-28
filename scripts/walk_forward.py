"""Walk-forward test: before each season, tune on earlier seasons only, then score that season.

Usage:  python scripts/walk_forward.py [--burn-in 1] [--min-train 2] [--metric log_loss]
        python scripts/walk_forward.py --simulate     # offline demo on simulated seasons

This is the most honest test of the tuning: it is exactly what you could have done live.
"""

import argparse

import pandas as pd

from nba_analyzer.simulate import demo_seasons
from nba_analyzer.tune import walk_forward


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", default="data/games.csv")
    ap.add_argument("--simulate", action="store_true", help="use simulated seasons instead of --games")
    ap.add_argument("--burn-in", type=int, default=1, help="warm-up seasons, never scored or tuned on")
    ap.add_argument("--min-train", type=int, default=2, help="seasons of history before the first test")
    ap.add_argument("--metric", choices=["log_loss", "brier"], default="log_loss")
    args = ap.parse_args()

    if args.simulate:
        games = demo_seasons()
    else:
        games = pd.read_csv(args.games, parse_dates=["date"], dtype={"game_id": str})

    seasons = sorted(games["season"].unique())
    targets = max(len(seasons) - args.burn_in - args.min_train, 0)
    print(f"Re-tuning before each of {targets} seasons; this can take a minute or two ...", flush=True)

    def report(row):
        gain = row["default"] - row["tuned"]
        print(f"  {row['season']} done: tuned {row['tuned']:.4f} vs default {row['default']:.4f}", flush=True)
        if gain < 0:
            print("    (defaults did better this season)", flush=True)

    table = walk_forward(
        games, burn_in=args.burn_in, min_train=args.min_train, metric=args.metric, progress=report
    )
    print()
    table["gain"] = table["default"] - table["tuned"]
    print(f"Each season scored with parameters tuned only on the seasons before it ({args.metric}):\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}" if abs(v) < 10 else f"{v:g}"))
    wins = int((table["gain"] > 0).sum())
    avg = table["gain"].mean()
    print(f"\nTuning beat the defaults in {wins} of {len(table)} seasons; average gain {avg:+.4f}.")


if __name__ == "__main__":
    main()
