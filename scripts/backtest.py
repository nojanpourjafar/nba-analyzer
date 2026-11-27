"""Run the Elo model over data/games.csv and report how good its probabilities are.

Usage:  python scripts/backtest.py [--burn-in 1]
The first --burn-in seasons are used only to warm up ratings and are left out of the scores.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nba_analyzer.elo import EloModel, run_elo  # noqa: E402
from nba_analyzer.evaluate import calibration_table, summarize  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="data/games.csv")
    ap.add_argument("--burn-in", type=int, default=1)
    ap.add_argument("--k", type=float, default=20.0)
    ap.add_argument("--home-adv", type=float, default=100.0)
    ap.add_argument("--out", default="data/predictions.csv")
    args = ap.parse_args()

    games = pd.read_csv(args.games, parse_dates=["date"])
    results, model = run_elo(games, EloModel(k=args.k, home_adv=args.home_adv))
    results.to_csv(args.out, index=False)

    seasons = sorted(results["season"].unique())
    scored = results[results["season"].isin(seasons[args.burn_in:])]

    print(f"Scored seasons: {seasons[args.burn_in]} .. {seasons[-1]}\n")
    for name, value in summarize(scored).items():
        print(f"  {name:<18} {value:.4f}" if isinstance(value, float) else f"  {name:<18} {value}")
    print("\nCalibration (predicted vs actual home-win rate):")
    print(calibration_table(scored["p_home"].to_numpy(), scored["home_win"].to_numpy()).to_string())
    print("\nTop 5 current ratings:")
    for team, r in sorted(model.ratings.items(), key=lambda kv: -kv[1])[:5]:
        print(f"  {team}  {r:.0f}")


if __name__ == "__main__":
    main()
