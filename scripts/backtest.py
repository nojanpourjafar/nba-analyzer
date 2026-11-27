"""Run the Elo model over data/games.csv and report how good its probabilities are.

Usage:  python scripts/backtest.py [--burn-in 1] [--odds data/odds.csv]
The first --burn-in seasons are used only to warm up ratings and are left out of the scores.
With --odds, the model is also scored against the market on the games that have a price.
"""

import argparse

import pandas as pd

from nba_analyzer.elo import EloModel, run_elo
from nba_analyzer.evaluate import calibration_table, compare_to_market, summarize
from nba_analyzer.market import attach_market, load_odds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="data/games.csv")
    ap.add_argument("--burn-in", type=int, default=1)
    ap.add_argument("--k", type=float, default=20.0)
    ap.add_argument("--home-adv", type=float, default=100.0)
    ap.add_argument("--carryover", type=float, default=0.75)
    ap.add_argument("--home-k", type=float, default=0.0, help="learn home advantage at this rate (0 = fixed)")
    ap.add_argument("--odds", help="odds CSV (date, home, away, home_ml, away_ml) to compare against")
    ap.add_argument("--out", default="data/predictions.csv")
    args = ap.parse_args()

    games = pd.read_csv(args.games, parse_dates=["date"], dtype={"game_id": str})
    seasons = sorted(games["season"].unique())
    if not 0 <= args.burn_in < len(seasons):
        ap.error(f"--burn-in must be between 0 and {len(seasons) - 1} (data has {len(seasons)} seasons)")

    model = EloModel(k=args.k, home_adv=args.home_adv, carryover=args.carryover, home_k=args.home_k)
    results, model = run_elo(games, model)
    if args.odds:
        results = attach_market(results, load_odds(args.odds))
    results.to_csv(args.out, index=False)

    scored = results[results["season"].isin(seasons[args.burn_in :])]

    print(f"Scored seasons: {seasons[args.burn_in]} .. {seasons[-1]}\n")
    for name, value in summarize(scored).items():
        print(f"  {name:<18} {value:.4f}" if isinstance(value, float) else f"  {name:<18} {value}")
    print("\nCalibration (predicted vs actual home-win rate):")
    print(calibration_table(scored["p_home"].to_numpy(), scored["home_win"].to_numpy()).to_string())
    if args.odds:
        priced = int(scored["p_market"].notna().sum())
        print(f"\nModel vs market ({priced} of {len(scored)} scored games have odds):")
        if priced:
            print(compare_to_market(scored).to_string(float_format="%.4f"))
    if args.home_k:
        print(f"\nLearned home advantage now: {model.home_adv:.0f} Elo points (started at {args.home_adv:g})")
    print("\nTop 5 current ratings:")
    for team, r in sorted(model.ratings.items(), key=lambda kv: -kv[1])[:5]:
        print(f"  {team}  {r:.0f}")


if __name__ == "__main__":
    main()
