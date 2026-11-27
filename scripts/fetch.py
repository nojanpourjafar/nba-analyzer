"""Download regular-season results to data/games.csv.

Usage:  python scripts/fetch.py --start 2015 --end 2025
(--end is the start year of the last season, so 2025 means 2025-26.)
"""

import argparse
import logging
from pathlib import Path

from nba_analyzer.data import fetch_seasons, season_label


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2015)
    ap.add_argument("--end", type=int, default=2024, help="start year of the last season (2024 = 2024-25)")
    ap.add_argument("--out", default="data/games.csv")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    seasons = [season_label(y) for y in range(args.start, args.end + 1)]
    print(f"Fetching {seasons[0]} .. {seasons[-1]}")
    games = fetch_seasons(seasons)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    games.to_csv(out, index=False)
    print(f"Saved {len(games)} games to {out}")


if __name__ == "__main__":
    main()
