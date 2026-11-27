# NBA Prediction Analyzer

[![CI](https://github.com/nojanpourjafar/nba-analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/nojanpourjafar/nba-analyzer/actions/workflows/ci.yml)

Predicts NBA game win probabilities and measures how good those probabilities are.
Later stages compare them against prediction-market prices (paper trading only).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # installs the package in editable mode, plus pytest and ruff
pytest -q                         # offline tests
ruff check . && ruff format --check .
```

## Run

```bash
python scripts/fetch.py --start 2015 --end 2024   # pulls 2015-16 .. 2024-25 into data/games.csv
python scripts/tune.py --burn-in 1 --test 2       # pick all Elo settings on early seasons, test on later ones
python scripts/backtest.py --burn-in 1            # Elo backtest + calibration table
python scripts/backtest.py --odds data/odds.csv   # ... plus model vs market on games with odds
python scripts/backtest.py --home-k 0.5           # learn home advantage as seasons go (0 = fixed)
python scripts/backtest.py --b2b-penalty 30       # teams on a back-to-back count as 30 Elo weaker
python scripts/backtest.py --travel-penalty 20    # ... and 20 Elo weaker per 1,000 extra miles flown
```

The odds file needs the columns `date, home, away, home_ml, away_ml`. The last two are American
moneyline odds, and team names can be written any common
way ("Golden State Warriors", "GS", "GSW"). The market's probabilities have
the bookmaker's margin removed before they're scored.

No network? `python scripts/tune.py --simulate` runs the whole tuning loop on simulated seasons.

`fetch.py` calls stats.nba.com through `nba_api`. The site throttles rapid requests, so each season is
retried with exponential backoff (5s, 10s, 20s, 40s) before giving up.

## Tuning without fooling yourself

`tune.py` splits seasons into three groups: burn-in (warms up ratings, never scored), train
(used to rank parameter combos) and test (the last `--test` seasons, used only to report how the
chosen combo does). Ratings flow continuously through all of them, since that only uses past games,
but the parameters are chosen without looking at test scores. The script prints the held-out score
of both the defaults and the tuned parameters; if tuning doesn't beat the defaults there, the
"improvement" was noise.

## What's here

| File | Purpose |
|---|---|
| `nba_analyzer/data.py` | Fetch seasons, collapse two team rows into one row per game |
| `nba_analyzer/elo.py` | Elo model: home-court advantage (fixed or learned), back-to-back and travel penalties, margin-of-victory multiplier, season regression |
| `nba_analyzer/evaluate.py` | Accuracy, Brier score, log loss, calibration table vs a naive baseline |
| `nba_analyzer/features.py` | Pre-game features: rest days, back-to-backs and travel miles for each team |
| `nba_analyzer/market.py` | Betting odds to probabilities, with the bookmaker margin removed |
| `nba_analyzer/teams.py` | Team-name aliases for odds sources, plus each team's arena location |
| `nba_analyzer/tune.py` | Grid search over Elo parameters (including whether to learn home advantage), train/test split by season |
| `nba_analyzer/simulate.py` | Simulated seasons with known team strengths, for tests and offline demos |
| `scripts/` | Command-line entry points |
| `tests/` | Offline tests, including an Elo check on simulated seasons |

## Reading the backtest

- **Brier / log loss**: lower is better. Beating the baseline (always predicting the league home-win rate) is the first bar.
- **Calibration**: in each bucket, `predicted` should be close to `actual`. If games you call 70% are won 60% of the time, the model is overconfident.
- **Accuracy**: good NBA models land somewhere around the mid-60s percent. Much higher usually means data leakage.
- **Predicted margin**: each game gets a `spread` (expected home margin, 28 Elo points = 1 point). NBA results are noisy, so even sharp betting lines miss by about 10 points on average; beat the "always predict the average margin" baseline first.

## Roadmap

1. ~~Game results + Elo baseline~~
2. ~~Tune `k` and `home_adv` on early seasons, test on later ones~~
3. Add features: ~~rest days / back-to-backs~~, ~~travel~~, injuries
4. Logistic regression / gradient boosting on Elo + features
5. Pull historical market prices and compare model vs market (paper trading only)
6. Dashboard
