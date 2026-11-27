# NBA Prediction Analyzer

Predicts NBA game win probabilities and measures how good those probabilities are.
Later stages compare them against prediction-market prices (paper trading only).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q          # offline tests, should print "4 passed"
```

## Run

```bash
python scripts/fetch.py --start 2015 --end 2024   # pulls 2015-16 .. 2024-25 into data/games.csv
python scripts/backtest.py --burn-in 1            # Elo backtest + calibration table
```

`fetch.py` calls stats.nba.com through `nba_api`. If it hangs or times out, wait a minute and retry;
the site throttles rapid requests.

## What's here

| File | Purpose |
|---|---|
| `nba_analyzer/data.py` | Fetch seasons, collapse two team rows into one row per game |
| `nba_analyzer/elo.py` | Elo model: home-court advantage, margin-of-victory multiplier, season regression |
| `nba_analyzer/evaluate.py` | Accuracy, Brier score, log loss, calibration table vs a naive baseline |
| `scripts/` | Command-line entry points |
| `tests/` | Offline tests, including an Elo check on simulated seasons |

## Reading the backtest

- **Brier / log loss**: lower is better. Beating the baseline (always predicting the league home-win rate) is the first bar.
- **Calibration**: in each bucket, `predicted` should be close to `actual`. If games you call 70% are won 60% of the time, the model is overconfident.
- **Accuracy**: good NBA models land somewhere around the mid-60s percent. Much higher usually means data leakage.

## Roadmap

1. ~~Game results + Elo baseline~~
2. Tune `k` and `home_adv` on early seasons, test on later ones
3. Add features: rest days / back-to-backs, travel, injuries
4. Logistic regression / gradient boosting on Elo + features
5. Pull historical market prices and compare model vs market (paper trading only)
6. Dashboard
