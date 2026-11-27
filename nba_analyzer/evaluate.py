"""Score probabilistic predictions: accuracy, Brier score, log loss, calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd


def brier(p: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error of probabilities. Lower is better; a coin flip scores 0.25."""
    return float(np.mean((p - y) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """Lower is better; a coin flip scores ~0.693."""
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def accuracy(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p >= 0.5) == (y == 1)))


def calibration_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Group predictions into probability buckets and compare to how often the home team won.

    A well-calibrated model has predicted ~= actual in every bucket.
    """
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    df = pd.DataFrame({"bucket": idx, "p": p, "y": y})
    table = df.groupby("bucket").agg(games=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))
    table.index = [f"{edges[i]:.1f}-{edges[i + 1]:.1f}" for i in table.index]
    return table.round(3)


def summarize(results: pd.DataFrame) -> dict[str, float]:
    """Compare the model to a naive baseline that always predicts the league home-win rate."""
    p = results["p_home"].to_numpy()
    y = results["home_win"].to_numpy()
    base = np.full_like(p, y.mean(), dtype=float)
    return {
        "games": len(y),
        "accuracy": accuracy(p, y),
        "brier": brier(p, y),
        "log_loss": log_loss(p, y),
        "baseline_brier": brier(base, y),
        "baseline_log_loss": log_loss(base, y),
        "home_win_rate": float(y.mean()),
    }


def compare_to_market(results: pd.DataFrame) -> pd.DataFrame:
    """Score the model and the market side by side, on only the games that have a market price.

    Needs columns p_home, p_market and home_win. Lower Brier / log loss wins.
    """
    priced = results.dropna(subset=["p_market"])
    if priced.empty:
        raise ValueError("No games have a market price to compare against")
    y = priced["home_win"].to_numpy()
    rows = {}
    for name, col in [("model", "p_home"), ("market", "p_market")]:
        p = priced[col].to_numpy()
        rows[name] = {
            "games": len(y),
            "accuracy": accuracy(p, y),
            "brier": brier(p, y),
            "log_loss": log_loss(p, y),
        }
    table = pd.DataFrame(rows).T
    table["games"] = table["games"].astype(int)
    return table
