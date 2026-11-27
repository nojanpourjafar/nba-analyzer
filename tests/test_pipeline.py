"""Offline tests: reshaping logic plus an Elo sanity check on simulated seasons."""

import pandas as pd
import pytest

from nba_analyzer.data import fetch_with_retry, season_label, to_games
from nba_analyzer.elo import EloModel, run_elo
from nba_analyzer.evaluate import summarize
from nba_analyzer.features import add_rest_days
from nba_analyzer.simulate import simulate_seasons


def test_to_games_pairs_home_and_away():
    raw = pd.DataFrame(
        {
            "GAME_ID": ["001", "001", "002", "002"],
            "GAME_DATE": ["2024-01-02", "2024-01-02", "2024-01-01", "2024-01-01"],
            "TEAM_ABBREVIATION": ["BOS", "NYK", "LAL", "GSW"],
            "MATCHUP": ["BOS vs. NYK", "NYK @ BOS", "LAL @ GSW", "GSW vs. LAL"],
            "PTS": [110, 102, 99, 120],
        }
    )
    g = to_games(raw, "2023-24")
    assert list(g["game_id"]) == ["002", "001"]  # sorted by date
    assert g.iloc[1][["home", "away", "home_pts", "away_pts"]].tolist() == ["BOS", "NYK", 110, 102]
    assert g.iloc[0][["home", "away"]].tolist() == ["GSW", "LAL"]


def test_season_label():
    assert season_label(2023) == "2023-24"
    assert season_label(1999) == "1999-00"


def test_elo_update_is_zero_sum_and_rewards_winner():
    m = EloModel()
    p = m.update("A", "B", 110, 100)
    assert 0.5 < p < 1  # home advantage makes the home team the favourite
    assert m.rating("A") > m.mean > m.rating("B")
    assert abs(m.rating("A") + m.rating("B") - 2 * m.mean) < 1e-9


def test_elo_beats_baseline_on_simulated_data():
    games = simulate_seasons()
    results, _ = run_elo(games)
    scored = results[results["season"] != results["season"].iloc[0]]
    s = summarize(scored)
    assert s["brier"] < s["baseline_brier"]
    assert s["accuracy"] > 0.6


def test_to_games_drops_unplayed_rows():
    raw = pd.DataFrame(
        {
            "GAME_ID": ["001", "001"],
            "GAME_DATE": ["2024-01-02", "2024-01-02"],
            "TEAM_ABBREVIATION": ["BOS", "NYK"],
            "MATCHUP": ["BOS vs. NYK", "NYK @ BOS"],
            "PTS": [None, None],
        }
    )
    assert to_games(raw, "2023-24").empty


def test_new_season_regresses_toward_mean():
    m = EloModel(carryover=0.5)
    m.ratings = {"A": 1605.0, "B": 1405.0}
    m.new_season()
    assert m.ratings == {"A": 1555.0, "B": 1455.0}


def test_run_elo_uses_only_pregame_information():
    games = simulate_seasons(seasons=1, teams=4, games_per_team=10)
    full, _ = run_elo(games)
    first_half, _ = run_elo(games.iloc[: len(games) // 2])
    # Appending later games must not change earlier predictions.
    assert full["p_home"].iloc[: len(first_half)].tolist() == first_half["p_home"].tolist()


def test_fetch_with_retry_recovers_from_transient_errors(monkeypatch):
    monkeypatch.setattr("nba_analyzer.data.time.sleep", lambda s: None)
    calls = []

    def flaky(season):
        calls.append(season)
        if len(calls) < 3:
            raise TimeoutError("throttled")
        return pd.DataFrame({"ok": [1]})

    assert fetch_with_retry("2023-24", fetch=flaky)["ok"].tolist() == [1]
    assert len(calls) == 3


def test_fetch_with_retry_gives_up(monkeypatch):
    monkeypatch.setattr("nba_analyzer.data.time.sleep", lambda s: None)

    def broken(season):
        raise TimeoutError("down")

    with pytest.raises(RuntimeError, match="after 3 attempts"):
        fetch_with_retry("2023-24", retries=2, fetch=broken)


def test_home_k_zero_keeps_home_advantage_fixed():
    m = EloModel()
    m.update("A", "B", 90, 110)
    assert m.home_adv == 100.0


def test_learned_home_advantage_tracks_a_vanishing_home_edge():
    # Two normal seasons, then two "bubble" seasons with no home edge at all.
    games = simulate_seasons(seasons=4, home_edge=[3, 3, 0, 0], seed=2)
    fixed, _ = run_elo(games, EloModel(home_adv=80))
    learned, model = run_elo(games, EloModel(home_adv=80, home_k=0.5))
    assert model.home_adv < 40  # it noticed home teams stopped over-performing
    bubble = fixed["season"].isin(["2017-18", "2018-19"])
    assert summarize(learned[bubble])["log_loss"] < summarize(fixed[bubble])["log_loss"]


def test_simulate_needs_one_home_edge_per_season():
    with pytest.raises(ValueError):
        simulate_seasons(seasons=3, home_edge=[3, 0])


def test_simulated_back_to_back_penalty_shows_up_in_results():
    games = add_rest_days(simulate_seasons(seasons=4, b2b_penalty=4, seed=5))
    home_wins = games["home_pts"] > games["away_pts"]
    tired_home = (games["home_b2b"] == 1) & (games["away_b2b"] == 0)
    rested_home = (games["home_b2b"] == 0) & (games["away_b2b"] == 1)
    assert home_wins[tired_home].mean() < home_wins[rested_home].mean() - 0.1


def test_b2b_penalty_lowers_a_tired_teams_chances():
    m = EloModel(b2b_penalty=50)
    assert m.win_prob("A", "B", home_b2b=1) < m.win_prob("A", "B") < m.win_prob("A", "B", away_b2b=1)
    assert m.win_prob("A", "B", home_b2b=1, away_b2b=1) == m.win_prob("A", "B")


def test_elo_with_b2b_penalty_beats_ignoring_fatigue():
    games = simulate_seasons(seasons=4, b2b_penalty=3, seed=6)  # ~85 Elo points
    tired = add_rest_days(games)
    tired = tired["home_b2b"] != tired["away_b2b"]
    plain, _ = run_elo(games, EloModel(home_adv=80))
    aware, _ = run_elo(games, EloModel(home_adv=80, b2b_penalty=80))
    later = plain["season"] != "2015-16"
    assert summarize(aware[later & tired])["log_loss"] < summarize(plain[later & tired])["log_loss"]


def test_travel_penalty_lowers_the_chances_of_the_team_that_flew_further():
    m = EloModel(travel_penalty=20)
    assert m.win_prob("A", "B", home_travel=0, away_travel=2500) > m.win_prob("A", "B")
    assert m.win_prob("A", "B", home_travel=2500, away_travel=0) < m.win_prob("A", "B")
    assert EloModel().win_prob("A", "B", home_travel=2500) == EloModel().win_prob("A", "B")


def test_elo_with_travel_penalty_beats_ignoring_travel():
    games = simulate_seasons(seasons=4, travel_penalty=3, seed=8)  # 3 points per 1,000 miles
    plain, _ = run_elo(games, EloModel(home_adv=80))
    aware, _ = run_elo(games, EloModel(home_adv=80, travel_penalty=80))
    later = plain["season"] != "2015-16"
    assert summarize(aware[later])["log_loss"] < summarize(plain[later])["log_loss"]
