import pandas as pd

from nba_analyzer.features import add_rest_days
from nba_analyzer.simulate import simulate_seasons


def _games(rows):
    return pd.DataFrame(rows, columns=["game_id", "season", "date", "home", "away"]).assign(
        date=lambda d: pd.to_datetime(d["date"])
    )


def test_rest_days_and_back_to_backs():
    g = add_rest_days(
        _games(
            [
                ("1", "2023-24", "2024-01-01", "BOS", "NYK"),
                ("2", "2023-24", "2024-01-02", "NYK", "MIA"),  # NYK on a back-to-back
                ("3", "2023-24", "2024-01-04", "MIA", "BOS"),  # MIA 2 days, BOS 3 days
                ("4", "2023-24", "2024-01-20", "BOS", "NYK"),  # long breaks are capped
            ]
        )
    )
    assert g[["home_rest", "away_rest"]].values.tolist() == [[4, 4], [1, 4], [2, 3], [4, 4]]
    assert g["home_b2b"].tolist() == [0, 1, 0, 0]
    assert g["away_b2b"].sum() == 0


def test_rest_resets_each_season():
    g = add_rest_days(
        _games(
            [
                ("1", "2022-23", "2023-04-09", "BOS", "NYK"),
                ("2", "2023-24", "2023-04-10", "BOS", "NYK"),  # new season, not a back-to-back
            ]
        )
    )
    assert g["home_rest"].tolist() == [4, 4]


def test_rest_ignores_time_of_day_and_keeps_rows():
    games = _games(
        [
            ("1", "2023-24", "2024-01-01 22:00", "BOS", "NYK"),
            ("2", "2023-24", "2024-01-02 12:00", "BOS", "MIA"),  # next calendar day: back-to-back
        ]
    )
    g = add_rest_days(games)
    assert len(g) == 2
    assert g.loc[1, "home_b2b"] == 1


def test_simulated_schedule_looks_like_the_nba():
    g = add_rest_days(simulate_seasons(seasons=1, seed=1))
    assert len(g) == 30 * 82 // 2
    assert (g[["home_rest", "away_rest"]] >= 1).all().all()  # never two games in one day
    b2b_share = (g["home_b2b"].mean() + g["away_b2b"].mean()) / 2
    assert 0.12 < b2b_share < 0.22
    assert 150 < g["date"].dt.normalize().nunique() < 200  # roughly a real season's length
