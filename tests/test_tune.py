import pytest

from nba_analyzer.simulate import simulate_seasons
from nba_analyzer.tune import split_seasons, tune


@pytest.fixture(scope="module")
def games():
    return simulate_seasons(seasons=5, seed=1)


def test_split_seasons(games):
    train, test = split_seasons(games, burn_in=1, test=2)
    assert train == ["2016-17", "2017-18"]
    assert test == ["2018-19", "2019-20"]


@pytest.mark.parametrize("burn_in,test", [(3, 2), (0, 5), (-1, 1), (1, 0)])
def test_split_seasons_rejects_bad_sizes(games, burn_in, test):
    with pytest.raises(ValueError):
        split_seasons(games, burn_in, test)


def test_tune_picks_by_train_and_beats_bad_params(games):
    grid = {"k": [2, 20], "home_adv": [0, 80], "carryover": [0.75]}
    res = tune(games, grid=grid, burn_in=1, test=2)
    assert len(res.table) == 4
    assert res.table["train"].is_monotonic_increasing
    assert res.best == {k: res.table.loc[0, k] for k in grid}
    # The simulated league has a real home edge and real team differences, so a tiny k with
    # no home advantage should lose on the held-out seasons too.
    assert res.test_score(res.best) < res.test_score({"k": 2, "home_adv": 0, "carryover": 0.75})
    with pytest.raises(KeyError):
        res.test_score({"k": 99, "home_adv": 0, "carryover": 0.75})
