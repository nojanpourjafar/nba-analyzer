import numpy as np
import pytest

from nba_analyzer.market import american_to_prob, decimal_to_prob, fair_probs, vig


def test_american_to_prob():
    assert american_to_prob(-150) == pytest.approx(0.6)
    assert american_to_prob(130) == pytest.approx(100 / 230)
    assert american_to_prob(100) == pytest.approx(0.5)
    assert american_to_prob(-100) == pytest.approx(0.5)
    np.testing.assert_allclose(american_to_prob([-200, 200]), [2 / 3, 1 / 3])


@pytest.mark.parametrize("bad", [0, 50, -99])
def test_american_rejects_impossible_odds(bad):
    with pytest.raises(ValueError):
        american_to_prob(bad)


def test_decimal_to_prob():
    assert decimal_to_prob(2.5) == pytest.approx(0.4)
    with pytest.raises(ValueError):
        decimal_to_prob(1.0)


def test_fair_probs_remove_standard_vig():
    # A typical -110 / -110 line: each side implies 52.4%, 4.8% margin, fair price is 50/50.
    raw_h, raw_a = american_to_prob(-110), american_to_prob(-110)
    assert vig(raw_h, raw_a) == pytest.approx(0.0476, abs=1e-4)
    h, a = fair_probs(raw_h, raw_a)
    assert h == pytest.approx(0.5)
    assert h + a == pytest.approx(1)


def test_fair_probs_keep_the_favourite():
    h, a = fair_probs(american_to_prob(-250), american_to_prob(205))
    assert h + a == pytest.approx(1)
    assert 0.68 < h < 0.71
