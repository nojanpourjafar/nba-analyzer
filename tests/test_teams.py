import pandas as pd
import pytest

from nba_analyzer.teams import ABBREVIATIONS, normalize_team, normalize_teams


@pytest.mark.parametrize(
    "name,abbr",
    [
        ("Golden State Warriors", "GSW"),
        ("GoldenState", "GSW"),
        ("gs", "GSW"),
        ("Warriors", "GSW"),
        ("LA Clippers", "LAC"),
        ("LAClippers", "LAC"),
        ("Los Angeles Lakers", "LAL"),
        ("Philadelphia 76ers", "PHI"),
        ("Portland Trail Blazers", "POR"),
        ("PHO", "PHX"),
        ("utah", "UTA"),
        ("WSH", "WAS"),
        ("NewOrleans", "NOP"),
        ("BOS", "BOS"),
    ],
)
def test_normalize_team(name, abbr):
    assert normalize_team(name) == abbr


def test_every_abbreviation_maps_to_itself():
    assert len(ABBREVIATIONS) == 30
    assert all(normalize_team(a) == a for a in ABBREVIATIONS)


def test_los_angeles_alone_is_ambiguous():
    with pytest.raises(KeyError):
        normalize_team("Los Angeles")


def test_normalize_teams_lists_all_unknowns():
    with pytest.raises(ValueError, match=r"\['Seattle', 'Vancouver'\]"):
        normalize_teams(pd.Series(["Boston", "Seattle", "Vancouver", "Seattle"]))
