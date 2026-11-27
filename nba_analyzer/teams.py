"""Map the many ways odds sources write team names onto nba_api's abbreviations.

"Golden State Warriors", "GoldenState", "Warriors", "GS" and "gsw" all become "GSW".
Matching ignores case, spaces and punctuation. Covers the 30 current franchises.
"""

from __future__ import annotations

import re

import pandas as pd

# abbreviation, city, nickname, extra aliases (other sites' codes and spellings)
_TEAMS = [
    ("ATL", "Atlanta", "Hawks", []),
    ("BOS", "Boston", "Celtics", []),
    ("BKN", "Brooklyn", "Nets", ["BRK", "BKLYN"]),
    ("CHA", "Charlotte", "Hornets", ["CHO", "CHH"]),
    ("CHI", "Chicago", "Bulls", []),
    ("CLE", "Cleveland", "Cavaliers", ["Cavs"]),
    ("DAL", "Dallas", "Mavericks", ["Mavs"]),
    ("DEN", "Denver", "Nuggets", []),
    ("DET", "Detroit", "Pistons", []),
    ("GSW", "Golden State", "Warriors", ["GS", "GOS"]),
    ("HOU", "Houston", "Rockets", []),
    ("IND", "Indiana", "Pacers", []),
    ("LAC", "Los Angeles", "Clippers", ["LA Clippers", "L.A. Clippers"]),
    ("LAL", "Los Angeles", "Lakers", ["LA Lakers", "L.A. Lakers"]),
    ("MEM", "Memphis", "Grizzlies", []),
    ("MIA", "Miami", "Heat", []),
    ("MIL", "Milwaukee", "Bucks", []),
    ("MIN", "Minnesota", "Timberwolves", ["Wolves"]),
    ("NOP", "New Orleans", "Pelicans", ["NO", "NOR"]),
    ("NYK", "New York", "Knicks", ["NY"]),
    ("OKC", "Oklahoma City", "Thunder", ["OKL"]),
    ("ORL", "Orlando", "Magic", []),
    ("PHI", "Philadelphia", "76ers", ["Sixers", "PHL"]),
    ("PHX", "Phoenix", "Suns", ["PHO"]),
    ("POR", "Portland", "Trail Blazers", ["Blazers"]),
    ("SAC", "Sacramento", "Kings", []),
    ("SAS", "San Antonio", "Spurs", ["SA"]),
    ("TOR", "Toronto", "Raptors", []),
    ("UTA", "Utah", "Jazz", ["UTAH", "UTH"]),
    ("WAS", "Washington", "Wizards", ["WSH"]),
]


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _build_aliases() -> dict[str, str]:
    aliases: dict[str, str] = {}
    shared_cities = {"Los Angeles"}  # two teams: a city alone is ambiguous
    for abbr, city, nick, extra in _TEAMS:
        names = [abbr, nick, f"{city} {nick}", *extra]
        if city not in shared_cities:
            names.append(city)
        for name in names:
            key = _key(name)
            if aliases.get(key, abbr) != abbr:
                raise AssertionError(f"Alias {name!r} is ambiguous")
            aliases[key] = abbr
    return aliases


ALIASES = _build_aliases()
ABBREVIATIONS = sorted({abbr for abbr, *_ in _TEAMS})


def normalize_team(name: str) -> str:
    """One team name -> abbreviation. Raises KeyError for names it doesn't know."""
    try:
        return ALIASES[_key(name)]
    except KeyError:
        raise KeyError(f"Unknown team name: {name!r}") from None


def normalize_teams(names: pd.Series) -> pd.Series:
    """Normalize a column of team names, listing every unknown name at once if any."""
    keys = names.map(_key)
    unknown = sorted(set(names[~keys.isin(ALIASES)].astype(str)))
    if unknown:
        raise ValueError(f"Unknown team names: {unknown}")
    return keys.map(ALIASES)
