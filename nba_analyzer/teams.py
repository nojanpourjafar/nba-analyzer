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


# Home arena (latitude, longitude). Approximate: good to a few miles, which is plenty for
# travel distance. Uses current arenas for teams that moved within the same metro area.
ARENAS: dict[str, tuple[float, float]] = {
    "ATL": (33.757, -84.396),
    "BOS": (42.366, -71.062),
    "BKN": (40.683, -73.975),
    "CHA": (35.225, -80.839),
    "CHI": (41.881, -87.674),
    "CLE": (41.496, -81.688),
    "DAL": (32.790, -96.810),
    "DEN": (39.749, -105.008),
    "DET": (42.341, -83.055),
    "GSW": (37.768, -122.388),
    "HOU": (29.751, -95.362),
    "IND": (39.764, -86.155),
    "LAC": (34.043, -118.267),
    "LAL": (34.043, -118.267),
    "MEM": (35.138, -90.051),
    "MIA": (25.781, -80.188),
    "MIL": (43.045, -87.917),
    "MIN": (44.979, -93.276),
    "NOP": (29.949, -90.082),
    "NYK": (40.751, -73.993),
    "OKC": (35.463, -97.515),
    "ORL": (28.539, -81.384),
    "PHI": (39.901, -75.172),
    "PHX": (33.446, -112.071),
    "POR": (45.532, -122.667),
    "SAC": (38.580, -121.500),
    "SAS": (29.427, -98.438),
    "TOR": (43.643, -79.379),
    "UTA": (40.768, -111.901),
    "WAS": (38.898, -77.021),
}
