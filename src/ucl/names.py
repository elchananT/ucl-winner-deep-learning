"""Club-name harmonisation across the two source datasets.

The two sources (and even different seasons of the same source) spell clubs
differently - ``"Bayern München"``, ``"FC Bayern München"``, ``"Bayern Munich"``.
Left alone, one club would be split into several "teams" and its rating history
would be fragmented, which silently hurts every model.  The strategy is:

1. normalise (ASCII-fold, lower-case, drop punctuation and generic tokens such
   as ``fc``/``cf``/``sk``);
2. map the few remaining spelling variants through an explicit alias table that
   was built by listing every (country, normalised-name) pair and reviewing
   near-duplicates by hand.
"""

from __future__ import annotations

import re
import unicodedata

_STOP_TOKENS = {
    "fc", "cf", "sc", "ac", "as", "ss", "sk", "fk", "nk", "bk", "if", "ik", "sv", "vfb", "vfl", "tsv",
    "afc", "bsc", "fsv", "rsc", "kv", "kaa", "rcd", "ud", "cd", "sd", "sl", "ssc", "us", "ca", "cs",
    "club", "clube", "calcio", "futbol", "de", "da", "do", "di", "del", "la", "le", "the", "cfr",
    "fotbal", "voetbal", "sad", "1", "jsc", "pfc", "fbc", "ifk", "gnk", "hnk",
    "ofk", "ofc", "gks", "kf", "ks", "rc", "ogc", "sco", "ec", "es", "vv", "sp", "spor", "kulubu",
}  # fmt: skip

# "normalised-variant|COUNTRY" -> "canonical-name|COUNTRY".  Built by listing all
# (country, normalised-name) pairs of the combined data and merging the variants
# of one club by hand (see tests/test_data.py::test_alias_targets_exist).
_ALIASES: dict[str, str] = {
    "rb salzburg|AUT": "red bull salzburg|AUT",
    "qarabag agdam|AZE": "qarabag|AZE",
    "athletic|ESP": "athletic bilbao|ESP",
    "olympique lyonnais|FRA": "olympique lyon|FRA",
    "racing lens|FRA": "lens|FRA",
    "monaco|MCO": "monaco|FRA",
    "bayer 04 leverkusen|GER": "bayer leverkusen|GER",
    "aek athen|GRE": "aek athens|GRE",
    "olympiakos piraeus|GRE": "olympiacos|GRE",
    "pae olympiakos sfp|GRE": "olympiacos|GRE",
    "ferencvarosi tc|HUN": "ferencvaros|HUN",
    "acf fiorentina|ITA": "fiorentina|ITA",
    "atalanta bc|ITA": "atalanta|ITA",
    "internazionale milano|ITA": "inter|ITA",
    "internazionale|ITA": "inter|ITA",
    "lazio roma|ITA": "lazio|ITA",
    "feyenoord rotterdam|NED": "feyenoord|NED",
    "psv eindhoven|NED": "psv|NED",
    "legia warsaw|POL": "legia warszawa|POL",
    "sport lisboa e benfica|POR": "benfica|POR",
    "sporting portugal|POR": "sporting cp|POR",
    "basel 1893|SUI": "basel|SUI",
    "besiktas istanbul jk|TUR": "besiktas|TUR",
    "galatasaray istanbul|TUR": "galatasaray|TUR",
}


def normalise(name: str) -> str:
    s = unicodedata.normalize("NFKD", name)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = (
        s.replace("&", " and ")
        .replace("ø", "o")
        .replace("æ", "ae")
        .replace("œ", "oe")
        .replace("ł", "l")
        .replace("đ", "d")
        .replace("ß", "ss")
    )
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    tokens = [t for t in s.split() if t not in _STOP_TOKENS]
    return " ".join(tokens)


def canonical_team(name: str, country: str) -> str:
    """Stable team key such as ``"bayern munchen|GER"`` (country disambiguates)."""
    full = f"{normalise(name)}|{country}"
    return _ALIASES.get(full, full)
