from __future__ import annotations

import pandas as pd
import pytest

from ucl import names
from ucl.data import (
    _parse_of_score,
    parse_footballcsv,
    parse_openfootball,
    season_winners,
)

OF_SAMPLE = """\
= UEFA Champions League 2023/24

▪ Group, Matchday 1
  Tue Sep 19 2023
    18:45  AC Milan (ITA)          v Newcastle United FC (ENG)  0-0
           BSC Young Boys (SUI)    v RB Leipzig (GER)         1-3 (1-1)

▪ Finals, Round of 16
  Tue Feb 13 2024
    21:00  FC Bayern München (GER) v SS Lazio (ITA)           0-1 (0-0)
  Wed Mar 6
    21:00  SS Lazio (ITA)          v FC Bayern München (GER)  0-3 (0-1)

▪ Finals, Final
  Sat Jun 1
    21:00  Borussia Dortmund (GER) v Real Madrid CF (ESP)     0-2 (0-0)
"""

FC_SAMPLE = pd.DataFrame(
    {
        "Stage": ["Qualifying", "Knockout", "Knockout"],
        "Round": ["Qual. Round 1 | Leg 1", "Round of 16 | Leg 2", "Final"],
        "Group": [None] * 3,
        "Date": ["(Tue) 30 Jun 2015 (W26)", "(Wed) 10 Mar 2004 (W11)", "(Sat) 26 May 2004 (W21)"],
        "Team 1": ["Crusaders › NIR (1)", "Real Madrid › ESP (2)", "AS Monaco › FRA (1)"],
        "FT": ["0-0", "1-0 (*)", "0-3"],
        "HT": ["0-0", "0-0", "0-1"],
        "Team 2": ["Levadia › EST (1)", "Bayern München › GER (3)", "FC Porto › POR (1)"],
        "∑FT": [None, None, None],
        "ET": [None, None, None],
        "P": [None, None, None],
        "Comments": [None] * 3,
    }
)


def test_parse_openfootball_basic():
    df = parse_openfootball(OF_SAMPLE, 2023)
    assert len(df) == 5
    assert df["date"].min() == pd.Timestamp("2023-09-19").date()
    assert (
        df.loc[df["round"] == "Final", "date"].iloc[0] == pd.Timestamp("2024-06-01").date()
    )  # year rolls over
    young_boys = df[df["home"].str.contains("Young Boys")].iloc[0]
    assert (young_boys["home_goals"], young_boys["away_goals"]) == (1, 3)
    assert set(df["stage"]) == {"group", "knockout", "final"}


@pytest.mark.parametrize(
    ("raw", "ft", "et", "pen"),
    [
        ("2-1 (1-0)", (2, 1), None, None),
        ("0-0", (0, 0), None, None),
        ("4-3 pen. 1-1 a.e.t. (1-1, 0-1)", (1, 1), (1, 1), (4, 3)),
        ("5-0 a.e.t. (3-0, 1-0)", (3, 0), (5, 0), None),
    ],
)
def test_parse_openfootball_scores(raw, ft, et, pen):
    out = _parse_of_score(raw)
    assert out["ft"] == ft and out["et"] == et and out["pen"] == pen


def test_parse_footballcsv_drops_qualifying_and_parses_scores():
    df = parse_footballcsv(FC_SAMPLE, 2003)
    assert len(df) == 2
    assert df["stage"].tolist() == ["knockout", "final"]
    assert df["leg"].tolist()[0] == 2
    assert df.iloc[1]["home_goals"] == 0 and df.iloc[1]["away_goals"] == 3
    assert df.iloc[0]["home_country"] == "ESP"


def test_name_normalisation_merges_spelling_variants():
    same = [
        ("Bayern München", "GER"),
        ("FC Bayern München", "GER"),
    ]
    assert len({names.canonical_team(n, c) for n, c in same}) == 1
    assert names.canonical_team("Olympique Lyonnais", "FRA") == names.canonical_team("Olympique Lyon", "FRA")
    assert names.canonical_team("AS Monaco FC", "MCO") == names.canonical_team("Monaco", "FRA")
    assert names.canonical_team("Nordsjælland", "DEN") == names.canonical_team("Nordsjaelland", "DEN")
    # the country keeps homonyms apart
    assert names.canonical_team("Dinamo Kiev", "UKR") != names.canonical_team("Dinamo Kiev", "RUS")


def test_alias_targets_are_stable():
    for target in names._ALIASES.values():
        assert names._ALIASES.get(target, target) == target  # no alias chains


@pytest.mark.data
def test_real_dataset_integrity(real_matches):
    m = real_matches
    assert m["season"].nunique() == 34 and m["season"].min() == 1992 and m["season"].max() == 2025
    assert not m[["home", "away", "home_goals", "away_goals", "date"]].isna().any().any()
    assert (m["home"] != m["away"]).all()
    winners = season_winners(m)
    assert len(winners) == 34
    assert winners[2004] == "liverpool|ENG" and winners[2019] == "bayern munchen|GER"
    assert winners[2022] == "manchester city|ENG" and winners[2008] == "barcelona|ESP"
    # a fixture is played once per (season, home, away, round)
    assert not m.duplicated(["season", "round", "home", "away", "date"]).any()
    assert m.groupby("season").size().min() >= 61


@pytest.mark.data
def test_cross_source_parsers_agree_on_overlapping_seasons():
    """footballcsv and openfootball both cover 2011-2014: the 90' scores must agree."""
    import pandas as pd

    from ucl.config import RAW_DIR
    from ucl.data import season_label

    fc_path = RAW_DIR / "check_fc"
    if not (RAW_DIR / "of_2011-12.txt").exists():
        pytest.skip("dataset not downloaded")
    import urllib.request

    from ucl.data import FOOTBALLCSV_SHA

    fc_path.mkdir(exist_ok=True)
    frames_fc, frames_of = [], []
    for y in (2011, 2012, 2013, 2014):
        label = season_label(y)
        f = fc_path / f"{label}.csv"
        if not f.exists():
            url = (
                "https://raw.githubusercontent.com/footballcsv/europe-champions-league/"
                f"{FOOTBALLCSV_SHA}/{label}/champs.csv"
            )
            try:
                f.write_bytes(urllib.request.urlopen(url, timeout=30).read())
            except OSError:
                pytest.skip("network unavailable")
        frames_fc.append(parse_footballcsv(pd.read_csv(f), y))
        frames_of.append(parse_openfootball((RAW_DIR / f"of_{label}.txt").read_text(encoding="utf-8"), y))
    fc, of = pd.concat(frames_fc), pd.concat(frames_of)
    for d in (fc, of):
        d["home"] = [names.canonical_team(n, c) for n, c in zip(d["home"], d["home_country"], strict=True)]
        d["away"] = [names.canonical_team(n, c) for n, c in zip(d["away"], d["away_country"], strict=True)]
        d["key"] = d["home"] + ">" + d["away"] + "@" + d["date"].astype(str)
    both = fc.merge(of, on="key", suffixes=("_fc", "_of"))
    assert (both["home_goals_fc"] == both["home_goals_of"]).all()
    assert (both["away_goals_fc"] == both["away_goals_of"]).all()
    # The only fixtures not matched are finals, which the two files list with the teams swapped.
    rest = fc[~fc["key"].isin(both["key"])].copy()
    assert (rest["round"] == "Final").all()
    rest["swapped"] = rest["away"] + ">" + rest["home"] + "@" + rest["date"].astype(str)
    mirrored = rest.merge(of, left_on="swapped", right_on="key", suffixes=("_fc", "_of"))
    assert len(both) + len(mirrored) == len(fc) == len(of)
    assert (mirrored["home_goals_fc"] == mirrored["away_goals_of"]).all()
