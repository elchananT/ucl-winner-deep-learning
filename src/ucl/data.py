"""Download, parse and harmonise open UEFA Champions League match data.

Two CC0 sources are combined (both pinned to an immutable git commit):

* ``footballcsv/europe-champions-league``  -> seasons 1992/93 .. 2010/11 (CSV)
* ``openfootball/champions-league``        -> seasons 2011/12 .. 2025/26 (Football.TXT)

Only the *main competition* is kept (league / group stages and knockout
rounds); qualifying rounds are discarded because they involve hundreds of
tiny clubs with almost no history.  Scores are the **90-minute** result; the
extra-time / penalty outcome is stored separately and only used to decide who
advanced or won the title.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from ucl.config import FIRST_SEASON, LAST_SEASON, MATCHES_CSV, RAW_DIR
from ucl.names import canonical_team

FOOTBALLCSV_SHA = "66393c8dc5c86b459dc700a0ae3a0bd0abedb56e"
OPENFOOTBALL_SHA = "abfaeddc2ee3d14f99ecc163c9ddb46cb4d67cef"
FOOTBALLCSV_LAST_SEASON = 2010  # 2010/11 is the last season taken from footballcsv
CHECKSUM_FILE = Path(__file__).with_name("checksums.json")

_FC_URL = (
    "https://raw.githubusercontent.com/footballcsv/europe-champions-league/"
    f"{FOOTBALLCSV_SHA}/{{label}}/champs.csv"
)
_OF_URL = (
    f"https://raw.githubusercontent.com/openfootball/champions-league/{OPENFOOTBALL_SHA}/{{label}}/cl.txt"
)

MATCH_COLUMNS = [
    "season",
    "date",
    "stage",
    "round",
    "group",
    "leg",
    "home",
    "away",
    "home_country",
    "away_country",
    "home_goals",
    "away_goals",
    "et_home_goals",
    "et_away_goals",
    "pen_home",
    "pen_away",
    "source",
]


def season_label(season: int) -> str:
    """``2022`` -> ``"2022-23"``."""
    return f"{season}-{(season + 1) % 100:02d}"


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RawFile:
    season: int
    source: str
    url: str
    path: Path


def raw_files(raw_dir: Path = RAW_DIR) -> list[RawFile]:
    out = []
    for season in range(FIRST_SEASON, LAST_SEASON + 1):
        label = season_label(season)
        if season <= FOOTBALLCSV_LAST_SEASON:
            out.append(
                RawFile(season, "footballcsv", _FC_URL.format(label=label), raw_dir / f"fc_{label}.csv")
            )
        else:
            out.append(
                RawFile(season, "openfootball", _OF_URL.format(label=label), raw_dir / f"of_{label}.txt")
            )
    return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(raw_dir: Path = RAW_DIR, *, force: bool = False, write_checksums: bool = False) -> list[Path]:
    """Fetch every raw file (cached) and verify it against ``checksums.json``.

    Because the URLs are pinned to commit hashes the bytes can never change; a
    checksum mismatch therefore means a corrupted download and is a hard error.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    expected = json.loads(CHECKSUM_FILE.read_text()) if CHECKSUM_FILE.exists() else {}
    observed: dict[str, str] = {}
    paths = []
    for rf in raw_files(raw_dir):
        if force or not rf.path.exists():
            with urllib.request.urlopen(rf.url, timeout=60) as resp:
                rf.path.write_bytes(resp.read())
        digest = _sha256(rf.path)
        observed[rf.path.name] = digest
        if not write_checksums and expected.get(rf.path.name, digest) != digest:
            raise RuntimeError(f"checksum mismatch for {rf.path.name}; delete it and re-run")
        paths.append(rf.path)
    if write_checksums:
        CHECKSUM_FILE.write_text(json.dumps(observed, indent=2, sort_keys=True) + "\n")
    return paths


# --------------------------------------------------------------------------- #
# Parsing: openfootball (Football.TXT)
# --------------------------------------------------------------------------- #
_MONTHS = {
    m: i
    for i, m in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1
    )
}
_OF_DATE = re.compile(r"^\s+[A-Z][a-z]{2}\s+([A-Z][a-z]{2})\s+(\d{1,2})(?:\s+(\d{4}))?\s*$")
_OF_MATCH = re.compile(
    r"^\s+(?:\d{1,2}:\d{2}\s+)?(?P<home>.+?)\s+\((?P<hc>[A-Z]{3})\)\s+v\s+"
    r"(?P<away>.+?)\s+\((?P<ac>[A-Z]{3})\)\s+(?P<score>\d+-\d+.*)$"
)
_PAIR = re.compile(r"(\d+)-(\d+)")


def _parse_of_score(score: str) -> dict:
    """Decode e.g. ``4-3 pen. 1-1 a.e.t. (1-1, 0-1)`` into 90' / ET / penalty parts."""
    out: dict = {"et": None, "pen": None}
    if "a.e.t." in score:
        paren = re.search(r"\(([^)]*)\)", score)
        ft = _PAIR.search(paren.group(1)) if paren else None
        if ft is None:
            raise ValueError(f"cannot find 90-minute score in {score!r}")
        out["ft"] = (int(ft.group(1)), int(ft.group(2)))
        head = score.split("a.e.t.")[0]
        pairs = _PAIR.findall(head)
        if "pen." in head:
            out["pen"] = (int(pairs[0][0]), int(pairs[0][1]))
            out["et"] = (int(pairs[1][0]), int(pairs[1][1]))
        else:
            out["et"] = (int(pairs[0][0]), int(pairs[0][1]))
    else:
        first = _PAIR.match(score)
        out["ft"] = (int(first.group(1)), int(first.group(2)))
    return out


def _of_stage(header: str) -> tuple[str, str, str | None]:
    """Map a ``▪`` header to ``(stage, round, group)``."""
    h = header.strip()
    m = re.match(r"^Gr(?:oup|uppe)\s+([A-H])$", h)
    if m:
        return "group", f"Group {m.group(1)}", m.group(1)
    if h.startswith("Group,"):
        return "group", h.split(",", 1)[1].strip(), None
    if h.startswith("League,"):
        return "league", h.split(",", 1)[1].strip(), None
    if h.startswith("Playoffs,"):
        return "playoff", "Playoffs", None
    rnd = h.split(",", 1)[1].strip() if h.startswith("Finals,") else h
    rnd = {"Final": "Final"}.get(rnd, rnd)
    return ("final" if rnd == "Final" else "knockout"), rnd, None


def parse_openfootball(text: str, season: int, source: str = "openfootball") -> pd.DataFrame:
    rows = []
    stage = rnd = group = None
    current: date | None = None
    for line in text.splitlines():
        if line.startswith("▪"):
            stage, rnd, group = _of_stage(line.lstrip("▪ ").strip())
            continue
        d = _OF_DATE.match(line)
        if d:
            mon, day, year = _MONTHS[d.group(1)], int(d.group(2)), d.group(3)
            yr = int(year) if year else (season if mon >= 7 else season + 1)
            current = date(yr, mon, day)
            continue
        m = _OF_MATCH.match(line)
        if not m:
            continue
        if current is None or stage is None:
            raise ValueError(f"match before date/stage header: {line!r}")
        sc = _parse_of_score(m.group("score"))
        rows.append(
            {
                "season": season,
                "date": current,
                "stage": stage,
                "round": rnd,
                "group": group,
                "home": m.group("home"),
                "away": m.group("away"),
                "home_country": m.group("hc"),
                "away_country": m.group("ac"),
                "home_goals": sc["ft"][0],
                "away_goals": sc["ft"][1],
                "et_home_goals": sc["et"][0] if sc["et"] else None,
                "et_away_goals": sc["et"][1] if sc["et"] else None,
                "pen_home": sc["pen"][0] if sc["pen"] else None,
                "pen_away": sc["pen"][1] if sc["pen"] else None,
                "source": source,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Parsing: footballcsv
# --------------------------------------------------------------------------- #
_FC_TEAM = re.compile(r"^(?P<name>.+?)\s+›\s+(?P<country>[A-Z]{3})\s+\(\d+\)$")
_FC_DATE = re.compile(r"(\d{1,2})\s+([A-Z][a-z]{2})\s+(\d{4})")
_FC_QUAL = re.compile(r"^(Qual\.|Prelim\.|Playoff Round)")


def _fc_stage(stage: str | float, rnd: str) -> tuple[str, str, str | None]:
    base = rnd.split("|")[0].strip()
    if base == "Final":
        return "final", "Final", None
    if isinstance(stage, str) and stage.startswith("Group"):
        return "group", base, None
    return "knockout", base, None


def parse_footballcsv(df: pd.DataFrame, season: int, source: str = "footballcsv") -> pd.DataFrame:
    rows = []
    for rec in df.to_dict("records"):
        rnd_raw = str(rec["Round"])
        if rec["Stage"] == "Qualifying" or _FC_QUAL.match(rnd_raw):
            continue
        th, ta = _FC_TEAM.match(rec["Team 1"]), _FC_TEAM.match(rec["Team 2"])
        if not (th and ta):
            raise ValueError(f"unparseable team in {rec}")
        d = _FC_DATE.search(str(rec["Date"]))
        when = datetime.strptime(f"{d.group(1)} {d.group(2)} {d.group(3)}", "%d %b %Y").date()
        ft = _PAIR.match(str(rec["FT"]))
        stage, rnd, group = _fc_stage(rec["Stage"], rnd_raw)
        leg = re.search(r"Leg (\d)", rnd_raw)
        et = _PAIR.match(str(rec["ET"])) if isinstance(rec["ET"], str) else None
        pen = _PAIR.match(str(rec["P"])) if isinstance(rec["P"], str) else None
        rows.append(
            {
                "season": season,
                "date": when,
                "stage": stage,
                "round": rnd,
                "group": group,
                "leg": int(leg.group(1)) if leg else None,
                "home": th.group("name"),
                "away": ta.group("name"),
                "home_country": th.group("country"),
                "away_country": ta.group("country"),
                "home_goals": int(ft.group(1)),
                "away_goals": int(ft.group(2)),
                "et_home_goals": int(et.group(1)) if et else None,
                "et_away_goals": int(et.group(2)) if et else None,
                "pen_home": int(pen.group(1)) if pen else None,
                "pen_away": int(pen.group(2)) if pen else None,
                "source": source,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def _infer_legs(df: pd.DataFrame) -> pd.Series:
    """Leg number (1, 2, or NaN for single matches) for knockout fixtures.

    A knockout pairing that appears twice in the same round is a two-legged tie;
    the earlier date is leg 1.  Groups / league phase get ``NaN``.
    """
    leg = pd.Series(float("nan"), index=df.index, dtype="float")
    ko = df[df["stage"].isin(["knockout", "playoff"])]
    pair_key = ko.apply(lambda r: (r["season"], r["round"], frozenset((r["home"], r["away"]))), axis=1)
    for _, idx in ko.groupby(pair_key).groups.items():
        if len(idx) == 2:
            order = df.loc[idx].sort_values("date", kind="stable").index
            leg.loc[order[0]], leg.loc[order[1]] = 1, 2
    return leg


def build_matches(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Parse every raw file and return one tidy, chronologically sorted frame."""
    frames = []
    for rf in raw_files(raw_dir):
        if not rf.path.exists():
            raise FileNotFoundError(f"{rf.path} missing - run `make data` first")
        if rf.source == "footballcsv":
            frames.append(parse_footballcsv(pd.read_csv(rf.path), rf.season))
        else:
            frames.append(parse_openfootball(rf.path.read_text(encoding="utf-8"), rf.season))
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["home"] = [canonical_team(n, c) for n, c in zip(df["home"], df["home_country"], strict=True)]
    df["away"] = [canonical_team(n, c) for n, c in zip(df["away"], df["away_country"], strict=True)]
    # openfootball does not label legs; footballcsv does.  Fill the gaps.
    inferred = _infer_legs(df)
    df["leg"] = df["leg"].fillna(inferred)
    df = df.sort_values(["date", "stage", "home"], kind="stable").reset_index(drop=True)
    df.insert(0, "match_id", range(len(df)))
    return df[["match_id", *MATCH_COLUMNS]]


def load_matches(path: Path = MATCHES_CSV) -> pd.DataFrame:
    """Load the processed match table (build it with ``python -m ucl.data``)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `make data` first")
    return pd.read_csv(path, parse_dates=["date"])


def season_winners(matches: pd.DataFrame) -> dict[int, str]:
    """Champion of each season = winner of its final (ET / penalties included)."""
    out = {}
    for season, g in matches[matches["stage"] == "final"].groupby("season"):
        if len(g) != 1:
            raise ValueError(f"season {season}: expected one final, found {len(g)}")
        r = g.iloc[0]
        if pd.notna(r["pen_home"]):
            home_won = r["pen_home"] > r["pen_away"]
        elif pd.notna(r["et_home_goals"]):
            home_won = r["et_home_goals"] > r["et_away_goals"]
        else:
            home_won = r["home_goals"] > r["away_goals"]
        out[int(season)] = r["home"] if home_won else r["away"]
    return out


def main() -> None:
    """``python -m ucl.data``: download, verify checksums and write matches.csv."""
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    ap.add_argument("--write-checksums", action="store_true", help="(maintainers) refresh checksums.json")
    args = ap.parse_args()
    download(force=args.force, write_checksums=args.write_checksums)
    df = build_matches()
    MATCHES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MATCHES_CSV, index=False)
    print(f"wrote {MATCHES_CSV} - {len(df)} matches, {df['season'].nunique()} seasons")


if __name__ == "__main__":
    main()
