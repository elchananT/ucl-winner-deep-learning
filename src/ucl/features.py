"""Leakage-safe, walk-forward feature construction.

The whole module revolves around :class:`RatingState`, a small mutable object
that summarises *everything known about every team up to now*: its Elo rating,
how many matches it has played and its most recent results.  The one rule that
keeps the pipeline honest is the order of operations used by
:func:`build_features`:

1. for a given match day, read the pre-match features of **every** match of
   that day from the state;
2. only then feed that day's results into the state.

A match therefore never influences its own features, nor those of any match
played on the same day or earlier.  ``tests/test_leakage.py`` proves this by
corrupting the results of the future and checking that earlier rows are
bit-for-bit unchanged.

The very same state object is used at prediction and simulation time
(:meth:`RatingState.pair_features`), so there is no train/serve skew.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

BASE_ELO = 1500.0
FORM_WINDOW = 5
FORM_DECAY = 0.8  # weight of the k-th most recent match is FORM_DECAY**k
GOALS_WINDOW = 10
FORM_PRIOR_WEIGHT = 2.0
GOALS_PRIOR_WEIGHT = 3.0
PRIOR_POINTS = 1.35  # mean points per team-match in the training era
PRIOR_GOALS = 1.4  # mean goals scored by a team per match

TEAM_FEATURES = ("elo", "form_pts", "form_gd", "att", "dfn", "log_n")
CONTEXT_FEATURES = ("neutral", "is_knockout", "is_second_leg", "is_final")


@dataclass(frozen=True)
class EloParams:
    """Hyper-parameters of the rating system (selected on validation data only)."""

    k: float = 25.0
    home_advantage: float = 75.0
    season_regression: float = 0.15  # pull towards BASE_ELO at each new season
    country_prior_weight: float = 0.5  # weight of the country mean for a debutant


@dataclass
class TeamRecord:
    country: str
    elo: float
    n: int = 0
    # most recent first; each entry = (points, goal_diff, goals_for, goals_against)
    recent: deque = field(default_factory=lambda: deque(maxlen=GOALS_WINDOW))


def expected_score(rating_diff: float) -> float:
    """Elo expectation for the first team given ``rating_h - rating_a`` (home bonus included)."""
    return 1.0 / (1.0 + 10.0 ** (-rating_diff / 400.0))


def goal_multiplier(goal_diff: int) -> float:
    g = abs(goal_diff)
    if g <= 1:
        return 1.0
    if g == 2:
        return 1.5
    return (11.0 + g) / 8.0


class RatingState:
    """Everything known about all teams *so far* (Elo, experience, recent form)."""

    def __init__(self, params: EloParams | None = None):
        self.params = params or EloParams()
        self.teams: dict[str, TeamRecord] = {}
        self.season: int | None = None

    # ------------------------------------------------------------------ #
    def _country_mean(self, country: str) -> float | None:
        elos = [t.elo for t in self.teams.values() if t.country == country and t.n > 0]
        return float(np.mean(elos)) if elos else None

    def initial_elo(self, country: str) -> float:
        """Prior rating of a team that has never been seen: shrunk country mean."""
        mean = self._country_mean(country)
        if mean is None:
            return BASE_ELO
        w = self.params.country_prior_weight
        return w * mean + (1 - w) * BASE_ELO

    def _get(self, name: str, country: str) -> TeamRecord:
        if name not in self.teams:
            self.teams[name] = TeamRecord(country=country, elo=self.initial_elo(country))
        return self.teams[name]

    # ------------------------------------------------------------------ #
    def team_features(self, name: str, country: str) -> dict[str, float]:
        """Pre-match features of one team (does not mutate the state)."""
        rec = self.teams.get(name)
        if rec is None:
            elo, n, recent = self.initial_elo(country), 0, ()
        else:
            elo, n, recent = rec.elo, rec.n, tuple(rec.recent)
        w = np.array([FORM_DECAY**k for k in range(min(len(recent), FORM_WINDOW))])
        pts = np.array([r[0] for r in recent[:FORM_WINDOW]], dtype=float)
        gd = np.array([r[1] for r in recent[:FORM_WINDOW]], dtype=float)
        form_pts = (float(w @ pts) + FORM_PRIOR_WEIGHT * PRIOR_POINTS) / (w.sum() + FORM_PRIOR_WEIGHT)
        form_gd = (float(w @ gd) + 0.0) / (w.sum() + FORM_PRIOR_WEIGHT)
        gw = np.array([FORM_DECAY**k for k in range(len(recent))])
        gf = np.array([r[2] for r in recent], dtype=float)
        ga = np.array([r[3] for r in recent], dtype=float)
        att = (float(gw @ gf) + GOALS_PRIOR_WEIGHT * PRIOR_GOALS) / (gw.sum() + GOALS_PRIOR_WEIGHT)
        dfn = (float(gw @ ga) + GOALS_PRIOR_WEIGHT * PRIOR_GOALS) / (gw.sum() + GOALS_PRIOR_WEIGHT)
        return {
            "elo": float(elo),
            "form_pts": form_pts,
            "form_gd": form_gd,
            "att": att,
            "dfn": dfn,
            "log_n": float(np.log1p(n)),
        }

    def pair_features(
        self,
        home: str,
        away: str,
        home_country: str,
        away_country: str,
        *,
        neutral: bool = False,
        is_knockout: bool = False,
        is_second_leg: bool = False,
        is_final: bool = False,
    ) -> dict[str, float]:
        """Feature row for a (possibly hypothetical) match, from the current state."""
        feats: dict[str, float] = {}
        for prefix, (team, country) in (("h", (home, home_country)), ("a", (away, away_country))):
            for key, value in self.team_features(team, country).items():
                feats[f"{key}_{prefix}"] = value
        feats["elo_diff"] = feats["elo_h"] - feats["elo_a"]
        feats["neutral"] = float(neutral)
        feats["is_knockout"] = float(is_knockout)
        feats["is_second_leg"] = float(is_second_leg)
        feats["is_final"] = float(is_final)
        return feats

    # ------------------------------------------------------------------ #
    def start_season(self, season: int) -> None:
        """Season boundary: regress every rating a little towards the mean."""
        if self.season is not None and season != self.season:
            r = self.params.season_regression
            for rec in self.teams.values():
                rec.elo += r * (BASE_ELO - rec.elo)
        self.season = season

    def update(self, row) -> None:
        """Feed one *finished* match into the state."""
        p = self.params
        h = self._get(row.home, row.home_country)
        a = self._get(row.away, row.away_country)
        hg, ag = int(row.home_goals), int(row.away_goals)
        bonus = 0.0 if row.neutral else p.home_advantage
        e_home = expected_score(h.elo + bonus - a.elo)
        s_home = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        delta = p.k * goal_multiplier(hg - ag) * (s_home - e_home)
        h.elo += delta
        a.elo -= delta
        h.recent.appendleft((3.0 if hg > ag else 1.0 if hg == ag else 0.0, hg - ag, hg, ag))
        a.recent.appendleft((3.0 if ag > hg else 1.0 if hg == ag else 0.0, ag - hg, ag, hg))
        h.n += 1
        a.n += 1

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return {
            "params": asdict(self.params),
            "season": self.season,
            "teams": {
                name: {
                    "country": t.country,
                    "elo": t.elo,
                    "n": t.n,
                    "recent": [list(r) for r in t.recent],
                }
                for name, t in self.teams.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> RatingState:
        st = cls(EloParams(**d["params"]))
        st.season = d["season"]
        for name, t in d["teams"].items():
            st.teams[name] = TeamRecord(
                country=t["country"],
                elo=t["elo"],
                n=t["n"],
                recent=deque((tuple(r) for r in t["recent"]), maxlen=GOALS_WINDOW),
            )
        return st


# --------------------------------------------------------------------------- #
def add_context(matches: pd.DataFrame) -> pd.DataFrame:
    """Add venue/stage flags that are known before kick-off (no results used)."""
    df = matches.copy()
    df["is_final"] = (df["stage"] == "final").astype(float)
    df["is_knockout"] = df["stage"].isin(["knockout", "playoff", "final"]).astype(float)
    df["is_second_leg"] = (df["leg"] == 2).astype(float)
    # Neutral venues: finals, plus the single-leg "Lisbon bubble" of 2019/20
    # (quarter-finals onward, played in August 2020).
    bubble = (df["season"] == 2019) & (df["date"] >= "2020-08-12")
    df["neutral"] = ((df["stage"] == "final") | (bubble & (df["stage"] != "group"))).astype(float)
    return df


def result_label(home_goals: pd.Series, away_goals: pd.Series) -> np.ndarray:
    """0 = home win, 1 = draw, 2 = away win (90-minute result)."""
    return np.where(home_goals > away_goals, 0, np.where(home_goals == away_goals, 1, 2))


def build_features(
    matches: pd.DataFrame, params: EloParams | None = None
) -> tuple[pd.DataFrame, RatingState]:
    """Walk through ``matches`` chronologically and return (features, final state).

    ``matches`` must contain the columns of :func:`ucl.data.build_matches`.  The
    returned frame has one row per match with only pre-match information plus
    the 90-minute ``y`` label (never used as a feature).
    """
    df = add_context(matches).sort_values(["date", "match_id"], kind="stable").reset_index(drop=True)
    state = RatingState(params)
    rows: list[dict[str, float]] = []
    for _, day in df.groupby("date", sort=True):
        state.start_season(int(day["season"].iloc[0]))
        # 1) read features of every match of the day ...
        for r in day.itertuples(index=False):
            rows.append(
                state.pair_features(
                    r.home,
                    r.away,
                    r.home_country,
                    r.away_country,
                    neutral=bool(r.neutral),
                    is_knockout=bool(r.is_knockout),
                    is_second_leg=bool(r.is_second_leg),
                    is_final=bool(r.is_final),
                )
            )
        # 2) ... and only then learn from the results.
        for r in day.itertuples(index=False):
            state.update(r)
    feats = pd.DataFrame(rows)
    meta = df[
        ["match_id", "season", "date", "stage", "round", "home", "away", "home_country", "away_country"]
    ]
    out = pd.concat([meta.reset_index(drop=True), feats], axis=1)
    out["y"] = result_label(df["home_goals"], df["away_goals"])
    out["goal_diff"] = (df["home_goals"] - df["away_goals"]).to_numpy()
    return out, state


# --------------------------------------------------------------------------- #
# Feature groups used by the models / ablations
# --------------------------------------------------------------------------- #
def _sides(names: tuple[str, ...]) -> list[str]:
    return [f"{n}_{s}" for n in names for s in ("h", "a")]


ELO_ONLY = ["elo_diff", "neutral"]
CONTEXT = list(CONTEXT_FEATURES)
FORM = _sides(("form_pts", "form_gd", "att", "dfn"))
EXPERIENCE = _sides(("log_n",))
FEATURES_NO_FORM = ["elo_diff", *CONTEXT, *EXPERIENCE]
FEATURES_ALL = ["elo_diff", *_sides(("elo",)), *CONTEXT, *FORM, *EXPERIENCE]


def split_frame(feats: pd.DataFrame, split: str) -> pd.DataFrame:
    """Rows of the requested chronological split (see :mod:`ucl.config`)."""
    from ucl.config import split_of_season

    mask = feats["season"].map(split_of_season) == split
    return feats[mask].reset_index(drop=True)
