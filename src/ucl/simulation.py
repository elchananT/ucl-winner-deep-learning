"""Monte-Carlo simulation of a classic 32-team Champions League edition.

Pipeline for one held-out season:

1. freeze a :class:`~ucl.features.RatingState` as of the day before the season;
2. precompute the model's (home, draw, away) probabilities for every ordered
   pair of participants in four match contexts (group, first leg, second leg,
   neutral final);
3. simulate the real group draw, the constrained round-of-16 draw, random
   quarter/semi-final draws and the final ``n_sims`` times.

Simplifications (documented in the model card):

* ratings are frozen at the start of the tournament, i.e. they are not updated
  by simulated results;
* the models forecast the 90-minute *outcome* only, so goal margins are drawn
  from the empirical margin distribution of the training period given the
  outcome; group ties are broken on points, simulated goal difference, then at
  random (no head-to-head, no goals scored);
* ties level after two legs, and drawn finals, are decided by a fair coin flip
  (extra time + penalties);
* the round-of-16 draw respects "winner vs runner-up, different group,
  different association"; later rounds are drawn at random.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ucl.features import RatingState, build_features

CONTEXTS = ("group", "leg1", "leg2", "final")
STAGE_NAMES = ("Group exit", "Round of 16", "Quarter-final", "Semi-final", "Final", "Winner")


@dataclass
class SeasonSetup:
    season: int
    teams: list[str]
    countries: dict[str, str]
    groups: dict[str, list[str]]
    fixtures: list[tuple[str, str, str]]  # (home, away, group letter)


def season_setup(matches: pd.DataFrame, season: int) -> SeasonSetup:
    """Participants, groups and the complete fixture list of a classic-format season.

    Groups are recovered as the connected components of the group-stage
    fixture graph, because some source files do not label groups by letter.
    """
    g = matches[(matches["season"] == season) & (matches["stage"] == "group")]
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for h, a in zip(g["home"], g["away"], strict=True):
        parent[find(h)] = find(a)
    components: dict[str, list[str]] = {}
    for t in list(parent):
        components.setdefault(find(t), []).append(t)
    members = sorted((sorted(v) for v in components.values()), key=lambda v: v[0])
    if len(members) != 8 or any(len(v) != 4 for v in members):
        raise ValueError(f"season {season} is not a classic 8-group x 4-team edition")
    groups = {chr(ord("A") + i): v for i, v in enumerate(members)}
    of_team = {t: letter for letter, v in groups.items() for t in v}
    teams = [t for letter in groups for t in groups[letter]]
    countries = dict(zip(g["home"], g["home_country"], strict=True)) | dict(
        zip(g["away"], g["away_country"], strict=True)
    )
    fixtures = [(h, a, of_team[h]) for h, a in zip(g["home"], g["away"], strict=True)]
    return SeasonSetup(season, teams, countries, groups, fixtures)


def state_before_season(matches: pd.DataFrame, season: int, params) -> RatingState:
    """Rating state as it stands at kick-off of ``season`` (uses only earlier seasons)."""
    _, state = build_features(matches[matches["season"] < season], params)
    state.start_season(season)
    return state


def probability_tensor(model, state: RatingState, setup: SeasonSetup) -> np.ndarray:
    """``P[context, home_idx, away_idx, outcome]`` for all ordered pairs of participants."""
    teams = setup.teams
    rows = []
    for ci, ctx in enumerate(CONTEXTS):
        flags = {
            "group": dict(),
            "leg1": dict(is_knockout=True),
            "leg2": dict(is_knockout=True, is_second_leg=True),
            "final": dict(is_knockout=True, is_final=True, neutral=True),
        }[ctx]
        for i, h in enumerate(teams):
            for j, a in enumerate(teams):
                if i == j:
                    continue
                f = state.pair_features(h, a, setup.countries[h], setup.countries[a], **flags)
                rows.append({"ctx": ci, "i": i, "j": j, "home": h, "away": a, **f})
    frame = pd.DataFrame(rows)
    p = model.predict_proba(frame)
    n = len(teams)
    out = np.full((len(CONTEXTS), n, n, 3), 1.0 / 3.0)
    out[frame["ctx"].to_numpy(), frame["i"].to_numpy(), frame["j"].to_numpy()] = p
    return out


class MarginSampler:
    """Goal margin given the 90-minute result, from the empirical training distribution."""

    def __init__(self, goal_diff: np.ndarray):
        gd = np.asarray(goal_diff)
        self.home = self._dist(gd[gd > 0])
        self.away = self._dist(-gd[gd < 0])

    @staticmethod
    def _dist(margins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values, counts = np.unique(margins, return_counts=True)
        return values, np.cumsum(counts) / counts.sum()

    def sample_goal_diff(self, outcome: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Signed home goal difference for outcomes 0 (home) / 1 (draw) / 2 (away)."""
        u = rng.random(outcome.shape)
        gd = np.zeros(outcome.shape, dtype=int)
        for code, (values, cdf), sign in ((0, self.home, 1), (2, self.away, -1)):
            mask = outcome == code
            gd[mask] = sign * values[np.searchsorted(cdf, u[mask])]
        return gd


def _sample_outcome(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Vectorised categorical draw from rows of ``p`` (shape ``(..., 3)``)."""
    u = rng.random(p.shape[:-1])
    cdf = np.cumsum(p, axis=-1)
    return np.minimum((u[..., None] > cdf).sum(axis=-1), 2)


def _round_of_16_draw(
    winners: np.ndarray, runners: np.ndarray, countries: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Pair each group winner with a runner-up from another group and association.

    ``winners``/``runners`` are team indices aligned by group (position g = group g).
    Rejection sampling over random permutations is uniform over valid pairings.
    """
    n = len(winners)
    for attempt in range(500):
        perm = rng.permutation(n)
        ok = all(perm[g] != g for g in range(n))
        if ok and attempt < 400:
            ok = all(countries[winners[g]] != countries[runners[perm[g]]] for g in range(n))
        if ok:
            return perm
    return rng.permutation(n)  # pragma: no cover - practically unreachable


def _play_two_legged(
    a: np.ndarray, b: np.ndarray, probs: np.ndarray, margins: MarginSampler, rng: np.random.Generator
) -> np.ndarray:
    """Winner (team index) of ties between ``a`` (hosts leg 1) and ``b`` (hosts leg 2)."""
    o1 = _sample_outcome(probs[1][a, b], rng)
    o2 = _sample_outcome(probs[2][b, a], rng)
    # aggregate from a's perspective
    agg = margins.sample_goal_diff(o1, rng) - margins.sample_goal_diff(o2, rng)
    coin = rng.random(a.shape) < 0.5
    a_wins = np.where(agg == 0, coin, agg > 0)
    return np.where(a_wins, a, b)


def _play_final(
    a: np.ndarray, b: np.ndarray, probs: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    swap = rng.random(a.shape) < 0.5  # arbitrary "home" slot at a neutral venue
    h, w = np.where(swap, b, a), np.where(swap, a, b)
    o = _sample_outcome(probs[3][h, w], rng)
    coin = rng.random(a.shape) < 0.5
    home_wins = np.where(o == 1, coin, o == 0)
    winner = np.where(home_wins, h, w)
    loser = np.where(home_wins, w, h)
    return winner, loser


def simulate_season(
    setup: SeasonSetup,
    probs: np.ndarray,
    margins: MarginSampler,
    n_sims: int = 20000,
    seed: int = 0,
) -> pd.DataFrame:
    """Simulate the edition ``n_sims`` times.

    Returns a frame indexed by team with, for every stage, the probability of
    *reaching* it (Round of 16, Quarter-final, Semi-final, Final) and of winning.
    Each column of ``reach`` sums to the number of teams that can reach it
    (16, 8, 4, 2, 1).
    """
    rng = np.random.default_rng(seed)
    idx = {t: i for i, t in enumerate(setup.teams)}
    n_teams = len(setup.teams)
    country_codes = {c: k for k, c in enumerate(sorted(set(setup.countries.values())))}
    countries = np.array([country_codes[setup.countries[t]] for t in setup.teams])

    # --- group stage ------------------------------------------------------ #
    pts = np.zeros((n_sims, n_teams))
    gd = np.zeros((n_sims, n_teams))
    for home, away, _ in setup.fixtures:
        i, j = idx[home], idx[away]
        o = _sample_outcome(np.broadcast_to(probs[0][i, j], (n_sims, 3)), rng)
        d = margins.sample_goal_diff(o, rng)
        pts[:, i] += np.where(o == 0, 3, np.where(o == 1, 1, 0))
        pts[:, j] += np.where(o == 2, 3, np.where(o == 1, 1, 0))
        gd[:, i] += d
        gd[:, j] -= d
    letters = sorted(setup.groups)
    winners = np.zeros((n_sims, 8), dtype=int)
    runners = np.zeros((n_sims, 8), dtype=int)
    for g, letter in enumerate(letters):
        members = np.array([idx[t] for t in setup.groups[letter]])
        key = pts[:, members] * 1e4 + gd[:, members] * 1e2 + rng.random((n_sims, 4))
        order = np.argsort(-key, axis=1)
        winners[:, g] = members[order[:, 0]]
        runners[:, g] = members[order[:, 1]]

    reached = np.zeros((n_teams, 6))  # columns: STAGE_NAMES[:]  reached-at-least counts
    # reached[:, 0] = everyone (group stage); filled below
    reached[:, 0] = n_sims
    for g in range(8):
        np.add.at(reached[:, 1], winners[:, g], 1)
        np.add.at(reached[:, 1], runners[:, g], 1)

    # --- round of 16 (constrained draw) ----------------------------------- #
    r16_a = np.zeros((n_sims, 8), dtype=int)  # runner-up hosts leg 1
    r16_b = np.zeros((n_sims, 8), dtype=int)  # group winner hosts leg 2
    for s in range(n_sims):
        perm = _round_of_16_draw(winners[s], runners[s], countries, rng)
        r16_a[s] = runners[s][perm]
        r16_b[s] = winners[s]
    qf = _play_two_legged(r16_a, r16_b, probs, margins, rng)  # (n_sims, 8)
    for g in range(8):
        np.add.at(reached[:, 2], qf[:, g], 1)

    # --- quarter-finals ---------------------------------------------------- #
    def draw_round(teams_: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        shuffled = np.take_along_axis(teams_, np.argsort(rng.random(teams_.shape), axis=1), axis=1)
        a, b = shuffled[:, 0::2], shuffled[:, 1::2]
        flip = rng.random(a.shape) < 0.5
        return np.where(flip, b, a), np.where(flip, a, b)

    a, b = draw_round(qf)
    sf = _play_two_legged(a, b, probs, margins, rng)  # (n_sims, 4)
    for g in range(4):
        np.add.at(reached[:, 3], sf[:, g], 1)
    a, b = draw_round(sf)
    fin = _play_two_legged(a, b, probs, margins, rng)  # (n_sims, 2)
    for g in range(2):
        np.add.at(reached[:, 4], fin[:, g], 1)
    champ, _ = _play_final(fin[:, 0], fin[:, 1], probs, rng)
    np.add.at(reached[:, 5], champ, 1)

    out = pd.DataFrame(reached / n_sims, index=setup.teams, columns=list(STAGE_NAMES))
    out.attrs["n_sims"] = n_sims
    return out


def actual_stage_reached(matches: pd.DataFrame, season: int) -> pd.Series:
    """Furthest stage each participant actually reached, as an index into ``STAGE_NAMES``."""
    s = matches[matches["season"] == season]
    level = {"Round of 16": 1, "Quarterfinals": 2, "Semifinals": 3, "Final": 4}
    reached: dict[str, int] = {}
    for r in s.itertuples(index=False):
        lv = 0 if r.stage == "group" else level[r.round]
        for t in (r.home, r.away):
            reached[t] = max(reached.get(t, 0), lv)
    final = s[s["stage"] == "final"].iloc[0]
    if pd.notna(final["pen_home"]):
        home_won = final["pen_home"] > final["pen_away"]
    elif pd.notna(final["et_home_goals"]):
        home_won = final["et_home_goals"] > final["et_away_goals"]
    else:
        home_won = final["home_goals"] > final["away_goals"]
    reached[final["home"] if home_won else final["away"]] = 5
    return pd.Series(reached)


def forecast_scores(sim: pd.DataFrame, actual: pd.Series, n_sims: int) -> dict[str, float]:
    """Score a simulated forecast against what happened.

    * ``champion_logscore``: -log P(actual champion) (probability floored at 1/n_sims)
    * ``champion_brier``: Brier score of the 32-way winner distribution
    * ``stage_brier``: mean Brier over the five "reached stage k" binary events
    """
    champ = actual[actual == 5].index[0]
    p_champ = max(float(sim.loc[champ, "Winner"]), 1.0 / n_sims)
    y_champ = (actual == 5).astype(float).reindex(sim.index).to_numpy()
    champion_brier = float(np.sum((sim["Winner"].to_numpy() - y_champ) ** 2))
    briers = []
    for k in range(1, 6):
        y = (actual.reindex(sim.index) >= k).astype(float).to_numpy()
        briers.append(float(np.mean((sim[STAGE_NAMES[k]].to_numpy() - y) ** 2)))
    return {
        "champion_prob": p_champ,
        "champion_logscore": -float(np.log(p_champ)),
        "champion_brier": champion_brier,
        "stage_brier": float(np.mean(briers)),
    }


__all__ = [
    "CONTEXTS",
    "STAGE_NAMES",
    "MarginSampler",
    "SeasonSetup",
    "actual_stage_reached",
    "forecast_scores",
    "probability_tensor",
    "season_setup",
    "simulate_season",
    "state_before_season",
]
