"""Synthetic fixtures: small, fast, deterministic stand-ins for the real match table."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

COUNTRIES = ["ENG", "ESP", "GER", "ITA", "FRA", "POR", "NED", "RUS"]


def make_matches(n_seasons: int = 6, n_teams: int = 16, seed: int = 0, start: int = 2000) -> pd.DataFrame:
    """Round-robin-ish fake seasons: groups then knockout, with latent team strengths."""
    rng = np.random.default_rng(seed)
    teams = [f"team{i:02d}|{COUNTRIES[i % len(COUNTRIES)]}" for i in range(n_teams)]
    strength = rng.normal(0, 0.5, n_teams)
    rows = []
    for s in range(n_seasons):
        season = start + s
        day = pd.Timestamp(f"{season}-09-15")
        for md in range(10):
            order = rng.permutation(n_teams)
            for k in range(0, n_teams, 2):
                i, j = order[k], order[k + 1]
                lam_h = np.exp(0.25 + 0.3 * (strength[i] - strength[j]))
                lam_a = np.exp(0.0 - 0.3 * (strength[i] - strength[j]))
                is_ko = md >= 8
                rows.append(
                    {
                        "season": season,
                        "date": day + pd.Timedelta(days=14 * md),
                        "stage": "knockout" if is_ko else "group",
                        "round": "Round of 16" if is_ko else f"Matchday {md + 1}",
                        "group": None,
                        "leg": float(md - 7) if is_ko else np.nan,
                        "home": teams[i],
                        "away": teams[j],
                        "home_country": teams[i].split("|")[1],
                        "away_country": teams[j].split("|")[1],
                        "home_goals": int(rng.poisson(lam_h)),
                        "away_goals": int(rng.poisson(lam_a)),
                        "et_home_goals": np.nan,
                        "et_away_goals": np.nan,
                        "pen_home": np.nan,
                        "pen_away": np.nan,
                        "source": "synthetic",
                    }
                )
    df = pd.DataFrame(rows).sort_values("date", kind="stable").reset_index(drop=True)
    df.insert(0, "match_id", range(len(df)))
    return df


@pytest.fixture(scope="session")
def synthetic_matches() -> pd.DataFrame:
    return make_matches()


@pytest.fixture(scope="session")
def real_matches() -> pd.DataFrame:
    from ucl.config import MATCHES_CSV
    from ucl.data import load_matches

    if not MATCHES_CSV.exists():
        pytest.skip("dataset not downloaded - run `make data`")
    return load_matches()
