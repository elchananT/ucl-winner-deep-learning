from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ucl.simulation import (
    STAGE_NAMES,
    MarginSampler,
    _round_of_16_draw,
    forecast_scores,
    season_setup,
    simulate_season,
)

COUNTRIES = ["ENG", "ESP", "GER", "ITA", "FRA", "POR", "NED", "RUS"]


def fake_group_season(season: int = 2030) -> pd.DataFrame:
    rows = []
    for g in range(8):
        teams = [f"g{g}t{k}|{COUNTRIES[(g + k) % 8]}" for k in range(4)]
        for h in teams:
            for a in teams:
                if h != a:
                    rows.append(
                        dict(season=season, stage="group", group=chr(65 + g), home=h, away=a,
                             home_country=h.split("|")[1], away_country=a.split("|")[1])
                    )  # fmt: skip
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def setup():
    return season_setup(fake_group_season(), 2030)


def uniform_probs(n=32):
    return np.full((4, n, n, 3), 1 / 3)


def margins():
    rng = np.random.default_rng(0)
    return MarginSampler(rng.integers(-4, 5, 2000))


def test_season_setup_recovers_groups(setup):
    assert len(setup.teams) == 32 and len(setup.groups) == 8
    assert all(len(v) == 4 for v in setup.groups.values())
    assert len(setup.fixtures) == 8 * 12


def test_each_stage_has_the_right_number_of_survivors(setup):
    sim = simulate_season(setup, uniform_probs(), margins(), n_sims=600, seed=1)
    expected = [32, 16, 8, 4, 2, 1]
    assert np.allclose(sim.sum().to_numpy(), expected)
    assert list(sim.columns) == list(STAGE_NAMES)
    assert ((sim >= 0) & (sim <= 1)).all().all()
    # monotone: reaching a later stage is never more likely than an earlier one
    assert (np.diff(sim.to_numpy(), axis=1) <= 1e-12).all()


def test_winner_probabilities_sum_to_one_and_are_symmetric_under_uniform_model(setup):
    sim = simulate_season(setup, uniform_probs(), margins(), n_sims=8000, seed=2)
    assert np.isclose(sim["Winner"].sum(), 1.0)
    assert np.abs(sim["Winner"] - 1 / 32).max() < 0.015


def test_stronger_team_wins_more_often(setup):
    p = uniform_probs()
    p[:, 0, :, :] = [0.8, 0.12, 0.08]  # team 0 as hosts: very strong
    p[:, :, 0, :] = [0.08, 0.12, 0.8]  # team 0 as visitors: very strong
    sim = simulate_season(setup, p, margins(), n_sims=3000, seed=3)
    assert sim["Winner"].iloc[0] > 3 / 32
    assert sim["Winner"].idxmax() == setup.teams[0]


def test_simulation_is_seeded(setup):
    a = simulate_season(setup, uniform_probs(), margins(), n_sims=200, seed=5)
    b = simulate_season(setup, uniform_probs(), margins(), n_sims=200, seed=5)
    pd.testing.assert_frame_equal(a, b)


def test_round_of_16_draw_respects_group_and_country_constraints():
    rng = np.random.default_rng(0)
    countries = np.array([i % 8 for i in range(16)])
    winners, runners = np.arange(8), np.arange(8, 16)
    for _ in range(200):
        perm = _round_of_16_draw(winners, runners, countries, rng)
        assert sorted(perm) == list(range(8))
        assert all(perm[g] != g for g in range(8))
        assert all(countries[winners[g]] != countries[runners[perm[g]]] for g in range(8))


def test_margin_sampler_respects_the_outcome():
    ms = margins()
    rng = np.random.default_rng(0)
    gd = ms.sample_goal_diff(np.array([0] * 100 + [1] * 100 + [2] * 100), rng)
    assert (gd[:100] >= 1).all() and (gd[100:200] == 0).all() and (gd[200:] <= -1).all()


def test_forecast_scores_reward_the_right_champion(setup):
    n = 1000
    sim = pd.DataFrame(0.0, index=setup.teams, columns=list(STAGE_NAMES))
    sim["Group exit"] = 1.0
    sim["Winner"] = 1 / 32
    actual = pd.Series(0, index=setup.teams)
    actual.iloc[0] = 5
    flat = forecast_scores(sim, actual, n)
    sim.iloc[0, sim.columns.get_loc("Winner")] = 0.5
    sim.iloc[1:, sim.columns.get_loc("Winner")] = 0.5 / 31
    sharp = forecast_scores(sim, actual, n)
    assert sharp["champion_logscore"] < flat["champion_logscore"]
    assert np.isclose(flat["champion_logscore"], np.log(32))
