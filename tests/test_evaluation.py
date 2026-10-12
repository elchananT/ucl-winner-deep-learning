from __future__ import annotations

import numpy as np
import pytest

from ucl.evaluation import (
    accuracy,
    apply_temperature,
    brier,
    ece,
    fit_temperature,
    log_loss,
    paired_bootstrap,
    per_match_log_loss,
    reliability_table,
    rps,
)


def test_log_loss_known_values():
    y = np.array([0, 1, 2])
    uniform = np.full((3, 3), 1 / 3)
    assert np.isclose(log_loss(y, uniform), np.log(3))
    perfect = np.eye(3)
    assert log_loss(y, perfect) < 1e-9
    p = np.array([[0.5, 0.3, 0.2]])
    assert np.isclose(log_loss([1], p), -np.log(0.3))


def test_brier_known_values():
    y = np.array([0, 1, 2])
    assert np.isclose(brier(y, np.full((3, 3), 1 / 3)), 2 / 3)
    assert brier(y, np.eye(3)) == 0.0
    # worst case: all mass on a wrong class -> squared error 2
    assert np.isclose(brier([0], np.array([[0.0, 1.0, 0.0]])), 2.0)


def test_rps_known_values_and_ordering():
    # hand computation: predicted (0.5, 0.3, 0.2), truth = draw
    # cum p = (0.5, 0.8), cum y = (0, 1) -> ((0.5)^2 + (0.2)^2)/2 = 0.145
    assert np.isclose(rps([1], np.array([[0.5, 0.3, 0.2]])), 0.145)
    # being wrong by two classes costs more than being wrong by one
    near = rps([0], np.array([[0.0, 1.0, 0.0]]))
    far = rps([0], np.array([[0.0, 0.0, 1.0]]))
    assert far > near
    assert rps([2], np.array([[0.0, 0.0, 1.0]])) == 0.0


def test_accuracy():
    assert accuracy([0, 1], np.array([[0.7, 0.2, 0.1], [0.5, 0.4, 0.1]])) == 0.5


def test_invalid_probabilities_are_rejected():
    with pytest.raises(ValueError):
        log_loss([0], np.array([[0.5, 0.5, 0.5]]))
    with pytest.raises(ValueError):
        log_loss([0, 1], np.array([[1.0, 0.0, 0.0]]))


def test_ece_is_small_when_calibrated_and_large_when_overconfident():
    rng = np.random.default_rng(0)
    n = 40000
    p = rng.dirichlet([2, 1.5, 1.2], size=n)
    y = (rng.random(n)[:, None] > np.cumsum(p, axis=1)).sum(axis=1).clip(0, 2)  # sampled from p
    assert ece(y, p) < 0.01
    sharp = apply_temperature(p, 0.4)  # same ranking, too confident
    assert ece(y, sharp) > 0.05
    assert log_loss(y, sharp) > log_loss(y, p)


def test_reliability_table_counts_everything():
    rng = np.random.default_rng(1)
    p = rng.dirichlet([1, 1, 1], size=500)
    y = rng.integers(0, 3, 500)
    t = reliability_table(y, p, n_bins=10)
    assert t["count"].sum() == 500 * 3
    t_top = reliability_table(y, p, n_bins=10, classwise=False)
    assert t_top["count"].sum() == 500


def test_temperature_scaling_recovers_overconfidence():
    rng = np.random.default_rng(2)
    n = 20000
    p = rng.dirichlet([2, 1.5, 1.2], size=n)
    y = (rng.random(n)[:, None] > np.cumsum(p, axis=1)).sum(axis=1).clip(0, 2)
    sharp = apply_temperature(p, 0.5)
    t = fit_temperature(y, sharp)
    assert 1.8 < t < 2.2  # undoes the 0.5 sharpening
    assert np.isclose(apply_temperature(p, 1.0), p).all()
    assert log_loss(y, apply_temperature(sharp, t)) < log_loss(y, sharp)
    assert np.allclose(apply_temperature(sharp, t).sum(axis=1), 1.0)


def test_paired_bootstrap_detects_a_better_model():
    rng = np.random.default_rng(3)
    n = 3000
    p = rng.dirichlet([2, 1.5, 1.2], size=n)
    y = (rng.random(n)[:, None] > np.cumsum(p, axis=1)).sum(axis=1).clip(0, 2)
    good, bad = p, np.full((n, 3), 1 / 3)
    res = paired_bootstrap(y, good, bad, per_match_log_loss, n_boot=500)
    assert res["diff"] < 0 and res["hi"] < 0 and res["p_a_better"] > 0.99
    same = paired_bootstrap(y, good, good, per_match_log_loss, n_boot=200)
    assert same["diff"] == 0.0 and same["lo"] == 0.0 == same["hi"]
