from __future__ import annotations

import numpy as np
import pytest

from ucl.evaluation import log_loss
from ucl.features import EloParams, build_features, split_frame
from ucl.models import (
    CalibratedModel,
    Ensemble,
    MLPConfig,
    MLPModel,
    PriorModel,
    UniformModel,
    elo_logreg,
    full_logreg,
    gradient_boosting,
)


@pytest.fixture(scope="module")
def splits(synthetic_matches):
    feats, _ = build_features(synthetic_matches, EloParams())
    n = len(feats)
    return feats.iloc[: int(n * 0.6)], feats.iloc[int(n * 0.6) : int(n * 0.8)], feats.iloc[int(n * 0.8) :]


@pytest.mark.parametrize(
    "factory",
    [UniformModel, PriorModel, elo_logreg, full_logreg, lambda: gradient_boosting(max_iter=20)],
)
def test_baselines_fit_and_output_valid_probabilities(factory, splits):
    tr, va, te = splits
    model = factory().fit(tr, va)
    p = model.predict_proba(te)
    assert p.shape == (len(te), 3)
    assert np.allclose(p.sum(axis=1), 1.0) and (p >= 0).all()


def test_logistic_regression_beats_uniform_on_learnable_signal(splits):
    tr, _va, te = splits
    p = elo_logreg().fit(tr).predict_proba(te)
    assert log_loss(te["y"], p) < log_loss(te["y"], UniformModel().predict_proba(te))


def test_mlp_fits_on_a_tiny_sample_and_is_reproducible(splits):
    tr, va, te = splits
    cfg = MLPConfig(hidden=(8,), embedding_dim=4, max_epochs=6, patience=3, min_team_matches=1)
    a = MLPModel(cfg, seed=7).fit(tr, va)
    b = MLPModel(cfg, seed=7).fit(tr, va)
    pa = a.predict_proba(te)
    assert pa.shape == (len(te), 3) and np.allclose(pa.sum(axis=1), 1.0)
    assert np.allclose(pa, b.predict_proba(te), atol=1e-5)  # same seed -> same network
    assert len(a.history["val_loss"]) >= 1 and a.best_epoch_ >= 1


def test_mlp_handles_unseen_teams_and_no_embedding_variant(splits):
    tr, va, te = splits
    unseen = te.copy()
    unseen["home"] = "brand_new_team|ENG"
    cfg = MLPConfig(hidden=(8,), max_epochs=3, patience=2, min_team_matches=1)
    m = MLPModel(cfg, seed=0).fit(tr, va)
    assert np.isfinite(m.predict_proba(unseen)).all()
    plain = MLPModel(MLPConfig(hidden=(8,), use_embeddings=False, max_epochs=3, patience=2), seed=0)
    assert plain.fit(tr, va).predict_proba(te).shape == (len(te), 3)


def test_mlp_save_load_roundtrip(splits, tmp_path):
    tr, va, te = splits
    m = MLPModel(MLPConfig(hidden=(8,), max_epochs=3, patience=2, min_team_matches=1), seed=1).fit(tr, va)
    m.save(tmp_path / "m")
    loaded = MLPModel.load(tmp_path / "m")
    assert np.allclose(m.predict_proba(te), loaded.predict_proba(te), atol=1e-6)


def test_ensemble_and_calibrated_wrappers_keep_probabilities_valid(splits):
    tr, _va, te = splits
    ens = Ensemble([elo_logreg().fit(tr), full_logreg().fit(tr)], name="e")
    p = ens.predict_proba(te)
    assert np.allclose(p.sum(axis=1), 1.0)
    cal = CalibratedModel(ens, 1.5)
    q = cal.predict_proba(te)
    assert np.allclose(q.sum(axis=1), 1.0)
    assert q.max() <= p.max() + 1e-9  # T > 1 softens


def test_split_frame_uses_season_ranges(real_matches):
    feats, _ = build_features(real_matches, EloParams())
    assert set(split_frame(feats, "test")["season"]) == {2021, 2022, 2023, 2024, 2025}
