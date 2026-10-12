from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from ucl.config import MODELS_DIR
from ucl.features import EloParams, build_features
from ucl.models import MLPConfig, MLPModel
from ucl.predict import DEFAULT_ARTIFACT, MatchRequest, Predictor, save_artifact


def test_request_validation():
    assert MatchRequest(home="Real Madrid", away="Barcelona").stage == "group"
    assert MatchRequest(home="a1", away="b2", stage="final").neutral is True
    with pytest.raises(ValidationError):
        MatchRequest(home="Real Madrid", away="real madrid")
    with pytest.raises(ValidationError):
        MatchRequest(home="x1", away="y2", stage="semifinal")
    with pytest.raises(ValidationError):
        MatchRequest(home="x1", away="y2", stage="group", second_leg=True)
    with pytest.raises(ValidationError):
        MatchRequest(home="x1", away="y2", surprise=1)


def test_artifact_roundtrip_on_synthetic_data(synthetic_matches, tmp_path):
    feats, state = build_features(synthetic_matches, EloParams())
    n = len(feats)
    tr, va = feats.iloc[: int(n * 0.7)], feats.iloc[int(n * 0.7) :]
    cfg = MLPConfig(hidden=(8,), max_epochs=3, patience=2, min_team_matches=1)
    members = [MLPModel(cfg, seed=s).fit(tr, va) for s in range(2)]
    save_artifact(members, 1.2, state, tmp_path / "art", {"as_of": "2005-06-01"})
    pred = Predictor.load(tmp_path / "art")
    out = pred.predict(MatchRequest(home="team00", away="team01", stage="knockout", second_leg=True))
    assert np.isclose(out.p_home + out.p_draw + out.p_away, 1.0)
    with pytest.raises(ValueError, match="unknown or ambiguous"):
        pred.predict(MatchRequest(home="Nonexistent FC", away="team01"))


@pytest.mark.skipif(not (DEFAULT_ARTIFACT / "artifact.json").exists(), reason="trained artifact not present")
def test_committed_artifact_predicts_sensible_probabilities():
    pred = Predictor.load(MODELS_DIR / "ucl_mlp")
    a = pred.predict(MatchRequest(home="Real Madrid", away="Skonto"))
    b = pred.predict(MatchRequest(home="Skonto", away="Real Madrid"))
    assert np.isclose(a.p_home + a.p_draw + a.p_away, 1.0)
    assert a.p_home > a.p_away and b.p_away > b.p_home  # strong club favoured home and away
