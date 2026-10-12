from __future__ import annotations

import numpy as np
import pandas as pd

from ucl.features import (
    FEATURES_ALL,
    EloParams,
    RatingState,
    add_context,
    build_features,
    expected_score,
    goal_multiplier,
    split_frame,
)


def test_shapes_and_no_missing_values(synthetic_matches):
    feats, state = build_features(synthetic_matches)
    assert len(feats) == len(synthetic_matches)
    assert feats[FEATURES_ALL].notna().all().all()
    assert set(feats["y"].unique()) <= {0, 1, 2}
    assert len(state.teams) == synthetic_matches["home"].nunique()


def test_debutants_start_from_the_prior(synthetic_matches):
    feats, _ = build_features(synthetic_matches)
    first = feats.iloc[0]
    assert first["elo_h"] == 1500.0 and first["elo_a"] == 1500.0
    assert first["log_n_h"] == 0.0 and first["log_n_a"] == 0.0


def test_elo_update_is_zero_sum_and_moves_the_right_way():
    st = RatingState(EloParams(k=20, home_advantage=0))
    row = type(
        "R",
        (),
        dict(
            home="a|X",
            away="b|X",
            home_country="X",
            away_country="X",
            home_goals=2,
            away_goals=0,
            neutral=0.0,
        ),
    )()
    st.update(row)
    assert np.isclose(st.teams["a|X"].elo + st.teams["b|X"].elo, 3000.0)
    assert st.teams["a|X"].elo > 1500.0 > st.teams["b|X"].elo


def test_expected_score_and_goal_multiplier():
    assert expected_score(0) == 0.5
    assert np.isclose(expected_score(100) + expected_score(-100), 1.0)
    assert goal_multiplier(1) == 1.0 < goal_multiplier(2) < goal_multiplier(5)


def test_pair_features_match_the_training_rows(synthetic_matches):
    """Train/serve parity: the state-based path used for inference reproduces training rows."""
    df = add_context(synthetic_matches).sort_values(["date", "match_id"]).reset_index(drop=True)
    feats, _ = build_features(synthetic_matches)
    k = len(df) // 2
    prefix_feats, state = build_features(synthetic_matches.iloc[:k])
    row = df.iloc[k]
    # advance the season boundary exactly like build_features does
    if int(row["season"]) != state.season:
        state.start_season(int(row["season"]))
    got = state.pair_features(
        row["home"], row["away"], row["home_country"], row["away_country"],
        neutral=bool(row["neutral"]), is_knockout=bool(row["is_knockout"]),
        is_second_leg=bool(row["is_second_leg"]), is_final=bool(row["is_final"]),
    )  # fmt: skip
    # only valid when no other match on the same day sits before it in the prefix
    if df.iloc[k - 1]["date"] != row["date"]:
        for key, value in got.items():
            assert np.isclose(value, feats.loc[k, key]), key
    assert len(prefix_feats) == k


def test_state_roundtrip(synthetic_matches):
    _, state = build_features(synthetic_matches)
    clone = RatingState.from_dict(state.to_dict())
    a = state.pair_features("team00|ENG", "team01|ESP", "ENG", "ESP")
    b = clone.pair_features("team00|ENG", "team01|ESP", "ENG", "ESP")
    assert a == b


def test_chronological_split_has_no_overlap(real_matches):
    feats, _ = build_features(real_matches, EloParams())
    tr, va, te = (split_frame(feats, s) for s in ("train", "val", "test"))
    assert len(tr) + len(va) + len(te) == len(feats)
    assert tr["date"].max() < va["date"].min() <= va["date"].max() < te["date"].min()


def test_context_flags(real_matches):
    ctx = add_context(real_matches)
    finals = ctx[ctx["stage"] == "final"]
    assert (finals["neutral"] == 1).all() and (finals["is_final"] == 1).all()
    group = ctx[ctx["stage"] == "group"]
    assert (group["is_knockout"] == 0).all() and (group["neutral"] == 0).all()
    assert pd.api.types.is_float_dtype(ctx["is_second_leg"])
