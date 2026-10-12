"""The most important tests: features must be built from the strict past only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ucl.features import FEATURES_ALL, EloParams, build_features

LABEL_LIKE = {"y", "goal_diff", "home_goals", "away_goals"}


def _feature_columns(feats: pd.DataFrame) -> list[str]:
    return [
        c
        for c in feats.columns
        if c
        not in {
            "match_id",
            "season",
            "date",
            "stage",
            "round",
            "home",
            "away",
            "home_country",
            "away_country",
            "y",
            "goal_diff",
        }
    ]


def _scramble_after(matches: pd.DataFrame, cutoff: pd.Timestamp, seed: int) -> pd.DataFrame:
    """Replace the results of every match played on/after ``cutoff`` with random scores."""
    rng = np.random.default_rng(seed)
    out = matches.copy()
    mask = out["date"] >= cutoff
    out.loc[mask, "home_goals"] = rng.integers(0, 6, mask.sum())
    out.loc[mask, "away_goals"] = rng.integers(0, 6, mask.sum())
    return out


def test_labels_are_never_features():
    assert not LABEL_LIKE & set(FEATURES_ALL)


@pytest.mark.parametrize("cut_fraction", [0.3, 0.6, 0.9])
def test_future_results_do_not_change_past_features(synthetic_matches, cut_fraction):
    dates = np.sort(synthetic_matches["date"].unique())
    cutoff = pd.Timestamp(dates[int(len(dates) * cut_fraction)])
    base, _ = build_features(synthetic_matches)
    corrupted, _ = build_features(_scramble_after(synthetic_matches, cutoff, seed=1))
    cols = _feature_columns(base)
    # Features of every match played on or before the cutoff day depend only on
    # results from *before* that day, so they must be identical.
    past = base["date"] <= cutoff
    pd.testing.assert_frame_equal(base.loc[past, cols], corrupted.loc[past, cols], check_exact=True)


def test_scrambling_is_detectable_afterwards(synthetic_matches):
    """Power check: the test above would fail if features did look at results."""
    dates = np.sort(synthetic_matches["date"].unique())
    cutoff = pd.Timestamp(dates[len(dates) // 2])
    base, _ = build_features(synthetic_matches)
    corrupted, _ = build_features(_scramble_after(synthetic_matches, cutoff, seed=1))
    later = base["date"] > cutoff
    cols = _feature_columns(base)
    assert not np.allclose(base.loc[later, cols].to_numpy(), corrupted.loc[later, cols].to_numpy())


def test_truncation_equivalence(synthetic_matches):
    """Building on a prefix of history gives exactly the same rows as the full build."""
    dates = np.sort(synthetic_matches["date"].unique())
    cutoff = pd.Timestamp(dates[len(dates) // 2])
    full, _ = build_features(synthetic_matches)
    prefix, _ = build_features(synthetic_matches[synthetic_matches["date"] <= cutoff])
    cols = _feature_columns(full)
    n = len(prefix)
    pd.testing.assert_frame_equal(full.loc[: n - 1, cols], prefix[cols], check_exact=True)


def test_same_day_results_do_not_leak_sideways(synthetic_matches):
    """Changing one result must not alter other matches played the same day."""
    day = synthetic_matches["date"].iloc[len(synthetic_matches) // 2]
    idx = synthetic_matches.index[synthetic_matches["date"] == day]
    assert len(idx) > 1
    changed = synthetic_matches.copy()
    changed.loc[idx[0], ["home_goals", "away_goals"]] = [7, 0]
    base, _ = build_features(synthetic_matches)
    other, _ = build_features(changed)
    cols = _feature_columns(base)
    same_day = base["date"] == day
    pd.testing.assert_frame_equal(base.loc[same_day, cols], other.loc[same_day, cols], check_exact=True)


def test_leakage_on_real_data(real_matches):
    """Same property on the real 4k-match table, cut at the start of the test period."""
    cutoff = pd.Timestamp("2021-09-01")
    base, _ = build_features(real_matches, EloParams())
    corrupted, _ = build_features(_scramble_after(real_matches, cutoff, seed=3), EloParams())
    cols = _feature_columns(base)
    past = base["date"] <= cutoff
    assert past.sum() > 3000
    pd.testing.assert_frame_equal(base.loc[past, cols], corrupted.loc[past, cols], check_exact=True)
