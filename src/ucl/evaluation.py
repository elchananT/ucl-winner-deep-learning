"""Probabilistic-forecast metrics, calibration tools and paired bootstrap.

Outcome classes are ordered ``home (0) < draw (1) < away (2)``; probabilities
are ``(n, 3)`` arrays whose rows sum to one.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

EPS = 1e-12


def _check(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    if p.ndim != 2 or p.shape[0] != y.shape[0]:
        raise ValueError(f"shape mismatch: y {y.shape} vs p {p.shape}")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("probability rows must sum to 1")
    return y, p


def log_loss(y, p) -> float:
    """Mean negative log-likelihood of the observed class (lower is better)."""
    y, p = _check(y, p)
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], EPS, 1.0))))


def per_match_log_loss(y, p) -> np.ndarray:
    y, p = _check(y, p)
    return -np.log(np.clip(p[np.arange(len(y)), y], EPS, 1.0))


def brier(y, p) -> float:
    """Multi-class Brier score: mean over matches of the summed squared error."""
    y, p = _check(y, p)
    onehot = np.eye(p.shape[1])[y]
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def per_match_brier(y, p) -> np.ndarray:
    y, p = _check(y, p)
    return np.sum((p - np.eye(p.shape[1])[y]) ** 2, axis=1)


def rps(y, p) -> float:
    """Ranked probability score for ordered outcomes (home < draw < away)."""
    return float(np.mean(per_match_rps(y, p)))


def per_match_rps(y, p) -> np.ndarray:
    y, p = _check(y, p)
    k = p.shape[1]
    cum_p = np.cumsum(p, axis=1)[:, :-1]
    cum_y = np.cumsum(np.eye(k)[y], axis=1)[:, :-1]
    return np.sum((cum_p - cum_y) ** 2, axis=1) / (k - 1)


def accuracy(y, p) -> float:
    y, p = _check(y, p)
    return float(np.mean(p.argmax(axis=1) == y))


def reliability_table(y, p, n_bins: int = 10, *, classwise: bool = True) -> pd.DataFrame:
    """Binned predicted probability vs observed frequency.

    With ``classwise=True`` every (match, class) pair is one binary prediction
    ("did class k occur?"), the usual multi-class extension of a reliability
    diagram; otherwise only the top-label confidence is used.
    """
    y, p = _check(y, p)
    if classwise:
        conf = p.reshape(-1)
        hit = np.eye(p.shape[1])[y].reshape(-1)
    else:
        conf = p.max(axis=1)
        hit = (p.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        rows.append(
            {
                "bin": b,
                "lo": edges[b],
                "hi": edges[b + 1],
                "count": int(m.sum()),
                "mean_confidence": float(conf[m].mean()) if m.any() else np.nan,
                "observed_frequency": float(hit[m].mean()) if m.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def ece(y, p, n_bins: int = 10, *, classwise: bool = True) -> float:
    """Expected calibration error: bin-weighted mean |confidence - frequency|."""
    t = reliability_table(y, p, n_bins, classwise=classwise)
    t = t[t["count"] > 0]
    return float(
        np.sum(t["count"] * (t["mean_confidence"] - t["observed_frequency"]).abs()) / t["count"].sum()
    )


def score_all(y, p) -> dict[str, float]:
    return {
        "log_loss": log_loss(y, p),
        "brier": brier(y, p),
        "rps": rps(y, p),
        "accuracy": accuracy(y, p),
        "ece": ece(y, p),
    }


# --------------------------------------------------------------------------- #
# Temperature scaling
# --------------------------------------------------------------------------- #
def apply_temperature(p, temperature: float) -> np.ndarray:
    """Soften (T>1) or sharpen (T<1) probabilities: ``softmax(log p / T)``."""
    logits = np.log(np.clip(np.asarray(p, dtype=float), EPS, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(y, p, bounds: tuple[float, float] = (0.2, 5.0)) -> float:
    """Single scalar T minimising validation log-loss (Guo et al., 2017)."""
    y, p = _check(y, p)
    res = minimize_scalar(lambda t: log_loss(y, apply_temperature(p, t)), bounds=bounds, method="bounded")
    return float(res.x)


# --------------------------------------------------------------------------- #
# Uncertainty
# --------------------------------------------------------------------------- #
def paired_bootstrap(
    y,
    p_a,
    p_b,
    metric: Callable[[np.ndarray, np.ndarray], np.ndarray] = per_match_log_loss,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap CI of ``mean(metric(A)) - mean(metric(B))`` over matches.

    Negative values mean model A has the lower (better) loss.  Resampling
    matches (not seasons) ignores within-season correlation, so the intervals
    are, if anything, slightly optimistic.
    """
    rng = np.random.default_rng(seed)
    d = metric(y, p_a) - metric(y, p_b)
    n = len(d)
    boots = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "diff": float(d.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "p_a_better": float((boots < 0).mean()),
    }


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    boots = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)
