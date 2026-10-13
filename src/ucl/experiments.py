"""Shared experiment plumbing used by the notebooks (keeps them short and honest).

All selection of hyper-parameters happens on the *validation* split; the test
split is only ever used to report final numbers.
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ucl.config import REPORTS_DIR, SEED
from ucl.data import load_matches
from ucl.evaluation import fit_temperature, log_loss, score_all
from ucl.features import ELO_ONLY, FEATURES_ALL, EloParams, RatingState, build_features, split_frame
from ucl.models import (
    CalibratedModel,
    MLPConfig,
    MLPModel,
    PriorModel,
    UniformModel,
    elo_logreg,
    full_logreg,
    gradient_boosting,
)

RESULTS_DIR = REPORTS_DIR / "results"


def save_json(obj, name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=float) + "\n")
    return path


def load_json(name: str):
    path = RESULTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run the earlier notebooks first (see Makefile)")
    return json.loads(path.read_text())


def record_runtime(key: str, seconds: float) -> None:
    """Merge one wall-clock measurement into ``reports/results/runtimes.json``."""
    path = RESULTS_DIR / "runtimes.json"
    current = json.loads(path.read_text()) if path.exists() else {}
    current[key] = round(float(seconds), 1)
    save_json(current, "runtimes.json")


@dataclass
class Dataset:
    matches: pd.DataFrame
    feats: pd.DataFrame
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    state: RatingState
    elo_params: EloParams


def load_dataset(elo_params: EloParams | None = None) -> Dataset:
    """Matches -> walk-forward features -> chronological splits.

    Without arguments the Elo parameters selected in notebook 02 are used.
    """
    if elo_params is None:
        elo_params = EloParams(**load_json("elo_params.json")["best"])
    matches = load_matches()
    feats, state = build_features(matches, elo_params)
    return Dataset(
        matches,
        feats,
        split_frame(feats, "train"),
        split_frame(feats, "val"),
        split_frame(feats, "test"),
        state,
        elo_params,
    )


# --------------------------------------------------------------------------- #
# Rating hyper-parameters
# --------------------------------------------------------------------------- #
def select_elo_params(matches: pd.DataFrame) -> tuple[EloParams, pd.DataFrame]:
    """Grid-search the Elo system on validation log-loss of the Elo-only classifier."""
    rows = []
    grid = itertools.product([5, 10, 15, 25, 40], [60, 100], [0.0, 0.1, 0.25], [0.0, 0.5])
    for k, ha, reg, cpw in grid:
        p = EloParams(k=k, home_advantage=ha, season_regression=reg, country_prior_weight=cpw)
        feats, _ = build_features(matches, p)
        tr, va = split_frame(feats, "train"), split_frame(feats, "val")
        model = elo_logreg().fit(tr)
        rows.append({**asdict(p), "val_log_loss": log_loss(va["y"], model.predict_proba(va))})
    table = pd.DataFrame(rows).sort_values("val_log_loss").reset_index(drop=True)
    best = table.iloc[0]
    return (
        EloParams(
            k=float(best["k"]),
            home_advantage=float(best["home_advantage"]),
            season_regression=float(best["season_regression"]),
            country_prior_weight=float(best["country_prior_weight"]),
        ),
        table,
    )


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def tune_logreg(tr: pd.DataFrame, va: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    rows = []
    for c in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
        m = full_logreg(c).fit(tr)
        rows.append({"C": c, "val_log_loss": log_loss(va["y"], m.predict_proba(va))})
    table = pd.DataFrame(rows).sort_values("val_log_loss").reset_index(drop=True)
    return float(table.iloc[0]["C"]), table


def tune_gbm(tr: pd.DataFrame, va: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    rows = []
    for depth, n_iter, leaf, l2 in itertools.product([2, 3], [25, 50, 100, 200], [20, 60], [1.0, 10.0]):
        kw = dict(max_depth=depth, max_iter=n_iter, min_samples_leaf=leaf, l2=l2, learning_rate=0.05)
        m = gradient_boosting(**kw, seed=SEED).fit(tr)
        rows.append({**kw, "val_log_loss": log_loss(va["y"], m.predict_proba(va))})
    table = pd.DataFrame(rows).sort_values("val_log_loss").reset_index(drop=True)
    best = table.iloc[0]
    kw = dict(
        max_depth=int(best["max_depth"]),
        max_iter=int(best["max_iter"]),
        min_samples_leaf=int(best["min_samples_leaf"]),
        l2=float(best["l2"]),
        learning_rate=0.05,
    )
    return kw, table


def fit_baselines(tr: pd.DataFrame, va: pd.DataFrame, hparams: dict) -> dict:
    """Fit every non-neural model; ``hparams`` comes from :func:`tune_logreg` / :func:`tune_gbm`."""
    models = {
        "Uniform": UniformModel(),
        "Base rates": PriorModel(),
        "Elo + logistic regression": elo_logreg(),
        "All features + logistic regression": full_logreg(hparams["logreg_C"]),
        "Gradient boosting": gradient_boosting(**hparams["gbm"], seed=SEED),
    }
    for m in models.values():
        m.fit(tr, va)
    return models


def calibrate(model, va: pd.DataFrame, *, name: str | None = None) -> CalibratedModel:
    """Fit one temperature on validation data and wrap the model."""
    t = fit_temperature(va["y"], model.predict_proba(va))
    cal = CalibratedModel(model, t)
    if name:
        cal.name = name
    return cal


def metrics_table(models: dict, df: pd.DataFrame) -> pd.DataFrame:
    rows = {name: score_all(df["y"], m.predict_proba(df)) for name, m in models.items()}
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- #
# Neural network
# --------------------------------------------------------------------------- #
def mlp_search_space(n_trials: int, seed: int = SEED) -> list[MLPConfig]:
    """Random search space (same budget philosophy as the GBM grid: small and cheap)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_trials):
        out.append(
            MLPConfig(
                use_embeddings=True,
                embedding_dim=int(rng.choice([4, 8, 16])),
                hidden=((16,), (32,), (32, 16), (64, 32))[int(rng.integers(0, 4))],
                dropout=float(rng.choice([0.1, 0.3, 0.5])),
                l2=float(rng.choice([1e-4, 1e-3, 1e-2])),
                lr=float(rng.choice([1e-3, 3e-3])),
                batch_size=int(rng.choice([32, 64, 128])),
            )
        )
    return out


def tune_mlp(
    tr: pd.DataFrame, va: pd.DataFrame, n_trials: int = 12, n_seeds: int = 2
) -> tuple[MLPConfig, pd.DataFrame]:
    """Random search; each config is scored by the validation log-loss of a small seed ensemble."""
    configs = mlp_search_space(n_trials)
    rows = []
    for i, cfg in enumerate(configs):
        t0 = time.time()
        members = [MLPModel(cfg, seed=SEED + s).fit(tr, va) for s in range(n_seeds)]
        p = np.mean([m.predict_proba(va) for m in members], axis=0)
        rows.append(
            {
                "trial": i,
                "embedding_dim": cfg.embedding_dim,
                "hidden": str(cfg.hidden),
                "dropout": cfg.dropout,
                "l2": cfg.l2,
                "lr": cfg.lr,
                "batch_size": cfg.batch_size,
                "mean_best_epoch": float(np.mean([m.best_epoch_ for m in members])),
                "val_log_loss": log_loss(va["y"], p),
                "seconds": time.time() - t0,
            }
        )
    table = pd.DataFrame(rows)
    best = int(table["val_log_loss"].idxmin())
    return configs[best], table.sort_values("val_log_loss").reset_index(drop=True)


def ablation_configs(base: MLPConfig) -> dict[str, MLPConfig]:
    """Variants of the tuned network used for the ablation study (everything else fixed)."""
    from dataclasses import replace

    from ucl.features import CONTEXT, EXPERIENCE, FEATURES_NO_FORM

    return {
        "Full (embeddings + form)": base,
        "No team embeddings": replace(base, use_embeddings=False),
        "No form features": replace(base, features=tuple(FEATURES_NO_FORM)),
        "No form, no embeddings": replace(base, use_embeddings=False, features=tuple(FEATURES_NO_FORM)),
        "Elo difference only": replace(base, use_embeddings=False, features=tuple(ELO_ONLY)),
        "Embeddings + context only (no Elo)": replace(
            base, features=tuple([*CONTEXT, *EXPERIENCE]), use_embeddings=True
        ),
    }


__all__ = [
    "FEATURES_ALL",
    "RESULTS_DIR",
    "Dataset",
    "ablation_configs",
    "calibrate",
    "fit_baselines",
    "load_dataset",
    "load_json",
    "metrics_table",
    "record_runtime",
    "save_json",
    "select_elo_params",
    "tune_gbm",
    "tune_logreg",
    "tune_mlp",
]
