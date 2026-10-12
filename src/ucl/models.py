"""Baselines and the deep model, behind one tiny interface.

Every model exposes ``fit(train, val)`` and ``predict_proba(frame) -> (n, 3)``
so that the evaluation code treats a coin flip and a neural network
identically.  ``val`` is only used for early stopping (neural net) - never for
fitting parameters of the other models.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ucl.features import ELO_ONLY, FEATURES_ALL

N_CLASSES = 3


# --------------------------------------------------------------------------- #
# Trivial and classical baselines
# --------------------------------------------------------------------------- #
class UniformModel:
    """Predicts 1/3 for each outcome; the 'knows nothing' reference."""

    name = "Uniform"

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> UniformModel:
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return np.full((len(df), N_CLASSES), 1.0 / N_CLASSES)


class PriorModel:
    """Training-set class frequencies (home / draw / away base rates)."""

    name = "Base rates"

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> PriorModel:
        self.p_ = np.bincount(train["y"].to_numpy(), minlength=N_CLASSES) / len(train)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return np.tile(self.p_, (len(df), 1))


class SklearnModel:
    """Wrap a scikit-learn classifier that works on a fixed list of columns."""

    def __init__(self, name: str, columns: Sequence[str], estimator):
        self.name, self.columns, self.estimator = name, list(columns), estimator

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> SklearnModel:
        self.estimator.fit(train[self.columns].to_numpy(), train["y"].to_numpy())
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(df[self.columns].to_numpy())


def elo_logreg(c: float = 1.0) -> SklearnModel:
    """Multinomial logistic regression on Elo difference (+ neutral-venue flag)."""
    est = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))
    return SklearnModel("Elo + logistic regression", ELO_ONLY, est)


def full_logreg(c: float = 0.1, columns: Sequence[str] = FEATURES_ALL) -> SklearnModel:
    est = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000))
    return SklearnModel("All features + logistic regression", columns, est)


def gradient_boosting(
    *,
    max_iter: int = 150,
    max_depth: int = 3,
    learning_rate: float = 0.04,
    min_samples_leaf: int = 40,
    l2: float = 5.0,
    seed: int = 0,
    columns: Sequence[str] = FEATURES_ALL,
) -> SklearnModel:
    est = HistGradientBoostingClassifier(
        max_iter=max_iter,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2,
        early_stopping=False,
        random_state=seed,
    )
    return SklearnModel("Gradient boosting", columns, est)


# --------------------------------------------------------------------------- #
# Deep model: MLP with team embeddings
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MLPConfig:
    features: tuple[str, ...] = tuple(FEATURES_ALL)
    use_embeddings: bool = True
    embedding_dim: int = 8
    hidden: tuple[int, ...] = (32, 16)
    dropout: float = 0.3
    l2: float = 1e-3
    lr: float = 3e-3
    batch_size: int = 64
    max_epochs: int = 200
    patience: int = 15
    min_team_matches: int = 5  # teams seen fewer times in training share an <UNK> embedding

    def to_json(self) -> dict:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}

    @classmethod
    def from_json(cls, d: dict) -> MLPConfig:
        d = {k: tuple(v) if isinstance(v, list) else v for k, v in d.items()}
        return cls(**d)


def _keras():
    """Import Keras lazily and quietly (keeps ``import ucl`` fast, CPU-only for determinism)."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    import keras

    return keras


class MLPModel:
    """Feed-forward net on engineered features plus learned team embeddings.

    The embedding table is shared between the home and the away slot, so a team
    has one learned "style vector" regardless of where it plays.
    """

    def __init__(self, config: MLPConfig | None = None, seed: int = 0):
        self.config, self.seed = config or MLPConfig(), seed
        self.name = "MLP + team embeddings" if self.config.use_embeddings else "MLP"
        self.model = None
        self.history: dict[str, list[float]] = {}

    # -- preprocessing ----------------------------------------------------- #
    def _fit_preprocessing(self, train: pd.DataFrame) -> None:
        x = train[list(self.config.features)].to_numpy(dtype=float)
        self.mean_, self.std_ = x.mean(axis=0), x.std(axis=0) + 1e-9
        counts = pd.concat([train["home"], train["away"]]).value_counts()
        keep = sorted(counts[counts >= self.config.min_team_matches].index)
        self.vocab_ = {t: i + 1 for i, t in enumerate(keep)}  # 0 == <UNK>

    def _inputs(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        x = (df[list(self.config.features)].to_numpy(dtype=float) - self.mean_) / self.std_
        out = {"num": x.astype("float32")}
        if self.config.use_embeddings:
            out["home_id"] = df["home"].map(self.vocab_).fillna(0).to_numpy(dtype="int32")
            out["away_id"] = df["away"].map(self.vocab_).fillna(0).to_numpy(dtype="int32")
        return out

    # -- graph ------------------------------------------------------------- #
    def _build(self):
        keras = _keras()
        cfg = self.config
        layers = keras.layers
        num = layers.Input(shape=(len(cfg.features),), name="num")
        inputs, parts = [num], [num]
        if cfg.use_embeddings:
            hid = layers.Input(shape=(), dtype="int32", name="home_id")
            aid = layers.Input(shape=(), dtype="int32", name="away_id")
            emb = layers.Embedding(
                len(self.vocab_) + 1,
                cfg.embedding_dim,
                embeddings_regularizer=keras.regularizers.l2(cfg.l2),
                name="team_embedding",
            )
            parts += [emb(hid), emb(aid)]
            inputs += [hid, aid]
        x = layers.Concatenate()(parts) if len(parts) > 1 else num
        for width in cfg.hidden:
            x = layers.Dense(width, activation="relu", kernel_regularizer=keras.regularizers.l2(cfg.l2))(x)
            x = layers.Dropout(cfg.dropout)(x)
        out = layers.Dense(N_CLASSES, activation="softmax")(x)
        model = keras.Model(inputs, out)
        model.compile(
            optimizer=keras.optimizers.Adam(cfg.lr),
            loss="sparse_categorical_crossentropy",
        )
        return model

    # -- API --------------------------------------------------------------- #
    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> MLPModel:
        if val is None:
            raise ValueError("MLPModel needs a validation frame for early stopping")
        keras = _keras()
        keras.utils.set_random_seed(self.seed)
        self._fit_preprocessing(train)
        self.model = self._build()
        stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=self.config.patience, restore_best_weights=True
        )
        hist = self.model.fit(
            self._inputs(train),
            train["y"].to_numpy(),
            validation_data=(self._inputs(val), val["y"].to_numpy()),
            epochs=self.config.max_epochs,
            batch_size=self.config.batch_size,
            callbacks=[stop],
            verbose=0,
            shuffle=True,
        )
        self.history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
        self.best_epoch_ = int(np.argmin(self.history["val_loss"])) + 1
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        p = np.asarray(self.model(self._inputs(df), training=False), dtype="float64")
        return p / p.sum(axis=1, keepdims=True)

    # -- persistence ------------------------------------------------------- #
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save(directory / "model.keras")
        meta = {
            "config": self.config.to_json(),
            "seed": self.seed,
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
            "vocab": self.vocab_,
            "best_epoch": self.best_epoch_,
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=1))

    @classmethod
    def load(cls, directory: Path) -> MLPModel:
        keras = _keras()
        meta = json.loads((directory / "meta.json").read_text())
        obj = cls(MLPConfig.from_json(meta["config"]), meta["seed"])
        obj.mean_, obj.std_ = np.array(meta["mean"]), np.array(meta["std"])
        obj.vocab_ = meta["vocab"]
        obj.best_epoch_ = meta["best_epoch"]
        obj.model = keras.models.load_model(directory / "model.keras")
        return obj


class Ensemble:
    """Arithmetic mean of several models' probabilities (a 'deep ensemble')."""

    def __init__(self, members: Sequence, name: str | None = None):
        self.members = list(members)
        self.name = name or f"{self.members[0].name} (x{len(self.members)} ensemble)"

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return np.mean([m.predict_proba(df) for m in self.members], axis=0)


class CalibratedModel:
    """Temperature-scaled wrapper: ``softmax(log p / T)`` with ``T`` fitted on validation data."""

    def __init__(self, base, temperature: float):
        self.base, self.temperature = base, float(temperature)
        self.name = base.name

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        from ucl.evaluation import apply_temperature

        return apply_temperature(self.base.predict_proba(df), self.temperature)
