"""One consistent, restrained plotting style for every figure in the project."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ucl.config import FIGURES_DIR
from ucl.evaluation import reliability_table

INK = "#1B2A49"
MUTED = "#6B7684"
GRID = "#E3E7EC"

COLORS = {
    "navy": "#1B2A49",
    "blue": "#3B6EA8",
    "sky": "#8FB8DE",
    "teal": "#2A9D8F",
    "gold": "#D9A441",
    "orange": "#E8833A",
    "red": "#C8423B",
    "grey": "#9AA3AF",
}
OUTCOME_COLORS = {"home": COLORS["blue"], "draw": COLORS["grey"], "away": COLORS["orange"]}
MODEL_COLORS = {
    "Uniform": "#C5CBD3",
    "Base rates": COLORS["grey"],
    "Elo + logistic regression": COLORS["blue"],
    "All features + logistic regression": COLORS["sky"],
    "Gradient boosting": COLORS["teal"],
    "MLP + team embeddings": COLORS["orange"],
    "MLP + team embeddings (x5 ensemble)": COLORS["red"],
}


def model_color(name: str) -> str:
    for key, color in MODEL_COLORS.items():
        if name.startswith(key):
            return color
    return COLORS["navy"]


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 140,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.family": "DejaVu Sans",
            "axes.prop_cycle": mpl.cycler(
                color=[COLORS[c] for c in ("blue", "orange", "teal", "red", "gold", "navy")]
            ),
        }
    )


def save(fig: plt.Figure, name: str, directory: Path = FIGURES_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=140)
    return path


def pretty_team(key: str) -> str:
    """``"real madrid|ESP"`` -> ``"Real Madrid"`` (display only)."""
    return key.split("|")[0].title().replace("Psv", "PSV").replace("Cska", "CSKA").replace("Rb ", "RB ")


# --------------------------------------------------------------------------- #
def reliability_plot(
    y: np.ndarray, probs: dict[str, np.ndarray], *, n_bins: int = 8, ax: plt.Axes | None = None
) -> plt.Axes:
    """Class-wise reliability diagram: predicted probability vs observed frequency."""
    ax = ax or plt.gca()
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="perfect calibration")
    for name, p in probs.items():
        t = reliability_table(y, p, n_bins=n_bins)
        t = t[t["count"] >= 10]
        ax.plot(
            t["mean_confidence"],
            t["observed_frequency"],
            marker="o",
            ms=4.5,
            lw=1.6,
            color=model_color(name),
            label=name,
        )
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_xlim(0, 0.85)
    ax.set_ylim(0, 0.85)
    ax.set_aspect("equal")
    return ax


def forest_plot(
    table: pd.DataFrame, value: str, lo: str, hi: str, *, ax: plt.Axes | None = None, ref: float | None = None
) -> plt.Axes:
    """Horizontal point-and-interval plot; rows are models (index)."""
    ax = ax or plt.gca()
    y = np.arange(len(table))[::-1]
    for yi, (name, r) in zip(y, table.iterrows(), strict=True):
        c = model_color(str(name))
        ax.plot([r[lo], r[hi]], [yi, yi], color=c, lw=2.2, solid_capstyle="round")
        ax.plot(r[value], yi, "o", color=c, ms=7, mec="white", mew=1.2)
    ax.set_yticks(y, [str(n) for n in table.index])
    ax.grid(axis="y", visible=False)
    if ref is not None:
        ax.axvline(ref, color=MUTED, lw=1, ls="--")
    return ax


def training_curves(histories: list[dict], *, ax: plt.Axes | None = None) -> plt.Axes:
    """Train / validation loss of several seeds (thin lines) with the mean (thick)."""
    ax = ax or plt.gca()
    for key, color, label in (
        ("loss", COLORS["blue"], "train"),
        ("val_loss", COLORS["orange"], "validation"),
    ):
        for h in histories:
            ax.plot(h[key], color=color, alpha=0.25, lw=1, label=None)
        n = min(len(h[key]) for h in histories)
        mean = np.mean([h[key][:n] for h in histories], axis=0)
        ax.plot(mean, color=color, lw=2.2, label=f"{label} (mean of {len(histories)} seeds)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy (incl. L2 penalty on train)")
    ax.legend()
    return ax
