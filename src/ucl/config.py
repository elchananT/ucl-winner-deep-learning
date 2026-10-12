"""Project-wide constants: paths, chronological split, seeds.

Everything that defines *what counts as train / validation / test* lives here so
that the notebooks, the tests and the CLI cannot silently disagree.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MATCHES_CSV = PROCESSED_DIR / "matches.csv"

SEED = 2026

# Seasons are labelled by their starting year (2022 == the 2022/23 edition).
FIRST_SEASON = 1992
LAST_SEASON = 2025

# Strict chronological split: every validation match is played after every
# training match, and every test match after every validation match.
TRAIN_SEASONS = (1992, 2016)  # 1992/93 .. 2016/17 (inclusive)
VAL_SEASONS = (2017, 2020)  # 2017/18 .. 2020/21
TEST_SEASONS = (2021, 2025)  # 2021/22 .. 2025/26

# Held-out editions that use the classic 32-team format and are therefore
# simulated end to end in notebook 04 (2024/25 introduced the league phase).
SIMULATION_SEASONS = (2021, 2022, 2023)

RESULT_LABELS = ("home", "draw", "away")


def split_of_season(season: int) -> str:
    """Return ``"train"``, ``"val"`` or ``"test"`` for a season start year."""
    if TRAIN_SEASONS[0] <= season <= TRAIN_SEASONS[1]:
        return "train"
    if VAL_SEASONS[0] <= season <= VAL_SEASONS[1]:
        return "val"
    if TEST_SEASONS[0] <= season <= TEST_SEASONS[1]:
        return "test"
    raise ValueError(f"season {season} is outside the supported range")
