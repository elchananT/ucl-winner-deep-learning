"""Inference API and command line: ``ucl-predict "Real Madrid" "Manchester City"``.

The saved artifact bundles (i) the neural-network ensemble, (ii) the fitted
temperature, and (iii) a snapshot of the :class:`~ucl.features.RatingState`
(Elo, experience, recent form of every club) as of the last match in the data,
so predictions use exactly the same feature code as training.
"""

from __future__ import annotations

import argparse
import difflib
import json
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ucl.config import MODELS_DIR
from ucl.evaluation import apply_temperature
from ucl.features import RatingState
from ucl.models import Ensemble, MLPModel
from ucl.names import normalise

DEFAULT_ARTIFACT = MODELS_DIR / "ucl_mlp"

# Convenience spellings accepted by the CLI (normalised form -> normalised data name).
SEARCH_ALIASES = {
    "bayern munich": "bayern munchen",
    "psg": "paris saint germain",
    "paris sg": "paris saint germain",
    "man city": "manchester city",
    "man united": "manchester united",
    "man utd": "manchester united",
    "inter milan": "inter",
    "juve": "juventus",
    "spurs": "tottenham hotspur",
    "atletico": "atletico madrid",
    "dortmund": "borussia dortmund",
    "barca": "barcelona",
}


class MatchRequest(BaseModel):
    """Validated description of the match to forecast."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    home: str = Field(min_length=2, description="Club listed first (hosts the match unless neutral)")
    away: str = Field(min_length=2, description="Club listed second")
    stage: Literal["group", "knockout", "final"] = "group"
    second_leg: bool = Field(False, description="Second leg of a two-legged knockout tie")
    neutral: bool = Field(False, description="Neutral venue (always true for a final)")

    @model_validator(mode="after")
    def _consistent(self) -> MatchRequest:
        if normalise(self.home) == normalise(self.away):
            raise ValueError("home and away must be different clubs")
        if self.second_leg and self.stage != "knockout":
            raise ValueError("second_leg only applies to stage='knockout'")
        if self.stage == "final":
            self.neutral = True
        return self


class MatchPrediction(BaseModel):
    home: str
    away: str
    p_home: float = Field(ge=0, le=1)
    p_draw: float = Field(ge=0, le=1)
    p_away: float = Field(ge=0, le=1)
    ratings_as_of: date
    elo_home: float
    elo_away: float


def save_artifact(
    members: list[MLPModel],
    temperature: float,
    state: RatingState,
    directory: Path,
    extra: dict | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(members):
        m.save(directory / f"member_{i}")
    payload = {
        "temperature": float(temperature),
        "n_members": len(members),
        "state": state.to_dict(),
        **(extra or {}),
    }
    (directory / "artifact.json").write_text(json.dumps(payload, indent=1, default=float))
    return directory


class Predictor:
    """Load once, predict many."""

    def __init__(self, ensemble: Ensemble, temperature: float, state: RatingState, as_of: date):
        self.ensemble, self.temperature, self.state, self.as_of = ensemble, temperature, state, as_of
        self._by_name: dict[str, list[str]] = {}
        for key in state.teams:
            self._by_name.setdefault(key.split("|")[0], []).append(key)

    @classmethod
    def load(cls, directory: Path = DEFAULT_ARTIFACT) -> Predictor:
        meta = json.loads((directory / "artifact.json").read_text())
        members = [MLPModel.load(directory / f"member_{i}") for i in range(meta["n_members"])]
        return cls(
            Ensemble(members),
            meta["temperature"],
            RatingState.from_dict(meta["state"]),
            date.fromisoformat(meta["as_of"]),
        )

    def resolve(self, name: str) -> str:
        """Map a free-text club name to the canonical team key, or raise with suggestions."""
        norm = normalise(name)
        norm = SEARCH_ALIASES.get(norm, norm)
        if norm in self._by_name and len(self._by_name[norm]) == 1:
            return self._by_name[norm][0]
        # allow partial matches such as "madrid" -> "real madrid"
        hits = [k for n, keys in self._by_name.items() for k in keys if norm and norm in n.split()]
        if len(hits) == 1:
            return hits[0]
        pool = sorted(self._by_name)
        close = difflib.get_close_matches(norm, pool, n=4, cutoff=0.5)
        raise ValueError(f"unknown or ambiguous club {name!r}; did you mean one of {close or pool[:5]}?")

    def predict(self, request: MatchRequest) -> MatchPrediction:
        h, a = self.resolve(request.home), self.resolve(request.away)
        if h == a:
            raise ValueError("home and away resolve to the same club")
        feats = self.state.pair_features(
            h,
            a,
            self.state.teams[h].country,
            self.state.teams[a].country,
            neutral=request.neutral,
            is_knockout=request.stage != "group",
            is_second_leg=request.second_leg,
            is_final=request.stage == "final",
        )
        frame = pd.DataFrame([{"home": h, "away": a, **feats}])
        p = apply_temperature(self.ensemble.predict_proba(frame), self.temperature)[0]
        return MatchPrediction(
            home=h.split("|")[0].title(),
            away=a.split("|")[0].title(),
            p_home=float(p[0]),
            p_draw=float(p[1]),
            p_away=float(p[2]),
            ratings_as_of=self.as_of,
            elo_home=feats["elo_h"],
            elo_away=feats["elo_a"],
        )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Forecast the 90-minute result of a Champions League match.")
    ap.add_argument("home")
    ap.add_argument("away")
    ap.add_argument("--stage", choices=["group", "knockout", "final"], default="group")
    ap.add_argument("--second-leg", action="store_true")
    ap.add_argument("--neutral", action="store_true")
    ap.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    ap.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = ap.parse_args(argv)
    try:
        req = MatchRequest(
            home=args.home, away=args.away, stage=args.stage, second_leg=args.second_leg, neutral=args.neutral
        )
        pred = Predictor.load(args.artifact).predict(req)
    except ValueError as err:  # pydantic.ValidationError is a ValueError
        raise SystemExit(f"error: {err}") from None
    if args.json:
        print(pred.model_dump_json(indent=2))
        return
    print(f"{pred.home} vs {pred.away}  ({args.stage}{', neutral' if req.neutral else ''})")
    print(
        f"  {pred.home} win  {pred.p_home:6.1%}\n  draw          {pred.p_draw:6.1%}\n  {pred.away} win  {pred.p_away:6.1%}"
    )
    print(f"  (Elo {pred.elo_home:.0f} vs {pred.elo_away:.0f}, ratings as of {pred.ratings_as_of})")


if __name__ == "__main__":
    main()
