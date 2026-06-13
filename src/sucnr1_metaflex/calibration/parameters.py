"""Parameter configuration utilities."""

from __future__ import annotations

import pathlib
from typing import Dict, Tuple

import yaml
from pydantic import BaseModel, Field


class ParameterDef(BaseModel):
    guess: float = Field(..., description="Log10 initial guess")
    bounds: Tuple[float, float] = Field(..., description="Log10 lower and upper bounds")


class FitConfig(BaseModel):
    parameters: Dict[str, ParameterDef]
    weights: Dict[str, float] = Field(default_factory=dict)
    observables: Dict[str, str] = Field(default_factory=dict)
    optimiser: Dict[str, float | int | str] = Field(default_factory=dict)


def load_fit_config(config_path: str) -> FitConfig:
    path = pathlib.Path(config_path)

    if not path.exists():
        repo_root = pathlib.Path(__file__).resolve().parents[3]
        alt = repo_root / "configs" / path.name
        if alt.exists():
            path = alt

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return FitConfig(**data)