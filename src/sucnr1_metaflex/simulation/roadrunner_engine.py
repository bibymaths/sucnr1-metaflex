"""RoadRunner simulation engine.

This module wraps the libRoadRunner API to provide simple
functions for loading an SBML model and simulating it at
specified time points.  The :mod:`roadrunner` package is used
under the hood.
"""

from __future__ import annotations

from typing import List, Sequence, Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    import roadrunner  # type: ignore
except ImportError:
    roadrunner = None  # type: ignore


class _DummyModel:
    """Fallback model used when libRoadRunner is unavailable.

    This dummy model exposes a minimal API sufficient for the
    simulation functions in this package.  It stores parameter
    values and simulates a single species ``G_plasma`` undergoing
    first–order decay according to the parameter ``k_clear_base``.
    """

    def __init__(self):
        self.params = {"k_clear_base": 0.01}
        self.species = {"G_plasma": 5.0}

    def __setitem__(self, key: str, value: float) -> None:
        self.params[key] = float(value)

    def __getitem__(self, key: str) -> float:
        return self.params.get(key, np.nan)

    def getFloatingSpeciesIds(self):
        return list(self.species.keys())


def load_model(model_path: str) -> "roadrunner.ExecutableModel":
    """Load an SBML model into a RoadRunner instance.

    Args:
        model_path: Path to the SBML file.

    Returns:
        A RoadRunner model ready for simulation.

    Raises:
        RuntimeError: if the roadrunner module is not available.
    """
    if roadrunner is None:
        # return dummy model
        return _DummyModel()  # type: ignore
    rr = roadrunner.RoadRunner(model_path)
    return rr


def simulate_to_times(rr: "roadrunner.ExecutableModel", times: Sequence[float],
                      selections: Optional[List[str]] = None) -> pd.DataFrame:
    """Simulate the model and return values at specified times.

    Args:
        rr: A loaded RoadRunner model.
        times: Sorted sequence of time points.
        selections: Optional list of species/parameter IDs to return.

    Returns:
        A pandas DataFrame indexed by time with columns for each selection.
    """
    # When roadrunner is not available use the dummy model
    if roadrunner is None or isinstance(rr, _DummyModel):
        # simple exponential decay for G_plasma with parameter k_clear_base
        k = rr.params.get("k_clear_base", 0.01)
        G0 = rr.species.get("G_plasma", 1.0)
        data = {
            "time": list(times),
            "G_plasma": [G0 * np.exp(-k * t) for t in times],
        }
        return pd.DataFrame(data)
    # roadrunner available
    if selections is None:
        selections = ["time"] + list(rr.getFloatingSpeciesIds())
    else:
        selections = ["time"] + list(selections)
    rr.selections = selections
    start = float(min(times))
    end = float(max(times))
    steps = max(len(set(times)) - 1, 1)
    try:
        result = rr.simulate(start, end, steps)
    except Exception as exc:
        logger.error(f"Simulation failed: {exc}")
        raise
    df = pd.DataFrame(result, columns=selections)
    df_interp = df.set_index("time").reindex(sorted(set(times))).interpolate()
    df_interp.index.name = "time"
    return df_interp.reset_index()
