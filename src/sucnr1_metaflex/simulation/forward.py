"""Forward simulation routines.

This module provides a convenience function for running a model
over a uniform time grid.  It leverages the RoadRunner engine and
handles setting parameters before the simulation.  The function
returns a DataFrame of model states across the requested time
range.
"""

from __future__ import annotations

from typing import Dict, Sequence, Optional

import numpy as np
import pandas as pd
from loguru import logger

from .roadrunner_engine import load_model, simulate_to_times


def run_forward(model_path: str, params: Dict[str, float], start: float, end: float, num_points: int = 50,
                selections: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Run a forward simulation on a uniform grid.

    Args:
        model_path: Path to the SBML model file.
        params: Mapping of parameter IDs to values to set before simulation.
        start: Start time for the simulation.
        end: End time for the simulation.
        num_points: Number of points in the grid (including start and end).
        selections: Optional list of species IDs to return.  If not
            provided, all floating species are returned.

    Returns:
        A pandas DataFrame of simulation results.
    """
    rr = load_model(model_path)
    # Override parameter values
    for pid, value in params.items():
        try:
            rr[pid] = float(value)
        except Exception as exc:
            logger.warning(f"Could not set parameter {pid} to {value}: {exc}")
    times = np.linspace(start, end, num_points)
    df = simulate_to_times(rr, times, selections=selections)
    return df
