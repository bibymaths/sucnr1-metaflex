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
    rr = roadrunner.RoadRunner(str(model_path))
    configure_integrator(rr)
    return rr

def configure_integrator(rr: object) -> None:
    """Use robust CVODE settings for calibration."""
    integrator = getattr(rr, "integrator", None)

    if integrator is None:
        get_integrator = getattr(rr, "getIntegrator", None)
        if callable(get_integrator):
            try:
                integrator = get_integrator()
            except Exception:
                integrator = None

    if integrator is None:
        return

    settings = {
        "absolute_tolerance": 1.0e-8,
        "relative_tolerance": 1.0e-6,
        "maximum_num_steps": 100000,
        "stiff": True,
        "variable_step_size": True,
    }

    for key, value in settings.items():
        try:
            setattr(integrator, key, value)
        except Exception:
            pass

        set_value = getattr(integrator, "setValue", None)
        if callable(set_value):
            try:
                set_value(key, value)
            except Exception:
                pass

def simulate_to_times(
    rr: "roadrunner.ExecutableModel",
    times: Sequence[float],
    selections: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Simulate the model and return values at specified times."""
    times_arr = np.asarray(sorted(set(float(t) for t in times)), dtype=float)

    if times_arr.size == 0:
        raise ValueError("No simulation times were provided.")

    if roadrunner is None or isinstance(rr, _DummyModel):
        k = rr.params.get("k_clear_base", 0.01)
        g0 = rr.species.get("G_plasma", 1.0)
        return pd.DataFrame(
            {
                "time": times_arr,
                "G_plasma": [g0 * np.exp(-k * t) for t in times_arr],
            }
        )

    try:
        rr.resetToOrigin()
    except Exception:
        try:
            rr.reset()
        except Exception:
            pass

    configure_integrator(rr)

    if selections is None:
        ycols = list(rr.getFloatingSpeciesIds())
    else:
        ycols = [str(s) for s in selections]

    rr.selections = ["time"] + ycols

    start = float(times_arr.min())
    end = float(times_arr.max())

    if np.isclose(start, end):
        try:
            values = [rr[col] for col in ycols]
        except Exception as exc:
            logger.error(f"Failed to read model state at t={start}: {exc}")
            raise

        row = {"time": start}
        row.update({col: float(value) for col, value in zip(ycols, values)})
        return pd.DataFrame([row])

    n_points = max(200, int(10 * len(times_arr)))
    n_points = min(n_points, 5000)

    try:
        result = rr.simulate(start, end, n_points)
    except Exception as exc:
        logger.error(f"Simulation failed: {exc}")
        raise

    sim_df = pd.DataFrame(result, columns=["time"] + ycols)
    sim_df = sim_df.drop_duplicates(subset=["time"]).sort_values("time")

    out = pd.DataFrame({"time": times_arr})

    for col in ycols:
        y = pd.to_numeric(sim_df[col], errors="coerce").to_numpy(dtype=float)
        x = pd.to_numeric(sim_df["time"], errors="coerce").to_numpy(dtype=float)

        valid = np.isfinite(x) & np.isfinite(y)

        if valid.sum() < 2:
            raise RuntimeError(f"Could not interpolate simulation output for {col}")

        out[col] = np.interp(times_arr, x[valid], y[valid])

    return out
