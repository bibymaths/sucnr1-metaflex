"""Steady state solver for RoadRunner models.

This module attempts to compute a steady state of the loaded SBML
model.  If the built–in steadyState() function fails, a long
forward simulation is performed as a fallback.  A dictionary of
steady state values is returned.
"""

from __future__ import annotations

from typing import Dict, Optional

from loguru import logger

try:
    import roadrunner
except ImportError:
    roadrunner = None  # type: ignore


def compute_steady_state(rr: "roadrunner.ExecutableModel", timeout: float = 1000.0) -> Optional[Dict[str, float]]:
    """Attempt to compute a steady state for the given RoadRunner model.

    Args:
        rr: An initialised RoadRunner model.
        timeout: Maximum simulation time used in fallback simulation.

    Returns:
        A mapping from species identifiers to steady state values, or
        ``None`` if no steady state could be found.
    """
    try:
        rr.steadyState()
        ids = rr.getFloatingSpeciesIds()
        values = rr.getFloatingSpeciesConcentrations()
        return dict(zip(ids, values))
    except Exception as exc:
        logger.warning(f"Analytical steady state failed: {exc}, falling back to simulation")
    # fallback: run long simulation
    try:
        rr.selections = ["time"] + list(rr.getFloatingSpeciesIds())
        res = rr.simulate(0, timeout, 100)
        ids = rr.getFloatingSpeciesIds()
        # take final row
        steady = res[-1, 1:]
        return dict(zip(ids, steady))
    except Exception as exc:
        logger.error(f"Fallback steady state simulation failed: {exc}")
    return None
