"""Objective function for parameter estimation."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
from loguru import logger

from ..simulation.roadrunner_engine import load_model, simulate_to_times


def _safe_numeric_frame(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[np.isfinite(df["time"]) & np.isfinite(df["value"])]
    return df


def _assay_scale(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return 1.0

    scale = float(np.nanstd(arr))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.nanmax(np.abs(arr)))

    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0

    return scale


def compute_residuals(
    log_params: Sequence[float],
    param_names: Sequence[str],
    data: pd.DataFrame,
    model_path: str,
    assay_weights: Dict[str, float],
    observables: Dict[str, str] | None = None,
) -> np.ndarray:
    """Compute normalized residuals for one parameter vector."""
    observables = observables or {}

    param_values = {
        name: float(10.0 ** value)
        for name, value in zip(param_names, log_params)
    }

    try:
        rr = load_model(model_path)
    except Exception as exc:
        logger.error(f"Could not load model {model_path}: {exc}")
        return np.full(1, 1.0e12)

    for pid, value in param_values.items():
        try:
            rr[pid] = value
        except Exception:
            continue

    df = _safe_numeric_frame(data)

    if df.empty:
        logger.error("No finite data rows available for objective.")
        return np.full(1, 1.0e12)

    residuals: list[float] = []

    for assay, df_assay in df.groupby("assay"):
        assay = str(assay)
        species_id = observables.get(assay, "G_plasma")
        weight = float(assay_weights.get(assay, 1.0))

        df_assay = (
            df_assay.groupby("time", as_index=False)["value"]
            .mean()
            .sort_values("time")
        )

        times = df_assay["time"].to_numpy(dtype=float)
        obs = df_assay["value"].to_numpy(dtype=float)

        if times.size == 0:
            continue

        scale = _assay_scale(df_assay["value"])

        try:
            sim = simulate_to_times(rr, times, selections=[species_id])
        except Exception as exc:
            logger.error(f"Simulation failed for assay={assay}, species={species_id}: {exc}")
            return np.full(max(1, len(residuals)), 1.0e12)

        if species_id not in sim.columns:
            logger.error(f"Simulation output does not contain species {species_id}")
            return np.full(max(1, len(residuals)), 1.0e12)

        pred = pd.to_numeric(sim[species_id], errors="coerce").to_numpy(dtype=float)

        if pred.shape[0] != obs.shape[0]:
            logger.error(f"Prediction/data length mismatch for assay={assay}")
            return np.full(max(1, len(residuals)), 1.0e12)

        valid = np.isfinite(pred) & np.isfinite(obs)
        if not np.any(valid):
            continue

        assay_residuals = weight * (pred[valid] - obs[valid]) / scale
        residuals.extend(assay_residuals.tolist())

    if not residuals:
        logger.error("Objective produced no residuals.")
        return np.full(1, 1.0e12)

    residual_array = np.asarray(residuals, dtype=float)

    if not np.all(np.isfinite(residual_array)):
        return np.full_like(residual_array, 1.0e12)

    return residual_array