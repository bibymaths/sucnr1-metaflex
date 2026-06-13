"""Objective function for parameter estimation."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
from loguru import logger

from ..simulation.roadrunner_engine import load_model, simulate_to_times


PENALTY_VALUE = 1.0e12


def _safe_numeric_frame(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df["assay"] = df["assay"].astype(str)
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

    if not np.isfinite(scale) or scale <= 1.0e-12:
        scale = float(np.nanmax(np.abs(arr)))

    if not np.isfinite(scale) or scale <= 1.0e-12:
        scale = 1.0

    return scale


def _available_model_symbols(rr: object) -> set[str]:
    symbols: set[str] = {"time"}

    objects = [rr, getattr(rr, "model", None)]

    method_names = [
        "getFloatingSpeciesIds",
        "getBoundarySpeciesIds",
        "getGlobalParameterIds",
        "getCompartmentIds",
        "getReactionIds",
    ]

    for obj in objects:
        if obj is None:
            continue

        for method_name in method_names:
            method = getattr(obj, method_name, None)

            if not callable(method):
                continue

            try:
                values = method()
            except Exception:
                continue

            try:
                symbols.update(str(x) for x in values)
            except Exception:
                continue

    return symbols


def _aggregated_assay_tables(data: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    tables: list[tuple[str, pd.DataFrame]] = []

    for assay, df_assay in data.groupby("assay", dropna=False):
        assay = str(assay)

        agg = (
            df_assay.groupby("time", as_index=False)["value"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values("time")
        )

        if not agg.empty:
            tables.append((assay, agg))

    return tables


def _expected_residual_length(tables: list[tuple[str, pd.DataFrame]]) -> int:
    return max(1, sum(len(table) for _, table in tables))


def _penalty(length: int) -> np.ndarray:
    return np.full(max(1, int(length)), PENALTY_VALUE, dtype=float)


def _set_parameters(rr: object, param_values: Dict[str, float]) -> None:
    for pid, value in param_values.items():
        try:
            rr[pid] = float(value)  # type: ignore[index]
        except Exception:
            continue


def compute_residuals(
    log_params: Sequence[float],
    param_names: Sequence[str],
    data: pd.DataFrame,
    model_path: str,
    assay_weights: Dict[str, float],
    observables: Dict[str, str] | None = None,
) -> np.ndarray:
    """Compute normalized residuals for one parameter vector.

    The returned residual vector has constant length for all parameter
    values. This is required by scipy.optimize.least_squares.
    """
    observables = observables or {}

    df = _safe_numeric_frame(data)

    if df.empty:
        logger.error("No finite data rows available for objective.")
        return _penalty(1)

    assay_tables = _aggregated_assay_tables(df)
    expected_len = _expected_residual_length(assay_tables)

    param_values = {
        str(name): float(10.0 ** value)
        for name, value in zip(param_names, log_params)
    }

    try:
        rr0 = load_model(model_path)
        _set_parameters(rr0, param_values)
    except Exception as exc:
        logger.error(f"Could not load model {model_path}: {exc}")
        return _penalty(expected_len)

    available_symbols = _available_model_symbols(rr0)

    residuals: list[float] = []

    for assay, agg in assay_tables:
        species_id = observables.get(assay)

        if species_id is None:
            logger.error(
                f"No observable mapping for assay '{assay}'. "
                "Add it under the 'observables:' section of the fit YAML."
            )
            return _penalty(expected_len)

        species_id = str(species_id)

        if available_symbols and species_id not in available_symbols:
            logger.error(
                f"Observable '{species_id}' requested for assay '{assay}', "
                f"but it is not present in model '{model_path}'. "
                f"Available symbols include: {sorted(available_symbols)[:40]}"
            )
            return _penalty(expected_len)

        times = agg["time"].to_numpy(dtype=float)
        obs = agg["mean"].to_numpy(dtype=float)
        scale = _assay_scale(agg["mean"])
        weight = float(assay_weights.get(assay, 1.0))

        try:
            rr = load_model(model_path)
            _set_parameters(rr, param_values)
            sim = simulate_to_times(rr, times, selections=[species_id])
        except Exception as exc:
            logger.error(f"Simulation failed for assay={assay}, species={species_id}: {exc}")
            return _penalty(expected_len)

        if species_id not in sim.columns:
            logger.error(f"Simulation output does not contain species '{species_id}'.")
            return _penalty(expected_len)

        pred = pd.to_numeric(sim[species_id], errors="coerce").to_numpy(dtype=float)

        if pred.shape[0] != obs.shape[0]:
            logger.error(
                f"Prediction/data length mismatch for assay={assay}: "
                f"pred={pred.shape[0]}, obs={obs.shape[0]}"
            )
            return _penalty(expected_len)

        valid = np.isfinite(pred) & np.isfinite(obs)

        if not np.any(valid):
            return _penalty(expected_len)

        assay_residuals = weight * (pred[valid] - obs[valid]) / scale
        residuals.extend(assay_residuals.tolist())

    if len(residuals) != expected_len:
        logger.error(
            f"Objective residual length changed: got {len(residuals)}, expected {expected_len}."
        )
        return _penalty(expected_len)

    residual_array = np.asarray(residuals, dtype=float)

    if not np.all(np.isfinite(residual_array)):
        return _penalty(expected_len)

    return residual_array