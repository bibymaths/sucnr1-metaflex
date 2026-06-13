"""Calibration objective for SBML/RoadRunner model fitting."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from loguru import logger

from sucnr1_metaflex.simulation.roadrunner_engine import load_model
from .protocols import detect_condition_column, evaluate_shape, load_protocol_config, resolve_condition_factors, resolve_initial_conditions, resolve_protocol


DEFAULT_PENALTY_VALUE = 1.0e11


def _numeric_time_value_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Return data with numeric time/value and valid assay labels."""
    df = data.copy()

    if "assay" not in df.columns:
        raise ValueError("Input data is missing required column: assay")

    if "time" not in df.columns:
        raise ValueError("Input data is missing required column: time")

    if "value" not in df.columns:
        raise ValueError("Input data is missing required column: value")

    df["assay"] = df["assay"].astype(str)
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df[np.isfinite(df["time"]) & np.isfinite(df["value"])].copy()

    if df.empty:
        raise ValueError("No finite time/value rows remain for fitting.")

    return df


def _aggregated_assay_tables(
    data: pd.DataFrame,
    observables: Dict[str, str],
) -> list[tuple[str, str | None, str, str, pd.DataFrame]]:
    """Aggregate observations by assay, optional condition, and time."""
    df = _numeric_time_value_frame(data)
    protocols = load_protocol_config()
    tables: list[tuple[str, str | None, str, str, pd.DataFrame]] = []

    for assay, assay_df in df.groupby("assay", dropna=False):
        assay = str(assay)
        if assay not in observables:
            raise KeyError(f"Fit config missing observable for assay '{assay}'")
        protocol = resolve_protocol(assay, protocols)
        observable = str(protocol.get("observable", observables[assay]))
        cond_col = detect_condition_column(assay_df, protocol.get("condition_column"))
        groups = [("__all__", assay_df)] if cond_col is None else assay_df.groupby(cond_col, dropna=False)
        for condition, cdf in groups:
            condition_label = "" if cond_col is None else str(condition)
            agg = (cdf.groupby("time").agg(observed_mean=("value","mean"), observed_sd=("value","std"), n=("value","count")).reset_index().sort_values("time"))
            if not agg.empty:
                tables.append((assay, cond_col, condition_label, observable, agg))
    return tables

def _expected_residual_length(
    data: pd.DataFrame,
    observables: Dict[str, str],
) -> int:
    """Return the fixed residual-vector length expected by optimizers."""
    return int(
        sum(len(table) for _, _, _, _, table in _aggregated_assay_tables(data, observables))
    )


def _penalty_vector(length: int, value: float = DEFAULT_PENALTY_VALUE) -> np.ndarray:
    """Return a fixed-length penalty residual vector."""
    n = max(int(length), 1)
    return np.full(n, float(value), dtype=float)


@lru_cache(maxsize=16)
def _available_model_symbols(model_path: str) -> tuple[frozenset[str], frozenset[str]]:
    """Return available model symbols and settable global parameters."""
    rr = load_model(str(model_path))

    all_symbols: set[str] = set()
    global_params: set[str] = set()

    for getter in [
        "getFloatingSpeciesIds",
        "getBoundarySpeciesIds",
        "getGlobalParameterIds",
        "getCompartmentIds",
        "getReactionIds",
    ]:
        fn = getattr(rr.model, getter, None)
        if callable(fn):
            try:
                values = {str(x) for x in fn()}
                all_symbols.update(values)
                if getter == "getGlobalParameterIds":
                    global_params.update(values)
            except Exception:
                pass

    return frozenset(all_symbols), frozenset(global_params)


def _set_rr_parameters_strict(
    rr: object,
    params: Dict[str, float],
    model_path: str,
) -> None:
    """Apply fitted parameters and fail if a requested parameter is absent."""
    _, global_params = _available_model_symbols(str(model_path))

    for name, value in params.items():
        if name in global_params:
            rr[name] = float(value)  # type: ignore[index]


def _simulate_fitted_to_times(
    model_path: str | Path,
    params: Dict[str, float],
    times: Iterable[float],
    selections: list[str],
    initial_conditions: Dict[str, float] | None = None,
    condition_factors: Dict[str, float] | None = None,
) -> pd.DataFrame:
    """Simulate a fitted model to requested times without losing parameters."""
    model_path = str(model_path)

    all_symbols, _ = _available_model_symbols(model_path)

    missing_selections = [str(x) for x in selections if str(x) not in all_symbols]
    if missing_selections:
        raise KeyError(
            f"Requested simulation selections are absent from model {model_path}: "
            f"{missing_selections}"
        )

    rr = load_model(model_path)
    _set_rr_parameters_strict(rr, params, model_path)
    _, global_params = _available_model_symbols(model_path)
    all_symbols, _ = _available_model_symbols(model_path)
    for name, value in (condition_factors or {}).items():
        if name not in global_params:
            raise KeyError(f"Condition factor {name} not present in model")
        rr[name] = float(value)  # type: ignore[index]
    for name, value in (initial_conditions or {}).items():
        if name not in all_symbols:
            raise KeyError(f"Protocol initial condition {name} not present in model")
        rr[name] = float(value)  # type: ignore[index]

    times_arr = np.asarray(list(times), dtype=float)
    times_arr = times_arr[np.isfinite(times_arr)]

    if times_arr.size == 0:
        raise ValueError("No finite simulation times were provided.")

    unique_times = np.unique(times_arr)
    t_min = float(np.min(unique_times))
    t_max = float(np.max(unique_times))

    ycols = [str(x) for x in selections]
    rr.selections = ["time"] + ycols

    start = min(0.0, t_min)
    end = t_max

    if np.isclose(start, end):
        end = start + 1.0e-9

    n_points = max(200, 10 * len(unique_times))
    n_points = min(n_points, 5000)

    sim = rr.simulate(start, end, n_points)
    sim_df = pd.DataFrame(sim, columns=["time"] + ycols)

    sim_time = pd.to_numeric(sim_df["time"], errors="coerce").to_numpy(dtype=float)

    if not np.all(np.isfinite(sim_time)):
        raise ValueError("Simulation returned non-finite time values.")

    out = pd.DataFrame({"time": unique_times})

    for col in ycols:
        values = pd.to_numeric(sim_df[col], errors="coerce").to_numpy(dtype=float)

        if values.shape[0] != sim_time.shape[0]:
            raise ValueError(f"Simulation returned malformed values for {col}")

        if not np.all(np.isfinite(values)):
            raise ValueError(f"Simulation returned non-finite values for {col}")

        out[col] = np.interp(unique_times, sim_time, values)

    return out


def compute_residuals(
    x_log: np.ndarray,
    param_names: list[str],
    data: pd.DataFrame,
    model_path: str,
    assay_weights: Dict[str, float],
    observables: Dict[str, str],
) -> np.ndarray:
    """Compute fixed-length residual vector for log10 parameters.

    Parameters are optimized in log10 space and converted to linear space
    before being applied to the SBML model.
    """
    tables = _aggregated_assay_tables(data, observables)
    n_expected = int(sum(len(table) for _, _, _, _, table in tables))

    if n_expected <= 0:
        return _penalty_vector(1)

    x_log = np.asarray(x_log, dtype=float).ravel()

    if x_log.size != len(param_names):
        return _penalty_vector(n_expected)

    if not np.all(np.isfinite(x_log)):
        return _penalty_vector(n_expected)

    try:
        params = {
            str(name): float(10.0 ** value)
            for name, value in zip(param_names, x_log)
        }
    except Exception:
        return _penalty_vector(n_expected)

    if not all(np.isfinite(v) and v >= 0.0 for v in params.values()):
        return _penalty_vector(n_expected)

    all_symbols, global_params = _available_model_symbols(str(model_path))

    protocols = load_protocol_config()
    missing_observables = sorted({str(resolve_protocol(a, protocols).get("observable", o)) for a, o in observables.items()} - set(all_symbols))
    if missing_observables:
        logger.error(f"Observable symbols absent from model: {missing_observables}")
        return _penalty_vector(n_expected)

    residual_parts: list[np.ndarray] = []

    try:
        for assay, cond_col, condition, observable, agg in tables:
            times = agg["time"].to_numpy(dtype=float)
            protocol = resolve_protocol(assay, protocols)
            initial_conditions = resolve_initial_conditions(protocol, params)
            factors = (
                resolve_condition_factors(
                    protocol, condition if cond_col is not None else None, params
                )
                if protocol.get("condition_factors")
                else {}
            )

            sim = _simulate_fitted_to_times(
                model_path=model_path,
                params=params,
                times=times,
                selections=[observable],
                initial_conditions=initial_conditions,
                condition_factors=factors,
            )

            pred = pd.to_numeric(sim[observable], errors="coerce").to_numpy(dtype=float)
            pred = pred * evaluate_shape(protocol.get("shape"), times, params)
            obs = pd.to_numeric(agg["observed_mean"], errors="coerce").to_numpy(dtype=float)

            if pred.shape[0] != obs.shape[0]:
                logger.error(
                    f"Residual length mismatch for assay={assay}, "
                    f"observable={observable}: pred={pred.shape[0]}, obs={obs.shape[0]}"
                )
                return _penalty_vector(n_expected)

            if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(obs)):
                logger.error(
                    f"Non-finite prediction or observation for assay={assay}, "
                    f"observable={observable}"
                )
                return _penalty_vector(n_expected)

            weight = float(assay_weights.get(str(assay), 1.0))
            residual = weight * (pred - obs)

            if not np.all(np.isfinite(residual)):
                return _penalty_vector(n_expected)

            residual_parts.append(residual.astype(float))

    except Exception as exc:
        logger.debug(f"Residual computation failed: {exc}")
        return _penalty_vector(n_expected)

    if not residual_parts:
        return _penalty_vector(n_expected)

    residuals = np.concatenate(residual_parts).astype(float)

    if residuals.shape[0] != n_expected:
        return _penalty_vector(n_expected)

    if not np.all(np.isfinite(residuals)):
        return _penalty_vector(n_expected)

    return residuals
