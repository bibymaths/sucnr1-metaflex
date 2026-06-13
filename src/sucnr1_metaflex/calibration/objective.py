"""Calibration objective for SBML/RoadRunner model fitting."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from loguru import logger

from sucnr1_metaflex.simulation.roadrunner_engine import load_model
from .numba_kernels import NUMBA_AVAILABLE, interp_linear, log10_transform, seahorse_shape, weighted_residuals
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
        model = getattr(rr, "model", rr)
        fn = getattr(model, getter, None)
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
    rr: object | None = None,
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

    rr = load_model(model_path) if rr is None else rr
    reset_all = getattr(rr, "resetAll", None)
    if callable(reset_all):
        reset_all()
    _set_rr_parameters_strict(rr, params, model_path)
    _, global_params = _available_model_symbols(model_path)
    all_symbols, _ = _available_model_symbols(model_path)
    for name, value in (condition_factors or {}).items():
        if name not in global_params:
            # logger.debug(
            #     f"Skipping condition factor {name!r}; not present in model {model_path}"
            # )
            continue

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

        out[col] = interp_linear(unique_times, sim_time, values, use_numba=False)

    return out



@dataclass(frozen=True)
class ResidualGroup:
    assay: str
    cond_col: str | None
    condition: str
    observable: str
    times: np.ndarray
    observed: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class ResidualContext:
    groups: tuple[ResidualGroup, ...]
    n_expected: int
    use_numba: bool
    numba_available: bool

    @property
    def simulations_per_evaluation(self) -> int:
        return len(self.groups)


def prepare_residual_context(
    data: pd.DataFrame,
    assay_weights: Dict[str, float],
    observables: Dict[str, str],
    use_numba: bool | None = None,
) -> ResidualContext:
    """Precompute numeric residual metadata outside optimizer hot loops."""
    groups: list[ResidualGroup] = []
    for assay, cond_col, condition, observable, agg in _aggregated_assay_tables(data, observables):
        times = agg["time"].to_numpy(dtype=float)
        observed = pd.to_numeric(agg["observed_mean"], errors="coerce").to_numpy(dtype=float)
        weight_value = float(assay_weights.get(str(assay), 1.0))
        weights = np.full(observed.shape, weight_value, dtype=float)
        groups.append(
            ResidualGroup(
                assay=str(assay),
                cond_col=cond_col,
                condition=str(condition),
                observable=str(observable),
                times=times,
                observed=observed,
                weights=weights,
            )
        )
    enabled = NUMBA_AVAILABLE if use_numba is None else bool(use_numba and NUMBA_AVAILABLE)
    return ResidualContext(
        groups=tuple(groups),
        n_expected=int(sum(g.observed.size for g in groups)),
        use_numba=enabled,
        numba_available=NUMBA_AVAILABLE,
    )


def _shape_multiplier(protocol: dict, times: np.ndarray, params: Dict[str, float], use_numba: bool) -> np.ndarray:
    name = protocol.get("shape")
    if not name:
        return np.ones_like(times, dtype=float)
    if str(name) == "seahorse_ocr":
        values = np.array([
            params.get("ocr_basal", 1.0), params.get("ocr_peak_amp", 0.0),
            params.get("ocr_peak_time", 1.0), params.get("ocr_peak_width", 0.18),
            params.get("ocr_drop_amp", 0.0), params.get("ocr_drop_time", 1.5),
            params.get("ocr_drop_width", 0.22), params.get("min_shape", 1.0e-6),
        ], dtype=float)
        return seahorse_shape(times, 1, values, use_numba=use_numba)
    if str(name) == "seahorse_ecar":
        values = np.array([
            params.get("ecar_basal", 1.0), params.get("ecar_peak_amp", 0.0),
            params.get("ecar_peak_time", 1.0), params.get("ecar_peak_width", 0.20),
            params.get("ecar_drop_amp", 0.0), params.get("ecar_drop_time", 1.5),
            params.get("ecar_drop_width", 0.25), params.get("min_shape", 1.0e-6),
        ], dtype=float)
        return seahorse_shape(times, 2, values, use_numba=use_numba)
    return evaluate_shape(name, times, params)

def compute_residuals(
    x_log: np.ndarray,
    param_names: list[str],
    data: pd.DataFrame,
    model_path: str,
    assay_weights: Dict[str, float],
    observables: Dict[str, str],
    context: ResidualContext | None = None,
    use_numba: bool | None = None,
) -> np.ndarray:
    """Compute fixed-length residual vector for log10 parameters.

    Parameters are optimized in log10 space and converted to linear space
    before being applied to the SBML model.
    """
    ctx = context or prepare_residual_context(data, assay_weights, observables, use_numba=use_numba)
    n_expected = ctx.n_expected

    if n_expected <= 0:
        return _penalty_vector(1)

    x_log = np.asarray(x_log, dtype=float).ravel()

    if x_log.size != len(param_names):
        return _penalty_vector(n_expected)

    if not np.all(np.isfinite(x_log)):
        return _penalty_vector(n_expected)

    try:
        params = {
            str(name): float(value)
            for name, value in zip(param_names, log10_transform(x_log, ctx.use_numba))
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
        rr = load_model(str(model_path))
        for group in ctx.groups:
            assay = group.assay
            cond_col = group.cond_col
            condition = group.condition
            observable = group.observable
            times = group.times
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
                rr=rr,
            )

            pred = pd.to_numeric(sim[observable], errors="coerce").to_numpy(dtype=float)
            pred = pred * _shape_multiplier(protocol, times, params, ctx.use_numba)
            obs = group.observed

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

            residual = weighted_residuals(pred, obs, group.weights, use_numba=ctx.use_numba)

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
