"""Optional Numba kernels for calibration pure-NumPy hot spots.

This module intentionally contains only numeric kernels. It must not accept
RoadRunner objects, pandas objects, dictionaries, YAML structures, strings, or
other dynamic Python objects in jitted functions.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - exercised when numba is installed
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    njit = None  # type: ignore[assignment]
    NUMBA_AVAILABLE = False

LOSS_LINEAR = 0
LOSS_SOFT_L1 = 1
LOSS_HUBER = 2
LOSS_CAUCHY = 3
DEFAULT_INVALID_LOSS = 1.0e30


def loss_name_to_code(loss_name: str) -> int:
    loss = str(loss_name).strip().lower()
    if loss in {"linear", "l2", "sse"}:
        return LOSS_LINEAR
    if loss in {"soft_l1", "soft-l1", "softl1"}:
        return LOSS_SOFT_L1
    if loss == "huber":
        return LOSS_HUBER
    if loss == "cauchy":
        return LOSS_CAUCHY
    raise ValueError(
        f"Unknown scalar loss '{loss_name}'. Use one of: linear, soft_l1, huber, cauchy."
    )


def log10_transform_numpy(x_log: np.ndarray) -> np.ndarray:
    return 10.0 ** np.asarray(x_log, dtype=float)


def interp_linear_numpy(times: np.ndarray, sim_time: np.ndarray, sim_values: np.ndarray) -> np.ndarray:
    return np.interp(
        np.asarray(times, dtype=float),
        np.asarray(sim_time, dtype=float),
        np.asarray(sim_values, dtype=float),
    )


def weighted_residuals_numpy(pred: np.ndarray, obs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.asarray(weights, dtype=float) * (
        np.asarray(pred, dtype=float) - np.asarray(obs, dtype=float)
    )


def robust_scalar_loss_numpy(residuals: np.ndarray, loss_code: int, f_scale: float) -> float:
    r = np.asarray(residuals, dtype=float).ravel()
    if r.size == 0 or not np.all(np.isfinite(r)):
        return DEFAULT_INVALID_LOSS
    f = max(float(f_scale), 1.0e-12)
    z = (r / f) ** 2
    if loss_code == LOSS_LINEAR:
        value = np.sum(r**2)
    elif loss_code == LOSS_SOFT_L1:
        value = np.sum(2.0 * f**2 * (np.sqrt(1.0 + z) - 1.0))
    elif loss_code == LOSS_HUBER:
        a = np.abs(r)
        value = np.sum(np.where(a <= f, r**2, 2.0 * f * a - f**2))
    elif loss_code == LOSS_CAUCHY:
        value = np.sum(f**2 * np.log1p(z))
    else:
        raise ValueError(f"Unknown loss code: {loss_code}")
    value = float(value)
    return value if np.isfinite(value) else DEFAULT_INVALID_LOSS


def seahorse_shape_numpy(t: np.ndarray, shape_code: int, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(t, dtype=float)
    if shape_code == 0:
        return np.ones_like(arr, dtype=float)
    basal, peak_amp, peak_time, peak_width, drop_amp, drop_time, drop_width, min_shape = np.asarray(values, dtype=float)
    peak_width = max(float(peak_width), 1.0e-9)
    drop_width = max(float(drop_width), 1.0e-9)
    shape = basal + peak_amp * np.exp(-0.5 * ((arr - peak_time) / peak_width) ** 2) - drop_amp * np.exp(-0.5 * ((arr - drop_time) / drop_width) ** 2)
    return np.maximum(shape, float(min_shape))


if NUMBA_AVAILABLE:  # pragma: no cover - compiled functions are tested through wrappers
    log10_transform_numba = njit(cache=True)(log10_transform_numpy)
    interp_linear_numba = njit(cache=True)(interp_linear_numpy)
    weighted_residuals_numba = njit(cache=True)(weighted_residuals_numpy)
    robust_scalar_loss_numba = njit(cache=True)(robust_scalar_loss_numpy)
    seahorse_shape_numba = njit(cache=True)(seahorse_shape_numpy)
else:
    log10_transform_numba = log10_transform_numpy
    interp_linear_numba = interp_linear_numpy
    weighted_residuals_numba = weighted_residuals_numpy
    robust_scalar_loss_numba = robust_scalar_loss_numpy
    seahorse_shape_numba = seahorse_shape_numpy


def log10_transform(x_log: np.ndarray, use_numba: bool = True) -> np.ndarray:
    fn = log10_transform_numba if use_numba and NUMBA_AVAILABLE else log10_transform_numpy
    return fn(np.asarray(x_log, dtype=float))


def interp_linear(times: np.ndarray, sim_time: np.ndarray, sim_values: np.ndarray, use_numba: bool = True) -> np.ndarray:
    fn = interp_linear_numba if use_numba and NUMBA_AVAILABLE else interp_linear_numpy
    return fn(np.asarray(times, dtype=float), np.asarray(sim_time, dtype=float), np.asarray(sim_values, dtype=float))


def weighted_residuals(pred: np.ndarray, obs: np.ndarray, weights: np.ndarray, use_numba: bool = True) -> np.ndarray:
    fn = weighted_residuals_numba if use_numba and NUMBA_AVAILABLE else weighted_residuals_numpy
    return fn(np.asarray(pred, dtype=float), np.asarray(obs, dtype=float), np.asarray(weights, dtype=float))


def robust_scalar_loss(residuals: np.ndarray, loss_code: int, f_scale: float, use_numba: bool = True) -> float:
    fn = robust_scalar_loss_numba if use_numba and NUMBA_AVAILABLE else robust_scalar_loss_numpy
    return float(fn(np.asarray(residuals, dtype=float), int(loss_code), float(f_scale)))


def seahorse_shape(t: np.ndarray, shape_code: int, values: np.ndarray, use_numba: bool = True) -> np.ndarray:
    fn = seahorse_shape_numba if use_numba and NUMBA_AVAILABLE else seahorse_shape_numpy
    return fn(np.asarray(t, dtype=float), int(shape_code), np.asarray(values, dtype=float))
