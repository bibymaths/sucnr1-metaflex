"""Parameter fitting routines using scalar SciPy optimizers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import differential_evolution, dual_annealing, minimize

from .parameters import load_fit_config
from .objective import compute_residuals


PENALTY_LOSS_THRESHOLD = 1.0e22
DEFAULT_INVALID_LOSS = 1.0e30


def _load_fit_dataset(data_dir: str | Path, dataset: str) -> pd.DataFrame:
    data_path = Path(data_dir)

    filename_map = {
        "body": "dynamic_body.csv",
        "dynamic_body": "dynamic_body.csv",
        "seahorse": "dynamic_seahorse.csv",
        "dynamic_seahorse": "dynamic_seahorse.csv",
        "all": "all_tidy.csv",
        "all_tidy": "all_tidy.csv",
    }

    filename = filename_map.get(str(dataset), str(dataset))
    path = data_path / filename

    if not path.exists():
        raise FileNotFoundError(f"Fit dataset not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Fit dataset is empty: {path}")

    required = {"assay", "time", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Fit dataset {path} is missing columns: {sorted(missing)}")

    return df


def _filter_assays(
    data: pd.DataFrame,
    include_assays: list[str] | None,
    exclude_assays: list[str],
) -> pd.DataFrame:
    df = data.copy()
    df["assay"] = df["assay"].astype(str)

    if include_assays:
        include = {str(x) for x in include_assays}
        df = df[df["assay"].isin(include)]

    if exclude_assays:
        exclude = {str(x) for x in exclude_assays}
        df = df[~df["assay"].isin(exclude)]

    if df.empty:
        raise ValueError("No data rows remain after assay filtering.")

    return df


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False

    return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _robust_scalar_loss(
    residuals: np.ndarray,
    loss_name: str = "soft_l1",
    f_scale: float = 1.0,
) -> float:
    """Convert residual vector to scalar objective.

    Supported losses:
    - linear: sum(r^2)
    - soft_l1: 2 f^2 (sqrt(1 + (r/f)^2) - 1)
    - huber: quadratic near zero, linear in the tails
    - cauchy: f^2 log(1 + (r/f)^2)
    """
    r = np.asarray(residuals, dtype=float).ravel()

    if r.size == 0:
        return DEFAULT_INVALID_LOSS

    if not np.all(np.isfinite(r)):
        return DEFAULT_INVALID_LOSS

    f = max(float(f_scale), 1.0e-12)
    z = (r / f) ** 2
    loss = str(loss_name).strip().lower()

    if loss in {"linear", "l2", "sse"}:
        value = np.sum(r**2)

    elif loss in {"soft_l1", "soft-l1", "softl1"}:
        value = np.sum(2.0 * f**2 * (np.sqrt(1.0 + z) - 1.0))

    elif loss == "huber":
        a = np.abs(r)
        value = np.sum(np.where(a <= f, r**2, 2.0 * f * a - f**2))

    elif loss == "cauchy":
        value = np.sum(f**2 * np.log1p(z))

    else:
        raise ValueError(
            f"Unknown scalar loss '{loss_name}'. "
            "Use one of: linear, soft_l1, huber, cauchy."
        )

    value = float(value)

    if not np.isfinite(value):
        return DEFAULT_INVALID_LOSS

    return value


def _clip_to_bounds(x: np.ndarray, bounds_log: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), bounds_log[:, 0], bounds_log[:, 1])


def _make_objective(
    param_names: list[str],
    data: pd.DataFrame,
    model_path: str,
    assay_weights: Dict[str, float],
    observables: Dict[str, str],
    loss_name: str,
    f_scale: float,
):
    """Build scalar objective function over log10 parameters."""

    def objective(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)

        if x.ndim != 1 or x.size != len(param_names):
            return DEFAULT_INVALID_LOSS

        if not np.all(np.isfinite(x)):
            return DEFAULT_INVALID_LOSS

        try:
            residuals = compute_residuals(
                x,
                param_names,
                data,
                model_path,
                assay_weights,
                observables,
            )
            value = _robust_scalar_loss(
                residuals,
                loss_name=loss_name,
                f_scale=f_scale,
            )
        except Exception as exc:
            logger.debug(f"Objective evaluation failed: {exc}")
            return DEFAULT_INVALID_LOSS

        if not np.isfinite(value):
            return DEFAULT_INVALID_LOSS

        return float(value)

    return objective


def _run_local_minimize(
    objective,
    x0: np.ndarray,
    bounds_log: np.ndarray,
    method: str,
    max_iter: int,
    tol: float | None,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """Run bounded local scalar optimization."""
    x0 = _clip_to_bounds(x0, bounds_log)
    bounds = [(float(lo), float(hi)) for lo, hi in bounds_log]

    method_name = str(method)

    options: Dict[str, Any] = {
        "maxiter": int(max_iter),
        "disp": False,
    }

    if method_name.upper() == "L-BFGS-B":
        options["maxfun"] = int(max_iter)

    res = minimize(
        objective,
        x0,
        method=method_name,
        bounds=bounds,
        tol=tol,
        options=options,
    )

    x = _clip_to_bounds(np.asarray(res.x, dtype=float), bounds_log)
    fun = float(objective(x))

    meta = {
        "optimizer": method_name,
        "success": bool(getattr(res, "success", False)),
        "message": str(getattr(res, "message", "")),
        "nfev": int(getattr(res, "nfev", -1)),
        "nit": int(getattr(res, "nit", -1)),
    }

    return x, fun, meta


def _run_differential_evolution(
    objective,
    bounds_log: np.ndarray,
    optimiser_cfg: Dict[str, Any],
    seed: int | None,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """Run SciPy differential evolution in log-parameter space."""
    bounds = [(float(lo), float(hi)) for lo, hi in bounds_log]

    global_max_iter = _safe_int(optimiser_cfg.get("global_max_iter"), 120)
    popsize = _safe_int(optimiser_cfg.get("popsize"), 8)
    tol = _safe_float(optimiser_cfg.get("global_tol"), 1.0e-4)
    workers = _safe_int(optimiser_cfg.get("workers"), 1)

    updating = "deferred" if workers != 1 else "immediate"

    res = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=global_max_iter,
        popsize=popsize,
        tol=tol,
        polish=False,
        seed=seed,
        workers=workers,
        updating=updating,
    )

    x = _clip_to_bounds(np.asarray(res.x, dtype=float), bounds_log)
    fun = float(objective(x))

    meta = {
        "optimizer": "differential_evolution",
        "success": bool(getattr(res, "success", False)),
        "message": str(getattr(res, "message", "")),
        "nfev": int(getattr(res, "nfev", -1)),
        "nit": int(getattr(res, "nit", -1)),
    }

    return x, fun, meta


def _run_dual_annealing(
    objective,
    bounds_log: np.ndarray,
    optimiser_cfg: Dict[str, Any],
    seed: int | None,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """Run SciPy dual annealing in log-parameter space."""
    bounds = [(float(lo), float(hi)) for lo, hi in bounds_log]

    global_max_iter = _safe_int(optimiser_cfg.get("global_max_iter"), 250)

    res = dual_annealing(
        objective,
        bounds=bounds,
        maxiter=global_max_iter,
        seed=seed,
        no_local_search=True,
    )

    x = _clip_to_bounds(np.asarray(res.x, dtype=float), bounds_log)
    fun = float(objective(x))

    meta = {
        "optimizer": "dual_annealing",
        "success": bool(getattr(res, "success", True)),
        "message": str(getattr(res, "message", "")),
        "nfev": int(getattr(res, "nfev", -1)),
        "nit": int(getattr(res, "nit", -1)),
    }

    return x, fun, meta


def _run_one_start(
    objective,
    x0: np.ndarray,
    bounds_log: np.ndarray,
    optimiser_cfg: Dict[str, Any],
    start_idx: int,
    base_seed: int | None,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """Dispatch one optimization run."""
    method = str(optimiser_cfg.get("method", "hybrid_de_l_bfgs_b")).strip()
    method_l = method.lower()

    local_method = str(optimiser_cfg.get("local_method", "L-BFGS-B")).strip()
    max_iter = _safe_int(optimiser_cfg.get("max_iter"), 1000)
    tol_raw = optimiser_cfg.get("tol", None)
    tol = None if tol_raw is None else _safe_float(tol_raw, 1.0e-8)

    seed = None if base_seed is None else int(base_seed) + int(start_idx)

    if method_l in {"differential_evolution", "de"}:
        return _run_differential_evolution(
            objective=objective,
            bounds_log=bounds_log,
            optimiser_cfg=optimiser_cfg,
            seed=seed,
        )

    if method_l in {"dual_annealing", "da"}:
        return _run_dual_annealing(
            objective=objective,
            bounds_log=bounds_log,
            optimiser_cfg=optimiser_cfg,
            seed=seed,
        )

    if method_l in {
        "hybrid_de_l_bfgs_b",
        "hybrid_de",
        "de_l_bfgs_b",
        "de+lbfgsb",
    }:
        x_de, loss_de, meta_de = _run_differential_evolution(
            objective=objective,
            bounds_log=bounds_log,
            optimiser_cfg=optimiser_cfg,
            seed=seed,
        )

        polish = _as_bool(optimiser_cfg.get("polish"), True)
        if not polish:
            return x_de, loss_de, meta_de

        x_loc, loss_loc, meta_loc = _run_local_minimize(
            objective=objective,
            x0=x_de,
            bounds_log=bounds_log,
            method=local_method,
            max_iter=max_iter,
            tol=tol,
        )

        meta_loc["optimizer"] = f"differential_evolution+{local_method}"
        meta_loc["global_loss"] = float(loss_de)
        meta_loc["global_nfev"] = int(meta_de.get("nfev", -1))
        return x_loc, loss_loc, meta_loc

    if method_l in {
        "hybrid_da_l_bfgs_b",
        "hybrid_da",
        "da_l_bfgs_b",
        "da+lbfgsb",
    }:
        x_da, loss_da, meta_da = _run_dual_annealing(
            objective=objective,
            bounds_log=bounds_log,
            optimiser_cfg=optimiser_cfg,
            seed=seed,
        )

        polish = _as_bool(optimiser_cfg.get("polish"), True)
        if not polish:
            return x_da, loss_da, meta_da

        x_loc, loss_loc, meta_loc = _run_local_minimize(
            objective=objective,
            x0=x_da,
            bounds_log=bounds_log,
            method=local_method,
            max_iter=max_iter,
            tol=tol,
        )

        meta_loc["optimizer"] = f"dual_annealing+{local_method}"
        meta_loc["global_loss"] = float(loss_da)
        meta_loc["global_nfev"] = int(meta_da.get("nfev", -1))
        return x_loc, loss_loc, meta_loc

    # Default: bounded local scalar optimization.
    return _run_local_minimize(
        objective=objective,
        x0=x0,
        bounds_log=bounds_log,
        method=method,
        max_iter=max_iter,
        tol=tol,
    )


def run_fit(
    data_dir: str,
    body_model_path: str,
    fit_config_path: str,
    out_dir: str | Path,
    n_starts: int = 1,
) -> Dict[str, Path]:
    """Run multistart scalar SciPy parameter fitting.

    Parameters are optimized in log10 space and exported in linear space.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fit_cfg = load_fit_config(fit_config_path)

    data = _load_fit_dataset(data_dir, fit_cfg.dataset)
    data = _filter_assays(data, fit_cfg.include_assays, fit_cfg.exclude_assays)

    param_defs = fit_cfg.parameters
    assay_weights = fit_cfg.weights or {}
    observables = fit_cfg.observables or {}

    missing_observables = sorted(
        set(data["assay"].astype(str).unique()) - set(observables.keys())
    )
    if missing_observables:
        raise ValueError(
            "The fit config is missing observable mappings for assays: "
            f"{missing_observables}"
        )

    param_names = list(param_defs.keys())
    bounds_log = np.array(
        [list(defn.bounds) for defn in param_defs.values()],
        dtype=float,
    )
    guess_log = np.array(
        [defn.guess for defn in param_defs.values()],
        dtype=float,
    )

    if bounds_log.ndim != 2 or bounds_log.shape[1] != 2:
        raise ValueError("Parameter bounds must be a two-column array.")

    if len(param_names) == 0:
        raise ValueError("No parameters were provided in the fit config.")

    if np.any(bounds_log[:, 0] >= bounds_log[:, 1]):
        bad = [
            name
            for name, bound in zip(param_names, bounds_log)
            if bound[0] >= bound[1]
        ]
        raise ValueError(f"Invalid bounds for parameters: {bad}")

    guess_log = _clip_to_bounds(guess_log, bounds_log)

    optimiser_cfg = dict(fit_cfg.optimiser or {})
    base_seed_raw = optimiser_cfg.get("random_seed", None)
    base_seed = None if base_seed_raw is None else int(base_seed_raw)

    scalar_loss = str(optimiser_cfg.get("loss", "soft_l1"))
    f_scale = _safe_float(optimiser_cfg.get("f_scale"), 1.0)

    objective = _make_objective(
        param_names=param_names,
        data=data,
        model_path=body_model_path,
        assay_weights=assay_weights,
        observables=observables,
        loss_name=scalar_loss,
        f_scale=f_scale,
    )

    rng = np.random.default_rng(base_seed)

    results: List[Tuple[int, np.ndarray, float, Dict[str, Any]]] = []

    for start_idx in range(int(n_starts)):
        if start_idx == 0:
            x0 = guess_log.copy()
        else:
            lows = bounds_log[:, 0]
            highs = bounds_log[:, 1]
            x0 = rng.uniform(lows, highs)

        try:
            x_opt, final_loss, meta = _run_one_start(
                objective=objective,
                x0=x0,
                bounds_log=bounds_log,
                optimiser_cfg=optimiser_cfg,
                start_idx=start_idx,
                base_seed=base_seed,
            )
        except Exception as exc:
            logger.exception(f"Optimization failed on start {start_idx}: {exc}")
            continue

        if not np.isfinite(final_loss) or final_loss >= PENALTY_LOSS_THRESHOLD:
            logger.error(
                f"Run {start_idx}: invalid objective; loss={final_loss:.3e}. "
                "Check model/config observable compatibility or unstable simulation."
            )
            continue

        results.append((start_idx, x_opt, float(final_loss), meta))

        logger.info(
            f"Run {start_idx}: loss={final_loss:.6g}; "
            f"optimizer={meta.get('optimizer')}; "
            f"success={meta.get('success')}; "
            f"nfev={meta.get('nfev')}"
        )

    if not results:
        raise RuntimeError(
            "Parameter fitting failed for all runs. "
            "Most likely the selected model does not contain the requested observables, "
            "or all multistart simulations were numerically unstable."
        )

    best_start, best_x, best_loss, best_meta = min(results, key=lambda t: t[2])

    best_params_lin = {
        name: float(10.0 ** val)
        for name, val in zip(param_names, best_x)
    }

    best_params_log = {
        name: float(val)
        for name, val in zip(param_names, best_x)
    }

    params_csv = out / "best_parameters.csv"
    params_json = out / "best_parameters.json"
    params_log_csv = out / "best_log10_parameters.csv"
    ranked_csv = out / "ranked_multistart.csv"
    diagnostics_json = out / "fit_diagnostics.json"

    pd.DataFrame.from_dict(
        best_params_lin,
        orient="index",
        columns=["value"],
    ).to_csv(params_csv)

    pd.DataFrame.from_dict(
        best_params_log,
        orient="index",
        columns=["log10_value"],
    ).to_csv(params_log_csv)

    with params_json.open("w", encoding="utf-8") as f:
        json.dump(best_params_lin, f, indent=2)

    ranked_rows = sorted(results, key=lambda t: t[2])

    ranked_records = []
    for rank, (run_idx, x_opt, loss, meta) in enumerate(ranked_rows, start=1):
        ranked_records.append(
            {
                "rank": rank,
                "run": run_idx,
                "loss": float(loss),
                "optimizer": meta.get("optimizer", ""),
                "success": meta.get("success", False),
                "nfev": meta.get("nfev", -1),
                "nit": meta.get("nit", -1),
                "message": meta.get("message", ""),
                "global_loss": meta.get("global_loss", np.nan),
                "global_nfev": meta.get("global_nfev", np.nan),
            }
        )

    pd.DataFrame(ranked_records).to_csv(ranked_csv, index=False)

    diagnostics = {
        "best_run": int(best_start),
        "best_loss": float(best_loss),
        "best_optimizer": best_meta.get("optimizer", ""),
        "best_success": bool(best_meta.get("success", False)),
        "best_message": str(best_meta.get("message", "")),
        "n_parameters": len(param_names),
        "n_successful_runs": len(results),
        "n_requested_starts": int(n_starts),
        "loss": scalar_loss,
        "f_scale": float(f_scale),
        "optimiser": optimiser_cfg,
    }

    with diagnostics_json.open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    logger.info(f"Best run: {best_start}; loss={best_loss:.6g}")
    logger.info(f"Best optimizer: {best_meta.get('optimizer')}")
    logger.info(f"Best parameters written to {params_csv} and {params_json}")
    logger.info(f"Ranked multistart table written to {ranked_csv}")

    return {
        "best_parameters_csv": params_csv,
        "best_parameters_json": params_json,
        "best_log10_parameters_csv": params_log_csv,
        "ranked_multistart_csv": ranked_csv,
        "fit_diagnostics_json": diagnostics_json,
    }
