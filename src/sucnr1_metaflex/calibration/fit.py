"""Parameter fitting routines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import least_squares

from .parameters import load_fit_config
from .objective import compute_residuals


PENALTY_LOSS_THRESHOLD = 1.0e22


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


def run_fit(
    data_dir: str,
    body_model_path: str,
    fit_config_path: str,
    out_dir: str | Path,
    n_starts: int = 1,
) -> Dict[str, Path]:
    """Run multistart least-squares parameter fitting."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fit_cfg = load_fit_config(fit_config_path)

    data = _load_fit_dataset(data_dir, fit_cfg.dataset)
    data = _filter_assays(data, fit_cfg.include_assays, fit_cfg.exclude_assays)

    param_defs = fit_cfg.parameters
    assay_weights = fit_cfg.weights or {}
    observables = fit_cfg.observables or {}

    missing_observables = sorted(set(data["assay"].astype(str).unique()) - set(observables.keys()))
    if missing_observables:
        raise ValueError(
            "The fit config is missing observable mappings for assays: "
            f"{missing_observables}"
        )

    param_names = list(param_defs.keys())
    bounds_log = np.array([list(defn.bounds) for defn in param_defs.values()], dtype=float)
    guess_log = np.array([defn.guess for defn in param_defs.values()], dtype=float)

    results: List[Tuple[int, np.ndarray, float]] = []

    rng = np.random.default_rng(fit_cfg.optimiser.get("random_seed", None))
    max_nfev = int(fit_cfg.optimiser.get("max_iter", 200))

    for start_idx in range(int(n_starts)):
        if start_idx == 0:
            x0 = guess_log
        else:
            lows = bounds_log[:, 0]
            highs = bounds_log[:, 1]
            x0 = rng.uniform(lows, highs)

        def fun(x: np.ndarray) -> np.ndarray:
            return compute_residuals(
                x,
                param_names,
                data,
                body_model_path,
                assay_weights,
                observables,
            )

        try:
            res = least_squares(
                fun,
                x0,
                bounds=(bounds_log[:, 0], bounds_log[:, 1]),
                max_nfev=max_nfev,
                loss="soft_l1",
            )
        except Exception as exc:
            logger.error(f"Least squares failed on start {start_idx}: {exc}")
            continue

        final_loss = float(np.sum(res.fun ** 2))

        if not np.isfinite(final_loss) or final_loss >= PENALTY_LOSS_THRESHOLD:
            logger.error(
                f"Run {start_idx}: invalid objective; loss={final_loss:.3e}. "
                "Check model/config observable compatibility or unstable simulation."
            )
            continue

        results.append((start_idx, res.x, final_loss))
        logger.info(f"Run {start_idx}: loss={final_loss:.3f}")

    if not results:
        raise RuntimeError(
            "Parameter fitting failed for all runs. "
            "Most likely the selected model does not contain the requested observables, "
            "or all multistart simulations were numerically unstable."
        )

    best_start, best_x, best_loss = min(results, key=lambda t: t[2])
    best_params_lin = {
        name: float(10.0 ** val)
        for name, val in zip(param_names, best_x)
    }

    params_csv = out / "best_parameters.csv"
    params_json = out / "best_parameters.json"
    ranked_csv = out / "ranked_multistart.csv"

    pd.DataFrame.from_dict(best_params_lin, orient="index", columns=["value"]).to_csv(params_csv)

    with params_json.open("w", encoding="utf-8") as f:
        json.dump(best_params_lin, f, indent=2)

    ranked_rows = sorted(results, key=lambda t: t[2])
    pd.DataFrame(
        {
            "rank": list(range(1, len(ranked_rows) + 1)),
            "run": [row[0] for row in ranked_rows],
            "loss": [row[2] for row in ranked_rows],
        }
    ).to_csv(ranked_csv, index=False)

    logger.info(f"Best run: {best_start}; loss={best_loss:.6g}")
    logger.info(f"Best parameters written to {params_csv} and {params_json}")

    return {
        "best_parameters_csv": params_csv,
        "best_parameters_json": params_json,
        "ranked_multistart_csv": ranked_csv,
    }