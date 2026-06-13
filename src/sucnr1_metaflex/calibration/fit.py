"""Parameter fitting routines.

This module orchestrates parameter optimisation using a simple
least–squares objective.  It supports multi–start optimisation by
sampling initial guesses within the specified bounds.  Results are
written to CSV/JSON files in the specified output directory.
"""

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
from ..simulation.protocols import load_dynamic_body_data


def run_fit(data_dir: str, body_model_path: str, fit_config_path: str, out_dir: str | Path, n_starts: int = 1) -> Dict[
    str, Path]:
    """Run parameter fitting against body dynamics.

    Args:
        data_dir: Directory containing processed dynamic CSV files.
        body_model_path: Path to the body SBML model.
        fit_config_path: Path to the fit configuration YAML.
        out_dir: Directory into which results will be written.
        n_starts: Number of random initialisations to perform.

    Returns:
        Mapping of result names to file paths.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Load data and config
    data = load_dynamic_body_data(data_dir)
    fit_cfg = load_fit_config(fit_config_path)
    param_defs = fit_cfg.parameters
    assay_weights = fit_cfg.weights or {}
    param_names = list(param_defs.keys())
    bounds_log = np.array([list(defn.bounds) for defn in param_defs.values()])  # shape (n_params, 2)
    guess_log = np.array([defn.guess for defn in param_defs.values()])
    # Collect results
    results: List[Tuple[np.ndarray, float]] = []
    rng = np.random.default_rng(fit_cfg.optimiser.get("random_seed", None))
    for start_idx in range(int(n_starts)):
        # initial guess in log10 space; random uniform within bounds
        if start_idx == 0:
            x0 = guess_log
        else:
            lows = bounds_log[:, 0]
            highs = bounds_log[:, 1]
            x0 = rng.uniform(lows, highs)
        # objective wrapper
        fun = lambda x: compute_residuals(
                                            x,
                                            param_names,
                                            data,
                                            body_model_path,
                                            assay_weights,
                                            fit_cfg.observables,
                                        )
        try:
            res = least_squares(
                fun,
                x0,
                bounds=(bounds_log[:, 0], bounds_log[:, 1]),
                max_nfev=int(fit_cfg.optimiser.get("max_iter", 200)),
                loss="soft_l1",
            )
        except Exception as exc:
            logger.error(f"Least squares failed on start {start_idx}: {exc}")
            continue
        final_loss = float(np.sum(res.fun ** 2))
        results.append((res.x, final_loss))
        logger.info(f"Run {start_idx}: loss={final_loss:.3f}")
    if not results:
        raise RuntimeError("Parameter fitting failed for all runs")
    # Pick best result
    best_x, best_loss = min(results, key=lambda t: t[1])
    # Convert to linear scale
    best_params_lin = {name: float(10 ** val) for name, val in zip(param_names, best_x)}
    # Write outputs
    params_csv = out / "best_parameters.csv"
    params_json = out / "best_parameters.json"
    pd.DataFrame.from_dict(best_params_lin, orient="index", columns=["value"]).to_csv(params_csv)
    with params_json.open("w", encoding="utf-8") as f:
        json.dump(best_params_lin, f, indent=2)
    # Save ranking of runs
    ranked_csv = out / "ranked_multistart.csv"
    with ranked_csv.open("w", encoding="utf-8") as f:
        f.write("run,loss\n")
        for idx, (_, loss) in enumerate(results):
            f.write(f"{idx},{loss}\n")
    logger.info(f"Best parameters written to {params_csv} and {params_json}")
    return {
        "best_parameters_csv": params_csv,
        "best_parameters_json": params_json,
        "ranked_multistart_csv": ranked_csv,
    }
