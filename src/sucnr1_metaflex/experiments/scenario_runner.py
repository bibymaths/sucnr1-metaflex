"""Scenario engine.

This module applies parameter multipliers to an SBML model and runs
a forward simulation for each scenario.  Multipliers are defined
in a YAML configuration file.  For each scenario, baseline
parameters are multiplied by the specified factors and the model is
simulated on a simple time grid.  Results are written to CSV
files in the designated output directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
import yaml

from ..simulation.forward import run_forward


def run_scenarios(model_path: str, params_path: str, scenarios_config: str, out_dir: str | Path) -> Dict[str, Path]:
    """Run a suite of in silico scenarios.

    Args:
        model_path: Path to the SBML model (typically combined model).
        params_path: Path to JSON file of baseline parameter values.
        scenarios_config: YAML file containing a list of scenarios.
        out_dir: Directory into which scenario results will be written.

    Returns:
        Mapping of scenario names to CSV file paths.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(params_path, "r", encoding="utf-8") as f:
        baseline_params: Dict[str, float] = json.load(f)
    import pathlib
    cfg_path = pathlib.Path(scenarios_config)
    if not cfg_path.exists():
        repo_root = pathlib.Path(__file__).resolve().parents[3]
        alt = repo_root / "configs" / cfg_path.name
        if alt.exists():
            cfg_path = alt
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    scenarios = cfg.get("scenarios", [])
    results: Dict[str, Path] = {}
    for sc in scenarios:
        name = sc.get("name")
        multipliers = sc.get("multipliers", {})
        params = baseline_params.copy()
        # apply multipliers
        for pid, mult in multipliers.items():
            try:
                params[pid] = params.get(pid, 1.0) * float(mult)
            except Exception:
                continue
        # run simulation
        try:
            df = run_forward(model_path, params, start=0.0, end=120.0, num_points=50, selections=["G_plasma"])
        except Exception as exc:
            logger.error(f"Simulation failed for scenario {name}: {exc}")
            continue
        # save CSV
        csv_path = out / f"{name}.csv"
        df.to_csv(csv_path, index=False)
        # optionally plot
        try:
            plt.figure()
            plt.plot(df["time"], df["G_plasma"], label=name)
            plt.xlabel("Time")
            plt.ylabel("G_plasma")
            plt.title(f"Scenario: {name}")
            plt.legend()
            png_path = out / f"{name}.png"
            plt.savefig(png_path)
            plt.close()
        except Exception:
            pass
        results[name] = csv_path
        logger.info(f"Scenario {name} simulated and saved to {csv_path}")
    return results
