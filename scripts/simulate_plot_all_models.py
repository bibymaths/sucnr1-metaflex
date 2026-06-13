#!/usr/bin/env python
"""Simulate and plot body, liver and combined SUCNR1 SBML models.

This script uses the package simulation module, not a separate ODE
implementation.  It writes one CSV per model plus PNG plots for each
selected observable.

Examples
--------
Pre-fit baseline simulation:

    python scripts/simulate_plot_all_models.py \
      --start 0 --end 24 --num 240 \
      --out results/simulations/prefit

Post-fit simulation, using best_parameters.json when available:

    python scripts/simulate_plot_all_models.py \
      --start 0 --end 24 --num 240 \
      --out results/simulations/postfit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import pandas as pd

from sucnr1_metaflex.simulation.forward import run_forward
from sucnr1_metaflex.simulation.roadrunner_engine import load_model


MODEL_SPECS = {
    "body": {
        "model": Path("results/models/body.xml"),
        "params": Path("results/runs/body_fit/fit/best_parameters.json"),
        "selections": [
            "G_mgdl",
            "G_plasma",
            "I_eff",
            "Pyr_plasma",
            "AA_plasma",
            "Ketone_plasma",
            "Succ_plasma",
        ],
    },
    "liver": {
        "model": Path("results/models/liver.xml"),
        "params": Path("results/runs/liver_fit/fit/best_parameters.json"),
        "selections": [
            "G6P_liver",
            "Glycogen_liver",
            "Pyr_liver",
            "Succ_mito",
            "Succ_extra",
            "Mito_capacity",
            "OCR_proxy",
            "ECAR_proxy",
            "mito_OCR_proxy",
        ],
    },
    "combined": {
        "model": Path("results/models/body_liver.xml"),
        "params": Path("results/runs/combined_fit/fit/best_parameters.json"),
        "selections": [
            "G_mgdl",
            "G_plasma",
            "I_eff",
            "Pyr_plasma",
            "AA_plasma",
            "Ketone_plasma",
            "Succ_plasma",
            "G6P_liver",
            "Glycogen_liver",
            "Pyr_liver",
            "Succ_mito",
            "Succ_extra",
            "Mito_capacity",
            "OCR_proxy",
            "ECAR_proxy",
            "mito_OCR_proxy",
        ],
    },
}


def load_params(path: Path, use_params: bool) -> Dict[str, float]:
    """Load fitted parameters if available."""
    if not use_params:
        return {}

    if not path.exists():
        print(f"[warn] parameter file not found; using SBML defaults: {path}")
        return {}

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    params: Dict[str, float] = {}
    for key, value in raw.items():
        try:
            params[str(key)] = float(value)
        except Exception:
            print(f"[warn] skipping non-numeric parameter {key}={value!r}")

    return params


def available_symbols(model_path: Path) -> set[str]:
    """Return species and global-parameter symbols available for selection."""
    rr = load_model(str(model_path))

    symbols: set[str] = set()

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
                symbols.update(str(x) for x in fn())
            except Exception:
                pass

    return symbols


def filter_selections(model_path: Path, requested: Iterable[str]) -> List[str]:
    """Keep only selections that exist in the SBML model."""
    symbols = available_symbols(model_path)
    selected = [x for x in requested if x in symbols]

    missing = [x for x in requested if x not in symbols]
    if missing:
        print(f"[warn] {model_path}: skipping missing selections: {missing}")

    if not selected:
        raise RuntimeError(f"No valid selections found for {model_path}")

    return selected


def save_wide_plot(df: pd.DataFrame, model_name: str, out_dir: Path) -> Path:
    """Save one overview plot with all selected series."""
    fig, ax = plt.subplots(figsize=(9.0, 5.2))

    for col in df.columns:
        if col == "time":
            continue
        ax.plot(df["time"], df[col], linewidth=1.6, label=col)

    ax.set_title(f"{model_name}: forward simulation")
    ax.set_xlabel("Time")
    # ax.set_xscale("log")
    ax.set_ylabel("Model value")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()

    path = out_dir / f"{model_name}__overview.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_single_series_plots(df: pd.DataFrame, model_name: str, out_dir: Path) -> list[Path]:
    """Save one plot per simulated variable."""
    written: list[Path] = []

    for col in df.columns:
        if col == "time":
            continue

        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        ax.plot(df["time"], df[col], linewidth=1.8)
        ax.set_title(f"{model_name}: {col}")
        ax.set_xlabel("Time")
        ax.set_ylabel(col)
        fig.tight_layout()

        safe_col = (
            col.replace("/", "_")
            .replace(" ", "_")
            .replace(":", "_")
            .replace("__", "_")
        )
        path = out_dir / f"{model_name}__{safe_col}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)

        written.append(path)

    return written


def simulate_one(
    model_name: str,
    model_path: Path,
    params_path: Path,
    selections: list[str],
    start: float,
    end: float,
    num: int,
    out_root: Path,
    use_params: bool,
) -> None:
    """Run one model simulation and write CSV/PNG outputs."""
    if not model_path.exists():
        print(f"[warn] skipping missing model: {model_path}")
        return

    out_dir = out_root / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = filter_selections(model_path, selections)
    params = load_params(params_path, use_params=use_params)

    df = run_forward(
        model_path=str(model_path),
        params=params,
        start=float(start),
        end=float(end),
        num_points=int(num),
        selections=selected,
    )

    csv_path = out_dir / f"{model_name}__forward.csv"
    df.to_csv(csv_path, index=False)

    overview_path = save_wide_plot(df, model_name, out_dir)
    single_paths = save_single_series_plots(df, model_name, out_dir)

    print(f"[ok] {model_name}")
    print(f"     csv:      {csv_path}")
    print(f"     overview: {overview_path}")
    print(f"     n_series_plots: {len(single_paths)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate and plot body, liver and combined SUCNR1 models."
    )
    parser.add_argument("--start", type=float, default=0.0, help="Simulation start time.")
    parser.add_argument("--end", type=float, default=24.0, help="Simulation end time.")
    parser.add_argument("--num", type=int, default=240, help="Number of output time points.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/simulations/all_models"),
        help="Output directory.",
    )
    parser.add_argument(
        "--no-fit-params",
        action="store_true",
        help="Ignore best_parameters.json files and use SBML defaults.",
    )

    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    for model_name, spec in MODEL_SPECS.items():
        simulate_one(
            model_name=model_name,
            model_path=spec["model"],
            params_path=spec["params"],
            selections=spec["selections"],
            start=args.start,
            end=args.end,
            num=args.num,
            out_root=args.out,
            use_params=not args.no_fit_params,
        )


if __name__ == "__main__":
    main()
