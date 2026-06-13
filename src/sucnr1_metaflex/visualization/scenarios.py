"""Scenario and perturbation plotting."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from .style import apply_mpl_style, ensure_dir, safe_slug, save_figure


def _scenario_csvs(scenario_dir: str | Path) -> list[Path]:
    path = Path(scenario_dir)

    if not path.exists():
        raise FileNotFoundError(f"Scenario directory not found: {path}")

    csvs = []
    for item in sorted(path.glob("*.csv")):
        if item.name.startswith("endpoint_delta"):
            continue
        csvs.append(item)

    return csvs


def _numeric_observables(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col == "time":
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().any():
            cols.append(col)
    return cols


def plot_scenario_timecourses(
    scenario_dir: str | Path,
    out_dir: str | Path,
    observables: Iterable[str] | None = None,
) -> list[Path]:
    """Overlay scenario time courses, one figure per observable."""
    apply_mpl_style()
    out = ensure_dir(out_dir)

    csvs = _scenario_csvs(scenario_dir)
    if not csvs:
        logger.warning(f"No scenario CSV files found in {scenario_dir}")
        return []

    loaded = []
    observable_set: set[str] = set()

    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        if "time" not in df.columns:
            continue

        df["time"] = pd.to_numeric(df["time"], errors="coerce")
        df = df[df["time"].notna()]

        obs = _numeric_observables(df)
        observable_set.update(obs)
        loaded.append((csv_path.stem, df))

    if observables is None:
        selected = sorted(observable_set)
    else:
        selected = [str(x) for x in observables]

    written: list[Path] = []

    for obs in selected:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))

        plotted = False

        for scenario_name, df in loaded:
            if obs not in df.columns:
                continue

            y = pd.to_numeric(df[obs], errors="coerce")
            valid = df["time"].notna() & y.notna()

            if not valid.any():
                continue

            ax.plot(
                df.loc[valid, "time"].to_numpy(dtype=float),
                y.loc[valid].to_numpy(dtype=float),
                linewidth=1.5,
                label=scenario_name,
            )
            plotted = True

        if not plotted:
            plt.close(fig)
            continue

        ax.set_title(f"Scenario trajectories: {obs}")
        ax.set_xlabel("Time")
        ax.set_ylabel(obs)
        ax.legend(fontsize=8)

        path = out / f"scenario_timecourse__{safe_slug(obs)}.png"
        save_figure(fig, path)
        written.append(path)

    logger.info(f"Wrote {len(written)} scenario time-course plots to {out}")
    return written


def compute_endpoint_deltas(
    scenario_dir: str | Path,
    baseline_name: str = "baseline",
) -> pd.DataFrame:
    """Compute final-time deltas relative to a baseline scenario."""
    csvs = _scenario_csvs(scenario_dir)

    if not csvs:
        return pd.DataFrame()

    loaded: dict[str, pd.DataFrame] = {}

    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        if "time" not in df.columns:
            continue

        df["time"] = pd.to_numeric(df["time"], errors="coerce")
        df = df[df["time"].notna()].sort_values("time")

        if not df.empty:
            loaded[csv_path.stem] = df

    if not loaded:
        return pd.DataFrame()

    if baseline_name in loaded:
        baseline_key = baseline_name
    else:
        baseline_key = sorted(loaded.keys())[0]
        logger.warning(f"Baseline '{baseline_name}' not found; using '{baseline_key}'.")

    baseline_df = loaded[baseline_key]
    baseline_final = baseline_df.iloc[-1]

    records: list[dict[str, object]] = []

    for scenario_name, df in loaded.items():
        final = df.iloc[-1]

        for obs in _numeric_observables(df):
            if obs not in baseline_final.index:
                continue

            base = pd.to_numeric(pd.Series([baseline_final[obs]]), errors="coerce").iloc[0]
            value = pd.to_numeric(pd.Series([final[obs]]), errors="coerce").iloc[0]

            if not np.isfinite(base) or not np.isfinite(value):
                continue

            delta = float(value - base)
            rel_pct = np.nan
            if abs(float(base)) > 1e-12:
                rel_pct = 100.0 * delta / float(base)

            records.append(
                {
                    "scenario": scenario_name,
                    "baseline": baseline_key,
                    "observable": obs,
                    "baseline_final": float(base),
                    "scenario_final": float(value),
                    "delta": delta,
                    "relative_percent": float(rel_pct) if np.isfinite(rel_pct) else np.nan,
                }
            )

    return pd.DataFrame.from_records(records)


def plot_endpoint_deltas(
    scenario_dir: str | Path,
    out_dir: str | Path,
    baseline_name: str = "baseline",
) -> list[Path]:
    """Plot endpoint deltas relative to baseline, one figure per observable."""
    apply_mpl_style()
    out = ensure_dir(out_dir)

    table = compute_endpoint_deltas(scenario_dir, baseline_name=baseline_name)

    if table.empty:
        logger.warning("No endpoint deltas computed.")
        return []

    csv_path = out / "endpoint_deltas.csv"
    table.to_csv(csv_path, index=False)

    written: list[Path] = []

    for obs, obs_df in table.groupby("observable", dropna=False):
        obs = str(obs)
        obs_df = obs_df.sort_values("delta")

        height = max(4.5, 0.3 * len(obs_df))
        fig, ax = plt.subplots(figsize=(8.0, height))

        ax.barh(
            obs_df["scenario"].astype(str),
            obs_df["delta"].to_numpy(dtype=float),
        )

        ax.axvline(0.0, linewidth=1.0)
        ax.set_title(f"Endpoint delta: {obs}")
        ax.set_xlabel("Scenario final - baseline final")
        ax.set_ylabel("Scenario")

        path = out / f"endpoint_delta__{safe_slug(obs)}.png"
        save_figure(fig, path)
        written.append(path)

    logger.info(f"Wrote {len(written)} endpoint-delta plots to {out}")
    return written


def plot_scenario_diagnostics(
    scenario_dir: str | Path,
    out_dir: str | Path,
    baseline_name: str = "baseline",
) -> list[Path]:
    """Generate all scenario plots."""
    out = ensure_dir(out_dir)

    written = []
    written.extend(plot_scenario_timecourses(scenario_dir, out / "timecourses"))
    written.extend(plot_endpoint_deltas(scenario_dir, out / "endpoint_deltas", baseline_name))

    return written