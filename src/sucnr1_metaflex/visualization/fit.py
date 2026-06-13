"""Calibration and model-fit plots."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from sucnr1_metaflex.calibration.parameters import load_fit_config
from sucnr1_metaflex.simulation.roadrunner_engine import load_model

from .io import load_parameter_file, numeric_time_value_frame
from .style import apply_mpl_style, ensure_dir, safe_slug, save_figure


def _load_fit_plot_dataset(data_dir: str | Path, dataset: str) -> pd.DataFrame:
    """Load the same processed dataset used during fitting."""
    filename_map = {
        "body": "dynamic_body.csv",
        "dynamic_body": "dynamic_body.csv",
        "seahorse": "dynamic_seahorse.csv",
        "dynamic_seahorse": "dynamic_seahorse.csv",
        "all": "all_tidy.csv",
        "all_tidy": "all_tidy.csv",
    }

    filename = filename_map.get(str(dataset), str(dataset))
    path = Path(data_dir) / filename

    if not path.exists():
        raise FileNotFoundError(f"Plot dataset not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Plot dataset is empty: {path}")

    required = {"assay", "time", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Plot dataset {path} is missing columns: {sorted(missing)}")

    return df


def _set_rr_parameters(rr: object, params: Dict[str, float]) -> None:
    for pid, value in params.items():
        try:
            rr[pid] = float(value)  # type: ignore[index]
        except Exception:
            continue

def _simulate_fitted_to_times(
    model_path: str | Path,
    params: Dict[str, float],
    times: np.ndarray,
    selections: list[str],
) -> pd.DataFrame:
    """Simulate a fitted model to requested times without losing parameters.

    This avoids using simulate_to_times here because that helper may reset the
    RoadRunner instance internally, which can restore SBML default parameters.
    """
    rr = load_model(str(model_path))
    _set_rr_parameters(rr, params)

    times = np.asarray(times, dtype=float)
    times = times[np.isfinite(times)]

    if times.size == 0:
        raise ValueError("No finite simulation times were provided.")

    unique_times = np.unique(times)
    t_min = float(np.min(unique_times))
    t_max = float(np.max(unique_times))

    ycols = [str(x) for x in selections]
    rr.selections = ["time"] + ycols

    # If there is only one time point, simulate a tiny interval and interpolate.
    # This is safer than trying to read all assignment-rule outputs manually.
    start = min(0.0, t_min)
    end = t_max

    if np.isclose(start, end):
        end = start + 1.0e-9

    n_points = max(200, 10 * len(unique_times))
    n_points = min(n_points, 5000)

    sim = rr.simulate(start, end, n_points)
    sim_df = pd.DataFrame(sim, columns=["time"] + ycols)

    out = pd.DataFrame({"time": unique_times})

    sim_time = sim_df["time"].to_numpy(dtype=float)

    for col in ycols:
        values = pd.to_numeric(sim_df[col], errors="coerce").to_numpy(dtype=float)

        if not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite simulated values for {col}")

        out[col] = np.interp(unique_times, sim_time, values)

    return out

def plot_ranked_multistart(
    fit_dir: str | Path,
    out_dir: str | Path,
) -> Path | None:
    """Plot ranked multistart losses."""
    apply_mpl_style()

    fit_path = Path(fit_dir)
    out = ensure_dir(out_dir)

    ranked_path = fit_path / "ranked_multistart.csv"
    if not ranked_path.exists():
        logger.warning(f"Missing ranked multistart file: {ranked_path}")
        return None

    df = pd.read_csv(ranked_path)

    if df.empty or "loss" not in df.columns:
        logger.warning(f"Ranked multistart file has no loss column: {ranked_path}")
        return None

    df = df.sort_values("loss").reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(df["rank"], df["loss"], marker="o", linewidth=1.5)
    ax.set_title("Ranked multistart losses")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Loss")

    positive_loss = df["loss"].to_numpy(dtype=float)
    positive_loss = positive_loss[np.isfinite(positive_loss) & (positive_loss > 0.0)]
    if positive_loss.size > 0:
        ax.set_yscale("log")

    path = out / "ranked_multistart_loss.png"
    return save_figure(fig, path)


def plot_best_parameters(
    params_path: str | Path,
    out_dir: str | Path,
    top_n: int | None = None,
) -> Path:
    """Plot fitted parameter values on log10 scale."""
    apply_mpl_style()

    out = ensure_dir(out_dir)
    params = load_parameter_file(params_path)

    series = pd.Series(params, dtype=float)
    series = series[np.isfinite(series) & (series > 0.0)]
    series = np.log10(series).sort_values()

    if top_n is not None and top_n > 0:
        series = series.iloc[:top_n]

    height = max(4.5, 0.28 * len(series))

    fig, ax = plt.subplots(figsize=(8.0, height))
    ax.barh(series.index.astype(str), series.to_numpy(dtype=float))
    ax.set_title("Best-fit parameters")
    ax.set_xlabel("log10(value)")
    ax.set_ylabel("Parameter")

    path = out / "best_parameters_log10.png"
    return save_figure(fig, path)


def compute_fit_residual_table(
    data_dir: str | Path,
    model_path: str | Path,
    params_path: str | Path,
    config_path: str | Path,
) -> pd.DataFrame:
    """Compute observed, predicted and residual values by assay/time."""
    fit_cfg = load_fit_config(str(config_path))

    data = _load_fit_plot_dataset(data_dir, fit_cfg.dataset)
    data = numeric_time_value_frame(data)
    data["assay"] = data["assay"].astype(str)

    observables = getattr(fit_cfg, "observables", {}) or {}
    params = load_parameter_file(params_path)

    missing = sorted(set(data["assay"].unique()) - set(observables.keys()))
    if missing:
        raise ValueError(
            "Plot config is missing observable mappings for assays: "
            f"{missing}"
        )

    records: list[dict[str, object]] = []

    for assay, assay_df in data.groupby("assay", dropna=False):
        assay = str(assay)
        species_id = str(observables[assay])

        agg = (
            assay_df.groupby("time")
            .agg(
                observed_mean=("value", "mean"),
                observed_sd=("value", "std"),
                n=("value", "count"),
            )
            .reset_index()
            .sort_values("time")
        )

        if agg.empty:
            continue

        times = agg["time"].to_numpy(dtype=float)

        try:
            sim = _simulate_fitted_to_times(
                model_path=model_path,
                params=params,
                times=times,
                selections=[species_id],
            )
        except Exception as exc:
            logger.warning(
                f"Skipping assay={assay}; simulation failed for {species_id}: {exc}"
            )
            continue

        if species_id not in sim.columns:
            logger.warning(
                f"Skipping assay={assay}; missing simulated species {species_id}"
            )
            continue

        pred = pd.to_numeric(sim[species_id], errors="coerce").to_numpy(dtype=float)

        if pred.shape[0] != agg.shape[0]:
            logger.warning(
                f"Skipping assay={assay}; prediction/data length mismatch: "
                f"pred={pred.shape[0]}, obs={agg.shape[0]}"
            )
            continue

        for i, row in agg.reset_index(drop=True).iterrows():
            observed = float(row["observed_mean"])
            predicted = float(pred[i])

            if not np.isfinite(observed) or not np.isfinite(predicted):
                continue

            records.append(
                {
                    "assay": assay,
                    "observable": species_id,
                    "time": float(row["time"]),
                    "observed_mean": observed,
                    "observed_sd": (
                        float(row["observed_sd"])
                        if pd.notna(row["observed_sd"])
                        else np.nan
                    ),
                    "n": int(row["n"]),
                    "predicted": predicted,
                    "residual": predicted - observed,
                }
            )

    return pd.DataFrame.from_records(records)


def plot_model_fit(
    data_dir: str | Path,
    model_path: str | Path,
    params_path: str | Path,
    config_path: str | Path,
    out_dir: str | Path,
) -> list[Path]:
    """Plot observed mean ± SD against fitted model prediction."""
    apply_mpl_style()

    out = ensure_dir(out_dir)

    table = compute_fit_residual_table(
        data_dir=data_dir,
        model_path=model_path,
        params_path=params_path,
        config_path=config_path,
    )

    if table.empty:
        logger.warning("No fit residual table rows generated.")
        return []

    residual_csv = out / "fit_observed_predicted_residuals.csv"
    table.to_csv(residual_csv, index=False)

    written: list[Path] = []

    for assay, assay_df in table.groupby("assay", dropna=False):
        assay = str(assay)
        species_id = str(assay_df["observable"].iloc[0])

        assay_df = assay_df.sort_values("time")

        fig, ax = plt.subplots(figsize=(7.0, 4.5))

        yerr = assay_df["observed_sd"].fillna(0.0).to_numpy(dtype=float)

        ax.errorbar(
            assay_df["time"].to_numpy(dtype=float),
            assay_df["observed_mean"].to_numpy(dtype=float),
            yerr=yerr,
            marker="o",
            linestyle="none",
            capsize=2,
            label="Observed mean",
        )

        ax.plot(
            assay_df["time"].to_numpy(dtype=float),
            assay_df["predicted"].to_numpy(dtype=float),
            marker="x",
            linewidth=1.5,
            label=f"Model: {species_id}",
        )

        ax.set_title(f"Fit: {assay}")
        ax.set_xlabel("Time")
        ax.set_ylabel(species_id)
        ax.legend(fontsize=8)

        path = out / f"fit__{safe_slug(assay)}__{safe_slug(species_id)}.png"
        save_figure(fig, path)
        written.append(path)

    logger.info(f"Wrote {len(written)} fit plots to {out}")
    return written


def plot_fit_diagnostics(
    fit_dir: str | Path,
    data_dir: str | Path,
    model_path: str | Path,
    config_path: str | Path,
    out_dir: str | Path,
) -> list[Path]:
    """Generate all standard fit diagnostic plots."""
    out = ensure_dir(out_dir)
    fit_path = Path(fit_dir)

    written: list[Path] = []

    ranked = plot_ranked_multistart(fit_path, out)
    if ranked is not None:
        written.append(ranked)

    params_json = fit_path / "best_parameters.json"
    params_csv = fit_path / "best_parameters.csv"

    if params_json.exists():
        params_path = params_json
    elif params_csv.exists():
        params_path = params_csv
    else:
        logger.warning(f"No best parameter file found in {fit_path}")
        return written

    written.append(plot_best_parameters(params_path, out))

    written.extend(
        plot_model_fit(
            data_dir=data_dir,
            model_path=model_path,
            params_path=params_path,
            config_path=config_path,
            out_dir=out / "model_fit",
        )
    )

    return written
