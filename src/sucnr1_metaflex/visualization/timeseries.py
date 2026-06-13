"""Plots for processed dynamic experimental data."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

from .io import numeric_time_value_frame
from .style import apply_mpl_style, ensure_dir, safe_slug, save_figure


def _available_group_columns(df: pd.DataFrame) -> List[str]:
    candidates = [
        "genotype",
        "condition",
        "treatment",
        "sex",
        "diet",
        "dose",
        "group",
        "protocol",
    ]
    return [col for col in candidates if col in df.columns]


def _label_from_key(key: object, columns: list[str]) -> str:
    if not columns:
        return "mean"

    if not isinstance(key, tuple):
        key = (key,)

    parts = []
    for col, value in zip(columns, key):
        parts.append(f"{col}={value}")
    return ", ".join(parts)


def plot_observed_timeseries(
    df: pd.DataFrame,
    out_dir: str | Path,
    dataset_name: str,
    group_columns: Iterable[str] | None = None,
) -> list[Path]:
    """Plot observed time-series data, one PNG per assay."""
    apply_mpl_style()
    out = ensure_dir(out_dir)

    df = numeric_time_value_frame(df)

    if "assay" not in df.columns:
        raise ValueError("Expected column 'assay' in processed data.")

    if group_columns is None:
        group_columns = _available_group_columns(df)

    group_columns = [col for col in group_columns if col in df.columns]

    written: list[Path] = []

    for assay, assay_df in df.groupby("assay", dropna=False):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))

        if group_columns:
            grouped_iter = assay_df.groupby(group_columns, dropna=False)
        else:
            grouped_iter = [(("mean",), assay_df)]

        for key, sub in grouped_iter:
            agg = (
                sub.groupby("time", as_index=False)["value"]
                .agg(["mean", "std", "count"])
                .reset_index()
                .sort_values("time")
            )

            yerr = agg["std"].fillna(0.0).to_numpy(dtype=float)

            ax.errorbar(
                agg["time"].to_numpy(dtype=float),
                agg["mean"].to_numpy(dtype=float),
                yerr=yerr,
                marker="o",
                linewidth=1.5,
                capsize=2,
                label=_label_from_key(key, group_columns),
            )

        ax.set_title(f"{dataset_name}: {assay}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Observed value")

        if group_columns:
            ax.legend(fontsize=8)

        path = out / f"{safe_slug(dataset_name)}__{safe_slug(assay)}.png"
        save_figure(fig, path)
        written.append(path)

    logger.info(f"Wrote {len(written)} observed time-series plots to {out}")
    return written


def plot_processed_data_directory(
    data_dir: str | Path,
    out_dir: str | Path,
) -> list[Path]:
    """Plot all recognized processed dynamic tables."""
    data_path = Path(data_dir)
    written: list[Path] = []

    candidates = [
        ("dynamic_body.csv", "body"),
        ("dynamic_seahorse.csv", "seahorse"),
        ("all_tidy.csv", "all_tidy"),
    ]

    for filename, label in candidates:
        path = data_path / filename
        if not path.exists():
            logger.warning(f"Skipping missing processed table: {path}")
            continue

        df = pd.read_csv(path)
        if df.empty:
            logger.warning(f"Skipping empty processed table: {path}")
            continue

        written.extend(
            plot_observed_timeseries(
                df=df,
                out_dir=Path(out_dir) / label,
                dataset_name=label,
            )
        )

    return written