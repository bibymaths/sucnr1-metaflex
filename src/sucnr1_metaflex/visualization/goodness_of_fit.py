from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


_REQUIRED_COLUMNS = {"predicted"}
_OBSERVED_CANDIDATES = ("observed_mean", "observed")


def _safe_r2(y_obs: np.ndarray, y_pred: np.ndarray) -> float:
    """Return R²; NaN if variance is zero or data are insufficient."""
    if y_obs.size < 2:
        return float("nan")

    ss_res = float(np.sum((y_obs - y_pred) ** 2))
    ss_tot = float(np.sum((y_obs - np.mean(y_obs)) ** 2))

    if ss_tot <= 0.0:
        return float("nan")

    return 1.0 - ss_res / ss_tot


def _safe_corr(y_obs: np.ndarray, y_pred: np.ndarray) -> float:
    """Return Pearson correlation; NaN if undefined."""
    if y_obs.size < 2:
        return float("nan")

    if np.std(y_obs) <= 0.0 or np.std(y_pred) <= 0.0:
        return float("nan")

    return float(np.corrcoef(y_obs, y_pred)[0, 1])

def _normalise_residual_table(
    residuals: pd.DataFrame,
    species_col: str = "observable",
) -> pd.DataFrame:
    """
    Return a copy with standard numeric columns:
        observed, predicted

    Accepts either:
        observed_mean, predicted
    or:
        observed, predicted
    """
    missing = sorted(_REQUIRED_COLUMNS - set(residuals.columns))
    if missing:
        raise ValueError(f"Residual table is missing required columns: {missing}")

    observed_col = None
    for col in _OBSERVED_CANDIDATES:
        if col in residuals.columns:
            observed_col = col
            break

    if observed_col is None:
        raise ValueError(
            "Residual table needs one observed-value column. "
            f"Expected one of: {list(_OBSERVED_CANDIDATES)}. "
            f"Available columns: {list(residuals.columns)}"
        )

    if species_col not in residuals.columns:
        raise ValueError(
            f"Residual table has no '{species_col}' column. "
            f"Available columns: {list(residuals.columns)}"
        )

    df = residuals.copy()
    df["observed"] = pd.to_numeric(df[observed_col], errors="coerce")
    df["predicted"] = pd.to_numeric(df["predicted"], errors="coerce")
    df = df[np.isfinite(df["observed"]) & np.isfinite(df["predicted"])]

    return df

def compute_goodness_of_fit_by_species(
    residuals: pd.DataFrame,
    species_col: str = "observable",
) -> pd.DataFrame:
    """
    Compute goodness-of-fit metrics per fitted species/observable.

    Expected columns:
        observed, predicted, and usually observable/species.

    Returns columns:
        species, n, sse, mse, rmse, mae, bias, r2, pearson_r
    """
    df = _normalise_residual_table(residuals, species_col=species_col)

    rows: list[dict[str, float | int | str]] = []

    for species, g in df.groupby(species_col, dropna=False):
        y_obs = pd.to_numeric(g["observed"], errors="coerce").to_numpy(float)
        y_pred = pd.to_numeric(g["predicted"], errors="coerce").to_numpy(float)

        mask = np.isfinite(y_obs) & np.isfinite(y_pred)
        y_obs = y_obs[mask]
        y_pred = y_pred[mask]

        if y_obs.size == 0:
            continue

        err = y_pred - y_obs

        rows.append(
            {
                "species": str(species),
                "n": int(y_obs.size),
                "sse": float(np.sum(err**2)),
                "mse": float(np.mean(err**2)),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "mae": float(np.mean(np.abs(err))),
                "bias": float(np.mean(err)),
                "r2": _safe_r2(y_obs, y_pred),
                "pearson_r": _safe_corr(y_obs, y_pred),
                "observed_min": float(np.min(y_obs)),
                "observed_max": float(np.max(y_obs)),
                "predicted_min": float(np.min(y_pred)),
                "predicted_max": float(np.max(y_pred)),
            }
        )

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values(["rmse", "species"], ascending=[False, True]).reset_index(
            drop=True
        )

    return out


def plot_observed_vs_predicted_by_species(
    residuals: pd.DataFrame | str | Path,
    out_dir: str | Path,
    species_col: str = "observable",
    label_cols: Iterable[str] = ("assay", "condition"),
    filename_prefix: str = "observed_vs_predicted_by_species",
    max_cols: int = 3,
    point_size: float = 28.0,
    alpha: float = 0.75,
    dpi: int = 300,
) -> pd.DataFrame:
    """
    Plot observed vs predicted values separately for each species/observable.

    Parameters
    ----------
    residuals:
        Either a DataFrame or path to a CSV file containing observed/predicted values.
    out_dir:
        Output directory.
    species_col:
        Column identifying species/observable. Usually 'observable'.
    label_cols:
        Optional metadata columns used for point labels if present.
    filename_prefix:
        Output file prefix.
    max_cols:
        Maximum number of subplot columns.
    point_size:
        Scatter point size.
    alpha:
        Scatter transparency.
    dpi:
        Output PNG DPI.

    Returns
    -------
    metrics:
        Per-species goodness-of-fit metrics.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if isinstance(residuals, (str, Path)):
        df = pd.read_csv(residuals)
    else:
        df = residuals.copy()

    df = _normalise_residual_table(df, species_col=species_col)

    metrics = compute_goodness_of_fit_by_species(df, species_col=species_col)
    metrics_path = out_path / "goodness_of_fit_by_species.csv"
    metrics.to_csv(metrics_path, index=False)

    species_order = list(metrics["species"]) if not metrics.empty else sorted(df[species_col].astype(str).unique())

    if len(species_order) == 0:
        raise ValueError("No finite observed/predicted values available for plotting.")

    n_species = len(species_order)
    n_cols = min(max_cols, n_species)
    n_rows = int(np.ceil(n_species / n_cols))

    fig_width = 4.8 * n_cols
    fig_height = 4.3 * n_rows

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )

    metric_lookup = {
        str(row["species"]): row.to_dict() for _, row in metrics.iterrows()
    }

    for ax, species in zip(axes.ravel(), species_order):
        g = df[df[species_col].astype(str) == str(species)].copy()

        x = g["observed"].to_numpy(float)
        y = g["predicted"].to_numpy(float)

        ax.scatter(x, y, s=point_size, alpha=alpha)

        lo = float(np.nanmin([np.min(x), np.min(y)]))
        hi = float(np.nanmax([np.max(x), np.max(y)]))

        if np.isclose(lo, hi):
            pad = max(abs(lo) * 0.05, 1.0)
        else:
            pad = 0.05 * (hi - lo)

        lo -= pad
        hi += pad

        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        m = metric_lookup.get(str(species))

        if m is None:
            r2 = np.nan
            rmse = np.nan
            n = len(g)
        else:
            r2 = float(m.get("r2", np.nan))
            rmse = float(m.get("rmse", np.nan))
            n = int(m.get("n", len(g)))

        ax.set_title(f"{species}\nR²={r2:.3g}, RMSE={rmse:.3g}, n={n}")
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")

        # Add compact labels only when there are few points.
        available_label_cols = [c for c in label_cols if c in g.columns]
        if len(g) <= 12 and available_label_cols:
            for _, row in g.iterrows():
                label = " | ".join(str(row[c]) for c in available_label_cols)
                ax.annotate(
                    label,
                    (float(row["observed"]), float(row["predicted"])),
                    fontsize=7,
                    xytext=(3, 3),
                    textcoords="offset points",
                )

    for ax in axes.ravel()[n_species:]:
        ax.axis("off")

    fig.suptitle("Observed vs predicted by species", y=0.995)
    fig.tight_layout()

    png_path = out_path / f"{filename_prefix}.png"
    pdf_path = out_path / f"{filename_prefix}.pdf"

    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)

    return metrics


def plot_observed_vs_predicted_from_fit_dir(
    fit_plot_dir: str | Path,
    species_col: str = "observable",
) -> pd.DataFrame:
    """
    Convenience wrapper for the existing `sucnr1-plot fit` output directory.

    Expected input:
        <fit_plot_dir>/model_fit/fit_observed_predicted_residuals.csv

    Example:
        plot_observed_vs_predicted_from_fit_dir("results/figures/body_fit")
    """
    fit_plot_dir = Path(fit_plot_dir)

    residual_csv = fit_plot_dir / "model_fit" / "fit_observed_predicted_residuals.csv"
    if not residual_csv.exists():
        raise FileNotFoundError(
            f"Could not find residual table: {residual_csv}. "
            "Run `sucnr1-plot fit` first, or pass the residual DataFrame directly."
        )

    out_dir = fit_plot_dir / "model_fit" / "goodness_of_fit"

    return plot_observed_vs_predicted_by_species(
        residuals=residual_csv,
        out_dir=out_dir,
        species_col=species_col,
    )