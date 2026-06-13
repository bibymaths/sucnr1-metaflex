"""CLI entry points for visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich import print

from .fit import plot_fit_diagnostics
from .scenarios import plot_scenario_diagnostics, plot_scenario_timecourses
from .timeseries import plot_processed_data_directory

plot_app = typer.Typer(help="Static plotting and visualization commands.")


@plot_app.command("data")
def plot_data(
    data: str = typer.Option("results/processed", "--data", help="Processed data directory."),
    out: str = typer.Option("results/figures/data", "--out", help="Output figure directory."),
) -> None:
    """Plot processed experimental time-series data."""
    written = plot_processed_data_directory(data, out)
    print(f"Wrote {len(written)} data plot(s) to {out}")


@plot_app.command("fit")
def plot_fit(
    fit_dir: str = typer.Option("results/runs/run1/fit", "--fit-dir", help="Fit result directory."),
    data: str = typer.Option("results/processed", "--data", help="Processed data directory."),
    model: str = typer.Option("results/models/body.xml", "--model", help="SBML model path."),
    config: str = typer.Option("configs/fit.yaml", "--config", help="Fit YAML config."),
    out: str = typer.Option("results/figures/fit", "--out", help="Output figure directory."),
) -> None:
    """Plot fit diagnostics, fitted parameters, and observed-vs-predicted curves."""
    written = plot_fit_diagnostics(
        fit_dir=fit_dir,
        data_dir=data,
        model_path=model,
        config_path=config,
        out_dir=out,
    )
    print(f"Wrote {len(written)} fit plot(s) to {out}")


@plot_app.command("scenarios")
def plot_scenarios(
    scenario_dir: str = typer.Option("results/scenarios", "--scenario-dir", help="Scenario CSV directory."),
    out: str = typer.Option("results/figures/scenarios", "--out", help="Output figure directory."),
    baseline: str = typer.Option("baseline", "--baseline", help="Baseline scenario file stem."),
    observable: Optional[str] = typer.Option(None, "--observable", help="Optional single observable to plot."),
) -> None:
    """Plot scenario time courses and endpoint deltas."""
    if observable:
        written = plot_scenario_timecourses(
            scenario_dir=scenario_dir,
            out_dir=Path(out) / "timecourses",
            observables=[observable],
        )
    else:
        written = plot_scenario_diagnostics(
            scenario_dir=scenario_dir,
            out_dir=out,
            baseline_name=baseline,
        )

    print(f"Wrote {len(written)} scenario plot(s) to {out}")


@plot_app.command("all")
def plot_all(
    data: str = typer.Option("results/processed", "--data"),
    fit_dir: str = typer.Option("results/runs/run1/fit", "--fit-dir"),
    model: str = typer.Option("results/models/body.xml", "--model"),
    config: str = typer.Option("configs/fit.yaml", "--config"),
    scenario_dir: str = typer.Option("results/scenarios", "--scenario-dir"),
    out: str = typer.Option("results/figures", "--out"),
) -> None:
    """Generate all available static plots."""
    total = 0

    data_out = Path(out) / "data"
    fit_out = Path(out) / "fit"
    scenario_out = Path(out) / "scenarios"

    total += len(plot_processed_data_directory(data, data_out))

    if Path(fit_dir).exists():
        total += len(
            plot_fit_diagnostics(
                fit_dir=fit_dir,
                data_dir=data,
                model_path=model,
                config_path=config,
                out_dir=fit_out,
            )
        )

    if Path(scenario_dir).exists():
        total += len(
            plot_scenario_diagnostics(
                scenario_dir=scenario_dir,
                out_dir=scenario_out,
            )
        )

    print(f"Wrote {total} plot(s) under {out}")


if __name__ == "__main__":
    plot_app()