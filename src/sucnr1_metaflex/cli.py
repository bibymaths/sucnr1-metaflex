"""Command line interface for the SUCNR1 metabolic model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich import print

from .data.inventory import build_inventory
from .data.tidy import ingest_all
from .model import (
    build_body_model,
    build_liver_model,
    build_combined_model,
    write_sbml_document,
    validate_sbml,
)
from .calibration.fit import run_fit
from .experiments.scenario_runner import run_scenarios
from .simulation.forward import run_forward
from .simulation.roadrunner_engine import load_model
from .simulation.steady_state import compute_steady_state


data_app = typer.Typer(help="Data inventory and ingestion commands.")
build_app = typer.Typer(help="SBML model build commands.")
fit_app = typer.Typer(help="Model calibration commands.")
sim_app = typer.Typer(help="Simulation commands.")
scenario_app = typer.Typer(help="KO/inhibition scenario commands.")
report_app = typer.Typer(help="Report generation commands.")
dashboard_app = typer.Typer(help="Dashboard command.")


@data_app.command("inventory")
def data_inventory(
    zip_path: str = typer.Option(..., "--zip", help="Path to supplementary zip file."),
    out: str = typer.Option("results", "--out", help="Output directory."),
) -> None:
    build_inventory(zip_path, out)


@data_app.command("ingest")
def data_ingest(
    zip_path: str = typer.Option(..., "--zip", help="Path to supplementary zip file."),
    out: str = typer.Option("results/processed", "--out", help="Output directory."),
    config: str = typer.Option("configs/data_sources.yaml", "--config", help="Data-source config YAML."),
) -> None:
    ingest_all(zip_path, config, out)


@build_app.command("body")
def build_body(
    config: str = typer.Option("configs/model_body.yaml", "--config"),
    out: str = typer.Option("results/models/body.xml", "--out"),
) -> None:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    doc = build_body_model(config)
    write_sbml_document(doc, out_path)
    validate_sbml(str(out_path))


@build_app.command("liver")
def build_liver(
    config: str = typer.Option("configs/model_liver.yaml", "--config"),
    out: str = typer.Option("results/models/liver.xml", "--out"),
) -> None:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    doc = build_liver_model(config)
    write_sbml_document(doc, out_path)
    validate_sbml(str(out_path))


@build_app.command("combined")
def build_combined(
    config: str = typer.Option("configs/model_combined.yaml", "--config"),
    body_config: str = typer.Option("configs/model_body.yaml", "--body-config"),
    liver_config: str = typer.Option("configs/model_liver.yaml", "--liver-config"),
    out: str = typer.Option("results/models/body_liver.xml", "--out"),
) -> None:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    doc = build_combined_model(body_config, liver_config, config)
    write_sbml_document(doc, out_path)
    validate_sbml(str(out_path))


@fit_app.command("run")
def fit_run(
    data: str = typer.Option("results/processed", "--data"),
    config: str = typer.Option("configs/fit.yaml", "--config"),
    out: str = typer.Option("results/runs/run1/fit", "--out"),
    model: str = typer.Option("results/models/body.xml", "--model"),
    n_starts: int = typer.Option(10, "--n-starts"),
) -> None:
    run_fit(data, model, config, out, n_starts)


@sim_app.command("steady-state")
def sim_steady_state(
    model: str = typer.Option(..., "--model"),
    params: Optional[str] = typer.Option(None, "--params"),
    out: str = typer.Option("results/simulations/steady_state.json", "--out"),
) -> None:
    rr = load_model(model)

    if params is not None:
        with open(params, "r", encoding="utf-8") as handle:
            param_dict = json.load(handle)
        for key, value in param_dict.items():
            try:
                rr[key] = float(value)
            except Exception:
                pass

    steady = compute_steady_state(rr)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(steady, handle, indent=2)

    print(f"Steady state written to {out_path}")


@sim_app.command("forward")
def sim_forward(
    model: str = typer.Option(..., "--model"),
    params: Optional[str] = typer.Option(None, "--params"),
    protocol: str = typer.Option("fasting", "--protocol"),
    fine_grid: bool = typer.Option(False, "--fine-grid"),
    out: str = typer.Option("results/simulations/forward.csv", "--out"),
    start: float = typer.Option(0.0, "--start"),
    end: Optional[float] = typer.Option(None, "--end"),
    step: Optional[float] = typer.Option(None, "--step"),
) -> None:
    if params is not None:
        with open(params, "r", encoding="utf-8") as handle:
            param_dict = json.load(handle)
    else:
        param_dict = {}

    if end is None:
        end = 24.0 if protocol == "fasting" else 120.0

    if step is None:
        if fine_grid:
            step = 0.05 if protocol == "fasting" else 0.25
        else:
            step = 1.0

    num_points = int(round((end - start) / step)) + 1

    df = run_forward(
        model_path=model,
        params=param_dict,
        start=start,
        end=end,
        num_points=num_points,
        selections=None,
    )

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Forward simulation written to {out_path}")


@scenario_app.command("run")
def scenarios_run(
    run: Optional[str] = typer.Option(None, "--run", help="Run directory."),
    model: Optional[str] = typer.Option(None, "--model", help="Explicit SBML model path."),
    params: Optional[str] = typer.Option(None, "--params", help="Explicit parameter JSON path."),
    config: str = typer.Option("configs/scenarios.yaml", "--config"),
    out: Optional[str] = typer.Option(None, "--out"),
) -> None:
    if run is not None:
        run_path = Path(run)
        model_path = model or str(run_path / "models" / "body_liver.xml")
        params_path = params or str(run_path / "fit" / "best_parameters.json")
        out_dir = out or str(run_path / "scenarios")
    else:
        if model is None or params is None:
            raise typer.BadParameter("Either --run or both --model and --params must be provided.")
        model_path = model
        params_path = params
        out_dir = out or "results/scenarios"

    run_scenarios(model_path, params_path, config, out_dir)


@report_app.command("build")
def report_build(
    run: str = typer.Option(..., "--run"),
) -> None:
    run_path = Path(run)
    print(f"Report for run at {run_path}:")
    for path in sorted(run_path.rglob("*")):
        if path.is_file():
            print(f"  - {path}")


@dashboard_app.callback(invoke_without_command=True)
def dashboard_main(
    run: str = typer.Option("results/runs/run1", "--run"),
) -> None:
    print(f"Dashboard requested for run at {run}")