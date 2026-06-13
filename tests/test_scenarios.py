import json
from pathlib import Path

from sucnr1_metaflex.model import build_body_model, build_liver_model, build_combined_model, write_sbml_document
from sucnr1_metaflex.experiments.scenario_runner import run_scenarios


def test_run_scenarios(tmp_path):
    # Build minimal combined model and parameter file
    body_doc = build_body_model("configs/model_body.yaml")
    liver_doc = build_liver_model("configs/model_liver.yaml")
    combined_doc = build_combined_model("configs/model_body.yaml", "configs/model_liver.yaml", "configs/model_combined.yaml")
    model_path = tmp_path / "body_liver.xml"
    write_sbml_document(combined_doc, model_path)
    # Baseline parameters (use defaults from configs for a couple of keys)
    baseline = {"k_clear_base": 0.01, "genotype_sucnr1": 1.0}
    params_path = tmp_path / "params.json"
    with params_path.open("w", encoding="utf-8") as f:
        json.dump(baseline, f)
    # Use provided scenarios config
    scenarios_path = "configs/scenarios.yaml"
    out_dir = tmp_path / "scens"
    results = run_scenarios(str(model_path), str(params_path), scenarios_path, out_dir)
    # Expect at least one scenario result file
    assert results
    for name, path in results.items():
        assert Path(path).exists()