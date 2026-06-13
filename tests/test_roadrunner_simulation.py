import pytest
from pathlib import Path

from sucnr1_metaflex.model import build_body_model, write_sbml_document
from sucnr1_metaflex.simulation.roadrunner_engine import load_model, simulate_to_times


def test_roadrunner_forward(tmp_path):
    body_doc = build_body_model("configs/model_body.yaml")
    model_path = tmp_path / "body.xml"
    write_sbml_document(body_doc, model_path)
    rr = load_model(str(model_path))
    times = [0.0, 10.0, 20.0]
    df = simulate_to_times(rr, times, selections=["G_plasma"])
    # Expect number of rows equal to len(unique times)
    assert len(df) == len(set(times))
    assert "G_plasma" in df.columns