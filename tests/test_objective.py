import pytest
import numpy as np
from pathlib import Path

from sucnr1_metaflex.model import build_body_model, write_sbml_document
from sucnr1_metaflex.calibration.objective import compute_residuals
from sucnr1_metaflex.calibration.parameters import load_fit_config
from sucnr1_metaflex.data.tidy import ingest_all


def test_objective_finite(tmp_path):
    # Build body model and save
    body_doc = build_body_model("configs/model_body.yaml")
    model_path = tmp_path / "body.xml"
    write_sbml_document(body_doc, model_path)
    # Ingest data
    zip_path = "Beltran2026_Supp.zip"
    config_path = "configs/data_sources.yaml"
    ingest_all(zip_path, config_path, tmp_path)
    import pandas as pd
    data = pd.read_csv(tmp_path / "dynamic_body.csv")
    # Load param definitions
    fit_cfg = load_fit_config("configs/fit.yaml")
    param_names = list(fit_cfg.parameters.keys())
    # use guesses as log10 values
    log_params = np.array([fit_cfg.parameters[p].guess for p in param_names])
    assay_weights = fit_cfg.weights
    res = compute_residuals(log_params, param_names, data, str(model_path), assay_weights)
    assert np.all(np.isfinite(res))