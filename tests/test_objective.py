import pytest
import numpy as np
import pandas as pd
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
    res = compute_residuals(log_params, param_names, data, str(model_path), assay_weights, fit_cfg.observables)
    assert np.all(np.isfinite(res))

def test_python_and_numba_residual_paths_match(tmp_path):
    import numpy as np
    from sucnr1_metaflex.calibration.objective import prepare_residual_context
    from sucnr1_metaflex.calibration.numba_kernels import NUMBA_AVAILABLE

    data = pd.DataFrame({"assay": ["glucose", "glucose"], "time": [0.0, 1.0], "value": [5.0, 4.9]})
    model_path = tmp_path / "dummy.xml"
    model_path.write_text("<sbml />")
    param_names = ["k_clear_base"]
    x = np.array([-2.0])
    observables = {"glucose": "G_plasma"}

    py_ctx = prepare_residual_context(data, {}, observables, use_numba=False)
    nb_ctx = prepare_residual_context(data, {}, observables, use_numba=True)
    residuals_python = compute_residuals(x, param_names, data, str(model_path), {}, observables, context=py_ctx, use_numba=False)
    residuals_numba = compute_residuals(x, param_names, data, str(model_path), {}, observables, context=nb_ctx, use_numba=True)
    np.testing.assert_allclose(residuals_python, residuals_numba, rtol=1e-10, atol=1e-10)
    assert nb_ctx.use_numba is NUMBA_AVAILABLE
