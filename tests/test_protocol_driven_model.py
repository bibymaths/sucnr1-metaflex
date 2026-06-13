from pathlib import Path
import numpy as np
import pandas as pd

from sucnr1_metaflex.model.sbml_body import build_body_model
from sucnr1_metaflex.model.sbml_liver import build_liver_model
from sucnr1_metaflex.calibration.protocols import evaluate_shape


def _param(doc, name):
    return doc.getModel().getParameter(name).getValue()


def test_rebuild_uses_corrected_defaults_and_body_inputs():
    body = build_body_model("configs/model_body.yaml").getModel()
    liver = build_liver_model("configs/model_liver.yaml").getModel()
    assert body.getSpecies("G_abs") is not None
    assert body.getSpecies("Pyr_abs") is not None
    assert _param(build_body_model("configs/model_body.yaml"), "k_AA_release_fasting") > 0
    assert _param(build_body_model("configs/model_body.yaml"), "k_succ_appearance") > 0
    assert body.getParameter("k_hgp_base").getValue() != 0.01
    assert body.getParameter("k_clear_base").getValue() != 0.01
    assert liver.getParameter("k_mito_adapt").getValue() != 0.01


def test_seahorse_shape_peak_drop_without_ode_state():
    t = np.array([0.0, 1.0, 1.5])
    y = evaluate_shape("seahorse_ocr", t, {"ocr_peak_amp": 1.0, "ocr_drop_amp": 0.5})
    assert y[1] > y[0]
    assert y[2] < y[1]
    assert np.all(y > 0)


def test_fit_has_no_forbidden_optimizers():
    text = Path("src/sucnr1_metaflex/calibration/fit.py").read_text()
    forbidden = ["least" + "_squares", "differential" + "_evolution", "dual" + "_annealing", "py" + "moo"]
    for token in forbidden:
        assert token not in text
