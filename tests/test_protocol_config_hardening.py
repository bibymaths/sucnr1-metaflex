from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from sucnr1_metaflex.calibration.fit import run_fit
from sucnr1_metaflex.calibration.objective import compute_residuals
from sucnr1_metaflex.calibration.parameters import load_fit_config
from sucnr1_metaflex.calibration.protocols import (
    NEUTRAL_CONDITION_FACTORS,
    collect_parameter_references,
    load_protocol_config,
    resolve_condition_factors,
    resolve_protocol,
)
from sucnr1_metaflex.model import build_body_model, build_liver_model, write_sbml_document


REQUIRED_FACTORS = set(NEUTRAL_CONDITION_FACTORS)


def test_every_seahorse_condition_declares_complete_factor_triplet():
    protocols = load_protocol_config()["seahorse_protocols"]
    for assay, protocol in protocols.items():
        factors = protocol.get("condition_factors") or {}
        for condition, overrides in factors.items():
            assert REQUIRED_FACTORS <= set(overrides), (assay, condition, overrides)


def test_resolve_condition_factors_resets_neutral_values_before_overrides():
    params = {
        "genotype_sucnr1": 0.02,
        "ligand_factor": 7.0,
        "antagonist_factor": 0.03,
        "siRNA_sucnr1_factor": 0.25,
        "cESA_ligand_factor": 3.0,
        "antagonist_factor_protocol": 0.1,
        "antagonist_rescue_factor": 0.5,
    }
    protocols = load_protocol_config()

    for assay, condition in [
        ("OCR_WT_vs_global_KO", "WT"),
        ("ECAR_WT_vs_global_KO", "WT"),
        ("OCR_siRNA", "Scramble"),
        ("ECAR_siRNA", "Scramble"),
        ("OCR_antagonist_agonist", "Control"),
        ("ECAR_antagonist_agonist", "Control"),
    ]:
        factors = resolve_condition_factors(
            resolve_protocol(assay, protocols), condition, params
        )
        assert factors == NEUTRAL_CONDITION_FACTORS

    ko = resolve_condition_factors(
        resolve_protocol("OCR_WT_vs_global_KO", protocols), "SUCNR1KO", params
    )
    assert ko == {"genotype_sucnr1": 0.0, "ligand_factor": 1.0, "antagonist_factor": 1.0}

    sirna = resolve_condition_factors(
        resolve_protocol("ECAR_siRNA", protocols), "Sucnr1 siRNA", params
    )
    assert sirna == {"genotype_sucnr1": 0.25, "ligand_factor": 1.0, "antagonist_factor": 1.0}

    rescue = resolve_condition_factors(
        resolve_protocol("ECAR_antagonist_agonist", protocols), "NF-56-EJ40 + cESA", params
    )
    assert rescue == {"genotype_sucnr1": 1.0, "ligand_factor": 3.0, "antagonist_factor": 0.5}


def test_fit_combined_defines_all_protocol_parameter_references():
    protocols = load_protocol_config()
    combined = load_fit_config("configs/fit_combined.yaml")
    refs = collect_parameter_references(protocols)
    assert refs <= set(combined.parameters)


def test_run_fit_fails_early_for_missing_protocol_parameter(tmp_path):
    cfg = yaml.safe_load(Path("configs/fit_combined.yaml").read_text())
    cfg["parameters"].pop("gtt_glucose_bolus")
    config_path = tmp_path / "fit_combined_missing_protocol.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with pytest.raises(ValueError, match="missing protocol parameter.*gtt_glucose_bolus"):
        run_fit(
            data_dir="results/processed",
            body_model_path="missing-is-not-loaded-before-validation.xml",
            fit_config_path=str(config_path),
            out_dir=tmp_path / "fit",
            n_starts=1,
        )


def _changed_residuals(model_path, data, observables, names, base, changed):
    x0 = np.array([base[name] for name in names], dtype=float)
    x1 = np.array([changed.get(name, base[name]) for name in names], dtype=float)
    r0 = compute_residuals(x0, names, data, str(model_path), {}, observables)
    r1 = compute_residuals(x1, names, data, str(model_path), {}, observables)
    assert np.all(np.isfinite(r0))
    assert np.all(np.isfinite(r1))
    assert not np.allclose(r0, r1)


def test_compute_residuals_changes_for_seahorse_protocol_parameters(tmp_path):
    model_path = tmp_path / "liver.xml"
    write_sbml_document(build_liver_model("configs/model_liver.yaml"), model_path)

    base = {
        "siRNA_sucnr1_factor": -0.7,
        "antagonist_factor": -1.0,
        "antagonist_rescue_factor": -0.3,
        "cESA_ligand_factor": 0.3,
    }

    _changed_residuals(
        model_path,
        pd.DataFrame({"assay": ["OCR_siRNA"], "genotype": ["Sucnr1 siRNA"], "time": [1.0], "value": [1.0]}),
        {"OCR_siRNA": "OCR_proxy"},
        list(base),
        base,
        {"siRNA_sucnr1_factor": -1.5},
    )
    _changed_residuals(
        model_path,
        pd.DataFrame({"assay": ["OCR_antagonist_agonist"], "genotype": ["NF-56-EJ40"], "time": [1.0], "value": [1.0]}),
        {"OCR_antagonist_agonist": "OCR_proxy"},
        list(base),
        base,
        {"antagonist_factor": -2.0},
    )
    _changed_residuals(
        model_path,
        pd.DataFrame({"assay": ["OCR_antagonist_agonist"], "genotype": ["NF-56-EJ40 + cESA"], "time": [1.0], "value": [1.0]}),
        {"OCR_antagonist_agonist": "OCR_proxy"},
        list(base),
        base,
        {"cESA_ligand_factor": 0.8},
    )


def test_compute_residuals_changes_for_body_protocol_parameters(tmp_path):
    model_path = tmp_path / "body.xml"
    write_sbml_document(build_body_model("configs/model_body.yaml"), model_path)
    base = {
        "gtt_glucose_bolus": 0.2,
        "ptt_pyruvate_bolus": 0.2,
        "itt_insulin_pulse": 0.0,
    }
    _changed_residuals(
        model_path,
        pd.DataFrame({"assay": ["GTT"], "time": [0.25], "value": [90.0]}),
        {"GTT": "G_mgdl"},
        list(base),
        base,
        {"gtt_glucose_bolus": 1.0},
    )
    _changed_residuals(
        model_path,
        pd.DataFrame({"assay": ["PTT"], "time": [0.5], "value": [90.0]}),
        {"PTT": "G_mgdl"},
        list(base),
        base,
        {"ptt_pyruvate_bolus": 1.0},
    )
    _changed_residuals(
        model_path,
        pd.DataFrame({"assay": ["ITT"], "time": [0.25], "value": [90.0]}),
        {"ITT": "G_mgdl"},
        list(base),
        base,
        {"itt_insulin_pulse": 1.0},
    )
