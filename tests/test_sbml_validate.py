import pytest
from pathlib import Path

from sucnr1_metaflex.model import build_body_model, build_liver_model, write_sbml_document, validate_sbml


def test_validate_models(tmp_path):
    body_doc = build_body_model("configs/model_body.yaml")
    liver_doc = build_liver_model("configs/model_liver.yaml")
    body_path = tmp_path / "body.xml"
    liver_path = tmp_path / "liver.xml"
    write_sbml_document(body_doc, body_path)
    write_sbml_document(liver_doc, liver_path)
    assert validate_sbml(str(body_path))
    assert validate_sbml(str(liver_path))