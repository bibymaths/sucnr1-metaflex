import pytest
import libsbml

from sucnr1_metaflex.model import build_body_model, build_liver_model


def test_build_body_model():
    doc = build_body_model("configs/model_body.yaml")
    assert isinstance(doc, libsbml.SBMLDocument)
    model = doc.getModel()
    # Expect at least one species
    assert model.getNumSpecies() >= 1


def test_build_liver_model():
    doc = build_liver_model("configs/model_liver.yaml")
    assert isinstance(doc, libsbml.SBMLDocument)
    model = doc.getModel()
    assert model.getNumSpecies() >= 1