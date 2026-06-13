"""SBML liver model builder."""

from __future__ import annotations

import pathlib
from typing import Any, Dict

import yaml
from loguru import logger

try:
    import libsbml  # type: ignore
except ImportError:
    libsbml = None  # type: ignore

from .sbml_body import (
    _create_common_units,
    _infer_parameter_units,
    _is_numeric_literal,
    _normalise_species_substance_units,
)


def _resolve_config_path(config_path: str) -> pathlib.Path:
    path = pathlib.Path(config_path)
    if path.exists():
        return path

    repo_root = pathlib.Path(__file__).resolve().parents[3]
    alt = repo_root / "configs" / path.name
    if alt.exists():
        return alt

    return path


def _create_compartments(model: libsbml.Model, compartments: Dict[str, Dict[str, Any]]) -> None:
    for cid, attrs in compartments.items():
        comp = model.createCompartment()
        comp.setId(str(cid))
        comp.setConstant(True)
        comp.setSize(float(attrs.get("size", 1.0)))

        units = attrs.get("units", "litre")
        if units:
            comp.setUnits(str(units))


def _create_species(model: libsbml.Model, species: Dict[str, Dict[str, Any]]) -> None:
    for sid, attrs in species.items():
        sid = str(sid)

        compartment = attrs.get("compartment")
        if compartment is None:
            raise ValueError(f"Species {sid} missing compartment")

        sp = model.createSpecies()
        sp.setId(sid)
        sp.setCompartment(str(compartment))
        sp.setInitialConcentration(float(attrs.get("initial_concentration", 0.0)))
        sp.setBoundaryCondition(False)
        sp.setHasOnlySubstanceUnits(False)
        sp.setConstant(False)

        units = _normalise_species_substance_units(attrs.get("units"))
        if units:
            sp.setSubstanceUnits(str(units))


def _create_parameters(model: libsbml.Model, parameters: Dict[str, Any]) -> None:
    for pid, value in parameters.items():
        pid = str(pid)

        par = model.createParameter()
        par.setId(pid)
        par.setValue(float(value))
        par.setConstant(False)
        par.setUnits(_infer_parameter_units(pid))


def _ensure_parameter(model: libsbml.Model, pid: str, value: float = 0.0) -> libsbml.Parameter:
    existing = model.getParameter(pid)
    if existing is not None:
        existing.setConstant(False)
        if not existing.isSetUnits():
            existing.setUnits(_infer_parameter_units(pid))
        return existing

    par = model.createParameter()
    par.setId(pid)
    par.setValue(float(value))
    par.setConstant(False)
    par.setUnits(_infer_parameter_units(pid))
    return par


def _create_assignment_rules(model: libsbml.Model, assignment_rules: Dict[str, Any]) -> None:
    for pid, expr in assignment_rules.items():
        pid = str(pid)

        if (
            model.getParameter(pid) is None
            and model.getSpecies(pid) is None
            and model.getCompartment(pid) is None
        ):
            target_param = _ensure_parameter(model, pid, 0.0)
        else:
            target_param = model.getParameter(pid)
            if target_param is not None:
                target_param.setConstant(False)
                if not target_param.isSetUnits():
                    target_param.setUnits(_infer_parameter_units(pid))

        if _is_numeric_literal(expr):
            if target_param is not None:
                target_param.setValue(float(str(expr)))
                target_param.setConstant(False)
                if not target_param.isSetUnits():
                    target_param.setUnits(_infer_parameter_units(pid))
            continue

        rule = model.createAssignmentRule()
        rule.setVariable(pid)

        try:
            ast = libsbml.parseL3Formula(str(expr))
        except Exception as exc:
            logger.warning(f"Could not parse assignment rule {pid}: {expr} ({exc})")
            continue

        rule.setMath(ast)


def _create_reactions(
    model: libsbml.Model,
    species: Dict[str, Dict[str, Any]],
    parameters: Dict[str, Any],
) -> None:
    """Create first-order sink reactions with SBML-valid amount/time units.

    Since species are represented with initialConcentration and
    hasOnlySubstanceUnits=False, a species symbol in a kinetic law has
    concentration units. Therefore k * species must be multiplied by
    the species compartment to produce amount/time.
    """
    clearance_map = {
        "G6P_liver": "k_glucose_uptake",
        "Glycogen_liver": "k_glycogenolysis",
        "Pyr_liver": "k_TCA_glucose",
        "Succ_mito": "k_succ_export",
        "Succ_extra": "k_succ_clear",
        "Mito_capacity": "k_mito_adapt",
    }

    for sid, attrs in species.items():
        sid = str(sid)
        param_id = clearance_map.get(sid)
        if not param_id:
            continue

        compartment_id = attrs.get("compartment")
        if compartment_id is None:
            raise ValueError(f"Species {sid} missing compartment")

        _ensure_parameter(model, param_id, float(parameters.get(param_id, 0.0)))

        reaction = model.createReaction()
        reaction.setId(f"decay_{sid}")
        reaction.setReversible(False)
        reaction.setFast(False)

        reactant = reaction.createReactant()
        reactant.setSpecies(sid)
        reactant.setStoichiometry(1.0)
        reactant.setConstant(True)

        kl = reaction.createKineticLaw()
        formula = f"{param_id} * {sid} * {compartment_id}"
        math_ast = libsbml.parseL3Formula(formula)
        kl.setMath(math_ast)


def build_liver_model(config_path: str) -> libsbml.SBMLDocument:
    """Build the liver SBML model from YAML configuration."""
    if libsbml is None:
        raise RuntimeError("libsbml is not available; cannot build SBML models.")

    path = _resolve_config_path(config_path)

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    level = int(config.get("level", 3))
    version = int(config.get("version", 2))

    doc = libsbml.SBMLDocument(level, version)
    model = doc.createModel()
    model.setId(config.get("id", "liver"))

    _create_common_units(model)
    model.setTimeUnits("hour")
    model.setExtentUnits("millimole")
    model.setSubstanceUnits("millimole")

    compartments = config.get("compartments", {})
    species = config.get("species", {})
    parameters = config.get("parameters", {})
    assignment_rules = config.get("assignment_rules", {})

    _create_compartments(model, compartments)
    _create_species(model, species)
    _create_parameters(model, parameters)
    _create_assignment_rules(model, assignment_rules)
    _create_reactions(model, species, parameters)

    return doc