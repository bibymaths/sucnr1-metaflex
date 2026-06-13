"""Combined SBML model builder."""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List

import yaml

try:
    import libsbml  # type: ignore
except ImportError:
    libsbml = None  # type: ignore

from .sbml_body import (
    _create_common_units,
    _infer_parameter_units,
)
from .sbml_body import build_body_model
from .sbml_liver import build_liver_model


def _resolve_config_path(config_path: str) -> pathlib.Path:
    path = pathlib.Path(config_path)
    if path.exists():
        return path

    repo_root = pathlib.Path(__file__).resolve().parents[3]
    alt = repo_root / "configs" / path.name
    if alt.exists():
        return alt

    return path


def _copy_math(math_obj: Any) -> Any:
    if hasattr(math_obj, "deepCopy"):
        return math_obj.deepCopy()
    return math_obj


def _copy_unit_definitions(source_model: libsbml.Model, dest_model: libsbml.Model) -> None:
    for i in range(source_model.getNumUnitDefinitions()):
        unit_def = source_model.getUnitDefinition(i)
        unit_id = unit_def.getId()

        if dest_model.getUnitDefinition(unit_id) is not None:
            continue

        copied = unit_def.clone() if hasattr(unit_def, "clone") else unit_def.deepCopy()
        dest_model.addUnitDefinition(copied)

def _copy_components(source_model: libsbml.Model, dest_model: libsbml.Model) -> None:
    """Copy compartments, species, parameters, assignment rules and reactions.

    Important:
    Modifier species references must be copied. Otherwise reactions whose
    kinetic laws use non-consumed species, for example I_eff, G_plasma or
    Mito_capacity, become invalid in the combined model.
    """
    _copy_unit_definitions(source_model, dest_model)

    # Compartments
    for i in range(source_model.getNumCompartments()):
        comp = source_model.getCompartment(i)
        comp_id = comp.getId()

        if dest_model.getCompartment(comp_id) is not None:
            continue

        new_comp = dest_model.createCompartment()
        new_comp.setId(comp_id)

        if comp.isSetName():
            new_comp.setName(comp.getName())

        if comp.isSetConstant():
            new_comp.setConstant(comp.getConstant())
        else:
            new_comp.setConstant(True)

        if comp.isSetSize():
            new_comp.setSize(comp.getSize())

        if comp.isSetUnits():
            new_comp.setUnits(comp.getUnits())

        if comp.isSetSpatialDimensions():
            new_comp.setSpatialDimensions(comp.getSpatialDimensions())

    # Species
    for i in range(source_model.getNumSpecies()):
        sp = source_model.getSpecies(i)
        sid = sp.getId()

        if dest_model.getSpecies(sid) is not None:
            continue

        new_sp = dest_model.createSpecies()
        new_sp.setId(sid)

        if sp.isSetName():
            new_sp.setName(sp.getName())

        new_sp.setCompartment(sp.getCompartment())

        if sp.isSetInitialConcentration():
            new_sp.setInitialConcentration(sp.getInitialConcentration())
        elif sp.isSetInitialAmount():
            new_sp.setInitialAmount(sp.getInitialAmount())

        new_sp.setBoundaryCondition(sp.getBoundaryCondition())
        new_sp.setHasOnlySubstanceUnits(sp.getHasOnlySubstanceUnits())
        new_sp.setConstant(sp.getConstant())

        if sp.isSetSubstanceUnits():
            new_sp.setSubstanceUnits(sp.getSubstanceUnits())

        if sp.isSetConversionFactor():
            new_sp.setConversionFactor(sp.getConversionFactor())

    # Parameters
    for i in range(source_model.getNumParameters()):
        par = source_model.getParameter(i)
        pid = par.getId()

        if dest_model.getParameter(pid) is not None:
            continue

        new_par = dest_model.createParameter()
        new_par.setId(pid)

        if par.isSetName():
            new_par.setName(par.getName())

        if par.isSetValue():
            new_par.setValue(par.getValue())

        if par.isSetConstant():
            new_par.setConstant(par.getConstant())
        else:
            new_par.setConstant(False)

        if par.isSetUnits():
            new_par.setUnits(par.getUnits())
        else:
            new_par.setUnits(_infer_parameter_units(pid))

    # Assignment rules
    for i in range(source_model.getNumRules()):
        rule = source_model.getRule(i)

        if rule.getTypeCode() != libsbml.SBML_ASSIGNMENT_RULE:
            continue

        variable = rule.getVariable()

        if dest_model.getRule(variable) is not None:
            continue

        new_rule = dest_model.createAssignmentRule()
        new_rule.setVariable(variable)
        new_rule.setMath(_copy_math(rule.getMath()))

    # Reactions
    for i in range(source_model.getNumReactions()):
        reaction = source_model.getReaction(i)
        rid = reaction.getId()

        if dest_model.getReaction(rid) is not None:
            continue

        new_reaction = dest_model.createReaction()
        new_reaction.setId(rid)

        if reaction.isSetName():
            new_reaction.setName(reaction.getName())

        new_reaction.setReversible(reaction.getReversible())

        if reaction.isSetFast():
            new_reaction.setFast(reaction.getFast())
        else:
            new_reaction.setFast(False)

        # Reactants
        for j in range(reaction.getNumReactants()):
            reactant = reaction.getReactant(j)
            new_reactant = new_reaction.createReactant()

            if reactant.isSetId():
                new_reactant.setId(reactant.getId())

            if reactant.isSetName():
                new_reactant.setName(reactant.getName())

            new_reactant.setSpecies(reactant.getSpecies())

            if reactant.isSetStoichiometry():
                new_reactant.setStoichiometry(reactant.getStoichiometry())
            else:
                new_reactant.setStoichiometry(1.0)

            if reactant.isSetConstant():
                new_reactant.setConstant(reactant.getConstant())
            else:
                new_reactant.setConstant(True)

        # Products
        for j in range(reaction.getNumProducts()):
            product = reaction.getProduct(j)
            new_product = new_reaction.createProduct()

            if product.isSetId():
                new_product.setId(product.getId())

            if product.isSetName():
                new_product.setName(product.getName())

            new_product.setSpecies(product.getSpecies())

            if product.isSetStoichiometry():
                new_product.setStoichiometry(product.getStoichiometry())
            else:
                new_product.setStoichiometry(1.0)

            if product.isSetConstant():
                new_product.setConstant(product.getConstant())
            else:
                new_product.setConstant(True)

        # Modifiers: REQUIRED for kinetic-law-only species references.
        for j in range(reaction.getNumModifiers()):
            modifier = reaction.getModifier(j)
            new_modifier = new_reaction.createModifier()

            if modifier.isSetId():
                new_modifier.setId(modifier.getId())

            if modifier.isSetName():
                new_modifier.setName(modifier.getName())

            new_modifier.setSpecies(modifier.getSpecies())

        # Kinetic law
        if reaction.isSetKineticLaw():
            kl = reaction.getKineticLaw()
            new_kl = new_reaction.createKineticLaw()
            new_kl.setMath(_copy_math(kl.getMath()))

            # Copy local kinetic-law parameters if any exist.
            for j in range(kl.getNumLocalParameters()):
                lp = kl.getLocalParameter(j)
                new_lp = new_kl.createLocalParameter()
                new_lp.setId(lp.getId())

                if lp.isSetName():
                    new_lp.setName(lp.getName())

                if lp.isSetValue():
                    new_lp.setValue(lp.getValue())

                if lp.isSetUnits():
                    new_lp.setUnits(lp.getUnits())

def _ensure_parameter(model: libsbml.Model, pid: str, value: float = 0.1) -> libsbml.Parameter:
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


def _species_compartment(model: libsbml.Model, species_id: str) -> str:
    species = model.getSpecies(species_id)
    if species is None:
        raise ValueError(f"Coupling refers to unknown species: {species_id}")

    compartment = species.getCompartment()
    if not compartment:
        raise ValueError(f"Species {species_id} has no compartment")

    return compartment


def _create_coupling_reactions(
    model: libsbml.Model,
    couplings: List[Dict[str, str]],
) -> None:
    for cpl in couplings:
        source = cpl["source"]
        target = cpl["target"]
        param_id = cpl["parameter"]

        src_species = source.split("::")[-1]
        tgt_species = target.split("::")[-1]
        src_compartment = _species_compartment(model, src_species)

        _species_compartment(model, tgt_species)
        _ensure_parameter(model, param_id, 0.1)

        reaction_id = f"couple_{src_species}_to_{tgt_species}"

        if model.getReaction(reaction_id) is not None:
            continue

        reaction = model.createReaction()
        reaction.setId(reaction_id)
        reaction.setReversible(False)
        reaction.setFast(False)

        reactant = reaction.createReactant()
        reactant.setSpecies(src_species)
        reactant.setStoichiometry(1.0)
        reactant.setConstant(True)

        product = reaction.createProduct()
        product.setSpecies(tgt_species)
        product.setStoichiometry(1.0)
        product.setConstant(True)

        kl = reaction.createKineticLaw()
        formula = f"{param_id} * {src_species} * {src_compartment}"
        ast = libsbml.parseL3Formula(formula)
        kl.setMath(ast)


def build_combined_model(
    body_config: str,
    liver_config: str,
    combined_config: str,
) -> libsbml.SBMLDocument:
    """Build a combined body-liver SBML model."""
    if libsbml is None:
        raise RuntimeError("libsbml is not available; cannot build SBML models.")

    doc_body = build_body_model(body_config)
    doc_liver = build_liver_model(liver_config)

    body = doc_body.getModel()
    liver = doc_liver.getModel()

    cfg_path = _resolve_config_path(combined_config)

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    doc = libsbml.SBMLDocument(3, 2)
    model = doc.createModel()
    model.setId(cfg.get("id", "body_liver"))

    _create_common_units(model)
    model.setTimeUnits("hour")
    model.setExtentUnits("millimole")
    model.setSubstanceUnits("millimole")

    _copy_components(body, model)
    _copy_components(liver, model)

    couplings: List[Dict[str, str]] = cfg.get("couplings", [])
    _create_coupling_reactions(model, couplings)

    return doc