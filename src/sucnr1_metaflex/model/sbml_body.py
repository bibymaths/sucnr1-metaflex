"""SBML body model builder.

This module constructs a simple SBML representation of the plasma
compartment used in the SUCNR1 model.  It reads a YAML
configuration file specifying compartments, species and parameters
and then assembles a minimal kinetic scheme.  For the purposes of
this example we implement a few first–order clearance reactions
so that the resulting model can be simulated with libRoadRunner.
"""

from __future__ import annotations

import pathlib
from typing import Dict, Any

import yaml
from loguru import logger

try:
    import libsbml  # type: ignore
except ImportError:
    libsbml = None  # type: ignore

def _add_unit_definition(
    model: "libsbml.Model",
    unit_id: str,
    units: list[dict[str, float | int]],
) -> None:
    """Add a UnitDefinition if it does not already exist."""
    if model.getUnitDefinition(unit_id) is not None:
        return

    ud = model.createUnitDefinition()
    ud.setId(unit_id)

    for spec in units:
        unit = ud.createUnit()
        unit.setKind(int(spec["kind"]))
        unit.setExponent(float(spec.get("exponent", 1.0)))
        unit.setScale(int(spec.get("scale", 0)))
        unit.setMultiplier(float(spec.get("multiplier", 1.0)))

def _is_numeric_literal(expr: Any) -> bool:
    try:
        float(str(expr))
        return True
    except Exception:
        return False

def _create_common_units(model: "libsbml.Model") -> None:
    """Create custom SBML units used by the model."""
    _add_unit_definition(
        model,
        "millimole",
        [
            {
                "kind": libsbml.UNIT_KIND_MOLE,
                "scale": -3,
                "multiplier": 1.0,
                "exponent": 1.0,
            }
        ],
    )

    _add_unit_definition(
        model,
        "hour",
        [
            {
                "kind": libsbml.UNIT_KIND_SECOND,
                "scale": 0,
                "multiplier": 3600.0,
                "exponent": 1.0,
            }
        ],
    )

    _add_unit_definition(
        model,
        "per_hour",
        [
            {
                "kind": libsbml.UNIT_KIND_SECOND,
                "scale": 0,
                "multiplier": 3600.0,
                "exponent": -1.0,
            }
        ],
    )

    _add_unit_definition(
        model,
        "millimole_per_litre",
        [
            {
                "kind": libsbml.UNIT_KIND_MOLE,
                "scale": -3,
                "multiplier": 1.0,
                "exponent": 1.0,
            },
            {
                "kind": libsbml.UNIT_KIND_LITRE,
                "scale": 0,
                "multiplier": 1.0,
                "exponent": -1.0,
            },
        ],
    )

    _add_unit_definition(
        model,
        "millimole_per_litre_per_hour",
        [
            {
                "kind": libsbml.UNIT_KIND_MOLE,
                "scale": -3,
                "multiplier": 1.0,
                "exponent": 1.0,
            },
            {
                "kind": libsbml.UNIT_KIND_LITRE,
                "scale": 0,
                "multiplier": 1.0,
                "exponent": -1.0,
            },
            {
                "kind": libsbml.UNIT_KIND_SECOND,
                "scale": 0,
                "multiplier": 3600.0,
                "exponent": -1.0,
            },
        ],
    )

def _infer_parameter_units(pid: str) -> str:
    """Infer SBML parameter units from parameter names.

    Important distinction:
    - K_sucnr1, Km_sucnr1, EC50_sucnr1 are concentration constants.
    - k_mito_adapt is a first-order adaptation/turnover rate, not a Km.
    """
    name = str(pid).lower()

    # Explicit concentration constants.
    concentration_exact = {
        "k_sucnr1",
        "km_sucnr1",
        "k_m_sucnr1",
        "kd_sucnr1",
        "k_d_sucnr1",
        "ec50_sucnr1",
        "ic50_sucnr1",
    }

    if name in concentration_exact:
        return "millimole_per_litre"

    # Do not let k_mito_* be mistaken for K_m.
    if name.startswith("k_mito"):
        return "per_hour"

    # Conservative concentration-constant patterns.
    # These catch Km/Kd/EC50/IC50-style names without matching k_mito.
    concentration_prefixes = (
        "km_",
        "kd_",
        "ec50_",
        "ic50_",
        "k_m_",
        "k_d_",
    )

    concentration_infixes = (
        "_km_",
        "_kd_",
        "_ec50_",
        "_ic50_",
        "_k_m_",
        "_k_d_",
    )

    if name.startswith(concentration_prefixes):
        return "millimole_per_litre"

    if any(tok in name for tok in concentration_infixes):
        return "millimole_per_litre"

    dimensionless_tokens = (
        "activity",
        "scale",
        "factor",
        "ratio",
        "fraction",
        "frac",
        "alpha",
        "beta",
        "gamma",
        "hill",
    )

    flux_tokens = (
        "hgp",
        "appearance",
        "release",
        "production",
        "secretion",
        "ketogenesis",
        "input",
        "source",
    )

    first_order_tokens = (
        "clear",
        "clearance",
        "decay",
        "uptake",
        "export",
        "import",
        "transport",
        "glycogenolysis",
        "tca",
        "adapt",
        "turnover",
        "degradation",
    )

    if any(tok in name for tok in dimensionless_tokens):
        return "dimensionless"

    if name.startswith("k_") and any(tok in name for tok in flux_tokens):
        return "millimole_per_litre_per_hour"

    if name.startswith("k_") and any(tok in name for tok in first_order_tokens):
        return "per_hour"

    if name.startswith("k_"):
        return "per_hour"

    return "dimensionless"

def _normalise_species_substance_units(unit_id: str | None) -> str | None:
    """Map YAML concentration/dimensionless labels to SBML substance units."""
    if unit_id is None:
        return "millimole"

    unit_id = str(unit_id)

    concentration_units = {
        "millimole_per_litre",
        "mmol_per_litre",
        "mmol/L",
        "mM",
    }

    dimensionless_like = {
        "dimensionless",
        "1",
        "unitless",
    }

    if unit_id in concentration_units:
        return "millimole"

    if unit_id in dimensionless_like:
        return "millimole"

    return unit_id

def _create_compartments(model: libsbml.Model, compartments: Dict[str, Dict[str, Any]]) -> None:
    """Create compartments in the SBML model.

    Args:
        model: The libSBML model object.
        compartments: Mapping of compartment identifiers to size and units.
    """
    for cid, attrs in compartments.items():
        comp = model.createCompartment()
        comp.setId(cid)
        comp.setConstant(True)
        size = float(attrs.get("size", 1.0))
        comp.setSize(size)
        units = attrs.get("units")
        if units:
            comp.setUnits(units)


def _create_species(model: libsbml.Model, species: Dict[str, Dict[str, Any]]) -> None:
    """Create species in the SBML model."""
    for sid, attrs in species.items():
        sp = model.createSpecies()
        sp.setId(sid)

        compartment = attrs.get("compartment")
        if compartment is None:
            raise ValueError(f"Species {sid} missing compartment")

        sp.setCompartment(compartment)

        init = float(attrs.get("initial_concentration", 0.0))
        sp.setInitialConcentration(init)

        sp.setBoundaryCondition(False)
        sp.setHasOnlySubstanceUnits(False)
        sp.setConstant(False)

        units = _normalise_species_substance_units(attrs.get("units"))
        if units:
            sp.setSubstanceUnits(units)


def _create_parameters(model: libsbml.Model, parameters: Dict[str, Any]) -> None:
    """Create global parameters in the SBML model."""
    for pid, value in parameters.items():
        param = model.createParameter()
        param.setId(str(pid))
        param.setValue(float(value))
        param.setConstant(False)
        param.setUnits(_infer_parameter_units(str(pid)))


def _create_reactions(
    model: libsbml.Model,
    species: Dict[str, Dict[str, Any]],
    parameters: Dict[str, Any],
) -> None:
    """Define first-order clearance reactions.

    SBML KineticLaw expressions must have units of substance/time.
    Since state variables are represented as concentrations, the
    concentration-based first-order flux k * S must be multiplied by
    the compartment volume: k * S * compartment.
    """
    clearance_map = {
        "G_plasma": "k_clear_base",
        "I_eff": "k_I_decay",
        "Pyr_plasma": "k_pyr_clear",
        "AA_plasma": "k_AA_liver_uptake",
        "Ketone_plasma": "k_ketone_clearance",
        "Succ_plasma": "k_succ_clear",
    }

    for sid, attrs in species.items():
        param_id = clearance_map.get(sid)
        if not param_id:
            continue

        compartment_id = attrs.get("compartment")
        if compartment_id is None:
            raise ValueError(f"Species {sid} missing compartment")

        reaction = model.createReaction()
        reaction.setId(f"decay_{sid}")
        reaction.setReversible(False)
        reaction.setFast(False)

        reactant = reaction.createReactant()
        reactant.setSpecies(sid)
        reactant.setStoichiometry(1.0)
        reactant.setConstant(True)

        kl = reaction.createKineticLaw()
        math_ast = libsbml.parseL3Formula(f"{param_id} * {sid} * {compartment_id}")
        kl.setMath(math_ast)


def _create_assignment_rules(model: libsbml.Model, assignment_rules: Dict[str, Any]) -> None:
    """Create assignment rules for parameters in the model.

    Assignment-rule variables must already exist in the SBML model.
    If a rule target is not present, create it as a non-constant
    global parameter before attaching the rule.
    """
    for pid, expr in assignment_rules.items():
        pid = str(pid)

        # SBML requires the assignment-rule target to already exist.
        if (
            model.getParameter(pid) is None
            and model.getSpecies(pid) is None
            and model.getCompartment(pid) is None
        ):
            param = model.createParameter()
            param.setId(pid)
            param.setValue(0.0)
            param.setConstant(False)
            param.setUnits(_infer_parameter_units(pid))

        # If the target is an existing parameter, it must be non-constant
        # because assignment rules define its value dynamically.
        existing_param = model.getParameter(pid)
        if _is_numeric_literal(expr):
            if existing_param is not None:
                existing_param.setValue(float(str(expr)))
                existing_param.setConstant(False)
                if not existing_param.isSetUnits():
                    existing_param.setUnits(_infer_parameter_units(pid))
            continue

        rule = model.createAssignmentRule()
        rule.setVariable(pid)

        try:
            ast = libsbml.parseL3Formula(str(expr))
        except Exception as exc:
            logger.warning(f"Could not parse assignment rule for {pid}: {expr} ({exc})")
            continue

        rule.setMath(ast)

def build_body_model(config_path: str) -> libsbml.SBMLDocument:
    """Build an SBML body model from a YAML configuration file.

    Args:
        config_path: Path to the model_body.yaml file.

    Returns:
        An SBMLDocument containing the assembled model.
    """
    # Resolve configuration path relative to repository if necessary
    import pathlib
    path = pathlib.Path(config_path)
    if not path.exists():
        # Attempt to find the config in the repository's configs directory
        repo_root = pathlib.Path(__file__).resolve().parents[3]
        alt = repo_root / "configs" / path.name
        if alt.exists():
            path = alt
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if libsbml is None:
        raise RuntimeError("libsbml is not available; cannot build SBML models.")
    level = int(config.get("level", 3))
    version = int(config.get("version", 2))
    doc = libsbml.SBMLDocument(level, version)
    model = doc.createModel()
    model.setId(config.get("id", "body"))

    _create_common_units(model)
    model.setTimeUnits("hour")
    model.setExtentUnits("millimole")
    model.setSubstanceUnits("millimole")
    # Create compartments
    compartments = config.get("compartments", {})
    _create_compartments(model, compartments)
    # Create species
    species = config.get("species", {})
    _create_species(model, species)
    # Create parameters
    parameters = config.get("parameters", {})
    _create_parameters(model, parameters)
    # Create assignment rules
    assignment_rules = config.get("assignment_rules", {})
    _create_assignment_rules(model, assignment_rules)
    # Create simple reactions
    _create_reactions(model, species, parameters)
    return doc


def write_sbml_document(doc: "libsbml.SBMLDocument", out_path: str | pathlib.Path) -> None:
    """Write an SBML document to an XML file.

    Args:
        doc: The SBMLDocument to write.
        out_path: Path to the output XML file.
    """
    out_path = pathlib.Path(out_path)
    # Validate before writing
    if libsbml is None:
        raise RuntimeError("libsbml is not available; cannot write SBML models.")
    errors = doc.checkConsistency()
    if errors > 0:
        for i in range(errors):
            e = doc.getError(i)
            logger.warning(f"SBML error ({e.getSeverity()}): {e.getMessage()}")
    libsbml.writeSBMLToFile(doc, str(out_path))
    logger.info(f"Wrote SBML model to {out_path}")
