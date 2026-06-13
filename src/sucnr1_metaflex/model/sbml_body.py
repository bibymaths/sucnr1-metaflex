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

    The model uses concentration-like species with hasOnlySubstanceUnits=False.
    Therefore:
    - first-order rate constants are per_hour;
    - zero-order source rates are millimole_per_litre_per_hour;
    - reference concentrations and proxy observables are millimole_per_litre;
    - fitted scale factors are dimensionless.
    """
    name = str(pid).lower()

    concentration_exact = {
        "k_sucnr1",
        "km_sucnr1",
        "k_m_sucnr1",
        "kd_sucnr1",
        "k_d_sucnr1",
        "ec50_sucnr1",
        "ic50_sucnr1",
        "i_eff_ref",
        "mito_target",
        "mito_capacity_ref",
        "g_mgdl",
        "ocr_proxy",
        "ecar_proxy",
        "mito_ocr_proxy",
    }

    if name in concentration_exact:
        return "millimole_per_litre"

    if name in {"k_hgp_base", "v_hgp_pyr", "v_hgp_aa", "k_aa_release_fasting", "k_succ_appearance"}:
        return "millimole_per_litre_per_hour"

    if name in {"k_pyr_hgp", "k_aa_hgp"}:
        return "millimole_per_litre"

    dimensionless_exact = {
        "glucose_mgdl_scale",
        "ocr_scale",
        "ecar_scale",
        "mito_ocr_scale",
        "genotype_sucnr1",
        "sucnr1_activity",
        "mito_sucnr1_baseline",
        "mito_sucnr1_gain",
        "ligand_factor",
        "antagonist_factor",
    }

    if name in dimensionless_exact:
        return "dimensionless"

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
        "gain",
    )

    if any(tok in name for tok in dimensionless_tokens):
        return "dimensionless"

    # Zero-order source/appearance/input rates.
    # Used in formulas such as: k_hgp_base * plasma
    # Units: concentration / time.
    flux_exact = {
        "k_hgp_base",
        "V_hgp_pyr",
        "V_hgp_aa",
        "k_aa_release_fasting",
        "k_succ_appearance",
        "k_gln_anaplerosis",
    }

    if name in flux_exact:
        return "millimole_per_litre_per_hour"

    flux_tokens = (
        "appearance",
        "release",
        "production",
        "secretion",
        "input",
        "source",
        "anaplerosis",
    )

    if name.startswith("k_") and any(tok in name for tok in flux_tokens):
        return "millimole_per_litre_per_hour"

    # First-order rates multiplying a concentration-like state.
    # Used in formulas such as: k * species * compartment.
    first_order_tokens = (
        "clear",
        "clearance",
        "decay",
        "uptake",
        "export",
        "import",
        "transport",
        "glycogenolysis",
        "glycogen_synthesis",
        "synthesis",
        "glycolysis",
        "ketogenesis",
        "tca",
        "gng",
        "adapt",
        "turnover",
        "degradation",
    )

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
    """Create minimal dynamic body reactions.

    This replaces the previous sink-only system with production,
    clearance, and simple conversion reactions.
    """

    def add_reaction(
            rid: str,
            reactants: list[str],
            products: list[str],
            formula: str,
            modifiers: list[str] | None = None,
    ) -> None:
        if model.getReaction(rid) is not None:
            return

        modifiers = modifiers or []

        reaction = model.createReaction()
        reaction.setId(rid)
        reaction.setReversible(False)
        reaction.setFast(False)

        for sid in reactants:
            sr = reaction.createReactant()
            sr.setSpecies(sid)
            sr.setStoichiometry(1.0)
            sr.setConstant(True)

        for sid in products:
            sr = reaction.createProduct()
            sr.setSpecies(sid)
            sr.setStoichiometry(1.0)
            sr.setConstant(True)

        already_declared = set(reactants) | set(products)

        for sid in modifiers:
            if sid in already_declared:
                continue
            msr = reaction.createModifier()
            msr.setSpecies(sid)

        kl = reaction.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula(formula))

    plasma = "plasma"

    # Hepatic glucose production / appearance into plasma.
    add_reaction(
        "source_G_plasma_hgp",
        [],
        ["G_plasma"],
        (
            f"(k_hgp_base "
            f"+ V_hgp_pyr * ptt_hgp_factor * Pyr_plasma / (K_pyr_hgp + Pyr_plasma) "
            f"+ V_hgp_aa * AA_plasma / (K_aa_hgp + AA_plasma)) * {plasma}"
        ),
        modifiers=["Pyr_plasma", "AA_plasma"],
    )

    add_reaction(
        "absorb_G_abs_to_G_plasma",
        ["G_abs"],
        ["G_plasma"],
        f"k_g_abs * G_abs * {plasma}",
    )

    add_reaction(
        "absorb_Pyr_abs_to_Pyr_plasma",
        ["Pyr_abs"],
        ["Pyr_plasma"],
        f"k_pyr_abs * Pyr_abs * {plasma}",
    )

    # Glucose clearance, insulin-enhanced.
    add_reaction(
        "clear_G_plasma",
        ["G_plasma"],
        [],
        f"(k_clear_base + k_clear_ins * (I_eff / I_eff_ref)) * G_plasma * {plasma}",
        modifiers=["I_eff"],
    )

    # Insulin action decay.
    add_reaction(
        "decay_I_eff",
        ["I_eff"],
        [],
        f"k_I_decay * I_eff * {plasma}",
    )

    # Basal pyruvate appearance and clearance.
    add_reaction(
        "source_Pyr_plasma",
        [],
        ["Pyr_plasma"],
        f"k_pyr_uptake * G_plasma * {plasma}",
        modifiers=["G_plasma"],
    )

    add_reaction(
        "clear_Pyr_plasma",
        ["Pyr_plasma"],
        [],
        f"k_pyr_clear * Pyr_plasma * {plasma}",
    )

    # Fasting amino-acid appearance and hepatic uptake.
    add_reaction(
        "source_AA_plasma",
        [],
        ["AA_plasma"],
        f"k_AA_release_fasting * {plasma}",
    )

    add_reaction(
        "clear_AA_plasma",
        ["AA_plasma"],
        [],
        f"k_AA_liver_uptake * AA_plasma * {plasma}",
    )

    # Ketogenesis from amino-acid substrate pool and ketone clearance.
    add_reaction(
        "ketogenesis_from_AA",
        ["AA_plasma"],
        ["Ketone_plasma"],
        f"k_ketogenesis * AA_plasma * {plasma}",
    )

    add_reaction(
        "clear_Ketone_plasma",
        ["Ketone_plasma"],
        [],
        f"k_ketone_clearance * Ketone_plasma * {plasma}",
    )

    # Succinate appearance and clearance.
    add_reaction(
        "source_Succ_plasma",
        [],
        ["Succ_plasma"],
        f"k_succ_appearance * {plasma}",
    )

    add_reaction(
        "clear_Succ_plasma",
        ["Succ_plasma"],
        [],
        f"k_succ_clear * Succ_plasma * {plasma}",
    )

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
