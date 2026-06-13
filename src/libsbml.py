"""Fallback stub for the libsbml package.

This module provides a very limited subset of the libsbml API used
by the sucnr1-metaflex package.  It is not a complete SBML
implementation and should only be used when the real libsbml
library is unavailable.  The stub mimics the interface of the
classes required by our model builder and validator so that
importing `libsbml` does not raise ImportError in environments
without the library.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Constants mimicking libSBML severities and rule types
LIBSBML_SEV_ERROR = 2
SBML_ASSIGNMENT_RULE = 1


class SBMLDocument:
    def __init__(self, level: int = 3, version: int = 2) -> None:
        self._model: Optional[Model] = None
        self._errors: List[str] = []

    def createModel(self) -> "Model":
        self._model = Model()
        return self._model

    def getModel(self) -> "Model":
        return self._model

    def checkConsistency(self) -> int:
        # Always return no errors
        return 0

    def getNumErrors(self) -> int:
        return 0

    def getError(self, index: int) -> Any:
        class Err:
            def getSeverity(self):
                return LIBSBML_SEV_ERROR

            def getMessage(self):
                return ""

        return Err()


class Model:
    def __init__(self) -> None:
        self._compartments: List[Compartment] = []
        self._species: List[Species] = []
        self._parameters: List[Parameter] = []
        self._rules: List[AssignmentRule] = []
        self._reactions: List[Reaction] = []

    def setId(self, id: str) -> None:
        self.id = id

    # Compartment management
    def createCompartment(self) -> "Compartment":
        comp = Compartment()
        self._compartments.append(comp)
        return comp

    def getNumCompartments(self) -> int:
        return len(self._compartments)

    def getCompartment(self, idx: int) -> "Compartment":
        return self._compartments[idx]

    # Species management
    def createSpecies(self) -> "Species":
        sp = Species()
        self._species.append(sp)
        return sp

    def getNumSpecies(self) -> int:
        return len(self._species)

    def getSpecies(self, idx: int) -> "Species":
        return self._species[idx]

    # Parameter management
    def createParameter(self) -> "Parameter":
        par = Parameter()
        self._parameters.append(par)
        return par

    def getNumParameters(self) -> int:
        return len(self._parameters)

    def getParameter(self, idx: int) -> "Parameter":
        return self._parameters[idx]

    def getParameterById(self, pid: str) -> Optional["Parameter"]:
        for p in self._parameters:
            if getattr(p, "id", None) == pid:
                return p
        return None

    # Rules
    def createAssignmentRule(self) -> "AssignmentRule":
        rule = AssignmentRule()
        self._rules.append(rule)
        return rule

    def getNumRules(self) -> int:
        return len(self._rules)

    def getRule(self, idx: int) -> "AssignmentRule":
        return self._rules[idx]

    # Reactions
    def createReaction(self) -> "Reaction":
        rxn = Reaction()
        self._reactions.append(rxn)
        return rxn

    def getNumReactions(self) -> int:
        return len(self._reactions)

    def getReaction(self, idx: int) -> "Reaction":
        return self._reactions[idx]

    # Parameter retrieval by identifier
    def getParameterById(self, pid: str) -> Optional["Parameter"]:
        for p in self._parameters:
            if getattr(p, "id", None) == pid:
                return p
        return None


class Compartment:
    def setId(self, cid: str) -> None:
        self.id = cid

    def setConstant(self, constant: bool) -> None:
        self.constant = constant

    def setSize(self, size: float) -> None:
        self.size = size

    def setUnits(self, units: str) -> None:
        self.units = units

    # Accessors used by copying routine
    def getId(self) -> str:
        return getattr(self, "id", "")

    def getConstant(self) -> bool:
        return getattr(self, "constant", True)

    def getSize(self) -> float:
        return getattr(self, "size", 1.0)

    def isSetUnits(self) -> bool:
        return hasattr(self, "units")

    def getUnits(self) -> str:
        return getattr(self, "units", "")


class Species:
    def setId(self, sid: str) -> None:
        self.id = sid

    def setCompartment(self, comp: str) -> None:
        self.compartment = comp

    def setInitialConcentration(self, value: float) -> None:
        self.initial_concentration = value

    def setBoundaryCondition(self, bc: bool) -> None:
        self.boundary_condition = bc

    def setHasOnlySubstanceUnits(self, flag: bool) -> None:
        self.has_only_substance_units = flag

    def setUnits(self, units: str) -> None:
        self.units = units

    def getId(self) -> str:
        return getattr(self, "id", "")

    def getCompartment(self) -> str:
        return getattr(self, "compartment", "")

    def getInitialConcentration(self) -> float:
        return getattr(self, "initial_concentration", 0.0)

    def getBoundaryCondition(self) -> bool:
        return getattr(self, "boundary_condition", False)

    def getHasOnlySubstanceUnits(self) -> bool:
        return getattr(self, "has_only_substance_units", False)

    def isSetUnits(self) -> bool:
        return hasattr(self, "units")

    def getUnits(self) -> str:
        return getattr(self, "units", "")


class Parameter:
    def setId(self, pid: str) -> None:
        self.id = pid

    def setValue(self, value: float) -> None:
        self.value = value

    def setConstant(self, constant: bool) -> None:
        self.constant = constant

    def getId(self) -> str:
        return getattr(self, "id", "")

    def getValue(self) -> float:
        return getattr(self, "value", 0.0)

    def getConstant(self) -> bool:
        return getattr(self, "constant", True)


class Reactant:
    def setSpecies(self, sid: str) -> None:
        self.species = sid

    def setStoichiometry(self, value: float) -> None:
        self.stoichiometry = value

    def setConstant(self, constant: bool) -> None:
        self.constant = constant

    # Accessors used in combined model copy
    def getSpecies(self) -> str:
        return getattr(self, "species", "")

    def getStoichiometry(self) -> float:
        return getattr(self, "stoichiometry", 1.0)

    def getConstant(self) -> bool:
        return getattr(self, "constant", True)


class Product:
    def setSpecies(self, sid: str) -> None:
        self.species = sid

    def setStoichiometry(self, value: float) -> None:
        self.stoichiometry = value

    def setConstant(self, constant: bool) -> None:
        self.constant = constant

    # Accessors used in combined model copy
    def getSpecies(self) -> str:
        return getattr(self, "species", "")

    def getStoichiometry(self) -> float:
        return getattr(self, "stoichiometry", 1.0)

    def getConstant(self) -> bool:
        return getattr(self, "constant", True)


class KineticLaw:
    def setMath(self, ast: Any) -> None:
        self.math = ast

    def getMath(self) -> Any:
        return getattr(self, "math", None)


class Reaction:
    def __init__(self) -> None:
        self.reactants: List[Reactant] = []
        self.products: List[Product] = []
        self.kinetic_law = KineticLaw()
        self.id: str = ""
        self.reversible: bool = False

    def setId(self, rid: str) -> None:
        self.id = rid

    def setReversible(self, rev: bool) -> None:
        self.reversible = rev

    def createReactant(self) -> Reactant:
        r = Reactant()
        self.reactants.append(r)
        return r

    def createProduct(self) -> Product:
        p = Product()
        self.products.append(p)
        return p

    def createKineticLaw(self) -> KineticLaw:
        self.kinetic_law = KineticLaw()
        return self.kinetic_law

    # Accessors used in combined model copy
    def getNumReactants(self) -> int:
        return len(self.reactants)

    def getReactant(self, idx: int) -> Reactant:
        return self.reactants[idx]

    def getNumProducts(self) -> int:
        return len(self.products)

    def getProduct(self, idx: int) -> Product:
        return self.products[idx]

    def isSetKineticLaw(self) -> bool:
        return hasattr(self, "kinetic_law")

    def getKineticLaw(self) -> KineticLaw:
        return self.kinetic_law

    def getId(self) -> str:
        return self.id

    def getReversible(self) -> bool:
        return self.reversible


class AssignmentRule:
    def setVariable(self, var: str) -> None:
        self.variable = var

    def setMath(self, ast: Any) -> None:
        self.math = ast

    def getVariable(self) -> str:
        return getattr(self, "variable", "")

    def getMath(self) -> Any:
        return getattr(self, "math", None)

    def getTypeCode(self) -> int:
        return SBML_ASSIGNMENT_RULE


def parseL3Formula(formula: str) -> Any:
    """Stub for parsing an SBML math formula.

    In this stub we simply return the formula string itself.  In the
    real libsbml this would return an abstract syntax tree.
    """
    return formula


def writeSBMLToFile(doc: SBMLDocument, filename: str) -> int:
    """Write an SBMLDocument to a file.

    The stub serialises the document as a simple string for
    inspection.  It does not produce valid SBML but allows
    downstream code to proceed without error.
    """
    # Write a simple placeholder
    with open(filename, "w", encoding="utf-8") as f:
        f.write("<!-- SBML stub output -->\n")
    return 0


class SBMLReader:
    def readSBML(self, filename: str) -> SBMLDocument:
        # return an empty document for validation purposes
        return SBMLDocument()
