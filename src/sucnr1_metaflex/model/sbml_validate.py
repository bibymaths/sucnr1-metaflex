"""SBML validation utilities.

This module wraps libSBML's consistency checking routines.  It
provides a single function, :func:`validate_sbml`, which reads an
SBML file and returns True if no errors or warnings were
encountered.  Any errors are logged via the loguru logger.
"""

from __future__ import annotations

from loguru import logger

try:
    import libsbml  # type: ignore
except ImportError:
    libsbml = None  # type: ignore


def validate_sbml(file_path: str) -> bool:
    """Validate an SBML document stored at ``file_path``.

    Args:
        file_path: Path to the SBML XML file.

    Returns:
        ``True`` if the document passes basic consistency checks,
        otherwise ``False``.  Errors are logged.
    """
    if libsbml is None:
        raise RuntimeError("libsbml is not available; cannot validate SBML models.")
    reader = libsbml.SBMLReader()
    doc = reader.readSBML(file_path)
    if doc.getNumErrors() > 0:
        for i in range(doc.getNumErrors()):
            err = doc.getError(i)
            logger.error(f"SBML parse error (severity {err.getSeverity()}): {err.getMessage()}")
        return False
    n = doc.checkConsistency()
    if n > 0:
        for i in range(n):
            err = doc.getError(i)
            level = err.getSeverity()
            msg = err.getMessage()
            if level >= libsbml.LIBSBML_SEV_ERROR:
                logger.error(f"SBML validation error: {msg}")
            else:
                logger.warning(f"SBML validation warning: {msg}")
        return False
    logger.info(f"SBML document {file_path} is valid")
    return True
