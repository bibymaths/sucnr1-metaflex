"""Top-level package for sucnr1-metaflex.

This package contains tools to ingest the supplemental data from
Marsal‑Beltran et al. (2026), build SBML models of hepatic
metabolic flexibility, calibrate them against experimental data,
simulate perturbations and generate reports and dashboards.

The public API is exposed through the CLI commands defined in
`cli.py`.  Individual modules can be imported for programmatic
access.
"""

from importlib.metadata import version as _version

__all__ = ["__version__"]

try:
    __version__ = _version("sucnr1-metaflex")
except Exception:
    __version__ = "0.0.0"
