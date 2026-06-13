"""Utilities for determining common file system paths.

Throughout the codebase we refer to a few well‑known locations,
such as the directory containing this package, the repository root
and the default results directory.  This module centralises that
logic to avoid hard‑coding relative paths in multiple places.
"""

from pathlib import Path


def package_root() -> Path:
    """Return the root directory of the installed package."""
    return Path(__file__).resolve().parent


def project_root() -> Path:
    """Return the root directory of the repository (two levels up)."""
    return package_root().parent.parent


def default_results_dir() -> Path:
    """Return the default directory for storing results.

    The directory is created if it does not exist.
    """
    root = project_root() / "results"
    root.mkdir(exist_ok=True, parents=True)
    return root
