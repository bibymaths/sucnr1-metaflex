"""Data ingestion and preprocessing.

This subpackage provides tools to extract the supplementary Excel
data, catalogue the available sheets and convert selected sheets
into a tidy, replicate‑level format suitable for model calibration.
"""

from .extract import extract_zip
from .inventory import build_inventory
from .tidy import ingest_all

__all__ = ["extract_zip", "build_inventory", "ingest_all"]
