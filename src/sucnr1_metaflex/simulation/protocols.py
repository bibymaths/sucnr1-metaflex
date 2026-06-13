"""Protocol utilities.

This module defines simple helpers for working with the dynamic
datasets produced by the ingestion pipeline.  Each tolerance test
or fasting measurement is represented as a pandas DataFrame with
columns ``time``, ``value``, ``genotype`` and ``assay``.  These
functions filter the tidy dataset for a specific assay and return
the relevant subset.  More elaborate protocol implementations,
including time–dependent inputs, can be layered on top of these
utilities in the future.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def load_dynamic_body_data(processed_dir: str | Path) -> pd.DataFrame:
    """Load the processed dynamic body dataset.

    Args:
        processed_dir: Directory containing the CSV files produced by
            :func:`sucnr1_metaflex.data.tidy.ingest_all`.

    Returns:
        A DataFrame with time–series measurements.
    """
    processed_dir = Path(processed_dir)
    path = processed_dir / "dynamic_body.csv"
    return pd.read_csv(path)


def load_dynamic_seahorse_data(processed_dir: str | Path) -> pd.DataFrame:
    """Load the processed Seahorse dynamic dataset."""
    processed_dir = Path(processed_dir)
    path = processed_dir / "dynamic_seahorse.csv"
    return pd.read_csv(path)


def filter_assay(df: pd.DataFrame, assay: str) -> pd.DataFrame:
    """Return records belonging to a given assay (case insensitive)."""
    mask = df["assay"].str.lower() == assay.lower()
    return df.loc[mask].copy()
