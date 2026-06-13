"""Excel parsing utilities.

The supplementary workbooks contain complex header layouts.  This
module implements heuristics for extracting time–series data from
wide tables.  The parser assumes that each sheet contains a header
row listing genotype or condition names, followed by rows of
numerical measurements across multiple replicates.  The first
numeric column is interpreted as time.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


def parse_dynamic_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """Parse a sheet with time–series data into a tidy DataFrame.

    The input ``df`` should be a raw DataFrame obtained from
    :func:`pandas.read_excel` with ``header=None``.  The parser
    searches for the first row that contains string entries in
    columns beyond the first two; these strings are assumed to be
    genotype or condition labels.  All subsequent rows with a
    numerical value in column 1 are interpreted as data rows.  The
    resulting table has columns ``genotype``, ``replicate``, ``time``
    and ``value``.

    Args:
        df: Raw DataFrame from Excel with no header.

    Returns:
        A tidy DataFrame with columns ``genotype``, ``replicate``,
        ``time`` and ``value``.  Empty or malformed rows are skipped.
    """
    if df.empty:
        return pd.DataFrame(columns=["genotype", "replicate", "time", "value"])
    # Identify the row containing genotype names.  We look for the
    # first row where there is at least one string in columns 2+.
    row_genotype: Optional[int] = None
    for idx, row in df.iterrows():
        # ignore completely blank rows
        if row.dropna().empty:
            continue
        # count strings in columns beyond column1 (index>1)
        if any(isinstance(x, str) and isinstance(x, str) and not pd.isna(x) for x in row.iloc[2:]):
            row_genotype = idx
            break
    if row_genotype is None:
        logger.warning("Could not find genotype row; returning empty DataFrame")
        return pd.DataFrame(columns=["genotype", "replicate", "time", "value"])
    # Build mapping from column index to genotype label
    header_row = df.loc[row_genotype]
    group_for_col: Dict[int, Optional[str]] = {}
    current_label: Optional[str] = None
    for col in range(len(header_row)):
        val = header_row.iloc[col]
        # If this cell contains a non‑null string, update the current label
        if isinstance(val, str) and pd.notna(val):
            current_label = val.strip()
        if col > 1:
            group_for_col[col] = current_label
    # Build lists of columns per genotype label
    cols_by_label: Dict[str, List[int]] = {}
    for col, label in group_for_col.items():
        if label is None:
            continue
        cols_by_label.setdefault(label, []).append(col)
    records: List[Dict[str, object]] = []
    # Iterate through rows after header row
    for idx in range(row_genotype + 1, len(df)):
        row = df.loc[idx]
        # Parse time from column 1 (index 1)
        time_val = row.iloc[1]
        time = pd.to_numeric(time_val, errors="coerce")
        if pd.isna(time):
            continue
        for label, cols in cols_by_label.items():
            for rep_idx, col in enumerate(cols, start=1):
                val = row.iloc[col]
                value = pd.to_numeric(val, errors="coerce")
                if pd.isna(value):
                    continue
                records.append({
                    "genotype": label,
                    "replicate": rep_idx,
                    "time": float(time),
                    "value": float(value),
                })
    tidy_df = pd.DataFrame.from_records(records)
    return tidy_df


def parse_workbook(buffer: bytes, file_name: str, assay_map: Dict[str, str]) -> pd.DataFrame:
    """Parse dynamic sheets in a workbook into a tidy DataFrame.

    Args:
        buffer: Byte contents of the Excel file.
        file_name: Name of the workbook (used for metadata).
        assay_map: Mapping of sheet names to assay identifiers.

    Returns:
        Concatenated tidy DataFrame with additional columns ``file``,
        ``sheet`` and ``assay``.
    """
    df_list: List[pd.DataFrame] = []
    try:
        xls = pd.ExcelFile(buffer)
    except Exception as exc:
        logger.error(f"Failed to open workbook {file_name}: {exc}")
        return pd.DataFrame()
    for sheet, assay in assay_map.items():
        if sheet not in xls.sheet_names:
            logger.warning(f"Sheet {sheet} not found in {file_name}")
            continue
        raw = xls.parse(sheet, header=None)
        tidy = parse_dynamic_sheet(raw)
        if tidy.empty:
            continue
        tidy["file"] = file_name
        tidy["sheet"] = sheet
        tidy["assay"] = assay
        df_list.append(tidy)
    if df_list:
        return pd.concat(df_list, ignore_index=True)
    return pd.DataFrame()
