"""Data inventory builder.

This module inspects a zip archive of Excel workbooks and records
the names of each workbook and its sheets.  The resulting inventory
is written to both CSV and Markdown files.  Inventory information
is useful when deciding which sheets to parse and when documenting
the provenance of the data used for model calibration.
"""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from loguru import logger


def build_inventory(zip_path: str, out_dir: str | Path) -> Tuple[pd.DataFrame, Path, Path]:
    """Build an inventory of all Excel workbooks and sheets.

    Args:
        zip_path: Path to the zip archive containing Excel files.
        out_dir: Directory into which the inventory should be written.

    Returns:
        A tuple ``(df, csv_path, md_path)`` where ``df`` is the
        inventory dataframe and ``csv_path``/``md_path`` are the
        locations of the CSV and Markdown files.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, object]] = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".xlsx"):
                continue
            try:
                with zf.open(name) as fh:
                    xls = pd.ExcelFile(fh)
                    for sheet in xls.sheet_names:
                        # Attempt to read first few rows to count dims
                        try:
                            df_sheet = xls.parse(sheet, nrows=5)
                            n_rows, n_cols = df_sheet.shape
                        except Exception:
                            n_rows, n_cols = (0, 0)
                        records.append({
                            "file": name,
                            "sheet": sheet,
                            "n_rows": n_rows,
                            "n_cols": n_cols,
                        })
            except Exception as exc:
                logger.warning(f"Failed to inspect {name}: {exc}")
    inv_df = pd.DataFrame.from_records(records)
    csv_path = out / "data_inventory.csv"
    md_path = out / "data_inventory.md"
    inv_df.to_csv(csv_path, index=False)
    # Write markdown table
    with md_path.open("w", encoding="utf-8") as md:
        md.write("# Data inventory\n\n")
        md.write("| File | Sheet | Rows | Cols |\n")
        md.write("|-----|-------|------|------|\n")
        for _, row in inv_df.iterrows():
            md.write(f"| {row['file']} | {row['sheet']} | {row['n_rows']} | {row['n_cols']} |\n")
    logger.info(f"Data inventory written to {csv_path} and {md_path}")
    return inv_df, csv_path, md_path
