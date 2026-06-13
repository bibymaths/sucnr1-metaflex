"""Create tidy datasets from the supplementary spreadsheets."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml
from loguru import logger

from .excel_parser import parse_workbook


def ingest_all(zip_path: str, config_path: str, out_dir: str | Path) -> Dict[str, Path]:
    """Ingest dynamic data from all configured workbooks."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg_path = Path(config_path)
    if not cfg_path.exists():
        repo_root = Path(__file__).resolve().parents[3]
        alt = repo_root / "configs" / cfg_path.name
        if alt.exists():
            cfg_path = alt

    with cfg_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assay_map: Dict[str, Dict[str, str]] = config.get("assay_map", {})
    all_records: List[pd.DataFrame] = []

    with zipfile.ZipFile(zip_path) as zf:
        zip_names = set(zf.namelist())

        for file_name, sheet_map in assay_map.items():
            if file_name not in zip_names:
                logger.warning(f"Workbook {file_name} not present in zip")
                continue

            with zf.open(file_name) as fh:
                workbook_buffer = io.BytesIO(fh.read())

            tidy_df = parse_workbook(
                workbook_buffer,
                file_name=file_name,
                assay_map=sheet_map,
            )

            if tidy_df is not None and not tidy_df.empty:
                all_records.append(tidy_df)

    if not all_records:
        logger.warning("No dynamic data found during ingestion")
        return {}

    all_df = pd.concat(all_records, ignore_index=True)

    if "time" in all_df.columns:
        all_df = all_df[all_df["time"] >= 0]

    all_csv = out / "all_tidy.csv"
    all_df.to_csv(all_csv, index=False)
    logger.info(f"Wrote tidy data with {len(all_df)} records to {all_csv}")

    assay_col = all_df["assay"].astype(str)

    body_df = all_df[~assay_col.str.contains("OCR|ECAR", case=False, na=False)]
    seahorse_df = all_df[assay_col.str.contains("OCR|ECAR", case=False, na=False)]

    body_csv = out / "dynamic_body.csv"
    seahorse_csv = out / "dynamic_seahorse.csv"

    body_df.to_csv(body_csv, index=False)
    seahorse_df.to_csv(seahorse_csv, index=False)

    logger.info(f"Wrote body dynamic data to {body_csv}")
    logger.info(f"Wrote Seahorse dynamic data to {seahorse_csv}")

    return {
        "all_tidy": all_csv,
        "dynamic_body": body_csv,
        "dynamic_seahorse": seahorse_csv,
    }