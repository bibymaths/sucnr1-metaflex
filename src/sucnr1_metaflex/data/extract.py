"""Utilities for unpacking the supplementary data zip.

The supplementary data are distributed as a single zip archive
containing 15 Excel workbooks.  The :func:`extract_zip` function
extracts all files to a given output directory and returns the
extracted paths.  It does not perform any parsing itself.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import List

from loguru import logger


def extract_zip(zip_path: str, out_dir: str | Path) -> List[Path]:
    """Extract all Excel files from ``zip_path`` into ``out_dir``.

    Args:
        zip_path: Path to the zip archive containing Excel files.
        out_dir: Directory into which the files should be extracted.

    Returns:
        A list of :class:`pathlib.Path` objects pointing to the
        extracted files.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"Not a valid zip archive: {zip_path}")
    extracted: List[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.lower().endswith(".xlsx"):
                logger.debug(f"Extracting {name} to {out}")
                dest = out / name
                with dest.open("wb") as fh:
                    fh.write(zf.read(name))
                extracted.append(dest)
    return extracted
