import pytest
from pathlib import Path

from sucnr1_metaflex.data.inventory import build_inventory


def test_inventory_counts(tmp_path):
    # Use the provided zip file
    zip_path = Path("Beltran2026_Supp.zip")
    assert zip_path.exists(), "Supplementary zip file missing"
    df, csv_path, md_path = build_inventory(str(zip_path), tmp_path)
    # There should be at least 15 Excel workbooks
    unique_files = set(df["file"])
    assert len(unique_files) >= 15
    # The output files should exist
    assert csv_path.exists()
    assert md_path.exists()