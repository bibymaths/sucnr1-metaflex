import pytest
import zipfile
import pandas as pd

from sucnr1_metaflex.data.excel_parser import parse_dynamic_sheet, parse_workbook


def test_parse_dynamic_sheet_nonempty():
    # Load a known sheet from the supplementary zip
    with zipfile.ZipFile("Beltran2026_Supp.zip") as zf:
        with zf.open("aec8873_data_s2.xlsx") as fh:
            xls = pd.ExcelFile(fh)
            raw = xls.parse("F2D", header=None)
    tidy = parse_dynamic_sheet(raw)
    # The tidy DataFrame should not be empty and have expected columns
    assert not tidy.empty
    assert set(["genotype", "replicate", "time", "value"]).issubset(tidy.columns)
    # There should be multiple replicates
    assert tidy["replicate"].nunique() > 1


def test_parse_workbook():
    assay_map = {"F2D": "GTT"}
    with zipfile.ZipFile("Beltran2026_Supp.zip") as zf:
        with zf.open("aec8873_data_s2.xlsx") as fh:
            buf = fh.read()
    tidy = parse_workbook(buf, "aec8873_data_s2.xlsx", assay_map)
    assert not tidy.empty
    assert set(["file", "sheet", "assay"]).issubset(tidy.columns)