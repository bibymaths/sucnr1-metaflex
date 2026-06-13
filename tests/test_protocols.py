import pytest
from pathlib import Path

from sucnr1_metaflex.data.tidy import ingest_all
from sucnr1_metaflex.simulation.protocols import load_dynamic_body_data, filter_assay


def test_filter_assay(tmp_path):
    # ingest data to temp directory
    zip_path = "Beltran2026_Supp.zip"
    config_path = "configs/data_sources.yaml"
    ingest_all(zip_path, config_path, tmp_path)
    df = load_dynamic_body_data(tmp_path)
    sub = filter_assay(df, "GTT")
    # Should only contain assay column equal to GTT
    assert not sub.empty
    assert sub["assay"].str.contains("GTT", case=False).all()