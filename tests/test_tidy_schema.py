import pytest
from pathlib import Path

from sucnr1_metaflex.data.tidy import ingest_all


def test_ingest_all(tmp_path):
    zip_path = "Beltran2026_Supp.zip"
    config_path = "configs/data_sources.yaml"
    results = ingest_all(zip_path, config_path, tmp_path)
    # Expect keys
    assert "all_tidy" in results
    all_csv = results["all_tidy"]
    assert Path(all_csv).exists()
    # Load and check no negative times
    import pandas as pd
    df = pd.read_csv(all_csv)
    assert (df["time"] >= 0).all()