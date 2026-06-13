"""I/O helpers for visualization modules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd


def load_parameter_file(path: str | Path) -> Dict[str, float]:
    """Load best-parameter output from JSON or CSV."""
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Parameter file not found: {p}")

    if p.suffix.lower() == ".json":
        with p.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return {str(k): float(v) for k, v in data.items()}

    df = pd.read_csv(p, index_col=0)

    if "value" not in df.columns:
        raise ValueError(f"Expected column 'value' in parameter CSV: {p}")

    return {str(k): float(v) for k, v in df["value"].items()}


def load_processed_table(data_dir: str | Path, filename: str) -> pd.DataFrame:
    path = Path(data_dir) / filename

    if not path.exists():
        raise FileNotFoundError(f"Processed table not found: {path}")

    return pd.read_csv(path)


def numeric_time_value_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with finite numeric time/value rows only."""
    out = df.copy()

    if "time" not in out.columns or "value" not in out.columns:
        raise ValueError("Expected columns 'time' and 'value'.")

    out["time"] = pd.to_numeric(out["time"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out[out["time"].notna() & out["value"].notna()]
    return out