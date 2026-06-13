"""Protocol forcing and observation shapes for calibration."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import yaml

CONDITION_COLUMNS = ("genotype", "condition", "group", "treatment")

def load_protocol_config(path: str | Path = "configs/protocols.yaml") -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        p = Path(__file__).resolve().parents[3] / "configs" / p.name
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _resolve_inherits(protocols: dict[str, Any], assay: str) -> dict[str, Any]:
    proto = dict(protocols.get(str(assay), {}))
    parent = proto.pop("inherit", None)
    if parent:
        base = _resolve_inherits(protocols, str(parent))
        base.update(proto)
        return base
    return proto

def resolve_protocol(assay: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config or load_protocol_config()
    pools = []
    if str(assay).startswith(("OCR", "ECAR", "mito_OCR")):
        pools.append(cfg.get("seahorse_protocols", {}))
    pools += [cfg.get("body_protocols", {}), cfg.get("seahorse_protocols", {})]
    for protocols in pools:
        if assay in protocols:
            return _resolve_inherits(protocols, assay)
    if str(assay).startswith("OCR"):
        return dict(cfg.get("seahorse_protocols", {}).get("default_ocr", {}))
    if str(assay).startswith("ECAR"):
        return dict(cfg.get("seahorse_protocols", {}).get("default_ecar", {}))
    raise KeyError(f"No protocol configured for assay '{assay}'")

def value_from_spec(spec: Any, params: dict[str, float]) -> float:
    if isinstance(spec, dict) and "parameter" in spec:
        name = str(spec["parameter"])
        if name not in params:
            raise KeyError(f"Protocol parameter '{name}' is missing")
        return float(params[name])
    return float(spec)

def detect_condition_column(df, preferred: str | None = None) -> str | None:
    if preferred and preferred in df.columns:
        return preferred
    for col in CONDITION_COLUMNS:
        if col in df.columns:
            return col
    return None

def smooth_pulse(t, center: float, width: float):
    arr = np.asarray(t, dtype=float)
    return np.exp(-0.5 * ((arr - float(center)) / max(float(width), 1.0e-9)) ** 2)

def evaluate_shape(name: str | None, t, params: dict[str, float]):
    arr = np.asarray(t, dtype=float)
    if not name:
        return np.ones_like(arr, dtype=float)
    defaults = {
        "ocr_basal": 1.0, "ocr_peak_time": 1.0, "ocr_peak_width": 0.18, "ocr_drop_time": 1.5, "ocr_drop_width": 0.22,
        "ecar_basal": 1.0, "ecar_peak_time": 1.0, "ecar_peak_width": 0.20, "ecar_drop_time": 1.5, "ecar_drop_width": 0.25,
        "min_shape": 1.0e-6,
    }
    p = {**defaults, **params}
    if name == "seahorse_ocr":
        shape = p["ocr_basal"] + p.get("ocr_peak_amp", 0.0) * smooth_pulse(arr, p["ocr_peak_time"], p["ocr_peak_width"]) - p.get("ocr_drop_amp", 0.0) * smooth_pulse(arr, p["ocr_drop_time"], p["ocr_drop_width"])
    elif name == "seahorse_ecar":
        shape = p["ecar_basal"] + p.get("ecar_peak_amp", 0.0) * smooth_pulse(arr, p["ecar_peak_time"], p["ecar_peak_width"]) - p.get("ecar_drop_amp", 0.0) * smooth_pulse(arr, p["ecar_drop_time"], p["ecar_drop_width"])
    else:
        raise KeyError(f"Unknown protocol shape '{name}'")
    return np.maximum(shape, float(p["min_shape"]))
