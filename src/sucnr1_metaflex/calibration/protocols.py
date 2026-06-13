"""Protocol forcing and observation shapes for calibration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

CONDITION_COLUMNS = ("genotype", "condition", "group", "treatment")
NEUTRAL_CONDITION_FACTORS = {
    "genotype_sucnr1": 1.0,
    "ligand_factor": 1.0,
    "antagonist_factor": 1.0,
}


def load_protocol_config(path: str | Path = "configs/protocols.yaml") -> dict[str, Any]:
    """Load the protocol configuration file."""
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
    """Resolve a body or Seahorse protocol for an assay name."""
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
    """Resolve a literal or ``{parameter: name}`` spec to a linear value."""
    if isinstance(spec, dict) and "parameter" in spec:
        name = str(spec["parameter"])
        if name not in params:
            raise KeyError(f"Protocol parameter '{name}' is missing")
        return float(params[name])
    return float(spec)


def resolve_initial_conditions(
    protocol: dict[str, Any],
    params: dict[str, float],
) -> dict[str, float]:
    """Resolve protocol initial conditions from fitted linear parameters."""
    return {
        str(key): value_from_spec(value, params)
        for key, value in protocol.get("initial_conditions", {}).items()
    }


def resolve_condition_factors(
    protocol: dict[str, Any],
    condition: str | None,
    params: dict[str, float],
) -> dict[str, float]:
    """Return complete, non-leaking Seahorse condition factors.

    The returned mapping always contains genotype, ligand, and antagonist
    factors. Fitted global values for these factors are intentionally ignored
    here: each assay-condition simulation starts from neutral factors, then
    applies explicit protocol overrides.
    """
    factors = dict(NEUTRAL_CONDITION_FACTORS)
    condition_factors = protocol.get("condition_factors", {}) or {}

    if not condition_factors:
        return factors

    condition_key = "" if condition is None else str(condition)
    if condition_key not in condition_factors:
        raise KeyError(
            f"No condition factors configured for condition '{condition_key}'"
        )

    overrides = condition_factors[condition_key] or {}
    for key, value in overrides.items():
        factors[str(key)] = value_from_spec(value, params)

    missing = sorted(set(NEUTRAL_CONDITION_FACTORS) - set(factors))
    if missing:
        raise ValueError(f"Condition factor resolution missing keys: {missing}")

    return {key: float(factors[key]) for key in NEUTRAL_CONDITION_FACTORS}


def detect_condition_column(df, preferred: str | None = None) -> str | None:
    """Detect a condition column in a tidy assay data frame."""
    if preferred and preferred in df.columns:
        return preferred
    for col in CONDITION_COLUMNS:
        if col in df.columns:
            return col
    return None


def smooth_pulse(t, center: float, width: float):
    """Gaussian pulse used by deterministic Seahorse protocol shapes."""
    arr = np.asarray(t, dtype=float)
    return np.exp(-0.5 * ((arr - float(center)) / max(float(width), 1.0e-9)) ** 2)


def evaluate_shape(name: str | None, t, params: dict[str, float]):
    """Evaluate a clamped deterministic observation-shape multiplier."""
    arr = np.asarray(t, dtype=float)
    if not name:
        return np.ones_like(arr, dtype=float)
    defaults = {
        "ocr_basal": 1.0,
        "ocr_peak_time": 1.0,
        "ocr_peak_width": 0.18,
        "ocr_drop_time": 1.5,
        "ocr_drop_width": 0.22,
        "ecar_basal": 1.0,
        "ecar_peak_time": 1.0,
        "ecar_peak_width": 0.20,
        "ecar_drop_time": 1.5,
        "ecar_drop_width": 0.25,
        "min_shape": 1.0e-6,
    }
    p = {**defaults, **params}
    if name == "seahorse_ocr":
        shape = (
            p["ocr_basal"]
            + p.get("ocr_peak_amp", 0.0)
            * smooth_pulse(arr, p["ocr_peak_time"], p["ocr_peak_width"])
            - p.get("ocr_drop_amp", 0.0)
            * smooth_pulse(arr, p["ocr_drop_time"], p["ocr_drop_width"])
        )
    elif name == "seahorse_ecar":
        shape = (
            p["ecar_basal"]
            + p.get("ecar_peak_amp", 0.0)
            * smooth_pulse(arr, p["ecar_peak_time"], p["ecar_peak_width"])
            - p.get("ecar_drop_amp", 0.0)
            * smooth_pulse(arr, p["ecar_drop_time"], p["ecar_drop_width"])
        )
    else:
        raise KeyError(f"Unknown protocol shape '{name}'")
    return np.maximum(shape, float(p["min_shape"]))


def collect_parameter_references(value: Any) -> set[str]:
    """Collect all ``{parameter: name}`` references in a nested object."""
    refs: set[str] = set()
    if isinstance(value, dict):
        if set(value.keys()) == {"parameter"}:
            refs.add(str(value["parameter"]))
        for item in value.values():
            refs.update(collect_parameter_references(item))
    elif isinstance(value, list):
        for item in value:
            refs.update(collect_parameter_references(item))
    return refs


def collect_protocol_parameter_references(
    assays: Iterable[str],
    protocol_config: dict[str, Any] | None = None,
) -> set[str]:
    """Collect parameter references used by protocols for selected assays."""
    cfg = protocol_config or load_protocol_config()
    refs: set[str] = set()
    for assay in sorted({str(a) for a in assays}):
        refs.update(collect_parameter_references(resolve_protocol(assay, cfg)))
    return refs


def validate_protocol_parameters(
    assays: Iterable[str],
    fit_parameter_names: Iterable[str],
    protocol_config: dict[str, Any] | None = None,
) -> None:
    """Fail early if selected protocols reference missing fit parameters."""
    required = collect_protocol_parameter_references(assays, protocol_config)
    available = {str(name) for name in fit_parameter_names}
    missing = sorted(required - available)
    if missing:
        raise ValueError(
            "Fit config is missing protocol parameter definitions: "
            f"{missing}"
        )
