"""Shared plotting utilities."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt


def apply_mpl_style() -> None:
    """Apply a restrained Matplotlib style without hard-coded colors."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 250,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.autolayout": True,
        }
    )


def safe_slug(text: object) -> str:
    """Convert arbitrary labels to safe filenames."""
    value = str(text)
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = value.strip("_")
    return value or "plot"


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_figure(fig: plt.Figure, path: str | Path, close: bool = True) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    if close:
        plt.close(fig)
    return out_path