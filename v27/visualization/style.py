"""Shared matplotlib styling, color palettes, and safe PNG save utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP_COLORS = {"raw_t1": "#1f77b4", "risk_t1": "#2ca02c", "risk_t5": "#ff7f0e"}
CLUSTER_COLORS = {
    "Price/ShortMom": "#1f77b4",
    "Volatility": "#d62728",
    "Price-Volume": "#2ca02c",
    "SlowMom/Quality": "#9467bd",
    "Risk/Tail/Micro": "#ff7f0e",
}


def save_fig(fig: plt.Figure, out_path: Path, dpi: int = 160) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path
