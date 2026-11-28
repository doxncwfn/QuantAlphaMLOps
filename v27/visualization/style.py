"""Shared matplotlib styling, color palettes, crisis shading, and safe PNG save utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

EXP_COLORS = {"raw_t1": "#1f77b4", "risk_t1": "#2ca02c", "risk_t5": "#ff7f0e"}
CLUSTER_COLORS = {
    "Price/ShortMom": "#1f77b4",
    "Volatility": "#d62728",
    "Price-Volume": "#2ca02c",
    "SlowMom/Quality": "#9467bd",
    "Risk/Tail/Micro": "#ff7f0e",
}
CRISIS_ERAS = [
    ("Euro", "2011-05-01", "2011-12-31"),
    ("China/Oil", "2015-08-01", "2016-03-31"),
    ("COVID", "2020-02-15", "2020-04-30"),
    ("Rate Hike", "2022-01-01", "2022-10-31"),
]


def shade_crisis_eras(ax: plt.Axes, add_labels: bool = False) -> None:
    """Overlay subtle crisis-regime vertical bands on time-series axes."""
    x_lo, x_hi = ax.get_xlim()
    for label, start_s, end_s in CRISIS_ERAS:
        t0 = pd.Timestamp(start_s)
        t1 = pd.Timestamp(end_s)
        ax.axvspan(t0, t1, color="#d62728", alpha=0.07, zorder=0)
        if add_labels:
            mid = t0 + (t1 - t0) / 2
            ax.text(
                mid,
                0.98,
                label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=7,
                color="#8c1d18",
                alpha=0.85,
            )
    ax.set_xlim(x_lo, x_hi)


def save_fig(fig: plt.Figure, out_path: Path, dpi: int = 160) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path
