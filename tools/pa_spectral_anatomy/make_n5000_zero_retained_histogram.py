#!/usr/bin/env python3
"""Compare exact multiplicity with zero-removed and zero-retained histograms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SOURCE = OUT = None

ZERO_ATOL = 1e-8


def multiplicity_groups(energy: np.ndarray, keep: np.ndarray) -> list[dict[str, float | int]]:
    values = np.sort(energy[keep])
    groups: list[dict[str, float | int]] = []
    start = 0
    for j in range(1, len(values) + 1):
        if j == len(values) or abs(values[j] - values[start]) > ZERO_ATOL:
            block = values[start:j]
            groups.append({"energy": float(np.mean(block)), "multiplicity": int(len(block))})
            start = j
    return groups


def bin_rows(
    energy: np.ndarray,
    nipr: np.ndarray,
    mask: np.ndarray,
    edges: np.ndarray,
) -> list[dict[str, float | int | None]]:
    rows = []
    for i in range(len(edges) - 1):
        inside = mask & (energy >= edges[i]) & (energy < edges[i + 1] if i < len(edges) - 2 else energy <= edges[i + 1])
        values = nipr[inside]
        rows.append(
            {
                "left": float(edges[i]),
                "right": float(edges[i + 1]),
                "count": int(inside.sum()),
                "mean_nipr": float(np.mean(values)) if len(values) else None,
                "median_nipr": float(np.median(values)) if len(values) else None,
                "min_nipr": float(np.min(values)) if len(values) else None,
                "max_nipr": float(np.max(values)) if len(values) else None,
            }
        )
    return rows


def main() -> None:
    global SOURCE, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-table', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    SOURCE, OUT = args.source_table, args.output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(SOURCE)
    energy = frame["eigenvalue_energy"].to_numpy(float)
    nipr = frame["NIPR"].to_numpy(float)
    perron = frame["is_perron"].to_numpy(bool)
    zero = frame["is_zero_atom"].to_numpy(bool)
    nonzero_nonperron = ~zero & ~perron
    zero_retained = ~perron

    edges = np.histogram_bin_edges(energy[nonzero_nonperron], bins="fd")
    width = float(np.diff(edges)[0])
    removed_rows = bin_rows(energy, nipr, nonzero_nonperron, edges)
    retained_rows = bin_rows(energy, nipr, zero_retained, edges)
    removed = np.asarray([row["count"] for row in removed_rows], dtype=int)
    retained = np.asarray([row["count"] for row in retained_rows], dtype=int)
    zero_bin = int(np.searchsorted(edges, 0.0, side="right") - 1)

    groups = multiplicity_groups(energy, ~perron)
    central_groups = [row for row in groups if -2.5 <= float(row["energy"]) <= 2.5]

    fig, axes = plt.subplots(2, 2, figsize=(15.0, 10.0))
    ax = axes[0, 0]
    gx = np.asarray([row["energy"] for row in central_groups])
    gm = np.asarray([row["multiplicity"] for row in central_groups])
    ordinary = np.abs(gx) > ZERO_ATOL
    ax.scatter(gx[ordinary], gm[ordinary], s=8, color="#6c757d", alpha=0.50, label="distinct nonzero energy")
    ax.scatter(gx[~ordinary], gm[~ordinary], s=70, color="#d62828", marker="D", label="zero atom")
    ax.set_yscale("log")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(0.8, 6000)
    ax.set_xlabel(r"exact energy $E=\lambda$")
    ax.set_ylabel("exact-value multiplicity")
    ax.set_title("Exact multiplicity: nonzero levels are simple")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.18)

    ax = axes[0, 1]
    ax.stairs(retained, edges, fill=True, color="#43aa8b", edgecolor="#277da1", linewidth=0.55, alpha=0.80)
    ax.set_yscale("log")
    ax.set_xlabel(r"energy $E=\lambda$")
    ax.set_ylabel("eigenvalue count in bin")
    ax.set_title(f"Full histogram with zero atom retained (same width={width:.6f})")
    ax.grid(alpha=0.18)

    for ax, counts, title, color in [
        (axes[1, 0], removed, "Central zoom: zero atom removed", "#8ecae6"),
        (axes[1, 1], retained, "Central zoom: zero atom retained", "#43aa8b"),
    ]:
        ax.stairs(counts, edges, fill=True, color=color, edgecolor="#277da1", linewidth=0.65, alpha=0.82)
        ax.set_yscale("log")
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(0.8, 6000)
        ax.axvline(0.0, color="#d62828", linewidth=1.0, alpha=0.8)
        ax.set_xlabel(r"energy $E=\lambda$")
        ax.set_ylabel("eigenvalue count in bin")
        ax.set_title(title)
        ax.grid(alpha=0.18)
    axes[1, 0].annotate(
        f"zero-crossing bin count={removed[zero_bin]}",
        (0.0, removed[zero_bin]), xytext=(10, 12), textcoords="offset points", fontsize=8,
    )
    axes[1, 1].annotate(
        f"zero-crossing bin count={retained[zero_bin]}\n=1,844 zeros + {removed[zero_bin]} nonzero",
        (0.0, retained[zero_bin]), xytext=(10, -30), textcoords="offset points", fontsize=8,
    )
    fig.suptitle("N=5,000: exact multiplicity versus interval-bin occupancy")
    fig.tight_layout()
    fig.savefig(OUT / "n5000-zero-retained-multiplicity-and-histograms.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "n5000-zero-retained-multiplicity-and-histograms.pdf", bbox_inches="tight")
    plt.close(fig)

    template = (Path(__file__).with_name("zero-retained-histogram.template.html")).read_text()
    html = template.replace("__GROUPS_JSON__", json.dumps(central_groups, separators=(",", ":")))
    html = html.replace("__REMOVED_JSON__", json.dumps(removed_rows, separators=(",", ":")))
    html = html.replace("__RETAINED_JSON__", json.dumps(retained_rows, separators=(",", ":")))
    html = html.replace("pa-zero-retained-v1", "pa-zero-retained-n5000-v1")
    html = html.replace("multiplicity 3752", "multiplicity 1844")
    html = html.replace("multiplicity 3,752", "multiplicity 1,844")
    html = html.replace("y(3752)", "y(1844)")
    html = html.replace("N=10,000 · fixed Freedman–Diaconis width 0.315860", "N=5,000 · fixed Freedman–Diaconis width 0.406867")
    path = OUT / "n5000-zero-retained-histogram.html"
    path.write_text(html)
    if path.stat().st_size >= 1_000_000:
        raise RuntimeError(f"visualization exceeds 1 MB: {path.stat().st_size}")


if __name__ == "__main__":
    main()
