#!/usr/bin/env python3
"""Create the N=5000 exact-spectrum/NIPR counterpart to the N=10000 plot."""

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

FOCAL_INDICES = {4, 5, 6, 7, 4989, 4990, 4991, 4992}


def export_ranked(frame: pd.DataFrame) -> None:
    table = frame[
        [
            "global_eigenvalue_index_0based",
            "global_eigenvalue_index_1based",
            "valid_spectral_rank_1based",
            "valid_spectral_quantile",
            "eigenvalue_energy",
            "ipr",
            "NIPR",
            "is_zero_atom",
            "is_perron",
            "left_gap",
            "right_gap",
        ]
    ].copy()
    table["focal_quartet_mode"] = table["global_eigenvalue_index_0based"].isin(FOCAL_INDICES)
    table.to_csv(OUT / "n5000_full_spectrum_ranked_energy_nipr.csv", index=False, float_format="%.15g")


def build(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    energy = frame["eigenvalue_energy"].to_numpy(float)
    nipr = frame["NIPR"].to_numpy(float)
    zero = frame["is_zero_atom"].to_numpy(bool)
    perron = frame["is_perron"].to_numpy(bool)
    regular = ~zero & ~perron
    edges = np.histogram_bin_edges(energy[regular], bins="fd")
    counts, _ = np.histogram(energy[regular], bins=edges)
    width = float(np.diff(edges)[0])

    fig, axes = plt.subplots(2, 1, figsize=(14.5, 9.5), sharex=True, gridspec_kw={"height_ratios": [1.35, 1.0]})
    ax = axes[0]
    ax.scatter(energy[regular], nipr[regular], s=8, alpha=0.48, color="#277da1", linewidths=0, label="individual nonzero, non-Perron eigenpairs")
    ax.scatter(energy[zero], nipr[zero], s=5, alpha=0.09, color="#6c757d", linewidths=0, label="zero eigenspace (basis-dependent IPR)")
    ax.scatter(energy[perron], nipr[perron], s=55, marker="D", color="#f8961e", label="Perron mode", zorder=4)
    focal = np.asarray(sorted(FOCAL_INDICES), dtype=int)
    ax.scatter(energy[focal], nipr[focal], s=65, marker="*", color="#d62828", edgecolor="black", linewidth=0.35, label="two target quartets", zorder=5)
    ax.set_yscale("log")
    ax.set_ylabel(r"exact individual $N\,\mathrm{IPR}$")
    ax.set_title("Every saved eigenpair: energy versus individual localization (no bin mean)")
    ax.grid(alpha=0.16)
    ax.legend(frameon=False, fontsize=8, ncol=2)

    ax = axes[1]
    ax.stairs(counts, edges, fill=True, color="#8ecae6", edgecolor="#277da1", alpha=0.82, linewidth=0.6)
    ax.set_yscale("log")
    for index in focal:
        ax.axvline(energy[index], color="#d62828", linewidth=1.0, alpha=0.82)
    ax.set_xlabel(r"raw-adjacency energy $E=\lambda$")
    ax.set_ylabel("eigenvalue count in bin")
    ax.set_title(
        f"Target-independent Freedman–Diaconis histogram of 3,155 nonzero, non-Perron levels "
        f"(width={width:.6f}; 1,844 zero modes and Perron omitted)"
    )
    ax.grid(alpha=0.16)
    fig.suptitle("N=5,000, alpha=1.05, seed=4011773341 — exact spectrum and spectral occupancy")
    fig.tight_layout()
    fig.savefig(OUT / "n5000-full-spectrum-exact-nipr-and-fd-histogram.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "n5000-full-spectrum-exact-nipr-and-fd-histogram.pdf", bbox_inches="tight")
    plt.close(fig)
    return edges, counts


def write_interactive(frame: pd.DataFrame, edges: np.ndarray, counts: np.ndarray) -> None:
    points = []
    for row in frame.itertuples(index=False):
        index = int(row.global_eigenvalue_index_0based)
        category = "zero" if bool(row.is_zero_atom) else "perron" if bool(row.is_perron) else "focal" if index in FOCAL_INDICES else "regular"
        points.append(
            [
                index + 1,
                round(float(row.eigenvalue_energy), 12),
                round(float(row.NIPR), 6),
                category,
            ]
        )
    bins = [[round(float(edges[i]), 12), round(float(edges[i + 1]), 12), int(counts[i])] for i in range(len(counts))]
    template = (Path(__file__).with_name("n5000-spectrum.template.html")).read_text()
    html = template.replace("__POINTS_JSON__", json.dumps(points, separators=(",", ":")))
    html = html.replace("__BINS_JSON__", json.dumps(bins, separators=(",", ":")))
    path = OUT / "n5000-spectrum-localization.html"
    path.write_text(html)
    if path.stat().st_size >= 1_000_000:
        raise RuntimeError(f"visualization exceeds 1 MB: {path.stat().st_size}")


def main() -> None:
    global SOURCE, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-table', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    SOURCE, OUT = args.source_table, args.output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(SOURCE)
    export_ranked(frame)
    edges, counts = build(frame)
    write_interactive(frame, edges, counts)


if __name__ == "__main__":
    main()
