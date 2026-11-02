#!/usr/bin/env python3
"""Create the exact-spectrum and hub-structure follow-up figures.

This script only reads the saved N=10000 graph/eigensystem and prior audit
tables.  It does not regenerate a graph or run an eigensolver.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse


CASE = AUDIT = OUT = None

FOCAL = {
    "v17, E=-31.833": 21,
    "v17, E=+31.620": 9978,
    "v40, E=-27.642": 22,
    "v40, E=+27.266": 9977,
}
MODE_TO_LABEL = {
    "negative_target": "v17, E=-31.833",
    "positive_target": "v17, E=+31.620",
    "negative_companion": "v40, E=-27.642",
    "positive_companion": "v40, E=+27.266",
}


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def export_ranked_table(frame: pd.DataFrame) -> Path:
    output = frame[
        [
            "ascending_index",
            "ascending_rank_1based",
            "decreasing_algebraic_rank",
            "energy",
            "ipr",
            "n_times_ipr",
            "perron",
            "maximum_coordinate_vertex",
            "maximum_coordinate_mass",
            "relative_eigenpair_residual",
            "support_size_50",
            "support_size_90",
            "support_size_95",
        ]
    ].copy()
    output = output.rename(
        columns={
            "ascending_index": "ascending_index_0based",
            "decreasing_algebraic_rank": "descending_rank_1based",
            "maximum_coordinate_vertex": "dominant_vertex_0based",
        }
    )
    output.insert(3, "numerical_zero", np.isclose(output["energy"], 0.0, atol=1e-8, rtol=0.0))
    output.insert(4, "zero_eigenspace_ipr_basis_dependent", output["numerical_zero"])
    output["dominant_vertex_1based"] = output["dominant_vertex_0based"] + 1
    output["focal_mode"] = ""
    for label, index in FOCAL.items():
        output.loc[output["ascending_index_0based"] == index, "focal_mode"] = label
    path = OUT / "n10000_full_spectrum_ranked_energy_nipr.csv"
    output.to_csv(path, index=False, float_format="%.15g")
    return path


def plot_full_spectrum(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    energy = frame["energy"].to_numpy(float)
    nipr = frame["n_times_ipr"].to_numpy(float)
    perron = frame["perron"].to_numpy(bool)
    zero = np.isclose(energy, 0.0, atol=1e-8, rtol=0.0)
    regular = ~zero & ~perron

    fd_edges = np.histogram_bin_edges(energy[regular], bins="fd")
    counts, _ = np.histogram(energy[regular], bins=fd_edges)
    fd_width = float(np.diff(fd_edges)[0])

    fig, axes = plt.subplots(2, 1, figsize=(14.5, 9.5), sharex=True, gridspec_kw={"height_ratios": [1.35, 1.0]})
    ax = axes[0]
    ax.scatter(energy[regular], nipr[regular], s=7, alpha=0.46, color="#277da1", linewidths=0, label="individual nonzero, non-Perron eigenpairs")
    ax.scatter(energy[zero], nipr[zero], s=4, alpha=0.08, color="#6c757d", linewidths=0, label="zero eigenspace (basis-dependent IPR)")
    ax.scatter(energy[perron], nipr[perron], s=55, marker="D", color="#f8961e", label="Perron mode", zorder=4)
    annotation_offsets = {
        21: (-62, 12),
        22: (-62, -24),
        9977: (10, -24),
        9978: (10, 12),
    }
    for label, index in FOCAL.items():
        ax.scatter(energy[index], nipr[index], s=72, marker="*", color="#d62828", edgecolor="black", linewidth=0.35, zorder=5)
        ax.annotate(label, (energy[index], nipr[index]), xytext=annotation_offsets[index], textcoords="offset points", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel(r"exact individual $N\,\mathrm{IPR}$")
    ax.set_title("Every saved eigenpair: energy versus individual localization (no bin mean)")
    ax.grid(alpha=0.16)
    ax.legend(frameon=False, fontsize=8, ncol=2)

    ax = axes[1]
    ax.stairs(counts, fd_edges, fill=True, color="#8ecae6", edgecolor="#277da1", alpha=0.82, linewidth=0.6)
    ax.set_yscale("log")
    for label, index in FOCAL.items():
        ax.axvline(energy[index], color="#d62828", linewidth=1.1, alpha=0.85)
    ax.set_xlabel(r"raw-adjacency energy $E=\lambda$")
    ax.set_ylabel("eigenvalue count in bin")
    ax.set_title(
        f"Target-independent Freedman–Diaconis histogram of 6,247 nonzero, non-Perron levels "
        f"(width={fd_width:.6f}; 3,752 zero modes and Perron omitted)"
    )
    ax.grid(alpha=0.16)
    fig.suptitle("N=10,000, alpha=1.05, seed=2748869207 — exact spectrum and spectral occupancy")
    fig.tight_layout()
    save_figure(fig, "n10000-full-spectrum-exact-nipr-and-fd-histogram")
    return fd_edges, counts, regular


def plot_graph_a() -> None:
    shell = pd.read_csv(AUDIT / "mode_support_and_shell_mass.csv")
    region = pd.read_csv(AUDIT / "candidate_region_residuals.csv")
    colors = {
        "negative_target": "#277da1",
        "positive_target": "#43aa8b",
        "negative_companion": "#f8961e",
        "positive_companion": "#6a4c93",
    }
    styles = {
        "negative_target": "-",
        "positive_target": "--",
        "negative_companion": "-",
        "positive_companion": "--",
    }
    order = list(MODE_TO_LABEL)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6))

    ax = axes[0]
    for name in order:
        rows = shell[(shell["record_type"] == "shell") & (shell["mode_name"] == name)].sort_values("graph_distance")
        ax.plot(rows["graph_distance"], rows["shell_mass"], "o", linestyle=styles[name], color=colors[name], label=MODE_TO_LABEL[name])
    ax.set_yscale("log")
    ax.set_xlabel("graph distance from labeled hub vertex")
    ax.set_ylabel("squared mass in shell")
    ax.set_title("Shell mass around v17 and v40")
    ax.set_xticks([0, 1, 2, 3])
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.18)

    ax = axes[1]
    radius_order = {"radius0": 0, "radius1": 1, "radius2": 2}
    for name in order:
        rows = region[(region["mode_name"] == name) & region["region_name"].isin(radius_order)].copy()
        rows["radius"] = rows["region_name"].map(radius_order)
        rows = rows.sort_values("radius")
        ax.plot(rows["region_fraction"], rows["captured_mass"], "o", linestyle=styles[name], color=colors[name], label=MODE_TO_LABEL[name])
    ax.axvline(0.20, color="black", linestyle=":", linewidth=1.0, label="20% reporting cap")
    ax.set_xlabel("radius-neighborhood size / N")
    ax.set_ylabel("captured squared mass")
    ax.set_title("Mass captured by hub neighborhoods")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.18)

    fig.suptitle("Graph A — Hub-centered localization for the ±32 and ±27 energy pairs")
    fig.tight_layout()
    save_figure(fig, "graph-a-hub-shells-and-radius-coverage")


def build_adjacency() -> tuple[sparse.csr_matrix, np.ndarray]:
    archive = np.load(CASE / "arrays/graph_edges.npz")
    n = int(archive["n"])
    u = archive["edge_u"].astype(np.int64)
    v = archive["edge_v"].astype(np.int64)
    rows = np.concatenate([u, v])
    cols = np.concatenate([v, u])
    adjacency = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    return adjacency, archive["degrees"].astype(np.int64)


def radial_positions(vertices: np.ndarray, mass: np.ndarray, center: int) -> dict[int, tuple[float, float]]:
    others = [int(v) for v in vertices if int(v) != center]
    others.sort(key=lambda v: (-mass[v], v))
    positions = {center: (0.0, 0.0)}
    for j, vertex in enumerate(others):
        radius = 1.0 if j < 18 else 1.55
        local_j = j if j < 18 else j - 18
        local_n = min(18, len(others)) if j < 18 else max(1, len(others) - 18)
        angle = 2.0 * np.pi * local_j / local_n + (0.12 if j >= 18 else 0.0)
        positions[vertex] = (radius * np.cos(angle), radius * np.sin(angle))
    return positions


def plot_hub_structure(frame: pd.DataFrame) -> None:
    adjacency, degrees = build_adjacency()
    vectors = np.load(CASE / "arrays/full_eigenvectors_ascending.npy", mmap_mode="r")
    pair_definitions = [
        (17, (21, 9978), "v17: ±32 energy pair"),
        (40, (22, 9977), "v40: ±27 energy pair"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 7.0))
    for ax, (center, pair, title) in zip(axes, pair_definitions):
        pair_mass = 0.5 * (np.asarray(vectors[:, pair[0]]) ** 2 + np.asarray(vectors[:, pair[1]]) ** 2)
        neighbors = adjacency.indices[adjacency.indptr[center] : adjacency.indptr[center + 1]]
        top_neighbors = neighbors[np.lexsort((neighbors, -pair_mass[neighbors]))[:35]]
        vertices = np.concatenate([[center], top_neighbors]).astype(np.int64)
        selected = set(int(x) for x in vertices)
        positions = radial_positions(vertices, pair_mass, center)
        selected_edges: list[tuple[int, int]] = []
        for vertex in vertices:
            for neighbor in adjacency.indices[adjacency.indptr[vertex] : adjacency.indptr[vertex + 1]]:
                if int(neighbor) in selected and int(vertex) < int(neighbor):
                    selected_edges.append((int(vertex), int(neighbor)))
        for u, v in selected_edges:
            x1, y1 = positions[u]
            x2, y2 = positions[v]
            ax.plot([x1, x2], [y1, y2], color="#adb5bd", linewidth=0.55, alpha=0.55, zorder=1)
        noncenter = np.asarray([v for v in vertices if v != center], dtype=np.int64)
        sizes = 32.0 + 1200.0 * np.sqrt(pair_mass[noncenter] / max(float(pair_mass[center]), 1e-15))
        coords = np.asarray([positions[int(v)] for v in noncenter])
        points = ax.scatter(coords[:, 0], coords[:, 1], s=sizes, c=np.log10(np.maximum(pair_mass[noncenter], 1e-16)), cmap="viridis", edgecolor="black", linewidth=0.25, zorder=3)
        ax.scatter([0], [0], s=340, marker="*", color="#d62828", edgecolor="black", linewidth=0.7, zorder=5)
        ax.annotate(f"v{center}\ndeg={degrees[center]}", (0, 0), xytext=(0, -28), textcoords="offset points", ha="center", fontsize=9)
        for vertex in noncenter[:6]:
            ax.annotate(f"v{int(vertex)}", positions[int(vertex)], xytext=(3, 3), textcoords="offset points", fontsize=7)
        displayed_mass = float(pair_mass[vertices].sum())
        ax.set_title(f"{title}\ncenter + 35 highest-mass neighbors; displayed mass={displayed_mass:.3f}")
        ax.set_aspect("equal")
        ax.set_xlim(-1.78, 1.78)
        ax.set_ylim(-1.78, 1.78)
        ax.axis("off")
        fig.colorbar(points, ax=ax, fraction=0.036, pad=0.01, label=r"$\log_{10}$ pair-mean squared mass")
    fig.suptitle("Actual induced hub-core structure (edges among the displayed vertices only)")
    fig.tight_layout()
    save_figure(fig, "n10000-v17-v40-hub-core-structure")


def write_interactive(frame: pd.DataFrame, fd_edges: np.ndarray, counts: np.ndarray) -> None:
    energy = frame["energy"].to_numpy(float)
    zero = np.isclose(energy, 0.0, atol=1e-8, rtol=0.0)
    perron = frame["perron"].to_numpy(bool)
    focal_by_index = {index: label for label, index in FOCAL.items()}
    points = []
    for row in frame.itertuples(index=False):
        index = int(row.ascending_index)
        category = "zero" if zero[index] else "perron" if perron[index] else "focal" if index in focal_by_index else "regular"
        points.append(
            [
                index + 1,
                round(float(row.energy), 12),
                round(float(row.n_times_ipr), 6),
                int(row.maximum_coordinate_vertex),
                category,
            ]
        )
    bins = [
        [round(float(fd_edges[i]), 12), round(float(fd_edges[i + 1]), 12), int(counts[i])]
        for i in range(len(counts))
    ]

    template_path = Path(__file__).with_name("full-spectrum-localization.template.html")
    output_path = OUT / "full-spectrum-localization.html"
    template = template_path.read_text()
    html = template.replace("__POINTS_JSON__", json.dumps(points, separators=(",", ":")))
    html = html.replace("__BINS_JSON__", json.dumps(bins, separators=(",", ":")))
    output_path.write_text(html)
    if output_path.stat().st_size >= 1_000_000:
        raise RuntimeError(f"interactive visualization exceeds 1 MB: {output_path.stat().st_size}")


def main() -> None:
    global CASE, AUDIT, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-dir', type=Path, required=True)
    parser.add_argument('--audit-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    CASE, AUDIT, OUT = args.case_dir, args.audit_dir, args.output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(CASE / "tables/full_spectrum_localization.csv.gz")
    export_ranked_table(frame)
    fd_edges, counts, _ = plot_full_spectrum(frame)
    plot_graph_a()
    plot_hub_structure(frame)
    write_interactive(frame, fd_edges, counts)


if __name__ == "__main__":
    main()
