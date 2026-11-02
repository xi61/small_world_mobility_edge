#!/usr/bin/env python3
"""Audit eigenvector structure for the N=5000 and N=10000 zero-crossing bins."""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh


ROOT = OUT = None
TOP_K = (1, 2, 5, 10, 20, 50, 100)


def adjacency_from_npz(path: Path, n_key: str = "n") -> tuple[sparse.csr_matrix, np.ndarray]:
    archive = np.load(path)
    n = int(archive[n_key])
    u = archive["edge_u"].astype(np.int64)
    v = archive["edge_v"].astype(np.int64)
    rows = np.concatenate([u, v])
    cols = np.concatenate([v, u])
    adjacency = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    return adjacency, archive["degrees"].astype(np.int64)


def degree_ranks(degrees: np.ndarray) -> np.ndarray:
    order = np.lexsort((np.arange(len(degrees)), -degrees))
    ranks = np.empty(len(degrees), dtype=np.int64)
    ranks[order] = np.arange(1, len(degrees) + 1)
    return ranks


def bfs(adjacency: sparse.csr_matrix, source: int) -> np.ndarray:
    distance = np.full(adjacency.shape[0], -1, dtype=np.int64)
    distance[source] = 0
    queue: deque[int] = deque([source])
    while queue:
        vertex = queue.popleft()
        for neighbor in adjacency.indices[adjacency.indptr[vertex] : adjacency.indptr[vertex + 1]]:
            if distance[neighbor] < 0:
                distance[neighbor] = distance[vertex] + 1
                queue.append(int(neighbor))
    return distance


def neighbor_set(adjacency: sparse.csr_matrix, vertex: int) -> set[int]:
    return set(int(x) for x in adjacency.indices[adjacency.indptr[vertex] : adjacency.indptr[vertex + 1]])


def pair_similarity(adjacency: sparse.csr_matrix, a: int, b: int) -> dict[str, float | int | bool]:
    na = neighbor_set(adjacency, a)
    nb = neighbor_set(adjacency, b)
    shared = len(na & nb)
    union = len(na | nb)
    return {
        "adjacent": b in na,
        "shared_neighbors": shared,
        "neighbor_jaccard": shared / union if union else 0.0,
        "neighbor_cosine": shared / math.sqrt(len(na) * len(nb)) if na and nb else 0.0,
        "neighbor_symmetric_difference": len(na ^ nb),
    }


def induced_metrics(adjacency: sparse.csr_matrix, degrees: np.ndarray, vertices: np.ndarray, vector: np.ndarray, energy: float) -> dict[str, float | int]:
    sub = adjacency[vertices][:, vertices]
    internal_edges = int(sub.nnz // 2)
    components, labels = csgraph.connected_components(sub, directed=False, return_labels=True)
    largest = int(np.bincount(labels).max()) if len(labels) else 0
    boundary_edges = int(degrees[vertices].sum() - 2 * internal_edges)
    captured = float(np.sum(vector[vertices] ** 2))
    truncated = np.zeros(adjacency.shape[0], dtype=float)
    truncated[vertices] = vector[vertices] / math.sqrt(captured)
    residual = float(np.linalg.norm(adjacency @ truncated - energy * truncated))
    return {
        "internal_edges": internal_edges,
        "components": int(components),
        "largest_component": largest,
        "induced_density": 2 * internal_edges / (len(vertices) * (len(vertices) - 1)) if len(vertices) > 1 else 0.0,
        "boundary_edges": boundary_edges,
        "captured_mass": captured,
        "truncated_residual": residual,
    }


def analyze_modes(
    label: str,
    values: np.ndarray,
    vectors: dict[int, np.ndarray],
    indices: np.ndarray,
    adjacency: sparse.csr_matrix,
    degrees: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ranks = degree_ranks(degrees)
    summaries: list[dict[str, object]] = []
    top_rows: list[dict[str, object]] = []
    induced_rows: list[dict[str, object]] = []
    for index in indices:
        index = int(index)
        vector = np.asarray(vectors[index], dtype=float)
        mass = vector * vector
        order = np.lexsort((np.arange(len(vector)), -mass))
        cumulative = np.cumsum(mass[order])
        center = int(order[0])
        distance = bfs(adjacency, center)
        left_gap = float(values[index] - values[index - 1]) if index > 0 else math.nan
        right_gap = float(values[index + 1] - values[index]) if index + 1 < len(values) else math.nan
        top_two = [int(order[0]), int(order[1])]
        top_pair = pair_similarity(adjacency, top_two[0], top_two[1])
        best_pair: tuple[float, int, int, dict[str, float | int | bool]] | None = None
        for a_pos in range(10):
            for b_pos in range(a_pos + 1, 10):
                a, b = int(order[a_pos]), int(order[b_pos])
                record = pair_similarity(adjacency, a, b)
                score = float(record["neighbor_cosine"])
                if best_pair is None or score > best_pair[0]:
                    best_pair = (score, a, b, record)
        assert best_pair is not None
        summary: dict[str, object] = {
            "case": label,
            "ascending_index_0based": index,
            "ascending_rank_1based": index + 1,
            "energy": float(values[index]),
            "left_gap": left_gap,
            "right_gap": right_gap,
            "min_adjacent_gap": min(left_gap, right_gap),
            "ipr": float(np.sum(mass * mass)),
            "n_times_ipr": float(len(vector) * np.sum(mass * mass)),
            "dominant_vertex": center,
            "dominant_vertex_birth_fraction": center / len(vector),
            "dominant_mass": float(mass[center]),
            "dominant_degree": int(degrees[center]),
            "dominant_degree_rank": int(ranks[center]),
            "top2_vertices_adjacent": bool(top_pair["adjacent"]),
            "top2_shared_neighbors": int(top_pair["shared_neighbors"]),
            "top2_neighbor_jaccard": float(top_pair["neighbor_jaccard"]),
            "top2_neighbor_cosine": float(top_pair["neighbor_cosine"]),
            "top2_neighbor_symmetric_difference": int(top_pair["neighbor_symmetric_difference"]),
            "top2_opposite_sign": bool(np.sign(vector[top_two[0]]) != np.sign(vector[top_two[1]])),
            "top2_combined_mass": float(mass[top_two].sum()),
            "top10_best_twin_a": int(best_pair[1]),
            "top10_best_twin_b": int(best_pair[2]),
            "top10_best_neighbor_cosine": float(best_pair[0]),
            "top10_best_pair_opposite_sign": bool(np.sign(vector[best_pair[1]]) != np.sign(vector[best_pair[2]])),
            "top10_best_pair_combined_mass": float(mass[[best_pair[1], best_pair[2]]].sum()),
        }
        for k in TOP_K:
            chosen = order[:k]
            summary[f"top{k}_mass"] = float(cumulative[k - 1])
            summary[f"top{k}_degree_mean"] = float(np.mean(degrees[chosen]))
            summary[f"top{k}_degree_median"] = float(np.median(degrees[chosen]))
            summary[f"top{k}_within_distance1_count"] = int(np.sum(distance[chosen] <= 1))
        for level in (0.50, 0.80, 0.90, 0.95, 0.99):
            summary[f"support_size_{int(level * 100)}"] = int(np.searchsorted(cumulative, level, side="left") + 1)
        summaries.append(summary)
        for rank, vertex in enumerate(order[:20], start=1):
            top_rows.append(
                {
                    "case": label,
                    "ascending_index_0based": index,
                    "energy": float(values[index]),
                    "mass_rank": rank,
                    "vertex": int(vertex),
                    "coordinate": float(vector[vertex]),
                    "mass": float(mass[vertex]),
                    "degree": int(degrees[vertex]),
                    "degree_rank": int(ranks[vertex]),
                    "distance_from_dominant": int(distance[vertex]),
                }
            )
        for k in (10, 20, 50, 100):
            metrics = induced_metrics(adjacency, degrees, order[:k], vector, float(values[index]))
            induced_rows.append({"case": label, "ascending_index_0based": index, "energy": float(values[index]), "top_k": k, **metrics})
    return pd.DataFrame(summaries), pd.DataFrame(top_rows), pd.DataFrame(induced_rows)


def representative_graph(
    axis: plt.Axes,
    case: str,
    index: int,
    energy: float,
    vector: np.ndarray,
    adjacency: sparse.csr_matrix,
    degrees: np.ndarray,
) -> None:
    mass = vector * vector
    order = np.lexsort((np.arange(len(vector)), -mass))
    top = order[:20].astype(int)
    selected = set(int(x) for x in top)
    center = int(top[0])
    angles = np.linspace(0, 2 * np.pi, len(top), endpoint=False)
    positions = {center: (0.0, 0.0)}
    for j, vertex in enumerate(top[1:]):
        positions[int(vertex)] = (math.cos(angles[j]), math.sin(angles[j]))
    for a in top:
        for b in adjacency.indices[adjacency.indptr[a] : adjacency.indptr[a + 1]]:
            if int(b) in selected and int(a) < int(b):
                x1, y1 = positions[int(a)]
                x2, y2 = positions[int(b)]
                axis.plot([x1, x2], [y1, y2], color="#adb5bd", lw=0.7, alpha=0.7, zorder=1)
    coords = np.asarray([positions[int(v)] for v in top[1:]])
    sizes = 45 + 900 * np.sqrt(mass[top[1:]] / mass[center])
    axis.scatter(coords[:, 0], coords[:, 1], s=sizes, c=np.sign(vector[top[1:]]), cmap="coolwarm", vmin=-1, vmax=1, edgecolor="black", linewidth=0.25, zorder=3)
    axis.scatter([0], [0], s=320, marker="*", color="#f8961e", edgecolor="black", linewidth=0.7, zorder=5)
    for vertex in top[:8]:
        axis.annotate(f"v{int(vertex)}\nd={int(degrees[vertex])}", positions[int(vertex)], xytext=(3, 3), textcoords="offset points", fontsize=7)
    axis.set_title(f"{case}: E={energy:.6f}, index={index}\ntop-20 mass induced subgraph; center v{center}")
    axis.set_aspect("equal")
    axis.set_xlim(-1.25, 1.25)
    axis.set_ylim(-1.25, 1.25)
    axis.axis("off")


def main() -> None:
    global ROOT, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--recover-missing-vectors', action='store_true',
                        help='Allow targeted sparse eigensolves for unsaved N=5000 modes.')
    args = parser.parse_args()
    ROOT, OUT = args.archive_root, args.output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    n10_root = ROOT / "outputs/linear_pa_two_part_20260924_v3/cases/N_10000/alpha_1p05"
    table10 = pd.read_csv(n10_root / "tables/full_spectrum_localization.csv.gz")
    values10 = table10["energy"].to_numpy(float)
    keep10 = (~np.isclose(values10, 0.0, atol=1e-8, rtol=0.0)) & (~table10["perron"].to_numpy(bool))
    edges10 = np.histogram_bin_edges(values10[keep10], bins="fd")
    z10 = int(np.searchsorted(edges10, 0.0, side="right") - 1)
    indices10 = np.flatnonzero(keep10 & (values10 >= edges10[z10]) & (values10 < edges10[z10 + 1]))
    vectors10_mm = np.load(n10_root / "arrays/full_eigenvectors_ascending.npy", mmap_mode="r")
    vectors10 = {int(i): np.asarray(vectors10_mm[:, int(i)], dtype=float).copy() for i in indices10}
    adjacency10, degrees10 = adjacency_from_npz(n10_root / "arrays/graph_edges.npz")
    s10, t10, i10 = analyze_modes("N=10000", values10, vectors10, indices10, adjacency10, degrees10)

    n5_root = ROOT / "outputs/preferential_attachment/single_graph_edge_bulk/affine_alpha_1p05"
    table5 = pd.read_csv(n5_root / "spectrum/single_graph_eigenpairs.csv.gz")
    values5 = table5["eigenvalue_energy"].to_numpy(float)
    keep5 = (~table5["is_zero_atom"].to_numpy(bool)) & (~table5["is_perron"].to_numpy(bool))
    edges5 = np.histogram_bin_edges(values5[keep5], bins="fd")
    z5 = int(np.searchsorted(edges5, 0.0, side="right") - 1)
    indices5 = np.flatnonzero(keep5 & (values5 >= edges5[z5]) & (values5 < edges5[z5 + 1]))
    archive5 = np.load(n5_root / "tables/selected_eigenvectors.npz")
    lookup5 = {int(index): pos for pos, index in enumerate(archive5["global_indices"])}
    adjacency5, degrees5 = adjacency_from_npz(n5_root / "tables/graph_realization.npz")
    vectors5 = {
        int(i): np.asarray(archive5["eigenvectors"][:, lookup5[int(i)]], dtype=float)
        for i in indices5
        if int(i) in lookup5
    }
    for index in (int(i) for i in indices5 if int(i) not in lookup5):
        if not args.recover_missing_vectors:
            raise RuntimeError('Missing saved eigenvector; opt in with --recover-missing-vectors')
        target = float(values5[index])
        solved_values, solved_vectors = eigsh(
            adjacency5, k=3, sigma=target, which="LM", tol=1e-12, maxiter=20000
        )
        position = int(np.argmin(np.abs(solved_values - target)))
        solved_value = float(solved_values[position])
        vector = np.asarray(solved_vectors[:, position], dtype=float)
        vector /= np.linalg.norm(vector)
        solved_nipr = len(vector) * float(np.sum(vector**4))
        expected_nipr = float(table5.iloc[index]["NIPR"])
        residual = float(np.linalg.norm(adjacency5 @ vector - target * vector))
        if abs(solved_value - target) > 2e-10:
            raise RuntimeError(f"targeted solve found {solved_value}, expected {target}")
        if not np.isclose(solved_nipr, expected_nipr, rtol=2e-7, atol=2e-6):
            raise RuntimeError(f"targeted solve NIPR {solved_nipr}, expected {expected_nipr}")
        if residual > 2e-9:
            raise RuntimeError(f"targeted solve residual {residual}")
        vectors5[index] = vector
    s5, t5, i5 = analyze_modes("N=5000", values5, vectors5, indices5, adjacency5, degrees5)

    summaries = pd.concat([s5, s10], ignore_index=True)
    top = pd.concat([t5, t10], ignore_index=True)
    induced = pd.concat([i5, i10], ignore_index=True)
    summaries.to_csv(OUT / "zero_crossing_mode_diagnostics.csv", index=False, float_format="%.15g")
    top.to_csv(OUT / "zero_crossing_top20_vertices.csv", index=False, float_format="%.15g")
    induced.to_csv(OUT / "zero_crossing_induced_subgraph_metrics.csv", index=False, float_format="%.15g")

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.0))
    markers = {"N=5000": "o", "N=10000": "s"}
    colors = {"N=5000": "#277da1", "N=10000": "#d62828"}
    for case, rows in summaries.groupby("case"):
        axes[0, 0].scatter(rows["energy"], rows["n_times_ipr"], s=45, marker=markers[case], color=colors[case], alpha=0.80, label=case)
        axes[0, 1].scatter(rows["energy"], rows["top10_mass"], s=45, marker=markers[case], color=colors[case], alpha=0.80, label=case)
        axes[1, 0].scatter(rows["energy"], rows["dominant_degree"], s=45, marker=markers[case], color=colors[case], alpha=0.80, label=case)
        axes[1, 1].scatter(rows["energy"], rows["top10_best_neighbor_cosine"], s=45, marker=markers[case], color=colors[case], alpha=0.80, label=case)
    axes[0, 0].set_ylabel("N×IPR")
    axes[0, 0].set_title("Localization of zero-crossing modes")
    axes[0, 1].set_ylabel("mass on top 10 vertices")
    axes[0, 1].set_title("Top-10 mass concentration")
    axes[1, 0].set_ylabel("dominant-vertex degree")
    axes[1, 0].set_title("Degree of the largest-mass vertex")
    axes[1, 1].set_ylabel("max neighbor-vector cosine among top 10")
    axes[1, 1].set_title("Near-twin neighborhood similarity")
    for ax in axes.flat:
        ax.set_xlabel(r"energy $E=\lambda$")
        ax.grid(alpha=0.18)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Near-zero nonzero eigenvectors: direct structural diagnostics")
    fig.tight_layout()
    fig.savefig(OUT / "zero-crossing-mode-structure-summary.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "zero-crossing-mode-structure-summary.pdf", bbox_inches="tight")
    plt.close(fig)

    rep5 = int(s5.loc[s5["n_times_ipr"].idxmax(), "ascending_index_0based"])
    rep10 = int(s10.loc[s10["n_times_ipr"].idxmax(), "ascending_index_0based"])
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.8))
    representative_graph(axes[0], "N=5000", rep5, float(values5[rep5]), vectors5[rep5], adjacency5, degrees5)
    representative_graph(axes[1], "N=10000", rep10, float(values10[rep10]), vectors10[rep10], adjacency10, degrees10)
    fig.suptitle("Most localized mode in each zero-crossing bin: actual top-mass induced subgraph")
    fig.tight_layout()
    fig.savefig(OUT / "zero-crossing-representative-induced-subgraphs.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "zero-crossing-representative-induced-subgraphs.pdf", bbox_inches="tight")
    plt.close(fig)

    report = {
        "n5000_zero_bin": [float(edges5[z5]), float(edges5[z5 + 1])],
        "n5000_mode_count": int(len(indices5)),
        "n10000_zero_bin": [float(edges10[z10]), float(edges10[z10 + 1])],
        "n10000_mode_count": int(len(indices10)),
        "representative_indices": {"N=5000": rep5, "N=10000": rep10},
    }
    (OUT / "zero_crossing_analysis_manifest.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
