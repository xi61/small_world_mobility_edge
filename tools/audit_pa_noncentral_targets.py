#!/usr/bin/env python3
"""Reproduce and diagnose the saved PA noncentral spectral/localization targets.

This is an audit-only analysis.  It reuses saved graph, spectrum, scalar, and
eigenvector artifacts; it never regenerates a graph or runs an eigensolver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import zipfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy.sparse import csgraph


ANALYSIS_VERSION = "pa_noncentral_diagnosis_v1"
ZERO_ATOL = 1e-8
COMPACTNESS_CAP = 0.20
TOP_K = (1, 2, 5, 10, 20, 50, 100)
SUPPORT_LEVELS = (0.50, 0.90, 0.95, 0.99)
TARGET_5000_INTERVALS = {
    "negative": (-36.118699046492, -35.259191631001),
    "positive": (33.501401608270, 34.360909023760),
}
EXPECTED_10000 = {
    "negative_target": (21, -31.8325975382),
    "negative_companion": (22, -27.6423),
    "positive_companion": (9977, 27.2661),
    "positive_target": (9978, 31.6195395463),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_edge_hash(n: int, edge_u: np.ndarray, edge_v: np.ndarray) -> str:
    edges = np.column_stack(
        [np.minimum(edge_u, edge_v), np.maximum(edge_u, edge_v)]
    ).astype(np.int64, copy=False)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges = np.ascontiguousarray(edges[order], dtype=np.int64)
    digest = hashlib.sha256()
    digest.update(np.asarray([n, len(edges)], dtype=np.int64).tobytes())
    digest.update(edges.tobytes(order="C"))
    return digest.hexdigest()


def build_adjacency(
    n: int, edge_u: np.ndarray, edge_v: np.ndarray, degrees: np.ndarray
) -> tuple[sparse.csr_matrix, dict[str, object]]:
    u = np.asarray(edge_u, dtype=np.int64)
    v = np.asarray(edge_v, dtype=np.int64)
    if u.shape != v.shape or u.ndim != 1:
        raise AssertionError("edge arrays must be aligned one-dimensional arrays")
    if np.any(u == v) or np.any(u < 0) or np.any(v < 0) or np.any(u >= n) or np.any(v >= n):
        raise AssertionError("edge list contains a loop or out-of-range vertex")
    canon = np.column_stack([np.minimum(u, v), np.maximum(u, v)])
    unique = np.unique(canon, axis=0)
    if len(unique) != len(canon):
        raise AssertionError("edge list contains duplicate undirected edges")
    rows = np.concatenate([u, v])
    cols = np.concatenate([v, u])
    adjacency = sparse.csr_matrix(
        (np.ones(2 * len(u), dtype=np.float64), (rows, cols)), shape=(n, n)
    )
    adjacency.sum_duplicates()
    actual_degrees = np.asarray(adjacency.sum(axis=1)).ravel().astype(np.int64)
    if not np.array_equal(actual_degrees, np.asarray(degrees, dtype=np.int64)):
        raise AssertionError("saved degrees disagree with the reconstructed graph")
    component_count, labels = csgraph.connected_components(
        adjacency, directed=False, return_labels=True
    )
    checks = {
        "n": n,
        "edge_count": int(len(u)),
        "simple": True,
        "symmetric": bool((adjacency != adjacency.T).nnz == 0),
        "unweighted": bool(np.all(adjacency.data == 1.0)),
        "connected": bool(component_count == 1),
        "component_count": int(component_count),
        "largest_component_size": int(np.bincount(labels).max()),
        "minimum_degree": int(actual_degrees.min()),
        "maximum_degree": int(actual_degrees.max()),
    }
    if not all(checks[key] for key in ("simple", "symmetric", "unweighted", "connected")):
        raise AssertionError(f"graph structural checks failed: {checks}")
    return adjacency, checks


def bin_mask(values: np.ndarray, left: float, right: float, final: bool = False) -> np.ndarray:
    return (values >= left) & (values <= right if final else values < right)


def covering_edges(values: np.ndarray, width: float, shift_fraction: float, anchor: float) -> np.ndarray:
    """Fixed-width edges with the requested origin, extended over the full sample."""
    origin = float(anchor) + float(shift_fraction) * float(width)
    low_k = math.floor((float(values.min()) - origin) / width)
    high_k = math.ceil((float(values.max()) - origin) / width)
    edges = origin + np.arange(low_k, high_k + 1, dtype=np.float64) * width
    if edges[0] > values.min():
        edges = np.insert(edges, 0, edges[0] - width)
    if edges[-1] < values.max():
        edges = np.append(edges, edges[-1] + width)
    if not (edges[0] <= values.min() and edges[-1] >= values.max()):
        raise AssertionError("shifted histogram edges do not cover the full sample")
    return edges


def containing_bin(edges: np.ndarray, value: float) -> int:
    index = int(np.searchsorted(edges, value, side="right") - 1)
    return min(max(index, 0), len(edges) - 2)


def tie_aware_quantile_edges(values: np.ndarray, requested_bins: int) -> np.ndarray:
    """Return positive-width quantile display bins without splitting equal values."""
    data = np.sort(np.asarray(values, dtype=np.float64))
    unique, counts = np.unique(data, return_counts=True)
    if len(unique) < 2:
        raise ValueError("tie-aware bins require at least two distinct values")
    cumulative = np.cumsum(counts)
    cut_indices: list[int] = []
    for target in np.linspace(0, len(data), requested_bins + 1)[1:-1]:
        unique_index = int(np.searchsorted(cumulative, target, side="left"))
        if unique_index < len(unique) - 1:
            cut_indices.append(unique_index)
    cut_indices = sorted(set(cut_indices))
    interior = [0.5 * (unique[j] + unique[j + 1]) for j in cut_indices]
    pad_left = np.nextafter(unique[0], -np.inf)
    pad_right = np.nextafter(unique[-1], np.inf)
    edges = np.asarray([pad_left, *interior, pad_right], dtype=np.float64)
    edges = np.unique(edges)
    if len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise AssertionError("tie-aware quantile edges are not strictly increasing")
    return edges


def validate_tie_fixture() -> dict[str, object]:
    fixture = np.asarray([0, 0, 0, 1, 2, 2, 3, 4], dtype=np.float64)
    edges = tie_aware_quantile_edges(fixture, 4)
    bins = np.searchsorted(edges, fixture, side="right") - 1
    bins = np.minimum(bins, len(edges) - 2)
    for value in np.unique(fixture):
        if len(np.unique(bins[fixture == value])) != 1:
            raise AssertionError("tie-aware fixture split equal values")
    counts, _ = np.histogram(fixture, bins=edges)
    if counts.sum() != len(fixture) or np.any(np.diff(edges) <= 0):
        raise AssertionError("tie-aware fixture failed count/width checks")
    return {
        "passed": True,
        "fixture": fixture.tolist(),
        "edge_count": int(len(edges)),
        "counts": counts.tolist(),
    }


def mode_rank_fields(index: int, n: int) -> dict[str, object]:
    return {
        "ascending_index_0based": int(index),
        "ascending_rank_1based": int(index + 1),
        "descending_rank_1based": int(n - index),
        "full_spectrum_midpoint_quantile": float((index + 0.5) / n),
    }


def sparse_residual(adjacency: sparse.csr_matrix, vector: np.ndarray, energy: float) -> float:
    return float(np.linalg.norm(adjacency @ vector - energy * vector))


def bfs_distances(adjacency: sparse.csr_matrix, source: int) -> np.ndarray:
    n = adjacency.shape[0]
    distance = np.full(n, -1, dtype=np.int64)
    distance[source] = 0
    queue: deque[int] = deque([source])
    indptr, indices = adjacency.indptr, adjacency.indices
    while queue:
        vertex = queue.popleft()
        for neighbor in indices[indptr[vertex] : indptr[vertex + 1]]:
            if distance[neighbor] < 0:
                distance[neighbor] = distance[vertex] + 1
                queue.append(int(neighbor))
    return distance


def region_graph_stats(adjacency: sparse.csr_matrix, mask: np.ndarray) -> tuple[int, int]:
    vertices = np.flatnonzero(mask)
    internal_edges = int(round(float(adjacency[vertices][:, vertices].sum()) / 2.0))
    degree_sum = int(round(float(adjacency[vertices].sum())))
    boundary_edges = degree_sum - 2 * internal_edges
    return internal_edges, boundary_edges


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False, float_format="%.15g")


def plot_n5000_hist_controls(
    path: Path,
    sample: np.ndarray,
    h: float,
    production_edges: np.ndarray,
    targets: dict[str, np.ndarray],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), sharey="row")
    viewports = {"negative": (-40.5, -31.5), "positive": (30.0, 37.0)}
    settings = [(0.5, 0.0, "h/2, shift 0"), (1.0, 0.0, "h, production"), (1.0, 0.5, "h, shift h/2"), (2.0, 0.0, "2h, shift 0")]
    colors = ["#277da1", "#43aa8b", "#f8961e", "#d62828"]
    for column, side in enumerate(("negative", "positive")):
        lo, hi = viewports[side]
        for multiple, shift, label in settings:
            width = h * multiple
            edges = covering_edges(sample, width, shift, production_edges[0])
            counts, _ = np.histogram(sample, bins=edges)
            density = counts / (len(sample) * np.diff(edges))
            centers = 0.5 * (edges[:-1] + edges[1:])
            visible = (centers >= lo - 2 * width) & (centers <= hi + 2 * width)
            axes[0, column].step(centers[visible], density[visible], where="mid", lw=1.4, label=label, color=colors[settings.index((multiple, shift, label))])
        for energy in targets[side]:
            axes[0, column].axvline(energy, color="black", alpha=0.30, lw=0.8)
        axes[0, column].axvspan(*TARGET_5000_INTERVALS[side], color="#90be6d", alpha=0.12, label="production target bar")
        axes[0, column].set_xlim(lo, hi)
        axes[0, column].set_title(f"{side.capitalize()} target: width/origin controls")
        axes[0, column].set_xlabel(r"raw-adjacency energy $E=\lambda$")
        axes[0, column].grid(alpha=0.18)
        axes[0, column].legend(fontsize=8, frameon=False)

        quantile_edges = tie_aware_quantile_edges(sample, 80)
        equal_counts, _ = np.histogram(sample, bins=production_edges)
        equal_density = equal_counts / (len(sample) * np.diff(production_edges))
        q_counts, _ = np.histogram(sample, bins=quantile_edges)
        q_density = q_counts / (len(sample) * np.diff(quantile_edges))
        axes[1, column].stairs(equal_density, production_edges, label="160 equal-width", color="#43aa8b")
        axes[1, column].stairs(q_density, quantile_edges, label="80 tie-aware quantile bins", color="#6a4c93")
        equal_visible = (production_edges[:-1] < hi) & (production_edges[1:] > lo)
        quantile_visible = (quantile_edges[:-1] < hi) & (quantile_edges[1:] > lo)
        local_max = max(float(equal_density[equal_visible].max()), float(q_density[quantile_visible].max()))
        axes[1, column].set_ylim(0.0, 1.15 * local_max)
        axes[1, column].scatter(targets[side], np.zeros(len(targets[side])), marker="|", s=90, color="black", label="quartet levels")
        axes[1, column].set_xlim(lo, hi)
        axes[1, column].set_xlabel(r"raw-adjacency energy $E=\lambda$")
        axes[1, column].set_title("Display control (density = count / N / width)")
        axes[1, column].grid(alpha=0.18)
        axes[1, column].legend(fontsize=8, frameon=False)
    axes[0, 0].set_ylabel("empirical density")
    axes[1, 0].set_ylabel("empirical density")
    figure.suptitle("N=5000, seed=4011773341: noncentral edge-level histogram controls")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_n5000_rank_gap(
    path: Path, eigenvalues: np.ndarray, target_indices: dict[str, np.ndarray]
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 8.5))
    for row, side in enumerate(("negative", "positive")):
        indices = target_indices[side]
        start = max(0, int(indices.min()) - 3)
        stop = min(len(eigenvalues), int(indices.max()) + 7)
        window_indices = np.arange(start, stop)
        values = eigenvalues[window_indices]
        sorted_window = np.sort(values)
        ecdf = np.arange(1, len(sorted_window) + 1) / len(sorted_window)
        axes[row, 0].step(sorted_window, ecdf, where="post", color="#277da1")
        axes[row, 0].scatter(eigenvalues[indices], np.searchsorted(sorted_window, eigenvalues[indices], side="right") / len(sorted_window), color="#d62828", zorder=3)
        axes[row, 0].set_title(f"{side.capitalize()} zoomed ECDF")
        axes[row, 0].set_xlabel(r"energy $E=\lambda$")
        axes[row, 0].set_ylabel("window ECDF")

        axes[row, 1].plot(window_indices + 1, values, "o-", ms=3.5, lw=0.9, color="#277da1")
        axes[row, 1].scatter(indices + 1, eigenvalues[indices], color="#d62828", zorder=3, label="target quartet")
        for idx in indices:
            axes[row, 1].annotate(str(int(idx + 1)), (idx + 1, eigenvalues[idx]), xytext=(2, 5), textcoords="offset points", fontsize=7)
        axes[row, 1].set_title("Individual levels versus ascending rank")
        axes[row, 1].set_xlabel("ascending rank (1-based)")
        axes[row, 1].set_ylabel(r"energy $E=\lambda$")
        axes[row, 1].legend(fontsize=8, frameon=False)

        gaps = np.diff(values)
        left_indices = window_indices[:-1]
        axes[row, 2].stem(left_indices + 1, gaps, linefmt="#43aa8b", markerfmt="o", basefmt=" ")
        for idx in indices[:-1]:
            pos = np.flatnonzero(left_indices == idx)
            if len(pos):
                axes[row, 2].scatter(idx + 1, gaps[pos[0]], color="#d62828", zorder=3)
        axes[row, 2].set_title("Adjacent gaps (left-rank indexed)")
        axes[row, 2].set_xlabel("ascending rank of lower level")
        axes[row, 2].set_ylabel(r"$\lambda_{i+1}-\lambda_i$")
        for axis in axes[row]:
            axis.grid(alpha=0.18)
    figure.suptitle("N=5000, seed=4011773341: the two target bars resolve into distinct edge levels")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_n5000_common_edges(
    path: Path, eigenvalues: np.ndarray, nonzero: np.ndarray, nonzero_nonperron: np.ndarray
) -> None:
    edges = np.linspace(float(eigenvalues.min()), float(eigenvalues.max()), 161)
    subsets = [("all levels", eigenvalues, "#277da1"), ("zero removed", nonzero, "#f8961e"), ("zero + Perron removed", nonzero_nonperron, "#43aa8b")]
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharex=True)
    for label, values, color in subsets:
        counts, _ = np.histogram(values, bins=edges)
        width = np.diff(edges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        axes[0].step(centers, counts / (len(eigenvalues) * width), where="mid", label=f"{label} (mass={len(values)/len(eigenvalues):.3f})", color=color)
        axes[1].step(centers, counts / (len(values) * width), where="mid", label=label, color=color)
    axes[0].set_title("Original-N normalization: removed mass stays removed")
    axes[1].set_title("Conditional normalization: each retained sample integrates to 1")
    for axis in axes:
        axis.set_xlabel(r"raw-adjacency energy $E=\lambda$")
        axis.set_ylabel("empirical density")
        axis.grid(alpha=0.18)
        axis.legend(fontsize=8, frameon=False)
    figure.suptitle("N=5000 common-edge overlays (unlike the original two panels)")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def fixed_width_bins(values: np.ndarray, width: float, shift_fraction: float, anchor: float) -> np.ndarray:
    return covering_edges(values, width, shift_fraction, anchor)


def aggregate_metric(values: np.ndarray, metric: np.ndarray, edges: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(len(edges) - 1):
        inside = bin_mask(values, edges[index], edges[index + 1], index == len(edges) - 2)
        usable = inside & np.isfinite(metric)
        selected = metric[usable]
        rows.append(
            {
                "bin_index": index,
                "left": float(edges[index]),
                "right": float(edges[index + 1]),
                "center": float(0.5 * (edges[index] + edges[index + 1])),
                "eigenvalue_count": int(inside.sum()),
                "finite_ipr_count": int(usable.sum()),
                "mean_n_times_ipr": float(np.mean(selected)) if len(selected) else math.nan,
                "median_n_times_ipr": float(np.median(selected)) if len(selected) else math.nan,
            }
        )
    return rows


def plot_n10000_density_ipr(
    path: Path,
    eigenvalues: np.ndarray,
    nipr: np.ndarray,
    density0: pd.DataFrame,
    localization0: pd.DataFrame,
    selected_modes: dict[str, int],
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(16.5, 9.5))
    sides = {
        "negative": (-42.0, -18.0, range(20, 24)),
        "positive": (18.0, 42.0, range(32, 36)),
    }
    labels = {
        "negative_target": "- target",
        "negative_companion": "- companion",
        "positive_companion": "+ companion",
        "positive_target": "+ target",
    }
    for row, (side, (lo, hi, bin_indices)) in enumerate(sides.items()):
        drows = density0[density0["bin_index"].isin(bin_indices)].copy()
        lrows = localization0[localization0["energy_bin_index"].isin(bin_indices)].copy()
        width = drows["energy_right"].to_numpy() - drows["energy_left"].to_numpy()
        axes[row, 0].bar(drows["energy_center"], drows["density"], width=0.90 * width, color="#8ecae6", edgecolor="#277da1", alpha=0.85)
        window_mask = (eigenvalues >= lo) & (eigenvalues <= hi)
        for energy in eigenvalues[window_mask]:
            axes[row, 0].axvline(energy, color="black", lw=0.65, alpha=0.45)
        for _, record in drows.iterrows():
            axes[row, 0].annotate(f"n={int(record['count'])}", (record["energy_center"], record["density"]), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
        axes[row, 0].set_xlim(lo, hi)
        axes[row, 0].set_title(f"{side.capitalize()}: fixed original-bin density + levels")
        axes[row, 0].set_ylabel("density = count / (10000 width)")

        axes[row, 1].scatter(eigenvalues[window_mask], nipr[window_mask], s=26, color="#6c757d", alpha=0.75, label="individual simple/usable modes")
        for name, index in selected_modes.items():
            if name.startswith(side):
                axes[row, 1].scatter(eigenvalues[index], nipr[index], s=64, color="#d62828", zorder=3)
                axes[row, 1].annotate(f"{labels[name]}\ni={index}, E={eigenvalues[index]:.3f}", (eigenvalues[index], nipr[index]), xytext=(5, 5), textcoords="offset points", fontsize=8)
        axes[row, 1].set_xlim(lo, hi)
        axes[row, 1].set_title("Individual energy versus N×IPR")
        axes[row, 1].set_ylabel("N×IPR")

        axes[row, 2].plot(lrows["energy_center"], lrows["mean_n_times_ipr"], "o-", label="bin mean", color="#d62828")
        axes[row, 2].plot(lrows["energy_center"], lrows["median_n_times_ipr"], "s--", label="bin median", color="#6a4c93")
        joined = lrows.merge(drows[["bin_index", "count"]], left_on="energy_bin_index", right_on="bin_index")
        for _, record in joined.iterrows():
            y = record["mean_n_times_ipr"]
            axes[row, 2].annotate(
                f"levels={int(record['count'])}\nfinite={int(record['usable_count'])}",
                (record["energy_center"], y),
                xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7,
            )
        axes[row, 2].set_xlim(lo, hi)
        axes[row, 2].set_title("Original-bin localization summaries")
        axes[row, 2].set_ylabel("N×IPR")
        axes[row, 2].legend(fontsize=8, frameon=False)
        for column in range(3):
            axes[row, column].set_xlabel(r"raw-adjacency energy $E=\lambda$")
            axes[row, column].grid(alpha=0.18)
    figure.suptitle("N=10000, seed=2748869207, a=0: density and localization on the same graph")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_localization_sensitivity(
    path: Path,
    sensitivity: pd.DataFrame,
    eigenvalues: np.ndarray,
    selected_modes: dict[str, int],
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14.5, 5.5), sharey=True)
    representative = [
        ("shared_disorder_range", 0.5, 0.0),
        ("shared_disorder_range", 1.0, 0.0),
        ("shared_disorder_range", 1.0, 0.5),
        ("shared_disorder_range", 2.0, 0.0),
        ("clean_spectrum_range_60", 1.0, 0.0),
    ]
    colors = ["#277da1", "#43aa8b", "#f8961e", "#d62828", "#6a4c93"]
    for axis, side, xlim in zip(axes, ("negative", "positive"), ((-42, -18), (18, 42))):
        for setting, color in zip(representative, colors):
            basis, multiple, shift = setting
            rows = sensitivity[
                (sensitivity["record_type"] == "bin")
                & (sensitivity["side"] == side)
                & (sensitivity["edge_basis"] == basis)
                & np.isclose(sensitivity["width_multiple"], multiple)
                & np.isclose(sensitivity["origin_shift_fraction"], shift)
            ].sort_values("bin_center")
            rows = rows[np.isfinite(rows["mean_n_times_ipr"])]
            label = f"{basis.replace('_', ' ')}, {multiple:g}w, shift {shift:g}w"
            axis.plot(rows["bin_center"], rows["mean_n_times_ipr"], "o-", ms=3.5, lw=1.0, label=label, color=color)
        for name, index in selected_modes.items():
            if name.startswith(side):
                axis.axvline(eigenvalues[index], color="black", lw=0.8, ls="--", alpha=0.55)
                axis.annotate(f"i={index}\n{eigenvalues[index]:.3f}", (eigenvalues[index], axis.get_ylim()[1] if axis.get_ylim()[1] > 1 else 2500), xytext=(2, -25), textcoords="offset points", fontsize=7, rotation=90, va="top")
        axis.set_xlim(*xlim)
        axis.set_xlabel(r"localization-bin center / mode energy $E=\lambda$")
        axis.set_title(f"{side.capitalize()} window")
        axis.grid(alpha=0.18)
        axis.legend(fontsize=7, frameon=False)
    axes[0].set_ylabel("mean N×IPR (empty bins omitted)")
    figure.suptitle("N=10000 localization-bin sensitivity; individual mode energies remain fixed")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_spatial_profiles(
    path: Path,
    shell_frame: pd.DataFrame,
    region_frame: pd.DataFrame,
    pair_frame: pd.DataFrame,
) -> None:
    targets = ["negative_target", "positive_target", "negative_companion", "positive_companion"]
    figure, axes = plt.subplots(2, 2, figsize=(14.5, 10.0))
    ax = axes[0, 0]
    for name in targets:
        rows = shell_frame[(shell_frame["record_type"] == "shell") & (shell_frame["mode_name"] == name)].sort_values("graph_distance")
        ax.plot(rows["graph_distance"], rows["shell_mass"], "o-", label=name.replace("_", " "))
    ax.set_yscale("log")
    ax.set_xlabel("graph distance from dominant vertex")
    ax.set_ylabel("squared mass in shell")
    ax.set_title("Graph-shell mass profiles")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.18)

    ax = axes[0, 1]
    for name in targets:
        rows = region_frame[(region_frame["mode_name"] == name) & region_frame["region_name"].isin(["S90", "S95", "S99"])].sort_values("captured_mass")
        ax.plot(rows["captured_mass"], rows["full_residual"], "o-", label=name.replace("_", " "))
    ax.set_yscale("log")
    ax.set_xlabel("captured squared mass")
    ax.set_ylabel(r"$\|Au-\lambda u\|_2$")
    ax.set_title("Truncated-support residual versus captured mass")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.18)

    ax = axes[1, 0]
    radius_order = {"radius0": 0, "radius1": 1, "radius2": 2}
    for name in targets:
        rows = region_frame[(region_frame["mode_name"] == name) & region_frame["region_name"].isin(radius_order)].copy()
        rows["radius"] = rows["region_name"].map(radius_order)
        rows = rows.sort_values("radius")
        ax.plot(rows["region_fraction"], rows["captured_mass"], "o-", label=name.replace("_", " "))
        for _, record in rows.iterrows():
            ax.annotate(record["region_name"].replace("radius", "r="), (record["region_fraction"], record["captured_mass"]), xytext=(2, 3), textcoords="offset points", fontsize=7)
    ax.axvline(COMPACTNESS_CAP, color="black", ls="--", lw=1.0, label="20% reporting cap")
    ax.set_xlabel("region size / N")
    ax.set_ylabel("captured squared mass")
    ax.set_title("Radius-neighborhood coverage (no automatic growth)")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.18)

    ax = axes[1, 1]
    x = np.arange(len(pair_frame))
    width = 0.34
    ax.bar(x - width / 2, pair_frame["squared_mass_overlap"], width, label=r"$\sum_v \min(p_v,q_v)$", color="#43aa8b")
    ax.bar(x + width / 2, pair_frame["squared_mass_cosine"], width, label="squared-mass cosine", color="#277da1")
    ax.set_xticks(x, pair_frame["pair_name"].str.replace("_", " "))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("spatial-profile agreement")
    ax.set_title("Cross-sign squared-mass pairing")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.18, axis="y")
    figure.suptitle("N=10000 saved-vector structural pilot: support, shells, leakage, and pairing")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def analyze(args: argparse.Namespace) -> None:
    started = utc_now()
    root = args.root.resolve()
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output directory already exists: {output}")
    output.mkdir(parents=True, exist_ok=args.overwrite)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)

    n5_root = root / "outputs/preferential_attachment/single_graph_edge_bulk/affine_alpha_1p05"
    n10_root = root / "outputs/linear_pa_two_part_20260924_v3/cases/N_10000/alpha_1p05"
    report_tables = root / "outputs/pa_diagonal_uniform_noise_20260924_v1/cases/N_10000/alpha_1p05/report/tables"
    readiness = root / "audit_outputs/pa_bulk_peak_readiness_20261004"
    paths = {
        "n5000_graph": n5_root / "tables/graph_realization.npz",
        "n5000_spectral": n5_root / "spectrum/single_graph_spectral_arrays.npz",
        "n5000_eigenpairs": n5_root / "spectrum/single_graph_eigenpairs.csv.gz",
        "n5000_vectors": n5_root / "tables/selected_eigenvectors.npz",
        "n5000_residual_audit": n5_root / "tables/eigenpair_residual_audit.csv",
        "n5000_manifest": n5_root / "run_manifest.json",
        "n5000_graph_metadata": n5_root / "tables/graph_metadata.json",
        "n10000_graph": n10_root / "arrays/graph_edges.npz",
        "n10000_values": n10_root / "arrays/full_eigenvalues_ascending.npy",
        "n10000_vectors": n10_root / "arrays/full_eigenvectors_ascending.npy",
        "n10000_abs_residuals": n10_root / "arrays/full_absolute_residuals.npy",
        "n10000_rel_residuals": n10_root / "arrays/full_relative_residuals.npy",
        "n10000_localization": n10_root / "tables/full_spectrum_localization.csv.gz",
        "n10000_graph_manifest": n10_root / "graph_cache_manifest.json",
        "n10000_spectral_manifest": n10_root / "spectral_cache_manifest.json",
        "n10000_part2_manifest": n10_root / "part2/manifest.json",
        "n10000_part2_targets": n10_root / "part2/tables/target_summary.csv",
        "report_localization": report_tables / "energy_binned_localization.csv.gz",
        "report_density": report_tables / "spectral_density.csv.gz",
        "report_edges": report_tables / "common_energy_bin_edges.csv",
        "report_manifest": report_tables.parent / "run_manifest.json",
        "report_frozen_source": root / "outputs/pa_diagonal_uniform_noise_20260924_v1/execution_project/src/small_world_mobility_edge/random_schrodinger/pa_diagonal_uniform_noise_reporting.py",
        "report_frozen_config": root / "outputs/pa_diagonal_uniform_noise_20260924_v1/execution_project/config/pa_diagonal_uniform_noise.json",
        "n5000_density_source": root / "src/small_world_mobility_edge/preferential_attachment/single_graph_edge_bulk.py",
        "readiness_report": readiness / "PA_BULK_PEAK_READINESS_REPORT.md",
        "readiness_n5000_targets": readiness / "n5000_target_modes.csv",
        "readiness_n5000_sensitivity": readiness / "n5000_bin_sensitivity.csv",
        "readiness_n10000_windows": readiness / "n10000_target_windows.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required inputs: " + "; ".join(missing))

    with paths["n5000_manifest"].open() as handle:
        n5_manifest = json.load(handle)
    with paths["n5000_graph_metadata"].open() as handle:
        n5_metadata = json.load(handle)
    with paths["n10000_graph_manifest"].open() as handle:
        n10_graph_manifest = json.load(handle)
    with paths["n10000_spectral_manifest"].open() as handle:
        n10_spectral_manifest = json.load(handle)
    with paths["report_manifest"].open() as handle:
        report_manifest = json.load(handle)

    # --- N=5000 graph and scalar spectrum ---
    with np.load(paths["n5000_graph"], allow_pickle=False) as archive:
        n5 = int(archive["n"])
        n5_u = archive["edge_u"].copy()
        n5_v = archive["edge_v"].copy()
        n5_degrees = archive["degrees"].copy()
        n5_seed = int(archive["graph_seed"])
        n5_m = int(archive["m"])
    a5, graph5_checks = build_adjacency(n5, n5_u, n5_v, n5_degrees)
    n5_canonical_hash = canonical_edge_hash(n5, n5_u, n5_v)
    with np.load(paths["n5000_spectral"], allow_pickle=False) as archive:
        values5 = archive["eigenvalues"].astype(np.float64, copy=True)
        ipr5 = archive["ipr"].astype(np.float64, copy=True)
        nipr5 = archive["n_times_ipr"].astype(np.float64, copy=True)
    if values5.shape != (n5,) or not np.all(np.isfinite(values5)) or np.any(np.diff(values5) < 0):
        raise AssertionError("N=5000 eigenvalues fail shape/finite/sorting checks")
    if not np.allclose(nipr5, n5 * ipr5, rtol=2e-12, atol=1e-10, equal_nan=True):
        raise AssertionError("N=5000 IPR and N*IPR are inconsistent")
    zero5 = np.isclose(values5, 0.0, atol=ZERO_ATOL, rtol=0.0)
    nonzero5 = values5[~zero5]
    continuous5_mask = ~zero5
    continuous5_mask[-1] = False
    continuous5 = values5[continuous5_mask]
    if (len(values5), int(zero5.sum()), len(nonzero5), len(continuous5)) != (5000, 1844, 3156, 3155):
        raise AssertionError("N=5000 source accounting does not match 5000/1844/3156/3155")
    production_counts, production_edges = np.histogram(continuous5, bins=160)
    h = float(production_edges[1] - production_edges[0])
    if not math.isclose(h, 0.85950741549, rel_tol=0.0, abs_tol=2e-11):
        raise AssertionError(f"unexpected production histogram width {h}")
    if production_counts.sum() != len(continuous5):
        raise AssertionError("production histogram dropped observations")

    residual5 = pd.read_csv(paths["n5000_residual_audit"])
    residual5 = residual5.set_index("global_index_0based")
    target5_indices: dict[str, np.ndarray] = {}
    target5_values: dict[str, np.ndarray] = {}
    target_map_rows: list[dict[str, object]] = []
    quartet_rows: list[dict[str, object]] = []
    for side, (left, right) in TARGET_5000_INTERVALS.items():
        indices = np.flatnonzero(bin_mask(values5, left, right))
        if len(indices) != 4:
            raise AssertionError(f"N=5000 {side} production target does not contain four levels")
        target5_indices[side] = indices
        target5_values[side] = values5[indices]
        internal = np.diff(values5[indices])
        for local, index in enumerate(indices):
            residual_row = residual5.loc[index] if index in residual5.index else None
            continuous_rank = int(np.searchsorted(continuous5, values5[index], side="left") + 1)
            target_map_rows.append(
                {
                    "graph_id": "N5000_seed4011773341",
                    "case": "n5000_density",
                    "mode_name": f"{side}_quartet_{local + 1}",
                    "eigenpair_id": int(index),
                    "exact_energy": float(values5[index]),
                    "target_interval_left": left,
                    "target_interval_right": right,
                    "in_target_bin": True,
                    **mode_rank_fields(int(index), n5),
                    "filtered_rank_1based": continuous_rank,
                    "filtered_sample_size": len(continuous5),
                    "ipr": float(ipr5[index]),
                    "n_times_ipr": float(nipr5[index]),
                    "absolute_residual": float(residual_row["absolute_residual_norm"]) if residual_row is not None else math.nan,
                    "relative_residual": float(residual_row["relative_residual_norm"]) if residual_row is not None else math.nan,
                    "vector_available": bool(side == "positive"),
                    "vector_source": "selected_eigenvectors.npz" if side == "positive" else "not saved",
                }
            )
            quartet_rows.append(
                {
                    "side": side,
                    "quartet_position": local + 1,
                    "ascending_index_0based": int(index),
                    "energy": float(values5[index]),
                    "gap_from_previous_quartet_level": float(internal[local - 1]) if local > 0 else math.nan,
                    "absolute_residual": float(residual_row["absolute_residual_norm"]) if residual_row is not None else math.nan,
                    "relative_residual": float(residual_row["relative_residual_norm"]) if residual_row is not None else math.nan,
                }
            )
        quartet_rows.append(
            {
                "side": side,
                "quartet_position": "summary",
                "ascending_index_0based": math.nan,
                "energy": math.nan,
                "gap_from_previous_quartet_level": math.nan,
                "diameter": float(values5[indices[-1]] - values5[indices[0]]),
                "left_external_gap": float(values5[indices[0]] - values5[indices[0] - 1]),
                "right_external_gap": float(values5[indices[-1] + 1] - values5[indices[-1]]),
                "minimum_internal_gap": float(internal.min()),
                "maximum_internal_gap": float(internal.max()),
            }
        )

    with np.load(paths["n5000_vectors"], allow_pickle=False) as archive:
        selected5_indices = archive["global_indices"]
        selected5_values = archive["eigenvalues"]
        selected5_vectors = archive["eigenvectors"]
    if selected5_vectors.shape[0] != n5 or not np.allclose(values5[selected5_indices], selected5_values, rtol=0.0, atol=5e-13):
        raise AssertionError("N=5000 selected vector IDs/energies are misaligned")
    for index in target5_indices["positive"]:
        where = np.flatnonzero(selected5_indices == index)
        if len(where) != 1:
            raise AssertionError("positive N=5000 target vector is not uniquely saved")
        vector = selected5_vectors[:, where[0]]
        if not math.isclose(float(np.linalg.norm(vector)), 1.0, rel_tol=0.0, abs_tol=3e-12):
            raise AssertionError("saved N=5000 target vector is not normalized")
        computed = sparse_residual(a5, vector, values5[index])
        if index in residual5.index and not math.isclose(computed, float(residual5.loc[index, "absolute_residual_norm"]), rel_tol=0.15, abs_tol=2e-14):
            raise AssertionError("recomputed N=5000 residual disagrees with saved audit")

    fixed_interval_rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []
    for side, (left, right) in TARGET_5000_INTERVALS.items():
        prod_bin = containing_bin(production_edges, 0.5 * (left + right))
        if not (math.isclose(production_edges[prod_bin], left, abs_tol=2e-12) and math.isclose(production_edges[prod_bin + 1], right, abs_tol=2e-12)):
            raise AssertionError("stored target interval does not match recovered production edges")
        intervals = {
            "production_target_bin": (left, right),
            "three_production_bin_window": (production_edges[prod_bin - 1], production_edges[prod_bin + 2]),
        }
        for interval_name, (fixed_left, fixed_right) in intervals.items():
            count = int(bin_mask(continuous5, fixed_left, fixed_right).sum())
            width = float(fixed_right - fixed_left)
            fixed_interval_rows.append(
                {
                    "case": "n5000_density",
                    "side": side,
                    "interval_name": interval_name,
                    "interval_left": fixed_left,
                    "interval_right": fixed_right,
                    "count": count,
                    "sample_size": len(continuous5),
                    "bin_width": width,
                    "empirical_mass": count / len(continuous5),
                    "density": count / (len(continuous5) * width),
                    "invariant_under_rebinning": True,
                }
            )
        viewport = (-40.5, -31.5) if side == "negative" else (30.0, 37.0)
        reference = 0.5 * (left + right)
        for multiple in (0.5, 1.0, 2.0):
            width = h * multiple
            for shift in (0.0, 0.25, 0.5, 0.75):
                edges = covering_edges(continuous5, width, shift, production_edges[0])
                counts, _ = np.histogram(continuous5, bins=edges)
                if counts.sum() != len(continuous5):
                    raise AssertionError("shifted histogram silently changed the sample size")
                densities = counts / (len(continuous5) * np.diff(edges))
                selected_index = containing_bin(edges, reference)
                target_members = int(bin_mask(target5_values[side], edges[selected_index], edges[selected_index + 1], selected_index == len(edges) - 2).sum())
                centers = 0.5 * (edges[:-1] + edges[1:])
                region = (centers >= viewport[0]) & (centers <= viewport[1])
                region_indices = np.flatnonzero(region)
                peak_index = int(region_indices[np.argmax(densities[region_indices])])
                sensitivity_rows.append(
                    {
                        "case": "n5000_density",
                        "side": side,
                        "width_multiple": multiple,
                        "origin_shift_fraction": shift,
                        "width": width,
                        "selected_bin_left": float(edges[selected_index]),
                        "selected_bin_right": float(edges[selected_index + 1]),
                        "selected_bin_center": float(centers[selected_index]),
                        "selected_bin_count": int(counts[selected_index]),
                        "selected_bin_density": float(densities[selected_index]),
                        "original_quartet_members_in_selected_bin": target_members,
                        "regional_peak_bin_left": float(edges[peak_index]),
                        "regional_peak_bin_right": float(edges[peak_index + 1]),
                        "regional_peak_center": float(centers[peak_index]),
                        "regional_peak_count": int(counts[peak_index]),
                        "regional_peak_density": float(densities[peak_index]),
                        "sample_size": len(continuous5),
                        "histogram_total_count": int(counts.sum()),
                        "density_integral": float(np.sum(densities * np.diff(edges))),
                        "viewport_left": viewport[0],
                        "viewport_right": viewport[1],
                    }
                )
    if not all(math.isclose(row["density_integral"], 1.0, abs_tol=2e-13) for row in sensitivity_rows):
        raise AssertionError("shifted histogram normalization failed")

    quantile_edges = tie_aware_quantile_edges(continuous5, 80)
    quantile_counts, _ = np.histogram(continuous5, bins=quantile_edges)
    quantile_rows = [
        {
            "bin_index": index,
            "left": quantile_edges[index],
            "right": quantile_edges[index + 1],
            "width": quantile_edges[index + 1] - quantile_edges[index],
            "count": int(quantile_counts[index]),
            "density": quantile_counts[index] / (len(continuous5) * (quantile_edges[index + 1] - quantile_edges[index])),
            "sample_size": len(continuous5),
        }
        for index in range(len(quantile_counts))
    ]
    tie_test = validate_tie_fixture()

    plot_n5000_hist_controls(figures / "n5000_spectral_zoom_controls.png", continuous5, h, production_edges, target5_values)
    plot_n5000_rank_gap(figures / "n5000_ecdf_rank_gap_zoom.png", values5, target5_indices)
    plot_n5000_common_edges(figures / "n5000_common_edge_overlays.png", values5, nonzero5, continuous5)

    # --- N=10000 graph, scalar spectrum, and report bins ---
    graph10_archive = np.load(paths["n10000_graph"], allow_pickle=False)
    n10 = int(graph10_archive["n"])
    n10_u = graph10_archive["edge_u"].copy()
    n10_v = graph10_archive["edge_v"].copy()
    n10_degrees = graph10_archive["degrees"].copy()
    original_ids = graph10_archive["original_vertex_ids"].copy()
    metadata10 = json.loads(str(graph10_archive["metadata_json"]))
    graph10_archive.close()
    if not np.array_equal(original_ids, np.arange(n10)):
        raise AssertionError("N=10000 original vertex IDs are not the identity map")
    a10, graph10_checks = build_adjacency(n10, n10_u, n10_v, n10_degrees)
    n10_canonical_hash = canonical_edge_hash(n10, n10_u, n10_v)
    if n10_canonical_hash != n10_graph_manifest["graph_sha256_edges"]:
        raise AssertionError("N=10000 canonical graph-content hash mismatch")
    values10 = np.load(paths["n10000_values"], mmap_mode="r")
    vectors10 = np.load(paths["n10000_vectors"], mmap_mode="r")
    abs_residuals10 = np.load(paths["n10000_abs_residuals"], mmap_mode="r")
    rel_residuals10 = np.load(paths["n10000_rel_residuals"], mmap_mode="r")
    if values10.shape != (n10,) or vectors10.shape != (n10, n10) or not vectors10.flags["F_CONTIGUOUS"]:
        raise AssertionError("N=10000 eigensystem shapes/layout are unexpected")
    if not np.all(np.isfinite(values10)) or np.any(np.diff(values10) < 0):
        raise AssertionError("N=10000 eigenvalues fail finite/sorting checks")
    localization10 = pd.read_csv(paths["n10000_localization"])
    if not np.array_equal(localization10["ascending_index"].to_numpy(), np.arange(n10)):
        raise AssertionError("N=10000 localization table indices are misaligned")
    if not np.allclose(localization10["energy"].to_numpy(), values10, rtol=0.0, atol=5e-12):
        raise AssertionError("N=10000 localization energies are misaligned")
    nipr10 = localization10["n_times_ipr"].to_numpy(dtype=np.float64)
    ipr10 = localization10["ipr"].to_numpy(dtype=np.float64)
    repeated_basis_dependent = np.zeros(n10, dtype=bool)
    repeated_links = np.isclose(np.diff(values10), 0.0, atol=ZERO_ATOL, rtol=0.0)
    repeated_basis_dependent[:-1] |= repeated_links
    repeated_basis_dependent[1:] |= repeated_links
    nipr10_reporting = nipr10.copy()
    nipr10_reporting[repeated_basis_dependent] = np.nan
    finite_ipr = np.isfinite(ipr10)
    if not np.allclose(nipr10[finite_ipr], n10 * ipr10[finite_ipr], rtol=3e-11, atol=2e-8):
        raise AssertionError("N=10000 IPR and N*IPR are inconsistent")

    common_edges_table = pd.read_csv(paths["report_edges"])
    common_edges = np.r_[common_edges_table["energy_left"].iloc[0], common_edges_table["energy_right"].to_numpy()]
    if len(common_edges) != 61 or np.any(np.diff(common_edges) <= 0):
        raise AssertionError("common report edges are invalid")
    common_width = float(common_edges[1] - common_edges[0])
    density = pd.read_csv(paths["report_density"])
    binned_localization = pd.read_csv(paths["report_localization"])
    density0 = density[np.isclose(density["raw_amplitude"], 0.0)].copy()
    localization0 = binned_localization[np.isclose(binned_localization["raw_amplitude"], 0.0)].copy()
    recovered_counts, _ = np.histogram(values10, bins=common_edges)
    if not np.array_equal(recovered_counts, density0.sort_values("bin_index")["count"].to_numpy()):
        raise AssertionError("a=0 density rows do not reproduce from the clean spectrum")
    recovered_density = recovered_counts / (n10 * np.diff(common_edges))
    if not np.allclose(recovered_density, density0.sort_values("bin_index")["density"].to_numpy(), rtol=2e-12, atol=1e-15):
        raise AssertionError("a=0 density normalization mismatch")
    recovered_agg = pd.DataFrame(aggregate_metric(np.asarray(values10), nipr10_reporting, common_edges))
    if not np.array_equal(recovered_agg["finite_ipr_count"].to_numpy(), localization0.sort_values("energy_bin_index")["usable_count"].to_numpy()):
        raise AssertionError("a=0 finite-IPR contributor counts do not reproduce")
    expected_means = localization0.sort_values("energy_bin_index")["mean_n_times_ipr"].to_numpy()
    if not np.allclose(recovered_agg["mean_n_times_ipr"].to_numpy(), expected_means, rtol=2e-11, atol=2e-8, equal_nan=True):
        raise AssertionError("a=0 localization means do not reproduce")

    selected_modes: dict[str, int] = {}
    for name, (index, cue) in EXPECTED_10000.items():
        if not math.isclose(float(values10[index]), cue, rel_tol=0.0, abs_tol=2e-4):
            raise AssertionError(f"N=10000 expected mode {name} index/energy mismatch")
        selected_modes[name] = index
    if not (common_edges[21] <= values10[21] < common_edges[22] and common_edges[34] <= values10[9978] < common_edges[35]):
        raise AssertionError("N=10000 target mode bin membership mismatch")
    for name, index in selected_modes.items():
        bin_index = containing_bin(common_edges, float(values10[index]))
        target_map_rows.append(
            {
                "graph_id": "N10000_seed2748869207",
                "case": "n10000_localization",
                "mode_name": name,
                "eigenpair_id": index,
                "exact_energy": float(values10[index]),
                "target_interval_left": float(common_edges[bin_index]),
                "target_interval_right": float(common_edges[bin_index + 1]),
                "in_target_bin": True,
                **mode_rank_fields(index, n10),
                "filtered_rank_1based": math.nan,
                "filtered_sample_size": math.nan,
                "ipr": float(ipr10[index]),
                "n_times_ipr": float(nipr10[index]),
                "absolute_residual": float(abs_residuals10[index]),
                "relative_residual": float(rel_residuals10[index]),
                "vector_available": True,
                "vector_source": "full_eigenvectors_ascending.npy (memory-mapped column)",
            }
        )

    target_bins = {"negative": [20, 21, 22, 23], "positive": [32, 33, 34, 35]}
    window_mode_rows: list[dict[str, object]] = []
    for side, indices in target_bins.items():
        for bin_index in indices:
            inside = bin_mask(np.asarray(values10), common_edges[bin_index], common_edges[bin_index + 1], bin_index == len(common_edges) - 2)
            for index in np.flatnonzero(inside):
                window_mode_rows.append(
                    {
                        "side": side,
                        "production_bin_index": bin_index,
                        "ascending_index_0based": int(index),
                        "energy": float(values10[index]),
                        "n_times_ipr": float(nipr10[index]) if np.isfinite(nipr10[index]) else math.nan,
                        "finite_ipr": bool(np.isfinite(nipr10[index])),
                        "simple_at_1e-8": bool((index == 0 or abs(values10[index] - values10[index - 1]) > ZERO_ATOL) and (index == n10 - 1 or abs(values10[index + 1] - values10[index]) > ZERO_ATOL)),
                    }
                )

    # Four deterministic lower-IPR comparison modes: minimum finite NIPR in the
    # nearest occupied outer/inner flanking production bin on each sign side.
    comparison_bins = {
        "negative_outer_comparison": 20,
        "negative_inner_comparison": 23,
        "positive_inner_comparison": 32,
        "positive_outer_comparison": 35,
    }
    comparison_rule_rows: list[dict[str, object]] = []
    for name, bin_index in comparison_bins.items():
        inside = bin_mask(np.asarray(values10), common_edges[bin_index], common_edges[bin_index + 1], bin_index == len(common_edges) - 2)
        candidates = np.flatnonzero(inside & np.isfinite(nipr10))
        candidates = np.asarray([i for i in candidates if (i == 0 or abs(values10[i] - values10[i - 1]) > ZERO_ATOL) and (i == n10 - 1 or abs(values10[i + 1] - values10[i]) > ZERO_ATOL)], dtype=int)
        if len(candidates) == 0:
            continue
        min_value = np.nanmin(nipr10[candidates])
        chosen = int(candidates[np.flatnonzero(np.isclose(nipr10[candidates], min_value, rtol=0.0, atol=1e-12))[0]])
        selected_modes[name] = chosen
        comparison_rule_rows.append(
            {
                "mode_name": name,
                "selection_rule": "minimum finite N*IPR among simple modes in the nearest occupied flanking production bin; ascending index breaks ties",
                "production_bin_index": bin_index,
                "candidate_count": len(candidates),
                "selected_ascending_index_0based": chosen,
                "selected_energy": float(values10[chosen]),
                "selected_n_times_ipr": float(nipr10[chosen]),
            }
        )

    plot_n10000_density_ipr(
        figures / "n10000_same_graph_density_and_individual_ipr.png",
        np.asarray(values10), nipr10, density0, localization0,
        {key: selected_modes[key] for key in EXPECTED_10000},
    )

    # Localization-bin controls; density remains the original fixed display above.
    sensitivity10_rows: list[dict[str, object]] = []
    constructions: list[tuple[str, float, float, np.ndarray]] = []
    for basis in ("shared_disorder_range", "clean_spectrum_range_60"):
        anchor = float(common_edges[0]) if basis == "shared_disorder_range" else float(values10.min())
        base_width = common_width if basis == "shared_disorder_range" else float((values10.max() - values10.min()) / 60.0)
        for multiple in (0.5, 1.0, 2.0):
            for shift in (0.0, 0.25, 0.5, 0.75):
                width = base_width * multiple
                edges = fixed_width_bins(np.asarray(values10), width, shift, anchor)
                constructions.append((basis, multiple, shift, edges))
    side_viewports = {"negative": (-42.0, -18.0), "positive": (18.0, 42.0)}
    for basis, multiple, shift, edges in constructions:
        aggregates = aggregate_metric(np.asarray(values10), nipr10_reporting, edges)
        for side, (lo, hi) in side_viewports.items():
            side_rows = [row for row in aggregates if row["center"] >= lo and row["center"] <= hi]
            finite_rows = [row for row in side_rows if np.isfinite(row["mean_n_times_ipr"])]
            peak = max(finite_rows, key=lambda row: (row["mean_n_times_ipr"], -row["bin_index"]))
            for row in side_rows:
                sensitivity10_rows.append(
                    {
                        "record_type": "bin",
                        "side": side,
                        "edge_basis": basis,
                        "width_multiple": multiple,
                        "origin_shift_fraction": shift,
                        "base_width": common_width if basis == "shared_disorder_range" else float((values10.max() - values10.min()) / 60.0),
                        "actual_width": float(edges[1] - edges[0]),
                        "bin_index": row["bin_index"],
                        "bin_left": row["left"],
                        "bin_right": row["right"],
                        "bin_center": row["center"],
                        "eigenvalue_count": row["eigenvalue_count"],
                        "finite_ipr_count": row["finite_ipr_count"],
                        "mean_n_times_ipr": row["mean_n_times_ipr"],
                        "median_n_times_ipr": row["median_n_times_ipr"],
                        "peak_bin_center_in_fixed_window": peak["center"],
                        "peak_bin_mean_in_fixed_window": peak["mean_n_times_ipr"],
                    }
                )
            for name, index in selected_modes.items():
                if not name.startswith(side) or "comparison" in name:
                    continue
                bin_index = containing_bin(edges, float(values10[index]))
                row = aggregates[bin_index]
                sensitivity10_rows.append(
                    {
                        "record_type": "fixed_mode",
                        "side": side,
                        "edge_basis": basis,
                        "width_multiple": multiple,
                        "origin_shift_fraction": shift,
                        "base_width": common_width if basis == "shared_disorder_range" else float((values10.max() - values10.min()) / 60.0),
                        "actual_width": float(edges[1] - edges[0]),
                        "mode_name": name,
                        "mode_index": index,
                        "mode_energy": float(values10[index]),
                        "mode_n_times_ipr": float(nipr10[index]),
                        "bin_index": bin_index,
                        "bin_left": row["left"],
                        "bin_right": row["right"],
                        "bin_center": row["center"],
                        "eigenvalue_count": row["eigenvalue_count"],
                        "finite_ipr_count": row["finite_ipr_count"],
                        "mean_n_times_ipr": row["mean_n_times_ipr"],
                        "median_n_times_ipr": row["median_n_times_ipr"],
                        "peak_bin_center_in_fixed_window": peak["center"],
                        "peak_bin_mean_in_fixed_window": peak["mean_n_times_ipr"],
                    }
                )
    sensitivity10 = pd.DataFrame(sensitivity10_rows)
    plot_localization_sensitivity(
        figures / "n10000_localization_bin_sensitivity.png",
        sensitivity10, np.asarray(values10),
        {key: selected_modes[key] for key in EXPECTED_10000},
    )

    # Materialize only the selected Fortran-contiguous columns.
    selected_indices = sorted(set(selected_modes.values()))
    selected_vector_matrix = np.asarray(vectors10[:, selected_indices], dtype=np.float64).copy()
    vector_by_index = {index: selected_vector_matrix[:, position] for position, index in enumerate(selected_indices)}
    degree_order = np.lexsort((np.arange(n10), -n10_degrees))
    degree_rank = np.empty(n10, dtype=np.int64)
    degree_rank[degree_order] = np.arange(1, n10 + 1)

    support_shell_rows: list[dict[str, object]] = []
    top_vertex_rows: list[dict[str, object]] = []
    region_rows: list[dict[str, object]] = []
    outside_rows: list[dict[str, object]] = []
    mode_summary: dict[str, dict[str, object]] = {}
    distances_by_center: dict[int, np.ndarray] = {}
    for mode_name, index in selected_modes.items():
        vector = vector_by_index[index]
        norm = float(np.linalg.norm(vector))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=8e-12):
            raise AssertionError(f"N=10000 mode {mode_name} is not normalized: {norm}")
        mass = vector * vector
        computed_ipr = float(np.sum(mass * mass))
        if not math.isclose(computed_ipr, float(ipr10[index]), rel_tol=3e-10, abs_tol=2e-13):
            raise AssertionError(f"N=10000 mode {mode_name} IPR mismatch")
        computed_residual = sparse_residual(a10, vector, float(values10[index]))
        if not math.isclose(computed_residual, float(abs_residuals10[index]), rel_tol=0.20, abs_tol=2.5e-14):
            raise AssertionError(f"N=10000 mode {mode_name} residual mismatch")
        order = np.lexsort((np.arange(n10), -mass))
        cumulative = np.cumsum(mass[order])
        center = int(order[0])
        distances = distances_by_center.setdefault(center, bfs_distances(a10, center))
        unreachable = int(np.sum(distances < 0))
        summary: dict[str, object] = {
            "record_type": "summary",
            "mode_name": mode_name,
            "ascending_index_0based": index,
            "energy": float(values10[index]),
            "norm": norm,
            "ipr": computed_ipr,
            "n_times_ipr": n10 * computed_ipr,
            "saved_absolute_residual": float(abs_residuals10[index]),
            "recomputed_absolute_residual": computed_residual,
            "dominant_vertex": center,
            "dominant_vertex_degree": int(n10_degrees[center]),
            "dominant_vertex_degree_rank": int(degree_rank[center]),
            "dominant_vertex_mass": float(mass[center]),
            "unreachable_vertex_count": unreachable,
        }
        for k in TOP_K:
            summary[f"top_{k}_mass"] = float(cumulative[k - 1])
        for level in SUPPORT_LEVELS:
            summary[f"support_size_{int(level * 100)}"] = int(np.searchsorted(cumulative, level, side="left") + 1)
        support_shell_rows.append(summary)
        mode_summary[mode_name] = summary
        for distance in np.unique(distances[distances >= 0]):
            shell = distances == distance
            support_shell_rows.append(
                {
                    "record_type": "shell",
                    "mode_name": mode_name,
                    "ascending_index_0based": index,
                    "energy": float(values10[index]),
                    "dominant_vertex": center,
                    "graph_distance": int(distance),
                    "shell_size": int(shell.sum()),
                    "shell_mass": float(mass[shell].sum()),
                    "cumulative_mass_through_shell": float(mass[(distances >= 0) & (distances <= distance)].sum()),
                    "unreachable_vertex_count": unreachable,
                }
            )

        top20 = order[:20]
        top20_mask = np.zeros(n10, dtype=bool)
        top20_mask[top20] = True
        for top_rank, vertex in enumerate(top20, start=1):
            neighbors = a10.indices[a10.indptr[vertex] : a10.indptr[vertex + 1]]
            internal_degree = int(top20_mask[neighbors].sum())
            top_vertex_rows.append(
                {
                    "mode_name": mode_name,
                    "ascending_index_0based": index,
                    "energy": float(values10[index]),
                    "top_mass_rank": top_rank,
                    "vertex": int(vertex),
                    "signed_coordinate": float(vector[vertex]),
                    "squared_mass": float(mass[vertex]),
                    "degree": int(n10_degrees[vertex]),
                    "degree_rank": int(degree_rank[vertex]),
                    "edges_to_top20": internal_degree,
                    "edges_outside_top20": int(n10_degrees[vertex] - internal_degree),
                    "distance_from_dominant_vertex": int(distances[vertex]),
                }
            )

        region_masks: dict[str, np.ndarray] = {}
        for level in (0.90, 0.95, 0.99):
            count = int(np.searchsorted(cumulative, level, side="left") + 1)
            region = np.zeros(n10, dtype=bool)
            region[order[:count]] = True
            region_masks[f"S{int(level * 100)}"] = region
        for radius in (0, 1, 2):
            region_masks[f"radius{radius}"] = (distances >= 0) & (distances <= radius)
        for region_name, region in region_masks.items():
            vertices = np.flatnonzero(region)
            captured_mass = float(mass[region].sum())
            x = vector[region] / math.sqrt(captured_mass)
            u = np.zeros(n10, dtype=np.float64)
            u[region] = x
            au = a10 @ u
            residual = au - float(values10[index]) * u
            internal_error = float(np.linalg.norm(residual[region]))
            boundary_leakage = float(np.linalg.norm(au[~region]))
            full_residual = float(np.linalg.norm(residual))
            decomposition_error = abs(full_residual**2 - internal_error**2 - boundary_leakage**2)
            if decomposition_error > 2e-10 * max(1.0, full_residual**2):
                raise AssertionError("candidate-region residual decomposition failed")
            internal_edges, boundary_edges = region_graph_stats(a10, region)
            outside = np.flatnonzero(~region)
            contribution_order = outside[np.argsort(-np.abs(au[outside]), kind="stable")[:10]]
            for outside_rank, vertex in enumerate(contribution_order, start=1):
                outside_rows.append(
                    {
                        "mode_name": mode_name,
                        "ascending_index_0based": index,
                        "region_name": region_name,
                        "outside_contribution_rank": outside_rank,
                        "vertex": int(vertex),
                        "signed_incoming_contribution": float(au[vertex]),
                        "absolute_incoming_contribution": float(abs(au[vertex])),
                        "degree": int(n10_degrees[vertex]),
                        "distance_from_dominant_vertex": int(distances[vertex]),
                    }
                )
            region_rows.append(
                {
                    "mode_name": mode_name,
                    "ascending_index_0based": index,
                    "energy": float(values10[index]),
                    "dominant_vertex": center,
                    "region_name": region_name,
                    "region_size": int(region.sum()),
                    "region_fraction": float(region.mean()),
                    "within_operational_20pct_cap": bool(region.mean() <= COMPACTNESS_CAP),
                    "captured_mass": captured_mass,
                    "mass_omitted": 1.0 - captured_mass,
                    "overlap_with_observed_vector": math.sqrt(captured_mass),
                    "squared_overlap_with_observed_vector": captured_mass,
                    "internal_error": internal_error,
                    "boundary_leakage": boundary_leakage,
                    "full_residual": full_residual,
                    "residual_squared_decomposition_error": decomposition_error,
                    "outside_signed_incoming_sum": float(au[~region].sum()),
                    "outside_absolute_incoming_sum": float(np.abs(au[~region]).sum()),
                    "internal_edge_count": internal_edges,
                    "boundary_edge_count": boundary_edges,
                }
            )

    pair_rows: list[dict[str, object]] = []
    for pair_name, negative_name, positive_name in (
        ("target_pair", "negative_target", "positive_target"),
        ("companion_pair", "negative_companion", "positive_companion"),
    ):
        ni, pi = selected_modes[negative_name], selected_modes[positive_name]
        negative_mass = vector_by_index[ni] ** 2
        positive_mass = vector_by_index[pi] ** 2
        denom = float(np.linalg.norm(negative_mass) * np.linalg.norm(positive_mass))
        pair_rows.append(
            {
                "pair_name": pair_name,
                "negative_mode": negative_name,
                "positive_mode": positive_name,
                "negative_index": ni,
                "positive_index": pi,
                "negative_energy": float(values10[ni]),
                "positive_energy": float(values10[pi]),
                "energy_sum": float(values10[ni] + values10[pi]),
                "squared_mass_overlap": float(np.minimum(negative_mass, positive_mass).sum()),
                "squared_mass_cosine": float(np.dot(negative_mass, positive_mass) / denom),
                "negative_dominant_vertex": mode_summary[negative_name]["dominant_vertex"],
                "positive_dominant_vertex": mode_summary[positive_name]["dominant_vertex"],
                "dominant_center_agreement": bool(mode_summary[negative_name]["dominant_vertex"] == mode_summary[positive_name]["dominant_vertex"]),
                "signed_inner_product_not_used_for_spatial_overlap": float(np.dot(vector_by_index[ni], vector_by_index[pi])),
            }
        )

    star_rows: list[dict[str, object]] = []
    for mode_name in ("negative_target", "positive_target", "negative_companion", "positive_companion"):
        index = selected_modes[mode_name]
        observed = vector_by_index[index]
        center = int(mode_summary[mode_name]["dominant_vertex"])
        neighbors = a10.indices[a10.indptr[center] : a10.indptr[center + 1]]
        degree = len(neighbors)
        sign = 1.0 if values10[index] > 0 else -1.0
        template = np.zeros(n10, dtype=np.float64)
        template[center] = 1.0 / math.sqrt(2.0)
        template[neighbors] = sign / math.sqrt(2.0 * degree)
        predicted = sign * math.sqrt(degree)
        atemplate = a10 @ template
        rayleigh = float(np.dot(template, atemplate))
        epsilon = float(np.linalg.norm(atemplate - rayleigh * template))
        overlap_sq = float(np.dot(template, observed) ** 2)
        other = np.ones(n10, dtype=bool)
        other[index] = False
        separation = float(np.min(np.abs(np.asarray(values10)[other] - rayleigh)))
        closest_other_index = int(np.flatnonzero(other)[np.argmin(np.abs(np.asarray(values10)[other] - rayleigh))])
        ratio = epsilon / separation if separation > 0 else math.inf
        observed_mass = observed * observed
        neighbor_mass = float(observed_mass[neighbors].sum())
        hub_mass = float(observed_mass[center])
        outside_mass = float(1.0 - hub_mass - neighbor_mass)
        energy_error = float(values10[index] - rayleigh)
        spatial_assessment = "strong" if overlap_sq >= 0.80 and hub_mass + neighbor_mass >= 0.85 else "partial" if overlap_sq >= 0.10 else "poor"
        spectral_assessment = "informative" if ratio < 1.0 and abs(energy_error) <= 0.10 * max(1.0, abs(values10[index])) else "poor"
        if spatial_assessment == "strong" and spectral_assessment == "informative":
            assessment = "good"
        elif spatial_assessment in {"strong", "partial"}:
            assessment = "partial"
        else:
            assessment = "poor"
        star_rows.append(
            {
                "mode_name": mode_name,
                "ascending_index_0based": index,
                "observed_energy": float(values10[index]),
                "dominant_hub": center,
                "hub_degree": degree,
                "template_sign": "+" if sign > 0 else "-",
                "predicted_energy_sign_sqrt_degree": predicted,
                "rayleigh_quotient_actual_graph": rayleigh,
                "energy_error_observed_minus_predicted": float(values10[index] - predicted),
                "energy_error_observed_minus_rayleigh": energy_error,
                "template_full_residual_about_rayleigh": epsilon,
                "squared_overlap_with_observed_vector": overlap_sq,
                "observed_hub_mass": hub_mass,
                "observed_neighbor_mass": neighbor_mass,
                "observed_outside_mass": outside_mass,
                "template_hub_mass": 0.5,
                "template_neighbor_mass": 0.5,
                "template_outside_mass": 0.0,
                "spectral_separation_to_other_saved_eigenvalues": separation,
                "closest_other_index": closest_other_index,
                "closest_other_energy": float(values10[closest_other_index]),
                "epsilon_over_separation": ratio,
                "separation_bound_informative": bool(ratio < 1.0),
                "spatial_assessment": spatial_assessment,
                "spectral_assessment": spectral_assessment,
                "operational_assessment": assessment,
                "assessment_rule": "good only if spatially strong and spectrally informative; partial if spatial match survives but spectral test fails; otherwise poor",
            }
        )

    support_frame = pd.DataFrame(support_shell_rows)
    region_frame = pd.DataFrame(region_rows)
    pair_frame = pd.DataFrame(pair_rows)
    plot_spatial_profiles(
        figures / "n10000_spatial_profiles_and_support_leakage.png",
        support_frame, region_frame, pair_frame,
    )

    # --- Output tables ---
    write_csv(output / "target_mode_map.csv", target_map_rows)
    write_csv(output / "fixed_interval_mass_and_density.csv", fixed_interval_rows)
    write_csv(output / "bin_sensitivity.csv", sensitivity_rows)
    write_csv(output / "n5000_target_quartet_gaps.csv", quartet_rows)
    write_csv(output / "n5000_tie_aware_quantile_bins.csv", quantile_rows)
    write_csv(output / "n10000_window_individual_modes.csv", window_mode_rows)
    write_csv(output / "n10000_comparison_mode_selection.csv", comparison_rule_rows)
    sensitivity10.to_csv(output / "localization_bin_sensitivity.csv", index=False, float_format="%.15g")
    write_csv(output / "n10000_sign_pair_mass_overlap.csv", pair_rows)
    support_frame.to_csv(output / "mode_support_and_shell_mass.csv", index=False, float_format="%.15g")
    write_csv(output / "mode_top_vertices.csv", top_vertex_rows)
    region_frame.to_csv(output / "candidate_region_residuals.csv", index=False, float_format="%.15g")
    write_csv(output / "candidate_region_outside_contributions.csv", outside_rows)
    write_csv(output / "star_template_validation.csv", star_rows)

    # Assertions tied directly to the requested conclusions.
    density_lookup = density0.set_index("bin_index")
    if not (int(density_lookup.loc[21, "count"]) == 1 and int(density_lookup.loc[34, "count"]) == 1):
        raise AssertionError("N=10000 target bins are not singletons")
    if not (density_lookup.loc[21, "density"] < density_lookup.loc[20, "density"] and density_lookup.loc[21, "density"] < density_lookup.loc[22, "density"]):
        raise AssertionError("negative target is not a density trough relative to flanks")
    if not (density_lookup.loc[34, "density"] < density_lookup.loc[33, "density"] and density_lookup.loc[34, "density"] < density_lookup.loc[35, "density"]):
        raise AssertionError("positive target is not a density trough relative to flanks")
    max_region_decomposition_error = float(region_frame["residual_squared_decomposition_error"].max())
    fixed_mass_invariance = True

    # Checksums are computed after the bounded analysis so the large-vector hash
    # is a sequential read and does not interfere with selected-column access.
    input_checksums = {}
    for key, path in paths.items():
        input_checksums[key] = {"path": str(path), "sha256_file_bytes": sha256_file(path), "size_bytes": path.stat().st_size}

    star_frame = pd.DataFrame(star_rows)
    target_pair = pair_frame.set_index("pair_name").loc["target_pair"]
    companion_pair = pair_frame.set_index("pair_name").loc["companion_pair"]
    star_assessments = ", ".join(f"{row.mode_name}: {row.operational_assessment}" for row in star_frame.itertuples())
    star_overlap_range = (float(star_frame["squared_overlap_with_observed_vector"].min()), float(star_frame["squared_overlap_with_observed_vector"].max()))
    star_ratio_range = (float(star_frame["epsilon_over_separation"].min()), float(star_frame["epsilon_over_separation"].max()))
    compact_target_regions = region_frame[region_frame["mode_name"].isin(EXPECTED_10000) & region_frame["region_name"].isin(["S90", "S95"])]
    compact_residual_range = (float(compact_target_regions["full_residual"].min()), float(compact_target_regions["full_residual"].max()))
    repeated_mask_count = int(repeated_basis_dependent.sum())
    n5_negative_diameter = float(values5[target5_indices["negative"][-1]] - values5[target5_indices["negative"][0]])
    n5_positive_diameter = float(values5[target5_indices["positive"][-1]] - values5[target5_indices["positive"][0]])
    report_text = f"""# PA noncentral spectral peak and localization diagnosis

Run completed: {utc_now()}

Reproduction command:

```bash
python3 tools/audit_pa_noncentral_targets.py --root {root} --output {output}
```

## Executive findings

The two observations remain separate finite-graph objects.  The N=5000 density bars come from seed 4011773341; the N=10000 localization observations come from seed 2748869207.  Similar energy scales do not identify eigenvectors across them and do not establish a common mechanism or a thermodynamic mobility transition.

1. **N=5000 bars.** Each production target bar contains four distinct extreme-tail eigenvalues, not a repeated level and not an unresolved numerical cluster.  The negative quartet is {', '.join(f'{x:.9f}' for x in target5_values['negative'])}; its diameter is {n5_negative_diameter:.6f}.  The positive quartet is {', '.join(f'{x:.9f}' for x in target5_values['positive'])}; its diameter is {n5_positive_diameter:.6f}.  Internal gaps are at least {min(np.diff(target5_values['negative']).min(), np.diff(target5_values['positive']).min()):.6f}, far above the available positive-vector residuals and floating-point scale; negative-vector residuals are unavailable.  What survives bin removal is a real grouping of four distinct edge levels in each fixed physical interval.  What does not survive unchanged is the visual bar prominence: counts/heights and the regional peak center move with width and origin.  With four levels, this is finite-sample grouping plus plotting sensitivity, not an established new density component.

2. **N=10000 density versus localization.** The modes at -31.8325975382 (ascending column 21) and +31.6195395463 (column 9978) are concentration peaks, not density peaks.  Their original bins contain one eigenvalue and one finite-IPR contributor each, with N×IPR {nipr10[21]:.6f} and {nipr10[9978]:.6f}; each density is {density_lookup.loc[21, 'density']:.9g}, below both flanks.  The reported centers near ±32 are bin centers.  Changing localization bins moves the displayed peak center and changes singleton/group means, while the eigenpair IDs, energies, and individual N×IPR values remain fixed.  Empty means remain unavailable and are never filled with zero.

3. **Carrying eigenvectors and regions.** The ±32 target pair is centered on vertex {int(target_pair['negative_dominant_vertex'])}, degree {int(n10_degrees[int(target_pair['negative_dominant_vertex'])])}; the ±27 companion pair is centered on vertex {int(companion_pair['negative_dominant_vertex'])}, degree {int(n10_degrees[int(companion_pair['negative_dominant_vertex'])])}.  Target squared-mass overlap is {target_pair['squared_mass_overlap']:.6f} with cosine {target_pair['squared_mass_cosine']:.6f} and energy sum {target_pair['energy_sum']:.6f}.  Companion overlap is {companion_pair['squared_mass_overlap']:.6f} with cosine {companion_pair['squared_mass_cosine']:.6f} and energy sum {companion_pair['energy_sum']:.6f}.  This is strong approximate cross-sign spatial pairing, not exact spectral symmetry.  Signed inner products are recorded only as an orthogonality check, not used as spatial evidence.

4. **Graph-defined star test and certification status.** The elementary actual-graph star templates are assessed overall as {star_assessments}.  Spatial overlap is strong (squared overlap {star_overlap_range[0]:.3f}–{star_overlap_range[1]:.3f}), but spectral performance is poor: epsilon/separation is {star_ratio_range[0]:.1f}–{star_ratio_range[1]:.1f}, so every separation bound is uninformative.  The degree-energy scale is suggestive, not a valid eigenpair approximation.  No numerical support threshold, truncated known-vector residual, or star fit is an exact algebraic certificate; no exact compactly supported explanation is established.

## Provenance and identity checks

- Both graphs are verified simple, symmetric, unweighted, and connected from saved edge lists.  N=5000 has {graph5_checks['edge_count']} edges and degree range {graph5_checks['minimum_degree']}–{graph5_checks['maximum_degree']}; N=10000 has {graph10_checks['edge_count']} edges and degree range {graph10_checks['minimum_degree']}–{graph10_checks['maximum_degree']}.
- Both models record alpha=1.05 as the degree-CCDF parameter, m=10, delta=-9.5.  The operator is the unscaled raw adjacency (a=0 for N=10000), in identity vertex order.
- N=5000 accounting reproduces 5000 total, 1844 numerical zeros at atol=1e-8/rtol=0, 3156 nonzero levels, and 3155 nonzero non-Perron levels.  The recovered 160-bin right-panel width is {h:.12f}.
- N=10000 canonical graph-content SHA-256 is `{n10_canonical_hash}` and matches its graph manifest.  Archive-byte hashes and canonical content hashes are separately labeled in `diagnostic_manifest.json`; they are not compared as if they represented the same object.
- The eight-amplitude study chooses one common 60-bin range; source and manifest provenance show amplitudes {json.dumps(report_manifest['amplitudes'])}.  Direct recomputation confirms the a=0 counts, densities, contributor counts, means, and medians use only the clean eigensystem, not pooled observations.  The production repeated-eigenspace mask removes {repeated_mask_count} basis-dependent zero-space IPR values and no noncentral target value.

## N=5000 spectral controls

`bin_sensitivity.csv` keeps the 3155-level sample and denominator fixed, extends every shifted grid across the full sample, and separates selected-bar membership from regional peak position/height.  `fixed_interval_mass_and_density.csv` records the production target interval and a predeclared three-production-bin surrounding interval; these counts and masses do not change under display rebinning.  Common-edge overlays separately show original-N mass and retained-sample conditional density.  The original production panels should not be compared as if they shared edges: their automatic ranges and denominators both differ.

The ECDF/rank/gap figure displays the discrete quartet levels and neighboring gaps.  The tie-aware quantile display uses density=count/(sample size×width), merges repeated cuts, and passed the saved repeated-value fixture.  Its different visual prominence is a display control, not proof that the finite-sample concentration is artificial.

Negative N=5000 quartet vectors were not saved, so their per-mode residuals and cross-sign spatial comparison remain unavailable.  The exact follow-up would reconstruct the saved graph, run one targeted dense symmetric eigensolve for the same N=5000/m=10/alpha=1.05/seed=4011773341 realization, and save only columns 4–7 into a new audit-only directory.  That recomputation was not performed here.

## N=10000 aligned localization and structural pilot

The aligned figure uses every individual scalar mode in the four production bins on each side.  It prints both eigenvalue counts and finite-IPR contributor counts.  The companion modes near -27.6423 and +27.2661 are also strongly concentrated, so the phenomenon is not solely a singleton-bin artifact even though the displayed ±32 maxima are singleton means.

For the four target/companion vectors plus four deterministic lower-IPR flank comparisons, `mode_support_and_shell_mass.csv` reports top-k mass, 50/90/95/99% support sizes, and complete graph-distance shell mass.  `mode_top_vertices.csv` gives signed top coordinates and graph connectivity.  `candidate_region_residuals.csv` covers S90/S95/S99 and radius-0/1/2 regions, marking the 20% operational compactness cap.  Although S90/S95 regions fit below the cap for the four focal modes, their full residuals remain {compact_residual_range[0]:.2f}–{compact_residual_range[1]:.2f}; radius 2 covers almost the whole graph.  This rejects a small-region eigenpair explanation under the tested truncations.  The maximum squared-residual decomposition error is {max_region_decomposition_error:.3e}.  Overlap with a truncated observed vector is explicitly retrospective and equals its captured mass after squaring.

## Evidence classes and limitations

**Directly measured:** target IDs/energies/ranks, bin membership, density and finite-IPR counts, IPR, saved and recomputed sparse residuals, support/shell mass, graph connectivity, cross-sign squared-mass overlaps, and actual-graph template calculations.

**Approximate model fits:** cross-sign pairing and the elementary hub-star templates.  Their quality is descriptive and mode-specific.

**Untested hypotheses:** a shared mechanism between the N=5000 and N=10000 realizations; an exact compact-support motif; persistence across graph ensembles; and any thermodynamic mobility transition.

## Smallest justified next experiment

Only if direct N=5000 cross-sign mechanism comparison is needed, recompute the four missing negative target eigenvectors for the exact saved N=5000 graph and compare their support/mass profiles with the already saved positive quartet.  Required inputs are already present (saved edges, seed, vertex order, and scalar eigenvalues); the uncertainty resolved would be whether the two N=5000 quartets localize on matching hubs/regions.  A broader ensemble or local symbolic theory is not needed for that narrow question and is not implied by this audit.

## Output guide

- `target_mode_map.csv`: exact IDs, energies, ranks, IPR, residuals, bin membership, vector availability.
- `fixed_interval_mass_and_density.csv`, `bin_sensitivity.csv`, `n5000_target_quartet_gaps.csv`, `n5000_tie_aware_quantile_bins.csv`: N=5000 controls.
- `n10000_window_individual_modes.csv`, `localization_bin_sensitivity.csv`: same-graph density/localization alignment and rebinning controls.
- `n10000_sign_pair_mass_overlap.csv`, `mode_support_and_shell_mass.csv`, `mode_top_vertices.csv`: saved-vector spatial diagnostics.
- `candidate_region_residuals.csv`, `candidate_region_outside_contributions.csv`, `star_template_validation.csv`: bounded graph-defined pilot.
- `figures/`: six labelled diagnostic figures.
- `diagnostic_manifest.json`: checksums, conventions, commands, verification results, and completed/skipped steps.
"""
    (output / "PA_NONCENTRAL_PEAK_DIAGNOSIS_REPORT.md").write_text(report_text, encoding="utf-8")

    command = f"python3 tools/audit_pa_noncentral_targets.py --root {root} --output {output}"
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "status": "COMPLETE",
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "repository_physical_root": str(root),
        "reproduction_command": command,
        "script_identity": {"path": str(Path(__file__).resolve()), "sha256_file_bytes": sha256_file(Path(__file__).resolve())},
        "parameters": {
            "degree_ccdf_alpha": 1.05,
            "m": 10,
            "attachment_offset_delta": -9.5,
            "operator": "unscaled raw adjacency; a=0",
            "zero_atol": ZERO_ATOL,
            "zero_rtol": 0.0,
            "compactness_reporting_cap_fraction": COMPACTNESS_CAP,
            "n5000_histogram_bins": 160,
            "n5000_production_width": h,
            "n10000_report_bin_count": 60,
            "n10000_common_width": common_width,
            "comparison_mode_selection": "minimum finite N*IPR simple mode in nearest occupied flanking production bin; ascending index tie-break",
        },
        "target_graphs": {
            "n5000": {"n": n5, "seed": n5_seed, "canonical_graph_content_sha256": n5_canonical_hash, "archive_byte_sha256": input_checksums["n5000_graph"]["sha256_file_bytes"]},
            "n10000": {"n": n10, "seed": int(metadata10["graph_seed"]), "canonical_graph_content_sha256": n10_canonical_hash, "archive_byte_sha256": input_checksums["n10000_graph"]["sha256_file_bytes"]},
        },
        "input_files": input_checksums,
        "target_eigenpair_ids": {key: int(value) for key, value in selected_modes.items()},
        "filters_and_bins": {
            "n5000": "np.isclose(E,0,atol=1e-8,rtol=0) removed; Perron removed after zero filtering; np.histogram 160 automatic equal-width bins",
            "n10000": "60 exact stored common edges set by eight-amplitude range; a=0 values aggregated alone; repeated-eigenspace N*IPR stays NaN",
            "shifted_histograms": "same 3155-level sample and denominator; edge grids extended over complete sample range",
            "tie_aware_quantiles": "cuts moved to distinct-value boundaries; repeated edges merged; density=count/(sample_size*width)",
        },
        "dependency_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "platform": platform.platform(),
        },
        "verification": {
            "n5000_accounting": {"total": len(values5), "zero": int(zero5.sum()), "nonzero": len(nonzero5), "nonzero_nonperron": len(continuous5)},
            "n5000_histogram_normalization_all_settings": True,
            "n5000_fixed_interval_mass_unchanged_under_rebinning": fixed_mass_invariance,
            "tie_aware_repeated_value_fixture": tie_test,
            "n10000_index_energy_alignment": True,
            "n10000_repeated_eigenspace_mask_count": repeated_mask_count,
            "n10000_vector_norm_ipr_sparse_residual_checks": True,
            "n10000_original_density_and_localization_rows_reproduced": True,
            "candidate_region_residual_squared_decomposition_max_error": max_region_decomposition_error,
            "graphs_simple_symmetric_unweighted_connected": True,
            "n10000_vertex_map_identity": True,
        },
        "completed_steps": [
            "provenance and graph/eigensystem validation",
            "N=5000 production histogram reproduction and width/origin controls",
            "N=5000 common-edge, ECDF/rank/gap, and tie-aware quantile controls",
            "N=10000 same-realization density/localization alignment",
            "N=10000 localization-bin sensitivity including clean-only edge construction",
            "memory-mapped selected-column vector diagnostics",
            "support, shell, candidate-region leakage, sign-pair overlap, and actual-graph star-template tests",
        ],
        "skipped_or_deferred_steps": [
            {"step": "N=5000 negative-vector spatial/residual analysis", "reason": "target columns 4-7 were not saved; targeted recomputation expressly deferred"},
            {"step": "local eigensolve or exact symbolic motif certification", "reason": "outside authorized bounded pilot"},
            {"step": "graph regeneration, full eigensolve, disorder sweep, or ensemble simulation", "reason": "not authorized and not needed"},
        ],
        "outputs": sorted(
            set(str(path.relative_to(output)) for path in output.rglob("*") if path.is_file())
            | {"diagnostic_manifest.json", "pa_noncentral_diagnosis_20261004_v1.zip"}
        ),
    }
    (output / "diagnostic_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Re-open written small tables and report before packaging.
    required_outputs = [
        "PA_NONCENTRAL_PEAK_DIAGNOSIS_REPORT.md",
        "target_mode_map.csv",
        "fixed_interval_mass_and_density.csv",
        "bin_sensitivity.csv",
        "localization_bin_sensitivity.csv",
        "n10000_sign_pair_mass_overlap.csv",
        "mode_support_and_shell_mass.csv",
        "candidate_region_residuals.csv",
        "star_template_validation.csv",
        "diagnostic_manifest.json",
    ]
    for relative in required_outputs:
        file_path = output / relative
        if not file_path.is_file() or file_path.stat().st_size == 0:
            raise AssertionError(f"required output missing or empty: {file_path}")

    script_path = Path(__file__).resolve()
    zip_path = output / "pa_noncentral_diagnosis_20261004_v1.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.write(script_path, arcname=f"tools/{script_path.name}")
        for path in sorted(output.rglob("*")):
            if path.is_file() and path != zip_path:
                archive.write(path, arcname=str(path.relative_to(output.parent)))
    if not zip_path.is_file() or zip_path.stat().st_size == 0:
        raise AssertionError("diagnostic ZIP was not created")
    print(json.dumps({"status": "COMPLETE", "output": str(output), "zip": str(zip_path)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("audit_outputs/pa_noncentral_diagnosis_20261004_v1"))
    parser.add_argument("--overwrite", action="store_true", help="allow use of an existing output directory")
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
