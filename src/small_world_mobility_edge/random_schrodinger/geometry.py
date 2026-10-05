"""Stage 1: cache-only localization geometry for saved PA eigenpairs."""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/random-schrodinger-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/random-schrodinger-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .common import (
    BFSDistanceCache,
    component_ids,
    core_numbers,
    deterministic_mass_order,
    graph_from_edge_cache,
    graph_to_csr,
    mass_overlap,
    pair_distance_rows,
    read_json,
    relative_residual,
    runtime_metadata,
    selected_mass_sets,
    sha256_file,
    stable_json_hash,
    utc_now,
    write_csv,
    write_json,
)


STAGE1_VERSION = "induced_geometry_v2"
PRIMARY_BALL_RADIUS = 2
PRIMARY_PEAK_SEPARATION = 3
PRIMARY_TARGET_MASS = 0.80
MAX_CENTERS = 4
SENSITIVITY_RADII = (1, 2, 3)
SENSITIVITY_SEPARATIONS = (2, 3, 4)


def _deterministic_degree_ranks(degrees: np.ndarray) -> np.ndarray:
    degrees = np.asarray(degrees, dtype=int)
    return np.asarray([1 + int(np.sum(degrees > degree)) for degree in degrees], dtype=int)


def _induced_summary(graph: nx.Graph, vertices: Sequence[int]) -> dict[str, object]:
    nodes = [int(vertex) for vertex in vertices]
    induced = graph.subgraph(nodes)
    components = list(nx.connected_components(induced))
    return {
        "selected_size": len(nodes),
        "induced_edge_count": induced.number_of_edges(),
        "induced_density": float(nx.density(induced)) if len(nodes) > 1 else 0.0,
        "induced_component_count": len(components),
        "induced_largest_component_size": max((len(part) for part in components), default=0),
    }


def _distance_summary_for_set(
    cache: BFSDistanceCache,
    vertices: Sequence[int],
    masses: Sequence[float],
) -> tuple[np.ndarray, list[dict[str, object]], dict[str, object]]:
    matrix = cache.matrix(vertices)
    rows, summary = pair_distance_rows(vertices, masses, matrix)
    return matrix, rows, summary


def _ball_vertices(cache: BFSDistanceCache, center: int, radius: int) -> np.ndarray:
    row = cache.row_array(int(center))
    return np.flatnonzero((row >= 0) & (row <= radius)).astype(int)


def _local_mass_peaks(
    graph: nx.Graph,
    masses: np.ndarray,
    candidate_vertices: Sequence[int],
) -> list[int]:
    peaks: list[int] = []
    for vertex in candidate_vertices:
        vertex = int(vertex)
        mass = float(masses[vertex])
        is_peak = True
        for neighbor in graph.neighbors(vertex):
            neighbor_mass = float(masses[int(neighbor)])
            if neighbor_mass > mass or (neighbor_mass == mass and int(neighbor) < vertex):
                is_peak = False
                break
        if is_peak:
            peaks.append(vertex)
    if not peaks and len(candidate_vertices):
        peaks = [int(candidate_vertices[0])]
    return peaks


def _select_centers(
    graph: nx.Graph,
    vector: np.ndarray,
    cache: BFSDistanceCache,
    radius: int,
    separation: int,
    target_mass: float,
    max_centers: int = MAX_CENTERS,
) -> dict[str, object]:
    order, masses = deterministic_mass_order(vector)
    peaks = _local_mass_peaks(graph, masses, order[: min(50, len(order))])
    candidates: list[tuple[float, float, int, int]] = []
    for vertex in peaks:
        ball = _ball_vertices(cache, vertex, radius)
        candidates.append(
            (
                float(np.sum(masses[ball])),
                float(masses[vertex]),
                int(graph.degree(vertex)),
                int(vertex),
            )
        )
    candidates.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
    selected: list[int] = []
    covered: set[int] = set()
    achieved = 0.0
    for _, _, _, vertex in candidates:
        if any(cache.distance(vertex, other) <= separation for other in selected):
            continue
        selected.append(vertex)
        covered.update(_ball_vertices(cache, vertex, radius).tolist())
        achieved = float(np.sum(masses[np.asarray(sorted(covered), dtype=int)]))
        if achieved >= target_mass or len(selected) >= max_centers:
            break
    resolved = bool(selected) and achieved >= target_mass
    if not resolved:
        label = "unresolved"
    elif len(selected) == 1:
        label = "one_center"
    elif len(selected) == 2:
        label = "two_center"
    else:
        label = "multi_center"
    return {
        "radius": int(radius),
        "peak_separation": int(separation),
        "target_mass": float(target_mass),
        "candidate_peak_count": len(peaks),
        "centers": selected,
        "center_count": len(selected),
        "covered_mass": achieved,
        "resolved": resolved,
        "geometry_label": label,
    }


def _linear_decay_fit(radius: np.ndarray, values: np.ndarray) -> dict[str, object]:
    radius = np.asarray(radius, dtype=float)
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values) & (values > 1e-14) & (radius >= 1.0)
    x = radius[valid]
    y = values[valid]
    if len(x) < 3:
        return {
            "fit_supported": False,
            "reason": "fewer_than_three_positive_shells",
            "slope": math.nan,
            "intercept": math.nan,
            "r_squared": math.nan,
            "point_count": len(x),
        }
    log_y = np.log(y)
    slope, intercept = np.polyfit(x, log_y, 1)
    predicted = slope * x + intercept
    ss_res = float(np.sum(np.square(log_y - predicted)))
    ss_tot = float(np.sum(np.square(log_y - np.mean(log_y))))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else math.nan
    supported = bool(slope < 0.0 and math.isfinite(r_squared) and r_squared >= 0.70)
    return {
        "fit_supported": supported,
        "reason": "supported" if supported else "requires_negative_slope_and_R2_at_least_0p70",
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "point_count": len(x),
    }


def _shell_profiles(
    vector: np.ndarray,
    centers: Sequence[int],
    cache: BFSDistanceCache,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    masses = np.square(np.abs(np.asarray(vector)))
    amplitudes = np.abs(np.asarray(vector))
    rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    for center_rank, center in enumerate(centers, start=1):
        distances = cache.row_array(int(center))
        finite_distances = distances[distances >= 0]
        maximum = int(np.max(finite_distances)) if len(finite_distances) else 0
        cumulative = 0.0
        center_rows: list[dict[str, object]] = []
        for radius in range(maximum + 1):
            shell = np.asarray(
                np.flatnonzero(distances == radius),
                dtype=int,
            )
            if not len(shell):
                continue
            shell_mass = float(np.sum(masses[shell]))
            cumulative += shell_mass
            center_rows.append(
                {
                    "center_rank": center_rank,
                    "center_vertex_id": int(center),
                    "radius": radius,
                    "shell_size": len(shell),
                    "shell_mass": shell_mass,
                    "ball_mass": cumulative,
                    "mean_mass_per_shell_vertex": shell_mass / len(shell),
                    "mean_absolute_component": float(np.mean(amplitudes[shell])),
                    "rms_component": float(math.sqrt(shell_mass / len(shell))),
                }
            )
        rows.extend(center_rows)
        radius = np.asarray([row["radius"] for row in center_rows], dtype=float)
        for quantity in ("rms_component", "mean_absolute_component", "shell_mass"):
            values = np.asarray([row[quantity] for row in center_rows], dtype=float)
            fit_rows.append(
                {
                    "center_rank": center_rank,
                    "center_vertex_id": int(center),
                    "quantity": quantity,
                    **_linear_decay_fit(radius, values),
                }
            )
    return rows, fit_rows


def _sample_uniform_set(
    component_nodes: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    return np.asarray(rng.choice(component_nodes, size=count, replace=False), dtype=int)


def _sample_degree_matched_set(
    target_vertices: Sequence[int],
    degrees: np.ndarray,
    component_nodes: np.ndarray,
    rng: np.random.Generator,
    nearest_pool: int = 12,
) -> tuple[np.ndarray, dict[str, float]]:
    targets = np.asarray(target_vertices, dtype=int)
    available: set[int] = set(int(value) for value in component_nodes)
    result = np.empty(len(targets), dtype=int)
    target_order = np.lexsort((targets, -degrees[targets]))
    for target_position in target_order:
        target = int(targets[target_position])
        candidates = np.asarray(sorted(available), dtype=int)
        differences = np.abs(np.log1p(degrees[candidates]) - math.log1p(float(degrees[target])))
        ordering = np.lexsort((candidates, np.abs(degrees[candidates] - degrees[target]), differences))
        pool = candidates[ordering[: min(nearest_pool, len(ordering))]]
        chosen = int(pool[int(rng.integers(0, len(pool)))])
        result[target_position] = chosen
        available.remove(chosen)
    target_degrees = degrees[targets].astype(float)
    matched_degrees = degrees[result].astype(float)
    quality = {
        "degree_match_mean_absolute_error": float(np.mean(np.abs(matched_degrees - target_degrees))),
        "degree_match_max_absolute_error": float(np.max(np.abs(matched_degrees - target_degrees))),
        "degree_match_mean_log_error": float(
            np.mean(np.abs(np.log1p(matched_degrees) - np.log1p(target_degrees)))
        ),
        "degree_match_exact_fraction": float(np.mean(matched_degrees == target_degrees)),
    }
    return result, quality


def _baseline_distribution(
    graph: nx.Graph,
    cache: BFSDistanceCache,
    selected_vertices: Sequence[int],
    degrees: np.ndarray,
    components: np.ndarray,
    sample_count: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rng = np.random.default_rng(seed)
    selected = np.asarray(selected_vertices, dtype=int)
    component = int(components[selected[0]])
    if np.any(components[selected] != component):
        raise ValueError("selected support spans multiple components")
    component_nodes = np.flatnonzero(components == component)
    rows: list[dict[str, object]] = []
    for method in ("uniform_within_component", "degree_matched_within_component"):
        for sample_index in range(sample_count):
            if method == "uniform_within_component":
                vertices = _sample_uniform_set(component_nodes, len(selected), rng)
                quality = {
                    "degree_match_mean_absolute_error": math.nan,
                    "degree_match_max_absolute_error": math.nan,
                    "degree_match_mean_log_error": math.nan,
                    "degree_match_exact_fraction": math.nan,
                }
            else:
                vertices, quality = _sample_degree_matched_set(
                    selected, degrees, component_nodes, rng
                )
            matrix = cache.matrix(vertices)
            _, summary = pair_distance_rows(
                vertices,
                np.full(len(vertices), 1.0 / len(vertices), dtype=float),
                matrix,
            )
            rows.append(
                {
                    "baseline_method": method,
                    "sample_index": sample_index,
                    "component_id": component,
                    "vertex_ids": vertices.tolist(),
                    "mean_degree": float(np.mean(degrees[vertices])),
                    "maximum_degree": int(np.max(degrees[vertices])),
                    **quality,
                    **summary,
                    **_induced_summary(graph, vertices),
                }
            )
    summary_row: dict[str, object] = {}
    for method in ("uniform_within_component", "degree_matched_within_component"):
        selected_rows = [row for row in rows if row["baseline_method"] == method]
        prefix = "uniform" if method.startswith("uniform") else "degree_matched"
        for metric in (
            "finite_distance_mean",
            "finite_distance_median",
            "unreachable_fraction",
            "induced_edge_count",
            "mean_degree",
            "maximum_degree",
        ):
            values = np.asarray([float(row[metric]) for row in selected_rows], dtype=float)
            finite = values[np.isfinite(values)]
            summary_row[prefix + "_" + metric + "_mean"] = (
                float(np.mean(finite)) if len(finite) else math.nan
            )
            summary_row[prefix + "_" + metric + "_sd"] = (
                float(np.std(finite, ddof=1)) if len(finite) > 1 else math.nan
            )
        if prefix == "degree_matched":
            for metric in (
                "degree_match_mean_absolute_error",
                "degree_match_max_absolute_error",
                "degree_match_mean_log_error",
                "degree_match_exact_fraction",
            ):
                values = np.asarray([float(row[metric]) for row in selected_rows], dtype=float)
                summary_row[metric + "_mean"] = float(np.mean(values))
    return rows, summary_row


def _standardized_comparison(value: float, sample: Sequence[float]) -> tuple[float, float]:
    values = np.asarray(sample, dtype=float)
    values = values[np.isfinite(values)]
    if not math.isfinite(value) or not len(values):
        return math.nan, math.nan
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    z = (value - float(np.mean(values))) / sd if sd > 0.0 else math.nan
    percentile = 100.0 * float(np.mean(values <= value))
    return z, percentile


def _plot_distance_heatmap(
    matrix: np.ndarray,
    support: Sequence[int],
    title: str,
    path: Path,
) -> None:
    display = np.asarray(matrix, dtype=float).copy()
    finite = display[np.isfinite(display)]
    replacement = float(np.max(finite) + 1.0) if len(finite) else 1.0
    display[~np.isfinite(display)] = replacement
    figure, axis = plt.subplots(figsize=(7.0, 6.0))
    image = axis.imshow(display, cmap="viridis", interpolation="nearest")
    labels = [str(int(vertex)) for vertex in support]
    axis.set_xticks(range(len(labels)))
    axis.set_yticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axis.set_yticklabels(labels, fontsize=8)
    axis.set_xlabel("original vertex ID")
    axis.set_ylabel("original vertex ID")
    axis.set_title(title)
    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            value = matrix[i, j]
            text = "inf" if not math.isfinite(float(value)) else str(int(value))
            axis.text(j, i, text, ha="center", va="center", fontsize=7, color="white")
    figure.colorbar(image, ax=axis, label="full-graph shortest-path distance")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_connecting_paths(
    graph: nx.Graph,
    support: Sequence[int],
    masses: Sequence[float],
    centers: Sequence[int],
    title: str,
    path: Path,
    seed: int,
) -> None:
    support_nodes = [int(value) for value in support]
    support_set = set(support_nodes)
    path_nodes: set[int] = set()
    path_edges: set[tuple[int, int]] = set()
    anchors = list(centers[:3]) if centers else support_nodes[:3]
    for left, right in zip(anchors[:-1], anchors[1:]):
        try:
            selected_path = nx.shortest_path(graph, int(left), int(right))
        except nx.NetworkXNoPath:
            continue
        path_nodes.update(int(value) for value in selected_path)
        path_edges.update(
            tuple(sorted((int(a), int(b)))) for a, b in zip(selected_path[:-1], selected_path[1:])
        )
    display_nodes = sorted(support_set | path_nodes)
    display = graph.subgraph(display_nodes).copy()
    positions = nx.spring_layout(display, seed=int(seed) % (2**32 - 1))
    mass_by_vertex = {int(vertex): float(mass) for vertex, mass in zip(support_nodes, masses)}
    figure, axis = plt.subplots(figsize=(7.4, 6.2))
    interior = [node for node in display_nodes if node not in support_set]
    if interior:
        nx.draw_networkx_nodes(
            display,
            positions,
            nodelist=interior,
            node_size=150,
            node_color="#bbbbbb",
            node_shape="s",
            label="one possible shortest-path interior",
            ax=axis,
        )
    sizes = [260.0 + 2400.0 * mass_by_vertex[node] / max(mass_by_vertex.values()) for node in support_nodes]
    colors = ["#d73027" if node in set(centers) else "#4575b4" for node in support_nodes]
    nx.draw_networkx_nodes(
        display,
        positions,
        nodelist=support_nodes,
        node_size=sizes,
        node_color=colors,
        edgecolors="black",
        label="selected top-mass vertex",
        ax=axis,
    )
    ordinary_edges = [edge for edge in display.edges() if tuple(sorted(edge)) not in path_edges]
    nx.draw_networkx_edges(display, positions, edgelist=ordinary_edges, alpha=0.35, width=1.0, ax=axis)
    nx.draw_networkx_edges(
        display,
        positions,
        edgelist=list(path_edges),
        edge_color="#fdae61",
        width=2.6,
        ax=axis,
    )
    nx.draw_networkx_labels(display, positions, font_size=7, ax=axis)
    axis.set_title(title + "\nHighlighted route is one possible shortest path")
    axis.axis("off")
    axis.legend(frameon=False, fontsize=7, loc="best")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def load_stage1_cache(source_dir: Path) -> dict[str, object]:
    source_dir = Path(source_dir)
    manifest_path = source_dir / "run_manifest.json"
    graph_path = source_dir / "tables" / "graph_realization.npz"
    vector_path = source_dir / "tables" / "selected_eigenvectors.npz"
    spectrum_path = source_dir / "spectrum" / "single_graph_spectral_arrays.npz"
    for path in (manifest_path, graph_path, vector_path, spectrum_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_manifest = read_json(manifest_path)
    if source_manifest.get("status") != "COMPLETE":
        raise ValueError("source run is not COMPLETE")
    if source_manifest.get("analysis_version") != "pa_single_graph_edge_bulk_v1":
        raise ValueError("unexpected source analysis version")
    graph, graph_metadata = graph_from_edge_cache(graph_path)
    with np.load(vector_path, allow_pickle=False) as data:
        global_indices = np.asarray(data["global_indices"], dtype=int)
        eigenvalues = np.asarray(data["eigenvalues"], dtype=float)
        eigenvectors = np.asarray(data["eigenvectors"], dtype=float)
    with np.load(spectrum_path, allow_pickle=False) as data:
        full_eigenvalues = np.asarray(data["eigenvalues"], dtype=float)
        valid_mask = np.asarray(data["valid_mask"], dtype=bool)
        ipr = np.asarray(data["ipr"], dtype=float)
        n_times_ipr = np.asarray(data["n_times_ipr"], dtype=float)
    n = graph.number_of_nodes()
    if eigenvectors.shape != (n, len(global_indices)):
        raise ValueError("selected eigenvector cache has incompatible dimensions")
    if not np.allclose(eigenvalues, full_eigenvalues[global_indices], atol=1e-10, rtol=1e-10):
        raise ValueError("selected eigenvalues do not match full-spectrum cache")
    if not np.allclose(np.linalg.norm(eigenvectors, axis=0), 1.0, atol=1e-8, rtol=0.0):
        raise ValueError("selected eigenvectors are not unit normalized")
    adjacency = graph_to_csr(graph, nodelist=range(n))
    residuals = []
    for column in range(eigenvectors.shape[1]):
        absolute, relative = relative_residual(adjacency, eigenvalues[column], eigenvectors[:, column])
        residuals.append(relative)
    if max(residuals) > 1e-8:
        raise ValueError("cached eigenpair residual exceeds 1e-8")
    hashes = {
        "source_manifest_sha256": sha256_file(manifest_path),
        "graph_cache_sha256": sha256_file(graph_path),
        "selected_eigenvectors_sha256": sha256_file(vector_path),
        "spectral_arrays_sha256": sha256_file(spectrum_path),
    }
    cache_identity = {
        "graph_sha256": hashes["graph_cache_sha256"],
        "vertex_ordering": "identity_0_to_N_minus_1",
        "operator": "unscaled_simple_undirected_adjacency",
        "operator_scalar_normalization": 1.0,
        "graph_seed": source_manifest.get("graph_seed"),
        "spectral_selection": source_manifest.get("edge_selection"),
        "source_analysis_version": source_manifest.get("analysis_version"),
        "selected_eigenvectors_sha256": hashes["selected_eigenvectors_sha256"],
    }
    return {
        "source_dir": source_dir,
        "source_manifest": source_manifest,
        "graph": graph,
        "adjacency": adjacency,
        "graph_metadata": graph_metadata,
        "global_indices": global_indices,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "full_eigenvalues": full_eigenvalues,
        "valid_mask": valid_mask,
        "ipr": ipr,
        "n_times_ipr": n_times_ipr,
        "maximum_relative_residual": max(residuals),
        "hashes": hashes,
        "cache_identity": cache_identity,
        "cache_id": stable_json_hash(cache_identity),
    }


def _decorate_rows(rows: list[dict[str, object]], prefix: Mapping[str, object]) -> None:
    for row in rows:
        for key, value in reversed(list(prefix.items())):
            row[key] = value


def run_stage1_cached_geometry(
    source_dir: Path,
    output_dir: Path,
    project_root: Path,
    baseline_samples: int = 100,
    baseline_seed: int = 945101,
) -> Path:
    start = time.time()
    source = load_stage1_cache(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    source_manifest = source["source_manifest"]
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if existing.get("status") == "COMPLETE":
            matching = (
                existing.get("stage_version") == STAGE1_VERSION
                and existing.get("source_cache_id") == source["cache_id"]
                and int(existing.get("baseline_samples_per_method_per_eigenvector", -1))
                == int(baseline_samples)
                and int(existing.get("baseline_seed", -1)) == int(baseline_seed)
            )
            if matching:
                return output_dir
            raise ValueError("completed Stage 1 output has a different cache identity or configuration")
    manifest: dict[str, object] = {
        **runtime_metadata(project_root),
        "stage": "stage1_cached_geometry",
        "stage_version": STAGE1_VERSION,
        "source_dir": str(Path(source_dir)),
        "output_dir": str(output_dir),
        "source_cache_id": source["cache_id"],
        "source_hashes": source["hashes"],
        "cache_identity": source["cache_identity"],
        "baseline_samples_per_method_per_eigenvector": int(baseline_samples),
        "baseline_seed": int(baseline_seed),
    }
    write_json(manifest_path, manifest)
    try:
        graph: nx.Graph = source["graph"]
        adjacency = source["adjacency"]
        n = graph.number_of_nodes()
        degrees = np.asarray([graph.degree(vertex) for vertex in range(n)], dtype=int)
        degree_ranks = _deterministic_degree_ranks(degrees)
        components = component_ids(graph)
        cores = core_numbers(graph)
        distance_cache = BFSDistanceCache(graph)
        global_indices = np.asarray(source["global_indices"], dtype=int)
        cached_values = np.asarray(source["eigenvalues"], dtype=float)
        cached_vectors = np.asarray(source["eigenvectors"], dtype=float)
        column_for_index = {int(index): position for position, index in enumerate(global_indices)}
        requested_ranks = list(range(1, 12))
        requested_indices = [n - rank for rank in requested_ranks]
        missing = [index for index in requested_indices if index not in column_for_index]
        if missing:
            raise ValueError("cache lacks requested upper-edge indices: %s" % missing)

        model_key = Path(source_dir).name
        common = {
            "model_key": model_key,
            "graph_seed": int(source_manifest["graph_seed"]),
            "graph_id": source["hashes"]["graph_cache_sha256"],
            "operator_id": stable_json_hash(
                {
                    "graph_id": source["hashes"]["graph_cache_sha256"],
                    "operator": "A",
                    "normalization": 1.0,
                }
            ),
            "operator": "A",
            "operator_scalar_normalization": 1.0,
            "original_N": n,
            "analyzed_N_eff": n,
            "vertex_mapping": "identity_0_to_N_minus_1",
            "stage_version": STAGE1_VERSION,
        }
        eigenpair_rows: list[dict[str, object]] = []
        support_rows: list[dict[str, object]] = []
        pair_rows: list[dict[str, object]] = []
        matrix_rows: list[dict[str, object]] = []
        sensitivity_rows: list[dict[str, object]] = []
        geometry_sensitivity_rows: list[dict[str, object]] = []
        shell_rows: list[dict[str, object]] = []
        decay_fit_rows: list[dict[str, object]] = []
        baseline_rows_all: list[dict[str, object]] = []
        vector_by_rank: dict[int, np.ndarray] = {}
        centers_by_rank: dict[int, list[int]] = {}
        support_by_rank: dict[int, np.ndarray] = {}
        masses_by_rank: dict[int, np.ndarray] = {}

        for rank in requested_ranks:
            index = n - rank
            column = column_for_index[index]
            value = float(cached_values[column])
            vector = np.asarray(cached_vectors[:, column], dtype=float)
            vector_by_rank[rank] = vector
            order, all_masses = deterministic_mass_order(vector)
            selections = selected_mass_sets(vector)
            top10 = selections["top_10"]
            top10_masses = all_masses[top10]
            support_by_rank[rank] = top10
            masses_by_rank[rank] = top10_masses
            matrix, local_pair_rows, distance_summary = _distance_summary_for_set(
                distance_cache, top10, top10_masses
            )
            prefix = {**common, "upper_edge_rank": rank, "global_spectrum_rank_ascending_1based": index + 1}
            _decorate_rows(local_pair_rows, prefix)
            pair_rows.extend(local_pair_rows)
            for i, left in enumerate(top10):
                for j, right in enumerate(top10):
                    matrix_rows.append(
                        {
                            **prefix,
                            "row_support_rank": i + 1,
                            "column_support_rank": j + 1,
                            "row_vertex_id": int(left),
                            "column_vertex_id": int(right),
                            "distance": float(matrix[i, j]),
                            "unreachable": not math.isfinite(float(matrix[i, j])),
                        }
                    )
            for support_rank, vertex in enumerate(top10, start=1):
                support_rows.append(
                    {
                        **prefix,
                        "literal_top10_including_perron": rank <= 10,
                        "top10_nonperron": 2 <= rank <= 11,
                        "support_rank": support_rank,
                        "vertex_id": int(vertex),
                        "signed_component": float(vector[vertex]),
                        "mass": float(all_masses[vertex]),
                        "full_graph_degree": int(degrees[vertex]),
                        "full_graph_degree_rank": int(degree_ranks[vertex]),
                        "component_id": int(components[vertex]),
                        "core_number": int(cores[vertex]),
                    }
                )
            for name, vertices in selections.items():
                vertices = np.asarray(vertices, dtype=int)
                masses = all_masses[vertices]
                _, _, selection_distance = _distance_summary_for_set(
                    distance_cache, vertices, masses
                )
                sensitivity_rows.append(
                    {
                        **prefix,
                        "selection": name,
                        "selection_size": len(vertices),
                        "selection_mass": float(np.sum(masses)),
                        **_induced_summary(graph, vertices),
                        **selection_distance,
                    }
                )
            sensitivity = []
            for radius in SENSITIVITY_RADII:
                for separation in SENSITIVITY_SEPARATIONS:
                    selected = _select_centers(
                        graph,
                        vector,
                        distance_cache,
                        radius=radius,
                        separation=separation,
                        target_mass=PRIMARY_TARGET_MASS,
                    )
                    sensitivity.append(selected)
                    geometry_sensitivity_rows.append({**prefix, **selected})
            primary = next(
                row
                for row in sensitivity
                if row["radius"] == PRIMARY_BALL_RADIUS
                and row["peak_separation"] == PRIMARY_PEAK_SEPARATION
            )
            centers = [int(value_) for value_ in primary["centers"]]
            centers_by_rank[rank] = centers
            local_shell_rows, local_fit_rows = _shell_profiles(vector, centers, distance_cache)
            _decorate_rows(local_shell_rows, prefix)
            _decorate_rows(local_fit_rows, prefix)
            shell_rows.extend(local_shell_rows)
            decay_fit_rows.extend(local_fit_rows)
            baseline_rows, baseline_summary = _baseline_distribution(
                graph,
                distance_cache,
                top10,
                degrees,
                components,
                sample_count=baseline_samples,
                seed=baseline_seed + rank,
            )
            _decorate_rows(baseline_rows, prefix)
            baseline_rows_all.extend(baseline_rows)
            comparison: dict[str, object] = {}
            for method, label in (
                ("uniform_within_component", "uniform"),
                ("degree_matched_within_component", "degree_matched"),
            ):
                sample = [
                    float(row["finite_distance_mean"])
                    for row in baseline_rows
                    if row["baseline_method"] == method
                ]
                zscore, percentile = _standardized_comparison(
                    float(distance_summary["finite_distance_mean"]), sample
                )
                comparison[label + "_distance_zscore"] = zscore
                comparison[label + "_distance_percentile"] = percentile
            absolute, relative = relative_residual(adjacency, value, vector)
            eigenpair_rows.append(
                {
                    **prefix,
                    "literal_top10_including_perron": rank <= 10,
                    "top10_nonperron": 2 <= rank <= 11,
                    "is_perron": rank == 1,
                    "eigenvalue": value,
                    "ipr": float(source["ipr"][index]),
                    "N_eff_times_ipr": float(source["n_times_ipr"][index]),
                    "top10_mass": float(np.sum(top10_masses)),
                    "maximum_mass": float(top10_masses[0]),
                    "absolute_residual_norm": absolute,
                    "relative_residual_norm": relative,
                    "primary_ball_radius": PRIMARY_BALL_RADIUS,
                    "primary_peak_separation": PRIMARY_PEAK_SEPARATION,
                    "primary_target_mass": PRIMARY_TARGET_MASS,
                    "geometry_label": primary["geometry_label"],
                    "center_count": primary["center_count"],
                    "center_vertex_ids": centers,
                    "center_covered_mass": primary["covered_mass"],
                    **_induced_summary(graph, top10),
                    **distance_summary,
                    **baseline_summary,
                    **comparison,
                }
            )
            slug = "rank_%03d" % rank
            _plot_distance_heatmap(
                matrix,
                top10,
                "%s upper-edge rank %d" % (model_key, rank),
                output_dir / "figures" / "distance_heatmaps" / (slug + ".png"),
            )
            _plot_connecting_paths(
                graph,
                top10,
                top10_masses,
                centers,
                "%s upper-edge rank %d" % (model_key, rank),
                output_dir / "figures" / "connecting_paths" / (slug + ".png"),
                seed=int(source_manifest["graph_seed"]) + rank,
            )

        adjacent_rows: list[dict[str, object]] = []
        cross_distance_rows: list[dict[str, object]] = []
        for upper_rank in range(1, 11):
            lower_rank = upper_rank + 1
            left_vector = vector_by_rank[upper_rank]
            right_vector = vector_by_rank[lower_rank]
            left_support = support_by_rank[upper_rank]
            right_support = support_by_rank[lower_rank]
            cross = distance_cache.matrix(left_support, right_support)
            left_center = centers_by_rank[upper_rank][0] if centers_by_rank[upper_rank] else None
            right_center = centers_by_rank[lower_rank][0] if centers_by_rank[lower_rank] else None
            center_distance = (
                distance_cache.distance(left_center, right_center)
                if left_center is not None and right_center is not None
                else math.inf
            )
            memberships = []
            if upper_rank <= 9:
                memberships.append("literal_top10_including_perron")
            if upper_rank >= 2:
                memberships.append("top10_nonperron")
            pair_id = "rank_%03d_%03d" % (upper_rank, lower_rank)
            observed_gap = float(source["full_eigenvalues"][n - upper_rank] - source["full_eigenvalues"][n - lower_rank])
            adjacent_rows.append(
                {
                    **common,
                    "pair_id": pair_id,
                    "upper_edge_rank_left": upper_rank,
                    "upper_edge_rank_right": lower_rank,
                    "window_membership": "|".join(memberships),
                    "left_eigenvalue": float(source["full_eigenvalues"][n - upper_rank]),
                    "right_eigenvalue": float(source["full_eigenvalues"][n - lower_rank]),
                    "observed_adjacent_gap_operator_units": observed_gap,
                    "left_dominant_center": left_center,
                    "right_dominant_center": right_center,
                    "dominant_center_distance": center_distance,
                    "dominant_center_unreachable": not math.isfinite(center_distance),
                    "mass_overlap_sum_min": mass_overlap(left_vector, right_vector),
                    "cross_distance_finite_mean": float(np.mean(cross[np.isfinite(cross)])),
                    "cross_distance_finite_median": float(np.median(cross[np.isfinite(cross)])),
                    "cross_distance_finite_maximum": float(np.max(cross[np.isfinite(cross)])),
                    "cross_distance_unreachable_fraction": float(np.mean(~np.isfinite(cross))),
                }
            )
            for i, left_vertex in enumerate(left_support):
                for j, right_vertex in enumerate(right_support):
                    cross_distance_rows.append(
                        {
                            **common,
                            "pair_id": pair_id,
                            "upper_edge_rank_left": upper_rank,
                            "upper_edge_rank_right": lower_rank,
                            "left_support_rank": i + 1,
                            "right_support_rank": j + 1,
                            "left_vertex_id": int(left_vertex),
                            "right_vertex_id": int(right_vertex),
                            "left_mass": float(masses_by_rank[upper_rank][i]),
                            "right_mass": float(masses_by_rank[lower_rank][j]),
                            "distance": float(cross[i, j]),
                            "unreachable": not math.isfinite(float(cross[i, j])),
                        }
                    )

        table_dir = output_dir / "tables"
        write_csv(table_dir / "eigenpair_geometry.csv", eigenpair_rows)
        write_csv(table_dir / "top10_support_vertices.csv", support_rows)
        write_csv(table_dir / "top10_pair_distances.csv", pair_rows)
        write_csv(table_dir / "top10_distance_matrices.csv.gz", matrix_rows)
        write_csv(table_dir / "mass_set_sensitivity.csv", sensitivity_rows)
        write_csv(table_dir / "center_rule_sensitivity.csv", geometry_sensitivity_rows)
        write_csv(table_dir / "center_shell_profiles.csv", shell_rows)
        write_csv(table_dir / "center_decay_fits.csv", decay_fit_rows)
        write_csv(table_dir / "structural_baselines.csv.gz", baseline_rows_all)
        write_csv(table_dir / "adjacent_edge_pairs.csv", adjacent_rows)
        write_csv(table_dir / "adjacent_cross_distances.csv.gz", cross_distance_rows)
        write_json(
            output_dir / "source_cache_provenance.json",
            {
                "source_dir": str(source_dir),
                "source_manifest": source_manifest,
                "hashes": source["hashes"],
                "cache_identity": source["cache_identity"],
                "cache_id": source["cache_id"],
                "cache_validation": {
                    "graph_vertex_mapping_recovered": True,
                    "graph_degrees_match_edges": True,
                    "selected_eigenvalues_match_full_spectrum": True,
                    "selected_eigenvectors_unit_normalized": True,
                    "maximum_relative_residual": source["maximum_relative_residual"],
                    "operator_reused": "identical unscaled adjacency A",
                },
            },
        )
        manifest.update(
            {
                "status": "COMPLETE",
                "finished_at_utc": utc_now(),
                "runtime_seconds": time.time() - start,
                "model_key": model_key,
                "eigenpair_count": len(eigenpair_rows),
                "literal_top10_rank_range": [1, 10],
                "nonperron_top10_rank_range": [2, 11],
                "adjacent_pair_count": len(adjacent_rows),
                "bfs_source_count": distance_cache.source_count,
                "distance_definition": "unweighted shortest paths in the full analyzed graph; paths may leave selected sets",
                "unreachable_storage": "IEEE infinity plus explicit boolean flag",
                "mass_weighted_distance_denominator": "sum_{i<j, finite d(i,j)} m_i m_j",
                "center_rule": {
                    "candidate_peaks": "local mass maxima among deterministic top-50 masses",
                    "ordering": "ball mass, vertex mass, degree descending; vertex ID ascending",
                    "greedy_separation": "candidate accepted only if distance from prior centers exceeds threshold",
                    "resolution": "minimum accepted centers (cap 4) whose radius-r ball union contains target mass",
                    "primary_radius": PRIMARY_BALL_RADIUS,
                    "primary_peak_separation": PRIMARY_PEAK_SEPARATION,
                    "primary_target_mass": PRIMARY_TARGET_MASS,
                },
                "historical_outputs_modified": False,
            }
        )
        write_json(manifest_path, manifest)
        return output_dir
    except Exception as error:
        manifest.update(
            {
                "status": "FAILED",
                "finished_at_utc": utc_now(),
                "runtime_seconds": time.time() - start,
                "failure_type": type(error).__name__,
                "failure_message": str(error),
            }
        )
        write_json(manifest_path, manifest)
        raise
