"""Shared metrics for the PA coupling and Anderson handoff execution.

The functions in this module are deliberately operator-agnostic. They keep
rank, quantile, energy-window, atom, localization, and graph-distance
conventions in one place so retained-case, replicated-PA, and RRG jobs cannot
silently drift apart.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import networkx as nx
import numpy as np
from scipy import sparse

from .common import BFSDistanceCache, deterministic_mass_order, pair_distance_rows


HANDOFF_METRICS_VERSION = "pa_coupling_and_anderson_metrics_v1"
TOP_COUNTS = (1, 5, 10, 20, 30, 50, 100, 200)
MASS_FRACTIONS = (0.50, 0.80, 0.90, 0.95)


def localization_metrics(
    vector: np.ndarray,
    *,
    top_counts: Sequence[int] = TOP_COUNTS,
    mass_fractions: Sequence[float] = MASS_FRACTIONS,
) -> dict[str, object]:
    """Return consistently named localization metrics for one unit vector."""
    vector = np.asarray(vector)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or abs(norm - 1.0) > 1e-7:
        raise ValueError("localization metrics require a unit vector")
    order, masses = deterministic_mass_order(vector)
    sorted_masses = masses[order]
    cumulative = np.cumsum(sorted_masses)
    ipr = float(np.sum(masses * masses))
    n = len(vector)
    result: dict[str, object] = {
        "n": n,
        "vector_norm": norm,
        "ipr": ipr,
        "n_times_ipr": float(n * ipr),
        "participation_ratio": float(1.0 / ipr) if ipr > 0.0 else math.inf,
        "participation_fraction": float(1.0 / (n * ipr)) if ipr > 0.0 else math.inf,
        "maximum_absolute_coordinate": float(np.max(np.abs(vector))),
        "maximum_coordinate_mass": float(sorted_masses[0]),
        "dominant_vertex": int(order[0]),
    }
    for count in top_counts:
        actual = min(int(count), n)
        result[f"top_{count}_mass"] = float(cumulative[actual - 1]) if actual else 0.0
    total = float(cumulative[-1]) if len(cumulative) else 0.0
    for fraction in mass_fractions:
        label = int(round(100.0 * float(fraction)))
        size = (
            int(np.searchsorted(cumulative, float(fraction) * total, side="left") + 1)
            if total > 0.0
            else 0
        )
        result[f"support_size_{label}"] = size
        result[f"support_fraction_{label}"] = float(size / n) if n else math.nan
    return result


def ascending_edge_band(n: int, edge: str, start_rank: int, end_rank: int) -> np.ndarray:
    """Return consecutive ascending indices for one algebraic edge rank band."""
    if not (1 <= start_rank <= end_rank <= n):
        raise ValueError("invalid rank band")
    if edge == "upper":
        return np.arange(n - end_rank, n - start_rank + 1, dtype=int)
    if edge == "lower":
        return np.arange(start_rank - 1, end_rank, dtype=int)
    raise ValueError("edge must be upper or lower")


def quantile_window_indices(n: int, quantile: float, width: float) -> np.ndarray:
    """Return one fixed-width consecutive graph-local quantile window."""
    if n < 3 or not 0.0 <= quantile <= 1.0 or not 0.0 < width <= 1.0:
        raise ValueError("invalid quantile window")
    count = max(3, int(round(width * n)))
    if count % 2 == 0:
        count += 1
    count = min(count, n if n % 2 else n - 1)
    center = int(round(float(quantile) * (n - 1)))
    lo = max(0, center - count // 2)
    hi = min(n, lo + count)
    lo = max(0, hi - count)
    return np.arange(lo, hi, dtype=int)


def physical_energy_window_indices(values: np.ndarray, half_width: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.flatnonzero(np.abs(values) <= float(half_width)).astype(int)


def predeclared_windows(
    values: np.ndarray,
    *,
    quantile_width: float,
    physical_half_width: float,
) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    return {
        "lower_rank_1_10": ascending_edge_band(n, "lower", 1, 10),
        "lower_rank_11_25": ascending_edge_band(n, "lower", 11, 25),
        "lower_rank_26_50": ascending_edge_band(n, "lower", 26, 50),
        "upper_rank_1_10": ascending_edge_band(n, "upper", 1, 10),
        "upper_rank_2_11": ascending_edge_band(n, "upper", 2, 11),
        "upper_rank_11_25": ascending_edge_band(n, "upper", 11, 25),
        "upper_rank_26_50": ascending_edge_band(n, "upper", 26, 50),
        "q25_width_0p02": quantile_window_indices(n, 0.25, quantile_width),
        "q50_width_0p02": quantile_window_indices(n, 0.50, quantile_width),
        "q75_width_0p02": quantile_window_indices(n, 0.75, quantile_width),
        "physical_abs_energy": physical_energy_window_indices(values, physical_half_width),
    }


def stable_zero_cluster(values: np.ndarray, *, enabled: bool) -> dict[str, object]:
    """Identify a scale-aware stable zero cluster only when atom handling is enabled."""
    values = np.asarray(values, dtype=float)
    scale = max(1.0, float(np.max(np.abs(values))) if len(values) else 1.0)
    base = float(32.0 * np.finfo(float).eps * max(1, len(values)) * scale)
    tolerances = (base / 10.0, base, base * 10.0)
    counts = tuple(int(np.count_nonzero(np.abs(values) <= tol)) for tol in tolerances)
    stable = counts[0] == counts[1] == counts[2] and counts[1] >= 2
    active = bool(enabled and stable)
    mask = np.abs(values) <= base if active else np.zeros(len(values), dtype=bool)
    return {
        "enabled": bool(enabled),
        "active": active,
        "tolerance": base,
        "tolerance_sensitivity": list(tolerances),
        "count_sensitivity": list(counts),
        "stable": stable,
        "dimension": int(np.count_nonzero(mask)),
        "mask": mask,
        "interpretation": (
            "stable scale-aware zero cluster excluded without bridging"
            if active
            else "no zero-atom exclusion applied"
        ),
    }


def gap_triplet_rows(
    values: np.ndarray,
    middle_indices: Sequence[int],
    *,
    atom_mask: np.ndarray | None,
    prefix: Mapping[str, object],
) -> list[dict[str, object]]:
    """Build raw adjacent-gap ratios without bridging excluded atoms or selections."""
    values = np.asarray(values, dtype=float)
    mask = np.zeros(len(values), dtype=bool) if atom_mask is None else np.asarray(atom_mask, dtype=bool)
    if mask.shape != values.shape:
        raise ValueError("atom mask shape mismatch")
    rows: list[dict[str, object]] = []
    for middle in np.asarray(middle_indices, dtype=int):
        if middle <= 0 or middle >= len(values) - 1:
            continue
        triplet = np.asarray([middle - 1, middle, middle + 1], dtype=int)
        excluded = bool(np.any(mask[triplet]))
        left_gap = float(values[middle] - values[middle - 1])
        right_gap = float(values[middle + 1] - values[middle])
        denominator = max(left_gap, right_gap)
        ratio = min(left_gap, right_gap) / denominator if denominator > 0.0 else math.nan
        rows.append({
            **prefix,
            "left_ascending_index": int(middle - 1),
            "middle_ascending_index": int(middle),
            "right_ascending_index": int(middle + 1),
            "left_energy": float(values[middle - 1]),
            "middle_energy": float(values[middle]),
            "right_energy": float(values[middle + 1]),
            "left_raw_gap": left_gap,
            "right_raw_gap": right_gap,
            "raw_gap_ratio": ratio if not excluded else math.nan,
            "atom_triplet_excluded": excluded,
            "triplet_is_consecutive_full_spectrum": True,
        })
    return rows


def selected_vector_rows(
    values: np.ndarray,
    vectors: np.ndarray,
    indices: Sequence[int],
    *,
    prefix: Mapping[str, object],
    windows: Mapping[str, np.ndarray],
    potential: np.ndarray | None = None,
    disorder_amplitude: float = 0.0,
    adjacency: sparse.spmatrix | None = None,
) -> list[dict[str, object]]:
    """Build one detailed localization row for each supplied vector column."""
    values = np.asarray(values, dtype=float)
    vectors = np.asarray(vectors, dtype=float)
    indices = np.asarray(indices, dtype=int)
    if vectors.shape != (len(values), len(indices)):
        raise ValueError("selected vector dimensions do not match spectrum")
    membership_sets = {name: set(members.tolist()) for name, members in windows.items()}
    rows: list[dict[str, object]] = []
    n = len(values)
    for column, index in enumerate(indices):
        vector = vectors[:, column]
        row = {
            **prefix,
            "ascending_index": int(index),
            "ascending_rank_1based": int(index + 1),
            "upper_algebraic_rank": int(n - index),
            "lower_algebraic_rank": int(index + 1),
            "graph_local_quantile": float(index / max(1, n - 1)),
            "energy": float(values[index]),
            "window_memberships": sorted(name for name, members in membership_sets.items() if int(index) in members),
            **localization_metrics(vector),
        }
        if potential is not None:
            xi = np.asarray(potential, dtype=float)
            if xi.shape != (n,):
                raise ValueError("potential shape mismatch")
            masses = np.square(np.abs(vector))
            center = int(row["dominant_vertex"])
            potential_expectation = float(np.dot(xi, masses))
            row.update({
                "center_potential": float(xi[center]),
                "weighted_potential_mean": potential_expectation,
                "diagonal_energy_contribution": float(disorder_amplitude * potential_expectation),
            })
            if adjacency is not None:
                adjacency_energy = float(np.real(np.vdot(vector, adjacency.dot(vector))))
                row.update({
                    "adjacency_energy_contribution": adjacency_energy,
                    "energy_identity_reconstruction": adjacency_energy + float(disorder_amplitude * potential_expectation),
                    "energy_identity_absolute_error": abs(float(values[index]) - adjacency_energy - float(disorder_amplitude * potential_expectation)),
                })
        rows.append(row)
    return rows


def within_vector_geometry(
    graph: nx.Graph,
    distance_cache: BFSDistanceCache,
    vector: np.ndarray,
    *,
    prefix: Mapping[str, object],
    potential: np.ndarray | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Return top-10 induced geometry and all 45 full-graph distances."""
    order, masses = deterministic_mass_order(vector)
    top = order[: min(10, len(order))]
    top_masses = masses[top]
    matrix = distance_cache.matrix(top)
    pair_rows, distance_summary = pair_distance_rows(top, top_masses, matrix)
    for row in pair_rows:
        row.update(prefix)
        row["distance_graph"] = "full original analyzed graph; intermediates may be outside selected set"
    induced = graph.subgraph(top.tolist())
    components = nx.number_connected_components(induced) if len(top) else 0
    center = int(order[0])
    summary: dict[str, object] = {
        **prefix,
        "dominant_vertex": center,
        "dominant_vertex_degree": int(graph.degree(center)),
        "top10_vertex_count": int(len(top)),
        "top10_induced_edge_count": int(induced.number_of_edges()),
        "top10_induced_component_count": int(components),
        "top10_induced_connected": bool(components == 1 if len(top) else False),
        "top10_induced_edge_density": float(nx.density(induced)) if len(top) >= 2 else math.nan,
        "top10_vertices": top.tolist(),
        "top10_masses": top_masses.tolist(),
        **distance_summary,
    }
    if potential is not None:
        xi = np.asarray(potential, dtype=float)
        denominator = float(np.sum(top_masses))
        numerator = float(np.dot(top_masses, xi[top]))
        summary.update({
            "dominant_vertex_potential": float(xi[center]),
            "top10_mass_weighted_potential_numerator": numerator,
            "top10_mass_weighted_potential_denominator": denominator,
            "top10_conditional_mass_weighted_potential": numerator / denominator if denominator > 0.0 else math.nan,
        })
    return summary, pair_rows


def between_vector_geometry(
    distance_cache: BFSDistanceCache,
    left_vector: np.ndarray,
    right_vector: np.ndarray,
    *,
    prefix: Mapping[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Return overlap and cross-distances between two top-10 mass sets."""
    left_order, left_mass_all = deterministic_mass_order(left_vector)
    right_order, right_mass_all = deterministic_mass_order(right_vector)
    left = left_order[:10]
    right = right_order[:10]
    left_mass = left_mass_all[left]
    right_mass = right_mass_all[right]
    matrix = distance_cache.matrix(left, right)
    rows: list[dict[str, object]] = []
    numerator = 0.0
    denominator = 0.0
    finite_values: list[float] = []
    for i, left_vertex in enumerate(left):
        for j, right_vertex in enumerate(right):
            distance = float(matrix[i, j])
            weight = float(left_mass[i] * right_mass[j])
            finite = math.isfinite(distance)
            rows.append({
                **prefix,
                "left_support_rank": i + 1,
                "right_support_rank": j + 1,
                "left_vertex": int(left_vertex),
                "right_vertex": int(right_vertex),
                "left_mass": float(left_mass[i]),
                "right_mass": float(right_mass[j]),
                "cross_mass_weight": weight,
                "distance": distance,
                "unreachable": not finite,
            })
            if finite:
                numerator += weight * distance
                denominator += weight
                finite_values.append(distance)
    left_masses = np.square(np.abs(left_vector))
    right_masses = np.square(np.abs(right_vector))
    summary = {
        **prefix,
        "left_dominant_vertex": int(left_order[0]),
        "right_dominant_vertex": int(right_order[0]),
        "dominant_vertex_distance": float(distance_cache.distance(left_order[0], right_order[0])),
        "top10_set_overlap_count": int(len(set(left.tolist()) & set(right.tolist()))),
        "full_mass_overlap": float(np.sum(np.minimum(left_masses, right_masses))),
        "cross_pair_count": int(len(rows)),
        "cross_finite_pair_count": int(len(finite_values)),
        "cross_minimum_distance": float(np.min(finite_values)) if finite_values else math.nan,
        "cross_mean_distance": float(np.mean(finite_values)) if finite_values else math.nan,
        "cross_mass_weighted_distance_numerator": numerator,
        "cross_mass_weighted_distance_denominator": denominator,
        "cross_mass_weighted_mean_distance": numerator / denominator if denominator > 0.0 else math.nan,
    }
    return summary, rows


def empirical_cdf(reference_values: np.ndarray, query: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference_values, dtype=float))
    query = np.asarray(query, dtype=float)
    if len(reference) == 0:
        return np.full(query.shape, np.nan)
    return np.searchsorted(reference, query, side="right") / float(len(reference))


def heldout_unfolded_triplets(
    rows: Sequence[Mapping[str, object]],
    reference_values: np.ndarray,
    *,
    spectrum_size: int,
) -> list[dict[str, object]]:
    """Attach whole-graph-held-out empirical-CDF gaps to raw triplet rows."""
    output: list[dict[str, object]] = []
    for row in rows:
        updated = dict(row)
        if bool(row.get("atom_triplet_excluded", False)):
            updated.update({"left_unfolded_gap": math.nan, "right_unfolded_gap": math.nan, "unfolded_gap_ratio": math.nan})
        else:
            energies = np.asarray([row["left_energy"], row["middle_energy"], row["right_energy"]], dtype=float)
            mapped = float(spectrum_size) * empirical_cdf(reference_values, energies)
            left = float(mapped[1] - mapped[0])
            right = float(mapped[2] - mapped[1])
            denominator = max(left, right)
            updated.update({
                "left_unfolded_gap": left,
                "right_unfolded_gap": right,
                "unfolded_gap_ratio": min(left, right) / denominator if denominator > 0.0 else math.nan,
            })
        output.append(updated)
    return output
