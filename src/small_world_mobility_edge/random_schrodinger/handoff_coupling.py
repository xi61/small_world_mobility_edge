"""Structural reference modes and fixed-partition clean-PA continuation.

This module implements the predictive track required by the September 2026
handoff.  Structural regions and modes are frozen without consulting the
target exact eigenvectors or observed adjacent gaps.  Exact eigenpairs enter
only in the subsequent matching and validation tables.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import networkx as nx
import numpy as np
from scipy import linalg, sparse
from scipy.optimize import linear_sum_assignment
from scipy.sparse import linalg as sparse_linalg

from .common import (
    BFSDistanceCache,
    graph_to_csr,
    read_json,
    runtime_metadata,
    source_configuration_identity,
    stable_json_hash,
    utc_now,
    write_csv,
    write_json,
)
from .geometry import load_stage1_cache
from .handoff_metrics import (
    between_vector_geometry,
    localization_metrics,
    within_vector_geometry,
)
from .tunneling import schur_effective_operator, two_hub_graph


COUPLING_HANDOFF_VERSION = "pa_structural_reference_and_continuation_v1"


@dataclass
class DictionaryResult:
    raw_states: np.ndarray
    orthonormal_states: np.ndarray
    transform: np.ndarray
    effective_matrix: np.ndarray
    residual_matrix: np.ndarray
    region_rows: list[dict[str, object]]
    mode_rows: list[dict[str, object]]
    orthogonal_rows: list[dict[str, object]]
    rejected_rows: list[dict[str, object]]
    regions: list[np.ndarray]
    region_centers: list[list[int]]


def structural_vertices(graph: nx.Graph, maximum: int = 50) -> np.ndarray:
    degrees = np.asarray([graph.degree(vertex) for vertex in range(graph.number_of_nodes())])
    order = np.lexsort((np.arange(len(degrees)), -degrees))
    return np.asarray(order[: min(int(maximum), len(order))], dtype=int)


def graph_ball(graph: nx.Graph, center: int, radius: int) -> np.ndarray:
    return np.asarray(
        sorted(nx.single_source_shortest_path_length(graph, int(center), cutoff=int(radius)).keys()),
        dtype=int,
    )


def overlap_coefficient(left: Sequence[int], right: Sequence[int]) -> float:
    left_set = set(int(value) for value in left)
    right_set = set(int(value) for value in right)
    denominator = min(len(left_set), len(right_set))
    return float(len(left_set & right_set) / denominator) if denominator else 0.0


def merge_structural_regions(
    graph: nx.Graph,
    centers: Sequence[int],
    *,
    radius: int,
    overlap_threshold: float,
    maximum_fraction: float,
) -> tuple[list[np.ndarray], list[list[int]], list[dict[str, object]], list[dict[str, object]]]:
    """Merge strongly overlapping balls and reject nonlocal merged regions."""
    centers = [int(value) for value in centers]
    balls = [graph_ball(graph, center, radius) for center in centers]
    parent = list(range(len(centers)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)

    overlap_rows: list[dict[str, object]] = []
    for left in range(len(centers)):
        for right in range(left + 1, len(centers)):
            coefficient = overlap_coefficient(balls[left], balls[right])
            merged = coefficient >= float(overlap_threshold)
            overlap_rows.append({
                "radius": int(radius),
                "left_center": centers[left],
                "right_center": centers[right],
                "left_size": len(balls[left]),
                "right_size": len(balls[right]),
                "overlap_count": int(len(set(balls[left].tolist()) & set(balls[right].tolist()))),
                "overlap_coefficient": coefficient,
                "merge_threshold": float(overlap_threshold),
                "merged_by_rule": merged,
            })
            if merged:
                union(left, right)

    groups: dict[int, list[int]] = {}
    for index in range(len(centers)):
        groups.setdefault(find(index), []).append(index)
    accepted_regions: list[np.ndarray] = []
    accepted_centers: list[list[int]] = []
    region_rows: list[dict[str, object]] = []
    n = graph.number_of_nodes()
    ordered_groups = sorted(groups.values(), key=lambda group: min(group))
    for group_order, group in enumerate(ordered_groups, start=1):
        union_nodes = np.asarray(sorted(set().union(*(set(balls[index].tolist()) for index in group))), dtype=int)
        group_centers = [centers[index] for index in group]
        fraction = float(len(union_nodes) / n)
        accepted = fraction <= float(maximum_fraction)
        row = {
            "region_candidate_order": group_order,
            "radius": int(radius),
            "center_count": len(group_centers),
            "centers": group_centers,
            "vertex_count": len(union_nodes),
            "vertex_fraction": fraction,
            "overlap_merge_threshold": float(overlap_threshold),
            "maximum_region_fraction": float(maximum_fraction),
            "accepted_local_region": accepted,
            "rejection_reason": "" if accepted else "merged_region_exceeds_local_fraction_cap",
        }
        region_rows.append(row)
        if accepted:
            accepted_regions.append(union_nodes)
            accepted_centers.append(group_centers)
    return accepted_regions, accepted_centers, region_rows, overlap_rows


def disjoint_partition(
    n: int, regions: Sequence[np.ndarray]
) -> tuple[list[np.ndarray], np.ndarray, list[dict[str, object]]]:
    """Resolve residual weak overlaps deterministically and append background."""
    assigned = np.full(int(n), -1, dtype=int)
    disjoint: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    for region_id, region in enumerate(regions):
        nodes = np.asarray(sorted(set(int(value) for value in region)), dtype=int)
        available = nodes[assigned[nodes] < 0]
        removed = int(len(nodes) - len(available))
        assigned[available] = region_id
        disjoint.append(available)
        rows.append({
            "region_id": region_id,
            "original_vertex_count": len(nodes),
            "disjoint_vertex_count": len(available),
            "overlap_vertices_assigned_to_earlier_region": removed,
        })
    background = np.flatnonzero(assigned < 0).astype(int)
    assigned[background] = len(disjoint)
    rows.append({
        "region_id": len(disjoint),
        "original_vertex_count": len(background),
        "disjoint_vertex_count": len(background),
        "overlap_vertices_assigned_to_earlier_region": 0,
        "role": "background",
    })
    return disjoint, assigned, rows


def _largest_local_modes(
    operator: sparse.spmatrix,
    nodes: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    restricted = sparse.csr_matrix(operator)[nodes][:, nodes]
    dimension = restricted.shape[0]
    count = min(int(count), dimension)
    if count <= 0:
        return np.empty(0), np.empty((dimension, 0))
    if dimension <= 256 or count >= dimension - 1:
        values, vectors = linalg.eigh(restricted.toarray(), check_finite=False)
        order = np.argsort(values)[::-1][:count]
        return np.asarray(values[order], dtype=float), np.asarray(vectors[:, order], dtype=float)
    values, vectors = sparse_linalg.eigsh(restricted, k=count, which="LA", tol=1e-10)
    order = np.argsort(values)[::-1]
    return np.asarray(values[order], dtype=float), np.asarray(vectors[:, order], dtype=float)


def orthogonalize_with_truncation(
    states: np.ndarray,
    *,
    relative_cutoff: float,
) -> dict[str, object]:
    states = np.asarray(states, dtype=float)
    gram = states.T @ states
    eigenvalues, eigenvectors = linalg.eigh(gram, check_finite=False)
    maximum = float(np.max(eigenvalues)) if len(eigenvalues) else 0.0
    threshold = float(relative_cutoff) * max(maximum, 1.0)
    keep = eigenvalues > threshold
    if not np.any(keep):
        raise ValueError("all reference directions rejected by Gram cutoff")
    transform = eigenvectors[:, keep] @ np.diag(1.0 / np.sqrt(eigenvalues[keep]))
    orthonormal = states @ transform
    return {
        "gram": gram,
        "gram_eigenvalues": eigenvalues,
        "threshold": threshold,
        "kept_mask": keep,
        "retained_rank": int(np.count_nonzero(keep)),
        "rejected_rank": int(np.count_nonzero(~keep)),
        "transform": transform,
        "states": orthonormal,
        "orthonormality_error": float(np.linalg.norm(orthonormal.T @ orthonormal - np.eye(orthonormal.shape[1]))),
    }


def build_structural_dictionary(
    graph: nx.Graph,
    operator: sparse.spmatrix,
    *,
    radius: int,
    local_mode_count: int,
    structural_vertex_count: int = 50,
    overlap_threshold: float = 0.5,
    maximum_region_fraction: float = 0.20,
    gram_cutoff: float = 1e-8,
    maximum_modes: int = 128,
) -> DictionaryResult:
    """Construct a target-independent structural local-mode dictionary."""
    operator = sparse.csr_matrix(operator)
    centers = structural_vertices(graph, structural_vertex_count)
    regions, region_centers, region_rows, overlap_rows = merge_structural_regions(
        graph,
        centers,
        radius=radius,
        overlap_threshold=overlap_threshold,
        maximum_fraction=maximum_region_fraction,
    )
    rejected_rows = [dict(row) for row in region_rows if not bool(row["accepted_local_region"])]
    if not regions:
        raise RuntimeError("no structural region remains under the locality cap")
    raw_columns: list[np.ndarray] = []
    mode_rows: list[dict[str, object]] = []
    n = graph.number_of_nodes()
    for region_id, (nodes, centers_in_region) in enumerate(zip(regions, region_centers)):
        values, local_vectors = _largest_local_modes(operator, nodes, local_mode_count)
        for local_rank, (value, local_vector) in enumerate(zip(values, local_vectors.T), start=1):
            embedded = np.zeros(n, dtype=float)
            embedded[nodes] = local_vector
            norm = float(np.linalg.norm(embedded))
            embedded /= norm
            sign_vertex = int(nodes[int(np.argmax(np.abs(local_vector)))])
            if embedded[sign_vertex] < 0.0:
                embedded *= -1.0
            residual = np.asarray(operator.dot(embedded)).reshape(-1) - float(value) * embedded
            raw_columns.append(embedded)
            mode_rows.append({
                "raw_mode_id": len(raw_columns) - 1,
                "region_id": region_id,
                "region_centers": centers_in_region,
                "radius": int(radius),
                "local_mode_rank": local_rank,
                "local_energy": float(value),
                "region_vertex_count": len(nodes),
                "region_vertex_fraction": float(len(nodes) / n),
                "preorthogonal_region_mass": float(np.sum(np.square(embedded[nodes]))),
                "full_residual_norm": float(np.linalg.norm(residual)),
                "boundary_condition": "induced/Dirichlet",
                "selection_independence": "structural degree-ranked region and largest algebraic local mode; no target exact eigenvector or gap used",
            })
            if len(raw_columns) >= int(maximum_modes):
                break
        if len(raw_columns) >= int(maximum_modes):
            break
    raw_states = np.column_stack(raw_columns)
    orthogonalized = orthogonalize_with_truncation(raw_states, relative_cutoff=gram_cutoff)
    u = np.asarray(orthogonalized["states"], dtype=float)
    transform = np.asarray(orthogonalized["transform"], dtype=float)
    k_matrix = u.T @ operator.dot(u)
    k_matrix = 0.5 * (k_matrix + k_matrix.T)
    residual = operator.dot(u) - u @ k_matrix
    orthogonal_rows: list[dict[str, object]] = []
    for column in range(u.shape[1]):
        masses = np.square(u[:, column])
        region_masses = [float(np.sum(masses[nodes])) for nodes in regions]
        coefficients = np.square(transform[:, column])
        orthogonal_rows.append({
            "orthogonal_mode_id": column,
            "dominant_raw_mode_id": int(np.argmax(coefficients)),
            "dominant_transform_weight": float(np.max(coefficients) / np.sum(coefficients)) if float(np.sum(coefficients)) > 0.0 else math.nan,
            "maximum_region_mass_after_orthogonalization": max(region_masses),
            "region_masses_after_orthogonalization": region_masses,
            "external_leakage_norm": float(np.linalg.norm(residual[:, column])),
        })
    accepted_index = 0
    for row in region_rows:
        if bool(row["accepted_local_region"]):
            row["accepted_region_id"] = accepted_index
            accepted_index += 1
    rejected_rows.extend({"kind": "pairwise_overlap", **row} for row in overlap_rows if bool(row["merged_by_rule"]))
    return DictionaryResult(
        raw_states=raw_states,
        orthonormal_states=u,
        transform=transform,
        effective_matrix=k_matrix,
        residual_matrix=np.asarray(residual),
        region_rows=region_rows,
        mode_rows=mode_rows,
        orthogonal_rows=orthogonal_rows,
        rejected_rows=rejected_rows,
        regions=regions,
        region_centers=region_centers,
    )


def dictionary_summary(dictionary: DictionaryResult) -> dict[str, object]:
    singular_values = np.linalg.svd(dictionary.residual_matrix, compute_uv=False)
    return {
        "accepted_region_count": len(dictionary.regions),
        "raw_mode_count": dictionary.raw_states.shape[1],
        "orthonormal_mode_count": dictionary.orthonormal_states.shape[1],
        "orthonormality_error": float(np.linalg.norm(dictionary.orthonormal_states.T @ dictionary.orthonormal_states - np.eye(dictionary.orthonormal_states.shape[1]))),
        "external_leakage_frobenius_norm": float(np.linalg.norm(dictionary.residual_matrix)),
        "external_leakage_operator_norm": float(singular_values[0]) if len(singular_values) else 0.0,
        "effective_matrix_hermiticity_error": float(np.linalg.norm(dictionary.effective_matrix - dictionary.effective_matrix.T)),
    }


def _pair_projection(operator: sparse.spmatrix, left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    raw = np.column_stack([left, right])
    result = orthogonalize_with_truncation(raw, relative_cutoff=1e-10)
    if int(result["retained_rank"]) != 2:
        return {"valid": False, "reason": "pair_rank_deficient"}
    u = np.asarray(result["states"])
    k = u.T @ sparse.csr_matrix(operator).dot(u)
    k = 0.5 * (k + k.T)
    residual = sparse.csr_matrix(operator).dot(u) - u @ k
    detuning = abs(float(k[0, 0] - k[1, 1]))
    coupling = float(k[0, 1])
    predicted = math.sqrt(detuning * detuning + 4.0 * coupling * coupling)
    predicted_minority = 0.5 * (1.0 - detuning / predicted) if predicted > 0.0 else math.nan
    ritz_values, ritz_vectors = linalg.eigh(k, check_finite=False)
    return {
        "valid": True,
        "reason": "valid",
        "u": u,
        "k": k,
        "residual": np.asarray(residual),
        "e_left": float(k[0, 0]),
        "e_right": float(k[1, 1]),
        "direct_coupling": coupling,
        "absolute_direct_coupling": abs(coupling),
        "uncoupled_detuning": detuning,
        "predicted_gap": predicted,
        "predicted_minority_weight_formula": predicted_minority,
        "ritz_values": ritz_values,
        "ritz_vectors": ritz_vectors,
        "gram_eigenvalues": np.asarray(result["gram_eigenvalues"]),
        "orthonormality_error": float(result["orthonormality_error"]),
    }


def structurally_ranked_mode_pairs(
    graph: nx.Graph,
    operator: sparse.spmatrix,
    dictionary: DictionaryResult,
    *,
    maximum_pairs: int = 20,
) -> list[dict[str, object]]:
    """Rank candidate pairs without exact target eigenpairs or observed gaps."""
    distance_cache = BFSDistanceCache(graph)
    candidates: list[dict[str, object]] = []
    for left in range(dictionary.raw_states.shape[1]):
        left_row = dictionary.mode_rows[left]
        for right in range(left + 1, dictionary.raw_states.shape[1]):
            right_row = dictionary.mode_rows[right]
            if int(left_row["region_id"]) == int(right_row["region_id"]):
                continue
            pair = _pair_projection(operator, dictionary.raw_states[:, left], dictionary.raw_states[:, right])
            if not bool(pair["valid"]):
                continue
            left_center = int(dictionary.region_centers[int(left_row["region_id"])][0])
            right_center = int(dictionary.region_centers[int(right_row["region_id"])][0])
            detuning = float(pair["uncoupled_detuning"])
            coupling = float(pair["absolute_direct_coupling"])
            resolution = 64.0 * np.finfo(float).eps * max(1.0, abs(float(pair["e_left"])), abs(float(pair["e_right"])))
            ratio = coupling / detuning if detuning > resolution else (math.inf if coupling > resolution else math.nan)
            candidates.append({
                "left_raw_mode_id": left,
                "right_raw_mode_id": right,
                "left_region_id": int(left_row["region_id"]),
                "right_region_id": int(right_row["region_id"]),
                "left_local_mode_rank": int(left_row["local_mode_rank"]),
                "right_local_mode_rank": int(right_row["local_mode_rank"]),
                "left_center": left_center,
                "right_center": right_center,
                "center_distance": float(distance_cache.distance(left_center, right_center)),
                "e_left": float(pair["e_left"]),
                "e_right": float(pair["e_right"]),
                "direct_coupling": float(pair["direct_coupling"]),
                "absolute_direct_coupling": coupling,
                "uncoupled_detuning": detuning,
                "coupling_over_detuning": ratio,
                "predicted_gap": float(pair["predicted_gap"]),
                "predicted_minority_weight_formula": float(pair["predicted_minority_weight_formula"]),
                "selection_score_role": "target-independent structural ranking; exact eigenvectors and observed gaps not used",
            })
    def sort_key(row: Mapping[str, object]) -> tuple[float, float, float, int, int]:
        ratio = float(row["coupling_over_detuning"])
        ratio_key = 1e300 if math.isinf(ratio) else (-1e300 if math.isnan(ratio) else ratio)
        mean_energy = 0.5 * (float(row["e_left"]) + float(row["e_right"]))
        return (-ratio_key, -mean_energy, float(row["center_distance"]), int(row["left_raw_mode_id"]), int(row["right_raw_mode_id"]))
    candidates.sort(key=sort_key)
    for order, row in enumerate(candidates[: int(maximum_pairs)], start=1):
        row["structural_candidate_order"] = order
    return candidates[: int(maximum_pairs)]


def _match_two_energies_to_adjacent_pair(
    predicted: np.ndarray,
    full_values: np.ndarray,
    *,
    first_rank: int = 2,
    last_rank: int = 50,
) -> dict[str, object]:
    predicted = np.sort(np.asarray(predicted, dtype=float))
    n = len(full_values)
    best: tuple[float, int, np.ndarray] | None = None
    for upper_rank in range(first_rank, last_rank):
        indices = np.asarray([n - (upper_rank + 1), n - upper_rank], dtype=int)
        exact = np.asarray(full_values[indices], dtype=float)
        error = float(np.max(np.abs(predicted - exact)))
        if best is None or (error, upper_rank) < (best[0], best[1]):
            best = (error, upper_rank, indices)
    if best is None:
        raise RuntimeError("no adjacent exact pair available for matching")
    _, upper_rank, indices = best
    return {
        "left_upper_rank": upper_rank + 1,
        "right_upper_rank": upper_rank,
        "ascending_indices": indices,
        "maximum_energy_match_error": best[0],
    }


def validate_candidate_pair(
    operator: sparse.spmatrix,
    pair: Mapping[str, object],
    dictionary: DictionaryResult,
    full_values: np.ndarray,
    exact_vector_by_index: Mapping[int, np.ndarray],
) -> dict[str, object]:
    left = int(pair["left_raw_mode_id"])
    right = int(pair["right_raw_mode_id"])
    projected = _pair_projection(operator, dictionary.raw_states[:, left], dictionary.raw_states[:, right])
    match = _match_two_energies_to_adjacent_pair(projected["ritz_values"], full_values)
    indices = np.asarray(match["ascending_indices"], dtype=int)
    if any(int(index) not in exact_vector_by_index for index in indices):
        return {**pair, **{key: value for key, value in match.items() if key != "ascending_indices"}, "validation_status": "matched_pair_vector_not_retained", "accepted_predictive_two_mode": False}
    exact = np.column_stack([exact_vector_by_index[int(index)] for index in indices])
    u = np.asarray(projected["u"])
    overlaps = np.square(np.abs(u.T @ exact))
    captures = np.sum(overlaps, axis=0)
    predicted_values = np.asarray(projected["ritz_values"])
    assignment = linear_sum_assignment(np.abs(predicted_values[:, None] - full_values[indices][None, :]))
    order = np.argsort(assignment[1])
    predicted_columns = assignment[0][order]
    exact_columns = assignment[1][order]
    predicted_reference_weights = np.square(np.abs(np.asarray(projected["ritz_vectors"])[:, predicted_columns]))
    conditional_exact_weights = overlaps[:, exact_columns] / np.maximum(captures[exact_columns][None, :], np.finfo(float).tiny)
    weight_error = float(np.max(np.abs(predicted_reference_weights - conditional_exact_weights)))
    observed_gap = float(full_values[indices[1]] - full_values[indices[0]])
    predicted_gap = float(projected["predicted_gap"])
    gap_error = abs(predicted_gap - observed_gap) / max(abs(observed_gap), np.finfo(float).eps)
    omitted = np.ones(len(full_values), dtype=bool)
    omitted[indices] = False
    exact_pair = full_values[indices]
    g_ext = float(min(np.min(np.abs(full_values[omitted] - energy)) for energy in exact_pair))
    leakage_operator = float(np.linalg.svd(np.asarray(projected["residual"]), compute_uv=False)[0])
    leakage_ratio = leakage_operator / g_ext if g_ext > 0.0 else math.inf
    capture_pass = bool(np.all(captures >= 0.90))
    leakage_pass = bool(leakage_ratio <= 0.10)
    gap_pass = bool(gap_error <= 0.10)
    weight_pass = bool(weight_error <= 0.05)
    left_region = dictionary.regions[int(pair["left_region_id"])]
    right_region = dictionary.regions[int(pair["right_region_id"])]
    exact_region_masses = [
        {
            "left_region_mass": float(np.sum(np.square(exact[:, column][left_region]))),
            "right_region_mass": float(np.sum(np.square(exact[:, column][right_region]))),
        }
        for column in range(2)
    ]
    return {
        **pair,
        **{key: value for key, value in match.items() if key != "ascending_indices"},
        "left_ascending_index": int(indices[0]),
        "right_ascending_index": int(indices[1]),
        "left_exact_energy": float(full_values[indices[0]]),
        "right_exact_energy": float(full_values[indices[1]]),
        "observed_adjacent_gap": observed_gap,
        "predicted_gap": predicted_gap,
        "relative_gap_prediction_error": gap_error,
        "left_exact_captured_mass": float(captures[0]),
        "right_exact_captured_mass": float(captures[1]),
        "minimum_exact_captured_mass": float(np.min(captures)),
        "predicted_reference_weights": predicted_reference_weights.tolist(),
        "measured_conditional_reference_weights": conditional_exact_weights.tolist(),
        "conditional_mode_weight_maximum_absolute_error": weight_error,
        "left_exact_vector_left_region_mass": exact_region_masses[0]["left_region_mass"],
        "left_exact_vector_right_region_mass": exact_region_masses[0]["right_region_mass"],
        "right_exact_vector_left_region_mass": exact_region_masses[1]["left_region_mass"],
        "right_exact_vector_right_region_mass": exact_region_masses[1]["right_region_mass"],
        "external_spectral_separation": g_ext,
        "external_spectral_separation_definition": "minimum distance from either matched exact eigenvalue to every other known full-spectrum eigenvalue",
        "external_leakage_operator_norm": leakage_operator,
        "external_leakage_over_separation": leakage_ratio,
        "capture_90_pass": capture_pass,
        "capture_95_sensitivity_pass": bool(np.all(captures >= 0.95)),
        "leakage_over_separation_0p10_pass": leakage_pass,
        "leakage_over_separation_0p05_sensitivity_pass": bool(leakage_ratio <= 0.05),
        "gap_error_0p10_pass": gap_pass,
        "conditional_weight_error_0p05_pass": weight_pass,
        "dictionary_enlargement_stability_verified": False,
        "accepted_before_stability_check": bool(capture_pass and leakage_pass and gap_pass and weight_pass),
        "accepted_predictive_two_mode": False,
        "validation_status": "stability_not_yet_established" if capture_pass and leakage_pass and gap_pass and weight_pass else "failed_predeclared_predictive_gate",
    }


def apply_dictionary_enlargement_gate(
    validation_rows: list[dict[str, object]],
    *,
    primary_variant: str,
    enlarged_variant: str,
    gap_relative_tolerance: float = 0.10,
    weight_absolute_tolerance: float = 0.05,
) -> list[dict[str, object]]:
    """Require predictions to remain stable when local modes are enlarged."""
    def key(row: Mapping[str, object]) -> tuple[int, int, int, int]:
        return (
            int(row["left_region_id"]),
            int(row["right_region_id"]),
            int(row["left_local_mode_rank"]),
            int(row["right_local_mode_rank"]),
        )

    enlarged = {
        key(row): row
        for row in validation_rows
        if row.get("variant") == enlarged_variant
    }
    for row in validation_rows:
        if row.get("variant") != primary_variant:
            continue
        comparison = enlarged.get(key(row))
        row["dictionary_enlargement_variant"] = enlarged_variant
        if comparison is None:
            row["dictionary_enlargement_stability_verified"] = False
            row["dictionary_enlargement_status"] = "no_structurally_corresponding_candidate"
            row["accepted_predictive_two_mode"] = False
            if bool(row.get("accepted_before_stability_check", False)):
                row["validation_status"] = "failed_dictionary_enlargement_correspondence"
            continue
        same_pair = (
            int(row.get("left_upper_rank", -1)) == int(comparison.get("left_upper_rank", -2))
            and int(row.get("right_upper_rank", -1)) == int(comparison.get("right_upper_rank", -2))
        )
        primary_gap = float(row.get("predicted_gap", math.nan))
        enlarged_gap = float(comparison.get("predicted_gap", math.nan))
        gap_change = abs(enlarged_gap - primary_gap) / max(abs(primary_gap), np.finfo(float).eps)
        try:
            primary_weights = np.asarray(row["predicted_reference_weights"], dtype=float)
            enlarged_weights = np.asarray(comparison["predicted_reference_weights"], dtype=float)
            weight_change = float(np.max(np.abs(primary_weights - enlarged_weights)))
        except (KeyError, TypeError, ValueError):
            weight_change = math.inf
        stable = bool(
            same_pair
            and gap_change <= float(gap_relative_tolerance)
            and weight_change <= float(weight_absolute_tolerance)
        )
        accepted = bool(row.get("accepted_before_stability_check", False) and stable)
        row.update({
            "dictionary_enlargement_same_matched_exact_pair": same_pair,
            "dictionary_enlargement_gap_relative_change": gap_change,
            "dictionary_enlargement_weight_maximum_absolute_change": weight_change,
            "dictionary_enlargement_stability_verified": stable,
            "dictionary_enlargement_status": "stable" if stable else "changed_under_enlargement",
            "accepted_predictive_two_mode": accepted,
            "validation_status": (
                "accepted_predictive_two_mode"
                if accepted
                else (
                    "failed_dictionary_enlargement_stability"
                    if bool(row.get("accepted_before_stability_check", False))
                    else str(row.get("validation_status", "failed_predeclared_predictive_gate"))
                )
            ),
        })
    return validation_rows


def schur_diagnostic_rows(
    operator: sparse.spmatrix,
    dictionary: DictionaryResult,
    candidates: Sequence[Mapping[str, object]],
    *,
    epsilon_resolvent_values: Sequence[float],
    maximum_cases: int,
) -> list[dict[str, object]]:
    """Evaluate explicitly regularized Schur corrections for selected pairs."""
    rows: list[dict[str, object]] = []
    for candidate in list(candidates)[: int(maximum_cases)]:
        raw_ids = [int(candidate["left_raw_mode_id"]), int(candidate["right_raw_mode_id"])]
        pair = _pair_projection(operator, dictionary.raw_states[:, raw_ids[0]], dictionary.raw_states[:, raw_ids[1]])
        if not bool(pair["valid"]):
            continue
        direct = np.asarray(pair["k"], dtype=float)
        energy = float(np.mean(np.asarray(pair["ritz_values"], dtype=float)))
        for epsilon in epsilon_resolvent_values:
            schur = schur_effective_operator(
                operator,
                np.asarray(pair["u"], dtype=float),
                energy,
                float(epsilon),
            )
            effective = np.asarray(schur.pop("effective_matrix"), dtype=complex)
            real_effective = 0.5 * (effective.real + effective.real.T)
            eigenvalues = np.linalg.eigvalsh(real_effective)
            rows.append({
                "structural_candidate_order": candidate.get("structural_candidate_order"),
                "left_raw_mode_id": raw_ids[0],
                "right_raw_mode_id": raw_ids[1],
                "evaluation_energy": energy,
                "epsilon_resolvent": float(epsilon),
                "epsilon_role": "resolvent regularizer; distinct from eta_couple continuation parameter",
                "direct_K_00": float(direct[0, 0]),
                "direct_K_11": float(direct[1, 1]),
                "direct_K_01": float(direct[0, 1]),
                "schur_diagonal_correction_00_real": float(real_effective[0, 0] - direct[0, 0]),
                "schur_diagonal_correction_11_real": float(real_effective[1, 1] - direct[1, 1]),
                "schur_offdiagonal_correction_01_real": float(real_effective[0, 1] - direct[0, 1]),
                "schur_real_effective_gap": float(eigenvalues[1] - eigenvalues[0]),
                "interpretation": "conditional diagnostic; predictive acceptance still requires captured mass and predeclared gates",
                **schur,
            })
    return rows


def validate_adjacent_pairs_against_dictionary(
    dictionary: DictionaryResult,
    full_values: np.ndarray,
    exact_vector_by_index: Mapping[int, np.ndarray],
    *,
    first_rank: int = 2,
    last_rank: int = 50,
) -> list[dict[str, object]]:
    u = dictionary.orthonormal_states
    k_values, k_vectors = linalg.eigh(dictionary.effective_matrix, check_finite=False)
    ritz_full = u @ k_vectors
    rows: list[dict[str, object]] = []
    n = len(full_values)
    for upper_rank in range(first_rank, last_rank):
        indices = np.asarray([n - (upper_rank + 1), n - upper_rank], dtype=int)
        if any(int(index) not in exact_vector_by_index for index in indices):
            continue
        exact = np.column_stack([exact_vector_by_index[int(index)] for index in indices])
        capture = np.sum(np.square(np.abs(u.T @ exact)), axis=0)
        cost = np.abs(k_values[:, None] - full_values[indices][None, :])
        assignment = linear_sum_assignment(cost)
        chosen = assignment[0]
        predicted = np.sort(k_values[chosen])
        observed = np.asarray(full_values[indices])
        rows.append({
            "left_upper_rank": upper_rank + 1,
            "right_upper_rank": upper_rank,
            "left_ascending_index": int(indices[0]),
            "right_ascending_index": int(indices[1]),
            "observed_gap": float(observed[1] - observed[0]),
            "dictionary_predicted_gap": float(predicted[1] - predicted[0]),
            "dictionary_gap_relative_error": float(abs((predicted[1] - predicted[0]) - (observed[1] - observed[0])) / max(abs(observed[1] - observed[0]), np.finfo(float).eps)),
            "left_dictionary_captured_mass": float(capture[0]),
            "right_dictionary_captured_mass": float(capture[1]),
            "minimum_dictionary_captured_mass": float(np.min(capture)),
            "comparison_role": "multimode dictionary a posteriori validation; not a pairwise tunneling estimate",
        })
    return rows


def fixed_partition_operator(
    operator: sparse.spmatrix,
    labels: np.ndarray,
    eta_couple: float,
) -> sparse.csr_matrix:
    operator = sparse.coo_matrix(operator)
    labels = np.asarray(labels, dtype=int)
    if labels.shape != (operator.shape[0],):
        raise ValueError("partition labels have incompatible shape")
    factors = np.where(labels[operator.row] == labels[operator.col], 1.0, float(eta_couple))
    result = sparse.coo_matrix((operator.data * factors, (operator.row, operator.col)), shape=operator.shape)
    return result.tocsr()


def _top_eigenpairs(operator: sparse.spmatrix, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = operator.shape[0]
    count = min(int(count), n - 2)
    values, vectors = sparse_linalg.eigsh(sparse.csr_matrix(operator), k=count, which="LA", tol=1e-10)
    order = np.argsort(values)[::-1]
    values = np.asarray(values[order], dtype=float)
    vectors = np.asarray(vectors[:, order], dtype=float)
    residuals = np.asarray([
        np.linalg.norm(operator.dot(vectors[:, column]) - values[column] * vectors[:, column]) / max(1.0, abs(values[column]))
        for column in range(count)
    ])
    return values, vectors, residuals


def run_fixed_partition_continuation(
    graph: nx.Graph,
    operator: sparse.spmatrix,
    dictionary: DictionaryResult,
    candidates: Sequence[Mapping[str, object]],
    *,
    eta_values: Sequence[float],
    maximum_cases: int = 2,
    eigenpair_count: int = 64,
    minimum_tracking_overlap: float = 0.50,
    gap_change_factor: float = 4.0,
    maximum_eta_points: int = 25,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Run overlap-tracked continuation with automatic ambiguous-interval refinement."""
    disjoint, labels, partition_rows = disjoint_partition(graph.number_of_nodes(), dictionary.regions)
    selected = list(candidates[: int(maximum_cases)])
    if not selected:
        return [], [], partition_rows
    eta_grid = sorted(set(float(value) for value in eta_values))
    spectra: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def spectrum(eta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if eta not in spectra:
            spectra[eta] = _top_eigenpairs(
                fixed_partition_operator(operator, labels, eta), eigenpair_count
            )
        return spectra[eta]

    def track(grid: Sequence[float]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for case_order, candidate in enumerate(selected, start=1):
            raw_ids = [int(candidate["left_raw_mode_id"]), int(candidate["right_raw_mode_id"])]
            reference = dictionary.raw_states[:, raw_ids]
            reference_orth = orthogonalize_with_truncation(reference, relative_cutoff=1e-10)
            if int(reference_orth["retained_rank"]) != 2:
                continue
            reference_u = np.asarray(reference_orth["states"])
            previous_vectors: np.ndarray | None = None
            for eta in grid:
                values, vectors, residuals = spectrum(float(eta))
                source_vectors = reference_u if previous_vectors is None else previous_vectors
                overlap_all = np.square(np.abs(source_vectors.T @ vectors))
                source_rows, chosen = linear_sum_assignment(-overlap_all)
                ordering = np.argsort(source_rows)
                chosen = np.asarray(chosen[ordering], dtype=int)
                current = vectors[:, chosen].copy()
                assignment_overlaps = overlap_all[np.arange(2), chosen]
                if previous_vectors is not None:
                    for column in range(2):
                        if float(np.dot(previous_vectors[:, column], current[:, column])) < 0.0:
                            current[:, column] *= -1.0
                selected_values = values[chosen]
                gap = float(abs(selected_values[1] - selected_values[0]))
                local_scale = max(1.0, float(np.max(np.abs(selected_values))))
                degeneracy = gap <= 1e-8 * local_scale
                projector_overlap = float(np.linalg.det(source_vectors.T @ current) ** 2)
                for state in range(2):
                    vector = current[:, state]
                    masses = np.square(vector)
                    region_masses = [float(np.sum(masses[nodes])) for nodes in disjoint]
                    reference_weights = np.square(np.abs(reference_u.T @ vector))
                    rows.append({
                        "case_order": case_order,
                        "left_region_id": int(candidate["left_region_id"]),
                        "right_region_id": int(candidate["right_region_id"]),
                        "eta_couple": float(eta),
                        "tracked_state": state + 1,
                        "current_upper_rank_within_computed_window": int(chosen[state] + 1),
                        "energy": float(selected_values[state]),
                        "tracked_pair_gap": gap,
                        "tracked_energy_order_sign": int(np.sign(selected_values[1] - selected_values[0])),
                        "degeneracy_or_unresolved_gap_flag": degeneracy,
                        "assigned_state_overlap": float(assignment_overlaps[state]),
                        "minimum_assignment_overlap": float(np.min(assignment_overlaps)),
                        "tracked_pair_projector_overlap": projector_overlap,
                        "relative_residual": float(residuals[chosen[state]]),
                        "reference_mode_1_weight": float(reference_weights[0]),
                        "reference_mode_2_weight": float(reference_weights[1]),
                        "reference_pair_captured_mass": float(np.sum(reference_weights)),
                        "partition_region_masses": region_masses,
                        "partition_background_mass": float(1.0 - sum(region_masses)),
                        **localization_metrics(vector),
                        "tracking_role": "Hungarian overlap assignment with sign alignment; near-degenerate rows interpreted through pair-projector overlap",
                    })
                previous_vectors = current
        return rows

    refinement_iterations = 0
    while True:
        tracked = track(eta_grid)
        by_case: dict[int, list[dict[str, object]]] = {}
        for row in tracked:
            if int(row["tracked_state"]) == 1:
                by_case.setdefault(int(row["case_order"]), []).append(row)
        intervals: set[tuple[float, float]] = set()
        for case_rows in by_case.values():
            case_rows.sort(key=lambda row: float(row["eta_couple"]))
            for left, right in zip(case_rows[:-1], case_rows[1:]):
                left_eta, right_eta = float(left["eta_couple"]), float(right["eta_couple"])
                left_gap, right_gap = float(left["tracked_pair_gap"]), float(right["tracked_pair_gap"])
                positive_min = max(min(left_gap, right_gap), np.finfo(float).tiny)
                gap_jump = max(left_gap, right_gap) / positive_min
                order_change = int(left["tracked_energy_order_sign"]) != int(right["tracked_energy_order_sign"])
                ambiguous = float(right["minimum_assignment_overlap"]) < float(minimum_tracking_overlap)
                if ambiguous or order_change or gap_jump > float(gap_change_factor):
                    intervals.add((left_eta, right_eta))
        available = max(0, int(maximum_eta_points) - len(eta_grid))
        midpoints = sorted({0.5 * (left + right) for left, right in intervals} - set(eta_grid))[:available]
        if not midpoints:
            break
        eta_grid = sorted(set(eta_grid) | set(midpoints))
        refinement_iterations += 1

    tracked_rows = track(eta_grid)
    for row in tracked_rows:
        row["eta_grid_size"] = len(eta_grid)
        row["automatic_refinement_iterations"] = refinement_iterations
        row["minimum_tracking_overlap_threshold"] = float(minimum_tracking_overlap)
        row["gap_change_refinement_factor"] = float(gap_change_factor)
    rank_rows: list[dict[str, object]] = []
    for eta in eta_grid:
        values, _, residuals = spectrum(eta)
        for rank, (value, residual) in enumerate(zip(values, residuals), start=1):
            rank_rows.append({
                "eta_couple": eta,
                "upper_algebraic_rank": rank,
                "energy": float(value),
                "relative_residual": float(residual),
                "eta_grid_size": len(eta_grid),
                "automatic_refinement_iterations": refinement_iterations,
                "record_role": "rank based; not a tracked-state identity",
            })
    return tracked_rows, rank_rows, partition_rows


def controlled_validation_rows() -> tuple[list[dict[str, object]], bool]:
    """Exercise exact, separated-hub, mediated, duplicate, and third-mode cases."""
    rows: list[dict[str, object]] = []
    all_pass = True
    for detuning in (0.0, 0.2, 1.0):
        for coupling in (0.0, 0.05, 0.4):
            matrix = np.asarray([[0.0, coupling], [coupling, detuning]], dtype=float)
            result = _pair_projection(sparse.csr_matrix(matrix), np.eye(2)[:, 0], np.eye(2)[:, 1])
            exact = np.linalg.eigvalsh(matrix)
            error = abs(float(result["predicted_gap"]) - float(exact[1] - exact[0]))
            passed = error <= 1e-12
            all_pass = all_pass and passed
            rows.append({
                "fixture": "exact_two_state_grid",
                "detuning": detuning,
                "coupling": coupling,
                "predicted_gap": float(result["predicted_gap"]),
                "exact_gap": float(exact[1] - exact[0]),
                "absolute_gap_error": error,
                "expected": "exact",
                "passed": passed,
            })
    for separation in (1, 3, 4, 5):
        for degrees in ((8, 8), (8, 14)):
            graph, left, right = two_hub_graph(degrees[0], degrees[1], separation)
            operator = graph_to_csr(graph, nodelist=range(graph.number_of_nodes()))
            left_nodes = graph_ball(graph, left, 1)
            right_nodes = graph_ball(graph, right, 1)
            _, left_vectors = _largest_local_modes(operator, left_nodes, 1)
            _, right_vectors = _largest_local_modes(operator, right_nodes, 1)
            left_state = np.zeros(graph.number_of_nodes())
            right_state = np.zeros(graph.number_of_nodes())
            left_state[left_nodes] = left_vectors[:, 0]
            right_state[right_nodes] = right_vectors[:, 0]
            result = _pair_projection(operator, left_state, right_state)
            exact = np.linalg.eigvalsh(operator.toarray())
            exact_top_gap = float(exact[-1] - exact[-2])
            mediated_expected = separation >= 4
            direct_zero = bool(abs(float(result.get("direct_coupling", math.nan))) <= 1e-12) if bool(result["valid"]) else False
            behavior_pass = bool(result["valid"] and exact_top_gap > 0.0 and (not mediated_expected or direct_zero))
            all_pass = all_pass and behavior_pass
            rows.append({
                "fixture": "two_hub_graph",
                "left_degree": degrees[0],
                "right_degree": degrees[1],
                "center_distance": separation,
                "direct_coupling": result.get("direct_coupling", math.nan),
                "exact_top_pair_gap": exact_top_gap,
                "expected": "zero-direct-path-mediated" if mediated_expected else "overlap_or_direct-boundary-bridge",
                "passed": behavior_pass,
            })
    duplicate = np.column_stack([np.asarray([1.0, 0.0]), np.asarray([1.0, 0.0])])
    duplicate_result = orthogonalize_with_truncation(duplicate, relative_cutoff=1e-8)
    duplicate_pass = int(duplicate_result["retained_rank"]) == 1
    all_pass = all_pass and duplicate_pass
    rows.append({
        "fixture": "duplicate_reference_regions",
        "retained_rank": int(duplicate_result["retained_rank"]),
        "expected": "truncate_to_one_direction",
        "passed": duplicate_pass,
    })
    path3 = sparse.csr_matrix(np.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]))
    two = _pair_projection(path3, np.eye(3)[:, 0], np.eye(3)[:, 2])
    three = orthogonalize_with_truncation(np.eye(3), relative_cutoff=1e-8)
    three_k = np.asarray(three["states"]).T @ path3.dot(np.asarray(three["states"]))
    three_error = float(np.max(np.abs(np.linalg.eigvalsh(three_k) - np.linalg.eigvalsh(path3.toarray()))))
    third_pass = bool(float(two["predicted_gap"]) == 0.0 and three_error <= 1e-12)
    all_pass = all_pass and third_pass
    rows.append({
        "fixture": "near_resonant_third_mode_path_graph",
        "two_mode_predicted_gap": float(two["predicted_gap"]),
        "three_mode_maximum_energy_error": three_error,
        "expected": "two-mode rejection and exact three-mode escalation",
        "passed": third_pass,
    })
    for epsilon_resolvent in (1e-1, 1e-2, 1e-3):
        schur = schur_effective_operator(
            path3,
            np.asarray(two["u"], dtype=float),
            energy=0.2,
            eta=epsilon_resolvent,
        )
        schur_pass = bool(
            float(schur["absolute_mediated_coupling"]) > 0.0
            and float(schur["linear_solve_residual_norm"]) <= 1e-10
        )
        all_pass = all_pass and schur_pass
        rows.append({
            "fixture": "mediated_path_schur_regularizer_sensitivity",
            "epsilon_resolvent": epsilon_resolvent,
            "eta_couple": math.nan,
            "direct_coupling": float(two["direct_coupling"]),
            "absolute_mediated_coupling": float(schur["absolute_mediated_coupling"]),
            "linear_solve_residual_norm": float(schur["linear_solve_residual_norm"]),
            "expected": "zero direct coupling but nonzero background-mediated Schur coupling",
            "passed": schur_pass,
        })
    return rows, all_pass


def run_retained_clean_pa(
    *,
    source_dir: Path,
    output_dir: Path,
    project_root: Path,
    freeze_manifest: Path,
    config: Mapping[str, object],
) -> Path:
    started = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "arrays").mkdir(parents=True, exist_ok=True)
    source = load_stage1_cache(source_dir)
    freeze = read_json(freeze_manifest)
    source_identity = source_configuration_identity(project_root)
    if freeze.get("status") != "COMPLETE" or freeze.get("source_configuration_id") != source_identity["sha256"]:
        raise ValueError("freeze manifest does not match current source/configuration")
    cache_id = stable_json_hash({
        "version": COUPLING_HANDOFF_VERSION,
        "source_configuration_id": source_identity["sha256"],
        "source_cache_id": source["cache_id"],
        "config": config,
    })
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if existing.get("status") == "COMPLETE" and existing.get("cache_id") == cache_id:
            return output_dir
        raise ValueError("existing clean-PA output has another cache identity")
    manifest = {
        **runtime_metadata(project_root),
        "status": "RUNNING",
        "stage": "retained_clean_pa_structural_coupling_and_continuation",
        "stage_version": COUPLING_HANDOFF_VERSION,
        "cache_id": cache_id,
        "source_configuration_id": source_identity["sha256"],
        "source_cache_id": source["cache_id"],
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "config": dict(config),
    }
    write_json(manifest_path, manifest)
    try:
        graph: nx.Graph = source["graph"]
        operator = sparse.csr_matrix(source["adjacency"])
        full_values = np.asarray(source["full_eigenvalues"], dtype=float)
        selected_indices = np.asarray(source["global_indices"], dtype=int)
        selected_vectors = np.asarray(source["eigenvectors"], dtype=float)
        vector_by_index = {int(index): selected_vectors[:, column] for column, index in enumerate(selected_indices)}
        n = graph.number_of_nodes()
        model_key = Path(source_dir).name
        distance_cache = BFSDistanceCache(graph)
        baseline_rows: list[dict[str, object]] = []
        within_rows: list[dict[str, object]] = []
        path_rows: list[dict[str, object]] = []
        adjacent_rows: list[dict[str, object]] = []
        cross_rows: list[dict[str, object]] = []
        requested_indices = sorted(set(selected_indices.tolist()))
        for index in requested_indices:
            vector = vector_by_index[index]
            upper_rank = n - index
            role = "upper_edge_retained" if upper_rank <= 50 else "retained_bulk_selection"
            prefix = {
                "model_key": model_key,
                "ascending_index": index,
                "upper_algebraic_rank": upper_rank,
                "energy": float(full_values[index]),
                "selection_role": role,
            }
            baseline_rows.append({**prefix, **localization_metrics(vector)})
            if upper_rank <= 50:
                summary, pairs = within_vector_geometry(graph, distance_cache, vector, prefix=prefix)
                within_rows.append(summary)
                if 2 <= upper_rank <= 11:
                    path_rows.extend(pairs)
        for upper_rank in range(2, 50):
            right_index = n - upper_rank
            left_index = n - (upper_rank + 1)
            if right_index not in vector_by_index or left_index not in vector_by_index:
                continue
            prefix = {
                "model_key": model_key,
                "left_upper_rank": upper_rank + 1,
                "right_upper_rank": upper_rank,
                "left_energy": float(full_values[left_index]),
                "right_energy": float(full_values[right_index]),
                "observed_adjacent_gap": float(full_values[right_index] - full_values[left_index]),
            }
            summary, rows = between_vector_geometry(
                distance_cache,
                vector_by_index[left_index],
                vector_by_index[right_index],
                prefix=prefix,
            )
            adjacent_rows.append(summary)
            if upper_rank <= 10:
                cross_rows.extend(rows)
        write_csv(output_dir / "tables/baseline_selected_vector_metrics.csv.gz", baseline_rows)
        write_csv(output_dir / "tables/upper_edge_within_vector_geometry.csv.gz", within_rows)
        write_csv(output_dir / "tables/first_ten_nonperron_all_45_distances.csv.gz", path_rows)
        write_csv(output_dir / "tables/adjacent_pair_geometry_ranks_2_to_50.csv.gz", adjacent_rows)
        write_csv(output_dir / "tables/first_ten_adjacent_cross_distances.csv.gz", cross_rows)

        dictionary_summaries: list[dict[str, object]] = []
        candidate_rows: list[dict[str, object]] = []
        validation_rows: list[dict[str, object]] = []
        multimode_rows: list[dict[str, object]] = []
        continuation_rows: list[dict[str, object]] = []
        continuation_rank_rows: list[dict[str, object]] = []
        partition_rows_all: list[dict[str, object]] = []
        schur_rows: list[dict[str, object]] = []
        gram_sensitivity_rows: list[dict[str, object]] = []
        primary_dictionary: DictionaryResult | None = None
        primary_candidates: list[dict[str, object]] = []
        for radius in config["reference_dictionary"]["radii"]:
            for mode_count in config["reference_dictionary"]["local_mode_counts"]:
                variant = f"radius_{radius}_modes_{mode_count}"
                try:
                    dictionary = build_structural_dictionary(
                        graph,
                        operator,
                        radius=int(radius),
                        local_mode_count=int(mode_count),
                        structural_vertex_count=int(config["reference_dictionary"]["structural_vertex_count"]),
                        overlap_threshold=float(config["reference_dictionary"]["overlap_coefficient_merge_threshold"]),
                        maximum_region_fraction=float(config["reference_dictionary"]["maximum_merged_region_fraction"]),
                        gram_cutoff=float(config["reference_dictionary"]["gram_relative_cutoff"]),
                        maximum_modes=int(config["reference_dictionary"]["maximum_retained_modes"]),
                    )
                    summary = {"model_key": model_key, "variant": variant, "radius": radius, "local_mode_count": mode_count, "construction_status": "COMPLETE", **dictionary_summary(dictionary)}
                    write_csv(output_dir / f"tables/{variant}_regions.csv", dictionary.region_rows)
                    write_csv(output_dir / f"tables/{variant}_raw_modes.csv", dictionary.mode_rows)
                    write_csv(output_dir / f"tables/{variant}_orthogonal_modes.csv", dictionary.orthogonal_rows)
                    write_csv(output_dir / f"tables/{variant}_rejections_and_merges.csv", dictionary.rejected_rows)
                    np.savez_compressed(
                        output_dir / f"arrays/{variant}_dictionary.npz",
                        raw_states=dictionary.raw_states,
                        orthonormal_states=dictionary.orthonormal_states,
                        transform=dictionary.transform,
                        effective_matrix=dictionary.effective_matrix,
                        residual_matrix=dictionary.residual_matrix,
                    )
                    candidates = structurally_ranked_mode_pairs(graph, operator, dictionary, maximum_pairs=20)
                    for row in candidates:
                        row.update({"model_key": model_key, "variant": variant})
                    candidate_rows.extend(candidates)
                    for candidate in candidates:
                        validated = validate_candidate_pair(operator, candidate, dictionary, full_values, vector_by_index)
                        validated.update({"model_key": model_key, "variant": variant})
                        validation_rows.append(validated)
                    for row in validate_adjacent_pairs_against_dictionary(dictionary, full_values, vector_by_index):
                        row.update({"model_key": model_key, "variant": variant})
                        multimode_rows.append(row)
                    if int(radius) == int(config["reference_dictionary"]["primary_radius"]) and int(mode_count) == int(config["reference_dictionary"]["primary_local_mode_count"]):
                        primary_dictionary = dictionary
                        primary_candidates = candidates
                        for cutoff in config["reference_dictionary"]["gram_relative_cutoff_sensitivity"]:
                            gram_result = orthogonalize_with_truncation(
                                dictionary.raw_states, relative_cutoff=float(cutoff)
                            )
                            gram_u = np.asarray(gram_result["states"], dtype=float)
                            gram_k = gram_u.T @ operator.dot(gram_u)
                            gram_r = operator.dot(gram_u) - gram_u @ gram_k
                            gram_sensitivity_rows.append({
                                "model_key": model_key,
                                "variant": variant,
                                "gram_relative_cutoff": float(cutoff),
                                "retained_rank": int(gram_result["retained_rank"]),
                                "rejected_rank": int(gram_result["rejected_rank"]),
                                "orthonormality_error": float(gram_result["orthonormality_error"]),
                                "residual_frobenius_norm": float(np.linalg.norm(gram_r)),
                                "residual_operator_norm": float(np.linalg.svd(gram_r, compute_uv=False)[0]),
                            })
                except RuntimeError as error:
                    summary = {"model_key": model_key, "variant": variant, "radius": radius, "local_mode_count": mode_count, "construction_status": "INADEQUATE", "failure_reason": str(error)}
                dictionary_summaries.append(summary)
        primary_variant = (
            f"radius_{config['reference_dictionary']['primary_radius']}_"
            f"modes_{config['reference_dictionary']['primary_local_mode_count']}"
        )
        enlarged_variant = (
            f"radius_{config['reference_dictionary']['primary_radius']}_"
            f"modes_{config['reference_dictionary']['enlarged_local_mode_count']}"
        )
        apply_dictionary_enlargement_gate(
            validation_rows,
            primary_variant=primary_variant,
            enlarged_variant=enlarged_variant,
            gap_relative_tolerance=float(config["acceptance_gates"]["enlargement_gap_relative_tolerance"]),
            weight_absolute_tolerance=float(config["acceptance_gates"]["enlargement_weight_absolute_tolerance"]),
        )
        write_csv(output_dir / "tables/dictionary_summary.csv", dictionary_summaries)
        write_csv(output_dir / "tables/gram_cutoff_sensitivity.csv", gram_sensitivity_rows)
        write_csv(output_dir / "tables/structural_candidate_pairs.csv.gz", candidate_rows)
        write_csv(output_dir / "tables/predictive_pair_validation.csv.gz", validation_rows)
        write_csv(output_dir / "tables/adjacent_pair_multimode_dictionary_validation.csv.gz", multimode_rows)

        if primary_dictionary is not None and primary_candidates:
            schur_rows = schur_diagnostic_rows(
                operator,
                primary_dictionary,
                primary_candidates,
                epsilon_resolvent_values=config["schur"]["epsilon_resolvent_values"],
                maximum_cases=int(config["schur"]["maximum_cases_per_graph"]),
            )
            for row in schur_rows:
                row.update({"model_key": model_key, "variant": primary_variant})
            continuation_rows, continuation_rank_rows, partition_rows = run_fixed_partition_continuation(
                graph,
                operator,
                primary_dictionary,
                primary_candidates,
                eta_values=config["continuation"]["eta_couple_values"],
                maximum_cases=int(config["continuation"]["maximum_cases_per_graph"]),
                eigenpair_count=int(config["continuation"]["computed_upper_eigenpair_count"]),
                minimum_tracking_overlap=float(config["continuation"]["minimum_tracking_overlap"]),
                gap_change_factor=float(config["continuation"]["gap_change_refinement_factor"]),
                maximum_eta_points=int(config["continuation"]["maximum_eta_points"]),
            )
            for row in continuation_rows:
                row["model_key"] = model_key
            for row in continuation_rank_rows:
                row["model_key"] = model_key
            for row in partition_rows:
                row["model_key"] = model_key
            partition_rows_all.extend(partition_rows)
        write_csv(output_dir / "tables/schur_feshbach_diagnostics.csv.gz", schur_rows)
        write_csv(output_dir / "tables/continuation_tracked_modes.csv.gz", continuation_rows)
        write_csv(output_dir / "tables/continuation_rank_spectra.csv.gz", continuation_rank_rows)
        write_csv(output_dir / "tables/continuation_partition.csv", partition_rows_all)
        manifest.update({
            "status": "COMPLETE",
            "finished_at_utc": utc_now(),
            "runtime_seconds": time.time() - started,
            "model_key": model_key,
            "baseline_vector_count": len(baseline_rows),
            "adjacent_pair_geometry_count": len(adjacent_rows),
            "dictionary_variant_count": len(dictionary_summaries),
            "structural_candidate_count": len(candidate_rows),
            "accepted_predictive_two_mode_count": sum(bool(row.get("accepted_predictive_two_mode", False)) for row in validation_rows),
            "schur_diagnostic_row_count": len(schur_rows),
            "continuation_row_count": len(continuation_rows),
            "scientific_status": "completed including failed/inadequate reference constructions; no acceptance inferred from completion",
        })
        write_json(manifest_path, manifest)
        return output_dir
    except Exception as error:
        manifest.update({
            "status": "FAILED",
            "finished_at_utc": utc_now(),
            "runtime_seconds": time.time() - started,
            "failure_type": type(error).__name__,
            "failure_message": str(error),
        })
        write_json(manifest_path, manifest)
        raise
