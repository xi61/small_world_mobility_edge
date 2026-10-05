"""Anderson-only execution utilities isolated from the clean-PA workflow."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import networkx as nx
import numpy as np
from scipy import sparse

from .common import (
    BFSDistanceCache,
    analyzed_to_original_ids,
    deterministic_mass_order,
    read_json,
    sha256_file,
    source_configuration_identity,
    stable_json_hash,
    write_csv,
    write_json,
)
from .handoff_metrics import (
    between_vector_geometry,
    gap_triplet_rows,
    predeclared_windows,
    selected_vector_rows,
    stable_zero_cluster,
    within_vector_geometry,
)


ANDERSON_UTILS_VERSION = "anderson_repair_extension_utils_v1"


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + path.name + ".", suffix=".json", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        write_json(temporary, payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path = Path(path)
    if not rows:
        return
    suffix = ".csv.gz" if path.suffix == ".gz" else ".csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + path.name + ".", suffix=suffix, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        write_csv(temporary, rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_savez(path: Path, *, compressed: bool = True, **arrays: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + path.name + ".", suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        if compressed:
            np.savez_compressed(temporary, **arrays)
        else:
            np.savez(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def verify_execution_context(project_root: Path, config_path: Path, freeze_manifest: Path) -> dict[str, object]:
    project_root, config_path, freeze_manifest = Path(project_root), Path(config_path), Path(freeze_manifest)
    config = read_json(config_path)
    freeze = read_json(freeze_manifest)
    current = source_configuration_identity(project_root)
    if freeze.get("status") != "COMPLETE":
        raise ValueError("freeze is not complete")
    if freeze.get("source_configuration_id") != current.get("sha256"):
        raise ValueError("live source/configuration does not match immutable freeze")
    frozen_config_sha = freeze.get("configuration_sha256")
    if frozen_config_sha is not None and frozen_config_sha != sha256_file(config_path):
        raise ValueError("configuration hash differs from freeze")
    return {
        "config": config,
        "source_configuration_identity": current,
        "configuration_sha256": sha256_file(config_path),
        "freeze_manifest_sha256": sha256_file(freeze_manifest),
        "freeze_label": freeze.get("freeze_label"),
    }


def width_count_quantile_indices(n: int, quantile: float, count: int) -> np.ndarray:
    if n < 3 or count < 3 or count > n or not 0.0 <= quantile <= 1.0:
        raise ValueError("invalid fixed-count quantile window")
    if count % 2 == 0:
        raise ValueError("fixed-count quantile window must have odd count")
    center = int(round(float(quantile) * (n - 1)))
    lo = max(0, center - count // 2)
    hi = min(n, lo + count)
    lo = max(0, hi - count)
    return np.arange(lo, hi, dtype=int)


def window_registry(values: np.ndarray, *, pa_physical_half_width: float, quantile_width: float = 0.02) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    windows = predeclared_windows(
        values,
        quantile_width=float(quantile_width),
        physical_half_width=float(pa_physical_half_width),
    )
    windows.update({
        "lower_rank_1_5": np.arange(0, min(5, n), dtype=int),
        "lower_rank_6_10": np.arange(5, min(10, n), dtype=int),
        "upper_rank_1_5": np.arange(max(0, n - 5), n, dtype=int),
        "upper_rank_6_10": np.arange(max(0, n - 10), max(0, n - 5), dtype=int),
        "q25_width_51": width_count_quantile_indices(n, 0.25, 51),
        "q50_width_51": width_count_quantile_indices(n, 0.50, 51),
        "q75_width_51": width_count_quantile_indices(n, 0.75, 51),
    })
    return windows


def selection_union(windows: Mapping[str, np.ndarray], *, n: int, guard: int = 1, atom_mask: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    expanded: dict[str, np.ndarray] = {}
    for name, members in windows.items():
        members = np.asarray(members, dtype=int)
        if len(members) == 0:
            expanded[name] = members
            continue
        lo = max(0, int(np.min(members)) - int(guard))
        hi = min(n, int(np.max(members)) + int(guard) + 1)
        expanded[name] = np.arange(lo, hi, dtype=int)
    union = np.unique(np.concatenate([v for v in expanded.values() if len(v)])) if expanded else np.empty(0, dtype=int)
    if atom_mask is not None:
        mask = np.asarray(atom_mask, dtype=bool)
        if mask.shape != (n,):
            raise ValueError("atom mask shape mismatch")
        union = union[~mask[union]]
    return union.astype(int), expanded


def graph_identity(graph_cache: Path, *, operator: str, source_id: str) -> str:
    return stable_json_hash({
        "graph_sha256": sha256_file(graph_cache),
        "operator": operator,
        "source_configuration_id": source_id,
    })


def potential_identity(xi: np.ndarray, *, graph_id: str, seed: int) -> str:
    xi = np.asarray(xi, dtype=np.float64)
    return stable_json_hash({
        "graph_id": graph_id,
        "seed": int(seed),
        "law": "numpy Generator(PCG64).uniform(-1,1)",
        "xi_sha256": hashlib.sha256(xi.tobytes(order="C")).hexdigest(),
    })


def array_hash(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def validate_selected_eigenpairs(
    operator: sparse.spmatrix | np.ndarray,
    values: np.ndarray,
    vectors: np.ndarray,
    *,
    sample_orthogonality_cap: int = 128,
) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    vectors = np.asarray(vectors, dtype=float)
    if vectors.shape != (operator.shape[0], len(values)):
        raise ValueError("selected eigenpair dimensions disagree")
    norms = np.linalg.norm(vectors, axis=0)
    residual = operator.dot(vectors) - vectors * values[np.newaxis, :]
    scale = np.maximum(1.0, np.abs(values))
    relative = np.linalg.norm(residual, axis=0) / scale
    positions = np.linspace(0, max(0, len(values) - 1), min(sample_orthogonality_cap, len(values)), dtype=int)
    sampled = vectors[:, positions]
    gram = sampled.T @ sampled
    return {
        "maximum_norm_error": float(np.max(np.abs(norms - 1.0))) if len(norms) else 0.0,
        "maximum_relative_residual": float(np.max(relative)) if len(relative) else 0.0,
        "sampled_orthogonality_spectral_error": float(np.linalg.norm(gram - np.eye(len(positions)), ord=2)) if len(positions) else 0.0,
        "validated_eigenpair_count": int(len(values)),
        "orthogonality_sample_count": int(len(positions)),
    }


def shell_ball_rows(
    distance_cache: BFSDistanceCache,
    vector: np.ndarray,
    *,
    prefix: Mapping[str, object],
    max_radius: int,
) -> list[dict[str, object]]:
    order, masses = deterministic_mass_order(vector)
    center = int(order[0])
    distances = distance_cache.row_array(center)
    n = len(masses)
    rows: list[dict[str, object]] = []
    for radius in range(int(max_radius) + 1):
        shell = distances == radius
        ball = (distances >= 0) & (distances <= radius)
        rows.append({
            **prefix,
            "center_vertex": center,
            "radius": radius,
            "shell_vertex_count": int(np.count_nonzero(shell)),
            "ball_vertex_count": int(np.count_nonzero(ball)),
            "shell_vertex_fraction": float(np.count_nonzero(shell) / n),
            "ball_vertex_fraction": float(np.count_nonzero(ball) / n),
            "shell_mass": float(np.sum(masses[shell])),
            "ball_mass": float(np.sum(masses[ball])),
        })
    return rows


def checkpoint_diagnostics(
    *,
    graph: nx.Graph,
    adjacency: sparse.spmatrix,
    values: np.ndarray,
    ipr: np.ndarray,
    full_maximum_mass: np.ndarray | None,
    selected_indices: np.ndarray,
    selected_vectors: np.ndarray,
    potential: np.ndarray | None,
    amplitude: float,
    prefix: Mapping[str, object],
    physical_half_width: float,
    atom_enabled: bool,
    shell_max_radius: int = 5,
) -> dict[str, list[dict[str, object]] | dict[str, object]]:
    values = np.asarray(values, dtype=float)
    ipr = np.asarray(ipr, dtype=float)
    selected_indices = np.asarray(selected_indices, dtype=int)
    selected_vectors = np.asarray(selected_vectors, dtype=float)
    n = len(values)
    if ipr.shape != (n,) or selected_vectors.shape != (n, len(selected_indices)):
        raise ValueError("checkpoint arrays have incompatible dimensions")
    if np.any(np.diff(values) < -1e-10):
        raise ValueError("eigenvalues are not ascending")
    windows = window_registry(values, pa_physical_half_width=physical_half_width)
    zero = stable_zero_cluster(values, enabled=atom_enabled)
    atom_mask = np.asarray(zero["mask"], dtype=bool)
    selected_position = {int(index): position for position, index in enumerate(selected_indices)}
    vector_rows = selected_vector_rows(
        values,
        selected_vectors,
        selected_indices,
        prefix=prefix,
        windows=windows,
        potential=potential,
        disorder_amplitude=float(amplitude),
        adjacency=adjacency,
    )
    degrees = np.asarray([graph.degree(vertex) for vertex in range(n)], dtype=float)
    original_ids = analyzed_to_original_ids(graph)
    for row, column in zip(vector_rows, range(len(selected_indices))):
        index = int(row["ascending_index"])
        masses = np.square(np.abs(selected_vectors[:, column]))
        row["dominant_original_vertex_id"] = int(original_ids[int(row["dominant_vertex"])])
        row["mass_weighted_degree"] = float(np.dot(masses, degrees))
        if potential is None:
            vector = selected_vectors[:, column]
            adjacency_energy = float(np.real(np.vdot(vector, adjacency.dot(vector))))
            row.update({
                "center_potential": math.nan,
                "weighted_potential_mean": math.nan,
                "diagonal_energy_contribution": 0.0,
                "adjacency_energy_contribution": adjacency_energy,
                "energy_identity_reconstruction": adjacency_energy,
                "energy_identity_absolute_error": abs(float(values[index]) - adjacency_energy),
            })
        row["atom_member"] = bool(atom_mask[index])
        row["individual_vector_physical_comparison_available"] = not bool(atom_mask[index])
        if atom_mask[index]:
            row["basis_dependent_cached_ipr_provenance"] = row["ipr"]
            for key in list(row):
                if key in {"ipr", "n_times_ipr", "participation_ratio", "participation_fraction", "maximum_absolute_coordinate", "maximum_coordinate_mass"} or key.startswith("top_") or key.startswith("support_"):
                    row[key] = math.nan
    spectrum_rows: list[dict[str, object]] = []
    maximum_mass = None if full_maximum_mass is None else np.asarray(full_maximum_mass, dtype=float)
    for index in range(n):
        value_ipr = float(ipr[index])
        inside_atom = bool(atom_mask[index])
        spectrum_rows.append({
            **prefix,
            "ascending_index": index,
            "energy": float(values[index]),
            "ipr": math.nan if inside_atom else value_ipr,
            "basis_dependent_ipr_provenance": value_ipr if inside_atom else math.nan,
            "n_times_ipr": math.nan if inside_atom else float(n * value_ipr),
            "participation_ratio": math.nan if inside_atom else float(1.0 / value_ipr),
            "participation_fraction": math.nan if inside_atom else float(1.0 / (n * value_ipr)),
            "maximum_coordinate_mass": float(maximum_mass[index]) if not inside_atom and maximum_mass is not None and np.isfinite(maximum_mass[index]) else math.nan,
            "atom_member": inside_atom,
            "raw_vector_retained": index in selected_position,
        })
    window_rows: list[dict[str, object]] = []
    gap_rows: list[dict[str, object]] = []
    for name, members in windows.items():
        members = np.asarray(members, dtype=int)
        usable = members[~atom_mask[members]]
        atom_intersection = int(np.count_nonzero(atom_mask[members]))
        ipr_values = ipr[usable]
        raw_coverage = int(sum(int(index) in selected_position for index in members))
        window_gap = gap_triplet_rows(
            values,
            members,
            atom_mask=atom_mask,
            prefix={**prefix, "window": name},
        )
        for row in window_gap:
            row["middle_level_belongs_to_window"] = True
            row["uses_boundary_guard"] = int(row["middle_ascending_index"]) in {int(members[0]), int(members[-1])} if len(members) else False
        gap_rows.extend(window_gap)
        valid_ratios = np.asarray([float(row["raw_gap_ratio"]) for row in window_gap if not bool(row["atom_triplet_excluded"]) and math.isfinite(float(row["raw_gap_ratio"]))], dtype=float)
        window_rows.append({
            **prefix,
            "window": name,
            "first_ascending_index": int(members[0]) if len(members) else -1,
            "last_ascending_index": int(members[-1]) if len(members) else -1,
            "scalar_level_count": int(len(members)),
            "nonatom_scalar_level_count": int(len(usable)),
            "atom_intersection_count": atom_intersection,
            "raw_vector_coverage_count": raw_coverage,
            "raw_vector_coverage_fraction": float(raw_coverage / len(members)) if len(members) else math.nan,
            "mean_ipr": float(np.mean(ipr_values)) if len(ipr_values) and atom_intersection == 0 else math.nan,
            "median_ipr": float(np.median(ipr_values)) if len(ipr_values) and atom_intersection == 0 else math.nan,
            "median_n_times_ipr": float(np.median(n * ipr_values)) if len(ipr_values) and atom_intersection == 0 else math.nan,
            "median_participation_fraction": float(np.median(1.0 / (n * ipr_values))) if len(ipr_values) and atom_intersection == 0 else math.nan,
            "candidate_gap_triplet_count": int(len(window_gap)),
            "valid_gap_triplet_count": int(len(valid_ratios)),
            "atom_excluded_gap_triplet_count": int(sum(bool(row["atom_triplet_excluded"]) for row in window_gap)),
            "boundary_guard_triplet_count": int(sum(bool(row["uses_boundary_guard"]) for row in window_gap)),
            "mean_raw_gap_ratio": float(np.mean(valid_ratios)) if len(valid_ratios) else math.nan,
            "median_raw_gap_ratio": float(np.median(valid_ratios)) if len(valid_ratios) else math.nan,
            "window_selection": "central level belongs to window; immediate full-spectrum neighbors are guards",
        })
    distance_cache = BFSDistanceCache(graph)
    within_summary: list[dict[str, object]] = []
    within_pairs: list[dict[str, object]] = []
    between_summary: list[dict[str, object]] = []
    adjacent_cross: list[dict[str, object]] = []
    shell_rows: list[dict[str, object]] = []
    for edge in ("lower", "upper"):
        indices = np.arange(0, min(10, n), dtype=int) if edge == "lower" else np.arange(max(0, n - 10), n, dtype=int)[::-1]
        available = [int(index) for index in indices if int(index) in selected_position and not atom_mask[int(index)]]
        for rank_position, index in enumerate(available, start=1):
            vector = selected_vectors[:, selected_position[index]]
            vector_prefix = {**prefix, "edge": edge, "edge_rank_1based": rank_position, "ascending_index": index, "energy": float(values[index])}
            summary, pairs = within_vector_geometry(graph, distance_cache, vector, prefix=vector_prefix, potential=potential)
            top_vertices = np.asarray(summary["top10_vertices"], dtype=int)
            summary["top10_original_vertex_ids"] = analyzed_to_original_ids(graph, top_vertices).tolist()
            within_summary.append(summary)
            within_pairs.extend(pairs)
            shell_rows.extend(shell_ball_rows(distance_cache, vector, prefix=vector_prefix, max_radius=shell_max_radius))
        for left_position in range(len(available)):
            for right_position in range(left_position + 1, len(available)):
                left_index, right_index = available[left_position], available[right_position]
                pair_prefix = {
                    **prefix,
                    "edge": edge,
                    "left_edge_rank_1based": left_position + 1,
                    "right_edge_rank_1based": right_position + 1,
                    "left_ascending_index": left_index,
                    "right_ascending_index": right_index,
                    "spectrally_adjacent_within_edge_order": right_position == left_position + 1,
                }
                summary, cross = between_vector_geometry(
                    distance_cache,
                    selected_vectors[:, selected_position[left_index]],
                    selected_vectors[:, selected_position[right_index]],
                    prefix=pair_prefix,
                )
                between_summary.append(summary)
                if right_position == left_position + 1:
                    adjacent_cross.extend(cross)
    return {
        "spectrum_rows": spectrum_rows,
        "window_rows": window_rows,
        "gap_rows": gap_rows,
        "selected_vector_rows": vector_rows,
        "within_vector_rows": within_summary,
        "within_distance_rows": within_pairs,
        "between_vector_rows": between_summary,
        "adjacent_cross_distance_rows": adjacent_cross,
        "shell_ball_rows": shell_rows,
        "zero_cluster": {key: value for key, value in zero.items() if key != "mask"},
        "distance_cache_source_count": distance_cache.source_count,
    }


def write_checkpoint_tables(output_dir: Path, bundle: Mapping[str, object]) -> None:
    output_dir = Path(output_dir)
    table_names = {
        "spectrum_rows": "eigenpair_scalars.csv.gz",
        "window_rows": "window_summary.csv",
        "gap_rows": "gap_triplets.csv.gz",
        "selected_vector_rows": "selected_vector_metrics.csv.gz",
        "within_vector_rows": "within_vector_geometry.csv.gz",
        "within_distance_rows": "within_vector_distances.csv.gz",
        "between_vector_rows": "edge_pair_geometry.csv.gz",
        "adjacent_cross_distance_rows": "adjacent_cross_distances.csv.gz",
        "shell_ball_rows": "shell_ball_mass.csv.gz",
    }
    for key, filename in table_names.items():
        rows = bundle.get(key, [])
        if rows:
            atomic_write_csv(output_dir / "tables" / filename, rows)  # type: ignore[arg-type]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))
