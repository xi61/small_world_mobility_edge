"""Raw-amplitude paired diagonal disorder on exact clean affine-PA graphs.

Production graph generation is intentionally absent from this module.  Every
production case must resolve to a completed clean Part-1 cache.  The only graph
generation entry point is the explicitly labelled small controlled pilot.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import networkx as nx
import numpy as np
from scipy import linalg, sparse

from small_world_mobility_edge.preferential_attachment.baseline import (
    MODEL_AFFINE_PA,
    generate_preferential_attachment_graph,
)
from small_world_mobility_edge.preferential_attachment.linear_pa_two_part import (
    _coordinate_and_geometry_tables,
    _cumulative_and_disjoint_windows,
    _edge_hash,
    _multiplicity_scale,
    alpha_slug,
    quantile_rank_windows,
)
from small_world_mobility_edge.random_schrodinger.anderson_utils import (
    atomic_savez,
    atomic_write_csv,
    atomic_write_json,
    array_hash,
)
from small_world_mobility_edge.random_schrodinger.common import (
    graph_from_edge_cache,
    graph_to_csr,
    read_json,
    save_graph_cache,
    sha256_file,
    stable_json_hash,
    utc_now,
)
from small_world_mobility_edge.random_schrodinger.handoff_metrics import stable_zero_cluster


PA_UNIFORM_NOISE_VERSION = "pa_diagonal_uniform_noise_raw_amplitude_v1"


def amplitude_slug(amplitude: float) -> str:
    text = f"{float(amplitude):.12g}"
    return text.replace("-", "m").replace(".", "p")


def case_directory(root: Path, n: int, alpha: float) -> Path:
    return Path(root) / "cases" / f"N_{int(n)}" / f"alpha_{alpha_slug(alpha)}"


def clean_case_directory(clean_root: Path, n: int, alpha: float) -> Path:
    return Path(clean_root) / f"N_{int(n)}" / f"alpha_{alpha_slug(alpha)}"


def operator_directory(root: Path, n: int, alpha: float, amplitude: float) -> Path:
    return case_directory(root, n, alpha) / "operators" / f"a_{amplitude_slug(amplitude)}"


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".npy", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.save(temporary, np.asarray(array))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_configuration(path: Path) -> dict[str, object]:
    with Path(path).open() as handle:
        return json.load(handle)


def potential_seed(config: Mapping[str, object], n: int, alpha: float) -> int:
    alpha_grid = [float(value) for value in config["degree_ccdf_alpha_grid"]]
    matches = [index for index, value in enumerate(alpha_grid) if math.isclose(value, float(alpha), rel_tol=0.0, abs_tol=1e-12)]
    if len(matches) != 1:
        raise ValueError("alpha is not uniquely present in the configured grid")
    seed = np.random.SeedSequence([
        int(config["potential_base_seed"]),
        int(n),
        int(matches[0]),
        99173,
    ]).generate_state(1, dtype=np.uint64)[0]
    return int(seed)


def draw_potential(config: Mapping[str, object], n: int, alpha: float) -> tuple[int, np.ndarray]:
    seed = potential_seed(config, n, alpha)
    xi = np.random.Generator(np.random.PCG64(seed)).uniform(-1.0, 1.0, size=int(n)).astype(np.float64)
    return seed, xi


def _clean_case_evidence(clean_root: Path, n: int, alpha: float, m: int) -> dict[str, object]:
    clean_dir = clean_case_directory(clean_root, n, alpha)
    required = {
        "resolved_case": clean_dir / "resolved_case.json",
        "graph_manifest": clean_dir / "graph_cache_manifest.json",
        "part1_manifest": clean_dir / "part1_manifest.json",
        "spectral_manifest": clean_dir / "spectral_cache_manifest.json",
        "graph_cache": clean_dir / "arrays/graph_edges.npz",
        "eigenvalues": clean_dir / "arrays/full_eigenvalues_ascending.npy",
        "eigenvectors": clean_dir / "arrays/full_eigenvectors_ascending.npy",
        "absolute_residuals": clean_dir / "arrays/full_absolute_residuals.npy",
        "relative_residuals": clean_dir / "arrays/full_relative_residuals.npy",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("clean Part-1 prerequisite missing: " + "; ".join(missing))
    resolved = read_json(required["resolved_case"])
    graph_manifest = read_json(required["graph_manifest"])
    part1 = read_json(required["part1_manifest"])
    spectral = read_json(required["spectral_manifest"])
    if part1.get("status") != "COMPLETE" or spectral.get("status") != "COMPLETE":
        raise ValueError("clean Part-1 or spectral manifest is not COMPLETE")
    expected_delta = int(m) * (float(alpha) - 2.0)
    checks = {
        "n": int(resolved.get("n", -1)) == int(n),
        "m": int(resolved.get("m", -1)) == int(m),
        "alpha": math.isclose(float(resolved.get("degree_ccdf_alpha", math.nan)), float(alpha), rel_tol=0.0, abs_tol=1e-12),
        "offset": math.isclose(float(resolved.get("attachment_offset", math.nan)), expected_delta, rel_tol=0.0, abs_tol=1e-12),
        "operator": resolved.get("operator") == "unscaled simple undirected adjacency in vertex order 0..N-1",
    }
    if not all(checks.values()):
        raise ValueError(f"clean resolved case failed identity checks: {checks}")
    graph, cache_metadata = graph_from_edge_cache(required["graph_cache"])
    canonical_hash = _edge_hash(graph)
    if graph.number_of_nodes() != int(n) or set(graph.nodes()) != set(range(int(n))):
        raise ValueError("clean graph vertex ordering is not identity 0..N-1")
    if graph_manifest.get("graph_sha256_edges") != canonical_hash:
        raise ValueError("clean graph canonical edge hash disagrees with graph manifest")
    if graph_manifest.get("simple_undirected") is not True or nx.number_of_selfloops(graph):
        raise ValueError("clean graph is not simple undirected")
    values = np.load(required["eigenvalues"], mmap_mode="r")
    vectors = np.load(required["eigenvectors"], mmap_mode="r")
    if values.shape != (int(n),) or vectors.shape != (int(n), int(n)):
        raise ValueError("clean eigensystem dimensions are incompatible")
    return {
        "clean_case_dir": str(clean_dir),
        "clean_case_id": resolved.get("case_id"),
        "resolved_case": resolved,
        "graph_manifest": graph_manifest,
        "part1_manifest_sha256": sha256_file(required["part1_manifest"]),
        "spectral_manifest": spectral,
        "spectral_manifest_sha256": sha256_file(required["spectral_manifest"]),
        "graph_cache_path": str(required["graph_cache"]),
        "graph_cache_sha256": sha256_file(required["graph_cache"]),
        "canonical_edge_hash": canonical_hash,
        "vertex_order_identity": cache_metadata["original_vertex_ids_sha256_raw_int64"],
        "graph_seed": int(resolved["graph_seed"]),
        "num_edges": int(graph.number_of_edges()),
        "paths": {key: str(value) for key, value in required.items()},
    }


def prepare_case(
    *,
    config: Mapping[str, object],
    run_root: Path,
    clean_root: Path,
    n: int,
    alpha: float,
    source_identity: str,
) -> Path:
    evidence = _clean_case_evidence(clean_root, n, alpha, int(config["m"]))
    output = case_directory(run_root, n, alpha)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "case_manifest.json"
    seed, xi = draw_potential(config, n, alpha)
    xi_hash = array_hash(xi)
    identity = stable_json_hash({
        "version": PA_UNIFORM_NOISE_VERSION,
        "source_identity": source_identity,
        "experiment_id": config["experiment_id"],
        "n": int(n),
        "alpha": float(alpha),
        "graph_hash": evidence["canonical_edge_hash"],
        "graph_cache_sha256": evidence["graph_cache_sha256"],
        "potential_seed": seed,
        "xi_hash": xi_hash,
    })
    if manifest_path.exists():
        old = read_json(manifest_path)
        if old.get("status") == "COMPLETE" and old.get("case_identity") == identity:
            stored = np.load(output / "arrays/xi.npy")
            if array_hash(stored) != xi_hash:
                raise ValueError("stored xi disagrees with deterministic replay")
            return output
        if old.get("case_identity") not in (None, identity):
            raise ValueError("existing case directory has an incompatible identity")
    atomic_save_npy(output / "arrays/xi.npy", xi)
    amplitudes = [float(value) for value in config["amplitudes"]]
    payload = {
        "status": "COMPLETE",
        "stage": "prepare_exact_clean_graph_and_paired_potential",
        "case_identity": identity,
        "source_identity": source_identity,
        "experiment_id": config["experiment_id"],
        "n": int(n),
        "degree_ccdf_alpha": float(alpha),
        "alpha_convention": "degree CCDF tail parameter for affine linear PA",
        "m": int(config["m"]),
        "amplitudes_raw": amplitudes,
        "graph_evidence": evidence,
        "potential_seed": seed,
        "potential_seed_derivation": config["potential_seed_derivation"],
        "potential_rng": config["potential_rng"],
        "xi_path": str(output / "arrays/xi.npy"),
        "xi_sha256_raw_float64": xi_hash,
        "xi_file_sha256": sha256_file(output / "arrays/xi.npy"),
        "xi_minimum": float(np.min(xi)),
        "xi_maximum": float(np.max(xi)),
        "xi_mean": float(np.mean(xi)),
        "xi_variance_population": float(np.var(xi)),
        "pairing_design": "one iid vertex potential stream reused across every raw amplitude on this graph; amplitudes are paired, vertices are iid",
        "finished_at_utc": utc_now(),
    }
    atomic_write_json(manifest_path, payload)
    return output


def load_prepared_case(run_root: Path, n: int, alpha: float) -> tuple[dict[str, object], nx.Graph, sparse.csr_matrix, np.ndarray]:
    case_dir = case_directory(run_root, n, alpha)
    manifest = read_json(case_dir / "case_manifest.json")
    if manifest.get("status") != "COMPLETE":
        raise ValueError("prepared case manifest is not COMPLETE")
    graph_path = Path(manifest["graph_evidence"]["graph_cache_path"])
    graph, _ = graph_from_edge_cache(graph_path)
    if _edge_hash(graph) != manifest["graph_evidence"]["canonical_edge_hash"]:
        raise ValueError("prepared graph edge identity changed")
    xi = np.load(case_dir / "arrays/xi.npy")
    if array_hash(xi) != manifest["xi_sha256_raw_float64"]:
        raise ValueError("prepared potential identity changed")
    adjacency = graph_to_csr(graph, nodelist=range(int(n)))
    return manifest, graph, adjacency, np.asarray(xi, dtype=np.float64)


def construct_operator(adjacency: sparse.csr_matrix, xi: np.ndarray, amplitude: float) -> tuple[np.ndarray, dict[str, object]]:
    n = adjacency.shape[0]
    xi = np.asarray(xi, dtype=np.float64)
    if xi.shape != (n,):
        raise ValueError("xi dimension disagrees with adjacency")
    potential = float(amplitude) * xi
    dense = adjacency.toarray(order="F")
    diagonal = np.diag_indices(n)
    dense[diagonal] = potential
    symmetry_error = float(np.max(np.abs(dense - dense.T)))
    diagonal_error = float(np.max(np.abs(np.diag(dense) - potential)))
    offdiag_nonzero = int(np.count_nonzero(dense) - np.count_nonzero(potential))
    offdiag_square_sum = float(np.sum(np.square(dense)) - np.sum(np.square(potential)))
    expected_offdiag = 2 * int(adjacency.nnz // 2)
    invariant = {
        "symmetry_maximum_absolute_error": symmetry_error,
        "diagonal_maximum_absolute_error": diagonal_error,
        "diagonal_minimum": float(np.min(potential)),
        "diagonal_maximum": float(np.max(potential)),
        "diagonal_bound": float(amplitude),
        "diagonal_bounds_pass": bool(np.all(potential >= -float(amplitude)) and np.all(potential <= float(amplitude))),
        "offdiagonal_nonzero_count": offdiag_nonzero,
        "expected_offdiagonal_nonzero_count": expected_offdiag,
        "offdiagonal_square_sum": offdiag_square_sum,
        "expected_offdiagonal_square_sum": float(expected_offdiag),
        "offdiagonal_unchanged_pass": bool(offdiag_nonzero == expected_offdiag and abs(offdiag_square_sum - expected_offdiag) <= 1e-8 * max(1, expected_offdiag)),
        "matrix_trace_direct": float(np.trace(dense)),
        "potential_sum": float(np.sum(potential)),
    }
    if symmetry_error > 1e-12 or diagonal_error > 1e-12 or not invariant["diagonal_bounds_pass"] or not invariant["offdiagonal_unchanged_pass"]:
        raise AssertionError(f"operator construction invariant failed: {invariant}")
    return dense, invariant


def verify_eigensystem(
    operator: sparse.csr_matrix,
    values: np.ndarray,
    vectors: np.ndarray,
    *,
    residual_block_size: int,
    orthogonality_sample_size: int,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    vectors = np.asarray(vectors, dtype=np.float64)
    n = len(values)
    absolute = np.empty(n, dtype=np.float64)
    relative = np.empty(n, dtype=np.float64)
    norms2 = np.sum(np.square(vectors), axis=0)
    for start in range(0, n, int(residual_block_size)):
        stop = min(n, start + int(residual_block_size))
        block = vectors[:, start:stop]
        residual = operator.dot(block) - block * values[None, start:stop]
        absolute[start:stop] = np.linalg.norm(residual, axis=0)
        relative[start:stop] = absolute[start:stop] / np.maximum(1.0, np.abs(values[start:stop]))
    count = min(n, int(orthogonality_sample_size))
    sample = np.unique(np.rint(np.linspace(0, n - 1, count)).astype(int))
    gram_error = vectors[:, sample].T @ vectors[:, sample] - np.eye(len(sample))
    summary = {
        "maximum_absolute_residual": float(np.max(absolute)),
        "maximum_relative_residual": float(np.max(relative)),
        "median_relative_residual": float(np.median(relative)),
        "maximum_column_norm_error": float(np.max(np.abs(norms2 - 1.0))),
        "orthogonality_sample_size": int(len(sample)),
        "orthogonality_sample_indices": sample.tolist(),
        "orthogonality_sample_maximum_absolute_error": float(np.max(np.abs(gram_error))),
        "orthogonality_sample_frobenius_error": float(np.linalg.norm(gram_error)),
    }
    return summary, absolute, relative


def multiplicity_labels(values: np.ndarray, tolerance_factor: float) -> tuple[np.ndarray, np.ndarray, float]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    resolution = _multiplicity_scale(values, float(tolerance_factor))
    group = np.empty(n, dtype=int)
    multiplicity = np.empty(n, dtype=int)
    start = 0
    group_id = 0
    while start < n:
        stop = start + 1
        while stop < n and abs(float(values[stop] - values[stop - 1])) <= resolution:
            stop += 1
        group[start:stop] = group_id
        multiplicity[start:stop] = stop - start
        group_id += 1
        start = stop
    return group, multiplicity, resolution


def zero_subspace_evidence(
    values: np.ndarray,
    vectors: np.ndarray,
    operator: sparse.csr_matrix,
    *,
    enabled: bool,
    output_dir: Path,
) -> tuple[dict[str, object], np.ndarray]:
    zero = stable_zero_cluster(values, enabled=enabled)
    mask = np.asarray(zero.pop("mask"), dtype=bool)
    evidence: dict[str, object] = dict(zero)
    if bool(zero["active"]):
        basis = np.asarray(vectors[:, mask], dtype=float)
        projector_diagonal = np.sum(np.square(basis), axis=1)
        residual_square = 0.0
        block_size = 64
        zero_values = np.asarray(values[mask], dtype=float)
        for start in range(0, basis.shape[1], block_size):
            stop = min(basis.shape[1], start + block_size)
            residual = operator.dot(basis[:, start:stop]) - basis[:, start:stop] * zero_values[None, start:stop]
            residual_square += float(np.sum(np.square(residual)))
        sample_count = min(256, basis.shape[1])
        sample = np.unique(np.rint(np.linspace(0, basis.shape[1] - 1, sample_count)).astype(int))
        gram = basis[:, sample].T @ basis[:, sample]
        atomic_save_npy(Path(output_dir) / "arrays/zero_subspace_projector_diagonal.npy", projector_diagonal)
        evidence.update({
            "projector_diagonal_path": str(Path(output_dir) / "arrays/zero_subspace_projector_diagonal.npy"),
            "projector_trace_from_diagonal": float(np.sum(projector_diagonal)),
            "projector_diagonal_minimum": float(np.min(projector_diagonal)),
            "projector_diagonal_maximum": float(np.max(projector_diagonal)),
            "subspace_residual_frobenius": float(math.sqrt(residual_square)),
            "basis_sample_orthogonality_frobenius": float(np.linalg.norm(gram - np.eye(len(sample)))),
            "interpretation": "basis-invariant projector diagonal saved; individual solver vectors in this clean atom are unavailable for primary comparison",
        })
    return evidence, mask


def localization_rows(
    *,
    graph: nx.Graph,
    adjacency: sparse.csr_matrix,
    values: np.ndarray,
    vectors: np.ndarray,
    absolute_residuals: np.ndarray,
    relative_residuals: np.ndarray,
    potential: np.ndarray,
    atom_mask: np.ndarray,
    multiplicity_group: np.ndarray,
    multiplicity: np.ndarray,
    prefix: Mapping[str, object],
    support_fractions: Sequence[float],
    coordinate_counts: Sequence[int],
    block_size: int = 48,
) -> list[dict[str, object]]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    degrees = np.fromiter((graph.degree(vertex) for vertex in range(n)), dtype=int, count=n)
    rows: list[dict[str, object]] = []
    for start in range(0, n, int(block_size)):
        stop = min(n, start + int(block_size))
        block = np.asarray(vectors[:, start:stop], dtype=float)
        raw_mass = np.square(np.abs(block))
        norms2 = np.sum(raw_mass, axis=0)
        probabilities = raw_mass / norms2[None, :]
        ipr = np.sum(np.square(probabilities), axis=0)
        maximum_vertices = np.argmax(probabilities, axis=0)
        sorted_mass = np.sort(probabilities, axis=0)[::-1]
        cumulative = np.cumsum(sorted_mass, axis=0)
        for column, index in enumerate(range(start, stop)):
            basis_dependent = bool(atom_mask[index] or multiplicity[index] > 1)
            row: dict[str, object] = {
                **prefix,
                "ascending_index": int(index),
                "ascending_rank_1based": int(index + 1),
                "decreasing_algebraic_rank": int(n - index),
                "energy": float(values[index]),
                "leading_algebraic_state": bool(index == n - 1),
                "vector_norm": float(math.sqrt(norms2[column])),
                "absolute_eigenpair_residual": float(absolute_residuals[index]),
                "relative_eigenpair_residual": float(relative_residuals[index]),
                "multiplicity_group": int(multiplicity_group[index]),
                "numerical_multiplicity": int(multiplicity[index]),
                "clean_zero_atom_member": bool(atom_mask[index]),
                "basis_dependent_solver_vector": basis_dependent,
                "individual_state_primary_available": not basis_dependent,
                "ipr_solver_basis": float(ipr[column]),
                "n_times_ipr_solver_basis": float(n * ipr[column]),
                "dominant_vertex_solver_basis": int(maximum_vertices[column]),
                "dominant_vertex_degree_solver_basis": int(degrees[maximum_vertices[column]]),
                "maximum_coordinate_mass_solver_basis": float(sorted_mass[0, column]),
            }
            row["ipr"] = math.nan if basis_dependent else row["ipr_solver_basis"]
            row["n_times_ipr"] = math.nan if basis_dependent else row["n_times_ipr_solver_basis"]
            row["dominant_vertex"] = -1 if basis_dependent else row["dominant_vertex_solver_basis"]
            row["dominant_vertex_degree"] = -1 if basis_dependent else row["dominant_vertex_degree_solver_basis"]
            row["maximum_coordinate_mass"] = math.nan if basis_dependent else row["maximum_coordinate_mass_solver_basis"]
            for count in coordinate_counts:
                actual = min(int(count), n)
                value = float(cumulative[actual - 1, column]) if actual else 0.0
                row[f"top_{int(count)}_mass_solver_basis"] = value
                row[f"top_{int(count)}_mass"] = math.nan if basis_dependent else value
            for fraction in support_fractions:
                label = int(round(100.0 * float(fraction)))
                size = int(np.searchsorted(cumulative[:, column], float(fraction), side="left") + 1)
                size = min(size, n)
                row[f"support_size_{label}_solver_basis"] = size
                row[f"support_fraction_{label}_solver_basis"] = float(size / n)
                row[f"support_size_{label}"] = -1 if basis_dependent else size
                row[f"support_fraction_{label}"] = math.nan if basis_dependent else float(size / n)
            weights = probabilities[:, column]
            diagonal_energy = float(np.dot(potential, weights))
            row["diagonal_energy_contribution_solver_basis"] = diagonal_energy
            row["adjacency_energy_by_difference_solver_basis"] = float(values[index] - diagonal_energy)
            rows.append(row)
    return rows


def spacing_rows(
    values: np.ndarray,
    *,
    atom_mask_ascending: np.ndarray,
    resolution: float,
    prefix: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    descending = np.asarray(values[::-1], dtype=float)
    atom = np.asarray(atom_mask_ascending[::-1], dtype=bool)
    gaps = descending[:-1] - descending[1:]
    gap_rows: list[dict[str, object]] = []
    for position, gap in enumerate(gaps, start=1):
        gap_rows.append({
            **prefix,
            "pair_start_decreasing_rank": int(position),
            "left_rank": int(position),
            "right_rank": int(position + 1),
            "left_energy": float(descending[position - 1]),
            "right_energy": float(descending[position]),
            "raw_gap": float(gap),
            "numerically_resolved": bool(gap > resolution),
            "touches_clean_zero_atom": bool(atom[position - 1] or atom[position]),
            "multiplicity_resolution": float(resolution),
        })
    ratio_rows: list[dict[str, object]] = []
    for middle in range(2, len(descending)):
        left = float(gaps[middle - 2])
        right = float(gaps[middle - 1])
        denominator = max(left, right)
        raw = min(left, right) / denominator if denominator > 0.0 else math.nan
        unresolved = bool(left <= resolution or right <= resolution)
        atom_hit = bool(np.any(atom[middle - 2:middle + 1]))
        ratio_rows.append({
            **prefix,
            "middle_decreasing_rank": int(middle),
            "middle_energy": float(descending[middle - 1]),
            "left_gap": left,
            "right_gap": right,
            "raw_gap_ratio_all_levels": raw,
            "zero_over_zero_undefined": bool(denominator == 0.0),
            "multiplicity_or_resolution_excluded": unresolved,
            "clean_zero_atom_triplet_excluded": atom_hit,
            "filtered_gap_ratio": raw if math.isfinite(raw) and not unresolved and not atom_hit else math.nan,
            "triplet_first_rank": int(middle - 1),
            "triplet_last_rank": int(middle + 1),
            "triplet_is_consecutive_original_spectrum": True,
        })
    return gap_rows, ratio_rows


def _finite(rows: Iterable[object]) -> np.ndarray:
    values = np.asarray([float(value) for value in rows], dtype=float)
    return values[np.isfinite(values)]


def window_summary_rows(
    *,
    n: int,
    localization: Sequence[Mapping[str, object]],
    gaps: Sequence[Mapping[str, object]],
    ratios: Sequence[Mapping[str, object]],
    windows: Sequence[tuple[str, str, Mapping[str, object]]],
    prefix: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    descending = list(reversed(localization))
    summaries: list[dict[str, object]] = []
    observations: list[dict[str, object]] = []
    for family, perron_policy, window in windows:
        first = int(window["first_decreasing_rank"])
        last = int(window["last_decreasing_rank"])
        state_rows = descending[first - 1:last]
        strict_gaps = [row for row in gaps if first <= int(row["left_rank"]) and int(row["right_rank"]) <= last]
        strict_ratios = [row for row in ratios if first <= int(row["triplet_first_rank"]) and int(row["triplet_last_rank"]) <= last]
        guard_ratios = [row for row in ratios if first <= int(row["middle_decreasing_rank"]) <= last]
        for policy, ratio_selection in (("strict_internal", strict_ratios), ("center_with_external_guard", guard_ratios)):
            valid_ratios = _finite(row["filtered_gap_ratio"] for row in ratio_selection)
            all_ratios = _finite(row["raw_gap_ratio_all_levels"] for row in ratio_selection)
            usable = [row for row in state_rows if bool(row["individual_state_primary_available"])]
            nipr = _finite(row["n_times_ipr"] for row in usable)
            support80 = _finite(row["support_size_80"] for row in usable)
            summary = {
                **prefix,
                "window_family": family,
                "window_id": window["window_id"],
                "perron_policy": perron_policy,
                "membership_policy": policy,
                **dict(window),
                "actual_state_count": int(len(state_rows)),
                "atom_state_count": int(sum(bool(row["clean_zero_atom_member"]) for row in state_rows)),
                "basis_dependent_state_count": int(sum(bool(row["basis_dependent_solver_vector"]) for row in state_rows)),
                "primary_usable_state_count": int(len(usable)),
                "first_energy": float(state_rows[0]["energy"]) if state_rows else math.nan,
                "last_energy": float(state_rows[-1]["energy"]) if state_rows else math.nan,
                "strict_internal_gap_count": int(len(strict_gaps)),
                "candidate_ratio_count": int(len(ratio_selection)),
                "mathematically_defined_ratio_count": int(len(all_ratios)),
                "valid_filtered_ratio_count": int(len(valid_ratios)),
                "invalid_or_excluded_ratio_count": int(len(ratio_selection) - len(valid_ratios)),
                "mean_gap_ratio_all_levels": float(np.mean(all_ratios)) if len(all_ratios) else math.nan,
                "median_gap_ratio_all_levels": float(np.median(all_ratios)) if len(all_ratios) else math.nan,
                "mean_gap_ratio_filtered": float(np.mean(valid_ratios)) if len(valid_ratios) else math.nan,
                "median_gap_ratio_filtered": float(np.median(valid_ratios)) if len(valid_ratios) else math.nan,
                "mean_n_times_ipr": float(np.mean(nipr)) if len(nipr) else math.nan,
                "median_n_times_ipr": float(np.median(nipr)) if len(nipr) else math.nan,
                "mean_support_size_80": float(np.mean(support80)) if len(support80) else math.nan,
                "median_support_size_80": float(np.median(support80)) if len(support80) else math.nan,
                "histogram_status": "sufficient" if len(strict_gaps) >= 10 else "insufficient_observations",
            }
            summaries.append(summary)
        resolved = [float(row["raw_gap"]) for row in strict_gaps if bool(row["numerically_resolved"]) and not bool(row["touches_clean_zero_atom"])]
        mean_gap = float(np.mean(resolved)) if resolved else math.nan
        for row in strict_gaps:
            observations.append({
                **prefix,
                "window_family": family,
                "window_id": window["window_id"],
                "perron_policy": perron_policy,
                "membership_policy": "strict_internal",
                **{key: value for key, value in row.items() if key not in prefix},
                "window_mean_resolved_raw_gap": mean_gap,
                "window_mean_rescaled_gap": float(row["raw_gap"]) / mean_gap if mean_gap > 0.0 else math.nan,
                "rescaling_label": "division by this one-window mean; not ensemble unfolding",
            })
    return summaries, observations


def first_edge_rows(values: np.ndarray, gaps: Sequence[Mapping[str, object]], prefix: Mapping[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    descending = np.asarray(values[::-1], dtype=float)
    eigen_rows: list[dict[str, object]] = []
    gap_export: list[dict[str, object]] = []
    for convention, first_rank in (("literal_including_rank_one", 1), ("rank_one_excluded", 2)):
        for rank in range(first_rank, min(len(values) + 1, first_rank + 10)):
            eigen_rows.append({
                **prefix,
                "rank_convention": convention,
                "position_within_convention": rank - first_rank + 1,
                "decreasing_algebraic_rank": rank,
                "ascending_index": len(values) - rank,
                "energy": float(descending[rank - 1]),
            })
        for rank in range(first_rank, min(len(values), first_rank + 10)):
            source = gaps[rank - 1]
            gap_export.append({
                **prefix,
                "rank_convention": convention,
                "gap_position_within_convention": rank - first_rank + 1,
                **{key: value for key, value in source.items() if key not in prefix},
            })
    return eigen_rows, gap_export


def enrich_geometry(
    *,
    graph: nx.Graph,
    adjacency: sparse.csr_matrix,
    values: np.ndarray,
    vectors: np.ndarray,
    potential: np.ndarray,
    ranks: Sequence[int],
    coordinate_counts: Sequence[int],
    prefix: Mapping[str, object],
) -> tuple[dict[str, list[dict[str, object]]], list[dict[str, object]]]:
    tables = _coordinate_and_geometry_tables(graph, values, vectors, ranks, coordinate_counts, prefix)
    for row in tables["coordinate_rows"]:
        row["diagonal_potential"] = float(potential[int(row["vertex_id"])])
    for row in tables["induced_vertex_rows"]:
        row["diagonal_potential"] = float(potential[int(row["vertex_id"])])
    selected_ranks = set(int(value) for value in ranks)
    for row in tables["cross_summary_rows"]:
        row["spectrally_adjacent"] = bool(abs(int(row["left_decreasing_rank"]) - int(row["right_decreasing_rank"])) == 1)
        row["whole_vector_mass_overlap_definition"] = "sum_v min(|psi_i(v)|^2,|psi_j(v)|^2)"
        row["top10_overlap_definition"] = "cardinality of shared vertex IDs; shared vertices have cross-distance zero"
    decompositions: list[dict[str, object]] = []
    n = len(values)
    for rank in sorted(selected_ranks):
        if rank < 1 or rank > n:
            continue
        index = n - rank
        vector = np.asarray(vectors[:, index], dtype=float)
        mass = np.square(np.abs(vector))
        adjacency_energy = float(np.real(np.vdot(vector, adjacency.dot(vector))))
        diagonal_energy = float(np.dot(potential, mass))
        decompositions.append({
            **prefix,
            "decreasing_algebraic_rank": rank,
            "ascending_index": index,
            "energy": float(values[index]),
            "adjacency_energy": adjacency_energy,
            "diagonal_energy": diagonal_energy,
            "reconstructed_energy": adjacency_energy + diagonal_energy,
            "absolute_reconstruction_error": abs(float(values[index]) - adjacency_energy - diagonal_energy),
        })
    return tables, decompositions


def _write_operator_tables(output: Path, tables: Mapping[str, Sequence[Mapping[str, object]]]) -> None:
    filenames = {
        "localization": "full_spectrum_localization.csv.gz",
        "gaps": "full_spectrum_raw_gaps.csv.gz",
        "ratios": "full_spectrum_gap_ratios.csv.gz",
        "window_summaries": "energy_and_rank_window_summary.csv.gz",
        "window_gap_observations": "window_gap_observations.csv.gz",
        "first_eigenvalues": "first_ten_eigenvalues.csv",
        "first_gaps": "first_ten_consecutive_edge_gaps.csv",
        "energy_decomposition": "detailed_state_energy_decomposition.csv",
    }
    filenames.update({
        "coordinate_rows": "coordinate_rows.csv.gz",
        "cumulative_rows": "cumulative_rows.csv.gz",
        "induced_vertex_rows": "induced_vertex_rows.csv.gz",
        "induced_edge_rows": "induced_edge_rows.csv.gz",
        "induced_pair_distance_rows": "induced_pair_distance_rows.csv.gz",
        "induced_pair_distance_summary_rows": "induced_pair_distance_summary_rows.csv.gz",
        "cross_summary_rows": "cross_summary_rows.csv.gz",
        "cross_pair_rows": "cross_pair_rows.csv.gz",
    })
    for key, filename in filenames.items():
        rows = tables.get(key, [])
        if rows:
            atomic_write_csv(output / "tables" / filename, rows)


def analyze_operator(
    *,
    graph: nx.Graph,
    adjacency: sparse.csr_matrix,
    values: np.ndarray,
    vectors: np.ndarray,
    absolute_residuals: np.ndarray,
    relative_residuals: np.ndarray,
    xi: np.ndarray,
    amplitude: float,
    config: Mapping[str, object],
    output: Path,
    prefix: Mapping[str, object],
) -> dict[str, object]:
    analysis = config["analysis"]
    solver = config["solver"]
    n = len(values)
    potential = float(amplitude) * np.asarray(xi, dtype=float)
    operator = adjacency + sparse.diags(potential, format="csr")
    groups, multiplicity, resolution = multiplicity_labels(values, float(solver["multiplicity_tolerance_factor"]))
    zero, atom_mask = zero_subspace_evidence(values, vectors, operator, enabled=bool(float(amplitude) == 0.0), output_dir=output)
    prefix = {**prefix, "multiplicity_resolution": resolution}
    localization = localization_rows(
        graph=graph,
        adjacency=adjacency,
        values=values,
        vectors=vectors,
        absolute_residuals=absolute_residuals,
        relative_residuals=relative_residuals,
        potential=potential,
        atom_mask=atom_mask,
        multiplicity_group=groups,
        multiplicity=multiplicity,
        prefix=prefix,
        support_fractions=analysis["support_mass_fractions"],
        coordinate_counts=analysis["coordinate_counts"],
    )
    gaps, ratios = spacing_rows(values, atom_mask_ascending=atom_mask, resolution=resolution, prefix=prefix)
    windows: list[tuple[str, str, Mapping[str, object]]] = []
    for window in quantile_rank_windows(n, analysis["quantile_edges"]):
        windows.append(("global_quantile_rank_band", "literal decreasing algebraic rank", window))
    windows.extend(_cumulative_and_disjoint_windows(n, analysis["cumulative_upper_counts"][str(n)]))
    summaries, observations = window_summary_rows(
        n=n,
        localization=localization,
        gaps=gaps,
        ratios=ratios,
        windows=windows,
        prefix=prefix,
    )
    first_eigenvalues, first_gaps = first_edge_rows(values, gaps, prefix)
    geometry, decomposition = enrich_geometry(
        graph=graph,
        adjacency=adjacency,
        values=values,
        vectors=vectors,
        potential=potential,
        ranks=analysis["detailed_decreasing_algebraic_ranks"],
        coordinate_counts=analysis["coordinate_counts"],
        prefix=prefix,
    )
    tables: dict[str, Sequence[Mapping[str, object]]] = {
        "localization": localization,
        "gaps": gaps,
        "ratios": ratios,
        "window_summaries": summaries,
        "window_gap_observations": observations,
        "first_eigenvalues": first_eigenvalues,
        "first_gaps": first_gaps,
        "energy_decomposition": decomposition,
        **geometry,
    }
    _write_operator_tables(output, tables)
    top_mass_violations = 0
    support_violations = 0
    ipr_violations = 0
    for row in localization:
        top = [float(row[f"top_{count}_mass_solver_basis"]) for count in analysis["coordinate_counts"]]
        top_mass_violations += int(any(value < -1e-12 or value > 1.0 + 1e-10 for value in top) or any(right + 1e-12 < left for left, right in zip(top, top[1:])))
        supports = [int(row[f"support_size_{int(round(100*q))}_solver_basis"]) for q in analysis["support_mass_fractions"]]
        support_violations += int(any(value < 1 or value > n for value in supports) or any(right < left for left, right in zip(supports, supports[1:])))
        ipr = float(row["ipr_solver_basis"])
        ipr_violations += int(ipr < 1.0 / n - 1e-10 or ipr > 1.0 + 1e-10)
    if top_mass_violations or support_violations or ipr_violations:
        raise AssertionError("localization bound validation failed")
    return {
        "zero_subspace": zero,
        "multiplicity_resolution": resolution,
        "numerically_degenerate_state_count": int(np.count_nonzero(multiplicity > 1)),
        "numerical_multiplicity_maximum": int(np.max(multiplicity)),
        "localization_row_count": len(localization),
        "gap_row_count": len(gaps),
        "ratio_row_count": len(ratios),
        "window_summary_row_count": len(summaries),
        "window_gap_observation_count": len(observations),
        "detailed_rank_count": len(decomposition),
        "top_mass_bound_violation_count": top_mass_violations,
        "support_bound_violation_count": support_violations,
        "ipr_bound_violation_count": ipr_violations,
    }


def run_operator(
    *,
    config: Mapping[str, object],
    run_root: Path,
    n: int,
    alpha: float,
    amplitude: float,
    source_identity: str,
) -> Path:
    case_manifest, graph, adjacency, xi = load_prepared_case(run_root, n, alpha)
    if float(amplitude) not in [float(value) for value in config["amplitudes"]]:
        raise ValueError("amplitude is outside the configured raw-amplitude grid")
    output = operator_directory(run_root, n, alpha, amplitude)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    identity = stable_json_hash({
        "version": PA_UNIFORM_NOISE_VERSION,
        "source_identity": source_identity,
        "case_identity": case_manifest["case_identity"],
        "amplitude_raw": float(amplitude),
        "solver": config["solver"],
        "analysis": config["analysis"],
    })
    if manifest_path.exists():
        old = read_json(manifest_path)
        if old.get("status") == "COMPLETE" and old.get("operator_identity") == identity:
            return output
        if old.get("operator_identity") not in (None, identity):
            raise ValueError("existing operator output has an incompatible identity")
    started = time.time()
    manifest: dict[str, object] = {
        "status": "RUNNING",
        "stage": "raw_uniform_diagonal_disorder_full_eigensystem_and_diagnostics",
        "operator_identity": identity,
        "source_identity": source_identity,
        "case_identity": case_manifest["case_identity"],
        "n": int(n),
        "degree_ccdf_alpha": float(alpha),
        "raw_amplitude": float(amplitude),
        "operator_definition": "H(a)=A+a*diag(xi); A unscaled; xi iid Uniform[-1,1); no centering, standardization, clipping, or cumulative perturbation",
        "graph_evidence": case_manifest["graph_evidence"],
        "potential_seed": case_manifest["potential_seed"],
        "xi_sha256_raw_float64": case_manifest["xi_sha256_raw_float64"],
        "started_at_utc": utc_now(),
    }
    atomic_write_json(manifest_path, manifest)
    try:
        potential = float(amplitude) * xi
        prefix = {
            "experiment_id": config["experiment_id"],
            "n": int(n),
            "degree_ccdf_alpha": float(alpha),
            "alpha_convention": "degree CCDF tail parameter",
            "m": int(config["m"]),
            "raw_amplitude": float(amplitude),
            "graph_case_id": case_manifest["graph_evidence"]["clean_case_id"],
            "graph_sha256_edges": case_manifest["graph_evidence"]["canonical_edge_hash"],
            "potential_seed": int(case_manifest["potential_seed"]),
            "xi_sha256_raw_float64": case_manifest["xi_sha256_raw_float64"],
        }
        spectral_manifest_path = output / "spectral_cache_manifest.json"
        values_path = output / "arrays/full_eigenvalues_ascending.npy"
        vectors_path = output / "arrays/full_eigenvectors_ascending.npy"
        absolute_path = output / "arrays/full_absolute_residuals.npy"
        relative_path = output / "arrays/full_relative_residuals.npy"
        if float(amplitude) == 0.0:
            paths = case_manifest["graph_evidence"]["paths"]
            values = np.load(paths["eigenvalues"], mmap_mode="r")
            vectors = np.load(paths["eigenvectors"], mmap_mode="r")
            absolute = np.load(paths["absolute_residuals"], mmap_mode="r")
            relative = np.load(paths["relative_residuals"], mmap_mode="r")
            spectral_summary = {
                "status": "COMPLETE",
                "mode": "reused_validated_clean_eigensystem_without_duplicate_storage",
                "source_spectral_manifest": paths["spectral_manifest"],
                "source_spectral_manifest_sha256": case_manifest["graph_evidence"]["spectral_manifest_sha256"],
                "values_path": paths["eigenvalues"],
                "vectors_path": paths["eigenvectors"],
                "absolute_residuals_path": paths["absolute_residuals"],
                "relative_residuals_path": paths["relative_residuals"],
                "H0_equals_A": True,
            }
            dense, construction = construct_operator(adjacency, xi, 0.0)
            del dense
        else:
            spectral_identity = stable_json_hash({"operator_identity": identity, "stage": "eigensystem"})
            if all(path.is_file() for path in (values_path, vectors_path, absolute_path, relative_path, spectral_manifest_path)):
                spectral_summary = read_json(spectral_manifest_path)
                if spectral_summary.get("status") != "COMPLETE" or spectral_summary.get("spectral_identity") != spectral_identity:
                    raise ValueError("incompatible partial spectral cache")
                values = np.load(values_path, mmap_mode="r")
                vectors = np.load(vectors_path, mmap_mode="r")
                absolute = np.load(absolute_path, mmap_mode="r")
                relative = np.load(relative_path, mmap_mode="r")
                construction = spectral_summary["operator_construction_validation"]
            else:
                dense, construction = construct_operator(adjacency, xi, float(amplitude))
                eig_started = time.time()
                values_array, vectors_array = linalg.eigh(
                    dense,
                    overwrite_a=True,
                    check_finite=False,
                    driver=str(config["solver"]["dense_driver"]),
                )
                eig_runtime = time.time() - eig_started
                del dense
                operator_sparse = adjacency + sparse.diags(potential, format="csr")
                verification, absolute_array, relative_array = verify_eigensystem(
                    operator_sparse,
                    values_array,
                    vectors_array,
                    residual_block_size=int(config["solver"]["residual_block_size"]),
                    orthogonality_sample_size=int(config["solver"]["orthogonality_sample_size"]),
                )
                if float(verification["maximum_relative_residual"]) > float(config["solver"]["relative_eigenpair_residual_tolerance"]):
                    raise RuntimeError("eigensystem failed configured relative residual tolerance")
                trace_expected = float(np.sum(potential))
                trace_observed = float(np.sum(values_array))
                square_expected = float(2 * graph.number_of_edges() + np.sum(np.square(potential)))
                square_observed = float(np.sum(np.square(values_array)))
                scale_trace = max(1.0, abs(trace_expected), float(np.sum(np.abs(values_array))))
                scale_square = max(1.0, square_expected)
                trace_error = abs(trace_observed - trace_expected)
                square_error = abs(square_observed - square_expected)
                tolerance = float(config["solver"]["identity_relative_tolerance"])
                identities = {
                    "trace_expected_sum_V": trace_expected,
                    "trace_observed_sum_eigenvalues": trace_observed,
                    "trace_absolute_error": trace_error,
                    "trace_relative_scaled_error": trace_error / scale_trace,
                    "squared_trace_expected_2E_plus_sum_V2": square_expected,
                    "squared_trace_observed_sum_lambda2": square_observed,
                    "squared_trace_absolute_error": square_error,
                    "squared_trace_relative_error": square_error / scale_square,
                    "identities_pass": bool(trace_error / scale_trace <= tolerance and square_error / scale_square <= tolerance),
                }
                if not identities["identities_pass"]:
                    raise RuntimeError("spectral trace identity validation failed")
                atomic_save_npy(values_path, np.asarray(values_array, dtype=np.float64))
                atomic_save_npy(vectors_path, np.asarray(vectors_array, dtype=np.float64))
                atomic_save_npy(absolute_path, absolute_array)
                atomic_save_npy(relative_path, relative_array)
                spectral_summary = {
                    "status": "COMPLETE",
                    "spectral_identity": spectral_identity,
                    "eigenvalue_order": "ascending",
                    "eigenvector_columns": "column j matches ascending eigenvalue j; rows are vertices 0..N-1",
                    "storage_dtype": "float64",
                    "values_path": str(values_path),
                    "vectors_path": str(vectors_path),
                    "absolute_residuals_path": str(absolute_path),
                    "relative_residuals_path": str(relative_path),
                    "values_sha256": sha256_file(values_path),
                    "vectors_sha256": sha256_file(vectors_path),
                    "eigendecomposition_runtime_seconds": eig_runtime,
                    "operator_construction_validation": construction,
                    "spectral_trace_identities": identities,
                    **verification,
                    "finished_at_utc": utc_now(),
                }
                atomic_write_json(spectral_manifest_path, spectral_summary)
                values, vectors, absolute, relative = values_array, vectors_array, absolute_array, relative_array
        trace_expected = float(np.sum(potential))
        square_expected = float(2 * graph.number_of_edges() + np.sum(np.square(potential)))
        trace_observed = float(np.sum(values))
        square_observed = float(np.sum(np.square(values)))
        trace_error = abs(trace_observed - trace_expected)
        square_error = abs(square_observed - square_expected)
        identity_tolerance = float(config["solver"]["identity_relative_tolerance"])
        identities = {
            "trace_expected_sum_V": trace_expected,
            "trace_observed_sum_eigenvalues": trace_observed,
            "trace_absolute_error": trace_error,
            "squared_trace_expected_2E_plus_sum_V2": square_expected,
            "squared_trace_observed_sum_lambda2": square_observed,
            "squared_trace_absolute_error": square_error,
            "identities_pass": bool(
                trace_error / max(1.0, abs(trace_expected), float(np.sum(np.abs(values)))) <= identity_tolerance
                and square_error / max(1.0, square_expected) <= identity_tolerance
            ),
        }
        if not identities["identities_pass"]:
            raise RuntimeError("operator spectral identities failed before diagnostics")
        analysis_summary = analyze_operator(
            graph=graph,
            adjacency=adjacency,
            values=np.asarray(values),
            vectors=vectors,
            absolute_residuals=np.asarray(absolute),
            relative_residuals=np.asarray(relative),
            xi=xi,
            amplitude=float(amplitude),
            config=config,
            output=output,
            prefix=prefix,
        )
        required_tables = [
            "full_spectrum_localization.csv.gz",
            "full_spectrum_raw_gaps.csv.gz",
            "full_spectrum_gap_ratios.csv.gz",
            "energy_and_rank_window_summary.csv.gz",
            "window_gap_observations.csv.gz",
            "first_ten_eigenvalues.csv",
            "first_ten_consecutive_edge_gaps.csv",
            "coordinate_rows.csv.gz",
            "cumulative_rows.csv.gz",
            "induced_vertex_rows.csv.gz",
            "induced_pair_distance_rows.csv.gz",
            "cross_summary_rows.csv.gz",
            "cross_pair_rows.csv.gz",
        ]
        missing = [name for name in required_tables if not (output / "tables" / name).is_file()]
        if missing:
            raise RuntimeError(f"required output tables missing: {missing}")
        manifest.update({
            "status": "COMPLETE",
            "finished_at_utc": utc_now(),
            "runtime_seconds": time.time() - started,
            "spectral_cache": spectral_summary,
            "operator_construction_validation": construction,
            "spectral_trace_identities": identities,
            "analysis_summary": analysis_summary,
            "required_tables": required_tables,
            "scientific_scope": "one graph and one paired potential stream; descriptive comparisons only; ranks are not mode tracking",
        })
        atomic_write_json(manifest_path, manifest)
        return output
    except Exception as error:
        manifest.update({
            "status": "FAILED",
            "finished_at_utc": utc_now(),
            "runtime_seconds": time.time() - started,
            "failure_type": type(error).__name__,
            "failure_message": str(error),
        })
        atomic_write_json(manifest_path, manifest)
        raise


def run_controlled_pilot(*, config: Mapping[str, object], output: Path) -> Path:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    n, m, alpha, amplitude, graph_seed = 80, int(config["m"]), 2.0, 2.0, 8020260924
    graph, diagnostics = generate_preferential_attachment_graph(
        n=n,
        m=m,
        seed=graph_seed,
        model=MODEL_AFFINE_PA,
        degree_ccdf_alpha=alpha,
    )
    adjacency = graph_to_csr(graph, nodelist=range(n))
    _, xi = draw_potential({**dict(config), "degree_ccdf_alpha_grid": [alpha]}, n, alpha)
    dense, construction = construct_operator(adjacency, xi, amplitude)
    values, vectors = linalg.eigh(dense, overwrite_a=True, check_finite=False, driver="evd")
    operator = adjacency + sparse.diags(amplitude * xi, format="csr")
    verification, absolute, relative = verify_eigensystem(
        operator,
        values,
        vectors,
        residual_block_size=32,
        orthogonality_sample_size=n,
    )
    atomic_save_npy(output / "arrays/full_eigenvalues_ascending.npy", values)
    atomic_save_npy(output / "arrays/full_eigenvectors_ascending.npy", vectors)
    atomic_save_npy(output / "arrays/xi.npy", xi)
    pilot_config = json.loads(json.dumps(config))
    pilot_config["analysis"]["cumulative_upper_counts"] = {"80": [10, 25, 50, 80]}
    summary = analyze_operator(
        graph=graph,
        adjacency=adjacency,
        values=values,
        vectors=vectors,
        absolute_residuals=absolute,
        relative_residuals=relative,
        xi=xi,
        amplitude=amplitude,
        config=pilot_config,
        output=output,
        prefix={
            "experiment_id": "controlled_pilot",
            "n": n,
            "degree_ccdf_alpha": alpha,
            "alpha_convention": "degree CCDF tail parameter",
            "m": m,
            "raw_amplitude": amplitude,
            "graph_case_id": "controlled_pilot_generated_graph",
            "graph_sha256_edges": _edge_hash(graph),
            "potential_seed": potential_seed({**dict(config), "degree_ccdf_alpha_grid": [alpha]}, n, alpha),
            "xi_sha256_raw_float64": array_hash(xi),
        },
    )
    manifest = {
        "status": "COMPLETE",
        "stage": "small_controlled_end_to_end_pilot_not_production",
        "n": n,
        "m": m,
        "alpha": alpha,
        "amplitude": amplitude,
        "graph_seed": graph_seed,
        "generator_diagnostics": diagnostics.to_dict(),
        "operator_construction_validation": construction,
        "eigensystem_validation": verification,
        "analysis_summary": summary,
        "finished_at_utc": utc_now(),
    }
    atomic_write_json(output / "run_manifest.json", manifest)
    return output
