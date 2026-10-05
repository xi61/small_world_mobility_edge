"""Subspace-based tunneling validation and independent-center applications."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import networkx as nx
import numpy as np
from scipy import linalg, sparse
from scipy.optimize import linear_sum_assignment

from .applications import SpectralBundle, load_er_bundle, load_pa_bundle
from .common import (
    BFSDistanceCache,
    analyzed_to_original_ids,
    graph_from_edge_cache,
    graph_to_csr,
    read_json,
    runtime_metadata,
    sha256_file,
    source_configuration_identity,
    stable_json_hash,
    utc_now,
    write_csv,
    write_json,
)
from .tunneling import (
    approximate_state_pair_rows,
    local_ball_reference_state,
    paper_center_sets,
    paper_global_approximate_states,
    schur_effective_operator,
    symmetric_orthogonalization,
    two_hub_graph,
)


TUNNELING_CORRECTION_VERSION = "subspace_tunneling_correction_v3"


@dataclass
class SubspaceResult:
    valid: bool
    summary: dict[str, object]
    state_rows: list[dict[str, object]]
    orthonormal_states: np.ndarray | None
    effective_matrix: np.ndarray | None
    residual_matrix: np.ndarray | None


def subspace_diagnostics(operator: sparse.spmatrix, states: np.ndarray) -> SubspaceResult:
    """Compute K=U*HU and R=HU-UK after symmetric orthogonalization."""
    operator = sparse.csr_matrix(operator)
    raw_states = np.asarray(states, dtype=float)
    if raw_states.ndim != 2 or raw_states.shape[0] != operator.shape[0]:
        raise ValueError("state matrix has incompatible shape")
    orthogonalized = symmetric_orthogonalization(raw_states)
    base = {
        "raw_state_count": raw_states.shape[1],
        "raw_gram_condition_number": float(orthogonalized["gram_condition_number"]),
        "orthogonalization_valid": bool(orthogonalized["valid"]),
        "orthogonalization_reason": str(orthogonalized["reason"]),
    }
    if not bool(orthogonalized["valid"]):
        return SubspaceResult(False, base, [], None, None, None)
    u = np.asarray(orthogonalized["states"], dtype=float)
    hu = np.asarray(operator.dot(u), dtype=float)
    k_matrix = u.T @ hu
    k_matrix = 0.5 * (k_matrix + k_matrix.T)
    residual = hu - u @ k_matrix
    orthogonality_error = float(np.linalg.norm(u.T @ u - np.eye(u.shape[1])))
    projected_residual_error = float(np.linalg.norm(u.T @ residual))
    rows: list[dict[str, object]] = []
    maximum_decomposition_error = 0.0
    for column in range(u.shape[1]):
        diagonal_energy = float(k_matrix[column, column])
        individual = hu[:, column] - diagonal_energy * u[:, column]
        internal_coordinates = k_matrix[:, column].copy()
        internal_coordinates[column] -= diagonal_energy
        internal_norm = float(np.linalg.norm(internal_coordinates))
        leakage_norm = float(np.linalg.norm(residual[:, column]))
        individual_norm = float(np.linalg.norm(individual))
        decomposition_error = abs(individual_norm**2 - internal_norm**2 - leakage_norm**2)
        maximum_decomposition_error = max(maximum_decomposition_error, decomposition_error)
        rows.append(
            {
                "state_index": column,
                "diagonal_rayleigh_energy": diagonal_energy,
                "individual_approximate_eigenvector_residual_norm": individual_norm,
                "internal_coupling_norm": internal_norm,
                "external_leakage_norm": leakage_norm,
                "residual_squared_decomposition_error": decomposition_error,
                "interpretation": "individual residual combines internal retained-subspace coupling and external leakage",
            }
        )
    ritz_values = np.linalg.eigvalsh(k_matrix)
    summary = {
        **base,
        "orthonormality_error": orthogonality_error,
        "projected_residual_orthogonality_error": projected_residual_error,
        "effective_matrix_frobenius_norm": float(np.linalg.norm(k_matrix)),
        "external_leakage_frobenius_norm": float(np.linalg.norm(residual)),
        "maximum_state_residual_decomposition_error": maximum_decomposition_error,
        "ritz_values_ascending": ritz_values.tolist(),
        "K_definition": "U.T @ H @ U for the actual orthonormal U returned by symmetric Lowdin orthogonalization",
        "R_definition": "H @ U - U @ K; only R measures leakage outside the retained subspace",
    }
    return SubspaceResult(True, summary, rows, u, k_matrix, residual)


def _reference_comparison(
    subspace: SubspaceResult,
    exact_values: np.ndarray,
    exact_vectors: np.ndarray,
    full_spectrum_known: bool,
    isolation_values: np.ndarray | None = None,
) -> dict[str, object]:
    if not subspace.valid:
        return {"comparison_valid": False, "reason": "invalid_reference_subspace"}
    u = np.asarray(subspace.orthonormal_states)
    k_matrix = np.asarray(subspace.effective_matrix)
    ritz_values, ritz_vectors = linalg.eigh(k_matrix, check_finite=False)
    exact_values = np.asarray(exact_values, dtype=float)
    exact_vectors = np.asarray(exact_vectors, dtype=float)
    if exact_vectors.shape[0] != u.shape[0] or exact_vectors.shape[1] != len(exact_values):
        raise ValueError("exact reference eigensystem has incompatible dimensions")
    cost = np.abs(ritz_values[:, None] - exact_values[None, :])
    ritz_positions, exact_positions = linear_sum_assignment(cost)
    order = np.argsort(ritz_positions)
    exact_positions = exact_positions[order]
    matched_values = exact_values[exact_positions]
    predicted_vectors = u @ ritz_vectors
    exact_subspace = exact_vectors[:, exact_positions]
    predicted_projector_diagonal = np.sum(np.square(predicted_vectors), axis=1)
    exact_projector_diagonal = np.sum(np.square(exact_subspace), axis=1)
    subspace_overlap_squared = float(np.sum(np.square(u.T @ exact_subspace)))
    projector_frobenius_error = math.sqrt(
        max(0.0, 2.0 * u.shape[1] - 2.0 * subspace_overlap_squared)
    )
    energy_errors = np.abs(ritz_values - matched_values)
    isolation_spectrum = (
        np.asarray(isolation_values, dtype=float)
        if isolation_values is not None
        else exact_values
    )
    omitted = np.ones(len(isolation_spectrum), dtype=bool)
    available = np.arange(len(isolation_spectrum), dtype=int)
    matched_isolation_positions: list[int] = []
    for value in matched_values:
        candidates = available[omitted]
        position = int(candidates[np.argmin(np.abs(isolation_spectrum[candidates] - value))])
        omitted[position] = False
        matched_isolation_positions.append(position)
    if np.any(omitted):
        nearest_omitted = float(
            np.min(np.abs(ritz_values[:, None] - isolation_spectrum[omitted][None, :]))
        )
    else:
        nearest_omitted = math.inf
    leakage = float(subspace.summary["external_leakage_frobenius_norm"])
    return {
        "comparison_valid": True,
        "ritz_values_ascending": ritz_values.tolist(),
        "matched_exact_values_ascending": matched_values.tolist(),
        "matched_exact_reference_positions": exact_positions.tolist(),
        "matched_isolation_spectrum_positions": matched_isolation_positions,
        "isolation_spectrum_size": len(isolation_spectrum),
        "maximum_ritz_energy_error": float(np.max(energy_errors)),
        "mean_ritz_energy_error": float(np.mean(energy_errors)),
        "projector_frobenius_error": projector_frobenius_error,
        "projector_diagonal_l1_error": float(
            np.sum(np.abs(predicted_projector_diagonal - exact_projector_diagonal))
        ),
        "projector_diagonal_maximum_error": float(
            np.max(np.abs(predicted_projector_diagonal - exact_projector_diagonal))
        ),
        "nearest_known_omitted_spectral_separation": nearest_omitted,
        "leakage_over_known_omitted_separation": (
            leakage / nearest_omitted if nearest_omitted > 0.0 and math.isfinite(nearest_omitted) else math.nan
        ),
        "full_spectrum_known": full_spectrum_known,
        "isolation_interpretation": (
            "all omitted exact modes included in separation audit"
            if full_spectrum_known
            else "separation only against retained known reference modes; omitted lower modes remain unverified"
        ),
    }


def _structural_centers(
    graph: nx.Graph, maximum: int = 6, minimum_separation: int = 3
) -> list[int]:
    degrees = np.asarray([graph.degree(vertex) for vertex in range(graph.number_of_nodes())], dtype=int)
    order = np.lexsort((np.arange(len(degrees)), -degrees))
    cache = BFSDistanceCache(graph)
    selected: list[int] = []
    for value in order:
        vertex = int(value)
        if all(cache.distance(vertex, other) >= minimum_separation for other in selected):
            selected.append(vertex)
            if len(selected) >= maximum:
                break
    return selected


def _local_state_matrix(
    graph: nx.Graph, operator: sparse.spmatrix, centers: Sequence[int], radius: int
) -> tuple[np.ndarray, list[dict[str, object]]]:
    states = []
    rows = []
    for center in centers:
        state = local_ball_reference_state(
            graph,
            operator,
            int(center),
            radius=radius,
            mode_selection="largest_algebraic",
            target_energy=None,
        )
        states.append(np.asarray(state["state"], dtype=float))
        rows.append({key: value for key, value in state.items() if key != "state"})
    return np.column_stack(states), rows


def _fixture_schur_rows(
    fixture: str,
    operator: sparse.spmatrix,
    subspace: SubspaceResult,
) -> list[dict[str, object]]:
    if not subspace.valid:
        return []
    u = np.asarray(subspace.orthonormal_states)
    k_matrix = np.asarray(subspace.effective_matrix)
    ritz = np.linalg.eigvalsh(k_matrix)
    energies = sorted(set([float(ritz[0]), float(np.mean(ritz)), float(ritz[-1])]))
    dense = sparse.csr_matrix(operator).toarray()
    complement = linalg.null_space(u.T)
    complement_orthogonality = float(np.linalg.norm(complement.T @ complement - np.eye(complement.shape[1])))
    complement_coupling = float(np.linalg.norm(u.T @ dense @ complement)) if complement.shape[1] else 0.0
    complement_values = (
        np.linalg.eigvalsh(complement.T @ dense @ complement)
        if complement.shape[1]
        else np.asarray([], dtype=float)
    )
    rows: list[dict[str, object]] = []
    for energy in energies:
        nearest = (
            float(np.min(np.abs(complement_values - energy)))
            if len(complement_values)
            else math.inf
        )
        for eta in (0.1, 0.01, 0.001):
            result = schur_effective_operator(operator, u, energy, eta)
            rows.append(
                {
                    "fixture": fixture,
                    "orthonormal_basis_error_passed_to_schur": float(np.linalg.norm(u.T @ u - np.eye(u.shape[1]))),
                    "complement_basis_orthogonality_error": complement_orthogonality,
                    "H_PQ_frobenius_norm": complement_coupling,
                    "nearest_complement_resonance_distance": nearest,
                    "energy_role": "evaluation point only; eigenvalues of H_eff(z) are not asserted to equal full eigenvalues",
                    **{key: value for key, value in result.items() if key != "effective_matrix"},
                }
            )
    return rows


def run_corrected_tunneling_validation(output_dir: Path, project_root: Path) -> Path:
    started = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_configuration = source_configuration_identity(project_root)
    cache_id = stable_json_hash(
        {
            "version": TUNNELING_CORRECTION_VERSION,
            "source_configuration_id": source_configuration["sha256"],
            "fixtures": "exact2x2+weak+strong+mediated+intentional_failure",
        }
    )
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if existing.get("status") == "COMPLETE" and existing.get("cache_id") == cache_id:
            return output_dir
        raise ValueError("existing corrected tunneling validation has another cache identity")
    manifest: dict[str, object] = {
        **runtime_metadata(project_root),
        "status": "RUNNING",
        "stage": "corrected_tunneling_validation",
        "stage_version": TUNNELING_CORRECTION_VERSION,
        "cache_id": cache_id,
        "source_configuration_id": source_configuration["sha256"],
        "output_dir": str(output_dir),
    }
    write_json(manifest_path, manifest)
    try:
        summary_rows: list[dict[str, object]] = []
        state_rows: list[dict[str, object]] = []
        comparison_rows: list[dict[str, object]] = []
        schur_rows: list[dict[str, object]] = []
        exact_cases = [
            ("exact_2x2_weak", np.asarray([[0.0, 0.01], [0.01, 1.0]])),
            ("exact_2x2_strong", np.asarray([[0.2, 0.1], [0.1, 0.25]])),
        ]
        for name, dense in exact_cases:
            operator = sparse.csr_matrix(dense)
            exact_values, exact_vectors = linalg.eigh(dense, check_finite=False)
            result = subspace_diagnostics(operator, np.eye(2))
            comparison = _reference_comparison(result, exact_values, exact_vectors, True)
            summary_rows.append({"fixture": name, "expected_behavior": "exact_retained_subspace", **result.summary})
            state_rows.extend({"fixture": name, **row} for row in result.state_rows)
            comparison_rows.append(
                {
                    "fixture": name,
                    "energy_tolerance": 1e-12,
                    "mass_projector_maximum_tolerance": 1e-12,
                    "expected_to_pass_two_state_approximation": True,
                    **comparison,
                }
            )
            schur_rows.extend(_fixture_schur_rows(name, operator, result))

        graph_cases = [
            ("two_hub_weak_mixing", 8, 14, 5, 1, 0.20, 0.20, True),
            ("two_hub_strong_mixing", 8, 8, 1, 1, 0.80, 0.35, True),
            ("two_hub_mediated", 8, 8, 5, 1, 0.20, 0.20, True),
            ("two_hub_expected_failure", 8, 8, 5, 0, 0.20, 0.20, False),
        ]
        for name, left_degree, right_degree, separation, radius, energy_tol, mass_tol, expected_pass in graph_cases:
            graph, left_hub, right_hub = two_hub_graph(left_degree, right_degree, separation)
            operator = graph_to_csr(graph, nodelist=range(graph.number_of_nodes()))
            exact_values, exact_vectors = linalg.eigh(operator.toarray(), check_finite=False)
            states, local_rows = _local_state_matrix(graph, operator, [left_hub, right_hub], radius)
            result = subspace_diagnostics(operator, states)
            comparison = _reference_comparison(result, exact_values, exact_vectors, True)
            summary_rows.append(
                {
                    "fixture": name,
                    "expected_behavior": "pass" if expected_pass else "fail_two_state_approximation",
                    "left_degree": left_degree,
                    "right_degree": right_degree,
                    "center_distance": separation,
                    "radius": radius,
                    **result.summary,
                }
            )
            state_rows.extend(
                {"fixture": name, "center": int(center), **local, **diagnostic}
                for center, local, diagnostic in zip([left_hub, right_hub], local_rows, result.state_rows)
            )
            measured_pass = bool(
                result.valid
                and float(comparison["maximum_ritz_energy_error"]) <= energy_tol
                and float(comparison["projector_diagonal_maximum_error"]) <= mass_tol
            )
            comparison_rows.append(
                {
                    "fixture": name,
                    "energy_tolerance": energy_tol,
                    "mass_projector_maximum_tolerance": mass_tol,
                    "expected_to_pass_two_state_approximation": expected_pass,
                    "measured_pass_two_state_approximation": measured_pass,
                    "expectation_satisfied": bool(measured_pass == expected_pass),
                    **comparison,
                }
            )
            schur_rows.extend(_fixture_schur_rows(name, operator, result))

        maximum_decomposition = max(
            float(row.get("maximum_state_residual_decomposition_error", 0.0))
            for row in summary_rows
        )
        maximum_orthogonality = max(
            float(row.get("orthonormality_error", 0.0)) for row in summary_rows
        )
        exact_pass = all(
            float(row["maximum_ritz_energy_error"]) <= 1e-12
            and float(row["projector_diagonal_maximum_error"]) <= 1e-12
            for row in comparison_rows
            if str(row["fixture"]).startswith("exact_2x2")
        )
        expectations_pass = all(
            bool(row.get("expectation_satisfied", True))
            for row in comparison_rows
        )
        validation_passed = bool(
            exact_pass
            and expectations_pass
            and maximum_decomposition <= 1e-10
            and maximum_orthogonality <= 1e-10
        )
        write_csv(output_dir / "subspace_summary.csv", summary_rows)
        write_csv(output_dir / "state_diagnostics.csv", state_rows)
        write_csv(output_dir / "exact_eigenvalue_and_mass_validation.csv", comparison_rows)
        write_csv(output_dir / "schur_energy_regulator_and_complement.csv", schur_rows)
        if not validation_passed:
            raise AssertionError("corrected controlled tunneling validation did not satisfy declared tolerances")
        manifest.update(
            {
                "status": "COMPLETE",
                "finished_at_utc": utc_now(),
                "runtime_seconds": time.time() - started,
                "validation_passed": True,
                "fixture_count": len(summary_rows),
                "maximum_residual_squared_decomposition_error": maximum_decomposition,
                "maximum_orthonormality_error": maximum_orthogonality,
                "schur_row_count": len(schur_rows),
            }
        )
        write_json(manifest_path, manifest)
        return output_dir
    except Exception as error:
        manifest.update(
            {
                "status": "FAILED",
                "finished_at_utc": utc_now(),
                "runtime_seconds": time.time() - started,
                "failure_type": type(error).__name__,
                "failure_message": str(error),
            }
        )
        write_json(manifest_path, manifest)
        raise


def _load_existing_cl_bundle(input_dir: Path) -> SpectralBundle:
    input_dir = Path(input_dir)
    cache_dir = input_dir / "generated_cache"
    graph_path = cache_dir / "graph_realization.npz"
    selected_path = cache_dir / "selected_eigenvectors.npz"
    graph, graph_metadata = graph_from_edge_cache(graph_path)
    with np.load(selected_path, allow_pickle=False) as data:
        values = np.asarray(data["eigenvalues"], dtype=float)
        vectors = np.asarray(data["eigenvectors"], dtype=float)
        ranks = np.asarray(data["largest_algebraic_ranks"], dtype=int)
    graph_id = stable_json_hash({"graph_cache_sha256": sha256_file(graph_path)})
    operator_id = stable_json_hash({"graph_id": graph_id, "operator": "unscaled adjacency A"})
    return SpectralBundle(
        model="matched_chung_lu",
        graph=graph,
        operator=graph_to_csr(graph, nodelist=range(graph.number_of_nodes())),
        eigenvalues=values,
        eigenvectors=vectors,
        ranks=ranks,
        graph_id=graph_id,
        operator_id=operator_id,
        metadata={
            "source_dir": str(input_dir),
            "graph_cache": graph_metadata,
            "selected_eigenvectors_sha256": sha256_file(selected_path),
        },
    )


def run_corrected_tunneling_application(
    *,
    case: str,
    input_dir: Path,
    output_dir: Path,
    project_root: Path,
) -> Path:
    started = time.time()
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if case == "er":
        bundle = load_er_bundle(input_dir)
    elif case == "pa":
        bundle = load_pa_bundle(input_dir)
    elif case == "cl":
        bundle = _load_existing_cl_bundle(input_dir)
    else:
        raise ValueError("case must be er, pa, or cl")
    source_configuration = source_configuration_identity(project_root)
    isolation_values: np.ndarray | None = None
    isolation_spectrum_path: Path | None = None
    if case == "er":
        isolation_spectrum_path = input_dir / "full_spectrum_diagnostics.npz"
        with np.load(isolation_spectrum_path, allow_pickle=False) as data:
            isolation_values = np.asarray(data["eigenvalues"], dtype=float)
    input_identity = {
        "graph_id": bundle.graph_id,
        "operator_id": bundle.operator_id,
        "source_configuration_id": source_configuration["sha256"],
        "selection": "degree structural centers, separation>=3, largest local mode, radii 1 and 2",
        "isolation_spectrum_sha256": (
            sha256_file(isolation_spectrum_path) if isolation_spectrum_path is not None else None
        ),
    }
    cache_id = stable_json_hash({"version": TUNNELING_CORRECTION_VERSION, **input_identity})
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if existing.get("status") == "COMPLETE" and existing.get("cache_id") == cache_id:
            return output_dir
        raise ValueError("existing corrected tunneling application has another cache identity")
    manifest: dict[str, object] = {
        **runtime_metadata(project_root),
        "status": "RUNNING",
        "stage": "corrected_tunneling_application",
        "stage_version": TUNNELING_CORRECTION_VERSION,
        "case": case,
        "cache_id": cache_id,
        "source_configuration_id": source_configuration["sha256"],
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "reference_construction": "structural centers chosen before and without exact pair eigenvector maxima; largest induced-ball mode; no exact-pair mean energy",
        "old_exact_solution_assisted_diagnostic_status": "preserved separately for comparison; not overwritten",
    }
    write_json(manifest_path, manifest)
    try:
        centers = _structural_centers(bundle.graph, maximum=6, minimum_separation=3)
        if len(centers) < 2:
            raise RuntimeError("fewer than two independent structural centers found")
        original_ids = analyzed_to_original_ids(bundle.graph)
        center_rows = [
            {
                "center_order": order,
                "analyzed_vertex_id": center,
                "original_vertex_id": int(original_ids[center]),
                "degree": int(bundle.graph.degree(center)),
                "selection_rule": "degree descending, analyzed ID tie-break, greedy distance at least 3",
            }
            for order, center in enumerate(centers, start=1)
        ]
        summary_rows: list[dict[str, object]] = []
        state_rows: list[dict[str, object]] = []
        comparison_rows: list[dict[str, object]] = []
        schur_rows: list[dict[str, object]] = []
        exact_order = np.argsort(bundle.eigenvalues)
        exact_values = np.asarray(bundle.eigenvalues[exact_order], dtype=float)
        exact_vectors = np.asarray(bundle.eigenvectors[:, exact_order], dtype=float)
        for radius in (1, 2):
            states, local_rows = _local_state_matrix(
                bundle.graph, bundle.operator, centers, radius
            )
            result = subspace_diagnostics(bundle.operator, states)
            comparison = _reference_comparison(
                result,
                exact_values,
                exact_vectors,
                case == "er",
                isolation_values=isolation_values,
            )
            summary_rows.append({"case": case, "radius": radius, **result.summary})
            state_rows.extend(
                {
                    "case": case,
                    "radius": radius,
                    "center_analyzed_vertex_id": center,
                    "center_original_vertex_id": int(original_ids[center]),
                    **local,
                    **diagnostic,
                }
                for center, local, diagnostic in zip(centers, local_rows, result.state_rows)
            )
            comparison_rows.append({"case": case, "radius": radius, **comparison})
            if result.valid:
                pair_result = subspace_diagnostics(bundle.operator, states[:, :2])
                u = np.asarray(pair_result.orthonormal_states)
                k_pair = np.asarray(pair_result.effective_matrix)
                ritz = np.linalg.eigvalsh(k_pair)
                for energy in sorted(set([float(ritz[0]), float(np.mean(ritz)), float(ritz[-1])])):
                    for eta in (0.1, 0.01, 0.001):
                        schur = schur_effective_operator(bundle.operator, u, energy, eta)
                        schur_rows.append(
                            {
                                "case": case,
                                "radius": radius,
                                "left_center": centers[0],
                                "right_center": centers[1],
                                "orthonormal_basis_error_passed_to_schur": float(np.linalg.norm(u.T @ u - np.eye(2))),
                                "energy_role": "evaluation point only; no fixed-energy eigenvalue reproduction claim",
                                "complement_resonance_scope": "implicit through projected resolvent condition and eta dependence; full complement spectrum not formed",
                                **{key: value for key, value in schur.items() if key != "effective_matrix"},
                            }
                        )
        write_csv(output_dir / "structural_centers.csv", center_rows)
        write_csv(output_dir / "subspace_summary.csv", summary_rows)
        write_csv(output_dir / "state_diagnostics.csv", state_rows)
        write_csv(output_dir / "ritz_reference_comparison.csv", comparison_rows)
        write_csv(output_dir / "schur_energy_and_regulator_sensitivity.csv", schur_rows)
        manifest.update(
            {
                "status": "COMPLETE",
                "finished_at_utc": utc_now(),
                "runtime_seconds": time.time() - started,
                "model": bundle.model,
                "graph_id": bundle.graph_id,
                "operator_id": bundle.operator_id,
                "analyzed_n": bundle.graph.number_of_nodes(),
                "structural_center_count": len(centers),
                "radius_count": len(summary_rows),
                "all_subspaces_valid": all(bool(row["orthogonalization_valid"]) for row in summary_rows),
                "isolation_claim_status": (
                    "full_omitted_eigenvalue_separation_audited"
                    if case == "er"
                    else "not_established_without_full_omitted_spectrum"
                ),
                "model_metadata": bundle.metadata,
            }
        )
        write_json(manifest_path, manifest)
        return output_dir
    except Exception as error:
        manifest.update(
            {
                "status": "FAILED",
                "finished_at_utc": utc_now(),
                "runtime_seconds": time.time() - started,
                "failure_type": type(error).__name__,
                "failure_message": str(error),
            }
        )
        write_json(manifest_path, manifest)
        raise


def run_paper_er_nonempty(
    *,
    input_dir: Path,
    old_application_dir: Path,
    output_dir: Path,
    project_root: Path,
) -> Path:
    started = time.time()
    input_dir = Path(input_dir)
    old_application_dir = Path(old_application_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_er_bundle(input_dir)
    source_configuration = source_configuration_identity(project_root)
    old_center_path = old_application_dir / "paper_center_sets.json"
    cache_id = stable_json_hash(
        {
            "version": TUNNELING_CORRECTION_VERSION,
            "source_configuration_id": source_configuration["sha256"],
            "graph_id": bundle.graph_id,
            "original_center_set_sha256": sha256_file(old_center_path),
            "attempts": [(0.02, 0.1), (0.02, 0.05), (0.02, 0.01), (0.05, 0.1), (0.1, 0.1), (0.2, 0.1), (0.3, 0.1)],
        }
    )
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if existing.get("status") == "COMPLETE" and existing.get("cache_id") == cache_id:
            return output_dir
        raise ValueError("existing paper ER correction has another cache identity")
    manifest: dict[str, object] = {
        **runtime_metadata(project_root),
        "status": "RUNNING",
        "stage": "paper_informed_er_nonempty_correction",
        "stage_version": TUNNELING_CORRECTION_VERSION,
        "cache_id": cache_id,
        "source_configuration_id": source_configuration["sha256"],
        "input_dir": str(input_dir),
        "old_application_dir": str(old_application_dir),
        "output_dir": str(output_dir),
        "original_empty_outcome_preserved": True,
        "adaptation_policy": "same graph/size/seed; bounded predeclared parameter attempts; first nonempty W is reported as exploratory",
    }
    write_json(manifest_path, manifest)
    try:
        degrees = [bundle.graph.degree(vertex) for vertex in range(bundle.graph.number_of_nodes())]
        original_n = int(bundle.metadata["original_n"])
        d = float(bundle.metadata["d"])
        attempts = [(0.02, 0.1), (0.02, 0.05), (0.02, 0.01), (0.05, 0.1), (0.1, 0.1), (0.2, 0.1), (0.3, 0.1)]
        attempt_rows: list[dict[str, object]] = []
        selected_parameters: tuple[float, float] | None = None
        for attempt_index, (mu, kappa) in enumerate(attempts):
            sets = paper_center_sets(degrees, original_n, d, mu, kappa)
            row = {
                "attempt_index": attempt_index,
                "mu": mu,
                "kappa": kappa,
                "source_prescribed_parameters": bool(mu == 0.02 and kappa == 0.1),
                "theorem_mu_range_satisfied": bool(sets["theorem_mu_range_satisfied"]),
                "alpha_star": float(sets["alpha_star"]),
                "V_count": len(sets["V"]),
                "W_count": len(sets["W"]),
                "outcome": "nonempty" if len(sets["W"]) else "empty",
            }
            attempt_rows.append(row)
            if selected_parameters is None and len(sets["W"]):
                selected_parameters = (mu, kappa)
        if selected_parameters is None:
            raise RuntimeError("bounded paper-informed ER attempts all produced empty W")
        mu, kappa = selected_parameters
        states, center_sets = paper_global_approximate_states(
            bundle.operator,
            degrees,
            original_n=original_n,
            d=d,
            mu=mu,
            kappa=kappa,
            maximum_centers=6,
        )
        if not states:
            raise AssertionError("selected nonempty center set produced no approximate states")
        state_matrix = np.column_stack([np.asarray(state["state"], dtype=float) for state in states])
        result = subspace_diagnostics(bundle.operator, state_matrix)
        state_rows = [
            {
                **{key: value for key, value in state.items() if key != "state"},
                **(result.state_rows[index] if index < len(result.state_rows) else {}),
            }
            for index, state in enumerate(states)
        ]
        pair_rows = approximate_state_pair_rows(states)
        for row in pair_rows:
            residual_scale = max(float(row["left_residual_norm"]), float(row["right_residual_norm"]))
            separation = float(row["approximate_energy_separation"])
            row["maximum_residual_over_approximate_energy_separation"] = (
                residual_scale / separation if separation > 0.0 else math.inf
            )
            row["isolation_claim_status"] = "not_established_without_omitted_mode_separation"
        pair_output_rows = pair_rows or [
            {
                "status": "no_pair_because_only_one_approximate_state",
                "isolation_claim_status": "not_applicable",
            }
        ]
        write_csv(output_dir / "parameter_attempts.csv", attempt_rows)
        write_csv(output_dir / "paper_approximate_states_corrected.csv", state_rows)
        write_csv(output_dir / "paper_approximate_energy_pairs_corrected.csv", pair_output_rows)
        write_json(
            output_dir / "selected_center_sets.json",
            {
                key: value
                for key, value in center_sets.items()
                if key != "normalized_degrees"
            },
        )
        write_json(output_dir / "subspace_summary.json", result.summary)
        manifest.update(
            {
                "status": "COMPLETE",
                "finished_at_utc": utc_now(),
                "runtime_seconds": time.time() - started,
                "original_source_prescribed_outcome": attempt_rows[0],
                "selected_exploratory_parameters": {"mu": mu, "kappa": kappa},
                "selected_parameters_within_theorem_mu_range": bool(mu < 1.0 / 24.0),
                "approximate_state_count": len(states),
                "approximate_pair_count": len(pair_rows),
                "subspace_valid": result.valid,
                "scientific_status": "nonempty exploratory execution; no localization theorem or tunneling claim transferred without spectral isolation",
            }
        )
        write_json(manifest_path, manifest)
        return output_dir
    except Exception as error:
        manifest.update(
            {
                "status": "FAILED",
                "finished_at_utc": utc_now(),
                "runtime_seconds": time.time() - started,
                "failure_type": type(error).__name__,
                "failure_message": str(error),
            }
        )
        write_json(manifest_path, manifest)
        raise
