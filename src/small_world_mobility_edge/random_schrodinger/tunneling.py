"""Paper-informed approximate states and exploratory effective couplings."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Sequence

import networkx as nx
import numpy as np
from scipy import linalg, sparse, stats
from scipy.sparse import linalg as sparse_linalg

from .common import (
    graph_to_csr,
    runtime_metadata,
    stable_json_hash,
    utc_now,
    write_csv,
    write_json,
)


TUNNELING_VERSION = "approximate_state_effective_coupling_v1"


def Lambda(alpha: float | np.ndarray) -> float | np.ndarray:
    value = np.asarray(alpha, dtype=float)
    result = value / np.sqrt(value - 1.0)
    return float(result) if result.ndim == 0 else result


def paper_alpha_star(n: int, d: float, mu: float, kappa: float) -> dict[str, object]:
    """Finite-N implementation of 2024 equation (1.6).

    Degrees in G(N,d/N) have Binomial(N-1,d/N) law.  Since normalized
    degrees lie on the grid k/d, the infimum is evaluated exactly on that grid.
    """
    if n < 2 or not 0.0 < d < n or not 0.0 <= mu <= 1.0 or not 0.0 < kappa < 1.0:
        raise ValueError("invalid paper center-set parameters")
    probability_threshold = float(n ** (mu - 1.0))
    p = float(d / n)
    selected_degree = n - 1
    selected_tail = float(stats.binom.sf(selected_degree - 1, n - 1, p))
    for degree in range(n):
        tail = float(stats.binom.sf(degree - 1, n - 1, p))
        if tail <= probability_threshold:
            selected_degree = degree
            selected_tail = tail
            break
    raw = float(selected_degree / d)
    alpha_star = max(raw, 2.0 + float(kappa))
    return {
        "alpha_star": alpha_star,
        "raw_binomial_quantile_alpha": raw,
        "threshold_degree": selected_degree,
        "tail_probability_at_threshold": selected_tail,
        "probability_threshold": probability_threshold,
        "binomial_trials": n - 1,
        "binomial_p": p,
    }


def paper_center_sets(
    degrees: Sequence[int],
    original_n: int,
    d: float,
    mu: float,
    kappa: float,
) -> dict[str, object]:
    degrees = np.asarray(degrees, dtype=float)
    threshold = paper_alpha_star(original_n, d, mu, kappa)
    alpha = degrees / float(d)
    alpha_star = float(threshold["alpha_star"])
    v_set = np.flatnonzero(alpha >= alpha_star).astype(int)
    w_threshold = float(Lambda(alpha_star) + kappa / 2.0)
    eligible = np.zeros(len(alpha), dtype=bool)
    eligible[v_set] = np.asarray(Lambda(alpha[v_set]), dtype=float) >= w_threshold
    w_set = np.flatnonzero(eligible).astype(int)
    return {
        **threshold,
        "normalized_degrees": alpha,
        "V": v_set,
        "W": w_set,
        "W_lambda_threshold": w_threshold,
        "definition": "2024 equations (1.6), (2.1), and (2.2) on the finite analyzed graph; d and original N are not recomputed after giant-component restriction",
        "theorem_mu_range_satisfied": bool(0.0 < mu < 1.0 / 24.0),
    }


def _top_eigenpairs(matrix: sparse.spmatrix, count: int = 3) -> tuple[np.ndarray, np.ndarray]:
    n = matrix.shape[0]
    if n <= max(12, count + 2):
        values, vectors = linalg.eigh(matrix.toarray(), check_finite=False)
        order = np.argsort(values)[::-1][:count]
        return np.asarray(values[order], dtype=float), np.asarray(vectors[:, order], dtype=float)
    k = min(count, n - 2)
    values, vectors = sparse_linalg.eigsh(matrix, k=k, which="LA", tol=1e-10)
    order = np.argsort(values)[::-1]
    return np.asarray(values[order], dtype=float), np.asarray(vectors[:, order], dtype=float)


def paper_global_approximate_states(
    operator: sparse.spmatrix,
    degrees: Sequence[int],
    original_n: int,
    d: float,
    mu: float,
    kappa: float,
    maximum_centers: int | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Implement Definition 2.1 using H^(V\\{x}) and its second eigenpair."""
    operator = sparse.csr_matrix(operator)
    center_sets = paper_center_sets(degrees, original_n, d, mu, kappa)
    v_set = np.asarray(center_sets["V"], dtype=int)
    w_set = np.asarray(center_sets["W"], dtype=int)
    if maximum_centers is not None:
        normalized = np.asarray(center_sets["normalized_degrees"], dtype=float)
        order = np.lexsort((w_set, -normalized[w_set]))
        w_set = w_set[order[: int(maximum_centers)]]
    n_eff = operator.shape[0]
    states: list[dict[str, object]] = []
    for center in w_set:
        removed = set(int(vertex) for vertex in v_set if int(vertex) != int(center))
        retained = np.asarray([vertex for vertex in range(n_eff) if vertex not in removed], dtype=int)
        restricted = operator[retained][:, retained]
        values, vectors = _top_eigenpairs(restricted, count=3)
        if len(values) < 2:
            raise RuntimeError("paper approximate-state matrix has fewer than two eigenpairs")
        energy = float(values[1])
        local = np.asarray(vectors[:, 1], dtype=float)
        embedded = np.zeros(n_eff, dtype=float)
        embedded[retained] = local
        if embedded[int(center)] < 0.0:
            embedded *= -1.0
        residual = np.asarray(operator.dot(embedded)).reshape(-1) - energy * embedded
        states.append(
            {
                "center": int(center),
                "normalized_degree": float(center_sets["normalized_degrees"][center]),
                "approximate_energy": energy,
                "state": embedded,
                "removed_center_count": len(removed),
                "retained_dimension": len(retained),
                "residual_norm": float(np.linalg.norm(residual)),
                "relative_residual_norm": float(np.linalg.norm(residual)) / max(1.0, abs(energy)),
                "center_component": float(embedded[int(center)]),
                "definition": "lambda_2 and normalized eigenvector of H^(V\\{x}), embedded with zeros",
            }
        )
    return states, center_sets


def approximate_state_pair_rows(states: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for left in range(len(states)):
        for right in range(left + 1, len(states)):
            left_state = np.asarray(states[left]["state"], dtype=float)
            right_state = np.asarray(states[right]["state"], dtype=float)
            left_energy = float(states[left]["approximate_energy"])
            right_energy = float(states[right]["approximate_energy"])
            rows.append(
                {
                    "left_center": int(states[left]["center"]),
                    "right_center": int(states[right]["center"]),
                    "left_approximate_energy": left_energy,
                    "right_approximate_energy": right_energy,
                    "approximate_energy_separation": abs(left_energy - right_energy),
                    "state_nonorthogonality": float(np.dot(left_state, right_state)),
                    "left_residual_norm": float(states[left]["residual_norm"]),
                    "right_residual_norm": float(states[right]["residual_norm"]),
                }
            )
    return rows


def local_ball_reference_state(
    graph: nx.Graph,
    operator: sparse.spmatrix,
    center: int,
    radius: int,
    mode_selection: str = "largest_algebraic",
    target_energy: float | None = None,
) -> dict[str, object]:
    """Diagonalize the induced radius-r ball with a Dirichlet boundary."""
    center = int(center)
    nodes = np.asarray(
        sorted(nx.single_source_shortest_path_length(graph, center, cutoff=radius).keys()),
        dtype=int,
    )
    restricted = sparse.csr_matrix(operator)[nodes][:, nodes]
    n_local = len(nodes)
    if n_local <= 256:
        values, vectors = linalg.eigh(restricted.toarray(), check_finite=False)
        if mode_selection == "largest_algebraic":
            position = len(values) - 1
        elif mode_selection == "nearest_target":
            if target_energy is None:
                raise ValueError("nearest_target requires target_energy")
            position = int(np.argmin(np.abs(values - float(target_energy))))
        else:
            raise ValueError("unsupported mode selection")
        local_value = float(values[position])
        local_vector = np.asarray(vectors[:, position], dtype=float)
    else:
        if mode_selection == "largest_algebraic":
            values, vectors = sparse_linalg.eigsh(restricted, k=1, which="LA", tol=1e-10)
        elif mode_selection == "nearest_target":
            if target_energy is None:
                raise ValueError("nearest_target requires target_energy")
            values, vectors = sparse_linalg.eigsh(
                restricted, k=1, sigma=float(target_energy), which="LM", tol=1e-10
            )
        else:
            raise ValueError("unsupported mode selection")
        local_value = float(values[0])
        local_vector = np.asarray(vectors[:, 0], dtype=float)
    embedded = np.zeros(operator.shape[0], dtype=float)
    embedded[nodes] = local_vector
    norm = float(np.linalg.norm(embedded))
    if norm <= 0.0:
        raise RuntimeError("zero local reference state")
    embedded /= norm
    if embedded[center] < 0.0:
        embedded *= -1.0
    residual = np.asarray(operator.dot(embedded)).reshape(-1) - local_value * embedded
    node_set = set(nodes.tolist())
    boundary_edges = sum(
        1
        for node in nodes
        for other in graph.neighbors(int(node))
        if int(other) not in node_set
    )
    return {
        "center": center,
        "radius": int(radius),
        "boundary_condition": "Dirichlet induced-ball restriction",
        "mode_selection": mode_selection,
        "target_energy": target_energy,
        "local_dimension": n_local,
        "local_energy": local_value,
        "state": embedded,
        "residual_norm": float(np.linalg.norm(residual)),
        "relative_residual_norm": float(np.linalg.norm(residual)) / max(1.0, abs(local_value)),
        "boundary_edge_count_with_multiplicity": int(boundary_edges),
    }


def symmetric_orthogonalization(states: np.ndarray, tolerance: float = 1e-10) -> dict[str, object]:
    states = np.asarray(states, dtype=float)
    gram = states.T @ states
    eigenvalues, eigenvectors = linalg.eigh(gram, check_finite=False)
    maximum = float(np.max(eigenvalues)) if len(eigenvalues) else 0.0
    minimum = float(np.min(eigenvalues)) if len(eigenvalues) else 0.0
    threshold = tolerance * max(maximum, 1.0)
    valid = minimum > threshold
    if not valid:
        return {
            "valid": False,
            "states": None,
            "gram": gram,
            "gram_eigenvalues": eigenvalues,
            "gram_condition_number": math.inf,
            "reason": "rank_deficient_or_ill_conditioned_Gram",
        }
    inverse_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    orthonormal = states @ inverse_sqrt
    return {
        "valid": True,
        "states": orthonormal,
        "transform": inverse_sqrt,
        "gram": gram,
        "gram_eigenvalues": eigenvalues,
        "gram_condition_number": maximum / minimum,
        "orthogonality_error": float(np.linalg.norm(orthonormal.T @ orthonormal - np.eye(states.shape[1]))),
        "reason": "valid",
    }


def projected_effective_pair(
    operator: sparse.spmatrix,
    left_state: np.ndarray,
    right_state: np.ndarray,
    observed_gap: float,
) -> dict[str, object]:
    operator = sparse.csr_matrix(operator)
    states = np.column_stack([left_state, right_state])
    raw_gram = states.T @ states
    raw_k = states.T @ operator.dot(states)
    orthogonalized = symmetric_orthogonalization(states)
    base = {
        "raw_gram_00": float(raw_gram[0, 0]),
        "raw_gram_01": float(raw_gram[0, 1]),
        "raw_gram_11": float(raw_gram[1, 1]),
        "raw_K_00": float(raw_k[0, 0]),
        "raw_K_01": float(raw_k[0, 1]),
        "raw_K_11": float(raw_k[1, 1]),
        "gram_condition_number": float(orthogonalized["gram_condition_number"]),
        "observed_adjacent_gap": float(observed_gap),
        "observed_gap_role": "full-operator mixed gap; not uncoupled detuning",
    }
    if not bool(orthogonalized["valid"]):
        return {
            **base,
            "valid": False,
            "validity_flag": orthogonalized["reason"],
            "orthonormal_states": None,
            "effective_matrix": None,
            "e_left": math.nan,
            "e_right": math.nan,
            "direct_coupling": math.nan,
            "uncoupled_detuning": math.nan,
            "predicted_two_state_splitting": math.nan,
        }
    orthonormal = np.asarray(orthogonalized["states"], dtype=float)
    effective = orthonormal.T @ operator.dot(orthonormal)
    effective = 0.5 * (effective + effective.T)
    e_left = float(effective[0, 0])
    e_right = float(effective[1, 1])
    coupling = float(effective[0, 1])
    detuning = abs(e_left - e_right)
    predicted = math.sqrt(detuning**2 + 4.0 * abs(coupling) ** 2)
    return {
        **base,
        "valid": True,
        "validity_flag": "valid_direct_projection",
        "orthonormal_states": orthonormal,
        "effective_matrix": effective,
        "orthogonality_error": float(orthogonalized["orthogonality_error"]),
        "e_left": e_left,
        "e_right": e_right,
        "direct_coupling": coupling,
        "absolute_direct_coupling": abs(coupling),
        "uncoupled_detuning": detuning,
        "predicted_two_state_splitting": predicted,
        "coupling_over_detuning": abs(coupling) / detuning if detuning > 0.0 else math.inf,
        "coupling_over_observed_gap": abs(coupling) / observed_gap if observed_gap > 0.0 else math.inf,
    }


def schur_effective_operator(
    operator: sparse.spmatrix,
    orthonormal_states: np.ndarray,
    energy: float,
    eta: float,
) -> dict[str, object]:
    """Evaluate the Feshbach/Schur operator via full-resolvent linear solves.

    For orthonormal P, H_eff(z)=zI-[P*(zI-H)^(-1)P]^(-1), which equals
    H_PP + H_PQ(zI-H_QQ)^(-1)H_QP.  No dense complement or inverse is formed.
    """
    operator = sparse.csc_matrix(operator)
    states = np.asarray(orthonormal_states, dtype=complex)
    if not np.allclose(states.conj().T @ states, np.eye(states.shape[1]), atol=1e-8, rtol=0.0):
        raise ValueError("Schur reference states must be orthonormal")
    z = complex(float(energy), float(eta))
    system = z * sparse.identity(operator.shape[0], dtype=complex, format="csc") - operator.astype(complex)
    solutions = sparse_linalg.spsolve(system, states)
    if solutions.ndim == 1:
        solutions = solutions[:, None]
    projected_resolvent = states.conj().T @ solutions
    condition = float(np.linalg.cond(projected_resolvent))
    effective = z * np.eye(states.shape[1], dtype=complex) - np.linalg.inv(projected_resolvent)
    solve_residual = system.dot(solutions) - states
    return {
        "energy": float(energy),
        "eta": float(eta),
        "effective_matrix": effective,
        "effective_00_real": float(np.real(effective[0, 0])),
        "effective_11_real": float(np.real(effective[1, 1])),
        "effective_01_real": float(np.real(effective[0, 1])),
        "effective_01_imag": float(np.imag(effective[0, 1])),
        "absolute_mediated_coupling": float(abs(effective[0, 1])),
        "projected_resolvent_condition_number": condition,
        "linear_solve_residual_norm": float(np.linalg.norm(solve_residual)),
        "near_pole_flag": bool(condition > 1e10 or np.linalg.norm(effective.imag) > 10.0 * max(abs(eta), 1e-15)),
        "method": "resolvent-equivalent Schur complement using sparse linear solves",
    }


def effective_pair_from_centers(
    graph: nx.Graph,
    operator: sparse.spmatrix,
    centers: tuple[int, int],
    radius: int,
    observed_gap: float,
    target_energy: float | None = None,
    eta_values: Sequence[float] = (),
) -> tuple[dict[str, object], list[dict[str, object]]]:
    mode = "largest_algebraic" if target_energy is None else "nearest_target"
    left = local_ball_reference_state(graph, operator, centers[0], radius, mode, target_energy)
    right = local_ball_reference_state(graph, operator, centers[1], radius, mode, target_energy)
    direct = projected_effective_pair(operator, left["state"], right["state"], observed_gap)
    summary = {
        "left_center": int(centers[0]),
        "right_center": int(centers[1]),
        "radius": int(radius),
        "boundary_condition": "Dirichlet induced-ball restriction",
        "mode_selection": mode,
        "target_energy": target_energy,
        "left_local_energy": float(left["local_energy"]),
        "right_local_energy": float(right["local_energy"]),
        "left_residual_norm": float(left["residual_norm"]),
        "right_residual_norm": float(right["residual_norm"]),
        "left_local_dimension": int(left["local_dimension"]),
        "right_local_dimension": int(right["local_dimension"]),
        **{key: value for key, value in direct.items() if key not in {"orthonormal_states", "effective_matrix"}},
    }
    schur_rows: list[dict[str, object]] = []
    if bool(direct["valid"]) and eta_values:
        energy = float(target_energy) if target_energy is not None else 0.5 * (
            float(direct["e_left"]) + float(direct["e_right"])
        )
        for eta in eta_values:
            schur_rows.append(
                {
                    "left_center": int(centers[0]),
                    "right_center": int(centers[1]),
                    "radius": int(radius),
                    **schur_effective_operator(
                        operator, np.asarray(direct["orthonormal_states"]), energy, float(eta)
                    ),
                }
            )
    return summary, schur_rows


def two_hub_graph(left_degree: int, right_degree: int, separation: int) -> tuple[nx.Graph, int, int]:
    if left_degree < 2 or right_degree < 2 or separation < 1:
        raise ValueError("invalid two-hub fixture")
    graph = nx.Graph()
    left_hub = 0
    next_vertex = 1
    left_leaves = list(range(next_vertex, next_vertex + left_degree))
    next_vertex += left_degree
    graph.add_edges_from((left_hub, leaf) for leaf in left_leaves)
    previous = left_hub
    for _ in range(separation - 1):
        graph.add_edge(previous, next_vertex)
        previous = next_vertex
        next_vertex += 1
    right_hub = next_vertex
    graph.add_edge(previous, right_hub)
    next_vertex += 1
    right_leaves = list(range(next_vertex, next_vertex + right_degree))
    graph.add_edges_from((right_hub, leaf) for leaf in right_leaves)
    graph = nx.convert_node_labels_to_integers(graph, ordering="sorted")
    return graph, left_hub, right_hub


def run_tunneling_validation(output_dir: Path, project_root: Path) -> Path:
    """Run exact two-state and controlled graph fixtures on a Hoffman node."""
    start = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    manifest: dict[str, object] = {
        **runtime_metadata(project_root),
        "stage": "stage2_tunneling_validation",
        "stage_version": TUNNELING_VERSION,
        "output_dir": str(output_dir),
        "fixtures_are_project_validation_designs_not_published_experiments": True,
    }
    write_json(manifest_path, manifest)
    try:
        rows: list[dict[str, object]] = []
        schur_rows: list[dict[str, object]] = []
        exact_cases = [(0.0, 1.0, 0.01), (0.2, 0.25, 0.1), (-0.3, 0.7, 0.25)]
        for case_index, (left_energy, right_energy, coupling) in enumerate(exact_cases):
            matrix = sparse.csr_matrix([[left_energy, coupling], [coupling, right_energy]], dtype=float)
            observed = np.linalg.eigvalsh(matrix.toarray())
            diagnostic = projected_effective_pair(
                matrix, np.array([1.0, 0.0]), np.array([0.0, 1.0]), float(observed[1] - observed[0])
            )
            expected = math.sqrt((left_energy - right_energy) ** 2 + 4.0 * coupling**2)
            error = abs(float(diagnostic["predicted_two_state_splitting"]) - expected)
            rows.append(
                {
                    "fixture": "exact_two_state",
                    "case_index": case_index,
                    "left_degree": math.nan,
                    "right_degree": math.nan,
                    "center_distance": 1,
                    "radius": 0,
                    "expected_splitting": expected,
                    "exact_observed_splitting": float(observed[1] - observed[0]),
                    "splitting_absolute_error": error,
                    **{key: value for key, value in diagnostic.items() if key not in {"orthonormal_states", "effective_matrix"}},
                }
            )
            schur = schur_effective_operator(
                matrix, np.eye(2), float(np.mean(observed)), eta=1e-3
            )
            schur_rows.append({"fixture": "exact_two_state", "case_index": case_index, **schur})

        graph_cases = [(8, 8, 1), (8, 8, 3), (8, 8, 5), (8, 12, 3), (8, 16, 5)]
        for case_index, (left_degree, right_degree, separation) in enumerate(graph_cases):
            graph, left_hub, right_hub = two_hub_graph(left_degree, right_degree, separation)
            operator = graph_to_csr(graph, nodelist=range(graph.number_of_nodes()))
            exact_values, exact_vectors = linalg.eigh(operator.toarray(), check_finite=False)
            observed_gap = float(exact_values[-1] - exact_values[-2])
            for radius in (1, 2):
                summary, mediated = effective_pair_from_centers(
                    graph,
                    operator,
                    (left_hub, right_hub),
                    radius=radius,
                    observed_gap=observed_gap,
                    eta_values=(1e-2, 1e-3),
                )
                rows.append(
                    {
                        "fixture": "two_hub_graph",
                        "case_index": case_index,
                        "left_degree": left_degree,
                        "right_degree": right_degree,
                        "center_distance": nx.shortest_path_length(graph, left_hub, right_hub),
                        "exact_observed_splitting": observed_gap,
                        "top_mode_left_hub_mass": float(exact_vectors[left_hub, -1] ** 2),
                        "top_mode_right_hub_mass": float(exact_vectors[right_hub, -1] ** 2),
                        "second_mode_left_hub_mass": float(exact_vectors[left_hub, -2] ** 2),
                        "second_mode_right_hub_mass": float(exact_vectors[right_hub, -2] ** 2),
                        **summary,
                    }
                )
                for item in mediated:
                    schur_rows.append(
                        {
                            "fixture": "two_hub_graph",
                            "case_index": case_index,
                            "left_degree": left_degree,
                            "right_degree": right_degree,
                            "center_distance": separation,
                            **item,
                        }
                    )

        one_hub = nx.star_graph(12)
        one_hub.add_edge(1, 13)
        one_hub.add_edge(13, 14)
        one_operator = graph_to_csr(one_hub, nodelist=range(one_hub.number_of_nodes()))
        one_values, one_vectors = linalg.eigh(one_operator.toarray(), check_finite=False)
        one_state = local_ball_reference_state(one_hub, one_operator, 0, radius=1)
        rows.append(
            {
                "fixture": "one_hub_graph",
                "case_index": 0,
                "left_degree": 12,
                "right_degree": math.nan,
                "center_distance": math.nan,
                "radius": 1,
                "exact_observed_splitting": float(one_values[-1] - one_values[-2]),
                "top_mode_hub_mass": float(one_vectors[0, -1] ** 2),
                "left_local_energy": one_state["local_energy"],
                "left_residual_norm": one_state["residual_norm"],
                "valid": True,
                "validity_flag": "single_center_fixture",
            }
        )
        write_csv(output_dir / "validation_cases.csv", rows)
        write_csv(output_dir / "schur_sensitivity.csv", schur_rows)
        exact_errors = [
            float(row["splitting_absolute_error"])
            for row in rows
            if row["fixture"] == "exact_two_state"
        ]
        passed = max(exact_errors) < 1e-12
        if not passed:
            raise AssertionError("exact two-state splitting validation failed")
        manifest.update(
            {
                "status": "COMPLETE",
                "finished_at_utc": utc_now(),
                "runtime_seconds": time.time() - start,
                "case_count": len(rows),
                "schur_case_count": len(schur_rows),
                "maximum_exact_two_state_splitting_error": max(exact_errors),
                "validation_passed": True,
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
