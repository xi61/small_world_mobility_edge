from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path

import networkx as nx
import numpy as np

from small_world_mobility_edge import chung_lu_eigenvector_percentile_scan as scan
from small_world_mobility_edge.chung_lu_baseline import generate_chung_lu_graph_from_weights


ATOM_ENERGY = 1.0
ATOM_ATOL = 1e-8
TARGET_QUANTILE = 0.683


def read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def quantile_row_by_graph(path: Path, target_quantile: float) -> dict[int, dict[str, str]]:
    selected: dict[int, dict[str, str]] = {}
    for row in read_csv(path):
        if math.isclose(float(row["quantile"]), target_quantile, rel_tol=0.0, abs_tol=1e-12):
            selected[int(row["graph_index"])] = row
    return selected


def exact_atom_multiplicity(graph: nx.Graph) -> int:
    if graph.number_of_nodes() == 0:
        return 0
    matrix = nx.to_numpy_array(graph, dtype=float)
    eigenvalues = np.linalg.eigvalsh(matrix)
    return int(np.sum(np.isclose(eigenvalues, ATOM_ENERGY, atol=ATOM_ATOL, rtol=0.0)))


def pendant_length_two_atom_lower_bound(graph: nx.Graph) -> tuple[int, int, int]:
    """Count the +1 modes forced by repeated pendant length-two arms.

    If a root has k arms root--middle--leaf with degree(middle)=2 and
    degree(leaf)=1, differences between arms give k-1 independent +1 modes.
    The same construction also supplies k-1 modes at -1.
    """

    predicted_multiplicity = 0
    roots_with_repeated_arms = 0
    total_arms_at_repeated_roots = 0
    degrees = dict(graph.degree())
    for root in graph:
        arm_count = 0
        for middle in graph.neighbors(root):
            if degrees[middle] != 2:
                continue
            middle_neighbors = list(graph.neighbors(middle))
            leaf = middle_neighbors[0] if middle_neighbors[1] == root else middle_neighbors[1]
            if degrees[leaf] == 1:
                arm_count += 1
        if arm_count >= 2:
            roots_with_repeated_arms += 1
            total_arms_at_repeated_roots += arm_count
            predicted_multiplicity += arm_count - 1
    return predicted_multiplicity, roots_with_repeated_arms, total_arms_at_repeated_roots


def analyze_graph(
    graph_index: int,
    graph_seed: int,
    sequence_seed: int,
    displayed_tau: float,
    eigenvalue_row: dict[str, str],
    eigenvector_row: dict[str, str],
) -> dict[str, object]:
    weights, diagnostics = scan.generate_legacy_expected_weights(
        displayed_tau=displayed_tau,
        sequence_seed=sequence_seed,
    )
    if not diagnostics.admissibility_holds:
        raise ValueError(f"graph {graph_index}: inadmissible expected-weight sequence")
    graph = generate_chung_lu_graph_from_weights(weights, seed=graph_seed)
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    giant_nodes = components[0]
    giant = graph.subgraph(giant_nodes)

    small_component_atom_count = 0
    isolated_edge_count = 0
    size_counts = {2: 0, 3: 0, 4: 0, 5: 0}
    components_size_6_plus = 0
    for nodes in components[1:]:
        component = graph.subgraph(nodes)
        size = component.number_of_nodes()
        if size in size_counts:
            size_counts[size] += 1
        elif size >= 6:
            components_size_6_plus += 1
        if size == 2 and component.number_of_edges() == 1:
            isolated_edge_count += 1
        if size > 1:
            small_component_atom_count += exact_atom_multiplicity(component)

    (
        pendant_p2_atom_lower_bound,
        roots_with_repeated_p2_arms,
        repeated_p2_arm_count,
    ) = pendant_length_two_atom_lower_bound(giant)
    full_atom_count = int(eigenvalue_row["full_spectrum_atom_count_at_1"])
    residual_giant_atom_count = full_atom_count - small_component_atom_count

    return {
        "graph_index": graph_index,
        "graph_seed": graph_seed,
        "sequence_seed": sequence_seed,
        "displayed_tau": displayed_tau,
        "number_of_nodes": graph.number_of_nodes(),
        "number_of_edges": graph.number_of_edges(),
        "number_of_components": len(components),
        "largest_component_size": len(giant_nodes),
        "isolated_vertex_count": sum(len(nodes) == 1 for nodes in components[1:]),
        "isolated_edge_count": isolated_edge_count,
        "size_2_component_count": size_counts[2],
        "size_3_component_count": size_counts[3],
        "size_4_component_count": size_counts[4],
        "size_5_component_count": size_counts[5],
        "size_6_plus_non_giant_component_count": components_size_6_plus,
        "full_graph_exact_plus_1_multiplicity": full_atom_count,
        "non_giant_exact_plus_1_multiplicity": small_component_atom_count,
        "residual_giant_exact_plus_1_multiplicity": residual_giant_atom_count,
        "giant_pendant_p2_plus_1_lower_bound": pendant_p2_atom_lower_bound,
        "giant_roots_with_repeated_p2_arms": roots_with_repeated_p2_arms,
        "giant_repeated_p2_arm_count": repeated_p2_arm_count,
        "q_0p683_selected_atom_count": int(eigenvalue_row["selected_atom_count_at_1"]),
        "q_0p683_selected_atom_fraction": float(eigenvalue_row["selected_atom_fraction_at_1"]),
        "q_0p683_mean_n_times_ipr": float(eigenvector_row["mean_n_times_ipr"]),
        "q_0p683_mean_top_50_mass": float(eigenvector_row["mean_top_50_mass"]),
        "q_0p683_mean_r_90": float(eigenvector_row["mean_r_90"]),
        "zero_eigenvalue_count": int(eigenvalue_row["zero_eigenvalue_count"]),
        "zero_eigenvalue_fraction": float(eigenvalue_row["zero_eigenvalue_fraction"]),
        "continuous_nonperron_dimension": int(eigenvalue_row["continuous_bulk_dimension"]),
    }


def mean_standard_error(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    if len(array) < 2:
        return mean, math.nan
    return mean, float(np.std(array, ddof=1) / np.sqrt(len(array)))


def correlation(rows: list[dict[str, object]], left: str, right: str) -> float:
    x = np.asarray([float(row[left]) for row in rows])
    y = np.asarray([float(row[right]) for row in rows])
    return float(np.corrcoef(x, y)[0, 1])


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    metric_names = (
        "largest_component_size",
        "isolated_vertex_count",
        "isolated_edge_count",
        "full_graph_exact_plus_1_multiplicity",
        "non_giant_exact_plus_1_multiplicity",
        "residual_giant_exact_plus_1_multiplicity",
        "giant_pendant_p2_plus_1_lower_bound",
        "q_0p683_selected_atom_fraction",
        "q_0p683_mean_n_times_ipr",
        "zero_eigenvalue_fraction",
    )
    metrics: dict[str, object] = {}
    for name in metric_names:
        mean, standard_error = mean_standard_error([float(row[name]) for row in rows])
        metrics[name] = {"mean": mean, "standard_error": standard_error}

    full = np.asarray([int(row["full_graph_exact_plus_1_multiplicity"]) for row in rows])
    non_giant = np.asarray([int(row["non_giant_exact_plus_1_multiplicity"]) for row in rows])
    residual = np.asarray([int(row["residual_giant_exact_plus_1_multiplicity"]) for row in rows])
    p2_lower_bound = np.asarray([int(row["giant_pendant_p2_plus_1_lower_bound"]) for row in rows])
    return {
        "number_of_graphs": len(rows),
        "displayed_tau": float(rows[0]["displayed_tau"]),
        "target_quantile": TARGET_QUANTILE,
        "metrics": metrics,
        "fraction_of_graphs_with_any_exact_plus_1_mode": float(np.mean(full > 0)),
        "fraction_of_graphs_with_non_giant_exact_plus_1_mode": float(np.mean(non_giant > 0)),
        "fraction_of_graphs_with_giant_component_exact_plus_1_mode": float(np.mean(residual > 0)),
        "fraction_of_full_plus_1_multiplicity_outside_giant": float(np.sum(non_giant) / np.sum(full)),
        "fraction_of_giant_plus_1_multiplicity_explained_by_repeated_p2_arms": (
            float(np.sum(np.minimum(p2_lower_bound, residual)) / np.sum(residual))
            if np.sum(residual) > 0
            else math.nan
        ),
        "correlation_zero_fraction_with_q_0p683_atom_fraction": correlation(
            rows, "zero_eigenvalue_fraction", "q_0p683_selected_atom_fraction"
        ),
        "correlation_q_0p683_atom_fraction_with_n_times_ipr": correlation(
            rows, "q_0p683_selected_atom_fraction", "q_0p683_mean_n_times_ipr"
        ),
        "interpretation_guardrail": (
            "Exact zero modes were removed before quantile assignment. Exact +1 multiplicity is "
            "basis-invariant, but IPRs of individual vectors in a degenerate +1 eigenspace are not."
        ),
    }


def default_paths(repo: Path, displayed_tau: float) -> tuple[Path, Path, Path]:
    tau = scan.tau_slug(displayed_tau)
    eigenvalue = (
        repo
        / "outputs/chung_lu/power_law/eigenvalues/resonance_zoom_q0p60_0p72_width_0p002"
        / f"n_5000_tau_{tau}_avgdeg_20p0/eigenvalue_resonance_by_graph.csv.gz"
    )
    eigenvector = (
        repo
        / "outputs/chung_lu/power_law/eigenvectors/percentile_scan_q0p5_1_width_0p002"
        / f"n_5000_tau_{tau}_avgdeg_20p0/eigenvector_percentile_scan_by_graph.csv.gz"
    )
    output = (
        repo
        / "outputs/chung_lu/power_law/audits/e1_resonance_connected_structure"
        / f"n_5000_tau_{tau}_avgdeg_20p0"
    )
    return eigenvalue, eigenvector, output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--displayed-tau", type=float, default=1.5)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=scan.NUM_GRAPHS)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    eigenvalue_path, eigenvector_path, default_output = default_paths(args.repo, args.displayed_tau)
    output_dir = args.output_dir or default_output
    eigenvalue_rows = quantile_row_by_graph(eigenvalue_path, TARGET_QUANTILE)
    eigenvector_rows = quantile_row_by_graph(eigenvector_path, TARGET_QUANTILE)
    graph_seeds, sequence_seeds = scan.graph_and_sequence_seeds(args.displayed_tau)

    rows: list[dict[str, object]] = []
    for graph_index in range(args.start, args.stop):
        rows.append(
            analyze_graph(
                graph_index=graph_index,
                graph_seed=graph_seeds[graph_index],
                sequence_seed=sequence_seeds[graph_index],
                displayed_tau=args.displayed_tau,
                eigenvalue_row=eigenvalue_rows[graph_index],
                eigenvector_row=eigenvector_rows[graph_index],
            )
        )
        if (graph_index - args.start + 1) % 25 == 0:
            print(f"finished graph {graph_index}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.start:04d}_{args.stop:04d}"
    write_csv(rows, output_dir / f"e1_resonance_structural_audit_{suffix}.csv")
    with (output_dir / f"e1_resonance_structural_audit_summary_{suffix}.json").open("w") as handle:
        json.dump(summarize(rows), handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
