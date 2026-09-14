from __future__ import annotations

import csv
import gzip
import json
import math
import os
import tempfile
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "small-world-mobility-edge-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import networkx as nx
import numpy as np

from . import chung_lu_eigenvector_percentile_scan as reference_scan
from .chung_lu_baseline import generate_chung_lu_graph_from_weights


N = reference_scan.N
NUM_GRAPHS = reference_scan.NUM_GRAPHS
DISPLAYED_TAU = 1.5
ATOM_ENERGIES = (-1.0, 1.0)
ATOM_ATOL = 1e-8
ZERO_ATOL = reference_scan.ZERO_EIGENVALUE_ATOL
PROJECTOR_TOP_R_VALUES = (1, 2, 5, 10, 20, 50, 100, 200)
PROJECTOR_MASS_LEVELS = (0.50, 0.80, 0.90, 0.95)
STRUCTURES = ("full_graph", "largest_connected_component", "largest_component_2_core")
OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "chung_lu"
    / "power_law"
    / "eigenvectors"
    / "e1_projector_component_scan"
    / "n_5000_tau_1p5_avgdeg_20p0"
)


def _empty_mode_data(dimension: int) -> dict[str, object]:
    return {
        "multiplicity": 0,
        "projector_diagonal": np.zeros(dimension, dtype=float),
        "basis_ipr": np.array([], dtype=float),
        "eigenvalue_min": math.nan,
        "eigenvalue_max": math.nan,
    }


def eigendecomposition_projector_data(
    graph: nx.Graph,
) -> tuple[np.ndarray, dict[float, dict[str, object]], list[int]]:
    """Diagonalize a graph and retain only basis-invariant E=+/-1 projector diagonals."""

    nodes = list(graph.nodes())
    dimension = len(nodes)
    if dimension == 0:
        return np.array([], dtype=float), {
            energy: _empty_mode_data(0) for energy in ATOM_ENERGIES
        }, nodes
    adjacency = nx.to_numpy_array(graph, nodelist=nodes, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(adjacency)
    modes: dict[float, dict[str, object]] = {}
    for energy in ATOM_ENERGIES:
        mask = np.isclose(eigenvalues, energy, atol=ATOM_ATOL, rtol=0.0)
        selected = eigenvectors[:, mask]
        if selected.shape[1] == 0:
            modes[energy] = _empty_mode_data(dimension)
            continue
        # diag(V V^T) is invariant under every within-eigenspace rotation V -> VQ.
        diagonal = np.sum(np.square(selected), axis=1)
        modes[energy] = {
            "multiplicity": int(selected.shape[1]),
            "projector_diagonal": diagonal,
            # Retained only as an explicitly basis-dependent comparison.
            "basis_ipr": np.sum(np.power(selected, 4), axis=0),
            "eigenvalue_min": float(np.min(eigenvalues[mask])),
            "eigenvalue_max": float(np.max(eigenvalues[mask])),
        }
    return eigenvalues, modes, nodes


def combine_component_spectra_and_projectors(
    graph: nx.Graph,
    component_results: list[tuple[np.ndarray, dict[float, dict[str, object]], list[int]]],
) -> tuple[np.ndarray, dict[float, dict[str, object]], list[int]]:
    """Assemble full-graph spectra/projector diagonals from connected components."""

    all_nodes = list(graph.nodes())
    node_position = {node: position for position, node in enumerate(all_nodes)}
    all_eigenvalues: list[np.ndarray] = []
    combined: dict[float, dict[str, object]] = {}
    for energy in ATOM_ENERGIES:
        combined[energy] = _empty_mode_data(len(all_nodes))
    for eigenvalues, modes, nodes in component_results:
        all_eigenvalues.append(eigenvalues)
        positions = np.asarray([node_position[node] for node in nodes], dtype=int)
        for energy in ATOM_ENERGIES:
            mode = modes[energy]
            combined_mode = combined[energy]
            diagonal = np.asarray(combined_mode["projector_diagonal"], dtype=float)
            diagonal[positions] = np.asarray(mode["projector_diagonal"], dtype=float)
            combined_mode["multiplicity"] = int(combined_mode["multiplicity"]) + int(
                mode["multiplicity"]
            )
            old_ipr = np.asarray(combined_mode["basis_ipr"], dtype=float)
            new_ipr = np.asarray(mode["basis_ipr"], dtype=float)
            combined_mode["basis_ipr"] = np.concatenate((old_ipr, new_ipr))
            if int(mode["multiplicity"]) > 0:
                values = [
                    value
                    for value in (
                        float(combined_mode["eigenvalue_min"]),
                        float(mode["eigenvalue_min"]),
                    )
                    if math.isfinite(value)
                ]
                combined_mode["eigenvalue_min"] = min(values)
                values = [
                    value
                    for value in (
                        float(combined_mode["eigenvalue_max"]),
                        float(mode["eigenvalue_max"]),
                    )
                    if math.isfinite(value)
                ]
                combined_mode["eigenvalue_max"] = max(values)
    spectrum = np.sort(np.concatenate(all_eigenvalues)) if all_eigenvalues else np.array([], dtype=float)
    return spectrum, combined, all_nodes


def continuous_nonperron_spectrum(eigenvalues: np.ndarray) -> np.ndarray:
    values = np.sort(np.asarray(eigenvalues, dtype=float))
    if len(values) > 0:
        values = values[:-1]
    zero = np.isclose(values, 0.0, atol=ZERO_ATOL, rtol=0.0)
    return values[~zero]


def atom_quantile_interval(eigenvalues: np.ndarray, energy: float) -> tuple[float, float, float]:
    values = continuous_nonperron_spectrum(eigenvalues)
    if len(values) == 0:
        return math.nan, math.nan, math.nan
    atom = np.isclose(values, energy, atol=ATOM_ATOL, rtol=0.0)
    if not np.any(atom):
        return math.nan, math.nan, math.nan
    indices = np.flatnonzero(atom)
    low = float(indices[0] / len(values))
    high = float((indices[-1] + 1) / len(values))
    return low, high, 0.5 * (low + high)


def projector_metrics(
    diagonal: np.ndarray,
    multiplicity: int,
    nodes: list[int],
    graph_for_degrees: nx.Graph,
    largest_component_nodes: set[int],
    basis_ipr: np.ndarray,
) -> dict[str, float | int]:
    diagonal = np.asarray(diagonal, dtype=float)
    basis_ipr = np.asarray(basis_ipr, dtype=float)
    result: dict[str, float | int] = {
        "multiplicity": int(multiplicity),
        "projector_trace": float(np.sum(diagonal)),
        "basis_mean_ipr": float(np.mean(basis_ipr)) if len(basis_ipr) else math.nan,
        "basis_mean_dimension_times_ipr": (
            float(len(nodes) * np.mean(basis_ipr)) if len(basis_ipr) else math.nan
        ),
    }
    if multiplicity == 0:
        result.update(
            {
                "projector_diagonal_ipr": math.nan,
                "projector_effective_support": math.nan,
                "projector_effective_support_fraction": math.nan,
                "projector_max_vertex_mass": math.nan,
                "projector_degree_0_mass_fraction": math.nan,
                "projector_degree_1_mass_fraction": math.nan,
                "projector_degree_2_mass_fraction": math.nan,
                "projector_outside_largest_component_mass_fraction": math.nan,
            }
        )
        for top_r in PROJECTOR_TOP_R_VALUES:
            result[f"projector_top_{top_r}_mass"] = math.nan
        for level in PROJECTOR_MASS_LEVELS:
            result[f"projector_r_{int(round(100 * level))}"] = math.nan
        return result

    if not math.isclose(float(np.sum(diagonal)), float(multiplicity), rel_tol=0.0, abs_tol=1e-6):
        raise AssertionError("spectral-projector trace does not match eigenvalue multiplicity")
    probability = diagonal / float(multiplicity)
    probability = probability / float(np.sum(probability))
    sorted_probability = np.sort(probability)[::-1]
    cumulative = np.cumsum(sorted_probability)
    diagonal_ipr = float(np.sum(np.square(probability)))
    degrees = np.asarray([graph_for_degrees.degree(node) for node in nodes], dtype=int)
    outside = np.asarray([node not in largest_component_nodes for node in nodes], dtype=bool)
    result.update(
        {
            "projector_diagonal_ipr": diagonal_ipr,
            "projector_effective_support": 1.0 / max(diagonal_ipr, 1e-300),
            "projector_effective_support_fraction": 1.0 / max(diagonal_ipr, 1e-300) / len(nodes),
            "projector_max_vertex_mass": float(sorted_probability[0]),
            "projector_degree_0_mass_fraction": float(np.sum(probability[degrees == 0])),
            "projector_degree_1_mass_fraction": float(np.sum(probability[degrees == 1])),
            "projector_degree_2_mass_fraction": float(np.sum(probability[degrees == 2])),
            "projector_outside_largest_component_mass_fraction": float(np.sum(probability[outside])),
        }
    )
    for top_r in PROJECTOR_TOP_R_VALUES:
        result[f"projector_top_{top_r}_mass"] = float(cumulative[min(top_r, len(nodes)) - 1])
    for level in PROJECTOR_MASS_LEVELS:
        result[f"projector_r_{int(round(100 * level))}"] = int(
            np.searchsorted(cumulative, level - 1e-12, side="left") + 1
        )
    return result


def pendant_length_two_atom_lower_bound(graph: nx.Graph) -> int:
    count = 0
    degrees = dict(graph.degree())
    for root in graph:
        arms = 0
        for middle in graph.neighbors(root):
            if degrees[middle] != 2:
                continue
            neighbors = list(graph.neighbors(middle))
            leaf = neighbors[0] if neighbors[1] == root else neighbors[1]
            if degrees[leaf] == 1:
                arms += 1
        count += max(arms - 1, 0)
    return count


def structure_rows(
    graph_index: int,
    graph_seed: int,
    sequence_seed: int,
) -> list[dict[str, object]]:
    weights, diagnostics = reference_scan.generate_legacy_expected_weights(
        displayed_tau=DISPLAYED_TAU,
        sequence_seed=sequence_seed,
    )
    if not diagnostics.admissibility_holds:
        raise ValueError("expected-weight sequence violates Chung-Lu admissibility")
    graph = generate_chung_lu_graph_from_weights(weights, seed=graph_seed)
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    largest_nodes = set(components[0])
    largest_graph = graph.subgraph(largest_nodes).copy()
    core_graph = nx.k_core(largest_graph, k=2)

    largest_result = eigendecomposition_projector_data(largest_graph)
    component_results = [largest_result]
    isolated_edge_count = 0
    for nodes in components[1:]:
        component = graph.subgraph(nodes).copy()
        if component.number_of_nodes() == 1:
            eigenvalues = np.array([0.0])
            modes = {energy: _empty_mode_data(1) for energy in ATOM_ENERGIES}
            component_results.append((eigenvalues, modes, list(component.nodes())))
        else:
            component_results.append(eigendecomposition_projector_data(component))
        if component.number_of_nodes() == 2 and component.number_of_edges() == 1:
            isolated_edge_count += 1
    full_result = combine_component_spectra_and_projectors(graph, component_results)
    core_result = eigendecomposition_projector_data(core_graph)
    results = {
        "full_graph": (graph, full_result),
        "largest_connected_component": (largest_graph, largest_result),
        "largest_component_2_core": (core_graph, core_result),
    }
    rows: list[dict[str, object]] = []
    for structure in STRUCTURES:
        structure_graph, (eigenvalues, modes, nodes) = results[structure]
        row: dict[str, object] = {
            "graph_index": graph_index,
            "graph_seed": graph_seed,
            "sequence_seed": sequence_seed,
            "displayed_tau": DISPLAYED_TAU,
            "structure": structure,
            "dimension": structure_graph.number_of_nodes(),
            "number_of_edges": structure_graph.number_of_edges(),
            "number_of_components_in_full_graph": len(components),
            "largest_component_size": len(largest_nodes),
            "largest_component_2_core_size": core_graph.number_of_nodes(),
            "isolated_vertex_count": sum(len(nodes_) == 1 for nodes_ in components[1:]),
            "isolated_edge_count": isolated_edge_count,
            "largest_component_pendant_length_two_atom_lower_bound": (
                pendant_length_two_atom_lower_bound(largest_graph)
            ),
            "realized_mean_degree_full_graph": float(
                np.mean([degree for _, degree in graph.degree()])
            ),
        }
        for energy in ATOM_ENERGIES:
            slug = "minus_1" if energy < 0 else "plus_1"
            mode = modes[energy]
            metrics = projector_metrics(
                diagonal=np.asarray(mode["projector_diagonal"], dtype=float),
                multiplicity=int(mode["multiplicity"]),
                nodes=nodes,
                graph_for_degrees=graph,
                largest_component_nodes=largest_nodes,
                basis_ipr=np.asarray(mode["basis_ipr"], dtype=float),
            )
            q_low, q_high, q_mid = atom_quantile_interval(eigenvalues, energy)
            row[f"{slug}_eigenvalue_min"] = mode["eigenvalue_min"]
            row[f"{slug}_eigenvalue_max"] = mode["eigenvalue_max"]
            row[f"{slug}_continuous_nonperron_quantile_low"] = q_low
            row[f"{slug}_continuous_nonperron_quantile_high"] = q_high
            row[f"{slug}_continuous_nonperron_quantile_mid"] = q_mid
            for key, value in metrics.items():
                row[f"{slug}_{key}"] = value
        rows.append(row)
    return rows


def fieldnames() -> list[str]:
    structural = [
        "graph_index", "graph_seed", "sequence_seed", "displayed_tau", "structure",
        "dimension", "number_of_edges", "number_of_components_in_full_graph",
        "largest_component_size", "largest_component_2_core_size", "isolated_vertex_count",
        "isolated_edge_count", "largest_component_pendant_length_two_atom_lower_bound",
        "realized_mean_degree_full_graph",
    ]
    metrics = [
        "eigenvalue_min", "eigenvalue_max", "continuous_nonperron_quantile_low",
        "continuous_nonperron_quantile_high", "continuous_nonperron_quantile_mid",
        "multiplicity", "projector_trace", "basis_mean_ipr", "basis_mean_dimension_times_ipr",
        "projector_diagonal_ipr", "projector_effective_support",
        "projector_effective_support_fraction", "projector_max_vertex_mass",
        "projector_degree_0_mass_fraction", "projector_degree_1_mass_fraction",
        "projector_degree_2_mass_fraction", "projector_outside_largest_component_mass_fraction",
    ]
    metrics.extend(f"projector_top_{top_r}_mass" for top_r in PROJECTOR_TOP_R_VALUES)
    metrics.extend(
        f"projector_r_{int(round(100 * level))}" for level in PROJECTOR_MASS_LEVELS
    )
    result = list(structural)
    for slug in ("minus_1", "plus_1"):
        result.extend(f"{slug}_{metric}" for metric in metrics)
    return result


def write_csv(rows: Iterable[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener: Callable[..., object] = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames())
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames()})


def read_csv_paths(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        opener: Callable[..., object] = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", newline="") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    return rows


def _mean_se(values: Iterable[float]) -> tuple[float, float, int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return math.nan, math.nan, 0
    mean = float(np.mean(array))
    se = float(np.std(array, ddof=1) / np.sqrt(len(array))) if len(array) > 1 else math.nan
    return mean, se, int(len(array))


def aggregate_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    summary_metrics = [
        "dimension", "isolated_edge_count", "largest_component_pendant_length_two_atom_lower_bound",
    ]
    for slug in ("minus_1", "plus_1"):
        summary_metrics.extend(
            [
                f"{slug}_multiplicity", f"{slug}_continuous_nonperron_quantile_mid",
                f"{slug}_projector_effective_support",
                f"{slug}_projector_effective_support_fraction",
                f"{slug}_projector_degree_1_mass_fraction",
                f"{slug}_projector_degree_2_mass_fraction",
                f"{slug}_projector_outside_largest_component_mass_fraction",
                f"{slug}_projector_top_2_mass", f"{slug}_projector_top_10_mass",
                f"{slug}_projector_r_90", f"{slug}_basis_mean_dimension_times_ipr",
            ]
        )
    for structure in STRUCTURES:
        selected = [row for row in rows if row["structure"] == structure]
        result: dict[str, object] = {
            "structure": structure,
            "number_of_graphs": len(selected),
        }
        for metric in summary_metrics:
            mean, se, count = _mean_se(float(row[metric]) for row in selected)
            result[f"mean_{metric}"] = mean
            result[f"se_{metric}"] = se
            result[f"graphs_with_{metric}"] = count
        for slug in ("minus_1", "plus_1"):
            multiplicity = np.asarray([float(row[f"{slug}_multiplicity"]) for row in selected])
            result[f"fraction_graphs_with_{slug}"] = float(np.mean(multiplicity > 0))
            result[f"total_{slug}_multiplicity"] = int(np.sum(multiplicity))
        output.append(result)
    return output


def paired_reduction_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    by_graph: dict[int, dict[str, dict[str, str]]] = {}
    for row in rows:
        by_graph.setdefault(int(row["graph_index"]), {})[row["structure"]] = row
    comparisons = (
        ("full_to_largest", "full_graph", "largest_connected_component"),
        ("largest_to_2_core", "largest_connected_component", "largest_component_2_core"),
        ("full_to_2_core", "full_graph", "largest_component_2_core"),
    )
    output: list[dict[str, object]] = []
    for label, before, after in comparisons:
        result: dict[str, object] = {"comparison": label, "number_of_graphs": len(by_graph)}
        for slug in ("minus_1", "plus_1"):
            differences = np.asarray(
                [
                    float(structures[before][f"{slug}_multiplicity"])
                    - float(structures[after][f"{slug}_multiplicity"])
                    for structures in by_graph.values()
                ]
            )
            before_values = np.asarray(
                [float(structures[before][f"{slug}_multiplicity"]) for structures in by_graph.values()]
            )
            mean, se, _ = _mean_se(differences)
            result[f"mean_removed_{slug}_multiplicity"] = mean
            result[f"se_removed_{slug}_multiplicity"] = se
            result[f"fraction_of_{slug}_multiplicity_removed"] = float(
                np.sum(differences) / np.sum(before_values)
            ) if np.sum(before_values) > 0 else math.nan
            result[f"fraction_graphs_with_reduction_{slug}"] = float(np.mean(differences > 0))
        output.append(result)
    return output


def write_dict_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_summary(summary: list[dict[str, object]], reductions: list[dict[str, object]]):
    plt = _pyplot()
    labels = ["Full graph", "Largest component", "2-core"]
    x = np.arange(len(labels))
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    for offset, (slug, name, color) in enumerate(
        (("minus_1", "E=-1", "steelblue"), ("plus_1", "E=+1", "darkorange"))
    ):
        means = [float(row[f"mean_{slug}_multiplicity"]) for row in summary]
        errors = [float(row[f"se_{slug}_multiplicity"]) for row in summary]
        axes[0].bar(x + (offset - 0.5) * 0.34, means, width=0.34, yerr=errors, label=name, color=color)
    axes[0].set_xticks(x, labels, rotation=15, ha="right")
    axes[0].set_ylabel("mean exact multiplicity")
    axes[0].set_title("Exact E=+/-1 atoms")
    axes[0].legend(frameon=False)

    for slug, name, color in (("minus_1", "E=-1", "steelblue"), ("plus_1", "E=+1", "darkorange")):
        values = [float(row[f"mean_{slug}_projector_degree_1_mass_fraction"]) for row in summary]
        axes[1].plot(x, values, marker="o", label=name, color=color)
    axes[1].set_xticks(x, labels, rotation=15, ha="right")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_ylabel("projector mass fraction")
    axes[1].set_title("Mass on degree-1 vertices")

    comparison_labels = ["Full -> LCC", "LCC -> 2-core", "Full -> 2-core"]
    for offset, (slug, name, color) in enumerate(
        (("minus_1", "E=-1", "steelblue"), ("plus_1", "E=+1", "darkorange"))
    ):
        values = [float(row[f"fraction_of_{slug}_multiplicity_removed"]) for row in reductions]
        axes[2].bar(x + (offset - 0.5) * 0.34, values, width=0.34, label=name, color=color)
    axes[2].set_xticks(x, comparison_labels, rotation=15, ha="right")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_ylabel("fraction of multiplicity removed")
    axes[2].set_title("Paired structural removal")
    for axis in axes:
        axis.grid(alpha=0.2, axis="y")
    figure.suptitle(
        "Chung-Lu E=+/-1 projector scan: full graph -> largest component -> 2-core, "
        f"N={N}, displayed tau={DISPLAYED_TAU:g}, G={NUM_GRAPHS}"
    )
    figure.tight_layout()
    return figure


def code_version() -> str | None:
    repo = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def run_manifest(num_partial_files: int) -> dict[str, object]:
    specification = reference_scan.tau_specification(DISPLAYED_TAU)
    return {
        "model": "power_law_chung_lu",
        "diagnostic": "same_seed_basis_invariant_e1_projector_full_lcc_2core_scan",
        "output_is_additive": True,
        "n": N,
        "num_graphs": NUM_GRAPHS,
        "displayed_tau": DISPLAYED_TAU,
        "tau_convention": specification.tau_convention,
        "effective_density_exponent_passed_to_generator": specification.effective_density_exponent,
        "target_mean_degree": reference_scan.TARGET_MEAN_DEGREE,
        "graph_seed_base": specification.graph_seed_base,
        "sequence_seed_base": specification.sequence_seed_base,
        "same_underlying_graph_seeds_as_q0p683_scan": True,
        "structures_in_order": list(STRUCTURES),
        "atom_energies": list(ATOM_ENERGIES),
        "atom_atol": ATOM_ATOL,
        "zero_atol_for_reported_continuous_quantiles": ZERO_ATOL,
        "projector_definition": "P_E=sum_{|lambda-E|<=atol} v v^T",
        "projector_diagonal_is_basis_invariant": True,
        "ordinary_basis_ipr_is_secondary_and_basis_dependent": True,
        "full_projector_method": "direct sum of connected-component projectors",
        "largest_component_2_core_definition": "networkx k_core(largest_component, k=2)",
        "projector_top_r_values": list(PROJECTOR_TOP_R_VALUES),
        "projector_mass_levels": list(PROJECTOR_MASS_LEVELS),
        "num_partial_files": num_partial_files,
        "code_version": code_version(),
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def aggregate_parameter() -> Path:
    partial_paths = sorted((OUTPUT_DIR / "partials").glob("e1_projector_graphs_*.csv.gz"))
    if not partial_paths:
        raise FileNotFoundError(f"missing partials in {OUTPUT_DIR / 'partials'}")
    rows = read_csv_paths(partial_paths)
    graph_ids = sorted({int(row["graph_index"]) for row in rows})
    if graph_ids != list(range(NUM_GRAPHS)):
        missing = sorted(set(range(NUM_GRAPHS)).difference(graph_ids))
        raise ValueError(f"missing graph indices: {missing[:20]}")
    if len(rows) != NUM_GRAPHS * len(STRUCTURES):
        raise ValueError("wrong number of graph/structure rows")
    summary = aggregate_rows(rows)
    reductions = paired_reduction_rows(rows)
    write_csv(rows, OUTPUT_DIR / "e1_projector_component_scan_by_graph.csv.gz")
    write_dict_csv(summary, OUTPUT_DIR / "e1_projector_component_scan_summary.csv")
    write_dict_csv(reductions, OUTPUT_DIR / "e1_projector_paired_reductions.csv")
    with (OUTPUT_DIR / "run_manifest.json").open("w") as handle:
        json.dump(run_manifest(len(partial_paths)), handle, indent=2, sort_keys=True)
    figure = plot_summary(summary, reductions)
    figure.savefig(OUTPUT_DIR / "e1_projector_component_scan_summary.png", dpi=200, bbox_inches="tight")
    _pyplot().close(figure)
    return OUTPUT_DIR
