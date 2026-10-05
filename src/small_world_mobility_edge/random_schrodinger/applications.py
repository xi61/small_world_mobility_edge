"""First tunneling-diagnostic applications to ER, PA, and matched Chung--Lu."""

from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/random-schrodinger-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/random-schrodinger-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from small_world_mobility_edge.chung_lu_baseline import (
    generate_chung_lu_graph_from_weights,
    generate_power_law_expected_weights,
)

from .common import (
    BFSDistanceCache,
    deterministic_mass_order,
    graph_from_edge_cache,
    graph_to_csr,
    hill_ccdf_exponent,
    mass_overlap,
    read_json,
    relative_residual,
    runtime_metadata,
    save_graph_cache,
    sha256_file,
    stable_json_hash,
    utc_now,
    write_csv,
    write_json,
)
from .tunneling import (
    approximate_state_pair_rows,
    effective_pair_from_centers,
    paper_global_approximate_states,
)


APPLICATION_VERSION = "ensemble_tunneling_application_v1"


@dataclass
class SpectralBundle:
    model: str
    graph: nx.Graph
    operator: sparse.csr_matrix
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    ranks: np.ndarray
    graph_id: str
    operator_id: str
    metadata: dict[str, object]


def _load_selected_cache(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        values = np.asarray(data["eigenvalues"], dtype=float)
        vectors = np.asarray(data["eigenvectors"], dtype=float)
        if "largest_algebraic_ranks" in data.files:
            ranks = np.asarray(data["largest_algebraic_ranks"], dtype=int)
        else:
            indices = np.asarray(data["global_indices"], dtype=int)
            ranks = vectors.shape[0] - indices
    order = np.lexsort((np.arange(len(ranks)), ranks))
    return values[order], vectors[:, order], ranks[order]


def load_er_bundle(input_dir: Path) -> SpectralBundle:
    input_dir = Path(input_dir)
    manifest = read_json(input_dir / "run_manifest.json")
    if manifest.get("status") != "COMPLETE" or not manifest.get("full_spectrum_coverage"):
        raise ValueError("ER benchmark cache is not a complete full-spectrum run")
    graph, graph_metadata = graph_from_edge_cache(input_dir / "graph_realization.npz")
    values, vectors, ranks = _load_selected_cache(input_dir / "selected_eigenvectors.npz")
    d = float(manifest["publication_setup"]["d"])
    operator = graph_to_csr(graph, nodelist=range(graph.number_of_nodes())) / math.sqrt(d)
    keep = ranks <= 51
    return SpectralBundle(
        model="erdos_renyi",
        graph=graph,
        operator=operator,
        eigenvalues=values[keep],
        eigenvectors=vectors[:, keep],
        ranks=ranks[keep],
        graph_id=str(manifest["graph_id"]),
        operator_id=str(manifest["operator_id"]),
        metadata={
            "source_dir": str(input_dir),
            "source_manifest_sha256": sha256_file(input_dir / "run_manifest.json"),
            "original_n": int(manifest["original_n"]),
            "analyzed_n": int(manifest["analyzed_n"]),
            "scaling": "A/sqrt(d)",
            "d": d,
            "graph_seed": int(manifest["implementation"]["graph_seed"]),
            "graph_cache": graph_metadata,
        },
    )


def load_pa_bundle(input_dir: Path) -> SpectralBundle:
    input_dir = Path(input_dir)
    source_manifest = read_json(input_dir / "run_manifest.json")
    if source_manifest.get("status") != "COMPLETE":
        raise ValueError("PA source cache is not complete")
    graph_path = input_dir / "tables" / "graph_realization.npz"
    graph, graph_metadata = graph_from_edge_cache(graph_path)
    values, vectors, ranks = _load_selected_cache(input_dir / "tables" / "selected_eigenvectors.npz")
    keep = ranks <= 51
    operator = graph_to_csr(graph, nodelist=range(graph.number_of_nodes()))
    graph_id = stable_json_hash(
        {
            "cache_sha256": sha256_file(graph_path),
            "mapping": graph_metadata["vertex_ordering"],
        }
    )
    operator_id = stable_json_hash(
        {"graph_id": graph_id, "operator": "unscaled adjacency A", "dimension": graph.number_of_nodes()}
    )
    return SpectralBundle(
        model="preferential_attachment",
        graph=graph,
        operator=operator,
        eigenvalues=values[keep],
        eigenvectors=vectors[:, keep],
        ranks=ranks[keep],
        graph_id=graph_id,
        operator_id=operator_id,
        metadata={
            "source_dir": str(input_dir),
            "source_manifest_sha256": sha256_file(input_dir / "run_manifest.json"),
            "original_n": graph.number_of_nodes(),
            "analyzed_n": graph.number_of_nodes(),
            "scaling": "unscaled adjacency A",
            "graph_seed": source_manifest.get("config", {}).get("graph_seed", source_manifest.get("graph_seed")),
            "source_config": source_manifest.get("config", source_manifest),
            "graph_cache": graph_metadata,
        },
    )


def build_matched_chung_lu_bundle(
    *,
    output_dir: Path,
    n: int,
    pa_degree_ccdf_alpha: float,
    target_mean_degree: float,
    graph_seed: int,
    sequence_seed: int,
    eigenpair_count: int = 51,
) -> SpectralBundle:
    """Generate the established simple Chung--Lu control matched to PA CCDF alpha.

    The repository's Chung--Lu tau is the density exponent, hence
    tau = PA degree-CCDF alpha + 1.
    """
    tau = float(pa_degree_ccdf_alpha + 1.0)
    weights, diagnostics = generate_power_law_expected_weights(
        n=n,
        tau=tau,
        target_mean_degree=target_mean_degree,
        w_min=1.0,
        w_max=None,
        cutoff_mode="structural",
        cutoff_coefficient=1.0,
        sequence_mode="sampled",
        sequence_seed=int(sequence_seed),
    )
    if not diagnostics.admissibility_holds:
        raise ValueError("matched Chung--Lu expected weights violate max(w)^2 <= sum(w)")
    original = generate_chung_lu_graph_from_weights(weights, seed=int(graph_seed))
    components = sorted(nx.connected_components(original), key=lambda part: (-len(part), min(part)))
    original_ids = np.asarray(sorted(components[0]), dtype=np.int64)
    induced = original.subgraph(original_ids.tolist()).copy()
    mapping = {int(vertex): index for index, vertex in enumerate(original_ids)}
    graph = nx.relabel_nodes(induced, mapping, copy=True)
    operator = graph_to_csr(graph, nodelist=range(graph.number_of_nodes()))
    k = min(int(eigenpair_count), graph.number_of_nodes() - 2)
    values, vectors = sparse_linalg.eigsh(operator, k=k, which="LA", tol=1e-10, maxiter=50 * graph.number_of_nodes())
    order = np.argsort(values)[::-1]
    values = np.asarray(values[order], dtype=float)
    vectors = np.asarray(vectors[:, order], dtype=float)
    ranks = np.arange(1, len(values) + 1, dtype=int)
    realized_degrees = np.asarray([graph.degree(vertex) for vertex in range(graph.number_of_nodes())], dtype=int)
    graph_metadata = {
        "model": "power_law_chung_lu",
        "n_requested": n,
        "analyzed_n": graph.number_of_nodes(),
        "component_rule": "largest component; ties by minimum original vertex ID",
        "component_count": len(components),
        "component_sizes_descending": [len(part) for part in components],
        "graph_seed": int(graph_seed),
        "sequence_seed": int(sequence_seed),
        "sequence_mode": "sampled",
        "cutoff_mode": "structural",
        "cutoff_coefficient": 1.0,
        "selfloops": False,
        "multiple_edges": False,
        "pa_degree_ccdf_alpha_target": pa_degree_ccdf_alpha,
        "chung_lu_density_tau": tau,
        "derivation": "weight CCDF exponent=tau-1, so tau=PA CCDF alpha+1",
        "target_mean_degree": target_mean_degree,
        "realized_mean_degree_original": 2.0 * original.number_of_edges() / original.number_of_nodes(),
        "realized_mean_degree_analyzed": float(np.mean(realized_degrees)),
        "realized_tail_audit": hill_ccdf_exponent(realized_degrees),
        "weight_diagnostics": asdict(diagnostics),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_path = output_dir / "graph_realization.npz"
    save_graph_cache(graph_path, graph, original_ids, graph_metadata)
    selected_path = output_dir / "selected_eigenvectors.npz"
    np.savez_compressed(
        selected_path,
        eigenvalues=values,
        eigenvectors=vectors,
        largest_algebraic_ranks=ranks,
        original_vertex_ids=original_ids,
    )
    graph_id = stable_json_hash(
        {
            "cache_sha256": sha256_file(graph_path),
            "mapping": "ascending_original_ids",
            "graph_seed": graph_seed,
            "sequence_seed": sequence_seed,
        }
    )
    operator_id = stable_json_hash(
        {"graph_id": graph_id, "operator": "unscaled adjacency A", "dimension": graph.number_of_nodes()}
    )
    return SpectralBundle(
        model="matched_chung_lu",
        graph=graph,
        operator=operator,
        eigenvalues=values,
        eigenvectors=vectors,
        ranks=ranks,
        graph_id=graph_id,
        operator_id=operator_id,
        metadata={
            **graph_metadata,
            "graph_cache_sha256": sha256_file(graph_path),
            "selected_eigenvectors_sha256": sha256_file(selected_path),
            "scaling": "unscaled adjacency A",
        },
    )


def _pair_rows(bundle: SpectralBundle) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    graph = bundle.graph
    cache = BFSDistanceCache(graph)
    rank_to_position = {int(rank): index for index, rank in enumerate(bundle.ranks)}
    rows: list[dict[str, object]] = []
    schur_rows: list[dict[str, object]] = []
    for left_rank in range(1, 51):
        right_rank = left_rank + 1
        if left_rank not in rank_to_position or right_rank not in rank_to_position:
            continue
        left_position = rank_to_position[left_rank]
        right_position = rank_to_position[right_rank]
        left_vector = bundle.eigenvectors[:, left_position]
        right_vector = bundle.eigenvectors[:, right_position]
        left_energy = float(bundle.eigenvalues[left_position])
        right_energy = float(bundle.eigenvalues[right_position])
        observed_gap = abs(left_energy - right_energy)
        left_order, _ = deterministic_mass_order(left_vector)
        right_order, _ = deterministic_mass_order(right_vector)
        left_center = int(left_order[0])
        right_center = int(right_order[0])
        center_distance = cache.distance(left_center, right_center)
        cross = cache.matrix(left_order[:10], right_order[:10])
        finite_cross = cross[np.isfinite(cross)]
        left_ipr = float(np.sum(np.abs(left_vector) ** 4))
        right_ipr = float(np.sum(np.abs(right_vector) ** 4))
        for radius in (1, 2, 3):
            summary, mediated = effective_pair_from_centers(
                graph,
                bundle.operator,
                (left_center, right_center),
                radius=radius,
                observed_gap=observed_gap,
                target_energy=0.5 * (left_energy + right_energy),
                eta_values=(1e-2, 1e-3) if 2 <= left_rank <= 5 and radius in (1, 2) else (),
            )
            poor_residual = bool(
                float(summary["left_residual_norm"]) > 0.25
                or float(summary["right_residual_norm"]) > 0.25
            )
            rows.append(
                {
                    "model": bundle.model,
                    "graph_id": bundle.graph_id,
                    "operator_id": bundle.operator_id,
                    "left_rank_largest_algebraic": left_rank,
                    "right_rank_largest_algebraic": right_rank,
                    "rank1_inclusive_top10_window": right_rank <= 10,
                    "nonperron_top10_window": left_rank >= 2 and right_rank <= 11,
                    "nonperron_top25_window": left_rank >= 2 and right_rank <= 26,
                    "nonperron_top50_window": left_rank >= 2 and right_rank <= 51,
                    "left_energy": left_energy,
                    "right_energy": right_energy,
                    "mean_energy": 0.5 * (left_energy + right_energy),
                    "observed_adjacent_full_operator_gap": observed_gap,
                    "observed_gap_is_not_uncoupled_detuning": True,
                    "left_center": left_center,
                    "right_center": right_center,
                    "same_dominant_center": left_center == right_center,
                    "dominant_center_distance": center_distance,
                    "top10_cross_distance_finite_mean": (
                        float(np.mean(finite_cross)) if len(finite_cross) else math.nan
                    ),
                    "top10_cross_distance_finite_minimum": (
                        float(np.min(finite_cross)) if len(finite_cross) else math.nan
                    ),
                    "top10_cross_distance_unreachable_fraction": float(
                        1.0 - len(finite_cross) / cross.size
                    ),
                    "sign_invariant_mass_overlap": mass_overlap(left_vector, right_vector),
                    "left_ipr": left_ipr,
                    "right_ipr": right_ipr,
                    "left_n_eff_times_ipr": graph.number_of_nodes() * left_ipr,
                    "right_n_eff_times_ipr": graph.number_of_nodes() * right_ipr,
                    "poor_local_state_residual_flag": poor_residual,
                    "interpretation_validity": (
                        "inconclusive_poor_local_reference_state"
                        if poor_residual
                        else summary["validity_flag"]
                    ),
                    **summary,
                }
            )
            for item in mediated:
                schur_rows.append(
                    {
                        "model": bundle.model,
                        "graph_id": bundle.graph_id,
                        "operator_id": bundle.operator_id,
                        "left_rank_largest_algebraic": left_rank,
                        "right_rank_largest_algebraic": right_rank,
                        "observed_adjacent_full_operator_gap": observed_gap,
                        "dominant_center_distance": center_distance,
                        **item,
                    }
                )
    return rows, schur_rows


def _plot_pair_diagnostics(output_dir: Path, rows: list[dict[str, object]], model: str) -> None:
    valid = [
        row
        for row in rows
        if row["valid"]
        and not row["poor_local_state_residual_flag"]
        and math.isfinite(float(row["dominant_center_distance"]))
    ]
    if not valid:
        return
    distance = np.asarray([row["dominant_center_distance"] for row in valid], dtype=float)
    coupling = np.asarray([row["absolute_direct_coupling"] for row in valid], dtype=float)
    detuning = np.asarray([row["uncoupled_detuning"] for row in valid], dtype=float)
    energy = np.asarray([row["mean_energy"] for row in valid], dtype=float)
    radius = np.asarray([row["radius"] for row in valid], dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.7))
    scatter = axes[0].scatter(distance, coupling, c=radius, cmap="viridis", s=25, alpha=0.75)
    axes[0].set_xlabel("dominant-center graph distance")
    axes[0].set_ylabel(r"direct projected $|t_{xy}|$ (operator units)")
    figure.colorbar(scatter, ax=axes[0], label="ball radius")
    positive = detuning > 0.0
    axes[1].scatter(energy[positive], coupling[positive] / detuning[positive], c=radius[positive], cmap="viridis", s=25, alpha=0.75)
    axes[1].set_xlabel("mean exact energy (operator units)")
    axes[1].set_ylabel(r"direct $|t_{xy}|$ / uncoupled detuning")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle(f"{model}: exploratory local-reference coupling (failed residuals excluded)")
    figure.tight_layout()
    figure.savefig(output_dir / "coupling_distance_energy.png", dpi=200)
    plt.close(figure)


def run_tunneling_application(
    *,
    case: str,
    input_dir: Path | None,
    output_dir: Path,
    project_root: Path,
    cl_n: int = 5000,
    cl_pa_alpha: float = 2.0,
    cl_target_mean_degree: float = 19.978,
    cl_graph_seed: int = 822301,
    cl_sequence_seed: int = 822302,
) -> Path:
    started = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    completed_manifest = read_json(manifest_path) if manifest_path.exists() else None
    manifest: dict[str, object] = {
        **runtime_metadata(project_root),
        "stage": "stage2c_tunneling_application",
        "stage_version": APPLICATION_VERSION,
        "case": case,
        "input_dir": None if input_dir is None else str(input_dir),
        "effective_coupling_definition": {
            "reference_states": "eigenmodes of induced graph balls, independently constructed from final exact eigenvectors after center selection",
            "boundary": "Dirichlet induced-ball",
            "mode": "nearest to adjacent-pair mean energy",
            "orthogonalization": "symmetric Lowdin via Gram inverse square root",
            "radii": [1, 2, 3],
            "direct_projection_is_basis_dependent": True,
            "schur_subset": "left ranks 2..5, radii 1..2, eta in {1e-2,1e-3}",
            "residual_inconclusive_threshold": 0.25,
        },
        "sw_status": "BLOCKED: repository audit found no explicit small-world generator; none was invented",
    }
    try:
        if case == "er":
            if input_dir is None:
                raise ValueError("ER application requires benchmark input_dir")
            bundle = load_er_bundle(input_dir)
        elif case == "pa":
            if input_dir is None:
                raise ValueError("PA application requires source input_dir")
            bundle = load_pa_bundle(input_dir)
        elif case == "cl":
            bundle = build_matched_chung_lu_bundle(
                output_dir=output_dir / "generated_cache",
                n=cl_n,
                pa_degree_ccdf_alpha=cl_pa_alpha,
                target_mean_degree=cl_target_mean_degree,
                graph_seed=cl_graph_seed,
                sequence_seed=cl_sequence_seed,
            )
        else:
            raise ValueError("case must be er, pa, or cl")
        if completed_manifest is not None and completed_manifest.get("status") == "COMPLETE":
            matching = (
                completed_manifest.get("stage_version") == APPLICATION_VERSION
                and completed_manifest.get("case") == case
                and completed_manifest.get("graph_id") == bundle.graph_id
                and completed_manifest.get("operator_id") == bundle.operator_id
            )
            if matching:
                return output_dir
            raise ValueError("completed application output has a different graph/operator identity")
        write_json(manifest_path, manifest)
        residual_rows: list[dict[str, object]] = []
        for position, rank in enumerate(bundle.ranks):
            absolute, relative = relative_residual(
                bundle.operator, bundle.eigenvalues[position], bundle.eigenvectors[:, position]
            )
            residual_rows.append(
                {
                    "largest_algebraic_rank_1based": int(rank),
                    "eigenvalue": float(bundle.eigenvalues[position]),
                    "absolute_residual_norm": absolute,
                    "relative_residual_norm": relative,
                }
            )
        if max(float(row["relative_residual_norm"]) for row in residual_rows) > 1e-7:
            raise AssertionError("source eigenpair residual exceeds tolerance")
        rows, schur_rows = _pair_rows(bundle)
        write_csv(output_dir / "adjacent_pair_effective_coupling.csv.gz", rows)
        write_csv(output_dir / "schur_complement_sensitivity.csv", schur_rows)
        write_csv(output_dir / "source_eigenpair_residuals.csv", residual_rows)
        paper_state_count = 0
        paper_pair_count = 0
        if case == "er":
            d = float(bundle.metadata["d"])
            states, center_sets = paper_global_approximate_states(
                bundle.operator,
                [bundle.graph.degree(vertex) for vertex in range(bundle.graph.number_of_nodes())],
                original_n=int(bundle.metadata["original_n"]),
                d=d,
                mu=0.02,
                kappa=0.1,
            )
            state_rows = [
                {
                    key: value
                    for key, value in state.items()
                    if key != "state"
                }
                for state in states
            ]
            approximate_pairs = approximate_state_pair_rows(states)
            write_csv(output_dir / "paper_approximate_states.csv", state_rows)
            write_csv(output_dir / "paper_approximate_energy_pairs.csv", approximate_pairs)
            write_json(
                output_dir / "paper_center_sets.json",
                {
                    key: value
                    for key, value in center_sets.items()
                    if key != "normalized_degrees"
                },
            )
            paper_state_count = len(states)
            paper_pair_count = len(approximate_pairs)
        _plot_pair_diagnostics(output_dir, rows, bundle.model)
        manifest.update(
            {
                "status": "COMPLETE",
                "finished_at_utc": utc_now(),
                "runtime_seconds": time.time() - started,
                "model": bundle.model,
                "graph_id": bundle.graph_id,
                "operator_id": bundle.operator_id,
                "model_metadata": bundle.metadata,
                "analyzed_n": bundle.graph.number_of_nodes(),
                "eigenpair_count": len(bundle.ranks),
                "pair_radius_row_count": len(rows),
                "schur_row_count": len(schur_rows),
                "paper_approximate_state_count": paper_state_count,
                "paper_approximate_pair_count": paper_pair_count,
                "maximum_source_relative_residual": max(
                    float(row["relative_residual_norm"]) for row in residual_rows
                ),
            }
        )
        write_json(manifest_path, manifest)
        return output_dir
    except Exception as error:
        if completed_manifest is not None and completed_manifest.get("status") == "COMPLETE":
            raise
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
