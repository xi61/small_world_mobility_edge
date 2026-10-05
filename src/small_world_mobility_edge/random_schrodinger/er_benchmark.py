"""Full-spectrum reproduction of the documented ER Figure 10 setup."""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/random-schrodinger-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/random-schrodinger-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import linalg

from .common import (
    graph_to_csr,
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


ER_BENCHMARK_VERSION = "er_figure10_full_spectrum_v1"


def _selected_indices(n: int, edge_count: int, interior_count: int) -> np.ndarray:
    selected: set[int] = set(range(min(edge_count, n)))
    selected.update(range(max(0, n - edge_count), n))
    for quantile in (0.25, 0.50, 0.75):
        center = int(round(quantile * (n - 1)))
        half = interior_count // 2
        selected.update(range(max(0, center - half), min(n, center + half + 1)))
    return np.asarray(sorted(selected), dtype=np.int64)


def _sample_indices(n: int) -> np.ndarray:
    fixed = {0, 1, n // 4, n // 2, 3 * n // 4, max(0, n - 2), n - 1}
    fixed.update(int(value) for value in np.linspace(0, n - 1, min(20, n), dtype=int))
    return np.asarray(sorted(fixed), dtype=int)


def _largest_component_identity_graph(
    graph: nx.Graph,
) -> tuple[nx.Graph, np.ndarray, list[int]]:
    components = sorted(nx.connected_components(graph), key=lambda part: (-len(part), min(part)))
    original_ids = np.asarray(sorted(components[0]), dtype=np.int64)
    induced = graph.subgraph(original_ids.tolist()).copy()
    mapping = {int(original): index for index, original in enumerate(original_ids)}
    analyzed = nx.relabel_nodes(induced, mapping, copy=True)
    return analyzed, original_ids, [len(part) for part in components]


def _plot_replication(
    path: Path,
    eigenvalues: np.ndarray,
    infinity_norms: np.ndarray,
    n_requested: int,
    d: float,
    n_eff: int,
) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 5.6))
    axis.scatter(eigenvalues[:-1], infinity_norms[:-1], s=7, alpha=0.58, color="black", linewidths=0)
    axis.scatter(
        [eigenvalues[-1]],
        [infinity_norms[-1]],
        s=55,
        marker="*",
        color="crimson",
        label="rank 1 (Perron)",
        zorder=3,
    )
    axis.set_xlabel(r"energy $\lambda$ of $A/\sqrt{d}$")
    axis.set_ylabel(r"eigenvector infinity norm $\|w\|_\infty$")
    axis.set_title(
        rf"ER $G(N,d/N)$ full spectrum: $N={n_requested}$, $d=0.6\log N={d:.3f}$, LCC $N_{{eff}}={n_eff}$"
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _plot_extension(
    path: Path,
    eigenvalues: np.ndarray,
    ipr: np.ndarray,
    n_eff: int,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    axes[0].scatter(eigenvalues, ipr, s=7, alpha=0.55, color="tab:blue", linewidths=0)
    axes[0].set_xlabel(r"energy $\lambda$ of $A/\sqrt{d}$")
    axes[0].set_ylabel("IPR")
    axes[1].scatter(eigenvalues, n_eff * ipr, s=7, alpha=0.55, color="tab:purple", linewidths=0)
    axes[1].set_xlabel(r"energy $\lambda$ of $A/\sqrt{d}$")
    axes[1].set_ylabel(r"$N_{eff}\,\mathrm{IPR}$")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle("Our localization extension (not part of published Figure 10)")
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def run_er_figure10_benchmark(
    *,
    n_requested: int,
    graph_seed: int,
    output_dir: Path,
    project_root: Path,
    run_label: str,
    edge_vector_count: int = 51,
    interior_vector_count: int = 25,
) -> Path:
    """Run one full-spectrum ER benchmark and preserve graph/mapping provenance."""
    started = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    d = float(0.6 * math.log(n_requested))
    p = float(d / n_requested)
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if existing.get("status") == "COMPLETE":
            publication = existing.get("publication_setup", {})
            implementation = existing.get("implementation", {})
            matching = (
                existing.get("stage_version") == ER_BENCHMARK_VERSION
                and int(publication.get("original_n", -1)) == int(n_requested)
                and int(implementation.get("graph_seed", -1)) == int(graph_seed)
                and existing.get("run_label") == run_label
            )
            if matching:
                return output_dir
            raise ValueError("completed ER output has a different cache identity or configuration")
    manifest: dict[str, object] = {
        **runtime_metadata(project_root),
        "stage": "stage2a_er_figure10_benchmark",
        "stage_version": ER_BENCHMARK_VERSION,
        "run_label": run_label,
        "pilot": run_label.lower() != "published_size",
        "publication_setup": {
            "ensemble": "G(N,d/N)",
            "original_n": n_requested,
            "d": d,
            "p": p,
            "operator": "A/sqrt(d)",
            "component": "largest_connected_component",
            "ordinate": "eigenvector_infinity_norm",
            "perron_point_included": True,
            "source": "Alt-Ducatez-Knowles (2021), Appendix A.1, Figure 10",
        },
        "undocumented_publication_choices": [
            "random seed",
            "number of attempted graph draws",
            "software and eigensolver",
        ],
        "implementation": {
            "generator": "networkx.fast_gnp_random_graph",
            "graph_seed": int(graph_seed),
            "full_dense_symmetric_eigendecomposition": True,
            "eigh_driver": "evd",
            "selected_eigenvectors_only_are_persisted": True,
            "edge_vector_count_per_side": edge_vector_count,
            "interior_vector_count_per_quantile": interior_vector_count,
        },
    }
    write_json(manifest_path, manifest)
    try:
        original = nx.fast_gnp_random_graph(n_requested, p, seed=int(graph_seed), directed=False)
        graph, original_ids, component_sizes = _largest_component_identity_graph(original)
        n_eff = graph.number_of_nodes()
        graph_metadata = {
            "model": "erdos_renyi",
            "ensemble": "G(N,d/N)",
            "original_n": n_requested,
            "analyzed_n": n_eff,
            "d_frozen_from_original_n": d,
            "p": p,
            "graph_seed": int(graph_seed),
            "component_rule": "largest component; ties by minimum original vertex ID",
            "original_component_count": len(component_sizes),
            "original_component_sizes_descending": component_sizes,
            "original_edge_count": original.number_of_edges(),
            "analyzed_edge_count": graph.number_of_edges(),
            "analyzed_realized_mean_degree": 2.0 * graph.number_of_edges() / n_eff,
            "simple_undirected": True,
            "self_loops": False,
            "multiple_edges": False,
            "analyzed_vertex_ordering": "ascending original vertex ID mapped to 0..N_eff-1",
        }
        graph_path = output_dir / "graph_realization.npz"
        save_graph_cache(graph_path, graph, original_ids, graph_metadata)
        operator_sparse = graph_to_csr(graph, nodelist=range(n_eff)) / math.sqrt(d)
        dense = operator_sparse.toarray()
        values, vectors = linalg.eigh(
            dense,
            overwrite_a=True,
            check_finite=False,
            driver="evd",
        )
        del dense
        order = np.argsort(values, kind="mergesort")
        values = np.asarray(values[order], dtype=float)
        vectors = np.asarray(vectors[:, order], dtype=float)
        if len(values) != n_eff or vectors.shape != (n_eff, n_eff):
            raise AssertionError("full spectral coverage was not obtained")
        infinity_norms = np.max(np.abs(vectors), axis=0)
        ipr = np.sum(np.square(vectors) ** 2, axis=0)
        selected = _selected_indices(n_eff, edge_vector_count, interior_vector_count)
        sampled = _sample_indices(n_eff)
        gram = vectors[:, sampled].T @ vectors[:, sampled]
        orthogonality_error = float(np.linalg.norm(gram - np.eye(len(sampled)), ord=2))
        residual_rows: list[dict[str, object]] = []
        for index in sampled:
            absolute, relative = relative_residual(operator_sparse, values[index], vectors[:, index])
            residual_rows.append(
                {
                    "ascending_index_0based": int(index),
                    "largest_algebraic_rank_1based": int(n_eff - index),
                    "eigenvalue": float(values[index]),
                    "absolute_residual_norm": absolute,
                    "relative_residual_norm": relative,
                }
            )
        maximum_relative_residual = max(float(row["relative_residual_norm"]) for row in residual_rows)
        norm_errors = np.abs(np.linalg.norm(vectors, axis=0) - 1.0)
        if maximum_relative_residual > 1e-8 or orthogonality_error > 1e-8 or float(np.max(norm_errors)) > 1e-8:
            raise AssertionError("eigenpair validation tolerance exceeded")
        spectrum_path = output_dir / "full_spectrum_diagnostics.npz"
        np.savez_compressed(
            spectrum_path,
            eigenvalues=values,
            infinity_norms=infinity_norms,
            ipr=ipr,
            n_eff_times_ipr=n_eff * ipr,
        )
        selected_path = output_dir / "selected_eigenvectors.npz"
        np.savez_compressed(
            selected_path,
            ascending_indices=selected,
            largest_algebraic_ranks=n_eff - selected,
            eigenvalues=values[selected],
            eigenvectors=vectors[:, selected],
            original_vertex_ids=original_ids,
        )
        table_rows = [
            {
                "ascending_index_0based": index,
                "largest_algebraic_rank_1based": n_eff - index,
                "eigenvalue": float(values[index]),
                "infinity_norm": float(infinity_norms[index]),
                "ipr": float(ipr[index]),
                "n_eff_times_ipr": float(n_eff * ipr[index]),
                "is_perron_rank_1": bool(index == n_eff - 1),
            }
            for index in range(n_eff)
        ]
        write_csv(output_dir / "full_spectrum_diagnostics.csv.gz", table_rows)
        write_csv(output_dir / "sampled_eigenpair_residuals.csv", residual_rows)
        write_json(output_dir / "graph_metadata.json", graph_metadata)
        _plot_replication(
            output_dir / "figure10_replication_energy_vs_infinity_norm.png",
            values,
            infinity_norms,
            n_requested,
            d,
            n_eff,
        )
        _plot_extension(
            output_dir / "our_extension_energy_vs_ipr.png",
            values,
            ipr,
            n_eff,
        )
        graph_id = stable_json_hash(
            {
                "graph_cache_sha256": sha256_file(graph_path),
                "original_n": n_requested,
                "d": d,
                "p": p,
                "seed": graph_seed,
                "mapping": "ascending_original_ids",
            }
        )
        operator_id = stable_json_hash(
            {
                "graph_id": graph_id,
                "operator": "A/sqrt(d)",
                "d": d,
                "dimension": n_eff,
            }
        )
        manifest.update(
            {
                "status": "COMPLETE",
                "finished_at_utc": utc_now(),
                "runtime_seconds": time.time() - started,
                "graph_id": graph_id,
                "operator_id": operator_id,
                "original_n": n_requested,
                "analyzed_n": n_eff,
                "full_spectrum_coverage": True,
                "eigenpair_count": len(values),
                "maximum_norm_error_all_eigenvectors": float(np.max(norm_errors)),
                "sampled_orthogonality_spectral_norm_error": orthogonality_error,
                "sampled_maximum_relative_residual_norm": maximum_relative_residual,
                "graph_cache_sha256": sha256_file(graph_path),
                "spectrum_cache_sha256": sha256_file(spectrum_path),
                "selected_eigenvector_cache_sha256": sha256_file(selected_path),
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
