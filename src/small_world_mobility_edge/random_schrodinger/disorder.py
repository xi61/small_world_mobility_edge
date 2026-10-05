"""Paired diagonal-disorder sweeps on cached preferential-attachment graphs."""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/random-schrodinger-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/random-schrodinger-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import linalg, sparse
from scipy.sparse import linalg as sparse_linalg

from .common import (
    BFSDistanceCache,
    deterministic_mass_order,
    graph_from_edge_cache,
    graph_to_csr,
    pair_distance_rows,
    read_json,
    relative_residual,
    runtime_metadata,
    sha256_file,
    stable_json_hash,
    utc_now,
    write_csv,
    write_json,
)
from .tunneling import effective_pair_from_centers


DISORDER_VERSION = "pa_diagonal_uniform_paired_v1"
AMPLITUDE_RATIOS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
S_REF = math.sqrt(20.0)


def _slug(value: float) -> str:
    return ("%g" % value).replace(".", "p")


def _gaps_and_ratios(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    gaps = np.diff(values)
    ratios = np.full(len(values), np.nan, dtype=float)
    if len(values) >= 3:
        left = gaps[:-1]
        right = gaps[1:]
        denominator = np.maximum(left, right)
        valid = denominator > 0.0
        interior = ratios[1:-1]
        interior[valid] = np.minimum(left[valid], right[valid]) / denominator[valid]
    return gaps, ratios


def _selection_indices(n: int, width: int = 51) -> dict[str, np.ndarray]:
    result = {
        "lower_edge": np.arange(0, min(width, n), dtype=int),
        "upper_edge": np.arange(max(0, n - width), n, dtype=int),
    }
    for label, quantile in (("interior_q25", 0.25), ("interior_q50", 0.50), ("interior_q75", 0.75)):
        center = int(round((n - 1) * quantile))
        half = width // 2
        result[label] = np.arange(max(0, center - half), min(n, center + half + 1), dtype=int)
    return result


def _load_a0_from_source(
    source_dir: Path,
    operator,
    width: int = 51,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_dir = Path(source_dir)
    with np.load(source_dir / "spectrum" / "single_graph_spectral_arrays.npz", allow_pickle=False) as data:
        all_values = np.asarray(data["eigenvalues"], dtype=float)
        all_ipr = np.asarray(data["ipr"], dtype=float)
        historical_bulk50_indices = (
            np.asarray(data["bulk50_indices"], dtype=int)
            if "bulk50_indices" in data.files
            else _selection_indices(len(all_values))["interior_q50"]
        )
    with np.load(source_dir / "tables" / "selected_eigenvectors.npz", allow_pickle=False) as data:
        source_indices = np.asarray(data["global_indices"], dtype=int)
        source_vectors = np.asarray(data["eigenvectors"], dtype=float)
    n = len(all_values)
    lower_count = min(width, n - 2)
    lower_values, lower_vectors = sparse_linalg.eigsh(
        operator,
        k=lower_count,
        which="SA",
        tol=1e-10,
        maxiter=50 * n,
    )
    lower_order = np.argsort(lower_values)
    lower_values = np.asarray(lower_values[lower_order], dtype=float)
    lower_vectors = np.asarray(lower_vectors[:, lower_order], dtype=float)
    lower_indices = np.arange(lower_count, dtype=int)
    combined_indices = np.concatenate([source_indices, lower_indices])
    combined_vectors = np.column_stack([source_vectors, lower_vectors])
    unique_positions: dict[int, int] = {}
    for position, index in enumerate(combined_indices):
        unique_positions.setdefault(int(index), position)
    selected_indices = np.asarray(sorted(unique_positions), dtype=int)
    selected_vectors = np.column_stack(
        [combined_vectors[:, unique_positions[int(index)]] for index in selected_indices]
    )
    selected_values = all_values[selected_indices]
    for position, index in enumerate(lower_indices):
        if abs(float(lower_values[position] - all_values[index])) > 1e-7:
            raise AssertionError("a=0 lower sparse eigenvalue does not reproduce cached dense spectrum")
    return (
        all_values,
        all_ipr,
        selected_indices,
        selected_values,
        selected_vectors,
        historical_bulk50_indices,
    )


def _selected_diagnostics(
    *,
    graph: nx.Graph,
    cache: BFSDistanceCache,
    values: np.ndarray,
    vectors: np.ndarray,
    indices: np.ndarray,
    potential: np.ndarray,
    ratio: float,
    amplitude: float,
) -> list[dict[str, object]]:
    n = graph.number_of_nodes()
    degrees = np.asarray([graph.degree(vertex) for vertex in range(n)], dtype=int)
    result: list[dict[str, object]] = []
    for position, index in enumerate(indices):
        vector = vectors[:, position]
        order, masses = deterministic_mass_order(vector)
        selected = order[:10]
        matrix = cache.matrix(selected)
        _, distance_summary = pair_distance_rows(selected, masses[selected], matrix)
        if index >= n - 51:
            region = "upper_edge"
            rank = n - int(index)
        elif index < 51:
            region = "lower_edge"
            rank = int(index) + 1
        else:
            region = "interior"
            rank = math.nan
        center = int(selected[0])
        result.append(
            {
                "a_over_s_ref": ratio,
                "a": amplitude,
                "potential_standard_deviation": amplitude / math.sqrt(3.0),
                "ascending_index_0based": int(index),
                "region": region,
                "edge_rank_1based": rank,
                "eigenvalue": float(values[position]),
                "ipr": float(np.sum(np.abs(vector) ** 4)),
                "n_eff_times_ipr": float(n * np.sum(np.abs(vector) ** 4)),
                "infinity_norm": float(np.max(np.abs(vector))),
                "top10_mass": float(np.sum(masses[selected])),
                "dominant_center": center,
                "dominant_center_degree": int(degrees[center]),
                "dominant_center_standardized_degree": float(
                    (degrees[center] - np.mean(degrees)) / max(np.std(degrees), 1e-15)
                ),
                "dominant_center_xi": float(potential[center]),
                "dominant_center_diagonal_potential": float(amplitude * potential[center]),
                "top10_mean_degree": float(np.mean(degrees[selected])),
                "top10_mean_xi": float(np.mean(potential[selected])),
                "top10_maximum_xi": float(np.max(potential[selected])),
                **distance_summary,
            }
        )
    return result


def _coupling_rows(
    *,
    graph: nx.Graph,
    operator,
    upper_values: np.ndarray,
    upper_vectors: np.ndarray,
    ratio: float,
    amplitude: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    order = np.argsort(upper_values)[::-1]
    values = np.asarray(upper_values[order], dtype=float)
    vectors = np.asarray(upper_vectors[:, order], dtype=float)
    rows: list[dict[str, object]] = []
    schur_rows: list[dict[str, object]] = []
    maximum_pair = min(10, len(values) - 1)
    for left_position in range(maximum_pair):
        left_rank = left_position + 1
        right_rank = left_rank + 1
        left_vector = vectors[:, left_position]
        right_vector = vectors[:, left_position + 1]
        left_center = int(deterministic_mass_order(left_vector)[0][0])
        right_center = int(deterministic_mass_order(right_vector)[0][0])
        observed_gap = abs(float(values[left_position] - values[left_position + 1]))
        for radius in (1, 2):
            use_schur = ratio in (0.0, 1.0, 8.0) and left_rank in (2, 3)
            summary, mediated = effective_pair_from_centers(
                graph,
                operator,
                (left_center, right_center),
                radius=radius,
                observed_gap=observed_gap,
                target_energy=0.5 * float(values[left_position] + values[left_position + 1]),
                eta_values=(1e-2, 1e-3) if use_schur else (),
            )
            rows.append(
                {
                    "a_over_s_ref": ratio,
                    "a": amplitude,
                    "left_rank_largest_algebraic": left_rank,
                    "right_rank_largest_algebraic": right_rank,
                    "includes_rank1_shifted_positive_top_mode": left_rank == 1,
                    "baseline_compatible_nonperron_comparison": left_rank >= 2,
                    "left_energy": float(values[left_position]),
                    "right_energy": float(values[left_position + 1]),
                    "observed_adjacent_full_operator_gap": observed_gap,
                    "observed_gap_is_not_uncoupled_detuning": True,
                    "left_exact_ipr": float(np.sum(np.abs(left_vector) ** 4)),
                    "right_exact_ipr": float(np.sum(np.abs(right_vector) ** 4)),
                    "exact_mass_overlap": float(
                        np.sum(
                            np.minimum(
                                np.square(np.abs(left_vector)),
                                np.square(np.abs(right_vector)),
                            )
                        )
                    ),
                    **summary,
                }
            )
            for item in mediated:
                schur_rows.append(
                    {
                        "a_over_s_ref": ratio,
                        "a": amplitude,
                        "left_rank_largest_algebraic": left_rank,
                        "right_rank_largest_algebraic": right_rank,
                        **item,
                    }
                )
    return rows, schur_rows


def _write_ratio_outputs(
    output_dir: Path,
    values: np.ndarray,
    ipr: np.ndarray,
    infinity_norms: np.ndarray,
    selected_indices: np.ndarray,
    selected_values: np.ndarray,
    selected_vectors: np.ndarray,
    selected_rows: list[dict[str, object]],
    coupling_rows: list[dict[str, object]],
    schur_rows: list[dict[str, object]],
) -> None:
    gaps, ratios = _gaps_and_ratios(values)
    np.savez_compressed(
        output_dir / "spectrum_diagnostics.npz",
        eigenvalues=values,
        ipr=ipr,
        n_eff_times_ipr=len(values) * ipr,
        infinity_norms=infinity_norms,
        adjacent_raw_gaps=gaps,
        adjacent_gap_ratios=ratios,
    )
    np.savez_compressed(
        output_dir / "selected_eigenvectors.npz",
        ascending_indices=selected_indices,
        eigenvalues=selected_values,
        eigenvectors=selected_vectors,
    )
    write_csv(output_dir / "selected_eigenvector_diagnostics.csv.gz", selected_rows)
    write_csv(output_dir / "upper_adjacent_effective_coupling.csv", coupling_rows)
    write_csv(output_dir / "schur_complement_sensitivity.csv", schur_rows)


def _plot_sweep(output_dir: Path, summaries: list[dict[str, object]]) -> None:
    ratios = np.asarray([row["a_over_s_ref"] for row in summaries], dtype=float)
    upper = np.asarray([row["upper_edge_mean_ipr"] for row in summaries], dtype=float)
    lower = np.asarray([row["lower_edge_mean_ipr"] for row in summaries], dtype=float)
    interior = np.asarray([row["interior_mean_ipr"] for row in summaries], dtype=float)
    figure, axis = plt.subplots(figsize=(7.4, 5.0))
    axis.plot(ratios, upper, "o-", label="upper edge, ranks 1–10")
    axis.plot(ratios, lower, "s-", label="lower edge, ranks 1–10")
    axis.plot(ratios, interior, "^-", label="interior q50 window")
    axis.set_xscale("symlog", linthresh=0.1)
    axis.set_yscale("log")
    axis.set_xlabel(r"$a/s_{ref}$")
    axis.set_ylabel("mean IPR")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    axis.set_title("Paired diagonal-disorder sweep (rank windows, not mode tracking)")
    figure.tight_layout()
    figure.savefig(output_dir / "paired_disorder_ipr_summary.png", dpi=200)
    plt.close(figure)


def run_pa_disorder_sweep(
    *,
    source_dir: Path,
    output_dir: Path,
    project_root: Path,
    potential_seed: int,
    amplitude_ratios: Sequence[float] = AMPLITUDE_RATIOS,
) -> Path:
    started = time.time()
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = read_json(source_dir / "run_manifest.json")
    if source_manifest.get("status") != "COMPLETE":
        raise ValueError("source PA run is not complete")
    graph_path = source_dir / "tables" / "graph_realization.npz"
    graph, graph_metadata = graph_from_edge_cache(graph_path)
    n = graph.number_of_nodes()
    adjacency = graph_to_csr(graph, nodelist=range(n))
    degrees = np.asarray([graph.degree(vertex) for vertex in range(n)], dtype=float)
    realized_s_b = float(math.sqrt(np.mean(degrees)))
    graph_id = stable_json_hash(
        {"graph_cache_sha256": sha256_file(graph_path), "mapping": graph_metadata["vertex_ordering"]}
    )
    rng = np.random.default_rng(int(potential_seed))
    xi = rng.uniform(-1.0, 1.0, size=n)
    potential_id = stable_json_hash(
        {
            "graph_id": graph_id,
            "potential_seed": int(potential_seed),
            "rng": "numpy Generator(PCG64).uniform(-1,1), half-open [-1,1)",
            "xi_sha256": stable_json_hash({"xi": xi.tolist()}),
        }
    )
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if existing.get("status") == "COMPLETE":
            matching = (
                existing.get("stage_version") == DISORDER_VERSION
                and existing.get("graph_id") == graph_id
                and existing.get("potential_id") == potential_id
                and [float(value) for value in existing.get("amplitude_ratios", [])]
                == [float(value) for value in amplitude_ratios]
            )
            if matching:
                return output_dir
            raise ValueError("completed disorder output has a different cache identity or configuration")
    manifest: dict[str, object] = {
        **runtime_metadata(project_root),
        "stage": "stage3_pa_diagonal_disorder_pilot",
        "stage_version": DISORDER_VERSION,
        "source_dir": str(source_dir),
        "source_manifest_sha256": sha256_file(source_dir / "run_manifest.json"),
        "graph_id": graph_id,
        "potential_id": potential_id,
        "potential_seed": int(potential_seed),
        "operator": "H(a)=A+a*diag(xi), unscaled historical adjacency A",
        "adjacency_scale_c": 1.0,
        "s_ref": S_REF,
        "s_ref_definition": "sqrt(prescribed mean degree 2m)=sqrt(20), frozen for all graphs and potentials",
        "realized_s_B": realized_s_b,
        "realized_s_B_definition": "sqrt(trace(A^2)/N)=sqrt(realized mean degree)",
        "amplitude_ratios": list(amplitude_ratios),
        "potential_law": "xi iid Uniform[-1,1), NumPy Generator uniform convention",
        "same_xi_reused_across_all_a": True,
        "no_post_disorder_spectral_rescaling": True,
        "distance_metric": "unweighted shortest paths in the unchanged PA graph",
        "top_mode_policy": "included by default for a>0; baseline-compatible non-Perron rows separately labeled",
        "mode_tracking": "not implemented in pilot; all comparisons explicitly rank/window based",
        "graph_metadata": graph_metadata,
    }
    write_json(manifest_path, manifest)
    np.savez_compressed(output_dir / "paired_potential.npz", xi=xi, potential_seed=np.asarray(potential_seed))
    cache = BFSDistanceCache(graph)
    ratio_summaries: list[dict[str, object]] = []
    try:
        for ratio in amplitude_ratios:
            ratio = float(ratio)
            amplitude = float(ratio * S_REF)
            ratio_dir = output_dir / ("ratio_" + _slug(ratio))
            ratio_dir.mkdir(parents=True, exist_ok=True)
            ratio_manifest_path = ratio_dir / "run_manifest.json"
            cache_id = stable_json_hash(
                {
                    "version": DISORDER_VERSION,
                    "graph_id": graph_id,
                    "potential_id": potential_id,
                    "ratio": ratio,
                    "a": amplitude,
                    "operator": "A+a*diag(xi)",
                    "selection": (
                        "a0_reuses_source_top50+bulk50_and_adds_bottom51; "
                        "positive_a_uses_edge51+q25/q50/q75_width51"
                    ),
                }
            )
            existing = read_json(ratio_manifest_path) if ratio_manifest_path.exists() else None
            if existing is not None and existing.get("status") == "COMPLETE":
                if existing.get("cache_id") != cache_id:
                    raise ValueError("existing ratio output has a different cache identity")
                ratio_summaries.append(dict(existing["summary"]))
                continue
            ratio_started = time.time()
            ratio_manifest: dict[str, object] = {
                "status": "RUNNING",
                "started_at_utc": utc_now(),
                "cache_id": cache_id,
                "a_over_s_ref": ratio,
                "a": amplitude,
                "potential_standard_deviation": amplitude / math.sqrt(3.0),
                "raw_adjacency_half_width": amplitude,
                "graph_id": graph_id,
                "potential_id": potential_id,
            }
            write_json(ratio_manifest_path, ratio_manifest)
            try:
                if ratio == 0.0:
                    (
                        all_values,
                        all_ipr,
                        selected_indices,
                        selected_values,
                        selected_vectors,
                        interior_reference_indices,
                    ) = _load_a0_from_source(source_dir, adjacency)
                    infinity_norms = np.full(n, np.nan, dtype=float)
                    infinity_norms[selected_indices] = np.max(np.abs(selected_vectors), axis=0)
                    operator = adjacency
                else:
                    dense = adjacency.toarray()
                    dense[np.diag_indices_from(dense)] += amplitude * xi
                    all_values, all_vectors = linalg.eigh(
                        dense,
                        overwrite_a=True,
                        check_finite=False,
                        driver="evd",
                    )
                    del dense
                    all_values = np.asarray(all_values, dtype=float)
                    all_vectors = np.asarray(all_vectors, dtype=float)
                    all_ipr = np.sum(np.abs(all_vectors) ** 4, axis=0)
                    infinity_norms = np.max(np.abs(all_vectors), axis=0)
                    selected_indices = np.unique(
                        np.concatenate(list(_selection_indices(n).values()))
                    ).astype(int)
                    selected_values = all_values[selected_indices]
                    selected_vectors = all_vectors[:, selected_indices]
                    interior_reference_indices = _selection_indices(n)["interior_q50"]
                    del all_vectors
                    operator = adjacency + sparse.diags(amplitude * xi, offsets=0, format="csr")
                selected_norm_error = float(
                    np.max(np.abs(np.linalg.norm(selected_vectors, axis=0) - 1.0))
                )
                validation_positions = np.linspace(
                    0,
                    len(selected_indices) - 1,
                    min(20, len(selected_indices)),
                    dtype=int,
                )
                validation_residuals = [
                    relative_residual(
                        operator,
                        float(selected_values[position]),
                        selected_vectors[:, position],
                    )[1]
                    for position in validation_positions
                ]
                validation_vectors = selected_vectors[:, validation_positions]
                selected_orthogonality_error = float(
                    np.linalg.norm(
                        validation_vectors.T @ validation_vectors
                        - np.eye(len(validation_positions)),
                        ord=2,
                    )
                )
                maximum_selected_residual = float(max(validation_residuals))
                if (
                    selected_norm_error > 1e-7
                    or selected_orthogonality_error > 1e-6
                    or maximum_selected_residual > 1e-7
                ):
                    raise AssertionError("disorder eigenpair validation tolerance exceeded")
                selected_rows = _selected_diagnostics(
                    graph=graph,
                    cache=cache,
                    values=selected_values,
                    vectors=selected_vectors,
                    indices=selected_indices,
                    potential=xi,
                    ratio=ratio,
                    amplitude=amplitude,
                )
                upper_mask = selected_indices >= n - 51
                coupling_rows, schur_rows = _coupling_rows(
                    graph=graph,
                    operator=operator,
                    upper_values=selected_values[upper_mask],
                    upper_vectors=selected_vectors[:, upper_mask],
                    ratio=ratio,
                    amplitude=amplitude,
                )
                _write_ratio_outputs(
                    ratio_dir,
                    all_values,
                    all_ipr,
                    infinity_norms,
                    selected_indices,
                    selected_values,
                    selected_vectors,
                    selected_rows,
                    coupling_rows,
                    schur_rows,
                )
                upper_ipr = np.asarray(
                    [row["ipr"] for row in selected_rows if row["region"] == "upper_edge" and float(row["edge_rank_1based"]) <= 10],
                    dtype=float,
                )
                lower_ipr = np.asarray(
                    [row["ipr"] for row in selected_rows if row["region"] == "lower_edge" and float(row["edge_rank_1based"]) <= 10],
                    dtype=float,
                )
                interior_ipr = all_ipr[np.asarray(interior_reference_indices, dtype=int)]
                gaps, gap_ratios = _gaps_and_ratios(all_values)
                summary = {
                    "a_over_s_ref": ratio,
                    "a": amplitude,
                    "potential_standard_deviation": amplitude / math.sqrt(3.0),
                    "minimum_eigenvalue": float(all_values[0]),
                    "maximum_eigenvalue": float(all_values[-1]),
                    "upper_edge_mean_ipr": float(np.mean(upper_ipr)),
                    "lower_edge_mean_ipr": float(np.mean(lower_ipr)),
                    "interior_mean_ipr": float(np.mean(interior_ipr)),
                    "interior_window_definition": (
                        "historical atom-aware bulk50" if ratio == 0.0 else "ascending-rank q50 width51"
                    ),
                    "mean_raw_gap_all": float(np.mean(gaps)),
                    "zero_gap_count_exact": int(np.sum(gaps == 0.0)),
                    "mean_adjacent_gap_ratio_finite": float(np.nanmean(gap_ratios)),
                    "selected_eigenvector_count": len(selected_indices),
                    "coupling_row_count": len(coupling_rows),
                    "schur_row_count": len(schur_rows),
                    "maximum_selected_norm_error": selected_norm_error,
                    "sampled_selected_orthogonality_error": selected_orthogonality_error,
                    "sampled_maximum_selected_relative_residual": maximum_selected_residual,
                    "full_infinity_norm_coverage": bool(ratio != 0.0),
                }
                ratio_summaries.append(summary)
                ratio_manifest.update(
                    {
                        "status": "COMPLETE",
                        "finished_at_utc": utc_now(),
                        "runtime_seconds": time.time() - ratio_started,
                        "summary": summary,
                    }
                )
                write_json(ratio_manifest_path, ratio_manifest)
            except Exception as ratio_error:
                ratio_manifest.update(
                    {
                        "status": "FAILED",
                        "finished_at_utc": utc_now(),
                        "runtime_seconds": time.time() - ratio_started,
                        "failure_type": type(ratio_error).__name__,
                        "failure_message": str(ratio_error),
                    }
                )
                write_json(ratio_manifest_path, ratio_manifest)
                raise
        ratio_summaries.sort(key=lambda row: float(row["a_over_s_ref"]))
        write_csv(output_dir / "sweep_summary.csv", ratio_summaries)
        _plot_sweep(output_dir, ratio_summaries)
        manifest.update(
            {
                "status": "COMPLETE",
                "finished_at_utc": utc_now(),
                "runtime_seconds": time.time() - started,
                "completed_ratio_count": len(ratio_summaries),
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
                "completed_ratio_count": len(ratio_summaries),
            }
        )
        write_json(manifest_path, manifest)
        raise
