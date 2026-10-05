"""Corrected, cache-only analysis of the diagonal-disorder pilot."""

from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/random-schrodinger-correction-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/random-schrodinger-correction-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import sparse

from .common import (
    analyzed_to_original_ids,
    graph_from_edge_cache,
    graph_to_csr,
    runtime_metadata,
    sha256_file,
    source_configuration_identity,
    stable_json_hash,
    utc_now,
    write_csv,
    write_json,
)


DISORDER_CORRECTION_VERSION = "pa_disorder_reanalysis_v3"
S_REF = math.sqrt(20.0)
RATIOS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def _slug(value: float) -> str:
    return ("%g" % float(value)).replace(".", "p")


def _window_indices(n: int, quantile: float, width: int = 51) -> np.ndarray:
    center = int(round((n - 1) * float(quantile)))
    half = width // 2
    start = max(0, min(n - width, center - half))
    return np.arange(start, min(n, start + width), dtype=int)


def _zero_tolerance(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    scale = max(1.0, float(np.max(np.abs(values))))
    n = len(values)
    return max(100.0 * np.finfo(float).eps * n * scale, 1e-12 * scale)


def _window_ipr_summary(
    ipr: np.ndarray, indices: np.ndarray, atom_mask: np.ndarray
) -> dict[str, object]:
    """Separate cached basis-dependent IPR from an interpretable statistic.

    Individual eigenvectors inside a degenerate subspace are not unique. If a
    window touches the near-zero atom, retain the cached values for auditability
    but do not expose them as the corrected window statistic.
    """
    indices = np.asarray(indices, dtype=int)
    atom_count = int(np.sum(np.asarray(atom_mask, dtype=bool)[indices]))
    cached_mean = float(np.mean(np.asarray(ipr, dtype=float)[indices]))
    cached_median = float(np.median(np.asarray(ipr, dtype=float)[indices]))
    interpretable = atom_count == 0
    return {
        "mean_ipr": cached_mean if interpretable else math.nan,
        "median_ipr": cached_median if interpretable else math.nan,
        "cached_basis_dependent_mean_ipr": cached_mean,
        "cached_basis_dependent_median_ipr": cached_median,
        "atom_level_count_in_window": atom_count,
        "individual_mode_ipr_interpretable": interpretable,
        "ipr_status": (
            "basis_invariant_window_statistic"
            if interpretable
            else "undefined_in_corrected_curve; cached_value_retained_but_basis_dependent"
        ),
    }


def _duplicate_neighborhood_groups(graph: nx.Graph) -> list[list[int]]:
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for vertex in range(graph.number_of_nodes()):
        groups[tuple(sorted(int(value) for value in graph.neighbors(vertex)))].append(vertex)
    return sorted(
        (sorted(vertices) for vertices in groups.values() if len(vertices) >= 2),
        key=lambda vertices: (-len(vertices), vertices[0]),
    )


def _projected_diagonal_summary(values: np.ndarray) -> dict[str, float]:
    """Trace moments of P diag(values) P on the sum-zero subspace."""
    values = np.asarray(values, dtype=float)
    k = len(values)
    total = float(np.sum(values))
    sum_squares = float(np.dot(values, values))
    trace = (1.0 - 1.0 / k) * total
    # tr((PDP)^2) for P=I-11*/k; this avoids forming a potentially
    # thousand-dimensional dense projector for condensed PA graphs.
    frobenius_squared = (1.0 - 2.0 / k) * sum_squares + total**2 / k**2
    result = {
        "projected_trace": trace,
        "projected_mean_eigenvalue": trace / (k - 1),
        "projected_frobenius_squared": frobenius_squared,
        "projected_eigenvalue_second_moment": frobenius_squared / (k - 1),
    }
    if k <= 64:
        basis, _ = np.linalg.qr(np.eye(k)[:, :-1] - np.eye(k)[:, [-1]])
        restricted = basis.T @ np.diag(values) @ basis
        eigenvalues = np.linalg.eigvalsh(restricted)
        result.update(
            {
                "exact_projected_minimum": float(eigenvalues[0]),
                "exact_projected_median": float(np.median(eigenvalues)),
                "exact_projected_maximum": float(eigenvalues[-1]),
            }
        )
    else:
        result.update(
            {
                "exact_projected_minimum": math.nan,
                "exact_projected_median": math.nan,
                "exact_projected_maximum": math.nan,
            }
        )
    return result


def _gap_triplets(
    values: np.ndarray,
    atom_mask: np.ndarray,
    ratio: float,
    window_name: str,
    window: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    values = np.asarray(values, dtype=float)
    selected = set(int(index) for index in window)
    rows: list[dict[str, object]] = []
    retained: list[float] = []
    excluded = 0
    for center in window:
        center = int(center)
        if center <= 0 or center >= len(values) - 1:
            continue
        triplet = (center - 1, center, center + 1)
        if not all(index in selected for index in triplet):
            continue
        touches_atom = bool(np.any(atom_mask[np.asarray(triplet, dtype=int)]))
        left = float(values[center] - values[center - 1])
        right = float(values[center + 1] - values[center])
        denominator = max(left, right)
        gap_ratio = min(left, right) / denominator if denominator > 0.0 else math.nan
        if touches_atom:
            excluded += 1
        elif math.isfinite(gap_ratio):
            retained.append(gap_ratio)
        rows.append(
            {
                "a_over_s_ref": ratio,
                "window": window_name,
                "left_full_spectrum_index_0based": center - 1,
                "center_full_spectrum_index_0based": center,
                "right_full_spectrum_index_0based": center + 1,
                "left_energy": float(values[center - 1]),
                "center_energy": float(values[center]),
                "right_energy": float(values[center + 1]),
                "left_gap": left,
                "right_gap": right,
                "gap_ratio": gap_ratio,
                "excluded_due_to_atom": touches_atom,
                "retained_for_corrected_gap_statistic": bool(not touches_atom and math.isfinite(gap_ratio)),
            }
        )
    finite = np.asarray(retained, dtype=float)
    summary = {
        "a_over_s_ref": ratio,
        "window": window_name,
        "candidate_consecutive_triplet_count": len(rows),
        "excluded_triplet_count": excluded,
        "retained_triplet_count": len(finite),
        "mean_gap_ratio_without_bridging": float(np.mean(finite)) if len(finite) else math.nan,
        "median_gap_ratio_without_bridging": float(np.median(finite)) if len(finite) else math.nan,
        "exclusion_rule": "drop any original consecutive triplet touching the atom; never connect levels across removed indices",
    }
    return rows, summary


def _plot_corrected_windows(output_dir: Path, rows: list[dict[str, object]], model: str, seed: int) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.3), sharey=True)
    for axis, window in zip(axes, ("q25_width51", "q50_width51", "q75_width51")):
        subset = sorted(
            (row for row in rows if row["window"] == window),
            key=lambda row: float(row["a_over_s_ref"]),
        )
        axis.plot(
            [float(row["a_over_s_ref"]) for row in subset],
            [float(row["mean_ipr"]) for row in subset],
            "o-",
        )
        axis.set_xscale("symlog", linthresh=0.1)
        axis.set_yscale("log")
        axis.set_xlabel(r"$a/s_{ref}$")
        axis.set_title(window.replace("_width51", ""))
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("mean IPR, fixed ascending-rank window")
    figure.suptitle("%s, potential seed %d: corrected consistent interior windows" % (model, seed))
    figure.tight_layout()
    figure.savefig(output_dir / "consistent_q25_q50_q75_ipr.png", dpi=200)
    plt.close(figure)


def run_disorder_correction(
    *,
    source_dir: Path,
    original_sweep_dir: Path,
    output_dir: Path,
    project_root: Path,
) -> Path:
    started = time.time()
    source_dir = Path(source_dir)
    original_sweep_dir = Path(original_sweep_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_path = source_dir / "tables" / "graph_realization.npz"
    graph, graph_metadata = graph_from_edge_cache(graph_path)
    n = graph.number_of_nodes()
    original_ids = analyzed_to_original_ids(graph)
    adjacency = graph_to_csr(graph, nodelist=range(n))
    potential_path = original_sweep_dir / "paired_potential.npz"
    with np.load(potential_path, allow_pickle=False) as data:
        xi = np.asarray(data["xi"], dtype=float)
        potential_seed = int(data["potential_seed"])
    if xi.shape != (n,):
        raise ValueError("paired potential does not match graph dimension")
    source_configuration = source_configuration_identity(project_root)
    ratio_paths: list[Path] = []
    for ratio in RATIOS:
        ratio_dir = original_sweep_dir / ("ratio_" + _slug(ratio))
        ratio_paths.extend(
            [
                ratio_dir / "spectrum_diagnostics.npz",
                ratio_dir / "selected_eigenvectors.npz",
                ratio_dir / "run_manifest.json",
            ]
        )
    required = [graph_path, potential_path, original_sweep_dir / "run_manifest.json", *ratio_paths]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    input_hashes = {str(path): sha256_file(path) for path in required}
    cache_id = stable_json_hash(
        {
            "version": DISORDER_CORRECTION_VERSION,
            "source_configuration_id": source_configuration["sha256"],
            "input_hashes": input_hashes,
            "ratios": RATIOS,
        }
    )
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("status") == "COMPLETE" and existing.get("cache_id") == cache_id:
            return output_dir
        raise ValueError("existing corrected disorder output has a different cache identity")
    manifest: dict[str, object] = {
        **runtime_metadata(project_root),
        "status": "RUNNING",
        "stage": "stage3_disorder_corrected_reanalysis",
        "stage_version": DISORDER_CORRECTION_VERSION,
        "cache_id": cache_id,
        "source_configuration_id": source_configuration["sha256"],
        "source_dir": str(source_dir),
        "original_sweep_dir": str(original_sweep_dir),
        "output_dir": str(output_dir),
        "potential_seed": potential_seed,
        "input_hashes": input_hashes,
        "historical_outputs_modified": False,
        "superseded_output": str(original_sweep_dir / "paired_disorder_ipr_summary.png"),
    }
    write_json(manifest_path, manifest)
    try:
        window_rows: list[dict[str, object]] = []
        gap_rows: list[dict[str, object]] = []
        gap_summary_rows: list[dict[str, object]] = []
        residual_rows: list[dict[str, object]] = []
        zero_summary_rows: list[dict[str, object]] = []
        a0_values: np.ndarray | None = None
        a0_tolerance = math.nan
        for ratio in RATIOS:
            ratio_dir = original_sweep_dir / ("ratio_" + _slug(ratio))
            with np.load(ratio_dir / "spectrum_diagnostics.npz", allow_pickle=False) as data:
                values = np.asarray(data["eigenvalues"], dtype=float)
                ipr = np.asarray(data["ipr"], dtype=float)
            if values.shape != (n,) or ipr.shape != (n,):
                raise ValueError("full disorder spectrum/IPR arrays have incompatible dimensions")
            tolerance = _zero_tolerance(values)
            atom_mask = np.abs(values) <= tolerance if ratio == 0.0 else np.zeros(n, dtype=bool)
            if ratio == 0.0:
                a0_values = values
                a0_tolerance = tolerance
            windows = {
                "q25_width51": _window_indices(n, 0.25),
                "q50_width51": _window_indices(n, 0.50),
                "q75_width51": _window_indices(n, 0.75),
            }
            for window_name, indices in windows.items():
                ipr_summary = _window_ipr_summary(ipr, indices, atom_mask)
                window_rows.append(
                    {
                        "a_over_s_ref": ratio,
                        "a": ratio * S_REF,
                        "window": window_name,
                        "width": len(indices),
                        "full_spectrum_index_min_0based": int(indices[0]),
                        "full_spectrum_index_max_0based": int(indices[-1]),
                        "energy_min": float(values[indices[0]]),
                        "energy_max": float(values[indices[-1]]),
                        **ipr_summary,
                        "window_definition_consistent_across_amplitudes": True,
                    }
                )
                local_rows, local_summary = _gap_triplets(
                    values, atom_mask, ratio, window_name, indices
                )
                gap_rows.extend(local_rows)
                gap_summary_rows.append(local_summary)
            zero_summary_rows.append(
                {
                    "a_over_s_ref": ratio,
                    "scale_aware_zero_tolerance": tolerance,
                    "near_zero_count_at_tolerance": int(np.sum(np.abs(values) <= tolerance)),
                    "near_zero_count_at_tolerance_over_10": int(np.sum(np.abs(values) <= tolerance / 10.0)),
                    "near_zero_count_at_10_times_tolerance": int(np.sum(np.abs(values) <= 10.0 * tolerance)),
                    "exact_float_zero_count": int(np.sum(values == 0.0)),
                }
            )
            if ratio > 0.0:
                with np.load(ratio_dir / "selected_eigenvectors.npz", allow_pickle=False) as data:
                    indices = np.asarray(data["ascending_indices"], dtype=int)
                    selected_values = np.asarray(data["eigenvalues"], dtype=float)
                    vectors = np.asarray(data["eigenvectors"], dtype=float)
                operator = adjacency + sparse.diags(ratio * S_REF * xi, offsets=0, format="csr")
                if vectors.shape != (n, len(indices)):
                    raise ValueError("selected positive-disorder vectors have incompatible dimensions")
                for position, index in enumerate(indices):
                    vector = vectors[:, position]
                    residual = np.asarray(operator.dot(vector)).reshape(-1) - selected_values[position] * vector
                    residual_rows.append(
                        {
                            "a_over_s_ref": ratio,
                            "full_spectrum_index_0based": int(index),
                            "cached_eigenvalue": float(selected_values[position]),
                            "full_spectrum_eigenvalue": float(values[index]),
                            "eigenvalue_reconstruction_error": abs(float(selected_values[position] - values[index])),
                            "absolute_residual_norm": float(np.linalg.norm(residual)),
                            "relative_residual_norm": float(np.linalg.norm(residual)) / max(1.0, abs(float(selected_values[position]))),
                            "operator_reconstructed_independently": True,
                        }
                    )

        if a0_values is None:
            raise AssertionError("a=0 spectrum was not loaded")
        duplicate_groups = _duplicate_neighborhood_groups(graph)
        zero_group_rows: list[dict[str, object]] = []
        projected_rows: list[dict[str, object]] = []
        explained_dimension = 0
        maximum_kernel_residual = 0.0
        for group_index, vertices in enumerate(duplicate_groups, start=1):
            multiplicity = len(vertices) - 1
            explained_dimension += multiplicity
            reference = vertices[0]
            residual_norms = []
            for vertex in vertices[1 : 1 + min(multiplicity, 16)]:
                vector = np.zeros(n, dtype=float)
                vector[reference] = -1.0 / math.sqrt(2.0)
                vector[vertex] = 1.0 / math.sqrt(2.0)
                residual_norms.append(float(np.linalg.norm(adjacency.dot(vector))))
            group_residual = max(residual_norms, default=0.0)
            maximum_kernel_residual = max(maximum_kernel_residual, group_residual)
            diagonal = 1.0 - 1.0 / len(vertices)
            zero_group_rows.append(
                {
                    "group_index": group_index,
                    "duplicate_open_neighborhood_size": len(vertices),
                    "certified_zero_subspace_dimension": multiplicity,
                    "shared_neighbor_count": len(tuple(graph.neighbors(reference))),
                    "analyzed_vertex_ids": vertices,
                    "original_vertex_ids": [int(original_ids[item]) for item in vertices],
                    "projector_diagonal_on_each_group_vertex": diagonal,
                    "projector_trace": float(multiplicity),
                    "basis_invariant_projector_effective_support": float(len(vertices)),
                    "sampled_maximum_kernel_residual_norm": group_residual,
                    "certificate": "identical adjacency rows imply exact zero modes e_i-e_j",
                }
            )
            xi_group = xi[np.asarray(vertices, dtype=int)]
            projected_rows.append(
                {
                    "group_index": group_index,
                    "potential_seed": potential_seed,
                    "group_size": len(vertices),
                    "zero_subspace_dimension": multiplicity,
                    "xi_minimum": float(np.min(xi_group)),
                    "xi_mean": float(np.mean(xi_group)),
                    "xi_maximum": float(np.max(xi_group)),
                    "exact_projected_spectrum_computed": len(vertices) <= 64,
                    **_projected_diagonal_summary(xi_group),
                }
            )
        a0_near_zero_count = int(np.sum(np.abs(a0_values) <= a0_tolerance))
        a0_count_tighter = int(np.sum(np.abs(a0_values) <= a0_tolerance / 10.0))
        a0_count_looser = int(np.sum(np.abs(a0_values) <= 10.0 * a0_tolerance))
        tolerance_stable = a0_near_zero_count == a0_count_tighter == a0_count_looser
        unresolved_dimension = max(0, a0_near_zero_count - explained_dimension)
        if a0_near_zero_count == 0:
            classification = "no_near_zero_cluster"
        elif unresolved_dimension == 0 and explained_dimension == a0_near_zero_count:
            classification = "fully_certified_by_exact_duplicate_neighborhood_modes"
        elif tolerance_stable:
            classification = "tolerance_stable_near_zero_cluster_not_exactly_certified_from_saved_arrays"
        else:
            classification = "threshold_sensitive_numerically_unresolved_near_zero_cluster"
        projector_summary = {
            "scale_aware_zero_tolerance": a0_tolerance,
            "near_zero_eigenvalue_count": a0_near_zero_count,
            "near_zero_count_at_tolerance_over_10": a0_count_tighter,
            "near_zero_count_at_10_times_tolerance": a0_count_looser,
            "near_zero_dimension_tolerance_stable": tolerance_stable,
            "duplicate_neighborhood_group_count": len(duplicate_groups),
            "certified_exact_zero_subspace_dimension": explained_dimension,
            "numerically_unresolved_additional_dimension": unresolved_dimension,
            "sampled_maximum_certified_kernel_residual_norm": maximum_kernel_residual,
            "classification": classification,
            "full_near_zero_projector_available_from_retained_arrays": unresolved_dimension == 0,
            "full_near_zero_projector_limitation": (
                None
                if unresolved_dimension == 0
                else "historical cache retained eigenvalues/IPRs but not a basis for the interior near-zero cluster"
            ),
            "individual_mode_ipr_policy": "not interpreted inside the degenerate/near-degenerate zero cluster; projector diagnostics are reported instead",
        }
        projected_perturbation_status = {
            "status": (
                "computed_on_all_certified_duplicate_neighborhood_zero_subspaces"
                if duplicate_groups
                else "no_certified_duplicate_neighborhood_zero_subspace_found"
            ),
            "certified_group_count": len(duplicate_groups),
            "unresolved_near_zero_dimension": unresolved_dimension,
            "limitation": (
                None
                if unresolved_dimension == 0
                else "P diag(xi) P for the unresolved full near-zero cluster cannot be reconstructed because its basis was not retained"
            ),
        }
        write_csv(output_dir / "tables" / "consistent_window_ipr.csv", window_rows)
        write_csv(output_dir / "tables" / "nonbridging_gap_triplets.csv.gz", gap_rows)
        write_csv(output_dir / "tables" / "nonbridging_gap_summary.csv", gap_summary_rows)
        write_csv(output_dir / "tables" / "positive_disorder_reconstructed_eigenpair_residuals.csv.gz", residual_rows)
        write_csv(output_dir / "tables" / "zero_tolerance_sensitivity.csv", zero_summary_rows)
        write_csv(output_dir / "tables" / "certified_zero_subspace_groups.csv", zero_group_rows)
        write_csv(output_dir / "tables" / "projected_diagonal_perturbation.csv", projected_rows)
        write_json(output_dir / "zero_subspace_projector_summary.json", projector_summary)
        write_json(
            output_dir / "projected_diagonal_perturbation_status.json",
            projected_perturbation_status,
        )
        _plot_corrected_windows(output_dir, window_rows, source_dir.name, potential_seed)
        max_residual = max((float(row["relative_residual_norm"]) for row in residual_rows), default=0.0)
        max_eigenvalue_error = max((float(row["eigenvalue_reconstruction_error"]) for row in residual_rows), default=0.0)
        if max_residual > 1e-7 or max_eigenvalue_error > 1e-10:
            raise AssertionError("independent positive-disorder eigenpair validation failed")
        manifest.update(
            {
                "status": "COMPLETE",
                "finished_at_utc": utc_now(),
                "runtime_seconds": time.time() - started,
                "model": source_dir.name,
                "analyzed_n": n,
                "graph_metadata": graph_metadata,
                "consistent_window_row_count": len(window_rows),
                "gap_triplet_row_count": len(gap_rows),
                "positive_disorder_eigenpair_count_checked": len(residual_rows),
                "maximum_positive_disorder_relative_residual": max_residual,
                "maximum_positive_disorder_eigenvalue_reconstruction_error": max_eigenvalue_error,
                "zero_subspace_projector_summary": projector_summary,
                "corrected_figure_count": 1,
                "old_curve_status": "preserved_unmodified_and_superseded",
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
