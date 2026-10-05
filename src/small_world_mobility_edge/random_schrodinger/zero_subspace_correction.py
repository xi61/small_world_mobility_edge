"""Bounded Hoffman recomputation of near-zero projectors and perturbations."""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
from scipy import linalg

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
from .disorder_correction import _zero_tolerance


ZERO_SUBSPACE_VERSION = "zero_subspace_projector_v1"


def _projector_diagnostics(
    operator, values: np.ndarray, basis: np.ndarray
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    basis = np.asarray(basis, dtype=float)
    dimension = basis.shape[1]
    if basis.shape[0] != operator.shape[0] or values.shape != (dimension,):
        raise ValueError("near-zero basis dimensions are incompatible")
    gram_error = float(np.linalg.norm(basis.T @ basis - np.eye(dimension)))
    residual = np.asarray(operator.dot(basis)) - basis * values[None, :]
    residual_norms = np.linalg.norm(residual, axis=0)
    projector_diagonal = np.sum(np.square(basis), axis=1)
    trace = float(np.sum(projector_diagonal))
    effective_support = (
        float(trace**2 / np.dot(projector_diagonal, projector_diagonal))
        if trace > 0.0
        else 0.0
    )
    summary = {
        "dimension": dimension,
        "orthonormality_frobenius_error": gram_error,
        "maximum_absolute_eigenpair_residual_norm": float(np.max(residual_norms, initial=0.0)),
        "residual_frobenius_norm": float(np.linalg.norm(residual)),
        "projector_trace": trace,
        "projector_trace_error": abs(trace - dimension),
        "projector_frobenius_norm": math.sqrt(float(dimension)),
        "projector_diagonal_minimum": float(np.min(projector_diagonal, initial=math.inf)),
        "projector_diagonal_maximum": float(np.max(projector_diagonal, initial=0.0)),
        "projector_effective_support": effective_support,
        "basis_invariance": "projector diagonal, trace, effective support, and projected-potential spectrum are invariant under orthogonal basis rotations",
    }
    return summary, projector_diagonal, residual_norms


def _projected_potential_spectrum(basis: np.ndarray, xi: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    basis = np.asarray(basis, dtype=float)
    xi = np.asarray(xi, dtype=float)
    projected = basis.T @ (xi[:, None] * basis)
    projected = 0.5 * (projected + projected.T)
    eigenvalues = linalg.eigvalsh(projected, check_finite=False, overwrite_a=True)
    return eigenvalues, {
        "projected_trace": float(np.sum(eigenvalues)),
        "projected_frobenius_squared": float(np.dot(eigenvalues, eigenvalues)),
        "projected_minimum": float(eigenvalues[0]),
        "projected_q25": float(np.quantile(eigenvalues, 0.25)),
        "projected_median": float(np.median(eigenvalues)),
        "projected_q75": float(np.quantile(eigenvalues, 0.75)),
        "projected_maximum": float(eigenvalues[-1]),
    }


def run_zero_subspace_correction(
    *,
    source_dir: Path,
    sweep_dirs: list[Path],
    output_dir: Path,
    project_root: Path,
) -> Path:
    started = time.time()
    source_dir = Path(source_dir)
    sweep_dirs = [Path(path) for path in sweep_dirs]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_path = source_dir / "tables" / "graph_realization.npz"
    spectrum_paths = [path / "ratio_0" / "spectrum_diagnostics.npz" for path in sweep_dirs]
    potential_paths = [path / "paired_potential.npz" for path in sweep_dirs]
    required = [graph_path, *spectrum_paths, *potential_paths]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    input_hashes = {str(path): sha256_file(path) for path in required}
    source_configuration = source_configuration_identity(project_root)
    cache_id = stable_json_hash(
        {
            "version": ZERO_SUBSPACE_VERSION,
            "source_configuration_id": source_configuration["sha256"],
            "input_hashes": input_hashes,
        }
    )
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        import json

        existing = json.loads(manifest_path.read_text())
        if existing.get("status") == "COMPLETE" and existing.get("cache_id") == cache_id:
            return output_dir
        raise ValueError("existing zero-subspace output has another cache identity")
    manifest: dict[str, object] = {
        **runtime_metadata(project_root),
        "status": "RUNNING",
        "stage": "zero_subspace_projector_and_projected_disorder",
        "stage_version": ZERO_SUBSPACE_VERSION,
        "cache_id": cache_id,
        "source_configuration_id": source_configuration["sha256"],
        "source_dir": str(source_dir),
        "sweep_dirs": [str(path) for path in sweep_dirs],
        "output_dir": str(output_dir),
        "input_hashes": input_hashes,
        "historical_outputs_modified": False,
    }
    write_json(manifest_path, manifest)
    try:
        graph, graph_metadata = graph_from_edge_cache(graph_path)
        operator = graph_to_csr(graph, nodelist=range(graph.number_of_nodes()))
        with np.load(spectrum_paths[0], allow_pickle=False) as data:
            cached_values = np.asarray(data["eigenvalues"], dtype=float)
        tolerance = _zero_tolerance(cached_values)
        near_indices = np.flatnonzero(np.abs(cached_values) <= tolerance)
        tighter_count = int(np.sum(np.abs(cached_values) <= tolerance / 10.0))
        looser_count = int(np.sum(np.abs(cached_values) <= 10.0 * tolerance))
        tolerance_stable = len(near_indices) == tighter_count == looser_count
        original_ids = analyzed_to_original_ids(graph)
        perturbation_rows: list[dict[str, object]] = []
        if len(near_indices) == 0:
            projector_summary = {
                "dimension": 0,
                "classification": "no_near_zero_cluster",
                "scale_aware_zero_tolerance": tolerance,
                "near_zero_dimension_tolerance_stable": tolerance_stable,
            }
            projector_rows: list[dict[str, object]] = []
            residual_rows: list[dict[str, object]] = []
        else:
            if not np.array_equal(near_indices, np.arange(near_indices[0], near_indices[-1] + 1)):
                raise AssertionError("near-zero spectrum indices are not contiguous")
            dense = operator.toarray()
            values, basis = linalg.eigh(
                dense,
                subset_by_index=(int(near_indices[0]), int(near_indices[-1])),
                driver="evr",
                overwrite_a=True,
                check_finite=False,
            )
            del dense
            projector_summary, projector_diagonal, residual_norms = _projector_diagnostics(
                operator, values, basis
            )
            maximum_cached_value_error = float(
                np.max(np.abs(values - cached_values[near_indices]), initial=0.0)
            )
            projector_summary.update(
                {
                    "classification": (
                        "numerically_resolved_tolerance_stable_cluster_believed_exact_not_symbolically_certified"
                        if tolerance_stable
                        else "numerically_resolved_threshold_sensitive_cluster"
                    ),
                    "scale_aware_zero_tolerance": tolerance,
                    "near_zero_count_at_tolerance_over_10": tighter_count,
                    "near_zero_count_at_10_times_tolerance": looser_count,
                    "near_zero_dimension_tolerance_stable": tolerance_stable,
                    "maximum_recomputed_vs_cached_eigenvalue_error": maximum_cached_value_error,
                    "exactness_scope": "strong numerical residual and tolerance-stability evidence; no symbolic exact-nullspace certificate",
                }
            )
            projector_rows = [
                {
                    "analyzed_vertex_id": vertex,
                    "original_vertex_id": int(original_ids[vertex]),
                    "projector_diagonal": float(projector_diagonal[vertex]),
                }
                for vertex in range(graph.number_of_nodes())
            ]
            residual_rows = [
                {
                    "full_spectrum_index_0based": int(index),
                    "recomputed_eigenvalue": float(value),
                    "cached_eigenvalue": float(cached_values[index]),
                    "absolute_residual_norm": float(residual),
                }
                for index, value, residual in zip(near_indices, values, residual_norms)
            ]
            for sweep_dir, potential_path in zip(sweep_dirs, potential_paths):
                with np.load(potential_path, allow_pickle=False) as data:
                    xi = np.asarray(data["xi"], dtype=float)
                    potential_seed = int(data["potential_seed"])
                projected_values, projected_summary = _projected_potential_spectrum(basis, xi)
                np.savez_compressed(
                    output_dir / ("projected_potential_eigenvalues_seed_%d.npz" % potential_seed),
                    eigenvalues=projected_values,
                    potential_seed=np.asarray(potential_seed, dtype=np.int64),
                )
                perturbation_rows.append(
                    {
                        "potential_seed": potential_seed,
                        "sweep_dir": str(sweep_dir),
                        "zero_subspace_dimension": len(values),
                        **projected_summary,
                        "interpretation": "first-order slopes for splitting under a*diag(xi) within the recomputed near-zero subspace",
                    }
                )
        write_csv(output_dir / "projector_diagonal.csv.gz", projector_rows)
        write_csv(output_dir / "near_zero_eigenpair_residuals.csv.gz", residual_rows)
        write_csv(output_dir / "projected_diagonal_perturbation_summary.csv", perturbation_rows)
        write_json(output_dir / "zero_subspace_projector_summary.json", projector_summary)
        residual_maximum = float(projector_summary.get("maximum_absolute_eigenpair_residual_norm", 0.0))
        orthogonality_error = float(projector_summary.get("orthonormality_frobenius_error", 0.0))
        if residual_maximum > 1e-7 or orthogonality_error > 1e-7:
            raise AssertionError("recomputed near-zero eigenspace validation failed")
        manifest.update(
            {
                "status": "COMPLETE",
                "finished_at_utc": utc_now(),
                "runtime_seconds": time.time() - started,
                "model": source_dir.name,
                "graph_metadata": graph_metadata,
                "projector_summary": projector_summary,
                "projected_potential_count": len(perturbation_rows),
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
