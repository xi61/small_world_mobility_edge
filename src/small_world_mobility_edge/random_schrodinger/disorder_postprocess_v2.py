"""Cache-only repair of the completed PA diagonal-disorder pilot."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Mapping

import numpy as np

from .anderson_utils import (
    atomic_write_csv,
    atomic_write_json,
    checkpoint_diagnostics,
    verify_execution_context,
    write_checkpoint_tables,
)
from .common import graph_from_edge_cache, graph_to_csr, read_json, sha256_file, utc_now


PILOT_POSTPROCESS_VERSION = "anderson_pilot_cache_only_postprocess_v1"


def ratio_slug(value: float) -> str:
    return ("%.12g" % float(value)).replace("-", "m").replace(".", "p")


def _model_spec(config: Mapping[str, object], model_key: str) -> dict[str, object]:
    models = list(config["pilot"]["models"])  # type: ignore[index]
    seeds = list(config["pilot"]["graph_seeds"])  # type: ignore[index]
    if model_key not in models:
        raise ValueError("unknown pilot model")
    position = models.index(model_key)
    return {"model_key": model_key, "graph_seed": int(seeds[position])}


def run_pilot_postprocess(
    *,
    project_root: Path,
    config_path: Path,
    freeze_manifest: Path,
    model_key: str,
    potential_seed: int,
    output_dir: Path,
) -> Path:
    started = time.time()
    context = verify_execution_context(project_root, config_path, freeze_manifest)
    config = context["config"]
    spec = _model_spec(config, model_key)
    pilot = config["pilot"]
    potential_seeds = [int(value) for value in pilot["potential_seeds"]]
    if int(potential_seed) not in potential_seeds:
        raise ValueError("unexpected potential seed")
    historical_root = Path(str(config["historical_pilot_root"]))
    retained_root = Path(str(config["retained_pa_root"]))
    corrected_root = Path(str(config["corrected_root"]))
    sweep_dir = historical_root / model_key / ("potential_seed_%d" % int(potential_seed))
    source_dir = retained_root / model_key
    graph_path = source_dir / "tables" / "graph_realization.npz"
    graph, graph_metadata = graph_from_edge_cache(graph_path)
    adjacency = graph_to_csr(graph, nodelist=range(graph.number_of_nodes()))
    with np.load(sweep_dir / "paired_potential.npz", allow_pickle=False) as data:
        xi = np.asarray(data["xi"], dtype=float)
        stored_seed = int(data["potential_seed"])
    if stored_seed != int(potential_seed):
        raise ValueError("paired potential seed mismatch")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    manifest = {
        "status": "RUNNING",
        "stage": "pilot_cache_only_postprocessing",
        "stage_version": PILOT_POSTPROCESS_VERSION,
        "started_at_utc": utc_now(),
        "model_key": model_key,
        "graph_seed": spec["graph_seed"],
        "potential_seed": int(potential_seed),
        "source_configuration_id": context["source_configuration_identity"]["sha256"],
        "configuration_sha256": context["configuration_sha256"],
        "freeze_manifest_sha256": context["freeze_manifest_sha256"],
        "historical_sweep": str(sweep_dir),
        "corrected_validation": str(corrected_root / "disorder" / model_key / ("potential_seed_%d" % int(potential_seed))),
        "graph_metadata": graph_metadata,
        "eigensolver_invoked": False,
        "historical_outputs_modified": False,
    }
    atomic_write_json(manifest_path, manifest)
    coverage_rows: list[dict[str, object]] = []
    processed = 0
    deduplicated = 0
    try:
        for ratio in [float(value) for value in pilot["amplitude_ratios"]]:
            ratio_dir = sweep_dir / ("ratio_" + ratio_slug(ratio))
            files = [
                ratio_dir / "run_manifest.json",
                ratio_dir / "spectrum_diagnostics.npz",
                ratio_dir / "selected_eigenvectors.npz",
            ]
            if any(not path.is_file() for path in files):
                raise FileNotFoundError("missing pilot input for ratio %.12g" % ratio)
            ratio_manifest = read_json(files[0])
            if ratio_manifest.get("status") != "COMPLETE":
                raise ValueError("pilot ratio is not complete")
            spectrum_path, vectors_path = files[1], files[2]
            with np.load(spectrum_path, allow_pickle=False) as data:
                values = np.asarray(data["eigenvalues"], dtype=float)
                ipr = np.asarray(data["ipr"], dtype=float)
                infinity_norms = np.asarray(data["infinity_norms"], dtype=float)
            with np.load(vectors_path, allow_pickle=False) as data:
                selected_indices = np.asarray(data["ascending_indices"], dtype=int)
                selected_values = np.asarray(data["eigenvalues"], dtype=float)
                selected_vectors = np.asarray(data["eigenvectors"], dtype=float)
            if not np.array_equal(selected_values, values[selected_indices]):
                raise ValueError("selected values do not exactly match stored full spectrum")
            selected_maximum_mass = np.max(np.square(np.abs(selected_vectors)), axis=0)
            stored_selected_mass = np.square(infinity_norms[selected_indices])
            maximum_mass_error = float(np.nanmax(np.abs(selected_maximum_mass - stored_selected_mass)))
            if maximum_mass_error > 1e-12:
                raise AssertionError("stored infinity norm is not the vector infinity norm")
            checkpoint_output = output_dir / ("ratio_" + ratio_slug(ratio))
            duplicate_a0 = bool(ratio == 0.0 and int(potential_seed) != potential_seeds[0])
            coverage_rows.append({
                "model_key": model_key,
                "graph_seed": spec["graph_seed"],
                "potential_seed": int(potential_seed),
                "a_over_s_ref": ratio,
                "a": float(ratio_manifest["a"]),
                "spectrum_path": str(spectrum_path),
                "spectrum_sha256": sha256_file(spectrum_path),
                "selected_vectors_path": str(vectors_path),
                "selected_vectors_sha256": sha256_file(vectors_path),
                "full_scalar_level_count": int(len(values)),
                "raw_vector_count": int(len(selected_indices)),
                "full_maximum_mass_coverage_count": int(np.count_nonzero(np.isfinite(infinity_norms))),
                "maximum_selected_infinity_norm_squared_error": maximum_mass_error,
                "a_zero_deduplicated": duplicate_a0,
                "postprocess_status": "DEDUPLICATED_REFERENCE" if duplicate_a0 else "COMPLETE",
            })
            if duplicate_a0:
                reference = output_dir.parent / ("potential_seed_%d" % potential_seeds[0]) / "ratio_0"
                atomic_write_json(checkpoint_output / "run_manifest.json", {
                    "status": "DEDUPLICATED_REFERENCE",
                    "stage_version": PILOT_POSTPROCESS_VERSION,
                    "model_key": model_key,
                    "potential_seed": int(potential_seed),
                    "a_over_s_ref": ratio,
                    "reference": str(reference),
                    "reason": "a=0 is graph-level and independent of potential stream",
                })
                deduplicated += 1
                continue
            amplitude = float(ratio_manifest["a"])
            prefix = {
                "source_kind": "historical_pilot_repaired_postprocessing",
                "model_key": model_key,
                "n": int(len(values)),
                "graph_seed": spec["graph_seed"],
                "potential_seed": int(potential_seed) if ratio > 0.0 else -1,
                "a_over_s_ref": ratio,
                "a": amplitude,
            }
            bundle = checkpoint_diagnostics(
                graph=graph,
                adjacency=adjacency,
                values=values,
                ipr=ipr,
                full_maximum_mass=np.square(infinity_norms),
                selected_indices=selected_indices,
                selected_vectors=selected_vectors,
                potential=xi if ratio > 0.0 else None,
                amplitude=amplitude,
                prefix=prefix,
                physical_half_width=0.1 * float(pilot["s_ref"]),
                atom_enabled=ratio == 0.0,
                shell_max_radius=int(config["localization"]["shell_max_radius"]),
            )
            write_checkpoint_tables(checkpoint_output, bundle)
            atomic_write_json(checkpoint_output / "run_manifest.json", {
                "status": "COMPLETE",
                "stage_version": PILOT_POSTPROCESS_VERSION,
                "source_configuration_id": context["source_configuration_identity"]["sha256"],
                "model_key": model_key,
                "graph_seed": spec["graph_seed"],
                "potential_seed": int(potential_seed) if ratio > 0.0 else None,
                "a_over_s_ref": ratio,
                "a": amplitude,
                "input_spectrum_sha256": sha256_file(spectrum_path),
                "input_selected_vectors_sha256": sha256_file(vectors_path),
                "selected_vector_count": int(len(selected_indices)),
                "zero_cluster": bundle["zero_cluster"],
                "distance_cache_source_count": bundle["distance_cache_source_count"],
                "eigensolver_invoked": False,
                "finished_at_utc": utc_now(),
            })
            processed += 1
        atomic_write_csv(output_dir / "tables" / "input_coverage_ledger.csv", coverage_rows)
        corrected_manifest = corrected_root / "disorder" / model_key / ("potential_seed_%d" % int(potential_seed)) / "run_manifest.json"
        corrected = read_json(corrected_manifest)
        if corrected.get("status") != "COMPLETE" or float(corrected.get("maximum_positive_disorder_relative_residual", math.inf)) > 1e-8:
            raise ValueError("corrected positive-disorder validation is missing or failed")
        manifest.update({
            "status": "COMPLETE",
            "finished_at_utc": utc_now(),
            "runtime_seconds": time.time() - started,
            "processed_checkpoint_count": processed,
            "deduplicated_a_zero_checkpoint_count": deduplicated,
            "input_checkpoint_count": len(coverage_rows),
            "corrected_manifest_sha256": sha256_file(corrected_manifest),
            "inherited_maximum_positive_disorder_relative_residual": corrected["maximum_positive_disorder_relative_residual"],
        })
        atomic_write_json(manifest_path, manifest)
        return output_dir
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
