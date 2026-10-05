"""Targeted PA baseline supplements and bounded fresh PA disorder replication."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Mapping

import networkx as nx
import numpy as np
from scipy import linalg, sparse

from small_world_mobility_edge.preferential_attachment.baseline import (
    MODEL_AFFINE_PA,
    MODEL_NONLINEAR_PA,
    generate_preferential_attachment_graph,
)

from .anderson_utils import (
    atomic_savez,
    atomic_write_json,
    checkpoint_diagnostics,
    graph_identity,
    potential_identity,
    selection_union,
    validate_selected_eigenpairs,
    verify_execution_context,
    width_count_quantile_indices,
    window_registry,
    write_checkpoint_tables,
)
from .common import (
    graph_from_edge_cache,
    graph_to_csr,
    read_json,
    save_graph_cache,
    sha256_file,
    stable_json_hash,
    utc_now,
)
from .handoff_metrics import stable_zero_cluster


BASELINE_SUPPLEMENT_VERSION = "anderson_targeted_baseline_supplement_v1"
PA_REPLICATION_VERSION = "anderson_fresh_pa_replication_v1"


def _ratio_slug(value: float) -> str:
    return ("%.12g" % float(value)).replace("-", "m").replace(".", "p")


def _fresh_pa_task(config: Mapping[str, object], task_id: int) -> dict[str, object]:
    fresh = config["fresh_pa"]
    models = list(fresh["models"])
    sizes = [int(value) for value in fresh["n_values"]]
    replicates = int(fresh["graph_replicates"])
    total = len(models) * len(sizes) * replicates
    if not 1 <= int(task_id) <= total:
        raise ValueError("PA task ID outside configured range")
    zero = int(task_id) - 1
    model = dict(models[zero // (len(sizes) * replicates)])
    remainder = zero % (len(sizes) * replicates)
    n = sizes[remainder // replicates]
    graph_rep = remainder % replicates
    key = str(model["model_key"])
    seed = int(fresh["graph_seed_table"][key][str(n)][graph_rep])
    return {**model, "n": n, "graph_replicate": graph_rep + 1, "graph_seed": seed, "task_id": int(task_id)}


def _retention_windows(values: np.ndarray, physical_half_width: float) -> dict[str, np.ndarray]:
    all_windows = window_registry(values, pa_physical_half_width=physical_half_width)
    names = [
        "lower_rank_1_10", "lower_rank_11_25", "lower_rank_26_50",
        "upper_rank_1_10", "upper_rank_11_25", "upper_rank_26_50",
        "q25_width_0p02", "q50_width_0p02", "q75_width_0p02", "physical_abs_energy",
    ]
    return {name: all_windows[name] for name in names}


def _solve_full(dense: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    started = time.time()
    values, vectors = linalg.eigh(dense, overwrite_a=True, check_finite=False, driver="evd")
    return np.asarray(values, dtype=float), np.asarray(vectors, dtype=float), time.time() - started


def _save_pa_checkpoint(
    *,
    graph: nx.Graph,
    adjacency: sparse.spmatrix,
    dense_operator: np.ndarray,
    output_dir: Path,
    prefix: Mapping[str, object],
    potential: np.ndarray | None,
    amplitude: float,
    physical_half_width: float,
    atom_enabled: bool,
    context: Mapping[str, object],
    graph_id: str,
    potential_id: str | None,
    config: Mapping[str, object],
) -> dict[str, object]:
    output_dir = Path(output_dir)
    manifest_path = output_dir / "run_manifest.json"
    cache_payload = {
        "stage_version": PA_REPLICATION_VERSION,
        "source_configuration_id": context["source_configuration_identity"]["sha256"],
        "configuration_sha256": context["configuration_sha256"],
        "graph_id": graph_id,
        "potential_id": potential_id,
        "operator": "A+a*diag(xi)",
        "n": graph.number_of_nodes(),
        "amplitude": float(amplitude),
        "solver": config["fresh_pa"]["dense_solver"],
        "windows": config["windows"],
        "tolerances": config["validation"],
    }
    cache_id = stable_json_hash(cache_payload)
    if manifest_path.is_file():
        existing = read_json(manifest_path)
        if existing.get("status") == "COMPLETE" and existing.get("cache_id") == cache_id:
            return dict(existing)
        if existing.get("status") == "COMPLETE":
            raise ValueError("completed PA checkpoint has incompatible cache identity")
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(manifest_path, {"status": "RUNNING", "cache_id": cache_id, **prefix, "started_at_utc": utc_now()})
    solve_started = time.time()
    try:
        values, vectors, eigensolve_seconds = _solve_full(dense_operator)
        ipr = np.sum(np.square(np.square(np.abs(vectors))), axis=0)
        maximum_mass = np.max(np.square(np.abs(vectors)), axis=0)
        zero = stable_zero_cluster(values, enabled=atom_enabled)
        atom_mask = np.asarray(zero["mask"], dtype=bool)
        retention = _retention_windows(values, physical_half_width)
        selected_indices, expanded = selection_union(retention, n=len(values), guard=1, atom_mask=atom_mask)
        selected_vectors = vectors[:, selected_indices]
        selected_values = values[selected_indices]
        validation = validate_selected_eigenpairs(adjacency + sparse.diags(float(amplitude) * potential, offsets=0, format="csr") if potential is not None else adjacency, selected_values, selected_vectors)
        limits = config["validation"]
        if validation["maximum_relative_residual"] > float(limits["maximum_relative_residual"]) or validation["maximum_norm_error"] > float(limits["maximum_norm_error"]) or validation["sampled_orthogonality_spectral_error"] > float(limits["maximum_sampled_orthogonality_error"]):
            raise AssertionError("PA selected eigenpair validation failed")
        arrays = {
            "eigenvalues": values,
            "ipr": ipr,
            "n_times_ipr": len(values) * ipr,
            "maximum_coordinate_mass": maximum_mass,
            "atom_mask": atom_mask,
        }
        if np.any(atom_mask):
            atom_vectors = vectors[:, atom_mask]
            projector_diagonal = np.sum(np.square(np.abs(atom_vectors)), axis=1)
            arrays["atom_projector_diagonal"] = projector_diagonal
            arrays["atom_projector_trace"] = np.asarray(float(np.sum(projector_diagonal)))
            arrays["atom_effective_support"] = np.asarray(float(np.sum(projector_diagonal) ** 2 / np.sum(projector_diagonal ** 2)))
        atomic_savez(output_dir / "spectrum_scalars.npz", **arrays)
        atomic_savez(
            output_dir / "selected_eigenvectors.npz",
            ascending_indices=selected_indices,
            eigenvalues=selected_values,
            eigenvectors=selected_vectors,
        )
        bundle = checkpoint_diagnostics(
            graph=graph,
            adjacency=adjacency,
            values=values,
            ipr=ipr,
            full_maximum_mass=maximum_mass,
            selected_indices=selected_indices,
            selected_vectors=selected_vectors,
            potential=potential,
            amplitude=amplitude,
            prefix=prefix,
            physical_half_width=physical_half_width,
            atom_enabled=atom_enabled,
            shell_max_radius=int(config["localization"]["shell_max_radius"]),
        )
        write_checkpoint_tables(output_dir, bundle)
        manifest = {
            "status": "COMPLETE",
            "stage_version": PA_REPLICATION_VERSION,
            "cache_id": cache_id,
            **prefix,
            "source_configuration_id": context["source_configuration_identity"]["sha256"],
            "configuration_sha256": context["configuration_sha256"],
            "graph_id": graph_id,
            "potential_id": potential_id,
            "operator": "H(a)=A+a*diag(xi), unscaled adjacency",
            "eigensolver": config["fresh_pa"]["dense_solver"],
            "eigensolve_seconds": eigensolve_seconds,
            "runtime_seconds": time.time() - solve_started,
            "full_spectrum_level_count": int(len(values)),
            "selected_vector_count": int(len(selected_indices)),
            "retention_window_endpoints": {name: [int(index[0]), int(index[-1]), int(len(index))] if len(index) else [-1, -1, 0] for name, index in retention.items()},
            "expanded_guard_endpoints": {name: [int(index[0]), int(index[-1]), int(len(index))] if len(index) else [-1, -1, 0] for name, index in expanded.items()},
            "zero_cluster": bundle["zero_cluster"],
            "validation": validation,
            "finished_at_utc": utc_now(),
        }
        atomic_write_json(manifest_path, manifest)
        return manifest
    except Exception as error:
        atomic_write_json(manifest_path, {"status": "FAILED", "cache_id": cache_id, **prefix, "failure_type": type(error).__name__, "failure_message": str(error), "finished_at_utc": utc_now()})
        raise


def run_pa_replication_task(
    *,
    project_root: Path,
    config_path: Path,
    freeze_manifest: Path,
    task_id: int,
    output_root: Path,
) -> Path:
    started = time.time()
    context = verify_execution_context(project_root, config_path, freeze_manifest)
    config = context["config"]
    spec = _fresh_pa_task(config, int(task_id))
    fresh = config["fresh_pa"]
    output_dir = Path(output_root) / str(spec["model_key"]) / ("N%d" % int(spec["n"])) / ("graph_rep_%d_seed_%d" % (int(spec["graph_replicate"]), int(spec["graph_seed"])))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    manifest = {
        "status": "RUNNING", "stage": "fresh_pa_replication", "stage_version": PA_REPLICATION_VERSION,
        "started_at_utc": utc_now(), **spec,
        "source_configuration_id": context["source_configuration_identity"]["sha256"],
        "configuration_sha256": context["configuration_sha256"],
    }
    atomic_write_json(manifest_path, manifest)
    try:
        graph_cache = output_dir / "graph" / "graph_realization.npz"
        if graph_cache.is_file():
            graph, graph_metadata = graph_from_edge_cache(graph_cache)
        else:
            if spec["kind"] == "affine":
                graph, diagnostics = generate_preferential_attachment_graph(
                    n=int(spec["n"]), m=int(fresh["m"]), seed=int(spec["graph_seed"]),
                    model=MODEL_AFFINE_PA, degree_ccdf_alpha=float(spec["degree_ccdf_alpha"]),
                )
            elif spec["kind"] == "nonlinear":
                graph, diagnostics = generate_preferential_attachment_graph(
                    n=int(spec["n"]), m=int(fresh["m"]), seed=int(spec["graph_seed"]),
                    model=MODEL_NONLINEAR_PA, attachment_zeta=float(spec["attachment_zeta"]),
                )
            else:
                raise ValueError("unsupported PA model kind")
            save_graph_cache(graph_cache, graph, np.arange(graph.number_of_nodes()), {**diagnostics.to_dict(), "model_key": spec["model_key"]})
            graph, graph_metadata = graph_from_edge_cache(graph_cache)
        adjacency = graph_to_csr(graph, nodelist=range(graph.number_of_nodes()))
        graph_id = graph_identity(graph_cache, operator="unscaled simple PA adjacency", source_id=str(context["source_configuration_identity"]["sha256"]))
        n = graph.number_of_nodes()
        physical_half_width = float(config["windows"]["pa_physical_half_width_in_s_ref"]) * float(fresh["s_ref"])
        checkpoints: list[dict[str, object]] = []
        baseline_prefix = {
            "source_kind": "fresh_pa_replication", "model_key": spec["model_key"], "n": n,
            "graph_replicate": spec["graph_replicate"], "graph_seed": spec["graph_seed"],
            "potential_replicate": 0, "potential_seed": -1, "a_over_s_ref": 0.0, "a": 0.0,
        }
        baseline = _save_pa_checkpoint(
            graph=graph, adjacency=adjacency, dense_operator=adjacency.toarray(), output_dir=output_dir / "baseline",
            prefix=baseline_prefix, potential=None, amplitude=0.0, physical_half_width=physical_half_width,
            atom_enabled=True, context=context, graph_id=graph_id, potential_id=None, config=config,
        )
        checkpoints.append(baseline)
        for potential_rep in range(1, int(fresh["potential_replicates"]) + 1):
            potential_seed = int(spec["graph_seed"]) + 100000000 + potential_rep
            xi = np.random.default_rng(potential_seed).uniform(-1.0, 1.0, size=n)
            potential_id = potential_identity(xi, graph_id=graph_id, seed=potential_seed)
            potential_dir = output_dir / ("potential_rep_%d_seed_%d" % (potential_rep, potential_seed))
            atomic_savez(potential_dir / "paired_potential.npz", xi=xi, potential_seed=np.asarray(potential_seed), potential_id=np.asarray(potential_id))
            for ratio in [float(value) for value in fresh["amplitude_ratios"] if float(value) > 0.0]:
                amplitude = ratio * float(fresh["s_ref"])
                dense = adjacency.toarray()
                dense[np.diag_indices_from(dense)] += amplitude * xi
                prefix = {
                    "source_kind": "fresh_pa_replication", "model_key": spec["model_key"], "n": n,
                    "graph_replicate": spec["graph_replicate"], "graph_seed": spec["graph_seed"],
                    "potential_replicate": potential_rep, "potential_seed": potential_seed,
                    "a_over_s_ref": ratio, "a": amplitude,
                }
                checkpoint = _save_pa_checkpoint(
                    graph=graph, adjacency=adjacency, dense_operator=dense,
                    output_dir=potential_dir / ("ratio_" + _ratio_slug(ratio)), prefix=prefix,
                    potential=xi, amplitude=amplitude, physical_half_width=physical_half_width,
                    atom_enabled=False, context=context, graph_id=graph_id, potential_id=potential_id, config=config,
                )
                checkpoints.append(checkpoint)
        if len(checkpoints) != 1 + int(fresh["potential_replicates"]) * (len(fresh["amplitude_ratios"]) - 1):
            raise AssertionError("unexpected PA checkpoint count")
        manifest.update({
            "status": "COMPLETE", "finished_at_utc": utc_now(), "runtime_seconds": time.time() - started,
            "graph_id": graph_id, "graph_cache": str(graph_cache), "graph_cache_sha256": sha256_file(graph_cache),
            "graph_metadata": graph_metadata, "checkpoint_count": len(checkpoints),
            "baseline_count": 1, "positive_disorder_count": len(checkpoints) - 1,
        })
        atomic_write_json(manifest_path, manifest)
        return output_dir
    except Exception as error:
        manifest.update({"status": "FAILED", "finished_at_utc": utc_now(), "runtime_seconds": time.time() - started, "failure_type": type(error).__name__, "failure_message": str(error)})
        atomic_write_json(manifest_path, manifest)
        raise


def run_baseline_supplement(
    *,
    project_root: Path,
    config_path: Path,
    freeze_manifest: Path,
    model_key: str,
    output_dir: Path,
) -> Path:
    started = time.time()
    context = verify_execution_context(project_root, config_path, freeze_manifest)
    config = context["config"]
    pilot = config["pilot"]
    if model_key not in pilot["models"]:
        raise ValueError("unknown supplement model")
    source_dir = Path(str(config["retained_pa_root"])) / model_key
    historical_dir = Path(str(config["historical_pilot_root"])) / model_key / ("potential_seed_%d" % int(pilot["potential_seeds"][0])) / "ratio_0"
    graph_path = source_dir / "tables" / "graph_realization.npz"
    graph, graph_metadata = graph_from_edge_cache(graph_path)
    adjacency = graph_to_csr(graph, nodelist=range(graph.number_of_nodes()))
    with np.load(source_dir / "spectrum" / "single_graph_spectral_arrays.npz", allow_pickle=False) as data:
        full_values = np.asarray(data["eigenvalues"], dtype=float)
        full_ipr = np.asarray(data["ipr"], dtype=float)
    with np.load(historical_dir / "selected_eigenvectors.npz", allow_pickle=False) as data:
        historical_indices = np.asarray(data["ascending_indices"], dtype=int)
    n = len(full_values)
    targets = {
        "q25_width_51": width_count_quantile_indices(n, 0.25, 51),
        "q75_width_51": width_count_quantile_indices(n, 0.75, 51),
    }
    if model_key == "affine_alpha_2p0":
        q50 = width_count_quantile_indices(n, 0.50, 51)
        missing = np.asarray([index for index in q50 if index not in set(historical_indices.tolist())], dtype=int)
        if len(missing) != 1:
            raise AssertionError("expected exactly one missing affine-alpha2 q50 vector")
        targets["q50_missing_only"] = missing
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    cache_id = stable_json_hash({
        "stage_version": BASELINE_SUPPLEMENT_VERSION,
        "source_configuration_id": context["source_configuration_identity"]["sha256"],
        "configuration_sha256": context["configuration_sha256"],
        "graph_sha256": sha256_file(graph_path),
        "targets": {name: members.tolist() for name, members in targets.items()},
        "solver": "scipy.linalg.eigh subset_by_index driver=evr with one guard per side",
    })
    if manifest_path.is_file():
        existing = read_json(manifest_path)
        if existing.get("status") == "COMPLETE" and existing.get("cache_id") == cache_id:
            return output_dir
        if existing.get("status") == "COMPLETE":
            raise ValueError("completed supplement has incompatible cache identity")
    manifest = {"status": "RUNNING", "stage": "targeted_baseline_vector_supplement", "stage_version": BASELINE_SUPPLEMENT_VERSION, "cache_id": cache_id, "model_key": model_key, "started_at_utc": utc_now()}
    atomic_write_json(manifest_path, manifest)
    try:
        dense = adjacency.toarray()
        recovered: dict[int, np.ndarray] = {}
        solve_rows: list[dict[str, object]] = []
        for name, members in targets.items():
            lo = max(0, int(np.min(members)) - 1)
            hi = min(n - 1, int(np.max(members)) + 1)
            solve_started = time.time()
            values, vectors = linalg.eigh(dense, subset_by_index=(lo, hi), check_finite=False, driver="evr")
            global_indices = np.arange(lo, hi + 1, dtype=int)
            maximum_match_error = float(np.max(np.abs(np.asarray(values) - full_values[global_indices])))
            if maximum_match_error > float(config["validation"]["selected_index_eigenvalue_tolerance"]):
                raise AssertionError("supplement eigenvalues do not match saved full spectrum")
            for index in members:
                recovered[int(index)] = np.asarray(vectors[:, int(index) - lo], dtype=float)
            solve_rows.append({
                "model_key": model_key, "target": name, "target_count": int(len(members)),
                "solved_first_index": lo, "solved_last_index": hi, "solved_count": int(hi - lo + 1),
                "maximum_saved_spectrum_match_error": maximum_match_error,
                "eigensolve_seconds": time.time() - solve_started,
            })
        selected_indices = np.asarray(sorted(recovered), dtype=int)
        selected_vectors = np.column_stack([recovered[int(index)] for index in selected_indices])
        selected_values = full_values[selected_indices]
        validation = validate_selected_eigenpairs(adjacency, selected_values, selected_vectors)
        if validation["maximum_relative_residual"] > float(config["validation"]["maximum_relative_residual"]):
            raise AssertionError("baseline supplement residual tolerance failed")
        maximum_mass = np.full(n, np.nan, dtype=float)
        maximum_mass[selected_indices] = np.max(np.square(np.abs(selected_vectors)), axis=0)
        atomic_savez(output_dir / "supplement_vectors.npz", ascending_indices=selected_indices, eigenvalues=selected_values, eigenvectors=selected_vectors)
        bundle = checkpoint_diagnostics(
            graph=graph, adjacency=adjacency, values=full_values, ipr=full_ipr,
            full_maximum_mass=maximum_mass, selected_indices=selected_indices, selected_vectors=selected_vectors,
            potential=None, amplitude=0.0,
            prefix={"source_kind": "targeted_pilot_baseline_supplement", "model_key": model_key, "n": n, "a_over_s_ref": 0.0, "a": 0.0, "potential_seed": -1},
            physical_half_width=0.1 * float(pilot["s_ref"]), atom_enabled=True,
            shell_max_radius=int(config["localization"]["shell_max_radius"]),
        )
        write_checkpoint_tables(output_dir, bundle)
        manifest.update({
            "status": "COMPLETE", "finished_at_utc": utc_now(), "runtime_seconds": time.time() - started,
            "source_configuration_id": context["source_configuration_identity"]["sha256"],
            "configuration_sha256": context["configuration_sha256"], "graph_metadata": graph_metadata,
            "graph_cache_sha256": sha256_file(graph_path), "target_windows": {name: members.tolist() for name, members in targets.items()},
            "selected_vector_count": int(len(selected_indices)), "solve_rows": solve_rows, "validation": validation,
            "heavy_tail_q50_policy": "no arbitrary q50 eigenvectors computed; corrected projector/subspace evidence reused" if model_key != "affine_alpha_2p0" else "one missing nondegenerate q50 vector recovered",
        })
        atomic_write_json(manifest_path, manifest)
        return output_dir
    except Exception as error:
        manifest.update({"status": "FAILED", "finished_at_utc": utc_now(), "runtime_seconds": time.time() - started, "failure_type": type(error).__name__, "failure_message": str(error)})
        atomic_write_json(manifest_path, manifest)
        raise
