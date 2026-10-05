"""Bounded random 3-regular Anderson benchmark at central energy."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Mapping

import networkx as nx
import numpy as np
from scipy import linalg, sparse

from small_world_mobility_edge.regular_baseline import generate_random_regular_graph

from .anderson_utils import (
    atomic_savez,
    atomic_write_csv,
    atomic_write_json,
    graph_identity,
    potential_identity,
    validate_selected_eigenpairs,
    verify_execution_context,
)
from .common import (
    deterministic_mass_order,
    graph_from_edge_cache,
    graph_to_csr,
    read_json,
    save_graph_cache,
    sha256_file,
    stable_json_hash,
    utc_now,
)
from .handoff_metrics import localization_metrics


RRG_BENCHMARK_VERSION = "anderson_random_3_regular_central_v2"


def _amplitude_slug(value: float) -> str:
    return ("%.12g" % float(value)).replace("-", "m").replace(".", "p")


def _rrg_task(config: Mapping[str, object], task_id: int) -> dict[str, int]:
    rrg = config["rrg"]
    sizes = [int(value) for value in rrg["n_values"]]
    reps = int(rrg["graph_replicates"])
    total = len(sizes) * reps
    if not 1 <= int(task_id) <= total:
        raise ValueError("RRG task ID outside configured range")
    zero = int(task_id) - 1
    n = sizes[zero // reps]
    rep = zero % reps + 1
    seed = int(rrg["graph_seed_table"][str(n)][rep - 1])
    return {"n": n, "graph_replicate": rep, "graph_seed": seed, "task_id": int(task_id)}


def _rrg_stream_task(config: Mapping[str, object], task_id: int) -> dict[str, int]:
    rrg = config["rrg"]
    total = len(rrg["n_values"]) * int(rrg["graph_replicates"]) * int(rrg["potential_replicates"])
    if not 1 <= int(task_id) <= total:
        raise ValueError("RRG refinement task ID outside configured range")
    zero = int(task_id) - 1
    potential_rep = zero % int(rrg["potential_replicates"]) + 1
    graph_task = zero // int(rrg["potential_replicates"]) + 1
    return {**_rrg_task(config, graph_task), "potential_replicate": potential_rep, "stream_task_id": int(task_id)}


def _generate_connected_rrg(n: int, degree: int, seed: int) -> tuple[nx.Graph, list[dict[str, object]]]:
    attempts: list[dict[str, object]] = []
    for attempt in range(100):
        attempt_seed = int(seed + attempt)
        graph = generate_random_regular_graph(n=n, degree=degree, seed=attempt_seed)
        connected = nx.is_connected(graph)
        attempts.append({"attempt": attempt + 1, "seed": attempt_seed, "connected": connected})
        if connected:
            if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
                raise AssertionError("RRG generator did not return a simple undirected graph")
            degrees = np.asarray([graph.degree(vertex) for vertex in range(n)], dtype=int)
            if not np.all(degrees == degree):
                raise AssertionError("RRG degree verification failed")
            if set(graph.nodes()) != set(range(n)):
                graph = nx.convert_node_labels_to_integers(graph, ordering="sorted")
            # The shared cache contract requires insertion order 0..N-1, not
            # merely that those labels are present. NetworkX may insert the
            # same integer labels in shuffled order.
            ordered_graph = nx.Graph()
            ordered_graph.add_nodes_from(range(n))
            ordered_graph.add_edges_from(
                sorted((min(int(u), int(v)), max(int(u), int(v))) for u, v in graph.edges())
            )
            return ordered_graph, attempts
    raise RuntimeError("failed to generate a connected random regular graph in 100 attempts")


def _central_interval_solve(dense: np.ndarray, *, target_half_width: float, initial_half_width: float) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    half_width = float(initial_half_width)
    solve_started = time.time()
    while True:
        values, vectors = linalg.eigh(
            dense,
            subset_by_value=(-half_width, half_width),
            check_finite=False,
            driver="evr",
        )
        values = np.asarray(values, dtype=float)
        vectors = np.asarray(vectors, dtype=float)
        target = np.flatnonzero(np.abs(values) <= target_half_width)
        lower_guard = np.flatnonzero(values < -target_half_width)
        upper_guard = np.flatnonzero(values > target_half_width)
        if len(target) and len(lower_guard) and len(upper_guard):
            break
        half_width *= 1.5
        if half_width > 2.0:
            raise RuntimeError("failed to obtain central interval guards")
    first = int(target[0])
    last = int(target[-1])
    selected_positions = np.arange(first - 1, last + 2, dtype=int)
    return values[selected_positions], vectors[:, selected_positions], {
        "solver_interval_low": -half_width,
        "solver_interval_high": half_width,
        "returned_interval_count": int(len(values)),
        "target_count": int(len(target)),
        "selected_with_guards_count": int(len(selected_positions)),
        "lower_guard_count": 1,
        "upper_guard_count": 1,
        "target_positions_in_selected": [1, int(len(selected_positions) - 2)],
        "interval_solver_contract": "scipy.linalg.eigh driver=evr subset_by_value returns all eigenpairs in (low,high]",
        "global_rank_claimed": False,
        "eigensolve_seconds": time.time() - solve_started,
    }


def _central_tables(
    values: np.ndarray,
    vectors: np.ndarray,
    *,
    target_half_width: float,
    prefix: Mapping[str, object],
    graph: nx.Graph,
    potential: np.ndarray | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    target_positions = np.flatnonzero(np.abs(values) <= target_half_width)
    rows: list[dict[str, object]] = []
    for position, value in enumerate(values):
        metrics = localization_metrics(vectors[:, position])
        center = int(metrics["dominant_vertex"])
        masses = np.square(np.abs(vectors[:, position]))
        row = {
            **prefix,
            "interval_position": position,
            "energy": float(value),
            "target_window_member": bool(position in set(target_positions.tolist())),
            "guard_level": "lower" if position == 0 else ("upper" if position == len(values) - 1 else "none"),
            "global_rank_available": False,
            "dominant_vertex_degree": int(graph.degree(center)),
            **metrics,
        }
        if potential is not None:
            row["center_potential"] = float(potential[center])
            row["weighted_potential_mean"] = float(np.dot(potential, masses))
        rows.append(row)
    gap_rows: list[dict[str, object]] = []
    for middle in target_positions:
        if middle <= 0 or middle >= len(values) - 1:
            raise AssertionError("target window is missing a guard level")
        left = float(values[middle] - values[middle - 1])
        right = float(values[middle + 1] - values[middle])
        denominator = max(left, right)
        gap_rows.append({
            **prefix,
            "middle_interval_position": int(middle),
            "left_energy": float(values[middle - 1]),
            "middle_energy": float(values[middle]),
            "right_energy": float(values[middle + 1]),
            "left_raw_gap": left,
            "right_raw_gap": right,
            "raw_gap_ratio": min(left, right) / denominator if denominator > 0.0 else math.nan,
            "triplet_is_consecutive_within_complete_physical_interval": True,
            "middle_level_belongs_to_target_window": True,
        })
    valid = np.asarray([row["raw_gap_ratio"] for row in gap_rows if math.isfinite(float(row["raw_gap_ratio"]))], dtype=float)
    target_rows = [rows[int(position)] for position in target_positions]
    summary = {
        **prefix,
        "target_level_count": int(len(target_positions)),
        "gap_triplet_count": int(len(gap_rows)),
        "mean_raw_gap_ratio": float(np.mean(valid)) if len(valid) else math.nan,
        "median_raw_gap_ratio": float(np.median(valid)) if len(valid) else math.nan,
        "median_ipr": float(np.median([float(row["ipr"]) for row in target_rows])) if target_rows else math.nan,
        "median_n_times_ipr": float(np.median([float(row["n_times_ipr"]) for row in target_rows])) if target_rows else math.nan,
        "median_participation_fraction": float(np.median([float(row["participation_fraction"]) for row in target_rows])) if target_rows else math.nan,
        "median_maximum_coordinate_mass": float(np.median([float(row["maximum_coordinate_mass"]) for row in target_rows])) if target_rows else math.nan,
        "median_support_fraction_90": float(np.median([float(row["support_fraction_90"]) for row in target_rows])) if target_rows else math.nan,
    }
    return rows, gap_rows, summary


def _save_rrg_checkpoint(
    *,
    graph: nx.Graph,
    adjacency: sparse.spmatrix,
    dense_operator: np.ndarray,
    output_dir: Path,
    prefix: Mapping[str, object],
    potential: np.ndarray | None,
    amplitude: float,
    context: Mapping[str, object],
    config: Mapping[str, object],
    graph_id: str,
    potential_id: str | None,
    full_reference: bool,
) -> dict[str, object]:
    rrg = config["rrg"]
    target_half_width = float(rrg["central_window"][1])
    initial_half_width = float(rrg["solver_interval_initial"][1])
    cache_id = stable_json_hash({
        "stage_version": RRG_BENCHMARK_VERSION,
        "source_configuration_id": context["source_configuration_identity"]["sha256"],
        "configuration_sha256": context["configuration_sha256"],
        "graph_id": graph_id,
        "potential_id": potential_id,
        "operator": "A+a*diag(xi)",
        "n": graph.number_of_nodes(),
        "amplitude": float(amplitude),
        "solver": rrg["solver"],
        "central_window": rrg["central_window"],
        "tolerances": config["validation"],
    })
    output_dir = Path(output_dir)
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.is_file():
        existing = read_json(manifest_path)
        if existing.get("status") == "COMPLETE" and existing.get("cache_id") == cache_id:
            return dict(existing)
        if existing.get("status") == "COMPLETE":
            raise ValueError("completed RRG checkpoint has incompatible cache identity")
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(manifest_path, {"status": "RUNNING", "cache_id": cache_id, **prefix, "started_at_utc": utc_now()})
    started = time.time()
    try:
        values, vectors, coverage = _central_interval_solve(dense_operator, target_half_width=target_half_width, initial_half_width=initial_half_width)
        operator = adjacency + sparse.diags(float(amplitude) * potential, offsets=0, format="csr") if potential is not None else adjacency
        validation = validate_selected_eigenpairs(operator, values, vectors)
        limits = config["validation"]
        if validation["maximum_relative_residual"] > float(limits["maximum_relative_residual"]) or validation["maximum_norm_error"] > float(limits["maximum_norm_error"]):
            raise AssertionError("RRG central eigenpair validation failed")
        full_reference_result: dict[str, object] = {"performed": False}
        if full_reference:
            full_values = np.asarray(linalg.eigh(dense_operator.copy(), eigvals_only=True, check_finite=False, driver="evd"), dtype=float)
            reference = full_values[np.abs(full_values) <= target_half_width]
            target = values[np.abs(values) <= target_half_width]
            error = float(np.max(np.abs(reference - target))) if len(reference) == len(target) and len(reference) else math.inf
            full_reference_result = {
                "performed": True,
                "full_spectrum_count": int(len(full_values)),
                "full_reference_target_count": int(len(reference)),
                "interval_target_count": int(len(target)),
                "maximum_eigenvalue_match_error": error,
                "coverage_exactly_matches_full_reference": bool(len(reference) == len(target) and error <= float(limits["selected_index_eigenvalue_tolerance"])),
            }
            if not full_reference_result["coverage_exactly_matches_full_reference"]:
                raise AssertionError("RRG interval coverage failed complete-reference comparison")
        rows, gap_rows, summary = _central_tables(values, vectors, target_half_width=target_half_width, prefix=prefix, graph=graph, potential=potential)
        atomic_savez(
            output_dir / "central_eigenpairs.npz",
            eigenvalues=values,
            eigenvectors=vectors,
            target_window_mask=np.abs(values) <= target_half_width,
            target_half_width=np.asarray(target_half_width),
        )
        atomic_write_csv(output_dir / "tables" / "central_eigenpair_metrics.csv.gz", rows)
        atomic_write_csv(output_dir / "tables" / "central_gap_triplets.csv.gz", gap_rows)
        atomic_write_csv(output_dir / "tables" / "central_summary.csv", [summary])
        manifest = {
            "status": "COMPLETE", "stage_version": RRG_BENCHMARK_VERSION, "cache_id": cache_id, **prefix,
            "source_configuration_id": context["source_configuration_identity"]["sha256"],
            "configuration_sha256": context["configuration_sha256"], "graph_id": graph_id, "potential_id": potential_id,
            "operator": "H(a)=A+a*diag(xi), unit hopping", "spectrum_scope": "partial central interval only",
            "global_rank_claimed": False, "coverage": coverage, "full_reference_validation": full_reference_result,
            "validation": validation, "summary": summary, "runtime_seconds": time.time() - started,
            "finished_at_utc": utc_now(),
        }
        atomic_write_json(manifest_path, manifest)
        return manifest
    except Exception as error:
        atomic_write_json(manifest_path, {"status": "FAILED", "cache_id": cache_id, **prefix, "failure_type": type(error).__name__, "failure_message": str(error), "finished_at_utc": utc_now()})
        raise


def run_rrg_task(
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
    rrg = config["rrg"]
    spec = _rrg_task(config, int(task_id))
    output_dir = Path(output_root) / ("N%d" % spec["n"]) / ("graph_rep_%d_seed_%d" % (spec["graph_replicate"], spec["graph_seed"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    manifest = {"status": "RUNNING", "stage": "rrg_benchmark", "stage_version": RRG_BENCHMARK_VERSION, **spec, "started_at_utc": utc_now(), "source_configuration_id": context["source_configuration_identity"]["sha256"], "configuration_sha256": context["configuration_sha256"]}
    atomic_write_json(manifest_path, manifest)
    try:
        graph_cache = output_dir / "graph" / "graph_realization.npz"
        attempts_path = output_dir / "graph" / "generation_attempts.csv"
        if graph_cache.is_file():
            graph, graph_metadata = graph_from_edge_cache(graph_cache)
            attempts = []
        else:
            graph, attempts = _generate_connected_rrg(spec["n"], int(rrg["degree"]), spec["graph_seed"])
            save_graph_cache(graph_cache, graph, np.arange(spec["n"]), {
                "ensemble": "simple random regular graph", "degree": int(rrg["degree"]),
                "requested_seed": spec["graph_seed"], "accepted_seed": attempts[-1]["seed"],
                "connectedness_conditioning": rrg["connectedness_policy"],
            })
            atomic_write_csv(attempts_path, attempts)
            graph, graph_metadata = graph_from_edge_cache(graph_cache)
        adjacency = graph_to_csr(graph, nodelist=range(spec["n"]))
        graph_id = graph_identity(graph_cache, operator="unit-hopping simple random 3-regular adjacency", source_id=str(context["source_configuration_identity"]["sha256"]))
        checkpoints: list[dict[str, object]] = []
        baseline_prefix = {"source_kind": "rrg_benchmark", **spec, "potential_replicate": 0, "potential_seed": -1, "a": 0.0, "W": 0.0}
        baseline = _save_rrg_checkpoint(
            graph=graph, adjacency=adjacency, dense_operator=adjacency.toarray(), output_dir=output_dir / "baseline",
            prefix=baseline_prefix, potential=None, amplitude=0.0, context=context, config=config,
            graph_id=graph_id, potential_id=None,
            full_reference=bool(spec["n"] == int(config["validation"]["rrg_full_reference_n"]) and spec["graph_replicate"] == 1),
        )
        checkpoints.append(baseline)
        for potential_rep in range(1, int(rrg["potential_replicates"]) + 1):
            seed = int(spec["graph_seed"]) + 100000000 + potential_rep
            xi = np.random.default_rng(seed).uniform(-1.0, 1.0, size=spec["n"])
            potential_id = potential_identity(xi, graph_id=graph_id, seed=seed)
            potential_dir = output_dir / ("potential_rep_%d_seed_%d" % (potential_rep, seed))
            atomic_savez(potential_dir / "paired_potential.npz", xi=xi, potential_seed=np.asarray(seed), potential_id=np.asarray(potential_id))
            for amplitude in [float(value) for value in rrg["amplitudes"] if float(value) > 0.0]:
                dense = adjacency.toarray()
                dense[np.diag_indices_from(dense)] += amplitude * xi
                prefix = {"source_kind": "rrg_benchmark", **spec, "potential_replicate": potential_rep, "potential_seed": seed, "a": amplitude, "W": 2.0 * amplitude}
                checkpoint = _save_rrg_checkpoint(
                    graph=graph, adjacency=adjacency, dense_operator=dense,
                    output_dir=potential_dir / ("a_" + _amplitude_slug(amplitude)), prefix=prefix,
                    potential=xi, amplitude=amplitude, context=context, config=config,
                    graph_id=graph_id, potential_id=potential_id,
                    full_reference=bool(spec["n"] == int(config["validation"]["rrg_full_reference_n"]) and spec["graph_replicate"] == 1 and potential_rep == 1 and amplitude == 10.0),
                )
                checkpoints.append(checkpoint)
        expected = 1 + int(rrg["potential_replicates"]) * (len(rrg["amplitudes"]) - 1)
        if len(checkpoints) != expected:
            raise AssertionError("unexpected RRG checkpoint count")
        manifest.update({
            "status": "COMPLETE", "finished_at_utc": utc_now(), "runtime_seconds": time.time() - started,
            "graph_id": graph_id, "graph_cache": str(graph_cache), "graph_cache_sha256": sha256_file(graph_cache),
            "graph_metadata": graph_metadata, "generation_attempts": attempts,
            "simplicity_verified": True, "degree_three_verified": True, "connected_verified": True,
            "checkpoint_count": len(checkpoints), "baseline_count": 1, "positive_disorder_count": len(checkpoints) - 1,
        })
        atomic_write_json(manifest_path, manifest)
        return output_dir
    except Exception as error:
        manifest.update({"status": "FAILED", "finished_at_utc": utc_now(), "runtime_seconds": time.time() - started, "failure_type": type(error).__name__, "failure_message": str(error)})
        atomic_write_json(manifest_path, manifest)
        raise


def run_rrg_a9_stream(
    *,
    project_root: Path,
    config_path: Path,
    freeze_manifest: Path,
    task_id: int,
    output_root: Path,
) -> Path:
    context = verify_execution_context(project_root, config_path, freeze_manifest)
    config = context["config"]
    spec = _rrg_stream_task(config, int(task_id))
    graph_dir = Path(output_root) / ("N%d" % spec["n"]) / ("graph_rep_%d_seed_%d" % (spec["graph_replicate"], spec["graph_seed"]))
    graph, _ = graph_from_edge_cache(graph_dir / "graph" / "graph_realization.npz")
    adjacency = graph_to_csr(graph, nodelist=range(spec["n"]))
    graph_id = graph_identity(graph_dir / "graph" / "graph_realization.npz", operator="unit-hopping simple random 3-regular adjacency", source_id=str(context["source_configuration_identity"]["sha256"]))
    seed = int(spec["graph_seed"]) + 100000000 + int(spec["potential_replicate"])
    potential_dir = graph_dir / ("potential_rep_%d_seed_%d" % (spec["potential_replicate"], seed))
    with np.load(potential_dir / "paired_potential.npz", allow_pickle=False) as data:
        xi = np.asarray(data["xi"], dtype=float)
        potential_id = str(data["potential_id"].item())
    amplitude = 9.0
    dense = adjacency.toarray()
    dense[np.diag_indices_from(dense)] += amplitude * xi
    prefix = {"source_kind": "rrg_benchmark_conditional_a9", **spec, "potential_seed": seed, "a": amplitude, "W": 18.0}
    _save_rrg_checkpoint(
        graph=graph, adjacency=adjacency, dense_operator=dense, output_dir=potential_dir / "a_9",
        prefix=prefix, potential=xi, amplitude=amplitude, context=context, config=config,
        graph_id=graph_id, potential_id=potential_id, full_reference=False,
    )
    return potential_dir / "a_9"
