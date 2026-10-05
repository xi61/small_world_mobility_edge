"""Two-part affine-PA localization and local-mode interaction experiment.

The implementation is intentionally separate from the historical PA pipelines.
It generates one clean affine graph for every configured ``(N, alpha)`` case,
stores the complete adjacency eigensystem, produces full-spectrum localization
and spacing diagnostics, and then performs explicitly a-posteriori local-mode
analyses for fixed adjacent eigenpairs.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import networkx as nx
import numpy as np
from scipy import linalg, sparse
from scipy.optimize import linear_sum_assignment
from scipy.sparse import linalg as sparse_linalg

from .baseline import (
    MODEL_AFFINE_PA,
    attachment_offset_for_degree_ccdf_alpha,
    expected_ba_edge_count,
    generate_preferential_attachment_graph,
)
from ..random_schrodinger.common import (
    BFSDistanceCache,
    graph_from_edge_cache,
    graph_to_csr,
    hill_ccdf_exponent,
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
from ..random_schrodinger.tunneling import schur_effective_operator


LINEAR_PA_TWO_PART_VERSION = "linear_pa_two_part_v3"


def alpha_slug(alpha: float) -> str:
    return ("%.8g" % float(alpha)).replace("-", "m").replace(".", "p")


def stable_graph_seed(base_seed: int, n: int, alpha_index: int, alpha: float) -> int:
    """Derive a reproducible 32-bit seed without Python's randomized hash."""
    alpha_integer = int(round(float(alpha) * 1_000_000.0))
    sequence = np.random.SeedSequence(
        [int(base_seed), int(n), int(alpha_index), alpha_integer]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def resolve_case(
    config: Mapping[str, object], *, n: int, alpha: float
) -> dict[str, object]:
    sizes = [int(value) for value in config["sizes"]]
    alphas = [float(value) for value in config["alpha_grid"]]
    if int(n) not in sizes:
        raise ValueError(f"N={n} is not configured")
    matches = [index for index, value in enumerate(alphas) if math.isclose(value, float(alpha), abs_tol=1e-12, rel_tol=0.0)]
    if len(matches) != 1:
        raise ValueError(f"alpha={alpha} is not uniquely configured")
    alpha_index = matches[0]
    m = int(config["model"]["m"])
    seed = stable_graph_seed(int(config["base_seed"]), int(n), alpha_index, float(alpha))
    return {
        "experiment_id": config["experiment_id"],
        "case_id": f"N_{int(n)}__alpha_{alpha_slug(alpha)}",
        "n": int(n),
        "m": m,
        "degree_ccdf_alpha": float(alpha),
        "attachment_offset": attachment_offset_for_degree_ccdf_alpha(m, float(alpha)),
        "alpha_index": alpha_index,
        "graph_seed": seed,
        "seed_derivation": "NumPy SeedSequence([base_seed,N,alpha_index,round(alpha*1e6)]).generate_state(uint32)",
        "expected_edge_count": expected_ba_edge_count(int(n), m),
        "operator": "unscaled simple undirected adjacency in vertex order 0..N-1",
        "realization_count": 1,
        "coupled_growth_across_sizes": False,
    }


def case_directory(output_root: Path, n: int, alpha: float) -> Path:
    return Path(output_root) / f"N_{int(n)}" / f"alpha_{alpha_slug(alpha)}"


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    write_json(temporary, payload)
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _edge_hash(graph: nx.Graph) -> str:
    edges = np.asarray(sorted((min(int(u), int(v)), max(int(u), int(v))) for u, v in graph.edges()), dtype=np.int64)
    digest = hashlib.sha256()
    digest.update(np.asarray([graph.number_of_nodes(), graph.number_of_edges()], dtype=np.int64).tobytes())
    digest.update(edges.tobytes(order="C"))
    return digest.hexdigest()


def _graph_summary(graph: nx.Graph, resolved: Mapping[str, object]) -> dict[str, object]:
    n = graph.number_of_nodes()
    degrees = np.fromiter((graph.degree(vertex) for vertex in range(n)), dtype=np.int64, count=n)
    components = sorted((len(part) for part in nx.connected_components(graph)), reverse=True)
    hill = hill_ccdf_exponent(degrees, tail_fraction=0.10)
    return {
        **resolved,
        "graph_sha256_edges": _edge_hash(graph),
        "num_nodes": n,
        "num_edges": graph.number_of_edges(),
        "realized_mean_degree": float(np.mean(degrees)),
        "minimum_degree": int(np.min(degrees)),
        "maximum_degree": int(np.max(degrees)),
        "degree_median": float(np.median(degrees)),
        "component_count": len(components),
        "largest_component_size": int(components[0]),
        "connected": bool(len(components) == 1),
        "self_loop_count": int(nx.number_of_selfloops(graph)),
        "simple_undirected": bool(type(graph) is nx.Graph and not graph.is_directed() and not graph.is_multigraph()),
        "vertex_order": "identity 0..N-1",
        "empirical_tail_note": "finite-graph Hill estimate is diagnostic and is not the nominal exponent",
        **hill,
    }


def _degree_rows(graph: nx.Graph, resolved: Mapping[str, object]) -> list[dict[str, object]]:
    degrees = np.asarray([graph.degree(vertex) for vertex in range(graph.number_of_nodes())], dtype=int)
    counts = np.bincount(degrees)
    survival = np.cumsum(counts[::-1])[::-1]
    return [
        {
            "case_id": resolved["case_id"],
            "degree": degree,
            "vertex_count": int(count),
            "ccdf_count_ge_degree": int(survival[degree]),
            "ccdf_fraction_ge_degree": float(survival[degree] / len(degrees)),
        }
        for degree, count in enumerate(counts)
        if count
    ]


def generate_or_load_graph(
    case_dir: Path, resolved: Mapping[str, object]
) -> tuple[nx.Graph, dict[str, object]]:
    case_dir = Path(case_dir)
    cache = case_dir / "arrays/graph_edges.npz"
    metadata_path = case_dir / "graph_cache_manifest.json"
    if cache.exists() and metadata_path.exists():
        graph, cached = graph_from_edge_cache(cache)
        metadata = read_json(metadata_path)
        if metadata.get("case_identity") != stable_json_hash(dict(resolved)):
            raise ValueError("existing graph cache has an incompatible case identity")
        if metadata.get("graph_sha256_edges") != _edge_hash(graph):
            raise ValueError("existing graph cache edge identity is inconsistent")
        return graph, metadata
    graph, diagnostics = generate_preferential_attachment_graph(
        n=int(resolved["n"]),
        m=int(resolved["m"]),
        seed=int(resolved["graph_seed"]),
        model=MODEL_AFFINE_PA,
        degree_ccdf_alpha=float(resolved["degree_ccdf_alpha"]),
    )
    summary = _graph_summary(graph, resolved)
    if graph.number_of_edges() != int(resolved["expected_edge_count"]):
        raise AssertionError("generated graph edge count differs from the resolved configuration")
    if summary["self_loop_count"] or not summary["simple_undirected"]:
        raise AssertionError("generated PA graph is not simple undirected")
    summary.update({
        "generator_diagnostics": diagnostics.to_dict(),
        "case_identity": stable_json_hash(dict(resolved)),
    })
    save_graph_cache(cache, graph, np.arange(graph.number_of_nodes()), summary)
    summary["graph_cache_path"] = str(cache)
    summary["graph_cache_sha256"] = sha256_file(cache)
    _atomic_json(metadata_path, summary)
    write_csv(case_dir / "tables/degree_distribution.csv", _degree_rows(graph, resolved))
    return graph, summary


def _orthogonality_sample_indices(n: int, sample_size: int) -> np.ndarray:
    count = min(int(sample_size), int(n))
    if count == n:
        return np.arange(n, dtype=int)
    fixed = {0, n - 1, n // 2}
    fixed.update(np.rint(np.linspace(0, n - 1, count)).astype(int).tolist())
    return np.asarray(sorted(fixed), dtype=int)


def _verify_eigensystem(
    operator: sparse.csr_matrix,
    values: np.ndarray,
    vectors: np.ndarray,
    *,
    residual_block_size: int,
    orthogonality_sample_size: int,
) -> dict[str, object]:
    n = len(values)
    absolute = np.empty(n, dtype=float)
    relative = np.empty(n, dtype=float)
    norms = np.sum(np.square(vectors), axis=0)
    for start in range(0, n, int(residual_block_size)):
        stop = min(n, start + int(residual_block_size))
        block = vectors[:, start:stop]
        residual = operator.dot(block) - block * values[None, start:stop]
        absolute[start:stop] = np.linalg.norm(residual, axis=0)
        relative[start:stop] = absolute[start:stop] / np.maximum(1.0, np.abs(values[start:stop]))
    sample = _orthogonality_sample_indices(n, orthogonality_sample_size)
    sample_gram = vectors[:, sample].T @ vectors[:, sample]
    orthogonality_error = sample_gram - np.eye(len(sample))
    return {
        "maximum_absolute_residual": float(np.max(absolute)),
        "maximum_relative_residual": float(np.max(relative)),
        "median_relative_residual": float(np.median(relative)),
        "maximum_column_norm_error": float(np.max(np.abs(norms - 1.0))),
        "orthogonality_check": "deterministic full-row Gram check on a memory-conscious column subset",
        "orthogonality_sample_size": int(len(sample)),
        "orthogonality_sample_indices": sample.tolist(),
        "orthogonality_sample_maximum_absolute_error": float(np.max(np.abs(orthogonality_error))),
        "orthogonality_sample_frobenius_error": float(np.linalg.norm(orthogonality_error)),
        "relative_residuals": relative,
        "absolute_residuals": absolute,
    }


def compute_or_load_full_spectrum(
    case_dir: Path,
    graph: nx.Graph,
    resolved: Mapping[str, object],
    solver_config: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    case_dir = Path(case_dir)
    values_path = case_dir / "arrays/full_eigenvalues_ascending.npy"
    vectors_path = case_dir / "arrays/full_eigenvectors_ascending.npy"
    manifest_path = case_dir / "spectral_cache_manifest.json"
    identity = stable_json_hash({
        "version": LINEAR_PA_TWO_PART_VERSION,
        "case": dict(resolved),
        "graph_sha256_edges": _edge_hash(graph),
        "solver": dict(solver_config),
    })
    if values_path.exists() and vectors_path.exists() and manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("spectral_cache_identity") != identity:
            raise ValueError("existing eigensystem cache has an incompatible identity")
        values = np.load(values_path, mmap_mode="r")
        vectors = np.load(vectors_path, mmap_mode="r")
        if values.shape != (graph.number_of_nodes(),) or vectors.shape != (graph.number_of_nodes(), graph.number_of_nodes()):
            raise ValueError("cached full eigensystem has incompatible dimensions")
        return np.asarray(values), np.asarray(vectors), manifest
    operator = graph_to_csr(graph, nodelist=range(graph.number_of_nodes()))
    dense = operator.toarray(order="F")
    started = time.time()
    values, vectors = linalg.eigh(
        dense,
        overwrite_a=True,
        check_finite=False,
        driver=str(solver_config.get("dense_driver", "evd")),
    )
    elapsed = time.time() - started
    del dense
    verification = _verify_eigensystem(
        operator,
        values,
        vectors,
        residual_block_size=int(solver_config["residual_block_size"]),
        orthogonality_sample_size=int(solver_config["orthogonality_sample_size"]),
    )
    tolerance = float(solver_config["relative_eigenpair_residual_tolerance"])
    if verification["maximum_relative_residual"] > tolerance:
        raise RuntimeError("full eigensystem failed the configured residual tolerance")
    values_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(values_path, np.asarray(values, dtype=np.float64))
    np.save(vectors_path, np.asarray(vectors, dtype=np.float64))
    np.save(case_dir / "arrays/full_absolute_residuals.npy", verification.pop("absolute_residuals"))
    np.save(case_dir / "arrays/full_relative_residuals.npy", verification.pop("relative_residuals"))
    manifest = {
        "status": "COMPLETE",
        "spectral_cache_identity": identity,
        "case_id": resolved["case_id"],
        "graph_sha256_edges": _edge_hash(graph),
        "operator": resolved["operator"],
        "eigenvalue_order": "ascending array order",
        "eigenvector_columns": "column j corresponds to ascending eigenvalue j and vertex rows 0..N-1",
        "matrix_normalization": "none",
        "perron_handling": "retained and explicitly labelled rank 1 in decreasing algebraic order",
        "dense_eigensolver": f"scipy.linalg.eigh driver={solver_config.get('dense_driver', 'evd')}",
        "eigendecomposition_runtime_seconds": elapsed,
        "values_path": str(values_path),
        "vectors_path": str(vectors_path),
        "values_sha256": sha256_file(values_path),
        "vectors_sha256": sha256_file(vectors_path),
        **verification,
    }
    _atomic_json(manifest_path, manifest)
    return values, vectors, manifest


def _multiplicity_scale(values: np.ndarray, factor: float) -> float:
    return float(factor * np.finfo(float).eps * max(1, len(values)) * max(1.0, float(np.max(np.abs(values)))))


def _localization_rows(
    values: np.ndarray,
    vectors: np.ndarray,
    relative_residuals: np.ndarray,
    *,
    case_prefix: Mapping[str, object],
    support_fractions: Sequence[float],
    block_size: int = 64,
) -> list[dict[str, object]]:
    n = len(values)
    rows: list[dict[str, object]] = []
    for start in range(0, n, int(block_size)):
        stop = min(n, start + int(block_size))
        block = np.asarray(vectors[:, start:stop], dtype=float)
        masses = np.square(np.abs(block))
        ipr = np.sum(np.square(masses), axis=0)
        maximum_vertices = np.argmax(masses, axis=0)
        sorted_masses = np.sort(masses, axis=0)[::-1, :]
        cumulative = np.cumsum(sorted_masses, axis=0)
        support_sizes = {
            float(q): np.sum(cumulative < float(q), axis=0) + 1 for q in support_fractions
        }
        for column, ascending_index in enumerate(range(start, stop)):
            row: dict[str, object] = {
                **case_prefix,
                "ascending_index": ascending_index,
                "ascending_rank_1based": ascending_index + 1,
                "decreasing_algebraic_rank": n - ascending_index,
                "energy": float(values[ascending_index]),
                "perron": bool(ascending_index == n - 1),
                "ipr": float(ipr[column]),
                "n_times_ipr": float(n * ipr[column]),
                "maximum_coordinate_vertex": int(maximum_vertices[column]),
                "maximum_coordinate_mass": float(masses[maximum_vertices[column], column]),
                "maximum_absolute_coordinate": float(math.sqrt(masses[maximum_vertices[column], column])),
                "relative_eigenpair_residual": float(relative_residuals[ascending_index]),
            }
            for fraction in support_fractions:
                label = int(round(100 * float(fraction)))
                size = int(support_sizes[float(fraction)][column])
                row[f"support_size_{label}"] = size
                row[f"support_fraction_{label}"] = float(size / n)
            rows.append(row)
    return rows


def _spacing_rows(
    values: np.ndarray,
    *,
    case_prefix: Mapping[str, object],
    tolerance_factor: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], float, np.ndarray]:
    descending = np.asarray(values[::-1], dtype=float)
    n = len(descending)
    resolution = _multiplicity_scale(descending, tolerance_factor)
    zero_mask = np.abs(descending) <= resolution
    gaps = descending[:-1] - descending[1:]
    gap_rows: list[dict[str, object]] = []
    for start, gap in enumerate(gaps, start=1):
        gap_rows.append({
            **case_prefix,
            "pair_start_decreasing_rank": start,
            "left_rank": start,
            "right_rank": start + 1,
            "left_energy": float(descending[start - 1]),
            "right_energy": float(descending[start]),
            "raw_gap": float(gap),
            "numerically_resolved": bool(gap > resolution),
            "touches_zero_atom": bool(zero_mask[start - 1] or zero_mask[start]),
            "multiplicity_resolution": resolution,
        })
    ratio_rows: list[dict[str, object]] = []
    for middle_rank in range(2, n):
        left_gap = float(gaps[middle_rank - 2])
        right_gap = float(gaps[middle_rank - 1])
        denominator = max(left_gap, right_gap)
        raw_ratio = min(left_gap, right_gap) / denominator if denominator > 0.0 else math.nan
        unresolved = bool(left_gap <= resolution or right_gap <= resolution)
        atom = bool(np.any(zero_mask[middle_rank - 2:middle_rank + 1]))
        ratio_rows.append({
            **case_prefix,
            "middle_decreasing_rank": middle_rank,
            "left_gap": left_gap,
            "right_gap": right_gap,
            "raw_gap_ratio_all_levels": raw_ratio,
            "multiplicity_or_resolution_excluded": unresolved,
            "zero_atom_triplet_excluded": atom,
            "filtered_gap_ratio": raw_ratio if not unresolved and not atom else math.nan,
            "triplet_first_rank": middle_rank - 1,
            "triplet_last_rank": middle_rank + 1,
        })
    return gap_rows, ratio_rows, resolution, zero_mask


def quantile_rank_windows(n: int, edges: Sequence[float]) -> list[dict[str, object]]:
    boundaries = sorted(set([0, n] + [min(n, max(0, int(math.floor(float(q) * n)))) for q in edges]))
    rows: list[dict[str, object]] = []
    for window_id, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        if right <= left:
            continue
        rows.append({
            "window_id": f"qrank_{window_id:02d}",
            "quantile_left": float(left / n),
            "quantile_right": float(right / n),
            "first_decreasing_rank": left + 1,
            "last_decreasing_rank": right,
            "state_count": right - left,
        })
    return rows


def _finite(values: Iterable[object]) -> np.ndarray:
    array = np.asarray([float(value) for value in values], dtype=float)
    return array[np.isfinite(array)]


def _window_summary_and_gaps(
    *,
    window: Mapping[str, object],
    localization_descending: Sequence[Mapping[str, object]],
    gap_rows: Sequence[Mapping[str, object]],
    ratio_rows: Sequence[Mapping[str, object]],
    case_prefix: Mapping[str, object],
    family: str,
    perron_policy: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    first_rank = int(window["first_decreasing_rank"])
    last_rank = int(window["last_decreasing_rank"])
    vectors = list(localization_descending[first_rank - 1:last_rank])
    internal_gaps = [row for row in gap_rows if first_rank <= int(row["pair_start_decreasing_rank"]) < last_rank]
    internal_ratios = [row for row in ratio_rows if first_rank + 1 <= int(row["middle_decreasing_rank"]) <= last_rank - 1]
    valid_all = _finite(row["raw_gap_ratio_all_levels"] for row in internal_ratios)
    valid_filtered = _finite(row["filtered_gap_ratio"] for row in internal_ratios)
    positive_resolved = np.asarray([
        float(row["raw_gap"]) for row in internal_gaps
        if bool(row["numerically_resolved"]) and not bool(row["touches_zero_atom"])
    ])
    gap_mean = float(np.mean(positive_resolved)) if len(positive_resolved) else math.nan
    gap_export: list[dict[str, object]] = []
    for row in internal_gaps:
        gap_export.append({
            **case_prefix,
            "window_family": family,
            "window_id": window["window_id"],
            "perron_policy": perron_policy,
            **{key: value for key, value in row.items() if key not in case_prefix},
            "window_mean_resolved_raw_gap": gap_mean,
            "window_mean_rescaled_gap": float(row["raw_gap"]) / gap_mean if gap_mean > 0.0 else math.nan,
            "rescaling_label": "division by this single-window mean; not ensemble unfolding",
        })
    result = {
        **case_prefix,
        "window_family": family,
        "window_id": window["window_id"],
        "perron_policy": perron_policy,
        **dict(window),
        "actual_count": len(vectors),
        "first_energy": float(vectors[0]["energy"]) if vectors else math.nan,
        "last_energy": float(vectors[-1]["energy"]) if vectors else math.nan,
        "internal_gap_count": len(internal_gaps),
        "raw_ratio_count_all_levels": int(len(valid_all)),
        "valid_ratio_count_multiplicity_and_atom_excluded": int(len(valid_filtered)),
        "mean_gap_ratio_all_levels": float(np.mean(valid_all)) if len(valid_all) else math.nan,
        "median_gap_ratio_all_levels": float(np.median(valid_all)) if len(valid_all) else math.nan,
        "mean_gap_ratio_filtered": float(np.mean(valid_filtered)) if len(valid_filtered) else math.nan,
        "median_gap_ratio_filtered": float(np.median(valid_filtered)) if len(valid_filtered) else math.nan,
        "mean_raw_resolved_gap": gap_mean,
        "mean_n_times_ipr": float(np.mean([float(row["n_times_ipr"]) for row in vectors])) if vectors else math.nan,
        "median_n_times_ipr": float(np.median([float(row["n_times_ipr"]) for row in vectors])) if vectors else math.nan,
        "mean_maximum_coordinate_mass": float(np.mean([float(row["maximum_coordinate_mass"]) for row in vectors])) if vectors else math.nan,
        "mean_support_size_80": float(np.mean([float(row["support_size_80"]) for row in vectors])) if vectors else math.nan,
        "histogram_status": "sufficient" if len(positive_resolved) >= 10 else "insufficient_observations_retained_in_table",
    }
    return result, gap_export


def _cumulative_and_disjoint_windows(n: int, counts: Sequence[int]) -> list[tuple[str, str, dict[str, object]]]:
    result: list[tuple[str, str, dict[str, object]]] = []
    clipped = [min(int(count), n) for count in counts]
    previous_literal = 0
    previous_nonperron = 0
    for requested, literal_count in zip(counts, clipped):
        result.append(("cumulative_literal", "including_Perron", {
            "window_id": f"literal_top_{requested}",
            "requested_count": int(requested),
            "first_decreasing_rank": 1,
            "last_decreasing_rank": literal_count,
            "state_count": literal_count,
            "clipped": bool(literal_count != int(requested)),
            "whole_spectrum": bool(literal_count == n),
        }))
        nonperron_count = min(int(requested), n - 1)
        result.append(("cumulative_nonperron", "excluding_Perron", {
            "window_id": f"nonperron_top_{requested}",
            "requested_count": int(requested),
            "first_decreasing_rank": 2,
            "last_decreasing_rank": 1 + nonperron_count,
            "state_count": nonperron_count,
            "clipped": bool(nonperron_count != int(requested)),
            "whole_nonperron_spectrum": bool(nonperron_count == n - 1),
        }))
        if literal_count > previous_literal:
            result.append(("disjoint_literal", "including_Perron_only_if_first_band", {
                "window_id": f"literal_band_{previous_literal + 1}_{literal_count}",
                "requested_count": literal_count - previous_literal,
                "first_decreasing_rank": previous_literal + 1,
                "last_decreasing_rank": literal_count,
                "state_count": literal_count - previous_literal,
                "clipped": False,
            }))
        if nonperron_count > previous_nonperron:
            result.append(("disjoint_nonperron", "excluding_Perron", {
                "window_id": f"nonperron_band_{previous_nonperron + 1}_{nonperron_count}",
                "requested_count": nonperron_count - previous_nonperron,
                "first_decreasing_rank": 2 + previous_nonperron,
                "last_decreasing_rank": 1 + nonperron_count,
                "state_count": nonperron_count - previous_nonperron,
                "clipped": False,
            }))
        previous_literal = literal_count
        previous_nonperron = nonperron_count
    return result


def _coordinate_and_geometry_tables(
    graph: nx.Graph,
    values: np.ndarray,
    vectors: np.ndarray,
    ranks: Sequence[int],
    coordinate_counts: Sequence[int],
    case_prefix: Mapping[str, object],
) -> dict[str, list[dict[str, object]]]:
    n = graph.number_of_nodes()
    distance_cache = BFSDistanceCache(graph)
    coordinate_rows: list[dict[str, object]] = []
    cumulative_rows: list[dict[str, object]] = []
    vertex_rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    pair_summary_rows: list[dict[str, object]] = []
    cross_summary_rows: list[dict[str, object]] = []
    cross_pair_rows: list[dict[str, object]] = []
    data: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for rank in sorted(set(int(value) for value in ranks if 1 <= int(value) <= n)):
        index = n - rank
        vector = np.asarray(vectors[:, index], dtype=float)
        masses = np.square(np.abs(vector))
        order = np.lexsort((np.arange(n), -masses))
        cumulative = np.cumsum(masses[order])
        prefix = {**case_prefix, "decreasing_algebraic_rank": rank, "ascending_index": index, "energy": float(values[index]), "perron": bool(rank == 1)}
        for coordinate_rank, vertex in enumerate(order[: min(50, n)], start=1):
            coordinate_rows.append({
                **prefix,
                "coordinate_rank": coordinate_rank,
                "vertex_id": int(vertex),
                "coordinate_mass": float(masses[vertex]),
                "absolute_coordinate": float(abs(vector[vertex])),
                "original_graph_degree": int(graph.degree(int(vertex))),
                "cumulative_mass": float(cumulative[coordinate_rank - 1]),
            })
        for count in coordinate_counts:
            actual = min(int(count), n)
            cumulative_rows.append({
                **prefix,
                "requested_coordinate_count": int(count),
                "actual_coordinate_count": actual,
                "cumulative_mass": float(cumulative[actual - 1]) if actual else 0.0,
            })
        top = np.asarray(order[: min(10, n)], dtype=int)
        top_masses = masses[top]
        induced = graph.subgraph(top.tolist()).copy()
        components = sorted(nx.connected_components(induced), key=lambda part: (-len(part), min(part)))
        component_by_vertex = {int(vertex): component_id for component_id, part in enumerate(components) for vertex in part}
        for support_rank, vertex in enumerate(top, start=1):
            vertex_rows.append({
                **prefix,
                "support_rank": support_rank,
                "vertex_id": int(vertex),
                "coordinate_mass": float(top_masses[support_rank - 1]),
                "full_graph_degree": int(graph.degree(int(vertex))),
                "induced_subgraph_degree": int(induced.degree(int(vertex))),
                "induced_component_id": int(component_by_vertex[int(vertex)]),
                "induced_component_count": len(components),
                "induced_edge_count": induced.number_of_edges(),
                "induced_density": float(nx.density(induced)) if len(top) >= 2 else math.nan,
            })
        for left, right in sorted((min(int(u), int(v)), max(int(u), int(v))) for u, v in induced.edges()):
            edge_rows.append({**prefix, "left_vertex": left, "right_vertex": right})
        full_matrix = distance_cache.matrix(top)
        induced_lengths = dict(nx.all_pairs_shortest_path_length(induced))
        rank_pair_rows: list[dict[str, object]] = []
        for left_index in range(len(top)):
            for right_index in range(left_index + 1, len(top)):
                left = int(top[left_index])
                right = int(top[right_index])
                induced_distance = float(induced_lengths.get(left, {}).get(right, math.inf))
                full_distance = float(full_matrix[left_index, right_index])
                rank_pair_rows.append({
                    **prefix,
                    "left_support_rank": left_index + 1,
                    "right_support_rank": right_index + 1,
                    "left_vertex": left,
                    "right_vertex": right,
                    "left_mass": float(top_masses[left_index]),
                    "right_mass": float(top_masses[right_index]),
                    "induced_subgraph_distance": induced_distance,
                    "induced_unreachable": not math.isfinite(induced_distance),
                    "full_original_graph_distance": full_distance,
                    "full_graph_unreachable": not math.isfinite(full_distance),
                })
        pair_rows.extend(rank_pair_rows)
        induced_finite = np.asarray([
            float(row["induced_subgraph_distance"])
            for row in rank_pair_rows
            if math.isfinite(float(row["induced_subgraph_distance"]))
        ])
        full_finite = np.asarray([
            float(row["full_original_graph_distance"])
            for row in rank_pair_rows
            if math.isfinite(float(row["full_original_graph_distance"]))
        ])
        pair_summary_rows.append({
            **prefix,
            "pair_count": len(rank_pair_rows),
            "induced_finite_pair_count": len(induced_finite),
            "induced_unreachable_pair_count": len(rank_pair_rows) - len(induced_finite),
            "induced_unreachable_fraction": float(1.0 - len(induced_finite) / len(rank_pair_rows)) if rank_pair_rows else 0.0,
            "induced_finite_distance_mean": float(np.mean(induced_finite)) if len(induced_finite) else math.nan,
            "induced_finite_distance_median": float(np.median(induced_finite)) if len(induced_finite) else math.nan,
            "induced_finite_distance_maximum": float(np.max(induced_finite)) if len(induced_finite) else math.nan,
            "full_graph_finite_pair_count": len(full_finite),
            "full_graph_unreachable_pair_count": len(rank_pair_rows) - len(full_finite),
            "full_graph_unreachable_fraction": float(1.0 - len(full_finite) / len(rank_pair_rows)) if rank_pair_rows else 0.0,
            "full_graph_finite_distance_mean": float(np.mean(full_finite)) if len(full_finite) else math.nan,
            "full_graph_finite_distance_median": float(np.median(full_finite)) if len(full_finite) else math.nan,
            "full_graph_finite_distance_maximum": float(np.max(full_finite)) if len(full_finite) else math.nan,
        })
        data[rank] = (vector, top, top_masses)
    selected_ranks = sorted(data)
    for position, left_rank in enumerate(selected_ranks):
        left_vector, left_top, left_top_mass = data[left_rank]
        left_mass_all = np.square(np.abs(left_vector))
        for right_rank in selected_ranks[position + 1:]:
            right_vector, right_top, right_top_mass = data[right_rank]
            right_mass_all = np.square(np.abs(right_vector))
            matrix = distance_cache.matrix(left_top, right_top)
            finite = matrix[np.isfinite(matrix)]
            numerator = 0.0
            denominator = 0.0
            for left_index, left_vertex in enumerate(left_top):
                for right_index, right_vertex in enumerate(right_top):
                    distance = float(matrix[left_index, right_index])
                    weight = float(left_top_mass[left_index] * right_top_mass[right_index])
                    cross_pair_rows.append({
                        **case_prefix,
                        "left_decreasing_rank": left_rank,
                        "right_decreasing_rank": right_rank,
                        "left_support_rank": left_index + 1,
                        "right_support_rank": right_index + 1,
                        "left_vertex": int(left_vertex),
                        "right_vertex": int(right_vertex),
                        "distance_in_original_graph": distance,
                        "unreachable": not math.isfinite(distance),
                        "conditional_mass_weight": weight,
                    })
                    if math.isfinite(distance):
                        numerator += weight * distance
                        denominator += weight
            overlap = len(set(left_top.tolist()) & set(right_top.tolist()))
            dominant_distance = distance_cache.distance(int(left_top[0]), int(right_top[0]))
            cross_summary_rows.append({
                **case_prefix,
                "left_decreasing_rank": left_rank,
                "right_decreasing_rank": right_rank,
                "top10_overlap_vertex_count": overlap,
                "whole_vector_mass_overlap": float(np.sum(np.minimum(left_mass_all, right_mass_all))),
                "structural_overlap_coefficient": float(overlap / 10.0),
                "dominant_vertex_distance": dominant_distance,
                "cross_pair_count": int(matrix.size),
                "cross_finite_pair_count": int(len(finite)),
                "cross_unreachable_count": int(matrix.size - len(finite)),
                "cross_minimum_finite_distance": float(np.min(finite)) if len(finite) else math.nan,
                "cross_median_finite_distance": float(np.median(finite)) if len(finite) else math.nan,
                "cross_maximum_finite_distance": float(np.max(finite)) if len(finite) else math.nan,
                "conditional_mass_weighted_mean_distance": numerator / denominator if denominator > 0.0 else math.nan,
                "conditional_mass_weight_denominator": denominator,
            })
    return {
        "coordinate_rows": coordinate_rows,
        "cumulative_rows": cumulative_rows,
        "induced_vertex_rows": vertex_rows,
        "induced_edge_rows": edge_rows,
        "induced_pair_distance_rows": pair_rows,
        "induced_pair_distance_summary_rows": pair_summary_rows,
        "cross_summary_rows": cross_summary_rows,
        "cross_pair_rows": cross_pair_rows,
    }


def select_part2_targets(
    n: int,
    quantile_windows: Sequence[Mapping[str, object]],
    cumulative_counts: Sequence[int],
    config: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    first_rank = int(config["primary_nonperron_first_rank"])
    last_rank = min(int(config["primary_nonperron_last_rank"]), n)
    roles: dict[int, set[str]] = {}
    coverage: list[dict[str, object]] = []
    for start in range(first_rank, last_rank):
        roles.setdefault(start, set()).add("exhaustive_first_50_nonperron")
    coverage.append({
        "selection_family": "primary_exhaustive_nonperron",
        "window_id": "ranks_2_51",
        "eligible_pair_count": max(0, last_rank - first_rank),
        "selected_pair_count": max(0, last_rank - first_rank),
        "first_pair_start_rank": first_rank,
        "last_pair_start_rank": last_rank - 1,
    })
    samples = int(config["samples_per_remaining_band_or_window"])
    clipped_counts = [min(int(value), n - 1) for value in cumulative_counts]
    previous = min(50, n - 1)
    for endpoint in clipped_counts:
        if endpoint <= previous:
            continue
        band_first_rank = previous + 2
        band_last_rank = endpoint + 1
        eligible = np.arange(band_first_rank, band_last_rank, dtype=int)
        chosen = evenly_spaced_integer_sample(eligible, samples)
        for start in chosen:
            roles.setdefault(int(start), set()).add(f"stratified_nonperron_band_{previous + 1}_{endpoint}")
        coverage.append({
            "selection_family": "remaining_disjoint_upper_nonperron_band",
            "window_id": f"nonperron_band_{previous + 1}_{endpoint}",
            "eligible_pair_count": int(len(eligible)),
            "selected_pair_count": int(len(chosen)),
            "first_pair_start_rank": int(eligible[0]) if len(eligible) else math.nan,
            "last_pair_start_rank": int(eligible[-1]) if len(eligible) else math.nan,
        })
        previous = endpoint
    for window in quantile_windows:
        lo = max(2, int(window["first_decreasing_rank"]))
        hi = min(n - 1, int(window["last_decreasing_rank"]) - 1)
        eligible = np.arange(lo, hi + 1, dtype=int) if hi >= lo else np.empty(0, dtype=int)
        chosen = evenly_spaced_integer_sample(eligible, samples)
        role = f"global_{window['window_id']}"
        for start in chosen:
            roles.setdefault(int(start), set()).add(role)
        coverage.append({
            "selection_family": "global_quantile_window_stratified",
            "window_id": window["window_id"],
            "eligible_pair_count": int(len(eligible)),
            "selected_pair_count": int(len(chosen)),
            "first_pair_start_rank": int(eligible[0]) if len(eligible) else math.nan,
            "last_pair_start_rank": int(eligible[-1]) if len(eligible) else math.nan,
            "empty_or_tiny": bool(len(eligible) < 1),
        })
    targets = [
        {
            "target_id": f"pair_rank_{start:05d}_{start + 1:05d}",
            "pair_start_decreasing_rank": start,
            "left_decreasing_rank": start,
            "right_decreasing_rank": start + 1,
            "selection_roles": sorted(role_set),
            "selection_used_coupling_or_model_success": False,
            "perron_in_pair": bool(start == 1),
        }
        for start, role_set in sorted(roles.items())
    ]
    return targets, coverage


def evenly_spaced_integer_sample(values: Sequence[int], maximum: int) -> np.ndarray:
    values = np.asarray(values, dtype=int)
    if len(values) <= int(maximum):
        return values.copy()
    positions = np.rint(np.linspace(0, len(values) - 1, int(maximum))).astype(int)
    return values[np.unique(positions)]


def run_part1(
    *,
    project_root: Path,
    config: Mapping[str, object],
    n: int,
    alpha: float,
    output_root: Path,
    freeze_manifest: Path,
) -> Path:
    resolved = resolve_case(config, n=n, alpha=alpha)
    case_dir = case_directory(output_root, n, alpha)
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "arrays").mkdir(exist_ok=True)
    (case_dir / "tables").mkdir(exist_ok=True)
    freeze = read_json(freeze_manifest)
    if freeze.get("status") != "COMPLETE":
        raise ValueError("source freeze is not complete")
    stage_identity = stable_json_hash({
        "version": LINEAR_PA_TWO_PART_VERSION,
        "stage": "part1",
        "source_configuration_id": freeze.get("source_configuration_id"),
        "resolved_case": resolved,
        "config": dict(config),
    })
    manifest_path = case_dir / "part1_manifest.json"
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if existing.get("status") == "COMPLETE" and existing.get("stage_identity") == stage_identity:
            return case_dir
        if existing.get("stage_identity") not in (None, stage_identity):
            raise ValueError("existing Part 1 output has an incompatible cache identity")
    manifest = {
        **runtime_metadata(project_root),
        "status": "RUNNING",
        "stage": "linear_pa_part1_full_spectrum_localization_spacing_geometry",
        "stage_identity": stage_identity,
        "source_configuration_id": freeze.get("source_configuration_id"),
        "resolved_case": resolved,
        "restart_policy": "same identity reuses verified graph/eigensystem caches; incompatible identity is rejected",
    }
    _atomic_json(manifest_path, manifest)
    started = time.time()
    try:
        _atomic_json(case_dir / "resolved_case.json", resolved)
        graph, graph_summary = generate_or_load_graph(case_dir, resolved)
        values, vectors, spectral = compute_or_load_full_spectrum(case_dir, graph, resolved, config["solver"])
        relative_residuals = np.load(case_dir / "arrays/full_relative_residuals.npy")
        prefix = {"case_id": resolved["case_id"], "n": n, "degree_ccdf_alpha": float(alpha), "graph_seed": resolved["graph_seed"], "graph_sha256_edges": graph_summary["graph_sha256_edges"]}
        localization = _localization_rows(
            values,
            vectors,
            relative_residuals,
            case_prefix=prefix,
            support_fractions=config["part1"]["support_mass_fractions"],
        )
        localization_descending = list(reversed(localization))
        write_csv(case_dir / "tables/full_spectrum_localization.csv.gz", localization)
        gaps, ratios, resolution, zero_mask = _spacing_rows(values, case_prefix=prefix, tolerance_factor=float(config["solver"]["multiplicity_tolerance_factor"]))
        write_csv(case_dir / "tables/full_spectrum_raw_gaps.csv.gz", gaps)
        write_csv(case_dir / "tables/full_spectrum_gap_ratios.csv.gz", ratios)
        quantile_windows = quantile_rank_windows(n, config["part1"]["quantile_edges"])
        window_summaries: list[dict[str, object]] = []
        window_gap_rows: list[dict[str, object]] = []
        for window in quantile_windows:
            summary, rows = _window_summary_and_gaps(
                window=window,
                localization_descending=localization_descending,
                gap_rows=gaps,
                ratio_rows=ratios,
                case_prefix=prefix,
                family="global_quantile_rank_band",
                perron_policy="literal rank band; Perron present only when rank 1 is included",
            )
            window_summaries.append(summary)
            window_gap_rows.extend(rows)
        counts = config["part1"]["cumulative_upper_counts"][str(n)]
        for family, perron_policy, window in _cumulative_and_disjoint_windows(n, counts):
            summary, rows = _window_summary_and_gaps(
                window=window,
                localization_descending=localization_descending,
                gap_rows=gaps,
                ratio_rows=ratios,
                case_prefix=prefix,
                family=family,
                perron_policy=perron_policy,
            )
            window_summaries.append(summary)
            window_gap_rows.extend(rows)
        write_csv(case_dir / "tables/energy_and_rank_window_summary.csv.gz", window_summaries)
        write_csv(case_dir / "tables/window_gap_observations.csv.gz", window_gap_rows)
        first_ten_gap_rows: list[dict[str, object]] = []
        for role, first_rank in (("literal_edge_ten_gaps_from_eleven_eigenvalues", 1), ("nonperron_edge_ten_gaps_from_eleven_eigenvalues", 2)):
            for start_rank in range(first_rank, min(n, first_rank + 10)):
                row = dict(gaps[start_rank - 1])
                row["edge_role"] = role
                row["uses_eleven_eigenvalues_when_available"] = bool(n >= first_rank + 10)
                first_ten_gap_rows.append(row)
        write_csv(case_dir / "tables/first_ten_consecutive_edge_gaps.csv", first_ten_gap_rows)
        detailed_ranks = sorted(set(config["part1"]["detailed_literal_ranks"]) | set(config["part1"]["detailed_nonperron_ranks"]))
        geometry = _coordinate_and_geometry_tables(graph, values, vectors, detailed_ranks, config["part1"]["coordinate_counts"], prefix)
        for name, rows in geometry.items():
            write_csv(case_dir / f"tables/{name}.csv.gz", rows)
        targets, coverage = select_part2_targets(n, quantile_windows, counts, config["part2"])
        for target in targets:
            start_rank = int(target["pair_start_decreasing_rank"])
            left_index = n - start_rank
            right_index = n - (start_rank + 1)
            left_vector = np.asarray(vectors[:, left_index])
            right_vector = np.asarray(vectors[:, right_index])
            left_mass = np.square(left_vector)
            right_mass = np.square(right_vector)
            target.update({
                **prefix,
                "left_ascending_index": left_index,
                "right_ascending_index": right_index,
                "left_energy": float(values[left_index]),
                "right_energy": float(values[right_index]),
                "observed_adjacent_gap": float(values[left_index] - values[right_index]),
                "left_n_times_ipr": float(n * np.sum(left_mass * left_mass)),
                "right_n_times_ipr": float(n * np.sum(right_mass * right_mass)),
                "left_support_size_80": int(localization[left_index]["support_size_80"]),
                "right_support_size_80": int(localization[right_index]["support_size_80"]),
                "whole_vector_mass_overlap": float(np.sum(np.minimum(left_mass, right_mass))),
                "analysis_role": config["execution"]["part2_analysis_role"],
            })
        for row in coverage:
            row.update(prefix)
        write_csv(case_dir / "tables/part2_target_selection.csv", targets)
        write_csv(case_dir / "tables/part2_target_coverage.csv", coverage)
        manifest.update({
            "status": "COMPLETE",
            "finished_at_utc": utc_now(),
            "runtime_seconds": time.time() - started,
            "graph_summary": graph_summary,
            "spectral_summary": spectral,
            "multiplicity_resolution": resolution,
            "zero_cluster_dimension_at_resolution": int(np.count_nonzero(zero_mask)),
            "localization_row_count": len(localization),
            "gap_row_count": len(gaps),
            "ratio_row_count": len(ratios),
            "window_summary_count": len(window_summaries),
            "part2_target_count": len(targets),
            "scientific_status": "complete Part 1; finite single-realization case study, not an ensemble phase-transition inference",
        })
        _atomic_json(manifest_path, manifest)
        return case_dir
    except Exception as error:
        manifest.update({"status": "FAILED", "finished_at_utc": utc_now(), "runtime_seconds": time.time() - started, "failure_type": type(error).__name__, "failure_message": str(error)})
        _atomic_json(manifest_path, manifest)
        raise


def mass_local_maximum_centers(
    graph: nx.Graph, vector: np.ndarray, top_coordinate_count: int
) -> tuple[list[int], list[dict[str, object]]]:
    masses = np.square(np.abs(np.asarray(vector, dtype=float)))
    order = np.lexsort((np.arange(len(masses)), -masses))
    inspected = order[: min(int(top_coordinate_count), len(order))]
    centers: list[int] = []
    rows: list[dict[str, object]] = []
    for coordinate_rank, vertex_value in enumerate(inspected, start=1):
        vertex = int(vertex_value)
        disqualifiers = [
            int(neighbor)
            for neighbor in graph.neighbors(vertex)
            if masses[int(neighbor)] > masses[vertex]
            or (masses[int(neighbor)] == masses[vertex] and int(neighbor) < vertex)
        ]
        accepted = not disqualifiers
        if accepted:
            centers.append(vertex)
        rows.append({
            "coordinate_rank": coordinate_rank,
            "vertex_id": vertex,
            "coordinate_mass": float(masses[vertex]),
            "graph_neighbor_local_maximum": accepted,
            "disqualifying_neighbor_count": len(disqualifiers),
            "smallest_disqualifying_neighbor": min(disqualifiers) if disqualifiers else math.nan,
            "tie_rule": "equal mass is won by smaller vertex ID",
        })
    if not centers and len(order):
        centers = [int(order[0])]
        rows.append({
            "coordinate_rank": 1,
            "vertex_id": int(order[0]),
            "coordinate_mass": float(masses[order[0]]),
            "graph_neighbor_local_maximum": False,
            "fallback_selected": True,
            "fallback_reason": "no local maximum found among inspected top coordinates",
        })
    return centers, rows


def _ball(graph: nx.Graph, center: int, radius: int) -> np.ndarray:
    return np.asarray(
        sorted(nx.single_source_shortest_path_length(graph, int(center), cutoff=int(radius))),
        dtype=int,
    )


def overlap_coefficient(left: Sequence[int], right: Sequence[int]) -> float:
    left_set = set(int(value) for value in left)
    right_set = set(int(value) for value in right)
    denominator = min(len(left_set), len(right_set))
    return float(len(left_set & right_set) / denominator) if denominator else 0.0


def greedy_mass_regions(
    graph: nx.Graph,
    left_vector: np.ndarray,
    right_vector: np.ndarray,
    centers: Sequence[int],
    *,
    radius: int,
    x: float,
    y: float,
    maximum_centers: int,
) -> dict[str, object]:
    n = graph.number_of_nodes()
    left_mass = np.square(np.abs(np.asarray(left_vector, dtype=float)))
    right_mass = np.square(np.abs(np.asarray(right_vector, dtype=float)))
    unique_centers = sorted(set(int(value) for value in centers))
    candidates = [{"center": center, "nodes": _ball(graph, center, radius)} for center in unique_centers]
    cap = int(math.floor(float(y) * n + 1e-12))
    selected: list[dict[str, object]] = []
    union: set[int] = set()
    trace: list[dict[str, object]] = []
    stopping_reason = "target_coverage_reached"
    while True:
        current_nodes = np.asarray(sorted(union), dtype=int)
        current_left = float(np.sum(left_mass[current_nodes])) if len(current_nodes) else 0.0
        current_right = float(np.sum(right_mass[current_nodes])) if len(current_nodes) else 0.0
        if current_left + 1e-14 >= float(x) and current_right + 1e-14 >= float(x):
            break
        if len(selected) >= int(maximum_centers):
            stopping_reason = "center_count_cap"
            break
        feasible: list[tuple[float, int, int, dict[str, object], set[int], float, float]] = []
        for candidate in candidates:
            if any(int(item["center"]) == int(candidate["center"]) for item in selected):
                continue
            proposed = union | set(int(value) for value in candidate["nodes"])
            if len(proposed) > cap:
                continue
            proposed_nodes = np.asarray(sorted(proposed), dtype=int)
            proposed_left = float(np.sum(left_mass[proposed_nodes]))
            proposed_right = float(np.sum(right_mass[proposed_nodes]))
            old_deficit = max(0.0, float(x) - current_left) + max(0.0, float(x) - current_right)
            new_deficit = max(0.0, float(x) - proposed_left) + max(0.0, float(x) - proposed_right)
            improvement = old_deficit - new_deficit
            if improvement > 1e-15:
                feasible.append((-improvement, len(proposed) - len(union), int(candidate["center"]), candidate, proposed, proposed_left, proposed_right))
        if not feasible:
            stopping_reason = "locality_cap_or_no_positive_coverage_improvement"
            break
        feasible.sort(key=lambda item: (item[0], item[1], item[2]))
        _, added, _, chosen, proposed, proposed_left, proposed_right = feasible[0]
        selected.append(chosen)
        union = proposed
        trace.append({
            "selection_step": len(selected),
            "center": int(chosen["center"]),
            "added_vertex_count": int(added),
            "union_vertex_count": len(union),
            "left_union_mass": proposed_left,
            "right_union_mass": proposed_right,
        })
    union_nodes = np.asarray(sorted(union), dtype=int)
    left_union_mass = float(np.sum(left_mass[union_nodes])) if len(union_nodes) else 0.0
    right_union_mass = float(np.sum(right_mass[union_nodes])) if len(union_nodes) else 0.0
    rejection_rows: list[dict[str, object]] = []
    selected_centers = {int(item["center"]) for item in selected}
    for candidate in candidates:
        center = int(candidate["center"])
        if center in selected_centers:
            continue
        proposed_size = len(union | set(int(value) for value in candidate["nodes"]))
        if proposed_size > cap:
            reason = "union_locality_cap"
        elif len(selected) >= int(maximum_centers):
            reason = "center_count_cap"
        elif left_union_mass >= x and right_union_mass >= x:
            reason = "coverage_already_reached"
        else:
            reason = "no_positive_deficit_reduction_or_not_selected_by_tie_break"
        rejection_rows.append({
            "center": center,
            "radius": int(radius),
            "ball_vertex_count": int(len(candidate["nodes"])),
            "proposed_union_vertex_count": proposed_size,
            "rejection_reason": reason,
        })
    return {
        "selected": selected,
        "union_nodes": union_nodes,
        "trace": trace,
        "rejections": rejection_rows,
        "left_union_mass": left_union_mass,
        "right_union_mass": right_union_mass,
        "union_vertex_count": len(union_nodes),
        "union_vertex_fraction": float(len(union_nodes) / n),
        "coverage_pass": bool(left_union_mass + 1e-14 >= x and right_union_mass + 1e-14 >= x and len(union_nodes) <= cap),
        "stopping_reason": stopping_reason,
        "locality_vertex_cap": cap,
    }


def grouped_selected_balls(
    selected: Sequence[Mapping[str, object]], overlap_threshold: float
) -> tuple[list[np.ndarray], list[dict[str, object]]]:
    if not selected:
        return [], []
    parent = list(range(len(selected)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    overlap_rows: list[dict[str, object]] = []
    for left in range(len(selected)):
        for right in range(left + 1, len(selected)):
            coefficient = overlap_coefficient(selected[left]["nodes"], selected[right]["nodes"])
            merge = coefficient + 1e-15 >= float(overlap_threshold)
            if merge:
                union(left, right)
            overlap_rows.append({
                "left_center": int(selected[left]["center"]),
                "right_center": int(selected[right]["center"]),
                "overlap_coefficient": coefficient,
                "merge_threshold": float(overlap_threshold),
                "merged_by_rule": merge,
            })
    groups: dict[int, list[int]] = {}
    for index in range(len(selected)):
        groups.setdefault(find(index), []).append(index)
    regions = [
        np.asarray(sorted(set().union(*(set(int(value) for value in selected[index]["nodes"]) for index in indices))), dtype=int)
        for _, indices in sorted(groups.items())
    ]
    return regions, overlap_rows


def degree_baseline_regions(
    graph: nx.Graph,
    *,
    radius: int,
    maximum_fraction: float,
    maximum_regions: int,
    overlap_threshold: float,
) -> tuple[list[np.ndarray], list[dict[str, object]], list[dict[str, object]]]:
    n = graph.number_of_nodes()
    degrees = np.asarray([graph.degree(vertex) for vertex in range(n)], dtype=int)
    centers = np.lexsort((np.arange(n), -degrees))[: min(50, n)]
    selected = [{"center": int(center), "nodes": _ball(graph, int(center), radius)} for center in centers]
    merged, overlap_rows = grouped_selected_balls(selected, overlap_threshold)
    accepted: list[np.ndarray] = []
    region_rows: list[dict[str, object]] = []
    for order, nodes in enumerate(merged):
        accepted_flag = bool(len(nodes) / n <= float(maximum_fraction) and len(accepted) < int(maximum_regions))
        region_rows.append({
            "baseline_group_order": order,
            "vertex_count": len(nodes),
            "vertex_fraction": float(len(nodes) / n),
            "accepted": accepted_flag,
            "rejection_reason": "" if accepted_flag else ("per_region_locality_cap" if len(nodes) / n > float(maximum_fraction) else "maximum_region_count"),
            "certification_note": "degree baseline; volume cap is not a mass-concentration certificate",
        })
        if accepted_flag:
            accepted.append(nodes)
    return accepted, region_rows, overlap_rows


def _vertex_set_hash(nodes: Sequence[int]) -> str:
    array = np.asarray(sorted(set(int(value) for value in nodes)), dtype=np.int64)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


@dataclass
class CachedLocalModes:
    nodes: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    method: str
    target_energy: float | None
    cache_key: str
    solver_status: str


class LocalModeCache:
    def __init__(self, root: Path, operator: sparse.csr_matrix, config: Mapping[str, object]):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.operator = sparse.csr_matrix(operator)
        self.config = config
        self.memory: dict[str, CachedLocalModes] = {}

    def get(
        self,
        nodes: Sequence[int],
        method: str,
        target_energy: float | None = None,
        maximum_requested: int | None = None,
    ) -> CachedLocalModes:
        nodes_array = np.asarray(sorted(set(int(value) for value in nodes)), dtype=int)
        if not len(nodes_array):
            raise ValueError("local region is empty")
        energy_key = None if target_energy is None else float(target_energy)
        if maximum_requested is None:
            maximum_requested = max(int(value) for value in self.config["mode_counts"])
        payload = {
            "version": LINEAR_PA_TWO_PART_VERSION,
            "vertex_set_hash": _vertex_set_hash(nodes_array),
            "method": str(method),
            "target_energy": energy_key,
            "maximum_mode_count": int(maximum_requested),
            "degeneracy_tolerance": float(self.config["local_degeneracy_relative_tolerance"]),
        }
        key = stable_json_hash(payload)
        if key in self.memory:
            return self.memory[key]
        array_path = self.root / f"{key}.npz"
        manifest_path = self.root / f"{key}.json"
        if array_path.exists() and manifest_path.exists():
            manifest = read_json(manifest_path)
            if manifest.get("cache_key") != key:
                raise ValueError("local eigensystem cache manifest identity mismatch")
            with np.load(array_path, allow_pickle=False) as data:
                result = CachedLocalModes(
                    nodes=np.asarray(data["nodes"], dtype=int),
                    eigenvalues=np.asarray(data["eigenvalues"], dtype=float),
                    eigenvectors=np.asarray(data["eigenvectors"], dtype=float),
                    method=str(manifest["method"]),
                    target_energy=manifest.get("target_energy"),
                    cache_key=key,
                    solver_status=str(manifest["solver_status"]),
                )
            if not np.array_equal(result.nodes, nodes_array):
                raise ValueError("local eigensystem cache vertex set mismatch")
            self.memory[key] = result
            return result
        restricted = self.operator[nodes_array][:, nodes_array].tocsr()
        dimension = restricted.shape[0]
        compute_count = min(dimension, maximum_requested + 8)
        dense_limit = int(self.config["local_dense_dimension_limit"])
        solver_status = "complete"
        if dimension <= dense_limit or compute_count >= dimension - 1:
            all_values, all_vectors = linalg.eigh(restricted.toarray(), check_finite=False)
            if method == "largest_algebraic":
                order = np.argsort(all_values)[::-1]
            elif method == "target_energy_nearest":
                if target_energy is None:
                    raise ValueError("target-energy selection requires target_energy")
                order = np.lexsort((-all_values, np.abs(all_values - float(target_energy))))
            else:
                raise ValueError(f"unsupported local mode method {method}")
            order = order[:compute_count]
            values = np.asarray(all_values[order], dtype=float)
            vectors = np.asarray(all_vectors[:, order], dtype=float)
        else:
            k = min(compute_count, dimension - 2)
            try:
                if method == "largest_algebraic":
                    values, vectors = sparse_linalg.eigsh(restricted, k=k, which="LA", tol=1e-10)
                    order = np.argsort(values)[::-1]
                elif method == "target_energy_nearest":
                    if target_energy is None:
                        raise ValueError("target-energy selection requires target_energy")
                    values, vectors = sparse_linalg.eigsh(restricted, k=k, sigma=float(target_energy), which="LM", tol=1e-10)
                    order = np.lexsort((-values, np.abs(values - float(target_energy))))
                else:
                    raise ValueError(f"unsupported local mode method {method}")
                values = np.asarray(values[order], dtype=float)
                vectors = np.asarray(vectors[:, order], dtype=float)
            except Exception as error:
                solver_status = f"sparse_solver_failed:{type(error).__name__}:{error}"
                raise RuntimeError(solver_status) from error
        for column in range(vectors.shape[1]):
            maximum = int(np.argmax(np.abs(vectors[:, column])))
            if vectors[maximum, column] < 0.0:
                vectors[:, column] *= -1.0
        np.savez_compressed(array_path, nodes=nodes_array, eigenvalues=values, eigenvectors=vectors)
        _atomic_json(manifest_path, {
            **payload,
            "status": "COMPLETE",
            "cache_key": key,
            "method": method,
            "target_energy": energy_key,
            "region_dimension": dimension,
            "computed_mode_count": len(values),
            "solver_status": solver_status,
            "array_path": str(array_path),
            "array_sha256": sha256_file(array_path),
        })
        result = CachedLocalModes(nodes_array, values, vectors, method, energy_key, key, solver_status)
        self.memory[key] = result
        return result


def actual_mode_count_with_degeneracy(values: np.ndarray, requested: int, relative_tolerance: float) -> int:
    count = min(int(requested), len(values))
    if count <= 0 or count == len(values):
        return count
    boundary = float(values[count - 1])
    scale = max(1.0, abs(boundary))
    while count < len(values) and abs(float(values[count]) - boundary) <= float(relative_tolerance) * scale:
        count += 1
    return count


def ordered_residual_orthogonalization(states: np.ndarray, relative_cutoff: float) -> dict[str, object]:
    states = np.asarray(states, dtype=float)
    if states.ndim != 2:
        raise ValueError("reference state array must be two-dimensional")
    if states.shape[1] == 0:
        return {"states": np.empty((states.shape[0], 0)), "transform": np.empty((0, 0)), "kept_indices": [], "discarded_indices": [], "orthonormality_error": 0.0}
    columns: list[np.ndarray] = []
    coefficients: list[np.ndarray] = []
    kept: list[int] = []
    discarded: list[int] = []
    scale = max(1.0, float(np.linalg.norm(states, axis=0).max()))
    threshold = float(relative_cutoff) * scale
    for index in range(states.shape[1]):
        vector = states[:, index].copy()
        coefficient = np.zeros(states.shape[1], dtype=float)
        coefficient[index] = 1.0
        if columns:
            u = np.column_stack(columns)
            transform = np.column_stack(coefficients)
            for _ in range(2):
                projection = u.T @ vector
                vector -= u @ projection
                coefficient -= transform @ projection
        norm = float(np.linalg.norm(vector))
        if norm <= threshold:
            discarded.append(index)
            continue
        columns.append(vector / norm)
        coefficients.append(coefficient / norm)
        kept.append(index)
    u = np.column_stack(columns) if columns else np.empty((states.shape[0], 0))
    transform = np.column_stack(coefficients) if coefficients else np.empty((states.shape[1], 0))
    return {
        "states": u,
        "transform": transform,
        "kept_indices": kept,
        "discarded_indices": discarded,
        "threshold": threshold,
        "orthonormality_error": float(np.linalg.norm(u.T @ u - np.eye(u.shape[1]))) if u.shape[1] else 0.0,
    }


def symmetric_pair_basis(states: np.ndarray, relative_cutoff: float) -> dict[str, object]:
    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or states.shape[1] != 2:
        raise ValueError("spatial pair basis requires exactly two raw columns")
    gram = states.T @ states
    eigenvalues, eigenvectors = linalg.eigh(gram, check_finite=False)
    maximum = max(1.0, float(np.max(eigenvalues)))
    threshold = float(relative_cutoff) * maximum
    if np.count_nonzero(eigenvalues > threshold) != 2:
        return {"valid": False, "status": "rank_one_or_duplicate_pair", "gram_eigenvalues": eigenvalues, "threshold": threshold}
    transform = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    u = states @ transform
    return {
        "valid": True,
        "status": "full_rank_symmetric_orthogonalization",
        "states": u,
        "transform": transform,
        "gram": gram,
        "gram_eigenvalues": eigenvalues,
        "threshold": threshold,
        "orthonormality_error": float(np.linalg.norm(u.T @ u - np.eye(2))),
    }


def _embedded_modes(cache: CachedLocalModes, n: int) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for column in range(cache.eigenvectors.shape[1]):
        vector = np.zeros(n, dtype=float)
        vector[cache.nodes] = cache.eigenvectors[:, column]
        result.append(vector)
    return result


def _region_union(regions: Sequence[np.ndarray]) -> np.ndarray:
    if not regions:
        return np.empty(0, dtype=int)
    return np.asarray(sorted(set().union(*(set(int(value) for value in region) for region in regions))), dtype=int)


def _dictionary_raw_modes(
    *,
    regions: Sequence[np.ndarray],
    local_cache: LocalModeCache,
    selection_method: str,
    target_energy: float | None,
    mode_counts: Sequence[int],
    degeneracy_tolerance: float,
) -> tuple[list[dict[str, object]], dict[int, dict[int, int]]]:
    requested_counts = sorted(set(int(value) for value in mode_counts))
    actual_by_request: dict[int, dict[int, int]] = {count: {} for count in requested_counts}
    candidates: list[dict[str, object]] = []
    for region_id, region in enumerate(regions):
        cached = local_cache.get(
            region,
            selection_method,
            target_energy,
            maximum_requested=max(requested_counts),
        )
        embedded = _embedded_modes(cached, local_cache.operator.shape[0])
        for requested in requested_counts:
            actual_by_request[requested][region_id] = actual_mode_count_with_degeneracy(
                cached.eigenvalues,
                requested,
                degeneracy_tolerance,
            )
        for local_index, (energy, vector) in enumerate(zip(cached.eigenvalues, embedded), start=1):
            activation = next(
                (
                    requested
                    for requested in requested_counts
                    if local_index <= actual_by_request[requested][region_id]
                ),
                None,
            )
            if activation is None:
                continue
            residual = local_cache.operator.dot(vector) - float(energy) * vector
            candidates.append({
                "region_id": region_id,
                "local_mode_rank": local_index,
                "local_energy": float(energy),
                "activation_requested_count": int(activation),
                "vector": vector,
                "region_nodes": np.asarray(region, dtype=int),
                "local_cache_key": cached.cache_key,
                "local_cache_mode_index": local_index - 1,
                "local_mode_method": selection_method,
                "full_operator_residual_norm": float(np.linalg.norm(residual)),
                "boundary_condition": "induced subgraph Dirichlet restriction; zero extension",
            })
    candidates.sort(key=lambda row: (int(row["activation_requested_count"]), int(row["local_mode_rank"]), int(row["region_id"])))
    for raw_mode_id, row in enumerate(candidates):
        row["raw_mode_id"] = raw_mode_id
    return candidates, actual_by_request


def _compatible_reduced_matching(
    *,
    exact_values: np.ndarray,
    exact_vectors: np.ndarray,
    u: np.ndarray,
    k_values: np.ndarray,
    k_vectors: np.ndarray,
    full_spectral_span: float,
    config: Mapping[str, object],
) -> dict[str, object]:
    observed_gap = float(abs(exact_values[0] - exact_values[1]))
    half_width = max(
        float(config["matching_energy_window_absolute_floor"]),
        float(config["matching_energy_window_gap_factor"]) * observed_gap,
        float(config["matching_energy_window_spectral_fraction"]) * float(full_spectral_span),
    )
    lower = float(np.min(exact_values) - half_width)
    upper = float(np.max(exact_values) + half_width)
    compatible = np.flatnonzero((k_values >= lower) & (k_values <= upper))
    if len(compatible) < 2:
        return {
            "status": "missing_two_reduced_states_in_compatible_energy_window",
            "compatible_state_count": int(len(compatible)),
            "matching_energy_lower": lower,
            "matching_energy_upper": upper,
        }
    coefficients = u.T @ exact_vectors
    overlaps = np.square(np.abs(k_vectors[:, compatible].T @ coefficients))
    predicted_rows, exact_columns = linear_sum_assignment(-overlaps)
    chosen_by_exact = np.full(2, -1, dtype=int)
    overlap_by_exact = np.full(2, np.nan, dtype=float)
    for predicted_row, exact_column in zip(predicted_rows, exact_columns):
        chosen_by_exact[int(exact_column)] = int(compatible[int(predicted_row)])
        overlap_by_exact[int(exact_column)] = float(overlaps[int(predicted_row), int(exact_column)])
    if np.any(chosen_by_exact < 0):
        return {
            "status": "overlap_assignment_incomplete",
            "compatible_state_count": int(len(compatible)),
            "matching_energy_lower": lower,
            "matching_energy_upper": upper,
        }
    competing = []
    for exact_column in range(2):
        ranked = compatible[np.argsort(overlaps[:, exact_column])[::-1][: min(3, len(compatible))]]
        competing.append([
            {
                "reduced_state_index": int(index),
                "reduced_energy": float(k_values[index]),
                "squared_overlap": float(np.square(abs(np.dot(u @ k_vectors[:, index], exact_vectors[:, exact_column])))),
            }
            for index in ranked
        ])
    return {
        "status": "matched_by_one_to_one_squared_overlap_within_energy_window",
        "compatible_state_count": int(len(compatible)),
        "matching_energy_lower": lower,
        "matching_energy_upper": upper,
        "chosen_reduced_indices": chosen_by_exact,
        "chosen_squared_overlaps": overlap_by_exact,
        "competing_matches": competing,
    }


def _pair_mode_diagnostics(
    *,
    operator: sparse.csr_matrix,
    exact_values: np.ndarray,
    exact_vectors: np.ndarray,
    regions: Sequence[np.ndarray],
    selected_raw_modes: Sequence[Mapping[str, object]],
    gram_cutoff: float,
    orientation_minimum_mass: float,
    resolution: float,
) -> tuple[dict[str, object], np.ndarray | None]:
    if len(regions) < 2:
        return {"pair_status": "inadequate_pair_fewer_than_two_regions", "spatial_pair_interpretable": False}, None
    exact_masses = np.square(np.abs(exact_vectors))
    combined_region_mass = np.asarray([
        float(np.sum(exact_masses[np.asarray(region, dtype=int), :])) for region in regions
    ])
    region_order = np.lexsort((np.arange(len(regions)), -combined_region_mass))
    left_region, right_region = int(region_order[0]), int(region_order[1])
    chosen_modes: list[Mapping[str, object]] = []
    for region_id in (left_region, right_region):
        available = [row for row in selected_raw_modes if int(row["region_id"]) == region_id]
        if not available:
            return {"pair_status": "inadequate_pair_region_has_no_retained_mode", "spatial_pair_interpretable": False}, None
        available.sort(
            key=lambda row: (
                -float(np.sum(np.square(np.abs(np.asarray(row["vector"]) @ exact_vectors)))),
                int(row["local_mode_rank"]),
            )
        )
        chosen_modes.append(available[0])
    raw = np.column_stack([np.asarray(row["vector"], dtype=float) for row in chosen_modes])
    basis = symmetric_pair_basis(raw, gram_cutoff)
    if not bool(basis["valid"]):
        return {
            "pair_status": str(basis["status"]),
            "spatial_pair_interpretable": False,
            "pair_gram_eigenvalues": np.asarray(basis["gram_eigenvalues"]).tolist(),
        }, None
    u = np.asarray(basis["states"], dtype=float)
    orientation = np.asarray([
        [float(np.sum(np.square(u[np.asarray(region, dtype=int), column]))) for column in range(2)]
        for region in (regions[left_region], regions[right_region])
    ])
    direct = u.T @ operator.dot(u)
    direct = 0.5 * (direct + direct.T)
    e_left = float(direct[0, 0])
    e_right = float(direct[1, 1])
    coupling = float(direct[0, 1])
    detuning = abs(e_left - e_right)
    observed_gap = float(abs(exact_values[0] - exact_values[1]))
    predicted_gap = math.sqrt(detuning * detuning + 4.0 * coupling * coupling)
    pair_capture = np.sum(np.square(np.abs(u.T @ exact_vectors)), axis=0)
    pair_values, pair_vectors = linalg.eigh(direct, check_finite=False)
    lifted = u @ pair_vectors
    lifted_overlaps = np.square(np.abs(lifted.T @ exact_vectors))
    assignment_rows, assignment_columns = linear_sum_assignment(-lifted_overlaps)
    chosen_pair_by_exact = np.full(2, -1, dtype=int)
    for predicted_row, exact_column in zip(assignment_rows, assignment_columns):
        chosen_pair_by_exact[int(exact_column)] = int(predicted_row)
    predicted_region_masses: list[list[float]] = []
    exact_region_masses: list[list[float]] = []
    for exact_column in range(2):
        predicted_vector = lifted[:, chosen_pair_by_exact[exact_column]]
        predicted_region_masses.append([
            float(np.sum(np.square(predicted_vector[np.asarray(regions[left_region], dtype=int)]))),
            float(np.sum(np.square(predicted_vector[np.asarray(regions[right_region], dtype=int)]))),
        ])
        exact_region_masses.append([
            float(np.sum(np.square(exact_vectors[np.asarray(regions[left_region], dtype=int), exact_column]))),
            float(np.sum(np.square(exact_vectors[np.asarray(regions[right_region], dtype=int), exact_column]))),
        ])
    region_mass_error = float(np.max(np.abs(np.asarray(predicted_region_masses) - np.asarray(exact_region_masses))))
    orientation_pass = bool(orientation[0, 0] >= orientation_minimum_mass and orientation[1, 1] >= orientation_minimum_mass)
    detuning_flag = "resolved" if detuning > resolution else ("exact_resonance_with_nonzero_direct_coupling" if abs(coupling) > resolution else "degenerate_unresolved_basis")
    observed_flag = "resolved" if observed_gap > resolution else "zero_or_unresolved_observed_gap"
    direct_edges = 0
    right_set = set(int(value) for value in regions[right_region])
    for vertex in regions[left_region]:
        direct_edges += sum(1 for neighbor in operator.indices[operator.indptr[int(vertex)]:operator.indptr[int(vertex) + 1]] if int(neighbor) in right_set)
    row = {
        "pair_status": "spatial_pair_interpretable" if orientation_pass else "pair_basis_failed_spatial_orientation_check",
        "spatial_pair_interpretable": orientation_pass,
        "left_region_id": left_region,
        "right_region_id": right_region,
        "left_raw_mode_id": int(chosen_modes[0]["raw_mode_id"]),
        "right_raw_mode_id": int(chosen_modes[1]["raw_mode_id"]),
        "left_local_cache_key": chosen_modes[0]["local_cache_key"],
        "right_local_cache_key": chosen_modes[1]["local_cache_key"],
        "left_local_cache_mode_index": int(chosen_modes[0]["local_cache_mode_index"]),
        "right_local_cache_mode_index": int(chosen_modes[1]["local_cache_mode_index"]),
        "left_region_vertex_hash": _vertex_set_hash(regions[left_region]),
        "right_region_vertex_hash": _vertex_set_hash(regions[right_region]),
        "left_region_vertices": np.asarray(regions[left_region], dtype=int).tolist(),
        "right_region_vertices": np.asarray(regions[right_region], dtype=int).tolist(),
        "pair_gram_eigenvalues": np.asarray(basis["gram_eigenvalues"]).tolist(),
        "pair_orthonormality_error": float(basis["orthonormality_error"]),
        "pair_basis_transform": np.asarray(basis["transform"]).tolist(),
        "assigned_region_mass_matrix": orientation.tolist(),
        "orientation_minimum_assigned_mass_threshold": float(orientation_minimum_mass),
        "reference_energy_left": e_left,
        "reference_energy_right": e_right,
        "direct_coupling": coupling,
        "absolute_direct_coupling": abs(coupling),
        "reference_detuning": detuning,
        "observed_adjacent_gap": observed_gap,
        "two_mode_predicted_gap": predicted_gap,
        "direct_coupling_over_reference_detuning": abs(coupling) / detuning if detuning > resolution else math.nan,
        "direct_coupling_over_observed_gap": abs(coupling) / observed_gap if observed_gap > resolution else math.nan,
        "twice_direct_coupling_over_reference_detuning": 2.0 * abs(coupling) / detuning if detuning > resolution else math.nan,
        "reference_detuning_status": detuning_flag,
        "observed_gap_status": observed_flag,
        "left_pair_only_capture": float(pair_capture[0]),
        "right_pair_only_capture": float(pair_capture[1]),
        "minimum_pair_only_capture": float(np.min(pair_capture)),
        "pair_predicted_squared_overlaps": [float(lifted_overlaps[chosen_pair_by_exact[column], column]) for column in range(2)],
        "pair_predicted_region_masses_including_interference": predicted_region_masses,
        "exact_region_masses": exact_region_masses,
        "maximum_region_mass_absolute_error": region_mass_error,
        "pair_relative_gap_error": abs(predicted_gap - observed_gap) / observed_gap if observed_gap > resolution else math.nan,
        "direct_interregion_edge_count": int(direct_edges),
        "analysis_role": "a_posteriori spatial pair selected by target mass and overlaps; not an independent prediction",
    }
    return row, u


def analyze_reference_dictionary(
    *,
    operator: sparse.csr_matrix,
    exact_values: np.ndarray,
    exact_vectors: np.ndarray,
    regions: Sequence[np.ndarray],
    local_cache: LocalModeCache,
    selection_method: str,
    target_energy: float | None,
    mode_counts: Sequence[int],
    config: Mapping[str, object],
    output_arrays_dir: Path,
    dictionary_id: str,
    resolution: float,
    full_spectral_span: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    n = operator.shape[0]
    union_nodes = _region_union(regions)
    union_mass = np.sum(np.square(np.abs(exact_vectors[union_nodes, :])), axis=0) if len(union_nodes) else np.zeros(2)
    raw_modes, actual_by_request = _dictionary_raw_modes(
        regions=regions,
        local_cache=local_cache,
        selection_method=selection_method,
        target_energy=target_energy,
        mode_counts=mode_counts,
        degeneracy_tolerance=float(config["local_degeneracy_relative_tolerance"]),
    )
    rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    gram_rows: list[dict[str, object]] = []
    previous_capture: np.ndarray | None = None
    maximum_modes = int(config["maximum_total_modes"])
    for requested in sorted(set(int(value) for value in mode_counts)):
        selected = [row for row in raw_modes if int(row["activation_requested_count"]) <= requested]
        budget_limited = len(selected) > maximum_modes
        selected = selected[:maximum_modes]
        if not selected:
            rows.append({
                "dictionary_id": dictionary_id,
                "selection_method": selection_method,
                "requested_modes_per_region": requested,
                "status": "no_local_modes",
            })
            continue
        w = np.column_stack([np.asarray(row["vector"], dtype=float) for row in selected])
        gram_eigenvalues = linalg.eigvalsh(w.T @ w, check_finite=False)
        orth = ordered_residual_orthogonalization(w, float(config["gram_relative_cutoff"]))
        u = np.asarray(orth["states"], dtype=float)
        if u.shape[1] == 0:
            rows.append({
                "dictionary_id": dictionary_id,
                "selection_method": selection_method,
                "requested_modes_per_region": requested,
                "status": "all_modes_discarded_by_gram_cutoff",
            })
            continue
        coefficients = u.T @ exact_vectors
        capture = np.sum(np.square(np.abs(coefficients)), axis=0)
        if np.any(capture > union_mass + 1e-8):
            raise AssertionError("dictionary capture exceeds union mass")
        monotone = True if previous_capture is None else bool(np.all(capture + 1e-10 >= previous_capture))
        if not monotone:
            raise AssertionError("nested dictionary capture decreased")
        previous_capture = capture
        k_matrix = u.T @ operator.dot(u)
        k_matrix = 0.5 * (k_matrix + k_matrix.T)
        residual = operator.dot(u) - u @ k_matrix
        k_values, k_vectors = linalg.eigh(k_matrix, check_finite=False)
        matching = _compatible_reduced_matching(
            exact_values=exact_values,
            exact_vectors=exact_vectors,
            u=u,
            k_values=k_values,
            k_vectors=k_vectors,
            full_spectral_span=full_spectral_span,
            config=config,
        )
        row: dict[str, object] = {
            "dictionary_id": dictionary_id,
            "selection_method": selection_method,
            "target_energy_for_mode_selection": target_energy,
            "requested_modes_per_region": requested,
            "actual_modes_by_region": actual_by_request[requested],
            "raw_mode_count_before_budget": len([item for item in raw_modes if int(item["activation_requested_count"]) <= requested]),
            "raw_mode_count": len(selected),
            "orthonormal_rank": u.shape[1],
            "budget_limited": budget_limited,
            "maximum_total_modes": maximum_modes,
            "discarded_raw_mode_indices": orth["discarded_indices"],
            "gram_eigenvalues": gram_eigenvalues.tolist(),
            "gram_relative_cutoff": float(config["gram_relative_cutoff"]),
            "orthonormality_error": float(orth["orthonormality_error"]),
            "left_union_mass": float(union_mass[0]),
            "right_union_mass": float(union_mass[1]),
            "left_dictionary_capture": float(capture[0]),
            "right_dictionary_capture": float(capture[1]),
            "minimum_dictionary_capture": float(np.min(capture)),
            "left_squared_reconstruction_error": float(1.0 - capture[0]),
            "right_squared_reconstruction_error": float(1.0 - capture[1]),
            "left_within_union_unrepresented_mass": float(union_mass[0] - capture[0]),
            "right_within_union_unrepresented_mass": float(union_mass[1] - capture[1]),
            "capture_monotone_under_enlargement": monotone,
            "residual_frobenius_norm": float(np.linalg.norm(residual)),
            "residual_operator_norm": float(np.linalg.svd(residual, compute_uv=False)[0]) if residual.size else 0.0,
            "status": str(matching["status"]),
            "compatible_state_count": int(matching.get("compatible_state_count", 0)),
            "matching_energy_lower": matching.get("matching_energy_lower", math.nan),
            "matching_energy_upper": matching.get("matching_energy_upper", math.nan),
            "analysis_role": "a_posteriori explanatory target-informed reference model",
        }
        if str(matching["status"]).startswith("matched"):
            chosen = np.asarray(matching["chosen_reduced_indices"], dtype=int)
            predicted_values = k_values[chosen]
            predicted_vectors = u @ k_vectors[:, chosen]
            predicted_residuals = residual @ k_vectors[:, chosen]
            direct_residuals = operator.dot(predicted_vectors) - predicted_vectors * predicted_values[None, :]
            identity_error = np.linalg.norm(direct_residuals - predicted_residuals, axis=0)
            predicted_region_masses = [
                [float(np.sum(np.square(predicted_vectors[np.asarray(region, dtype=int), column]))) for region in regions]
                for column in range(2)
            ]
            exact_region_masses = [
                [float(np.sum(np.square(exact_vectors[np.asarray(region, dtype=int), column]))) for region in regions]
                for column in range(2)
            ]
            predicted_gap = float(abs(predicted_values[0] - predicted_values[1]))
            observed_gap = float(abs(exact_values[0] - exact_values[1]))
            near_degenerate = bool(observed_gap <= resolution)
            overlap_values = np.asarray(matching["chosen_squared_overlaps"], dtype=float)
            region_error = float(np.max(np.abs(np.asarray(predicted_region_masses) - np.asarray(exact_region_masses)))) if regions else math.nan
            row.update({
                "left_predicted_energy": float(predicted_values[0]),
                "right_predicted_energy": float(predicted_values[1]),
                "left_exact_energy": float(exact_values[0]),
                "right_exact_energy": float(exact_values[1]),
                "predicted_adjacent_gap": predicted_gap,
                "observed_adjacent_gap": observed_gap,
                "relative_gap_error": abs(predicted_gap - observed_gap) / observed_gap if observed_gap > resolution else math.nan,
                "left_reconstructed_squared_overlap": float(overlap_values[0]),
                "right_reconstructed_squared_overlap": float(overlap_values[1]),
                "individual_vector_overlap_meaningful": not near_degenerate,
                "exact_pair_near_degenerate_cluster": near_degenerate,
                "exact_pair_projector_capture": float(np.sum(capture)),
                "predicted_state_residual_norms": np.linalg.norm(predicted_residuals, axis=0).tolist(),
                "predicted_state_relative_residuals": (np.linalg.norm(predicted_residuals, axis=0) / np.maximum(1.0, np.abs(predicted_values))).tolist(),
                "lifted_residual_identity_errors": identity_error.tolist(),
                "predicted_region_masses_including_interference": predicted_region_masses,
                "exact_region_masses": exact_region_masses,
                "maximum_region_mass_absolute_error": region_error,
                "competing_matches": matching["competing_matches"],
            })
        pair, _ = _pair_mode_diagnostics(
            operator=operator,
            exact_values=exact_values,
            exact_vectors=exact_vectors,
            regions=regions,
            selected_raw_modes=selected,
            gram_cutoff=float(config["gram_relative_cutoff"]),
            orientation_minimum_mass=float(config["spatial_orientation_minimum_assigned_mass"]),
            resolution=resolution,
        )
        pair.update({
            "dictionary_id": dictionary_id,
            "selection_method": selection_method,
            "requested_modes_per_region": requested,
        })
        pair_rows.append(pair)
        row.update({
            "pair_status": pair.get("pair_status"),
            "pair_only_minimum_capture": pair.get("minimum_pair_only_capture", math.nan),
            "spatial_pair_interpretable": pair.get("spatial_pair_interpretable", False),
        })
        rows.append(row)
        output_arrays_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_arrays_dir / f"{dictionary_id}__modes_{requested}.npz",
            raw_mode_ids=np.asarray([int(item["raw_mode_id"]) for item in selected], dtype=int),
            raw_mode_region_ids=np.asarray([int(item["region_id"]) for item in selected], dtype=int),
            raw_mode_local_ranks=np.asarray([int(item["local_mode_rank"]) for item in selected], dtype=int),
            raw_mode_energies=np.asarray([float(item["local_energy"]) for item in selected], dtype=float),
            w_to_u_transform=np.asarray(orth["transform"], dtype=float),
            gram_eigenvalues=np.asarray(gram_eigenvalues, dtype=float),
            effective_matrix=np.asarray(k_matrix, dtype=float),
            residual_column_norms=np.linalg.norm(residual, axis=0),
        )
        if requested == max(int(value) for value in mode_counts):
            for cutoff in config["gram_cutoff_sensitivity"]:
                sensitivity = ordered_residual_orthogonalization(w, float(cutoff))
                sensitivity_u = np.asarray(sensitivity["states"], dtype=float)
                sensitivity_capture = np.sum(np.square(np.abs(sensitivity_u.T @ exact_vectors)), axis=0) if sensitivity_u.shape[1] else np.zeros(2)
                gram_rows.append({
                    "dictionary_id": dictionary_id,
                    "selection_method": selection_method,
                    "requested_modes_per_region": requested,
                    "gram_relative_cutoff": float(cutoff),
                    "retained_rank": sensitivity_u.shape[1],
                    "discarded_rank": len(sensitivity["discarded_indices"]),
                    "left_capture": float(sensitivity_capture[0]),
                    "right_capture": float(sensitivity_capture[1]),
                    "orthonormality_error": float(sensitivity["orthonormality_error"]),
                    "interpretation": "sensitivity includes target capture; rank alone is not the conclusion",
                })
    return rows, pair_rows, gram_rows


def _target_row_types(row: Mapping[str, str]) -> dict[str, object]:
    result: dict[str, object] = dict(row)
    integer_fields = {
        "n",
        "graph_seed",
        "pair_start_decreasing_rank",
        "left_decreasing_rank",
        "right_decreasing_rank",
        "left_ascending_index",
        "right_ascending_index",
        "left_support_size_80",
        "right_support_size_80",
    }
    float_fields = {
        "degree_ccdf_alpha",
        "left_energy",
        "right_energy",
        "observed_adjacent_gap",
        "left_n_times_ipr",
        "right_n_times_ipr",
        "whole_vector_mass_overlap",
    }
    for field in integer_fields:
        if field in result and str(result[field]) != "":
            result[field] = int(float(str(result[field])))
    for field in float_fields:
        if field in result and str(result[field]) != "":
            result[field] = float(str(result[field]))
    for field in ("selection_roles",):
        if field in result and isinstance(result[field], str):
            result[field] = json.loads(str(result[field]))
    return result


def _target_spatial_summary(
    graph: nx.Graph,
    left_vector: np.ndarray,
    right_vector: np.ndarray,
    distance_cache: BFSDistanceCache,
) -> dict[str, object]:
    left_mass = np.square(np.abs(left_vector))
    right_mass = np.square(np.abs(right_vector))
    left_order = np.lexsort((np.arange(len(left_mass)), -left_mass))
    right_order = np.lexsort((np.arange(len(right_mass)), -right_mass))
    left_top = left_order[:10]
    right_top = right_order[:10]
    overlap = len(set(left_top.tolist()) & set(right_top.tolist()))
    dominant_distance = distance_cache.distance(int(left_order[0]), int(right_order[0]))
    mass_overlap = float(np.sum(np.minimum(left_mass, right_mass)))
    if overlap > 0 or dominant_distance == 0.0:
        description = "shared_concentration_vertices"
    elif math.isfinite(dominant_distance) and dominant_distance >= 2.0 and mass_overlap < 0.5:
        description = "distinct_measured_concentrations"
    else:
        description = "unresolved_or_overlapping_description"
    return {
        "left_dominant_vertex": int(left_order[0]),
        "right_dominant_vertex": int(right_order[0]),
        "dominant_vertex_distance": dominant_distance,
        "top10_overlap_vertex_count": overlap,
        "whole_vector_mass_overlap": mass_overlap,
        "descriptive_concentration_relationship": description,
        "description_basis": "dominant-vertex distance, top-10 overlap, and whole-vector mass overlap; not a one-center assumption",
    }


def _dictionary_signature(regions: Sequence[np.ndarray], method: str, target_energy: float | None) -> str:
    return stable_json_hash({
        "region_hashes": [_vertex_set_hash(region) for region in regions],
        "method": method,
        "target_energy": target_energy,
    })[:20]


def analyze_target_pair(
    *,
    graph: nx.Graph,
    operator: sparse.csr_matrix,
    full_values: np.ndarray,
    full_vectors: np.ndarray,
    target: Mapping[str, object],
    config: Mapping[str, object],
    acceptance: Mapping[str, object],
    local_cache: LocalModeCache,
    target_dir: Path,
    resolution: float,
) -> dict[str, object]:
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "arrays").mkdir(exist_ok=True)
    left_index = int(target["left_ascending_index"])
    right_index = int(target["right_ascending_index"])
    exact_vectors = np.column_stack([
        np.asarray(full_vectors[:, left_index], dtype=float),
        np.asarray(full_vectors[:, right_index], dtype=float),
    ])
    exact_values = np.asarray([full_values[left_index], full_values[right_index]], dtype=float)
    left_vector = exact_vectors[:, 0]
    right_vector = exact_vectors[:, 1]
    target_prefix = {
        "target_id": target["target_id"],
        "pair_start_decreasing_rank": int(target["pair_start_decreasing_rank"]),
        "left_decreasing_rank": int(target["left_decreasing_rank"]),
        "right_decreasing_rank": int(target["right_decreasing_rank"]),
        "left_energy": float(exact_values[0]),
        "right_energy": float(exact_values[1]),
        "observed_adjacent_gap": float(abs(exact_values[0] - exact_values[1])),
        "analysis_role": "a_posteriori_explanatory_target_informed",
    }
    distance_cache = BFSDistanceCache(graph)
    spatial = _target_spatial_summary(graph, left_vector, right_vector, distance_cache)
    left_centers, left_center_rows = mass_local_maximum_centers(
        graph, left_vector, int(config["mass_center_top_coordinate_count"])
    )
    right_centers, right_center_rows = mass_local_maximum_centers(
        graph, right_vector, int(config["mass_center_top_coordinate_count"])
    )
    centers = sorted(set(left_centers) | set(right_centers))
    center_rows: list[dict[str, object]] = []
    for role, rows in (("left_exact_vector", left_center_rows), ("right_exact_vector", right_center_rows)):
        for row in rows:
            center_rows.append({**target_prefix, "source_vector_role": role, **row})
    write_csv(target_dir / "center_candidates.csv.gz", center_rows)
    sensitivity_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    rejection_rows: list[dict[str, object]] = []
    overlap_rows: list[dict[str, object]] = []
    region_vertex_rows: list[dict[str, object]] = []
    dictionary_specs: dict[str, dict[str, object]] = {}
    for radius in config["radii"]:
        for x in config["x_grid"]:
            for y in config["y_grid"]:
                construction = greedy_mass_regions(
                    graph,
                    left_vector,
                    right_vector,
                    centers,
                    radius=int(radius),
                    x=float(x),
                    y=float(y),
                    maximum_centers=int(config["maximum_selected_centers"]),
                )
                setting_base = f"mass_r{int(radius)}_x{int(round(100*float(x)))}_y{int(round(100*float(y)))}"
                for row in construction["trace"]:
                    trace_rows.append({**target_prefix, "setting_id": setting_base, **row})
                for row in construction["rejections"]:
                    rejection_rows.append({**target_prefix, "setting_id": setting_base, "construction_family": "mass_informed_greedy", **row})
                ungrouped = [np.asarray(item["nodes"], dtype=int) for item in construction["selected"]]
                grouped, pair_overlaps = grouped_selected_balls(
                    construction["selected"],
                    float(config["group_overlap_threshold"]),
                )
                for row in pair_overlaps:
                    overlap_rows.append({**target_prefix, "setting_id": setting_base, **row})
                for grouping, regions in (("selected_balls_ungrouped", ungrouped), ("overlap_component_grouped", grouped)):
                    setting_id = f"{setting_base}__{grouping}"
                    union_nodes = _region_union(regions)
                    left_union_mass = float(np.sum(np.square(left_vector[union_nodes]))) if len(union_nodes) else 0.0
                    right_union_mass = float(np.sum(np.square(right_vector[union_nodes]))) if len(union_nodes) else 0.0
                    pass_flag = bool(
                        left_union_mass + 1e-14 >= float(x)
                        and right_union_mass + 1e-14 >= float(x)
                        and len(union_nodes) / graph.number_of_nodes() <= float(y) + 1e-14
                    )
                    row = {
                        **target_prefix,
                        "setting_id": setting_id,
                        "construction_family": "mass_informed_greedy",
                        "grouping_variant": grouping,
                        "radius": int(radius),
                        "x_target_mass": float(x),
                        "y_union_fraction_cap": float(y),
                        "selected_center_count": len(construction["selected"]),
                        "region_count": len(regions),
                        "union_vertex_count": len(union_nodes),
                        "union_vertex_fraction": float(len(union_nodes) / graph.number_of_nodes()),
                        "left_union_mass": left_union_mass,
                        "right_union_mass": right_union_mass,
                        "paired_coverage_and_union_cap_pass": pass_flag,
                        "construction_stopping_reason": construction["stopping_reason"],
                        "primary_reference_setting": bool(
                            math.isclose(float(x), float(config["reference_x"]))
                            and math.isclose(float(y), float(config["reference_y"]))
                            and int(radius) == 2
                            and grouping == "overlap_component_grouped"
                        ),
                        "region_rule_status": "passed" if pass_flag else "descriptive_failure_preserved",
                    }
                    sensitivity_rows.append(row)
                    for region_id, region in enumerate(regions):
                        for vertex in region:
                            region_vertex_rows.append({
                                **target_prefix,
                                "setting_id": setting_id,
                                "region_id": region_id,
                                "vertex_id": int(vertex),
                            })
                    if pass_flag and regions:
                        upper_limit = int(config["maximum_upper_count_by_size"][str(graph.number_of_nodes())])
                        is_upper = int(target["pair_start_decreasing_rank"]) <= upper_limit
                        methods = ["largest_algebraic"] if is_upper else ["target_energy_nearest"]
                        if is_upper and bool(config.get("run_energy_window_variant_for_upper_edge", False)):
                            methods.append("target_energy_nearest")
                        for method in methods:
                            target_energy = float(np.mean(exact_values)) if method == "target_energy_nearest" else None
                            signature = _dictionary_signature(regions, method, target_energy)
                            spec = dictionary_specs.setdefault(signature, {
                                "dictionary_id": signature,
                                "regions": regions,
                                "selection_method": method,
                                "target_energy": target_energy,
                                "equivalent_setting_ids": [],
                                "reference_setting_in_equivalents": False,
                                "construction_family": "mass_informed_greedy",
                            })
                            spec["equivalent_setting_ids"].append(setting_id)
                            spec["reference_setting_in_equivalents"] = bool(spec["reference_setting_in_equivalents"] or row["primary_reference_setting"])
    for radius in config["degree_baseline_radii"]:
        regions, baseline_rows, baseline_overlaps = degree_baseline_regions(
            graph,
            radius=int(radius),
            maximum_fraction=float(config["reference_y"]),
            maximum_regions=int(config["maximum_selected_centers"]),
            overlap_threshold=float(config["group_overlap_threshold"]),
        )
        setting_id = f"degree_baseline_r{int(radius)}_per_region_cap_y{int(round(100*float(config['reference_y'])))}"
        union_nodes = _region_union(regions)
        left_union_mass = float(np.sum(np.square(left_vector[union_nodes]))) if len(union_nodes) else 0.0
        right_union_mass = float(np.sum(np.square(right_vector[union_nodes]))) if len(union_nodes) else 0.0
        sensitivity_rows.append({
            **target_prefix,
            "setting_id": setting_id,
            "construction_family": "degree_ranked_historical_baseline",
            "grouping_variant": "overlap_component_grouped",
            "radius": int(radius),
            "x_target_mass": float(config["reference_x"]),
            "y_union_fraction_cap": float(config["reference_y"]),
            "selected_center_count": int(config["degree_baseline_center_count"]),
            "region_count": len(regions),
            "union_vertex_count": len(union_nodes),
            "union_vertex_fraction": float(len(union_nodes) / graph.number_of_nodes()),
            "left_union_mass": left_union_mass,
            "right_union_mass": right_union_mass,
            "paired_coverage_and_union_cap_pass": bool(
                left_union_mass >= float(config["reference_x"])
                and right_union_mass >= float(config["reference_x"])
                and len(union_nodes) / graph.number_of_nodes() <= float(config["reference_y"])
            ),
            "primary_reference_setting": False,
            "region_rule_status": "historical_per_region_cap_baseline_not_mass_certified",
        })
        for row in baseline_rows:
            rejection_rows.append({**target_prefix, "setting_id": setting_id, "construction_family": "degree_ranked_historical_baseline", **row})
        for row in baseline_overlaps:
            overlap_rows.append({**target_prefix, "setting_id": setting_id, **row})
        if regions:
            method = "largest_algebraic" if float(np.mean(exact_values)) >= 0.0 else "target_energy_nearest"
            target_energy = None if method == "largest_algebraic" else float(np.mean(exact_values))
            signature = _dictionary_signature(regions, method, target_energy)
            dictionary_specs[signature] = {
                "dictionary_id": signature,
                "regions": regions,
                "selection_method": method,
                "target_energy": target_energy,
                "equivalent_setting_ids": [setting_id],
                "reference_setting_in_equivalents": False,
                "construction_family": "degree_ranked_historical_baseline",
            }
    write_csv(target_dir / "region_sensitivity.csv.gz", sensitivity_rows)
    write_csv(target_dir / "region_selection_trace.csv.gz", trace_rows)
    write_csv(target_dir / "region_rejections.csv.gz", rejection_rows)
    write_csv(target_dir / "region_ball_overlaps.csv.gz", overlap_rows)
    write_csv(target_dir / "region_vertices.csv.gz", region_vertex_rows)
    dictionary_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    gram_rows: list[dict[str, object]] = []
    solver_failure_rows: list[dict[str, object]] = []
    main_counts = [int(value) for value in config["mode_counts"]]
    for signature, spec in sorted(dictionary_specs.items()):
        spec_prefix = {
            **target_prefix,
            "dictionary_id": signature,
            "equivalent_setting_ids": spec["equivalent_setting_ids"],
            "reference_setting_in_equivalents": spec["reference_setting_in_equivalents"],
            "construction_family": spec["construction_family"],
            "region_count": len(spec["regions"]),
            "region_vertex_hashes": [_vertex_set_hash(region) for region in spec["regions"]],
        }
        try:
            rows, pairs, grams = analyze_reference_dictionary(
                operator=operator,
                exact_values=exact_values,
                exact_vectors=exact_vectors,
                regions=spec["regions"],
                local_cache=local_cache,
                selection_method=str(spec["selection_method"]),
                target_energy=spec["target_energy"],
                mode_counts=main_counts,
                config=config,
                output_arrays_dir=target_dir / "arrays",
                dictionary_id=signature,
                resolution=resolution,
                full_spectral_span=float(full_values[-1] - full_values[0]),
            )
            for row in rows:
                row.update(spec_prefix)
            for row in pairs:
                row.update(spec_prefix)
            for row in grams:
                row.update(spec_prefix)
            dictionary_rows.extend(rows)
            pair_rows.extend(pairs)
            gram_rows.extend(grams)
            final_main = [row for row in rows if int(row.get("requested_modes_per_region", -1)) == max(main_counts)]
            needs_extension = any(
                float(row.get("left_union_mass", 0.0)) >= 0.90
                and float(row.get("right_union_mass", 0.0)) >= 0.90
                and float(row.get("minimum_dictionary_capture", 0.0)) < 0.90
                for row in final_main
            )
            extended = [int(value) for value in config.get("extended_mode_counts", [])]
            if needs_extension and extended:
                all_counts = main_counts + extended
                ext_rows, ext_pairs, ext_grams = analyze_reference_dictionary(
                    operator=operator,
                    exact_values=exact_values,
                    exact_vectors=exact_vectors,
                    regions=spec["regions"],
                    local_cache=local_cache,
                    selection_method=str(spec["selection_method"]),
                    target_energy=spec["target_energy"],
                    mode_counts=all_counts,
                    config=config,
                    output_arrays_dir=target_dir / "arrays",
                    dictionary_id=signature,
                    resolution=resolution,
                    full_spectral_span=float(full_values[-1] - full_values[0]),
                )
                for row in ext_rows:
                    if int(row.get("requested_modes_per_region", -1)) in extended:
                        row.update(spec_prefix)
                        row["extension_trigger"] = "union_mass_at_least_0p90_but_20_mode_capture_below_0p90"
                        dictionary_rows.append(row)
                for row in ext_pairs:
                    if int(row.get("requested_modes_per_region", -1)) in extended:
                        row.update(spec_prefix)
                        pair_rows.append(row)
                for row in ext_grams:
                    row.update(spec_prefix)
                    gram_rows.append(row)
        except RuntimeError as error:
            solver_failure_rows.append({
                **spec_prefix,
                "failure_class": "numerical_solver_failure",
                "failure_type": type(error).__name__,
                "failure_message": str(error),
            })
    pair_lookup = {
        (str(row["dictionary_id"]), int(row["requested_modes_per_region"])): row for row in pair_rows
    }
    for row in dictionary_rows:
        capture_pass = bool(float(row.get("minimum_dictionary_capture", -math.inf)) >= float(acceptance["minimum_capture"]))
        overlap_pass = bool(
            not bool(row.get("individual_vector_overlap_meaningful", True))
            or (
                float(row.get("left_reconstructed_squared_overlap", -math.inf)) >= float(acceptance["minimum_squared_overlap"])
                and float(row.get("right_reconstructed_squared_overlap", -math.inf)) >= float(acceptance["minimum_squared_overlap"])
            )
        )
        gap_pass = bool(float(row.get("relative_gap_error", math.inf)) <= float(acceptance["maximum_relative_gap_error"]))
        region_pass = bool(float(row.get("maximum_region_mass_absolute_error", math.inf)) <= float(acceptance["maximum_region_mass_absolute_error"]))
        pair = pair_lookup.get((str(row["dictionary_id"]), int(row["requested_modes_per_region"])), {})
        row.update({
            "capture_0p90_pass": capture_pass,
            "capture_0p95_sensitivity_pass": bool(float(row.get("minimum_dictionary_capture", -math.inf)) >= float(acceptance["minimum_capture_sensitivity"])),
            "individual_overlap_0p90_pass_when_meaningful": overlap_pass,
            "relative_gap_error_0p10_pass": gap_pass,
            "region_mass_error_0p05_pass": region_pass,
            "full_dictionary_quality_screen_pass": bool(capture_pass and overlap_pass and gap_pass and region_pass),
            "two_mode_quality_screen_pass": bool(
                pair.get("spatial_pair_interpretable", False)
                and float(pair.get("minimum_pair_only_capture", -math.inf)) >= float(acceptance["minimum_capture"])
                and float(pair.get("pair_relative_gap_error", math.inf)) <= float(acceptance["maximum_relative_gap_error"])
                and float(pair.get("maximum_region_mass_absolute_error", math.inf)) <= float(acceptance["maximum_region_mass_absolute_error"])
            ),
        })
        if row["two_mode_quality_screen_pass"]:
            row["scientific_interpretation"] = "validated_two_center_mixing_for_this_a_posteriori_reference_description"
        elif row["full_dictionary_quality_screen_pass"]:
            row["scientific_interpretation"] = "multimode_explanation_for_this_a_posteriori_reference_description"
        else:
            row["scientific_interpretation"] = "reference_model_inadequate_or_unresolved"
    write_csv(target_dir / "dictionary_mode_sensitivity.csv.gz", dictionary_rows)
    write_csv(target_dir / "spatial_pair_coupling.csv.gz", pair_rows)
    write_csv(target_dir / "gram_cutoff_sensitivity.csv.gz", gram_rows)
    write_csv(target_dir / "solver_failures.csv.gz", solver_failure_rows)
    return {
        **target_prefix,
        **spatial,
        "candidate_center_count": len(centers),
        "region_sensitivity_row_count": len(sensitivity_rows),
        "unique_dictionary_count": len(dictionary_specs),
        "dictionary_mode_row_count": len(dictionary_rows),
        "spatial_pair_row_count": len(pair_rows),
        "solver_failure_count": len(solver_failure_rows),
        "full_dictionary_quality_pass_count": sum(bool(row.get("full_dictionary_quality_screen_pass", False)) for row in dictionary_rows),
        "two_mode_quality_pass_count": sum(bool(row.get("two_mode_quality_screen_pass", False)) for row in dictionary_rows),
    }


def _aggregate_target_tables(part2_root: Path, targets: Sequence[Mapping[str, object]]) -> dict[str, list[dict[str, str]]]:
    names = (
        "region_sensitivity",
        "region_rejections",
        "dictionary_mode_sensitivity",
        "spatial_pair_coupling",
        "gram_cutoff_sensitivity",
        "solver_failures",
    )
    aggregated: dict[str, list[dict[str, str]]] = {name: [] for name in names}
    for target in targets:
        target_dir = Path(part2_root) / "targets" / str(target["target_id"])
        manifest_path = target_dir / "manifest.json"
        if not manifest_path.exists() or read_json(manifest_path).get("status") != "COMPLETE":
            continue
        for name in names:
            path = target_dir / f"{name}.csv.gz"
            if path.exists():
                aggregated[name].extend(_read_csv(path))
    return aggregated


def _json_list(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return json.loads(str(value))


def _boolean(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def schur_rows_from_pair_table(
    *,
    operator: sparse.csr_matrix,
    local_cache_root: Path,
    pair_rows: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
) -> list[dict[str, object]]:
    maximum_count = max(int(value) for value in config["mode_counts"])
    eligible = [
        row for row in pair_rows
        if _boolean(row.get("reference_setting_in_equivalents", False))
        and int(float(str(row.get("requested_modes_per_region", -1)))) == maximum_count
        and _boolean(row.get("spatial_pair_interpretable", False))
    ]
    eligible.sort(key=lambda row: (int(float(str(row["pair_start_decreasing_rank"]))), str(row["dictionary_id"])))
    selected = eligible[: int(config["schur_maximum_cases_per_graph"])]
    output: list[dict[str, object]] = []
    for case_order, row in enumerate(selected, start=1):
        try:
            columns: list[np.ndarray] = []
            for side in ("left", "right"):
                key = str(row[f"{side}_local_cache_key"])
                with np.load(Path(local_cache_root) / f"{key}.npz", allow_pickle=False) as data:
                    nodes = np.asarray(data["nodes"], dtype=int)
                    local_vectors = np.asarray(data["eigenvectors"], dtype=float)
                    mode_index = int(float(str(row[f"{side}_local_cache_mode_index"])))
                    vector = np.zeros(operator.shape[0], dtype=float)
                    vector[nodes] = local_vectors[:, mode_index]
                    columns.append(vector)
            basis = symmetric_pair_basis(np.column_stack(columns), relative_cutoff=float(config["gram_relative_cutoff"]))
            if not bool(basis["valid"]):
                output.append({
                    "schur_case_order": case_order,
                    "target_id": row["target_id"],
                    "status": "pair_became_rank_deficient_on_reconstruction",
                })
                continue
            u = np.asarray(basis["states"], dtype=float)
            direct = u.T @ operator.dot(u)
            direct = 0.5 * (direct + direct.T)
            energies = [float(direct[0, 0]), float(np.mean(np.diag(direct))), float(direct[1, 1])]
            scale = max(1.0, abs(energies[0]), abs(energies[-1]))
            for energy_role, energy in zip(("reference_left", "reference_mean", "reference_right"), energies):
                for relative_regularizer in config["schur_relative_regularizers"]:
                    eta = float(relative_regularizer) * scale
                    try:
                        result = schur_effective_operator(operator, u, energy, eta)
                        effective = np.asarray(result["effective_matrix"], dtype=complex)
                        correction = effective - direct
                        output.append({
                            "schur_case_order": case_order,
                            "target_id": row["target_id"],
                            "pair_start_decreasing_rank": int(float(str(row["pair_start_decreasing_rank"]))),
                            "dictionary_id": row["dictionary_id"],
                            "status": "COMPLETE",
                            "selection_rule": "first spatially interpretable reference-setting cases in target-rank order; coupling ratio not used",
                            "evaluation_energy_role": energy_role,
                            "evaluation_energy": energy,
                            "reference_energy_scale": scale,
                            "relative_regularizer": float(relative_regularizer),
                            "absolute_regularizer": eta,
                            "direct_K_00": float(direct[0, 0]),
                            "direct_K_11": float(direct[1, 1]),
                            "direct_K_01": float(direct[0, 1]),
                            "total_effective_00_real": float(np.real(effective[0, 0])),
                            "total_effective_00_imag": float(np.imag(effective[0, 0])),
                            "total_effective_11_real": float(np.real(effective[1, 1])),
                            "total_effective_11_imag": float(np.imag(effective[1, 1])),
                            "total_effective_coupling_real": float(np.real(effective[0, 1])),
                            "total_effective_coupling_imag": float(np.imag(effective[0, 1])),
                            "absolute_total_effective_coupling": float(abs(effective[0, 1])),
                            "mediated_correction_00_real": float(np.real(correction[0, 0])),
                            "mediated_correction_00_imag": float(np.imag(correction[0, 0])),
                            "mediated_correction_11_real": float(np.real(correction[1, 1])),
                            "mediated_correction_11_imag": float(np.imag(correction[1, 1])),
                            "mediated_coupling_correction_real": float(np.real(correction[0, 1])),
                            "mediated_coupling_correction_imag": float(np.imag(correction[0, 1])),
                            "absolute_mediated_coupling_correction": float(abs(correction[0, 1])),
                            "projected_resolvent_condition_number": float(result["projected_resolvent_condition_number"]),
                            "linear_solve_residual_norm": float(result["linear_solve_residual_norm"]),
                            "near_pole_flag": bool(result["near_pole_flag"]),
                            "interpretation": "energy-dependent complex diagnostic; not inserted into the real two-level gap formula without separate validation",
                        })
                    except Exception as error:
                        output.append({
                            "schur_case_order": case_order,
                            "target_id": row["target_id"],
                            "dictionary_id": row["dictionary_id"],
                            "status": "NUMERICAL_SOLVER_FAILURE",
                            "evaluation_energy_role": energy_role,
                            "evaluation_energy": energy,
                            "relative_regularizer": float(relative_regularizer),
                            "absolute_regularizer": eta,
                            "failure_type": type(error).__name__,
                            "failure_message": str(error),
                        })
        except Exception as error:
            output.append({
                "schur_case_order": case_order,
                "target_id": row.get("target_id"),
                "status": "PAIR_RECONSTRUCTION_FAILURE",
                "failure_type": type(error).__name__,
                "failure_message": str(error),
            })
    return output


def run_part2(
    *,
    project_root: Path,
    config: Mapping[str, object],
    n: int,
    alpha: float,
    output_root: Path,
    freeze_manifest: Path,
) -> Path:
    resolved = resolve_case(config, n=n, alpha=alpha)
    case_dir = case_directory(output_root, n, alpha)
    part1_manifest = read_json(case_dir / "part1_manifest.json")
    if part1_manifest.get("status") != "COMPLETE":
        raise ValueError("Part 1 must complete before Part 2")
    freeze = read_json(freeze_manifest)
    if freeze.get("status") != "COMPLETE":
        raise ValueError("source freeze is not complete")
    stage_identity = stable_json_hash({
        "version": LINEAR_PA_TWO_PART_VERSION,
        "stage": "part2",
        "source_configuration_id": freeze.get("source_configuration_id"),
        "part1_stage_identity": part1_manifest.get("stage_identity"),
        "resolved_case": resolved,
        "config": dict(config),
    })
    part2_root = case_dir / "part2"
    part2_root.mkdir(parents=True, exist_ok=True)
    (part2_root / "tables").mkdir(exist_ok=True)
    manifest_path = part2_root / "manifest.json"
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if existing.get("status") == "COMPLETE" and existing.get("stage_identity") == stage_identity:
            return part2_root
        if existing.get("stage_identity") not in (None, stage_identity):
            raise ValueError("existing Part 2 output has an incompatible cache identity")
    manifest = {
        **runtime_metadata(project_root),
        "status": "RUNNING",
        "stage": "linear_pa_part2_a_posteriori_local_mode_interactions",
        "stage_identity": stage_identity,
        "source_configuration_id": freeze.get("source_configuration_id"),
        "part1_stage_identity": part1_manifest.get("stage_identity"),
        "resolved_case": resolved,
        "restart_policy": "completed target subdirectories are reused only for the same target identity",
    }
    _atomic_json(manifest_path, manifest)
    started = time.time()
    graph, _ = graph_from_edge_cache(case_dir / "arrays/graph_edges.npz")
    operator = graph_to_csr(graph, nodelist=range(graph.number_of_nodes()))
    full_values = np.load(case_dir / "arrays/full_eigenvalues_ascending.npy", mmap_mode="r")
    full_vectors = np.load(case_dir / "arrays/full_eigenvectors_ascending.npy", mmap_mode="r")
    targets = [_target_row_types(row) for row in _read_csv(case_dir / "tables/part2_target_selection.csv")]
    resolution = float(part1_manifest["multiplicity_resolution"])
    local_cache = LocalModeCache(part2_root / "local_mode_cache", operator, config["part2"])
    summary_rows: list[dict[str, object]] = []
    try:
        for target in targets:
            target_dir = part2_root / "targets" / str(target["target_id"])
            target_dir.mkdir(parents=True, exist_ok=True)
            target_identity = stable_json_hash({
                "stage_identity": stage_identity,
                "target_id": target["target_id"],
                "left_ascending_index": target["left_ascending_index"],
                "right_ascending_index": target["right_ascending_index"],
            })
            target_manifest_path = target_dir / "manifest.json"
            if target_manifest_path.exists():
                prior = read_json(target_manifest_path)
                if prior.get("status") == "COMPLETE" and prior.get("target_identity") == target_identity:
                    summary_rows.append(dict(prior["summary"]))
                    continue
                if prior.get("target_identity") not in (None, target_identity):
                    raise ValueError(f"target cache identity mismatch for {target['target_id']}")
            target_manifest = {
                "status": "RUNNING",
                "target_identity": target_identity,
                "target": target,
                "started_at_utc": utc_now(),
            }
            _atomic_json(target_manifest_path, target_manifest)
            target_started = time.time()
            summary = analyze_target_pair(
                graph=graph,
                operator=operator,
                full_values=np.asarray(full_values),
                full_vectors=full_vectors,
                target=target,
                config=config["part2"],
                acceptance=config["acceptance"],
                local_cache=local_cache,
                target_dir=target_dir,
                resolution=resolution,
            )
            target_manifest.update({
                "status": "COMPLETE",
                "finished_at_utc": utc_now(),
                "runtime_seconds": time.time() - target_started,
                "summary": summary,
            })
            _atomic_json(target_manifest_path, target_manifest)
            summary_rows.append(summary)
        aggregated = _aggregate_target_tables(part2_root, targets)
        for name, rows in aggregated.items():
            write_csv(part2_root / f"tables/all_targets_{name}.csv.gz", rows)
        local_mode_index: list[dict[str, object]] = []
        for cache_manifest_path in sorted((part2_root / "local_mode_cache").glob("*.json")):
            cache_manifest = read_json(cache_manifest_path)
            local_mode_index.append({
                "cache_key": cache_manifest.get("cache_key"),
                "vertex_set_hash": cache_manifest.get("vertex_set_hash"),
                "region_dimension": cache_manifest.get("region_dimension"),
                "computed_mode_count": cache_manifest.get("computed_mode_count"),
                "method": cache_manifest.get("method"),
                "target_energy": cache_manifest.get("target_energy"),
                "solver_status": cache_manifest.get("solver_status"),
                "array_path": cache_manifest.get("array_path"),
                "array_sha256": cache_manifest.get("array_sha256"),
                "boundary_condition": "induced subgraph Dirichlet restriction; vectors stored in local vertex order",
            })
        write_csv(part2_root / "tables/local_mode_cache_index.csv", local_mode_index)
        schur_config = {
            **config["part2"],
            "schur_maximum_cases_per_graph": int(config["schur"]["maximum_cases_per_graph"]),
            "schur_relative_regularizers": config["schur"]["relative_regularizers"],
        }
        schur_rows = schur_rows_from_pair_table(
            operator=operator,
            local_cache_root=part2_root / "local_mode_cache",
            pair_rows=aggregated["spatial_pair_coupling"],
            config=schur_config,
        )
        write_csv(part2_root / "tables/schur_effective_coupling.csv.gz", schur_rows)
        write_csv(part2_root / "tables/target_summary.csv", summary_rows)
        manifest.update({
            "status": "COMPLETE",
            "finished_at_utc": utc_now(),
            "runtime_seconds": time.time() - started,
            "target_count": len(targets),
            "completed_target_count": len(summary_rows),
            "region_sensitivity_row_count": len(aggregated["region_sensitivity"]),
            "dictionary_mode_row_count": len(aggregated["dictionary_mode_sensitivity"]),
            "pair_coupling_row_count": len(aggregated["spatial_pair_coupling"]),
            "raw_local_eigensystem_cache_count": len(local_mode_index),
            "schur_row_count": len(schur_rows),
            "scientific_status": "complete including scientifically inadequate and unresolved reference descriptions",
        })
        _atomic_json(manifest_path, manifest)
        return part2_root
    except Exception as error:
        manifest.update({
            "status": "FAILED",
            "finished_at_utc": utc_now(),
            "runtime_seconds": time.time() - started,
            "failure_type": type(error).__name__,
            "failure_message": str(error),
        })
        _atomic_json(manifest_path, manifest)
        raise


def controlled_validation_rows() -> tuple[list[dict[str, object]], bool]:
    rows: list[dict[str, object]] = []

    def record(fixture: str, passed: bool, **values: object) -> None:
        rows.append({"fixture": fixture, "passed": bool(passed), **values})

    operator = np.asarray([
        [0.0, 1.0, 0.2, 0.0],
        [1.0, 0.0, 0.0, 0.3],
        [0.2, 0.0, 0.0, 1.0],
        [0.0, 0.3, 1.0, 0.0],
    ])
    raw = np.column_stack([
        np.asarray([1.0, 1.0, 0.0, 0.0]) / math.sqrt(2.0),
        np.asarray([0.0, 0.0, 1.0, 1.0]) / math.sqrt(2.0),
    ])
    orth = ordered_residual_orthogonalization(raw, 1e-10)
    u = np.asarray(orth["states"])
    k = u.T @ operator @ u
    residual = operator @ u - u @ k
    mu, y = linalg.eigh(k)
    lifted = u @ y
    direct = operator @ lifted - lifted * mu[None, :]
    projector = u @ u.T
    rectangular_pass = bool(
        u.shape == (4, 2)
        and np.linalg.norm(u.T @ u - np.eye(2)) <= 1e-12
        and np.linalg.norm(projector @ projector - projector) <= 1e-12
        and np.linalg.norm(direct - residual @ y) <= 1e-12
    )
    record(
        "rectangular_basis_projector_action_and_lifted_residual_identity",
        rectangular_pass,
        orthonormality_error=float(np.linalg.norm(u.T @ u - np.eye(2))),
        projector_idempotence_error=float(np.linalg.norm(projector @ projector - projector)),
        lifted_residual_identity_error=float(np.linalg.norm(direct - residual @ y)),
    )

    strong = np.asarray([[0.0, 2.5], [2.5, 0.4]])
    invariant = ordered_residual_orthogonalization(np.eye(2), 1e-10)
    invariant_u = np.asarray(invariant["states"])
    invariant_k = invariant_u.T @ strong @ invariant_u
    invariant_r = strong @ invariant_u - invariant_u @ invariant_k
    record(
        "invariant_two_mode_space_strong_internal_coupling",
        bool(abs(invariant_k[0, 1]) > 1.0 and np.linalg.norm(invariant_r) <= 1e-12),
        absolute_internal_coupling=float(abs(invariant_k[0, 1])),
        action_escape_norm=float(np.linalg.norm(invariant_r)),
    )

    block = sparse.block_diag(
        [np.asarray([[0.0, 1.0], [1.0, 0.0]]), np.asarray([[0.0, 1.0], [1.0, 0.0]])],
        format="csr",
    )
    positive = np.column_stack([
        np.asarray([1.0, 1.0, 0.0, 0.0]) / math.sqrt(2.0),
        np.asarray([0.0, 0.0, 1.0, 1.0]) / math.sqrt(2.0),
    ])
    omitted_sign_target = np.asarray([1.0, -1.0, 0.0, 0.0]) / math.sqrt(2.0)
    omitted_capture = float(np.sum(np.square(positive.T @ omitted_sign_target)))
    positive_k = positive.T @ block.dot(positive)
    record(
        "disconnected_regions_omitted_local_sign_mode",
        bool(omitted_capture <= 1e-12 and abs(float(positive_k[0, 1])) <= 1e-12),
        captured_mass=omitted_capture,
        interregion_coupling=float(positive_k[0, 1]),
    )

    weak_raw = np.column_stack([
        np.asarray([1.0, 0.0, 0.12]),
        np.asarray([0.0, 1.0, 0.12]),
    ])
    weak_operator = np.asarray([[1.0, 0.25, 0.0], [0.25, 0.7, 0.0], [0.0, 0.0, -0.2]])
    spatial = symmetric_pair_basis(weak_raw, 1e-10)
    canonical_gram_values, canonical_gram_vectors = linalg.eigh(weak_raw.T @ weak_raw)
    canonical_u = weak_raw @ canonical_gram_vectors @ np.diag(1.0 / np.sqrt(canonical_gram_values))
    spatial_k = np.asarray(spatial["states"]).T @ weak_operator @ np.asarray(spatial["states"])
    canonical_k = canonical_u.T @ weak_operator @ canonical_u
    orientation = np.asarray([
        np.sum(np.square(np.asarray(spatial["states"])[[0, 2], 0])),
        np.sum(np.square(np.asarray(spatial["states"])[[1, 2], 1])),
    ])
    record(
        "weakly_overlapping_spatial_basis_vs_canonical_gram_rotation",
        bool(
            np.max(np.abs(np.linalg.eigvalsh(spatial_k) - np.linalg.eigvalsh(canonical_k))) <= 1e-12
            and abs(float(spatial_k[0, 1]) - float(canonical_k[0, 1])) > 1e-3
            and np.min(orientation) > 0.8
        ),
        projected_spectrum_error=float(np.max(np.abs(np.linalg.eigvalsh(spatial_k) - np.linalg.eigvalsh(canonical_k)))),
        spatial_coupling=float(spatial_k[0, 1]),
        canonical_coupling=float(canonical_k[0, 1]),
        assigned_region_masses=orientation.tolist(),
    )

    path3 = np.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    two_u = np.column_stack([np.eye(3)[:, 0], np.eye(3)[:, 2]])
    two_k = two_u.T @ path3 @ two_u
    three_k = np.eye(3).T @ path3 @ np.eye(3)
    two_error = float(
        np.min(
            np.abs(np.linalg.eigvalsh(path3)[:, None] - np.linalg.eigvalsh(two_k)[None, :]),
            axis=1,
        ).max()
    )
    three_error = float(np.max(np.abs(np.linalg.eigvalsh(three_k) - np.linalg.eigvalsh(path3))))
    record(
        "nearby_third_mode_changes_genuinely_enlarged_prediction",
        bool(two_error > 0.1 and three_error <= 1e-12),
        two_mode_spectral_error=two_error,
        three_mode_spectral_error=three_error,
    )

    nested_raw = np.column_stack([np.eye(5)[:, index] for index in range(5)])
    nested_target = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    nested_target /= np.linalg.norm(nested_target)
    captures = []
    for count in range(1, 6):
        nested_u = np.asarray(ordered_residual_orthogonalization(nested_raw[:, :count], 1e-10)["states"])
        captures.append(float(np.sum(np.square(nested_u.T @ nested_target))))
    record(
        "nested_enlargement_capture_monotonicity_and_complete_basis_control",
        bool(np.all(np.diff(captures) >= -1e-12) and abs(captures[-1] - 1.0) <= 1e-12),
        captures=captures,
    )

    schur = schur_effective_operator(
        sparse.csr_matrix(path3),
        two_u,
        energy=0.2,
        eta=0.1,
    )
    effective = np.asarray(schur["effective_matrix"])
    correction = effective - two_k
    record(
        "analytic_mediated_coupling_distinct_from_direct_and_total",
        bool(abs(two_k[0, 1]) <= 1e-12 and abs(correction[0, 1]) > 0.0 and abs(effective[0, 1]) > 0.0),
        direct_coupling=float(two_k[0, 1]),
        mediated_correction_real=float(np.real(correction[0, 1])),
        total_effective_coupling_real=float(np.real(effective[0, 1])),
        linear_solve_residual_norm=float(schur["linear_solve_residual_norm"]),
    )

    duplicate = symmetric_pair_basis(np.column_stack([np.eye(2)[:, 0], np.eye(2)[:, 0]]), 1e-8)
    record(
        "rank_one_duplicate_pair_status",
        bool(not duplicate["valid"] and duplicate["status"] == "rank_one_or_duplicate_pair"),
        status=duplicate["status"],
    )

    gap_rows, ratio_rows, _, _ = _spacing_rows(
        np.asarray([-1.0, 0.0, 0.0, 0.0, 1.0]),
        case_prefix={"fixture": "zero_gap"},
        tolerance_factor=64.0,
    )
    zero_over_zero = [row for row in ratio_rows if float(row["left_gap"]) == 0.0 and float(row["right_gap"]) == 0.0]
    undefined = all(not math.isfinite(float(row["raw_gap_ratio_all_levels"])) for row in zero_over_zero)
    record(
        "zero_gap_and_near_degenerate_ratio_handling",
        bool(undefined and any(not bool(row["numerically_resolved"]) for row in gap_rows)),
        zero_over_zero_row_count=len(zero_over_zero),
    )

    windows = _cumulative_and_disjoint_windows(
        500,
        [10, 25, 50, 100, 200, 300, 400, 500],
    )
    nonperron_500 = [
        window
        for family, _, window in windows
        if family == "cumulative_nonperron" and int(window["requested_count"]) == 500
    ][0]
    record(
        "n500_nonperron_top500_clipping_and_ten_gap_index_count",
        bool(
            int(nonperron_500["state_count"]) == 499
            and int(nonperron_500["last_decreasing_rank"]) == 500
            and len(range(2, min(500, 12))) == 10
        ),
        nonperron_actual_count=int(nonperron_500["state_count"]),
        nonperron_last_rank=int(nonperron_500["last_decreasing_rank"]),
        nonperron_first_ten_gap_count=len(range(2, min(500, 12))),
    )

    distance_graph = nx.Graph([(0, 2), (2, 1)])
    induced = distance_graph.subgraph([0, 1])
    induced_connected = nx.has_path(induced, 0, 1)
    full_distance = nx.shortest_path_length(distance_graph, 0, 1)
    record(
        "disconnected_induced_distance_but_finite_original_graph_distance",
        bool(not induced_connected and full_distance == 2),
        induced_unreachable=not induced_connected,
        full_graph_distance=full_distance,
    )

    seed_a = stable_graph_seed(20260924, 500, 0, 1.05)
    seed_b = stable_graph_seed(20260924, 500, 0, 1.05)
    seed_c = stable_graph_seed(20260924, 10000, 0, 1.05)
    record(
        "stable_seed_derivation_and_size_separation",
        bool(seed_a == seed_b and seed_a != seed_c),
        n500_seed=seed_a,
        n10000_seed=seed_c,
    )

    rejection_graph = nx.path_graph(8)
    construction = greedy_mass_regions(
        rejection_graph,
        np.eye(8)[:, 0],
        np.eye(8)[:, 7],
        [0, 7],
        radius=2,
        x=0.9,
        y=0.1,
        maximum_centers=8,
    )
    record(
        "rejection_provenance_preserved",
        bool(not construction["coverage_pass"] and construction["rejections"]),
        stopping_reason=construction["stopping_reason"],
        rejection_reasons=sorted(set(row["rejection_reason"] for row in construction["rejections"])),
    )
    return rows, bool(rows and all(bool(row["passed"]) for row in rows))
