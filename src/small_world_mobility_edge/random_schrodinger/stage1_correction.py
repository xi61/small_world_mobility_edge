"""Independent Stage 1 corrections built from cached graph/eigenvector data."""

from __future__ import annotations

import csv
import gzip
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/random-schrodinger-correction-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/random-schrodinger-correction-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .common import (
    analyzed_to_original_ids,
    deterministic_mass_order,
    runtime_metadata,
    sha256_file,
    source_configuration_identity,
    stable_json_hash,
    utc_now,
    write_csv,
    write_json,
)
from .geometry import load_stage1_cache


STAGE1_CORRECTION_VERSION = "stage1_geometry_correction_v1"


def _read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_json_list(value: str) -> list[int]:
    parsed = json.loads(value) if value else []
    return [int(item) for item in parsed]


def _local_mass_peaks(graph: nx.Graph, masses: np.ndarray, order: np.ndarray) -> list[int]:
    peaks: list[int] = []
    for vertex_value in order[: min(50, len(order))]:
        vertex = int(vertex_value)
        mass = float(masses[vertex])
        if all(
            float(masses[int(neighbor)]) < mass
            or (float(masses[int(neighbor)]) == mass and int(neighbor) > vertex)
            for neighbor in graph.neighbors(vertex)
        ):
            peaks.append(vertex)
    if not peaks and len(order):
        peaks.append(int(order[0]))
    return peaks


def _corrected_path_figure(
    graph: nx.Graph,
    support: np.ndarray,
    masses: np.ndarray,
    centers: list[int],
    route: list[int],
    original_ids: np.ndarray,
    title: str,
    path: Path,
    seed: int,
) -> None:
    support_set = set(int(value) for value in support)
    center_set = set(int(value) for value in centers)
    route_set = set(int(value) for value in route)
    display_nodes = sorted(support_set | route_set | center_set)
    display = graph.subgraph(display_nodes).copy()
    positions = nx.spring_layout(display, seed=int(seed) % (2**32 - 1))
    route_edges = {
        tuple(sorted((int(left), int(right))))
        for left, right in zip(route[:-1], route[1:])
    }
    figure, axis = plt.subplots(figsize=(7.6, 6.4))
    ordinary = [edge for edge in display.edges() if tuple(sorted(edge)) not in route_edges]
    nx.draw_networkx_edges(display, positions, edgelist=ordinary, alpha=0.25, width=0.9, ax=axis)
    nx.draw_networkx_edges(
        display,
        positions,
        edgelist=list(route_edges),
        edge_color="#fdae61",
        width=2.8,
        label="recomputed shortest path",
        ax=axis,
    )
    interiors = sorted(route_set - support_set)
    if interiors:
        nx.draw_networkx_nodes(
            display,
            positions,
            nodelist=interiors,
            node_shape="s",
            node_color="#bdbdbd",
            node_size=150,
            label="path interior",
            ax=axis,
        )
    noncenters = [int(value) for value in support if int(value) not in center_set]
    mass_by_vertex = {int(vertex): float(mass) for vertex, mass in zip(support, masses)}
    scale = max(mass_by_vertex.values()) if mass_by_vertex else 1.0
    if noncenters:
        nx.draw_networkx_nodes(
            display,
            positions,
            nodelist=noncenters,
            node_color="#4575b4",
            edgecolors="black",
            node_size=[230.0 + 1500.0 * mass_by_vertex[node] / scale for node in noncenters],
            label="top-10 mass vertex",
            ax=axis,
        )
    visible_centers = sorted(center_set & set(display_nodes))
    if visible_centers:
        nx.draw_networkx_nodes(
            display,
            positions,
            nodelist=visible_centers,
            node_shape="*",
            node_color="#d73027",
            edgecolors="black",
            node_size=520,
            label="rule-selected center",
            ax=axis,
        )
    labels = {node: str(int(original_ids[node])) for node in display_nodes}
    nx.draw_networkx_labels(display, positions, labels=labels, font_size=7, ax=axis)
    axis.set_title(title + "\nLabels are original vertex IDs; route independently recomputed")
    axis.axis("off")
    axis.legend(frameon=False, fontsize=7, loc="best")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def run_stage1_correction(
    *,
    source_dir: Path,
    original_stage1_dir: Path,
    output_dir: Path,
    project_root: Path,
) -> Path:
    started = time.time()
    source_dir = Path(source_dir)
    original_stage1_dir = Path(original_stage1_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = load_stage1_cache(source_dir)
    graph: nx.Graph = source["graph"]
    n = graph.number_of_nodes()
    original_ids = analyzed_to_original_ids(graph)
    vectors = np.asarray(source["eigenvectors"], dtype=float)
    global_indices = np.asarray(source["global_indices"], dtype=int)
    column_for_index = {int(index): position for position, index in enumerate(global_indices)}
    source_configuration = source_configuration_identity(project_root)
    required_original = [
        original_stage1_dir / "tables" / "top10_pair_distances.csv",
        original_stage1_dir / "tables" / "center_rule_sensitivity.csv",
        original_stage1_dir / "run_manifest.json",
    ]
    for path in required_original:
        if not path.is_file():
            raise FileNotFoundError(path)
    input_hashes = {
        "source_graph": source["hashes"]["graph_cache_sha256"],
        "source_vectors": source["hashes"]["selected_eigenvectors_sha256"],
        **{"original_" + path.name: sha256_file(path) for path in required_original},
    }
    cache_id = stable_json_hash(
        {
            "version": STAGE1_CORRECTION_VERSION,
            "source_configuration_id": source_configuration["sha256"],
            "input_hashes": input_hashes,
        }
    )
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("status") == "COMPLETE" and existing.get("cache_id") == cache_id:
            return output_dir
        raise ValueError("existing corrected Stage 1 output has a different cache identity")
    manifest: dict[str, object] = {
        **runtime_metadata(project_root),
        "status": "RUNNING",
        "stage": "stage1_independent_geometry_correction",
        "stage_version": STAGE1_CORRECTION_VERSION,
        "cache_id": cache_id,
        "source_configuration_id": source_configuration["sha256"],
        "source_dir": str(source_dir),
        "original_stage1_dir": str(original_stage1_dir),
        "output_dir": str(output_dir),
        "input_hashes": input_hashes,
        "historical_outputs_modified": False,
    }
    write_json(manifest_path, manifest)
    try:
        old_distance_rows = _read_csv(required_original[0])
        old_distance = {
            (
                int(row["upper_edge_rank"]),
                min(int(row["left_vertex_id"]), int(row["right_vertex_id"])),
                max(int(row["left_vertex_id"]), int(row["right_vertex_id"])),
            ): float(row["distance"])
            for row in old_distance_rows
        }
        path_rows: list[dict[str, object]] = []
        rank_payload: dict[int, tuple[np.ndarray, np.ndarray, list[int]]] = {}
        primary_centers: dict[int, list[int]] = {}
        sensitivity_rows = _read_csv(required_original[1])
        label_counts: dict[int, dict[str, int]] = {}
        for row in sensitivity_rows:
            rank = int(row["upper_edge_rank"])
            label = row["geometry_label"]
            label_counts.setdefault(rank, {})[label] = label_counts.setdefault(rank, {}).get(label, 0) + 1
            if int(row["radius"]) == 2 and int(row["peak_separation"]) == 3:
                primary_centers[rank] = _parse_json_list(row["centers"])

        for rank in range(1, 12):
            index = n - rank
            vector = vectors[:, column_for_index[index]]
            order, all_masses = deterministic_mass_order(vector)
            support = np.asarray(order[:10], dtype=int)
            support_masses = all_masses[support]
            recomputed: list[tuple[int, int, list[int]]] = []
            for left_position in range(10):
                for right_position in range(left_position + 1, 10):
                    left = int(support[left_position])
                    right = int(support[right_position])
                    route = [int(item) for item in nx.shortest_path(graph, left, right)]
                    distance = len(route) - 1
                    stored = old_distance[(rank, min(left, right), max(left, right))]
                    recomputed.append((left, right, route))
                    path_rows.append(
                        {
                            "upper_edge_rank": rank,
                            "left_support_rank": left_position + 1,
                            "right_support_rank": right_position + 1,
                            "left_analyzed_vertex_id": left,
                            "right_analyzed_vertex_id": right,
                            "left_original_vertex_id": int(original_ids[left]),
                            "right_original_vertex_id": int(original_ids[right]),
                            "stored_distance": stored,
                            "recomputed_distance": distance,
                            "distance_agrees": bool(abs(stored - distance) <= 1e-12),
                            "path_analyzed_vertex_ids": route,
                            "path_original_vertex_ids": [int(original_ids[item]) for item in route],
                        }
                    )
            farthest = max(
                recomputed,
                key=lambda item: (len(item[2]) - 1, -min(item[0], item[1]), -max(item[0], item[1])),
            )
            rank_payload[rank] = (support, support_masses, farthest[2])

        degree_order = np.lexsort((np.arange(n), -np.asarray([graph.degree(v) for v in range(n)])))
        hub_candidates = set(int(value) for value in degree_order[: min(50, n)])
        center_rows: list[dict[str, object]] = []
        reassessment_rows: list[dict[str, object]] = []
        maximum_mass_error = 0.0
        for old in sensitivity_rows:
            rank = int(old["upper_edge_rank"])
            radius = int(old["radius"])
            separation = int(old["peak_separation"])
            centers = _parse_json_list(old["centers"])
            index = n - rank
            vector = vectors[:, column_for_index[index]]
            order, masses = deterministic_mass_order(vector)
            peaks = _local_mass_peaks(graph, masses, order)
            peak_set = set(peaks)
            union: set[int] = set()
            for center_order, center in enumerate(centers, start=1):
                ball = set(
                    int(vertex)
                    for vertex in nx.single_source_shortest_path_length(
                        graph, int(center), cutoff=radius
                    )
                )
                union.update(ball)
                other_peaks = sorted((peak_set & ball) - {int(center)})
                other_hubs = sorted((hub_candidates & ball) - {int(center)})
                center_rows.append(
                    {
                        "upper_edge_rank": rank,
                        "radius": radius,
                        "peak_separation": separation,
                        "original_rule_label": old["geometry_label"],
                        "center_order": center_order,
                        "center_analyzed_vertex_id": int(center),
                        "center_original_vertex_id": int(original_ids[center]),
                        "center_degree": int(graph.degree(center)),
                        "center_mass": float(masses[center]),
                        "ball_mass": float(np.sum(masses[np.asarray(sorted(ball), dtype=int)])),
                        "ball_size": len(ball),
                        "ball_fraction_of_all_vertices": len(ball) / n,
                        "other_candidate_peak_count_in_ball": len(other_peaks),
                        "other_candidate_peak_analyzed_ids": other_peaks,
                        "other_candidate_peak_original_ids": [int(original_ids[item]) for item in other_peaks],
                        "other_top50_degree_hub_count_in_ball": len(other_hubs),
                        "other_top50_degree_hub_analyzed_ids": other_hubs,
                        "other_top50_degree_hub_original_ids": [int(original_ids[item]) for item in other_hubs],
                        "single_hub_localization_claim": "not_demonstrated_by_rule_label",
                    }
                )
            union_mass = float(np.sum(masses[np.asarray(sorted(union), dtype=int)])) if union else 0.0
            maximum_mass_error = max(maximum_mass_error, abs(union_mass - float(old["covered_mass"])))
            counts = label_counts[rank]
            reassessment_rows.append(
                {
                    "upper_edge_rank": rank,
                    "radius": radius,
                    "peak_separation": separation,
                    "original_rule_label": old["geometry_label"],
                    "original_rule_resolved": old["resolved"],
                    "center_count": len(centers),
                    "recomputed_union_ball_mass": union_mass,
                    "recomputed_union_ball_size": len(union),
                    "union_ball_fraction_of_all_vertices": len(union) / n,
                    "candidate_peak_count": len(peaks),
                    "one_center_fraction_across_nine_sensitivity_rules": counts.get("one_center", 0) / 9.0,
                    "unresolved_fraction_across_nine_sensitivity_rules": counts.get("unresolved", 0) / 9.0,
                    "corrected_interpretation": (
                        "rule_based_one_center; does_not_by_itself_demonstrate_single_hub_localization"
                        if old["geometry_label"] == "one_center"
                        else "rule_based_" + old["geometry_label"] + "; descriptive_not_mechanistic"
                    ),
                }
            )

        mapping_rows = [
            {
                "analyzed_vertex_id": index,
                "original_vertex_id": int(original_ids[index]),
                "mapping_is_identity_at_vertex": bool(index == int(original_ids[index])),
            }
            for index in range(n)
        ]
        write_csv(output_dir / "tables" / "independently_recomputed_shortest_paths.csv.gz", path_rows)
        write_csv(output_dir / "tables" / "center_ball_diagnostics.csv", center_rows)
        write_csv(output_dir / "tables" / "center_label_reassessment.csv", reassessment_rows)
        write_csv(output_dir / "tables" / "analyzed_to_original_vertex_mapping.csv.gz", mapping_rows)
        for rank, (support, support_masses, route) in rank_payload.items():
            _corrected_path_figure(
                graph,
                support,
                support_masses,
                primary_centers.get(rank, []),
                route,
                original_ids,
                "%s upper-edge rank %d" % (source_dir.name, rank),
                output_dir / "figures" / ("rank_%03d_corrected_paths.png" % rank),
                seed=int(source["source_manifest"]["graph_seed"]) + rank,
            )
        distance_failures = sum(not bool(row["distance_agrees"]) for row in path_rows)
        if distance_failures:
            raise AssertionError("independent shortest paths disagree with stored distances")
        if maximum_mass_error > 1e-12:
            raise AssertionError("recomputed center-union mass disagrees with original table")
        manifest.update(
            {
                "status": "COMPLETE",
                "finished_at_utc": utc_now(),
                "runtime_seconds": time.time() - started,
                "analyzed_n": n,
                "mapping_nonidentity_count": int(np.count_nonzero(original_ids != np.arange(n))),
                "mapping_unique": bool(len(np.unique(original_ids)) == n),
                "shortest_path_row_count": len(path_rows),
                "shortest_path_mismatch_count": distance_failures,
                "center_ball_row_count": len(center_rows),
                "label_reassessment_row_count": len(reassessment_rows),
                "maximum_center_union_mass_recompute_error": maximum_mass_error,
                "corrected_figure_count": len(rank_payload),
                "interpretation": "Original geometry labels are retained as rule outputs and are not treated as proof of single-hub localization.",
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
