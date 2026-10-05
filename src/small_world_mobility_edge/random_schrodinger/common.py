"""Shared provenance, graph, and deterministic-selection helpers."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import networkx as nx
import numpy as np
import scipy
from scipy import sparse

from . import ANALYSIS_VERSION


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_json_hash(payload: Mapping[str, object]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=True)
    return sha256_text(text)


def code_revision(project_root: Path) -> dict[str, object]:
    """Return Git provenance when available, otherwise a source-tree digest."""
    root = Path(project_root).resolve()
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
        return {"kind": "git", "commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        files = sorted((root / "src").rglob("*.py")) + sorted((root / "tools").glob("*.py"))
        digest = hashlib.sha256()
        for path in files:
            digest.update(str(path.relative_to(root)).encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        return {
            "kind": "source_tree_sha256",
            "sha256": digest.hexdigest(),
            "file_count": len(files),
        }


def source_configuration_identity(project_root: Path) -> dict[str, object]:
    """Hash the complete executable analysis definition used by new runs.

    This deliberately includes Python source, shell launchers, tests, and JSON
    configurations.  It is separate from ``code_revision`` so historical
    manifests keep their original semantics while corrected caches can reject
    reuse after any scientifically relevant implementation change.
    """
    root = Path(project_root).resolve()
    files: set[Path] = set()
    for directory, suffixes in (
        (root / "src", {".py"}),
        (root / "tools", {".py", ".sh"}),
        (root / "tests", {".py"}),
        (root / "experiments", {".json"}),
    ):
        if directory.is_dir():
            files.update(
                path
                for path in directory.rglob("*")
                if path.is_file()
                and path.suffix in suffixes
                and "__pycache__" not in path.parts
            )
    for name in ("pyproject.toml", "README.md"):
        path = root / name
        if path.is_file():
            files.add(path)
    ordered = sorted(files, key=lambda path: str(path.relative_to(root)))
    digest = hashlib.sha256()
    for path in ordered:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {
        "kind": "complete_source_configuration_sha256",
        "sha256": digest.hexdigest(),
        "file_count": len(ordered),
        "scope": "src/**/*.py; tools/**/*.{py,sh}; tests/**/*.py; experiments/**/*.json; pyproject.toml; README.md",
    }


def runtime_metadata(project_root: Path) -> dict[str, object]:
    return {
        "analysis_version": ANALYSIS_VERSION,
        "started_at_utc": utc_now(),
        "status": "RUNNING",
        "host": platform.node(),
        "job_id": os.environ.get("JOB_ID"),
        "sge_task_id": os.environ.get("SGE_TASK_ID"),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "networkx_version": nx.__version__,
        "code_revision": code_revision(project_root),
        "source_configuration_identity": source_configuration_identity(project_root),
    }


def _json_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (complex, np.complexfloating)):
        return {"real": float(np.real(value)), "imag": float(np.imag(value))}
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(payload), indent=2, sort_keys=True, allow_nan=True) + "\n")


def read_json(path: Path) -> dict[str, object]:
    with Path(path).open() as handle:
        return json.load(handle)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    fields.extend(sorted({key for row in rows for key in row.keys()} - set(fields)))
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(_json_value(value), sort_keys=True, allow_nan=True)
                    if isinstance(value, (dict, list, tuple, np.ndarray))
                    else _json_value(value)
                    for key, value in row.items()
                }
            )


def graph_from_edge_cache(path: Path) -> tuple[nx.Graph, dict[str, object]]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        required = {"edge_u", "edge_v", "degrees", "n"}
        missing = required - set(data.files)
        if missing:
            raise ValueError("graph cache missing keys: %s" % sorted(missing))
        n = int(data["n"])
        edge_u = np.asarray(data["edge_u"], dtype=int)
        edge_v = np.asarray(data["edge_v"], dtype=int)
        cached_degrees = np.asarray(data["degrees"], dtype=int)
        original_vertex_ids = (
            np.asarray(data["original_vertex_ids"], dtype=np.int64)
            if "original_vertex_ids" in data.files
            else np.arange(n, dtype=np.int64)
        )
        scalar_metadata = {
            key: _json_value(data[key])
            for key in data.files
            if np.asarray(data[key]).ndim == 0
        }
    if (
        len(edge_u) != len(edge_v)
        or cached_degrees.shape != (n,)
        or original_vertex_ids.shape != (n,)
    ):
        raise ValueError("invalid graph cache dimensions")
    if len(np.unique(original_vertex_ids)) != n:
        raise ValueError("original vertex IDs are not one-to-one")
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(zip(edge_u.tolist(), edge_v.tolist()))
    if graph.is_multigraph() or graph.is_directed() or nx.number_of_selfloops(graph):
        raise ValueError("cached graph is not simple undirected")
    if set(graph.nodes()) != set(range(n)):
        raise ValueError("cached vertex mapping is not identity 0..N-1")
    degrees = np.asarray([graph.degree(vertex) for vertex in range(n)], dtype=int)
    if not np.array_equal(degrees, cached_degrees):
        raise ValueError("cached degrees do not match cached edges/vertex ordering")
    graph.graph["original_vertex_ids"] = original_vertex_ids.copy()
    original_id_sha256 = hashlib.sha256(original_vertex_ids.tobytes(order="C")).hexdigest()
    nonidentity_count = int(np.count_nonzero(original_vertex_ids != np.arange(n)))
    metadata = {
        "path": str(path),
        "sha256": sha256_file(path),
        "n": n,
        "num_edges": graph.number_of_edges(),
        "vertex_ordering": "analyzed_identity_0_to_N_minus_1",
        "vertex_mapping": (
            "identity_analyzed_equals_original"
            if nonidentity_count == 0
            else "explicit_analyzed_index_to_original_vertex_id"
        ),
        "original_vertex_id_count": n,
        "original_vertex_ids_nonidentity_count": nonidentity_count,
        "original_vertex_ids_sha256_raw_int64": original_id_sha256,
        "component_count": nx.number_connected_components(graph),
        **scalar_metadata,
    }
    return graph, metadata


def analyzed_to_original_ids(
    graph: nx.Graph, vertices: Sequence[int] | None = None
) -> np.ndarray:
    """Map analyzed coordinates to the original graph's vertex identifiers."""
    mapping = np.asarray(
        graph.graph.get("original_vertex_ids", np.arange(graph.number_of_nodes())),
        dtype=np.int64,
    )
    if mapping.shape != (graph.number_of_nodes(),):
        raise ValueError("graph original-vertex mapping has incompatible shape")
    if vertices is None:
        return mapping.copy()
    indices = np.asarray(vertices, dtype=int)
    if np.any(indices < 0) or np.any(indices >= len(mapping)):
        raise IndexError("analyzed vertex index outside mapping")
    return mapping[indices]


def graph_to_csr(graph: nx.Graph, nodelist: Sequence[int] | None = None) -> sparse.csr_matrix:
    nodes = list(graph.nodes()) if nodelist is None else [int(value) for value in nodelist]
    if hasattr(nx, "to_scipy_sparse_array"):
        matrix = nx.to_scipy_sparse_array(graph, nodelist=nodes, dtype=float, format="csr")
    else:  # NetworkX 2.6 on Hoffman
        matrix = nx.to_scipy_sparse_matrix(graph, nodelist=nodes, dtype=float, format="csr")
    return sparse.csr_matrix(matrix)


def save_graph_cache(
    path: Path,
    graph: nx.Graph,
    original_vertex_ids: Sequence[int],
    metadata: Mapping[str, object],
) -> None:
    nodes = list(graph.nodes())
    if nodes != list(range(len(nodes))):
        raise ValueError("save_graph_cache expects analyzed labels 0..N_eff-1")
    edges = np.asarray(sorted((int(u), int(v)) for u, v in graph.edges()), dtype=np.int64)
    if edges.size == 0:
        edges = np.empty((0, 2), dtype=np.int64)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        n=np.asarray(len(nodes), dtype=np.int64),
        edge_u=edges[:, 0],
        edge_v=edges[:, 1],
        original_vertex_ids=np.asarray(original_vertex_ids, dtype=np.int64),
        degrees=np.asarray([graph.degree(node) for node in nodes], dtype=np.int64),
        metadata_json=np.asarray(json.dumps(_json_value(metadata), sort_keys=True)),
    )


def deterministic_mass_order(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vector = np.asarray(vector)
    masses = np.square(np.abs(vector))
    vertex_ids = np.arange(len(masses), dtype=int)
    order = np.lexsort((vertex_ids, -masses))
    return order.astype(int), masses


def selected_mass_sets(
    vector: np.ndarray,
    counts: Sequence[int] = (10, 25, 50),
    fractions: Sequence[float] = (0.5, 0.8),
) -> dict[str, np.ndarray]:
    order, masses = deterministic_mass_order(vector)
    result: dict[str, np.ndarray] = {}
    for count in counts:
        result["top_%d" % count] = order[: min(int(count), len(order))]
    cumulative = np.cumsum(masses[order])
    total = float(cumulative[-1]) if len(cumulative) else 0.0
    for fraction in fractions:
        if total <= 0.0:
            size = 0
        else:
            size = int(np.searchsorted(cumulative, float(fraction) * total, side="left") + 1)
        result["mass_%02d" % int(round(100 * fraction))] = order[:size]
    return result


class BFSDistanceCache:
    """Lazy unweighted BFS cache keyed by source vertex."""

    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self._rows: dict[int, np.ndarray] = {}

    def row_array(self, source: int) -> np.ndarray:
        """Return int32 distances, using -1 for unreachable vertices.

        Dense integer rows avoid the very large Python-object overhead of
        retaining thousands of BFS dictionaries when a mass-threshold set is
        large.  Rows are still computed lazily only for requested sources.
        """
        source = int(source)
        if source not in self._rows:
            result = np.full(self.graph.number_of_nodes(), -1, dtype=np.int32)
            for vertex, distance in nx.single_source_shortest_path_length(
                self.graph, source
            ).items():
                result[int(vertex)] = int(distance)
            self._rows[source] = result
        return self._rows[source]

    def row(self, source: int) -> dict[int, int]:
        array = self.row_array(source)
        return {int(vertex): int(array[vertex]) for vertex in np.flatnonzero(array >= 0)}

    def distance(self, source: int, target: int) -> float:
        value = int(self.row_array(source)[int(target)])
        return float(value) if value >= 0 else math.inf

    def matrix(self, left: Sequence[int], right: Sequence[int] | None = None) -> np.ndarray:
        left_nodes = [int(value) for value in left]
        right_nodes = left_nodes if right is None else [int(value) for value in right]
        matrix = np.full((len(left_nodes), len(right_nodes)), np.inf, dtype=float)
        for i, source in enumerate(left_nodes):
            distances = self.row_array(source)
            values = distances[np.asarray(right_nodes, dtype=int)]
            finite = values >= 0
            matrix[i, finite] = values[finite]
        return matrix

    @property
    def source_count(self) -> int:
        return len(self._rows)


def pair_distance_rows(
    vertices: Sequence[int],
    masses: Sequence[float],
    matrix: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    vertices = np.asarray(vertices, dtype=int)
    masses = np.asarray(masses, dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (len(vertices), len(vertices)):
        raise ValueError("distance matrix shape does not match selected vertices")
    rows: list[dict[str, object]] = []
    finite_values: list[float] = []
    finite_weighted_sum = 0.0
    finite_weight_denominator = 0.0
    for left in range(len(vertices)):
        for right in range(left + 1, len(vertices)):
            distance = float(matrix[left, right])
            finite = math.isfinite(distance)
            weight = float(masses[left] * masses[right])
            rows.append(
                {
                    "left_support_rank": left + 1,
                    "right_support_rank": right + 1,
                    "left_vertex_id": int(vertices[left]),
                    "right_vertex_id": int(vertices[right]),
                    "left_mass": float(masses[left]),
                    "right_mass": float(masses[right]),
                    "pair_mass_weight": weight,
                    "distance": distance,
                    "unreachable": not finite,
                }
            )
            if finite:
                finite_values.append(distance)
                finite_weighted_sum += weight * distance
                finite_weight_denominator += weight
    finite_array = np.asarray(finite_values, dtype=float)
    summary = {
        "pair_count": len(rows),
        "finite_pair_count": len(finite_array),
        "unreachable_pair_count": len(rows) - len(finite_array),
        "unreachable_fraction": float(1.0 - len(finite_array) / len(rows)) if rows else 0.0,
        "finite_distance_mean": float(np.mean(finite_array)) if len(finite_array) else math.nan,
        "finite_distance_median": float(np.median(finite_array)) if len(finite_array) else math.nan,
        "finite_distance_maximum": float(np.max(finite_array)) if len(finite_array) else math.nan,
        "finite_mass_weighted_distance_numerator": finite_weighted_sum,
        "finite_mass_weighted_distance_denominator": finite_weight_denominator,
        "finite_mass_weighted_mean_distance": (
            finite_weighted_sum / finite_weight_denominator
            if finite_weight_denominator > 0.0
            else math.nan
        ),
    }
    return rows, summary


def component_ids(graph: nx.Graph) -> np.ndarray:
    result = np.full(graph.number_of_nodes(), -1, dtype=int)
    components = sorted(nx.connected_components(graph), key=lambda part: (-len(part), min(part)))
    for component_id, nodes in enumerate(components):
        result[np.asarray(sorted(nodes), dtype=int)] = component_id
    if np.any(result < 0):
        raise AssertionError("component labeling failed")
    return result


def core_numbers(graph: nx.Graph) -> np.ndarray:
    values = nx.core_number(graph)
    return np.asarray([values[node] for node in range(graph.number_of_nodes())], dtype=int)


def mass_overlap(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sum(np.minimum(np.square(np.abs(left)), np.square(np.abs(right)))))


def relative_residual(matrix: sparse.spmatrix, value: float, vector: np.ndarray) -> tuple[float, float]:
    residual = np.asarray(matrix.dot(vector)).reshape(-1) - float(value) * np.asarray(vector)
    absolute = float(np.linalg.norm(residual))
    return absolute, absolute / max(1.0, abs(float(value)))


def hill_ccdf_exponent(degrees: Sequence[float], tail_fraction: float = 0.1) -> dict[str, float]:
    values = np.sort(np.asarray(degrees, dtype=float))[::-1]
    positive = values[values > 0.0]
    k = min(max(10, int(round(tail_fraction * len(positive)))), max(1, len(positive) - 1))
    if len(positive) < 11 or k < 2:
        return {"hill_k": float(k), "degree_ccdf_alpha_hill": math.nan, "threshold": math.nan}
    threshold = positive[k]
    logs = np.log(positive[:k] / threshold)
    denominator = float(np.sum(logs))
    alpha = float(k / denominator) if denominator > 0.0 else math.nan
    return {"hill_k": float(k), "degree_ccdf_alpha_hill": alpha, "threshold": float(threshold)}


def temporary_matplotlib_dir(prefix: str) -> str:
    root = Path(tempfile.gettempdir()) / prefix
    root.mkdir(parents=True, exist_ok=True)
    return str(root)
