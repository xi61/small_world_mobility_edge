from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

try:
    from .regular_baseline import adjacency_eigendecomposition
except ImportError:  # pragma: no cover - script-style fallback
    from regular_baseline import adjacency_eigendecomposition


@dataclass(frozen=True)
class ErdosRenyiSpectrum:
    graph_index: int
    seed: int
    n_requested: int
    average_degree_requested: float
    p: float
    component_mode: str
    analyzed_dimension: int
    realized_mean_degree: float
    num_components: int
    largest_component_size: int
    num_isolated_vertices: int
    degrees: np.ndarray
    eigenvalues: np.ndarray
    bulk_eigenvalues: np.ndarray
    perron_eigenvalue: float


@dataclass(frozen=True)
class ErdosRenyiEdgeLevels:
    graph_index: int
    seed: int
    component_mode: str
    analyzed_dimension: int
    average_degree_requested: float
    normalized_by_sqrt_d: bool
    positive_eigenvalues: np.ndarray
    positive_vectors: np.ndarray | None
    negative_eigenvalues: np.ndarray
    zero_multiplicity: int
    zero_gap_count_near_center: int
    negative_vectors: np.ndarray | None = None


def generate_erdos_renyi_graph(
    n: int,
    average_degree: float,
    seed: int | np.random.Generator | None = None,
) -> nx.Graph:
    if n <= 1:
        raise ValueError("n must be at least 2")
    if average_degree <= 0:
        raise ValueError("average_degree must be positive")
    p = average_degree / (n - 1)
    p = min(max(p, 0.0), 1.0)
    return nx.fast_gnp_random_graph(n=n, p=p, seed=seed)


def adjacency_eigenvalues_only(graph: nx.Graph) -> np.ndarray:
    eigenvalues, _ = adjacency_eigendecomposition(graph)
    return np.sort(np.asarray(eigenvalues, dtype=float))


def adjacency_sparse_matrix(graph: nx.Graph) -> sparse.csr_matrix:
    return nx.to_scipy_sparse_array(graph, dtype=float, format="csr")


def bulk_eigenvalues_excluding_perron(eigenvalues: np.ndarray) -> tuple[np.ndarray, float]:
    eigenvalues = np.sort(np.asarray(eigenvalues, dtype=float))
    if len(eigenvalues) < 2:
        raise ValueError("need at least two eigenvalues to exclude the top Perron outlier")
    return eigenvalues[:-1], float(eigenvalues[-1])


def normalized_adjacency_sparse_matrix(
    graph: nx.Graph,
    average_degree: float,
) -> sparse.csr_matrix:
    if average_degree <= 0:
        raise ValueError("average_degree must be positive")
    return adjacency_sparse_matrix(graph) / np.sqrt(float(average_degree))


def flat_vector_overlap_index(eigenvectors: np.ndarray) -> int:
    eigenvectors = np.asarray(eigenvectors, dtype=float)
    if eigenvectors.ndim != 2 or eigenvectors.shape[1] == 0:
        raise ValueError("eigenvectors must be a 2D array with at least one eigenvector")
    n = eigenvectors.shape[0]
    flat = np.ones(n, dtype=float) / np.sqrt(n)
    overlaps = np.abs(eigenvectors.T @ flat)
    return int(np.argmax(overlaps))


def sparse_normalized_edge_levels(
    graph: nx.Graph,
    average_degree: float,
    edge_levels: int,
    graph_index: int,
    seed: int,
    component_mode: str,
    dense_threshold: int = 3000,
    zero_tolerance: float = 1e-10,
) -> ErdosRenyiEdgeLevels:
    if edge_levels < 2:
        raise ValueError("edge_levels must be at least 2")
    n = graph.number_of_nodes()
    normalized_matrix = normalized_adjacency_sparse_matrix(graph, average_degree=average_degree)

    if n <= dense_threshold:
        dense_matrix = normalized_matrix.toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(dense_matrix)
        positive_eigenvalues = np.sort(eigenvalues)[-edge_levels:]
        negative_eigenvalues = np.sort(eigenvalues)[:edge_levels]
        positive_indices = np.argsort(eigenvalues)[-edge_levels:]
        negative_indices = np.argsort(eigenvalues)[:edge_levels]
        positive_vectors = eigenvectors[:, positive_indices]
        negative_vectors = eigenvectors[:, negative_indices]
        zero_multiplicity = int(np.sum(np.isclose(eigenvalues, 0.0, atol=zero_tolerance, rtol=0.0)))
        center_order = np.argsort(np.abs(eigenvalues))[: min(50, len(eigenvalues))]
        center_eigenvalues = np.sort(eigenvalues[center_order])
        center_gaps = np.diff(center_eigenvalues)
        zero_gap_count = int(np.sum(np.isclose(center_gaps, 0.0, atol=zero_tolerance, rtol=0.0)))
    else:
        k_upper = min(edge_levels, max(1, n - 2))
        k_lower = min(edge_levels, max(1, n - 2))
        upper_values, upper_vectors = eigsh(normalized_matrix, k=k_upper, which="LA")
        lower_values, lower_vectors = eigsh(normalized_matrix, k=k_lower, which="SA")
        positive_order = np.argsort(upper_values)
        lower_order = np.argsort(lower_values)
        positive_eigenvalues = np.asarray(upper_values[positive_order], dtype=float)
        positive_vectors = np.asarray(upper_vectors[:, positive_order], dtype=float)
        negative_eigenvalues = np.asarray(lower_values[lower_order], dtype=float)
        negative_vectors = np.asarray(lower_vectors[:, lower_order], dtype=float)
        center_count = min(max(4, 2 * edge_levels), 20, max(2, n - 2))
        try:
            center_values = eigsh(normalized_matrix, k=center_count, sigma=1e-8, which="LM", return_eigenvectors=False)
        except (ArpackNoConvergence, RuntimeError):
            try:
                center_values = eigsh(normalized_matrix, k=center_count, which="SM", return_eigenvectors=False)
            except ArpackNoConvergence as exc:
                center_values = getattr(exc, "eigenvalues", np.array([], dtype=float))
            except RuntimeError:
                center_values = np.array([], dtype=float)
        center_values = np.sort(np.asarray(center_values, dtype=float))
        zero_multiplicity = int(np.sum(np.isclose(center_values, 0.0, atol=zero_tolerance, rtol=0.0)))
        center_gaps = np.diff(center_values)
        zero_gap_count = int(np.sum(np.isclose(center_gaps, 0.0, atol=zero_tolerance, rtol=0.0)))

    return ErdosRenyiEdgeLevels(
        graph_index=graph_index,
        seed=seed,
        component_mode=component_mode,
        analyzed_dimension=n,
        average_degree_requested=float(average_degree),
        normalized_by_sqrt_d=True,
        positive_eigenvalues=np.sort(np.asarray(positive_eigenvalues, dtype=float)),
        positive_vectors=positive_vectors,
        negative_eigenvalues=np.sort(np.asarray(negative_eigenvalues, dtype=float)),
        zero_multiplicity=zero_multiplicity,
        zero_gap_count_near_center=zero_gap_count,
        negative_vectors=negative_vectors,
    )


def analyze_erdos_renyi_graph(
    n: int,
    average_degree: float,
    seed: int,
    component_mode: str = "full",
    graph_index: int = 0,
) -> ErdosRenyiSpectrum:
    graph = generate_erdos_renyi_graph(n=n, average_degree=average_degree, seed=seed)
    num_components = nx.number_connected_components(graph)
    component_sizes = sorted((len(component) for component in nx.connected_components(graph)), reverse=True)
    largest_component_size = component_sizes[0]
    degrees = np.array([degree for _, degree in graph.degree()], dtype=float)
    num_isolated_vertices = int(np.sum(degrees == 0))
    realized_mean_degree = float(np.mean(degrees))
    p = average_degree / (n - 1)

    if component_mode == "full":
        analyzed_graph = graph
    elif component_mode == "largest_connected_component":
        largest_nodes = max(nx.connected_components(graph), key=len)
        analyzed_graph = graph.subgraph(largest_nodes).copy()
    else:
        raise ValueError(f"unsupported component_mode={component_mode!r}")

    eigenvalues = adjacency_eigenvalues_only(analyzed_graph)
    bulk_eigenvalues, perron_eigenvalue = bulk_eigenvalues_excluding_perron(eigenvalues)
    return ErdosRenyiSpectrum(
        graph_index=graph_index,
        seed=seed,
        n_requested=n,
        average_degree_requested=average_degree,
        p=p,
        component_mode=component_mode,
        analyzed_dimension=analyzed_graph.number_of_nodes(),
        realized_mean_degree=realized_mean_degree,
        num_components=num_components,
        largest_component_size=largest_component_size,
        num_isolated_vertices=num_isolated_vertices,
        degrees=degrees,
        eigenvalues=eigenvalues,
        bulk_eigenvalues=bulk_eigenvalues,
        perron_eigenvalue=perron_eigenvalue,
    )


def generate_erdos_renyi_ensemble(
    n: int,
    average_degree: float,
    num_graphs: int,
    seed: int,
    component_mode: str = "full",
) -> list[ErdosRenyiSpectrum]:
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**32 - 1, size=num_graphs, dtype=np.uint32)
    return generate_erdos_renyi_ensemble_from_seeds(
        n=n,
        average_degree=average_degree,
        seeds=[int(graph_seed) for graph_seed in seeds],
        component_mode=component_mode,
    )


def generate_erdos_renyi_ensemble_from_seeds(
    n: int,
    average_degree: float,
    seeds: list[int],
    component_mode: str = "full",
) -> list[ErdosRenyiSpectrum]:
    return [
        analyze_erdos_renyi_graph(
            n=n,
            average_degree=average_degree,
            seed=int(graph_seed),
            component_mode=component_mode,
            graph_index=index,
        )
        for index, graph_seed in enumerate(seeds)
    ]
