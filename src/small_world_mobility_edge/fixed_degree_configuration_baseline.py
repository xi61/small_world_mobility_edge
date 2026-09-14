from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterator, Sequence
import warnings

import networkx as nx
import numpy as np

from .chung_lu_baseline import (
    WeightSequenceDiagnostics,
    generate_chung_lu_graph_from_weights,
    generate_power_law_expected_weights,
)
from .regular_baseline import adjacency_eigendecomposition


MODEL_CHUNG_LU = "Chung-Lu"
MODEL_CONFIGURATION = "Fixed-degree configuration model"
TAU_CONVENTION_DENSITY = "density_exponent_ccdf_tau_minus_one"
TAU_CONVENTION_CCDF = "experimental_ccdf_exponent"


class DegreePreservingRewireError(RuntimeError):
    """Raised when the requested degree-preserving randomization cannot finish."""


@dataclass(frozen=True)
class GraphSpectralResult:
    model_variant: str
    graph_index: int
    graph_seed: int
    sequence_seed: int
    rewire_seed: int
    n_requested: int
    tau: float
    tau_convention: str
    target_mean_degree: float
    component_mode: str
    analyzed_dimension: int
    num_edges: int
    realized_mean_degree: float
    num_components: int
    largest_component_size: int
    num_isolated_vertices: int
    max_realized_degree: int
    node_labels: np.ndarray
    degrees: np.ndarray
    eigenvalues: np.ndarray
    bulk_eigenvalues: np.ndarray
    perron_eigenvalue: float
    bulk_eigenvectors: np.ndarray | None


@dataclass(frozen=True)
class PairedConfigurationResult:
    original: GraphSpectralResult
    configuration: GraphSpectralResult
    expected_weights: np.ndarray
    weight_diagnostics: WeightSequenceDiagnostics
    requested_swaps: int
    swaps_per_edge: int
    max_tries: int
    degree_match: bool
    edge_overlap_fraction: float


def generate_expected_weights_for_tau_convention(
    *,
    n: int,
    tau: float,
    target_mean_degree: float,
    w_min: float,
    w_max: float | None,
    cutoff_mode: str,
    cutoff_coefficient: float,
    sequence_mode: str,
    sequence_seed: int,
    tau_convention: str = TAU_CONVENTION_DENSITY,
) -> tuple[np.ndarray, WeightSequenceDiagnostics]:
    """Generate the exact sequence used by the selected Chung-Lu experiment.

    The checked-in baseline uses a density exponent ``tau`` and therefore a
    CCDF exponent ``tau - 1``. The separately run tau=0.8 experiment used tau
    directly as the CCDF exponent. Keeping that mode explicit prevents the two
    parameterizations from being silently mixed in manifests or plots.
    """
    if tau_convention == TAU_CONVENTION_DENSITY:
        effective_tau = tau
    elif tau_convention == TAU_CONVENTION_CCDF:
        if tau <= 0:
            raise ValueError("the experimental CCDF exponent must be positive")
        effective_tau = tau + 1.0
    else:
        raise ValueError(f"unsupported tau_convention={tau_convention!r}")

    weights, diagnostics = generate_power_law_expected_weights(
        n=n,
        tau=effective_tau,
        target_mean_degree=target_mean_degree,
        w_min=w_min,
        w_max=w_max,
        cutoff_mode=cutoff_mode,
        cutoff_coefficient=cutoff_coefficient,
        sequence_mode=sequence_mode,
        sequence_seed=sequence_seed,
    )
    if tau_convention == TAU_CONVENTION_CCDF:
        diagnostics = replace(diagnostics, tau=float(tau))
    return weights, diagnostics


def _canonical_edges(graph: nx.Graph) -> set[frozenset[int]]:
    return {frozenset((u, v)) for u, v in graph.edges()}


def _validate_simple_graph(graph: nx.Graph, label: str) -> None:
    if type(graph) is not nx.Graph:
        raise TypeError(f"{label} must be an ordinary networkx.Graph")
    if graph.is_multigraph():
        raise ValueError(f"{label} must not contain parallel edges")
    if nx.number_of_selfloops(graph):
        raise ValueError(f"{label} must not contain self-loops")


def rewire_graph_preserving_degrees(
    graph: nx.Graph,
    rewire_seed: int,
    swaps_per_edge: int = 10,
    max_tries_factor: int = 20,
    overlap_warning_threshold: float | None = 0.25,
) -> tuple[nx.Graph, int, int, float]:
    """Randomize a simple graph by a finite double-edge-swap Markov chain.

    This is an approximate randomized simple graph conditional on the exact
    realized labeled degree sequence. A finite number of swaps is not claimed
    to produce a provably uniform configuration-model sample.
    """
    _validate_simple_graph(graph, "input graph")
    if swaps_per_edge < 1 or max_tries_factor < 1:
        raise ValueError("swaps_per_edge and max_tries_factor must be positive")
    if graph.number_of_edges() < 2:
        raise DegreePreservingRewireError("graph has fewer than two edges and cannot be rewired")

    original_degrees = dict(graph.degree())
    original_edges = _canonical_edges(graph)
    rewired = graph.copy()
    requested_swaps = int(swaps_per_edge * graph.number_of_edges())
    max_tries = int(max(requested_swaps + 100, max_tries_factor * requested_swaps))
    try:
        nx.double_edge_swap(
            rewired,
            nswap=requested_swaps,
            max_tries=max_tries,
            seed=int(rewire_seed),
        )
    except (nx.NetworkXAlgorithmError, nx.NetworkXError) as exc:
        raise DegreePreservingRewireError(
            f"double-edge swaps failed after requesting {requested_swaps} swaps "
            f"with max_tries={max_tries}: {exc}"
        ) from exc

    _validate_simple_graph(rewired, "rewired graph")
    if set(rewired.nodes()) != set(graph.nodes()):
        raise AssertionError("rewiring changed the node labels")
    if rewired.number_of_nodes() != graph.number_of_nodes():
        raise AssertionError("rewiring changed the number of vertices")
    if rewired.number_of_edges() != graph.number_of_edges():
        raise AssertionError("rewiring changed the number of edges")
    if dict(rewired.degree()) != original_degrees:
        raise AssertionError("rewiring did not preserve every labeled vertex degree")

    overlap = len(original_edges & _canonical_edges(rewired)) / max(len(original_edges), 1)
    if overlap_warning_threshold is not None and overlap > overlap_warning_threshold:
        warnings.warn(
            f"edge overlap remains high after rewiring: {overlap:.3f}",
            RuntimeWarning,
            stacklevel=2,
        )
    return rewired, requested_swaps, max_tries, float(overlap)


def _select_component(graph: nx.Graph, component_mode: str) -> nx.Graph:
    if component_mode == "full":
        return graph
    if component_mode == "largest_connected_component":
        nodes = max(nx.connected_components(graph), key=len)
        return graph.subgraph(nodes).copy()
    raise ValueError(f"unsupported component_mode={component_mode!r}")


def analyze_graph_spectrum(
    graph: nx.Graph,
    *,
    model_variant: str,
    graph_index: int,
    graph_seed: int,
    sequence_seed: int,
    rewire_seed: int,
    n_requested: int,
    tau: float,
    tau_convention: str,
    target_mean_degree: float,
    component_mode: str,
    retain_eigenvectors: bool,
) -> GraphSpectralResult:
    _validate_simple_graph(graph, model_variant)
    node_labels = np.asarray(sorted(graph.nodes()), dtype=int)
    degree_map = dict(graph.degree())
    degrees = np.asarray([degree_map[int(node)] for node in node_labels], dtype=int)
    components = list(nx.connected_components(graph))
    analyzed_graph = _select_component(graph, component_mode)
    eigenvalues, eigenvectors = adjacency_eigendecomposition(analyzed_graph)
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    if len(eigenvalues) < 2:
        raise ValueError("need at least two eigenvalues to exclude the Perron eigenvalue")
    return GraphSpectralResult(
        model_variant=model_variant,
        graph_index=graph_index,
        graph_seed=int(graph_seed),
        sequence_seed=int(sequence_seed),
        rewire_seed=int(rewire_seed),
        n_requested=n_requested,
        tau=float(tau),
        tau_convention=tau_convention,
        target_mean_degree=float(target_mean_degree),
        component_mode=component_mode,
        analyzed_dimension=analyzed_graph.number_of_nodes(),
        num_edges=graph.number_of_edges(),
        realized_mean_degree=float(np.mean(degrees)),
        num_components=len(components),
        largest_component_size=max(map(len, components)),
        num_isolated_vertices=int(np.sum(degrees == 0)),
        max_realized_degree=int(np.max(degrees)),
        node_labels=node_labels,
        degrees=degrees,
        eigenvalues=eigenvalues,
        bulk_eigenvalues=eigenvalues[:-1],
        perron_eigenvalue=float(eigenvalues[-1]),
        bulk_eigenvectors=np.asarray(eigenvectors[:, :-1], dtype=float) if retain_eigenvectors else None,
    )


def generate_paired_configuration_realization(
    *,
    n: int,
    tau: float,
    target_mean_degree: float,
    w_min: float,
    w_max: float | None,
    cutoff_mode: str,
    cutoff_coefficient: float,
    sequence_mode: str,
    graph_seed: int,
    sequence_seed: int,
    rewire_seed: int,
    component_mode: str = "full",
    graph_index: int = 0,
    swaps_per_edge: int = 10,
    max_tries_factor: int = 20,
    tau_convention: str = TAU_CONVENTION_DENSITY,
    retain_eigenvectors: bool = True,
) -> PairedConfigurationResult:
    weights, diagnostics = generate_expected_weights_for_tau_convention(
        n=n,
        tau=tau,
        target_mean_degree=target_mean_degree,
        w_min=w_min,
        w_max=w_max,
        cutoff_mode=cutoff_mode,
        cutoff_coefficient=cutoff_coefficient,
        sequence_mode=sequence_mode,
        sequence_seed=sequence_seed,
        tau_convention=tau_convention,
    )
    if not diagnostics.admissibility_holds:
        raise ValueError("expected-weight sequence violates max(w)^2 <= sum(w)")
    original_graph = generate_chung_lu_graph_from_weights(weights, seed=graph_seed)
    configuration_graph, requested_swaps, max_tries, overlap = rewire_graph_preserving_degrees(
        original_graph,
        rewire_seed=rewire_seed,
        swaps_per_edge=swaps_per_edge,
        max_tries_factor=max_tries_factor,
    )
    common = dict(
        graph_index=graph_index,
        graph_seed=graph_seed,
        sequence_seed=sequence_seed,
        rewire_seed=rewire_seed,
        n_requested=n,
        tau=tau,
        tau_convention=tau_convention,
        target_mean_degree=target_mean_degree,
        component_mode=component_mode,
        retain_eigenvectors=retain_eigenvectors,
    )
    original = analyze_graph_spectrum(original_graph, model_variant=MODEL_CHUNG_LU, **common)
    configuration = analyze_graph_spectrum(
        configuration_graph,
        model_variant=MODEL_CONFIGURATION,
        **common,
    )
    degree_match = bool(
        np.array_equal(original.node_labels, configuration.node_labels)
        and np.array_equal(original.degrees, configuration.degrees)
    )
    if not degree_match:
        raise AssertionError("paired graphs do not have identical labeled degree sequences")
    return PairedConfigurationResult(
        original=original,
        configuration=configuration,
        expected_weights=weights,
        weight_diagnostics=diagnostics,
        requested_swaps=requested_swaps,
        swaps_per_edge=swaps_per_edge,
        max_tries=max_tries,
        degree_match=degree_match,
        edge_overlap_fraction=overlap,
    )


def generate_paired_configuration_ensemble_from_seeds(
    *,
    graph_seeds: Sequence[int],
    sequence_seeds: Sequence[int],
    rewire_seeds: Sequence[int],
    **kwargs: object,
) -> Iterator[PairedConfigurationResult]:
    if not (len(graph_seeds) == len(sequence_seeds) == len(rewire_seeds)):
        raise ValueError("graph_seeds, sequence_seeds, and rewire_seeds must have equal length")
    for graph_index, (graph_seed, sequence_seed, rewire_seed) in enumerate(
        zip(graph_seeds, sequence_seeds, rewire_seeds)
    ):
        yield generate_paired_configuration_realization(
            graph_index=graph_index,
            graph_seed=int(graph_seed),
            sequence_seed=int(sequence_seed),
            rewire_seed=int(rewire_seed),
            **kwargs,
        )
