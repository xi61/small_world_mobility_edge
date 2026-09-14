"""Validated preferential-attachment graph generators and diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import networkx as nx
import numpy as np


MODEL_BA = "Barabasi-Albert"
MODEL_BA_REWIRED = "Barabasi-Albert degree-preserving rewired"
MODEL_AFFINE_PA = "Affine linear preferential attachment"
MODEL_AFFINE_PA_REWIRED = "Affine linear preferential attachment degree-preserving rewired"
MODEL_NONLINEAR_PA = "Nonlinear preferential attachment"
MODEL_NONLINEAR_PA_REWIRED = "Nonlinear preferential attachment degree-preserving rewired"
MODEL_HOLME_KIM = "Holme-Kim"
MODEL_HOLME_KIM_REWIRED = "Holme-Kim degree-preserving rewired"

DEFAULT_N = 5000
DEFAULT_M = 10
TARGET_MEAN_DEGREE = 20.0
THEORETICAL_DEGREE_CCDF_TAU = 2.0
INITIAL_GRAPH_CONVENTION = "complete_graph_on_m_plus_1_nodes"
HOLME_KIM_INITIAL_GRAPH_CONVENTION = "empty_graph_on_m_nodes_networkx_internal"


@dataclass(frozen=True)
class PreferentialAttachmentDiagnostics:
    """Basic invariants recorded for one generated graph."""

    model: str
    n_requested: int
    m: int
    triad_probability: float
    attachment_offset: float
    theoretical_degree_ccdf_tau: float
    initial_graph_convention: str
    graph_seed: int
    num_nodes: int
    num_edges: int
    expected_num_edges: int | None
    realized_mean_degree: float
    max_realized_degree: int
    min_realized_degree: int
    num_components: int
    largest_component_size: int
    num_isolated_vertices: int
    is_simple_undirected: bool
    labels_are_zero_to_n_minus_one: bool
    attachment_zeta: float = 1.0

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["degree_ccdf_alpha"] = self.degree_ccdf_alpha
        return values

    @property
    def degree_ccdf_alpha(self) -> float:
        """Canonical affine degree-CCDF exponent name."""
        return float(self.theoretical_degree_ccdf_tau)


def canonical_model_name(model: str) -> str:
    """Normalize supported CLI model spellings to their display names."""
    normalized = model.strip().lower().replace("_", "-")
    if normalized in {"ba", "barabasi-albert", "barabasi albert"}:
        return MODEL_BA
    if normalized in {"affine", "affine-pa", "affine-linear", "linear-pa"}:
        return MODEL_AFFINE_PA
    if normalized in {
        "nonlinear",
        "nonlinear-pa",
        "sublinear",
        "superlinear",
        "power-pa",
        "uniform-attachment",
    }:
        return MODEL_NONLINEAR_PA
    if normalized in {"holme-kim", "holme kim", "powerlaw-cluster"}:
        return MODEL_HOLME_KIM
    if model in {MODEL_BA, MODEL_AFFINE_PA, MODEL_NONLINEAR_PA, MODEL_HOLME_KIM}:
        return model
    raise ValueError(f"unsupported preferential-attachment model {model!r}")


def initial_graph_for_ba(m: int) -> nx.Graph:
    """Return the explicit BA seed graph used by this experiment."""
    if m < 1:
        raise ValueError("m must be positive")
    return nx.complete_graph(m + 1)


def expected_ba_edge_count(n: int, m: int) -> int:
    """Exact edge count for the explicit K_(m+1) initial graph convention."""
    if not 1 <= m < n:
        raise ValueError("BA requires 1 <= m < n")
    initial_edges = m * (m + 1) // 2
    added_vertices = n - (m + 1)
    return int(initial_edges + m * added_vertices)


def attachment_offset_for_degree_ccdf_alpha(m: int, degree_ccdf_alpha: float) -> float:
    """Return delta in P(new -> i) proportional to degree(i) + delta.

    For the affine linear model, the asymptotic degree CCDF exponent is
    alpha = 2 + delta / m. Strict positivity of every newborn vertex's
    attractiveness requires delta > -m, equivalently alpha > 1.
    """
    if m < 1:
        raise ValueError("m must be positive")
    if not math.isfinite(degree_ccdf_alpha) or degree_ccdf_alpha <= 1.0:
        raise ValueError("affine preferential attachment requires degree_ccdf_alpha > 1")
    return float(m * (degree_ccdf_alpha - 2.0))


def attachment_offset_for_ccdf_tau(m: int, tau: float) -> float:
    """Compatibility alias for :func:`attachment_offset_for_degree_ccdf_alpha`."""
    return attachment_offset_for_degree_ccdf_alpha(m, tau)


def generate_affine_preferential_attachment_graph(
    *,
    n: int,
    m: int,
    seed: int,
    degree_ccdf_alpha: float | None = None,
    degree_ccdf_tau: float | None = None,
) -> tuple[nx.Graph, float]:
    """Generate simple affine PA using weighted sampling without replacement."""
    if not 1 <= m < n:
        raise ValueError("affine preferential attachment requires 1 <= m < n")
    if degree_ccdf_alpha is None:
        degree_ccdf_alpha = degree_ccdf_tau
    elif degree_ccdf_tau is not None and not math.isclose(
        degree_ccdf_alpha, degree_ccdf_tau, abs_tol=1e-12, rel_tol=0.0
    ):
        raise ValueError("canonical and legacy affine exponent values disagree")
    if degree_ccdf_alpha is None:
        raise ValueError("degree_ccdf_alpha is required")
    offset = attachment_offset_for_degree_ccdf_alpha(m, degree_ccdf_alpha)
    graph = initial_graph_for_ba(m)
    degree_array = np.zeros(n, dtype=np.int64)
    degree_array[: m + 1] = m
    rng = np.random.default_rng(int(seed))
    for source in range(m + 1, n):
        existing = np.arange(source, dtype=int)
        degrees = degree_array[:source].astype(float, copy=False)
        attractiveness = degrees + offset
        if np.any(attractiveness <= 0.0):
            raise AssertionError("affine attachment kernel has nonpositive attractiveness")
        probabilities = attractiveness / np.sum(attractiveness)
        targets = rng.choice(existing, size=m, replace=False, p=probabilities)
        graph.add_node(source)
        graph.add_edges_from((source, int(target)) for target in targets)
        degree_array[source] = m
        degree_array[targets] += 1
    return graph, offset


def generate_nonlinear_preferential_attachment_graph(
    *,
    n: int,
    m: int,
    seed: int,
    attachment_zeta: float,
) -> nx.Graph:
    """Generate simple PA with attachment weights ``degree**attachment_zeta``.

    Targets are sampled without replacement.  ``zeta=0`` is uniform attachment
    over the existing positive-degree vertices.  The special case ``zeta=1``
    intentionally uses the same arithmetic and RNG calls as affine PA with
    zero offset, so identical seeds produce identical graphs.
    """
    if not 1 <= m < n:
        raise ValueError("nonlinear preferential attachment requires 1 <= m < n")
    if not math.isfinite(attachment_zeta) or attachment_zeta < 0.0:
        raise ValueError("attachment_zeta must be finite and nonnegative")
    graph = initial_graph_for_ba(m)
    degree_array = np.zeros(n, dtype=np.int64)
    degree_array[: m + 1] = m
    rng = np.random.default_rng(int(seed))
    for source in range(m + 1, n):
        existing = np.arange(source, dtype=int)
        degrees = degree_array[:source].astype(float, copy=False)
        if math.isclose(attachment_zeta, 0.0, abs_tol=0.0, rel_tol=0.0):
            weights = np.ones_like(degrees)
        elif math.isclose(attachment_zeta, 1.0, abs_tol=0.0, rel_tol=0.0):
            weights = degrees
        else:
            log_weights = attachment_zeta * np.log(degrees)
            weights = np.exp(log_weights - np.max(log_weights))
        probabilities = weights / np.sum(weights)
        targets = rng.choice(existing, size=m, replace=False, p=probabilities)
        graph.add_node(source)
        graph.add_edges_from((source, int(target)) for target in targets)
        degree_array[source] = m
        degree_array[targets] += 1
    _validate_graph(graph, n)
    return graph


def _validate_graph(graph: nx.Graph, n: int) -> None:
    if type(graph) is not nx.Graph or graph.is_directed() or graph.is_multigraph():
        raise TypeError("preferential-attachment generator must return a simple undirected nx.Graph")
    if nx.number_of_selfloops(graph):
        raise ValueError("preferential-attachment graph contains a self-loop")
    if graph.number_of_nodes() != n:
        raise AssertionError("preferential-attachment generator returned the wrong number of nodes")
    if set(graph.nodes()) != set(range(n)):
        raise AssertionError("preferential-attachment node labels must be 0, ..., n-1")


def generate_preferential_attachment_graph(
    *,
    n: int = DEFAULT_N,
    m: int = DEFAULT_M,
    seed: int,
    model: str = MODEL_BA,
    triad_probability: float = 0.0,
    attachment_ccdf_tau: float = THEORETICAL_DEGREE_CCDF_TAU,
    degree_ccdf_alpha: float | None = None,
    attachment_zeta: float = 1.0,
) -> tuple[nx.Graph, PreferentialAttachmentDiagnostics]:
    """Generate a BA or optional Holme-Kim graph and record its invariants."""
    model_name = canonical_model_name(model)
    if not 1 <= m < n:
        raise ValueError("preferential attachment requires 1 <= m < n")
    if not 0.0 <= triad_probability <= 1.0:
        raise ValueError("triad_probability must lie in [0, 1]")

    expected_edges: int | None
    attachment_offset = 0.0
    theoretical_tau = THEORETICAL_DEGREE_CCDF_TAU
    if model_name == MODEL_BA:
        graph = nx.barabasi_albert_graph(
            n=n,
            m=m,
            seed=int(seed),
            initial_graph=initial_graph_for_ba(m),
        )
        convention = INITIAL_GRAPH_CONVENTION
        expected_edges = expected_ba_edge_count(n, m)
    elif model_name == MODEL_AFFINE_PA:
        canonical_alpha = (
            float(attachment_ccdf_tau)
            if degree_ccdf_alpha is None
            else float(degree_ccdf_alpha)
        )
        graph, attachment_offset = generate_affine_preferential_attachment_graph(
            n=n,
            m=m,
            seed=seed,
            degree_ccdf_alpha=canonical_alpha,
        )
        theoretical_tau = canonical_alpha
        convention = INITIAL_GRAPH_CONVENTION
        expected_edges = expected_ba_edge_count(n, m)
    elif model_name == MODEL_NONLINEAR_PA:
        graph = generate_nonlinear_preferential_attachment_graph(
            n=n,
            m=m,
            seed=seed,
            attachment_zeta=attachment_zeta,
        )
        theoretical_tau = math.nan
        convention = INITIAL_GRAPH_CONVENTION
        expected_edges = expected_ba_edge_count(n, m)
    else:
        graph = nx.powerlaw_cluster_graph(
            n=n,
            m=m,
            p=float(triad_probability),
            seed=int(seed),
        )
        convention = HOLME_KIM_INITIAL_GRAPH_CONVENTION
        expected_edges = int(m * (n - m))

    graph = nx.convert_node_labels_to_integers(graph, ordering="sorted")
    _validate_graph(graph, n)
    degrees = np.asarray([graph.degree(node) for node in range(n)], dtype=int)
    components = list(nx.connected_components(graph))
    diagnostics = PreferentialAttachmentDiagnostics(
        model=model_name,
        n_requested=int(n),
        m=int(m),
        triad_probability=float(triad_probability),
        attachment_offset=float(attachment_offset),
        theoretical_degree_ccdf_tau=float(theoretical_tau),
        initial_graph_convention=convention,
        graph_seed=int(seed),
        num_nodes=graph.number_of_nodes(),
        num_edges=graph.number_of_edges(),
        expected_num_edges=expected_edges,
        realized_mean_degree=float(np.mean(degrees)),
        max_realized_degree=int(np.max(degrees)),
        min_realized_degree=int(np.min(degrees)),
        num_components=len(components),
        largest_component_size=max(map(len, components)),
        num_isolated_vertices=int(np.sum(degrees == 0)),
        is_simple_undirected=(
            type(graph) is nx.Graph
            and not graph.is_directed()
            and not graph.is_multigraph()
            and nx.number_of_selfloops(graph) == 0
        ),
        labels_are_zero_to_n_minus_one=set(graph.nodes()) == set(range(n)),
        attachment_zeta=float(attachment_zeta),
    )
    if expected_edges is not None and graph.number_of_edges() != expected_edges:
        raise AssertionError(
            f"BA edge count {graph.number_of_edges()} does not match expected {expected_edges}"
        )
    if not math.isclose(
        diagnostics.realized_mean_degree,
        2.0 * graph.number_of_edges() / graph.number_of_nodes(),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise AssertionError("mean degree is inconsistent with the graph edge count")
    return graph, diagnostics
