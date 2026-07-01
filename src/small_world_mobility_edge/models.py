from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class GraphSample:
    adjacency: sparse.csr_matrix
    degrees: np.ndarray
    metadata: dict


def random_regular_graph(n: int, degree: int, seed: int | np.random.Generator | None = None) -> GraphSample:
    """Return a random simple d-regular graph sample.

    The graph is generated with NetworkX and converted to the internal sample
    format used by the rest of the project.
    """
    graph = nx.random_regular_graph(degree, n, seed=seed)
    adjacency = nx.to_scipy_sparse_array(graph, dtype=float, format="csr")
    degrees = np.asarray(adjacency.sum(axis=1)).ravel().astype(int)
    return GraphSample(adjacency, degrees, {"model": "random_regular", "degree": degree})
