from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class GraphSample:
    adjacency: sparse.csr_matrix
    degrees: np.ndarray
    metadata: dict


def _csr_from_edges(n: int, edges: Iterable[tuple[int, int]]) -> sparse.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    seen: set[tuple[int, int]] = set()
    for u, v in edges:
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        rows.extend((a, b))
        cols.extend((b, a))
        data.extend((1.0, 1.0))
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))


def random_regular_graph(n: int, degree: int, rng: np.random.Generator) -> GraphSample:
    if n * degree % 2 != 0:
        raise ValueError("n * degree must be even")
    if degree <= 0 or degree >= n:
        raise ValueError("degree must satisfy 0 < degree < n")

    stubs = np.repeat(np.arange(n), degree)
    for _ in range(200):
        rng.shuffle(stubs)
        edges: list[tuple[int, int]] = []
        ok = True
        used: set[tuple[int, int]] = set()
        for i in range(0, len(stubs), 2):
            u = int(stubs[i])
            v = int(stubs[i + 1])
            if u == v:
                ok = False
                break
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in used:
                ok = False
                break
            used.add((a, b))
            edges.append((a, b))
        if ok:
            adj = _csr_from_edges(n, edges)
            degrees = np.asarray(adj.sum(axis=1)).ravel().astype(int)
            if np.all(degrees == degree):
                return GraphSample(adj, degrees, {"model": "random_regular", "degree": degree})
    raise RuntimeError("failed to generate a simple random regular graph")


def watts_strogatz_graph(n: int, degree: int, rewire_prob: float, rng: np.random.Generator) -> GraphSample:
    if degree % 2 != 0:
        raise ValueError("degree must be even")
    if not 0.0 <= rewire_prob <= 1.0:
        raise ValueError("rewire_prob must be in [0, 1]")

    half = degree // 2
    neighbors: set[tuple[int, int]] = set()
    edge_set: set[tuple[int, int]] = set()

    for u in range(n):
        for offset in range(1, half + 1):
            v = (u + offset) % n
            a, b = (u, v) if u < v else (v, u)
            edge_set.add((a, b))

    for u in range(n):
        for offset in range(1, half + 1):
            v = (u + offset) % n
            a, b = (u, v) if u < v else (v, u)
            if rng.random() >= rewire_prob:
                continue
            edge_set.discard((a, b))
            candidates = np.arange(n)
            rng.shuffle(candidates)
            for w in candidates:
                if w == u:
                    continue
                x, y = (u, int(w)) if u < int(w) else (int(w), u)
                if (x, y) not in edge_set:
                    edge_set.add((x, y))
                    break

    adj = _csr_from_edges(n, edge_set)
    degrees = np.asarray(adj.sum(axis=1)).ravel().astype(int)
    return GraphSample(adj, degrees, {"model": "watts_strogatz", "degree": degree, "rewire_prob": rewire_prob})


def chung_lu_graph(expected_degrees: np.ndarray, rng: np.random.Generator) -> GraphSample:
    weights = np.asarray(expected_degrees, dtype=float)
    if np.any(weights < 0):
        raise ValueError("expected degrees must be nonnegative")
    n = len(weights)
    total = weights.sum()
    if total <= 0:
        raise ValueError("sum of expected degrees must be positive")

    edges: list[tuple[int, int]] = []
    for i in range(n):
        wi = weights[i]
        probs = wi * weights[i + 1 :] / total
        probs = np.clip(probs, 0.0, 1.0)
        draws = rng.random(len(probs)) < probs
        js = np.nonzero(draws)[0] + i + 1
        edges.extend((i, int(j)) for j in js)

    adj = _csr_from_edges(n, edges)
    degrees = np.asarray(adj.sum(axis=1)).ravel().astype(int)
    return GraphSample(adj, degrees, {"model": "chung_lu", "expected_degree_sum": float(total)})


def configuration_model(degree_sequence: np.ndarray, rng: np.random.Generator) -> GraphSample:
    degrees = np.asarray(degree_sequence, dtype=int)
    if np.any(degrees < 0):
        raise ValueError("degrees must be nonnegative")
    if degrees.sum() % 2 != 0:
        raise ValueError("sum of degrees must be even")

    stubs = np.repeat(np.arange(len(degrees)), degrees)
    for _ in range(200):
        rng.shuffle(stubs)
        edges: list[tuple[int, int]] = []
        ok = True
        used: set[tuple[int, int]] = set()
        for i in range(0, len(stubs), 2):
            u = int(stubs[i])
            v = int(stubs[i + 1])
            if u == v:
                ok = False
                break
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in used:
                ok = False
                break
            used.add((a, b))
            edges.append((a, b))
        if ok:
            adj = _csr_from_edges(len(degrees), edges)
            realized = np.asarray(adj.sum(axis=1)).ravel().astype(int)
            if np.array_equal(realized, degrees):
                return GraphSample(adj, realized, {"model": "configuration"})
    raise RuntimeError("failed to generate a simple configuration-model graph")


def preferential_attachment_graph(n: int, m: int, rng: np.random.Generator) -> GraphSample:
    if m <= 0 or m >= n:
        raise ValueError("m must satisfy 0 < m < n")

    edge_set: set[tuple[int, int]] = set()
    degree_counts = np.zeros(n, dtype=int)

    for u in range(m + 1):
        for v in range(u + 1, m + 1):
            edge_set.add((u, v))
            degree_counts[u] += 1
            degree_counts[v] += 1

    repeated = []
    for node, deg in enumerate(degree_counts[: m + 1]):
        repeated.extend([node] * deg)

    for new_node in range(m + 1, n):
        targets: set[int] = set()
        while len(targets) < m:
            target = int(rng.choice(repeated))
            if target != new_node:
                targets.add(target)
        for target in targets:
            a, b = (target, new_node) if target < new_node else (new_node, target)
            edge_set.add((a, b))
            degree_counts[a] += 1
            degree_counts[b] += 1
            repeated.extend((a, b))

    adj = _csr_from_edges(n, edge_set)
    degrees = np.asarray(adj.sum(axis=1)).ravel().astype(int)
    return GraphSample(adj, degrees, {"model": "preferential_attachment", "m": m})


def truncated_power_law_expected_degrees(
    n: int,
    exponent: float,
    min_degree: float,
    max_degree: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if exponent <= 1.0:
        raise ValueError("exponent should exceed 1 for a normalizable tail")
    u = rng.random(n)
    a = min_degree ** (1.0 - exponent)
    b = max_degree ** (1.0 - exponent)
    samples = (a + u * (b - a)) ** (1.0 / (1.0 - exponent))
    return np.clip(samples, min_degree, max_degree)
