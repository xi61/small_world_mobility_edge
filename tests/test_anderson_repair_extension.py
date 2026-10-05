from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import linalg, sparse

from small_world_mobility_edge.random_schrodinger.anderson_reporting import _graph_points, _rrg_a9_decision
from small_world_mobility_edge.random_schrodinger.anderson_rrg_benchmark import _central_interval_solve, _generate_connected_rrg
from small_world_mobility_edge.random_schrodinger.anderson_utils import (
    atomic_savez,
    selection_union,
    validate_selected_eigenpairs,
    width_count_quantile_indices,
    window_registry,
)
from small_world_mobility_edge.random_schrodinger.common import stable_json_hash
from small_world_mobility_edge.random_schrodinger.handoff_metrics import gap_triplet_rows, selected_vector_rows, stable_zero_cluster


class AndersonExecutionTests(unittest.TestCase):
    def test_width_registry_and_endpoints(self) -> None:
        values = np.linspace(-3.0, 3.0, 5000)
        windows = window_registry(values, pa_physical_half_width=0.1)
        self.assertEqual(len(windows["q25_width_0p02"]), 101)
        self.assertEqual(len(windows["q50_width_0p02"]), 101)
        self.assertEqual(len(windows["q75_width_0p02"]), 101)
        self.assertEqual(len(windows["q25_width_51"]), 51)
        self.assertTrue(np.all(np.diff(windows["q50_width_0p02"]) == 1))

    def test_guard_union_and_atom_exclusion(self) -> None:
        windows = {"q": np.arange(4, 7, dtype=int)}
        atom = np.zeros(12, dtype=bool); atom[5] = True
        union, expanded = selection_union(windows, n=12, guard=1, atom_mask=atom)
        np.testing.assert_array_equal(expanded["q"], np.arange(3, 8))
        np.testing.assert_array_equal(union, np.asarray([3, 4, 6, 7]))

    def test_atom_triplets_never_bridge(self) -> None:
        values = np.asarray([-2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0])
        zero = stable_zero_cluster(values, enabled=True)
        self.assertEqual(zero["dimension"], 3)
        rows = gap_triplet_rows(values, range(1, 6), atom_mask=zero["mask"], prefix={})
        excluded = [row for row in rows if row["atom_triplet_excluded"]]
        self.assertEqual(len(excluded), 5)
        self.assertEqual([(row["left_ascending_index"], row["middle_ascending_index"], row["right_ascending_index"]) for row in rows], [(0,1,2),(1,2,3),(2,3,4),(3,4,5),(4,5,6)])

    def test_energy_decomposition(self) -> None:
        adjacency = sparse.csr_matrix(np.asarray([[0.0, 1.0], [1.0, 0.0]]))
        xi = np.asarray([-0.4, 0.7])
        amplitude = 2.3
        operator = adjacency.toarray() + amplitude * np.diag(xi)
        values, vectors = np.linalg.eigh(operator)
        rows = selected_vector_rows(values, vectors, [0, 1], prefix={}, windows={"all": np.asarray([0, 1])}, potential=xi, disorder_amplitude=amplitude, adjacency=adjacency)
        self.assertLess(max(float(row["energy_identity_absolute_error"]) for row in rows), 1e-12)

    def test_operator_pairing_and_cache_invalidation(self) -> None:
        rng = np.random.default_rng(1234)
        xi = rng.uniform(-1.0, 1.0, 8)
        adjacency = nx.to_numpy_array(nx.path_graph(8))
        h1 = adjacency + 2.0 * np.diag(xi)
        h2 = adjacency + 4.0 * np.diag(xi)
        np.testing.assert_allclose(np.diag(h2 - adjacency), 2.0 * np.diag(h1 - adjacency))
        first = stable_json_hash({"source": "a", "graph": "g", "a": 2.0})
        second = stable_json_hash({"source": "b", "graph": "g", "a": 2.0})
        self.assertNotEqual(first, second)

    def test_a_zero_graph_level_nesting(self) -> None:
        rows = [
            {"model": "m", "n": 10, "a": 0.0, "graph_replicate": 1, "graph_seed": 4, "metric": 1.0},
            {"model": "m", "n": 10, "a": 0.0, "graph_replicate": 2, "graph_seed": 5, "metric": 3.0},
            {"model": "m", "n": 10, "a": 0.0, "graph_replicate": 3, "graph_seed": 6, "metric": 5.0},
        ]
        points, summary = _graph_points(rows, dimensions=("model", "n", "a", "graph_replicate", "graph_seed"), metrics=("metric",))
        self.assertEqual(len(points), 3)
        self.assertEqual(summary[0]["graph_count"], 3)
        self.assertEqual(summary[0]["metric_graph_mean"], 3.0)

    def test_rrg_connected_simple_degree_three(self) -> None:
        graph, attempts = _generate_connected_rrg(100, 3, 9123)
        self.assertTrue(nx.is_connected(graph))
        self.assertEqual(list(graph.nodes()), list(range(100)))
        self.assertEqual(set(dict(graph.degree()).values()), {3})
        self.assertEqual(nx.number_of_selfloops(graph), 0)
        self.assertGreaterEqual(len(attempts), 1)

    def test_rrg_interval_completeness_against_full(self) -> None:
        graph, _ = _generate_connected_rrg(120, 3, 5512)
        dense = nx.to_numpy_array(graph, nodelist=range(120), dtype=float)
        xi = np.random.default_rng(887).uniform(-1.0, 1.0, 120)
        dense[np.diag_indices_from(dense)] += 9.0 * xi
        values, vectors, coverage = _central_interval_solve(dense, target_half_width=0.1, initial_half_width=0.4)
        full = linalg.eigh(dense, eigvals_only=True, check_finite=False, driver="evd")
        target = values[np.abs(values) <= 0.1]
        reference = full[np.abs(full) <= 0.1]
        np.testing.assert_allclose(target, reference, rtol=0.0, atol=1e-10)
        validation = validate_selected_eigenpairs(sparse.csr_matrix(dense), values, vectors)
        self.assertLess(validation["maximum_relative_residual"], 1e-10)
        self.assertEqual(coverage["target_count"], len(reference))

    def test_atomic_npz_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            atomic_savez(path, values=np.arange(7))
            with np.load(path, allow_pickle=False) as data:
                np.testing.assert_array_equal(data["values"], np.arange(7))

    def test_conditional_a9_rule(self) -> None:
        config = {
            "statistics": {"poisson_mean_raw_gap_ratio": 0.3862943611, "goe_reference": 0.53},
            "rrg": {
                "n_values": [1250, 2500, 5000],
                "conditional_a9_rule": "frozen",
                "conditional_a9_max_new_instances": 18,
            },
        }
        rows = []
        for n, p8, p10 in ((1250, .02, .015), (2500, .02, .012), (5000, .02, .009)):
            rows.extend([
                {"n": n, "a": 8.0, "mean_raw_gap_ratio_graph_mean": .49, "median_participation_fraction_graph_mean": p8},
                {"n": n, "a": 10.0, "mean_raw_gap_ratio_graph_mean": .42, "median_participation_fraction_graph_mean": p10},
            ])
        decision = _rrg_a9_decision(rows, config)
        self.assertTrue(decision["trigger_a9"])


if __name__ == "__main__":
    unittest.main()
