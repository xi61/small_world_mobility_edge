from __future__ import annotations

import unittest

import numpy as np

from small_world_mobility_edge.models import (
    chung_lu_graph,
    configuration_model,
    preferential_attachment_graph,
    random_regular_graph,
    watts_strogatz_graph,
)


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(123)

    def test_random_regular_degree_sequence(self) -> None:
        sample = random_regular_graph(n=20, degree=4, rng=self.rng)
        self.assertTrue(np.all(sample.degrees == 4))

    def test_watts_strogatz_is_symmetric(self) -> None:
        sample = watts_strogatz_graph(n=30, degree=4, rewire_prob=0.2, rng=self.rng)
        diff = (sample.adjacency - sample.adjacency.T).nnz
        self.assertEqual(diff, 0)

    def test_chung_lu_is_symmetric(self) -> None:
        sample = chung_lu_graph(np.full(25, 3.0), rng=self.rng)
        diff = (sample.adjacency - sample.adjacency.T).nnz
        self.assertEqual(diff, 0)

    def test_configuration_model_matches_degree_sequence(self) -> None:
        sample = configuration_model(np.array([2, 2, 2, 2, 2, 2]), rng=self.rng)
        self.assertTrue(np.array_equal(sample.degrees, np.array([2, 2, 2, 2, 2, 2])))

    def test_preferential_attachment_has_expected_edge_count(self) -> None:
        n = 25
        m = 3
        sample = preferential_attachment_graph(n=n, m=m, rng=self.rng)
        expected_edges = (m + 1) * m // 2 + (n - (m + 1)) * m
        self.assertEqual(sample.adjacency.nnz // 2, expected_edges)


if __name__ == "__main__":
    unittest.main()
