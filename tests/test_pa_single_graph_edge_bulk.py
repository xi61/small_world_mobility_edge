from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import networkx as nx
import numpy as np

from small_world_mobility_edge.preferential_attachment.single_graph_edge_bulk import (
    SingleGraphConfig,
    adjacent_gap_ratio_arrays,
    distance_table_for_sets,
    histogram_window,
    nested_bulk_windows,
    nested_edge_windows,
    rows_for_vertex_set,
    shortest_path_statistics,
    sorted_dense_eigendecomposition,
    top_mass_vertices,
    valid_atom_free_segments,
)


class SingleGraphEdgeBulkTests(unittest.TestCase):
    def test_production_requires_held_out_cdf(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                SingleGraphConfig(
                    n=100,
                    m=4,
                    model_family="affine_linear",
                    parameter_value=2.0,
                    graph_seed=11,
                    output_root=Path(folder),
                )

    def test_sorted_eigenvectors_stay_paired(self) -> None:
        graph = nx.path_graph(9)
        values, vectors, order = sorted_dense_eigendecomposition(graph)
        adjacency = nx.to_numpy_array(graph, nodelist=list(range(9)))
        self.assertTrue(np.all(np.diff(values) >= -1e-12))
        self.assertEqual(order.shape, (9,))
        for index in range(9):
            residual = adjacency @ vectors[:, index] - values[index] * vectors[:, index]
            self.assertLess(np.linalg.norm(residual), 1e-10)

    def test_zero_atom_splits_gap_ratios(self) -> None:
        values = np.asarray([-4.0, -2.0, -1.0, 0.0, 0.0, 0.5, 1.5, 3.5])
        valid, segments = valid_atom_free_segments(values)
        left, right, ratio = adjacent_gap_ratio_arrays(values, segments)
        self.assertEqual([part.tolist() for part in segments], [[0, 1, 2], [5, 6, 7]])
        self.assertFalse(valid[3])
        self.assertFalse(valid[4])
        self.assertAlmostEqual(ratio[1], 0.5)
        self.assertAlmostEqual(ratio[6], 0.5)
        self.assertTrue(math.isnan(right[2]))
        self.assertTrue(math.isnan(left[5]))
        self.assertTrue(math.isnan(ratio[7]))
    def test_histogram_windows_preserve_requested_centers(self) -> None:
        for center in (0.005, 0.01, 0.025, 0.5, 0.99, 0.999):
            low, high = histogram_window(center, 0.05)
            self.assertAlmostEqual((low + high) / 2.0, center)
            self.assertGreaterEqual(low, 0.0)
            self.assertLessEqual(high, 1.0)
        self.assertEqual(histogram_window(0.005, 0.05), (0.0, 0.01))
        self.assertEqual(histogram_window(0.999, 0.05), (0.998, 1.0))


    def test_edge_and_bulk_windows_are_nested(self) -> None:
        values = np.concatenate(
            [np.linspace(-12.0, -0.1, 70), np.asarray([0.0, 0.0]), np.linspace(0.1, 12.0, 70)]
        )
        _, segments = valid_atom_free_segments(values)
        edge = nested_edge_windows(values)
        center, bulk = nested_bulk_windows(values, segments)
        self.assertEqual(edge[10][-1], len(values) - 1)
        self.assertTrue(set(edge[10]).issubset(set(edge[25])))
        self.assertTrue(set(edge[25]).issubset(set(edge[50])))
        self.assertTrue(set(bulk[10]).issubset(set(bulk[25])))
        self.assertTrue(set(bulk[25]).issubset(set(bulk[50])))
        containing = next(part for part in segments if center in set(part))
        self.assertGreater(bulk[50][0], containing[0])
        self.assertLess(bulk[50][-1], containing[-1])

    def test_top_mass_coordinates_are_vertex_ids(self) -> None:
        vector = np.asarray([0.1, -0.9, 0.3, -0.7, 0.2])
        vertices, masses = top_mass_vertices(vector, count=3)
        self.assertEqual(vertices.tolist(), [1, 3, 2])
        np.testing.assert_allclose(masses, [0.81, 0.49, 0.09])

    def test_full_graph_distances_are_not_induced_distances(self) -> None:
        graph = nx.path_graph(6)
        adjacency = nx.to_scipy_sparse_array(graph, nodelist=list(range(6)), format="csr")
        vertices = np.asarray([0, 2, 5])
        distances, source_rows = distance_table_for_sets(adjacency, [vertices])
        rows = rows_for_vertex_set(distances, source_rows, vertices)
        stats = shortest_path_statistics(rows, vertices)
        self.assertAlmostEqual(stats["mean_pairwise_distance"], (2 + 5 + 3) / 3)
        self.assertEqual(stats["connected_pair_count"], 3)
        self.assertEqual(graph.subgraph(vertices).number_of_edges(), 0)


if __name__ == "__main__":
    unittest.main()
