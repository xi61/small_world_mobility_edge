import math
import unittest

import networkx as nx
import numpy as np

from small_world_mobility_edge.random_schrodinger.common import (
    BFSDistanceCache,
    deterministic_mass_order,
    pair_distance_rows,
    selected_mass_sets,
)
from small_world_mobility_edge.random_schrodinger.geometry import _select_centers


class RandomSchrodingerGeometryTests(unittest.TestCase):
    def test_deterministic_mass_ties_use_vertex_id(self):
        vector = np.array([0.5, -0.5, 0.1, -0.1], dtype=float)
        order, masses = deterministic_mass_order(vector)
        self.assertEqual(order.tolist(), [0, 1, 2, 3])
        self.assertTrue(np.allclose(masses, [0.25, 0.25, 0.01, 0.01]))

    def test_mass_sets_are_minimal_threshold_sets(self):
        vector = np.sqrt(np.array([0.4, 0.3, 0.2, 0.1]))
        sets = selected_mass_sets(vector, counts=(2,), fractions=(0.5, 0.8))
        self.assertEqual(sets["top_2"].tolist(), [0, 1])
        self.assertEqual(sets["mass_50"].tolist(), [0, 1])
        self.assertEqual(sets["mass_80"].tolist(), [0, 1, 2])

    def test_full_graph_distance_and_unreachable_flag(self):
        graph = nx.Graph([(0, 3), (3, 1)])
        graph.add_node(2)
        cache = BFSDistanceCache(graph)
        matrix = cache.matrix([0, 1, 2])
        rows, summary = pair_distance_rows([0, 1, 2], [0.5, 0.3, 0.2], matrix)
        self.assertEqual(matrix[0, 1], 2.0)
        self.assertTrue(math.isinf(matrix[0, 2]))
        self.assertEqual(sum(bool(row["unreachable"]) for row in rows), 2)
        self.assertAlmostEqual(summary["unreachable_fraction"], 2.0 / 3.0)
        self.assertEqual(summary["finite_mass_weighted_mean_distance"], 2.0)
        self.assertAlmostEqual(summary["finite_mass_weighted_distance_denominator"], 0.15)

    def test_star_mass_selects_hub_not_each_leaf_as_center(self):
        graph = nx.star_graph(8)
        vector = np.zeros(9, dtype=float)
        vector[0] = math.sqrt(0.55)
        vector[1:] = math.sqrt(0.45 / 8.0)
        selected = _select_centers(
            graph,
            vector,
            BFSDistanceCache(graph),
            radius=1,
            separation=2,
            target_mass=0.8,
        )
        self.assertEqual(selected["centers"], [0])
        self.assertEqual(selected["geometry_label"], "one_center")
        self.assertGreater(selected["covered_mass"], 0.999999)


if __name__ == "__main__":
    unittest.main()
