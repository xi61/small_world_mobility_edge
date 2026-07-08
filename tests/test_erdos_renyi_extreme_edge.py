from __future__ import annotations

from pathlib import Path
import sys
import unittest

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from small_world_mobility_edge.erdos_renyi_baseline import flat_vector_overlap_index, sparse_normalized_edge_levels
from small_world_mobility_edge.erdos_renyi_extreme_edge import (
    build_edge_levels_ensemble,
    build_edge_unfolding_model,
    compute_edge_summary,
    edge_values_from_levels,
    theorem_range_diagnostic,
)


class ErdosRenyiExtremeEdgeTests(unittest.TestCase):
    def test_flat_vector_overlap_index(self) -> None:
        vectors = np.array(
            [
                [0.5, 1.0 / np.sqrt(2)],
                [0.5, -1.0 / np.sqrt(2)],
                [0.5, 0.0],
                [0.5, 0.0],
            ],
            dtype=float,
        )
        self.assertEqual(flat_vector_overlap_index(vectors), 0)

    def test_sparse_normalized_edge_levels_are_sorted(self) -> None:
        graph = nx.path_graph(40)
        levels = sparse_normalized_edge_levels(
            graph,
            average_degree=3.0,
            edge_levels=6,
            graph_index=0,
            seed=17,
            component_mode="full",
        )
        self.assertEqual(len(levels.positive_eigenvalues), 6)
        self.assertEqual(len(levels.negative_eigenvalues), 6)
        self.assertTrue(np.all(np.diff(levels.positive_eigenvalues) >= 0))
        self.assertTrue(np.all(np.diff(levels.negative_eigenvalues) >= 0))

    def test_edge_summary_uses_within_graph_unfolded_gaps_only(self) -> None:
        density_levels = build_edge_levels_ensemble(
            n=200,
            average_degree=6.0,
            graph_seeds=[101, 103, 107, 109],
            component_mode="largest_connected_component",
        )
        spacing_levels = build_edge_levels_ensemble(
            n=200,
            average_degree=6.0,
            graph_seeds=[113, 127, 131, 137],
            component_mode="largest_connected_component",
        )
        result = compute_edge_summary(
            spacing_levels=spacing_levels,
            density_levels=density_levels,
            side="positive_edge",
            perron_mode="largest_eigenvalue",
            component_mode="largest_connected_component",
        )
        expected_gaps = sum(len(edge_values_from_levels(levels, "positive_edge", "largest_eigenvalue")) - 1 for levels in spacing_levels)
        self.assertEqual(int(result["number_of_gaps"]), expected_gaps)
        self.assertTrue(np.all(np.isfinite(np.asarray(result["spacings"], dtype=float))))

    def test_flat_vector_overlap_perron_exclusion_changes_positive_edge_sample(self) -> None:
        levels = build_edge_levels_ensemble(
            n=200,
            average_degree=6.0,
            graph_seeds=[201],
            component_mode="largest_connected_component",
        )[0]
        largest_mode_values = edge_values_from_levels(levels, "positive_edge", "largest_eigenvalue")
        overlap_mode_values = edge_values_from_levels(levels, "positive_edge", "flat_vector_overlap")
        self.assertEqual(len(largest_mode_values), len(overlap_mode_values))
        self.assertTrue(np.all(np.diff(largest_mode_values) >= 0))
        self.assertTrue(np.all(np.diff(overlap_mode_values) >= 0))

    def test_edge_unfolding_model_has_positive_total_count(self) -> None:
        density_levels = build_edge_levels_ensemble(
            n=200,
            average_degree=6.0,
            graph_seeds=[301, 307, 311, 313],
            component_mode="largest_connected_component",
        )
        density, total_count = build_edge_unfolding_model(
            density_levels=density_levels,
            side="negative_edge",
            perron_mode="largest_eigenvalue",
        )
        self.assertGreater(total_count, 0.0)
        self.assertTrue(np.all(np.isfinite(density.density)))

    def test_theorem_range_diagnostic_fields(self) -> None:
        diagnostic = theorem_range_diagnostic(n=5000, d=10.0)
        self.assertIn("logN", diagnostic)
        self.assertIn("loglogN_to_4", diagnostic)
        self.assertIn("d_over_logN", diagnostic)
        self.assertIn("heuristic_theorem_range", diagnostic)


if __name__ == "__main__":
    unittest.main()
