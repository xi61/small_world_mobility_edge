from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from small_world_mobility_edge.regular_baseline import (
    generate_random_regular_graph,
    adjacency_eigendecomposition,
    local_window_spacings,
)


class LocalWindowSpacingTests(unittest.TestCase):
    def test_window_selection_and_gap_count(self) -> None:
        eigenvalues = np.array([-0.30, -0.10, 0.00, 0.20, 0.40, 3.00], dtype=float)
        result = local_window_spacings(
            eigenvalues=eigenvalues,
            E=0.0,
            window_half_width=0.21,
            degree=3,
            min_eigenvalues=3,
        )

        np.testing.assert_allclose(result.selected_eigenvalues, np.array([-0.10, 0.00, 0.20]))
        np.testing.assert_allclose(result.raw_gaps, np.array([0.10, 0.20]))
        self.assertEqual(result.num_gaps, result.num_eigenvalues - 1)
        self.assertFalse(np.any(np.isclose(result.selected_eigenvalues, 3.0, atol=1e-10, rtol=0.0)))

    def test_fixed_and_midpoint_methods_are_close_in_narrow_bulk_window(self) -> None:
        graph = generate_random_regular_graph(n=1200, degree=3, seed=7)
        eigenvalues, _ = adjacency_eigendecomposition(graph)

        fixed = local_window_spacings(
            eigenvalues=eigenvalues,
            E=0.0,
            window_half_width=0.15,
            degree=3,
            reweight_within_window=False,
            min_eigenvalues=6,
        )
        midpoint = local_window_spacings(
            eigenvalues=eigenvalues,
            E=0.0,
            window_half_width=0.15,
            degree=3,
            reweight_within_window=True,
            min_eigenvalues=6,
        )

        mean_difference = float(np.mean(np.abs(fixed.spacings - midpoint.spacings)))
        self.assertLess(mean_difference / max(fixed.mean_spacing, 1e-12), 0.10)

    def test_bulk_mean_spacing_is_finite_and_near_one(self) -> None:
        graph = generate_random_regular_graph(n=1600, degree=3, seed=11)
        eigenvalues, _ = adjacency_eigendecomposition(graph)

        result = local_window_spacings(
            eigenvalues=eigenvalues,
            E=0.0,
            window_half_width=0.20,
            degree=3,
            min_eigenvalues=8,
        )

        self.assertTrue(np.all(np.isfinite(result.spacings)))
        self.assertLess(abs(result.mean_spacing - 1.0), 0.35)


if __name__ == "__main__":
    unittest.main()
