from __future__ import annotations

import unittest

import numpy as np

from small_world_mobility_edge import chung_lu_eigenvalue_resonance_scan as scan


class ChungLuEigenvalueResonanceScanTests(unittest.TestCase):
    def test_bins_cover_resonance_window(self) -> None:
        self.assertEqual(len(scan.LOCAL_QUANTILE_BINS), 60)
        self.assertAlmostEqual(scan.LOCAL_QUANTILE_BINS[0][0], 0.60)
        self.assertAlmostEqual(scan.LOCAL_QUANTILE_BINS[-1][1], 0.72)
        for left, right in scan.LOCAL_QUANTILE_BINS:
            self.assertAlmostEqual(right - left, 0.002)

    def test_gap_diagnostics_preserve_degeneracy(self) -> None:
        result = scan.gap_diagnostics(np.array([0.9, 1.0, 1.0, 1.1]))
        self.assertEqual(result["number_of_raw_gaps"], 3)
        self.assertEqual(result["number_of_degenerate_gaps"], 1)
        self.assertAlmostEqual(result["degenerate_gap_fraction"], 1.0 / 3.0)
        np.testing.assert_allclose(result["inclusive_gap_ratios"], [0.0, 0.0])
        self.assertEqual(len(result["regular_gap_ratios"]), 0)

    def test_regular_gap_ratio_uses_adjacent_original_gaps(self) -> None:
        result = scan.gap_diagnostics(np.array([0.0, 1.0, 3.0, 7.0]))
        np.testing.assert_allclose(result["regular_gap_ratios"], [0.5, 0.5])

    def test_filter_removes_perron_and_zero_atom_only(self) -> None:
        values, zero_count, perron = scan.filter_continuous_nonperron_eigenvalues(
            np.array([-1.0, 0.0, 1.0, 3.0])
        )
        np.testing.assert_array_equal(values, [-1.0, 1.0])
        self.assertEqual(zero_count, 1)
        self.assertEqual(perron, 3.0)

    def test_analytic_spacing_cdfs(self) -> None:
        self.assertAlmostEqual(float(scan.goe_spacing_cdf(np.array([0.0]))[0]), 0.0)
        self.assertAlmostEqual(float(scan.poisson_spacing_cdf(np.array([0.0]))[0]), 0.0)
        self.assertGreater(float(scan.goe_spacing_cdf(np.array([1.0]))[0]), 0.0)

    def test_small_end_to_end_graph_scan(self) -> None:
        rows = scan.graph_scan_rows(
            graph_index=0,
            graph_seed=101,
            sequence_seed=201,
            displayed_tau=0.8,
            n=240,
            local_quantile_bins=[(0.60, 0.65), (0.65, 0.70)],
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["effective_density_exponent"], 1.8)
        self.assertGreater(rows[0]["number_of_eigenvalues"], 0)


if __name__ == "__main__":
    unittest.main()
