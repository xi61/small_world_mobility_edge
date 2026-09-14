from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np

from small_world_mobility_edge import chung_lu_eigenvector_percentile_scan as scan


class ChungLuEigenvectorPercentileScanTests(unittest.TestCase):
    def test_equal_width_bins_cover_upper_half(self) -> None:
        bins = scan.quantile_bins()
        self.assertEqual(len(bins), 250)
        self.assertAlmostEqual(bins[0][0], 0.5)
        self.assertAlmostEqual(bins[-1][1], 1.0)
        for left, right in bins:
            self.assertAlmostEqual(right - left, 0.002)

    def test_legacy_tau_mapping_is_explicit(self) -> None:
        tau_08 = scan.tau_specification(0.8)
        tau_105 = scan.tau_specification(1.05)
        tau_15 = scan.tau_specification(1.5)
        self.assertEqual(tau_08.tau_convention, "experimental_ccdf_exponent")
        self.assertEqual(tau_08.effective_density_exponent, 1.8)
        self.assertEqual(tau_105.effective_density_exponent, 1.05)
        self.assertEqual(tau_15.effective_density_exponent, 1.5)

    def test_seed_streams_match_saved_eigenvalue_runs(self) -> None:
        graph_08, _ = scan.graph_and_sequence_seeds(0.8)
        graph_105, _ = scan.graph_and_sequence_seeds(1.05)
        graph_15, sequence_15 = scan.graph_and_sequence_seeds(1.5)
        self.assertEqual(graph_08[0], 2832232046)
        self.assertEqual(graph_105[0], 4011773341)
        self.assertEqual(graph_15[0], 4006230582)
        self.assertEqual(sequence_15[0], 2948808800)

    def test_localization_arrays_distinguish_flat_and_one_site_vectors(self) -> None:
        flat = np.full(10, 1.0 / np.sqrt(10.0))
        localized = np.zeros(10)
        localized[0] = 1.0
        arrays = scan.localization_arrays(np.column_stack((flat, localized)))
        np.testing.assert_allclose(arrays["ipr"], [0.1, 1.0])
        np.testing.assert_allclose(arrays["top_1_mass"], [0.1, 1.0])
        np.testing.assert_array_equal(arrays["r_90"], [9, 1])

    def test_zero_filter_removes_matching_vector_columns(self) -> None:
        values = np.array([-1.0, 0.0, 1.0, 3.0])
        vectors = np.eye(4)
        kept_values, kept_vectors, zero_count, perron = scan.filter_continuous_nonperron_eigenpairs(
            values,
            vectors,
        )
        np.testing.assert_array_equal(kept_values, [-1.0, 1.0])
        np.testing.assert_array_equal(kept_vectors, vectors[:, [0, 2]])
        self.assertEqual(zero_count, 1)
        self.assertEqual(perron, 3.0)

    def test_goe_top_200_baseline_exceeds_flat_baseline(self) -> None:
        top_200 = scan.asymptotic_goe_top_r_mass(5000, 200)
        self.assertGreater(top_200, 200 / 5000)
        self.assertAlmostEqual(top_200, 0.239, places=2)

    def test_small_end_to_end_graph_scan(self) -> None:
        rows, decay, diagnostics = scan.graph_scan_rows(
            graph_index=0,
            graph_seed=101,
            sequence_seed=201,
            displayed_tau=0.8,
            n=240,
            local_quantile_bins=[(0.50, 0.55), (0.95, 1.00)],
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(decay.shape, (2, 200))
        self.assertAlmostEqual(diagnostics.final_mean, scan.TARGET_MEAN_DEGREE)
        self.assertEqual(rows[0]["effective_density_exponent"], 1.8)
        self.assertGreater(rows[0]["number_of_eigenvectors"], 0)

    def test_lcc_scan_records_component_mode_and_reduces_dimension(self) -> None:
        rows, decay, _ = scan.graph_scan_rows(
            graph_index=0,
            graph_seed=101,
            sequence_seed=201,
            displayed_tau=1.5,
            n=180,
            local_quantile_bins=[(0.50, 0.55), (0.95, 1.00)],
            component_mode="largest_connected_component",
        )
        self.assertEqual(decay.shape[0], 2)
        self.assertTrue(all(row["component_mode"] == "largest_connected_component" for row in rows))
        self.assertTrue(all(row["analyzed_dimension"] == row["largest_component_size"] for row in rows))
        self.assertTrue(all(row["analyzed_dimension"] <= 180 for row in rows))
        self.assertEqual(decay.shape[1], min(200, rows[0]["analyzed_dimension"]))

    def test_streaming_aggregation_matches_in_memory_aggregation(self) -> None:
        rows, _, _ = scan.graph_scan_rows(
            graph_index=0,
            graph_seed=101,
            sequence_seed=201,
            displayed_tau=0.8,
            n=120,
            local_quantile_bins=[(0.50, 0.55), (0.95, 1.00)],
        )
        string_rows = [{key: str(value) for key, value in row.items()} for row in rows]
        expected = scan.aggregate_graph_rows(string_rows)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partial = root / "partial.csv.gz"
            scan.write_csv(rows, partial, scan.graph_row_fieldnames())
            actual, graph_ids = scan.aggregate_graph_csv_paths([partial], root / "all.csv.gz")
        self.assertEqual(graph_ids, [0])
        self.assertEqual(len(actual), len(expected))
        self.assertAlmostEqual(actual[0]["mean_n_times_ipr"], expected[0]["mean_n_times_ipr"])


if __name__ == "__main__":
    unittest.main()
