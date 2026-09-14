from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from small_world_mobility_edge.fixed_degree_configuration_baseline import (
    generate_paired_configuration_ensemble_from_seeds,
)
from small_world_mobility_edge.fixed_degree_configuration_joint_tail import (
    aggregate_difference_rows,
    aggregate_joint_rows,
    graph_variant_tail_rows,
    paired_difference_rows,
)


class FixedDegreeConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pairs = list(
            generate_paired_configuration_ensemble_from_seeds(
                n=300,
                tau=1.5,
                target_mean_degree=20.0,
                w_min=1.0,
                w_max=None,
                cutoff_mode="structural",
                cutoff_coefficient=1.0,
                sequence_mode="quantile",
                graph_seeds=[101, 103, 107],
                sequence_seeds=[201, 203, 207],
                rewire_seeds=[301, 303, 307],
                component_mode="full",
                swaps_per_edge=2,
                max_tries_factor=20,
                retain_eigenvectors=True,
            )
        )

    def test_smoke_pairs_preserve_every_labeled_degree(self) -> None:
        self.assertEqual(len(self.pairs), 3)
        for pair in self.pairs:
            original = pair.original
            configuration = pair.configuration
            self.assertTrue(pair.degree_match)
            self.assertTrue(np.array_equal(original.node_labels, configuration.node_labels))
            self.assertTrue(np.array_equal(original.degrees, configuration.degrees))
            self.assertEqual(len(original.node_labels), len(configuration.node_labels))
            self.assertEqual(int(np.sum(original.degrees)), int(np.sum(configuration.degrees)))
            self.assertGreater(pair.requested_swaps, 0)
            self.assertGreaterEqual(pair.edge_overlap_fraction, 0.0)
            self.assertLessEqual(pair.edge_overlap_fraction, 1.0)

    def test_spectra_and_eigenvectors_are_consistent(self) -> None:
        for pair in self.pairs:
            for result in (pair.original, pair.configuration):
                self.assertTrue(np.all(np.isfinite(result.eigenvalues)))
                self.assertEqual(len(result.bulk_eigenvalues), result.analyzed_dimension - 1)
                self.assertEqual(result.bulk_eigenvectors.shape, (result.analyzed_dimension, result.analyzed_dimension - 1))
                norms = np.linalg.norm(result.bulk_eigenvectors, axis=0)
                self.assertTrue(np.allclose(norms, 1.0, atol=1e-10))
                self.assertEqual(result.perron_eigenvalue, result.eigenvalues[-1])
                self.assertTrue(np.array_equal(result.bulk_eigenvalues, result.eigenvalues[:-1]))

    def test_rank_gap_joint_rows_and_paired_differences(self) -> None:
        bins = ((0.95, 0.97), (0.97, 0.99))
        rows = []
        decay_rows = []
        for pair in self.pairs:
            original_rows, original_decay = graph_variant_tail_rows(pair.original, pair, bins)
            configuration_rows, configuration_decay = graph_variant_tail_rows(pair.configuration, pair, bins)
            rows.extend(original_rows)
            rows.extend(configuration_rows)
            decay_rows.extend(original_decay)
            decay_rows.extend(configuration_decay)
        self.assertEqual(len(rows), 3 * 2 * len(bins))
        self.assertTrue(all(int(row["number_of_gaps"]) >= 1 for row in rows))
        summaries = aggregate_joint_rows(rows, rng_seed=99)
        differences = paired_difference_rows(rows)
        difference_summaries = aggregate_difference_rows(differences)
        self.assertEqual(len(summaries), 2 * len(bins))
        self.assertEqual(len(differences), 3 * len(bins))
        self.assertEqual(len(difference_summaries), len(bins))
        self.assertTrue(all(int(row["number_of_gaps"]) > 0 for row in summaries))
        self.assertGreater(len(decay_rows), 0)


if __name__ == "__main__":
    unittest.main()
