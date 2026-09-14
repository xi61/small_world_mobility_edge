from __future__ import annotations

import unittest

import networkx as nx
import numpy as np

from small_world_mobility_edge.preferential_attachment.baseline import (
    MODEL_AFFINE_PA,
    attachment_offset_for_ccdf_tau,
    generate_preferential_attachment_graph,
)
from small_world_mobility_edge.preferential_attachment import full_spectrum_scan as scan


class PreferentialAttachmentFullSpectrumTests(unittest.TestCase):
    def test_attachment_mapping_and_connected_graph(self) -> None:
        self.assertAlmostEqual(attachment_offset_for_ccdf_tau(10, 1.8), -2.0)
        self.assertAlmostEqual(attachment_offset_for_ccdf_tau(10, 1.5), -5.0)
        self.assertAlmostEqual(attachment_offset_for_ccdf_tau(10, 1.05), -9.5)
        graph, diagnostics = generate_preferential_attachment_graph(
            n=120,
            m=4,
            seed=31,
            model=MODEL_AFFINE_PA,
            attachment_ccdf_tau=1.5,
        )
        self.assertEqual(graph.number_of_nodes(), 120)
        self.assertEqual(diagnostics.num_components, 1)
        self.assertEqual(diagnostics.min_realized_degree, 4)
        self.assertEqual(
            sum(1 for component in nx.connected_components(graph) if len(component) == 2),
            0,
        )

    def test_quantile_bins_cover_full_spectrum(self) -> None:
        self.assertEqual(len(scan.QUANTILE_BINS), 500)
        self.assertEqual(scan.QUANTILE_BINS[0], (0.0, 0.002))
        self.assertEqual(scan.QUANTILE_BINS[-1], (0.998, 1.0))

    def test_gap_ratios_retain_zero_positive_pairs(self) -> None:
        spacings = np.asarray([0.0, 1.0, 2.0, 0.0])
        inclusive = scan._gap_ratios(spacings, positive_only=False)
        positive = scan._gap_ratios(spacings, positive_only=True)
        np.testing.assert_allclose(inclusive, [0.0, 0.5, 0.0])
        np.testing.assert_allclose(positive, [0.5])

    def test_localization_rows_have_full_quantile_grid(self) -> None:
        rng = np.random.default_rng(4)
        matrix = rng.normal(size=(30, 30))
        q, _ = np.linalg.qr(matrix)
        values = np.linspace(-3.0, 3.0, 30)
        diagnostics = {
            "zero_eigenvalue_count": 0,
            "zero_eigenvalue_fraction": 0.0,
            "perron_eigenvalue": 4.0,
            "realized_mean_degree": 8.0,
            "maximum_degree": 15,
        }
        rows, decay = scan.localization_rows(
            continuous_values=values,
            continuous_vectors=q,
            graph_index=0,
            graph_seed=1,
            alpha=1.5,
            diagnostics=diagnostics,
        )
        self.assertEqual(len(rows), 500)
        self.assertEqual(decay.shape, (500, 200))
        self.assertTrue(np.all(np.isfinite(decay)))


if __name__ == "__main__":
    unittest.main()
