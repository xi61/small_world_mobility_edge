from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile
import unittest

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from small_world_mobility_edge.chung_lu_degree_spectral_tail_match import sparse_tail_spectrum
from small_world_mobility_edge.fixed_degree_configuration_baseline import rewire_graph_preserving_degrees
from small_world_mobility_edge.preferential_attachment.baseline import (
    INITIAL_GRAPH_CONVENTION,
    MODEL_AFFINE_PA,
    MODEL_BA,
    attachment_offset_for_ccdf_tau,
    expected_ba_edge_count,
    generate_preferential_attachment_graph,
)
from small_world_mobility_edge.preferential_attachment.experiments import (
    PreferentialAttachmentConfig,
    aggregate_partials,
    fixed_rank_windows,
    load_graph_partial,
    run_experiment,
    run_graph_partial,
    spacing_localization_rows,
    validate_rank_windows,
)


class PreferentialAttachmentExperimentTests(unittest.TestCase):
    def test_affine_attachment_tau_mapping_and_invariants(self) -> None:
        self.assertAlmostEqual(attachment_offset_for_ccdf_tau(10, 1.8), -2.0)
        self.assertAlmostEqual(attachment_offset_for_ccdf_tau(10, 1.5), -5.0)
        self.assertAlmostEqual(attachment_offset_for_ccdf_tau(10, 1.05), -9.5)
        with self.assertRaises(ValueError):
            attachment_offset_for_ccdf_tau(10, 1.0)
        graph, diagnostics = generate_preferential_attachment_graph(
            n=160,
            m=4,
            seed=13,
            model=MODEL_AFFINE_PA,
            attachment_ccdf_tau=1.5,
        )
        self.assertEqual(graph.number_of_edges(), expected_ba_edge_count(160, 4))
        self.assertEqual(nx.number_connected_components(graph), 1)
        self.assertAlmostEqual(diagnostics.attachment_offset, -2.0)
        self.assertAlmostEqual(diagnostics.theoretical_degree_ccdf_tau, 1.5)

    def test_ba_generator_is_explicit_simple_and_has_exact_edge_count(self) -> None:
        graph, diagnostics = generate_preferential_attachment_graph(
            n=120,
            m=4,
            seed=17,
            model=MODEL_BA,
        )
        expected_edges = expected_ba_edge_count(120, 4)
        self.assertIs(type(graph), nx.Graph)
        self.assertFalse(graph.is_directed())
        self.assertFalse(graph.is_multigraph())
        self.assertEqual(nx.number_of_selfloops(graph), 0)
        self.assertEqual(set(graph.nodes()), set(range(120)))
        self.assertEqual(graph.number_of_edges(), expected_edges)
        self.assertEqual(diagnostics.expected_num_edges, expected_edges)
        self.assertEqual(diagnostics.initial_graph_convention, INITIAL_GRAPH_CONVENTION)
        self.assertAlmostEqual(diagnostics.realized_mean_degree, 2.0 * expected_edges / 120)

    def test_sparse_edges_match_dense_and_eigenpairs_are_normalized(self) -> None:
        graph, _ = generate_preferential_attachment_graph(n=120, m=4, seed=31)
        spectrum = sparse_tail_spectrum(
            graph,
            model_variant=MODEL_BA,
            k_per_side=24,
            eigsh_tol=1e-10,
            eigsh_maxiter=10000,
            eigsh_ncv_factor=3,
            validate_dense=True,
        )
        self.assertLess(spectrum.dense_max_abs_error, 1e-7)
        self.assertLess(float(np.max(spectrum.residuals_la)), 1e-7)
        self.assertLess(float(np.max(spectrum.residuals_sa)), 1e-7)
        self.assertTrue(np.allclose(np.linalg.norm(spectrum.eigenvectors_la, axis=0), 1.0, atol=1e-10))
        self.assertTrue(np.allclose(np.linalg.norm(spectrum.eigenvectors_sa, axis=0), 1.0, atol=1e-10))
        self.assertEqual(spectrum.perron_eigenvalue, spectrum.eigenvalues_la[-1])
        self.assertNotIn(spectrum.perron_eigenvalue, spectrum.positive_nonperron)

    def test_rank_windows_are_disjoint_and_keep_matched_vectors(self) -> None:
        windows = fixed_rank_windows(width=5, count=3)
        validate_rank_windows(windows)
        used = [rank for start, stop in windows for rank in range(start, stop + 1)]
        self.assertEqual(len(used), len(set(used)))
        graph, _ = generate_preferential_attachment_graph(n=120, m=4, seed=37)
        spectrum = sparse_tail_spectrum(
            graph,
            model_variant=MODEL_BA,
            k_per_side=24,
            eigsh_tol=1e-10,
            eigsh_maxiter=10000,
            eigsh_ncv_factor=3,
        )
        config = PreferentialAttachmentConfig(
            n=120,
            m=4,
            num_graphs=1,
            k_per_side=24,
            target_tail_observations=20,
            rank_window_width=5,
            num_rank_windows=3,
        )
        rows, decay = spacing_localization_rows(
            config=config,
            graph_index=0,
            model_variant=MODEL_BA,
            spectrum=spectrum,
            analyzed_dimension=120,
        )
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(int(row["number_of_gaps"]) == 5 for row in rows))
        self.assertTrue(all(int(row["number_of_eigenvalues"]) == 6 for row in rows))
        self.assertTrue(all(np.isfinite(float(row["mean_n_times_ipr"])) for row in rows))
        self.assertGreater(len(decay), 0)

    def test_rewiring_preserves_every_labeled_degree(self) -> None:
        graph, _ = generate_preferential_attachment_graph(n=120, m=4, seed=41)
        rewired, requested, max_tries, overlap = rewire_graph_preserving_degrees(
            graph,
            rewire_seed=43,
            swaps_per_edge=1,
            max_tries_factor=20,
        )
        self.assertEqual(dict(graph.degree()), dict(rewired.degree()))
        self.assertEqual(set(graph.nodes()), set(rewired.nodes()))
        self.assertEqual(graph.number_of_edges(), rewired.number_of_edges())
        self.assertGreater(requested, 0)
        self.assertGreater(max_tries, requested)
        self.assertGreaterEqual(overlap, 0.0)
        self.assertLessEqual(overlap, 1.0)

    def test_partial_aggregation_uses_n_minus_one_and_reproduces_rows(self) -> None:
        root = Path(tempfile.mkdtemp())
        config = PreferentialAttachmentConfig(
            n=80,
            m=2,
            num_graphs=2,
            target_mean_degree=4.0,
            swaps_per_edge=1,
            k_per_side=16,
            target_tail_observations=20,
            tail_lower_quantiles=(0.75,),
            rank_window_width=5,
            num_rank_windows=2,
            output_root=root / "partial_run",
        )
        serial_config = PreferentialAttachmentConfig(
            **{
                **config.__dict__,
                "output_root": root / "serial_run",
            }
        )
        serial_destination = run_experiment(serial_config)
        partial_dir = root / "partials"
        paths = [run_graph_partial(config, index, partial_dir=partial_dir) for index in range(2)]
        loaded_before = [load_graph_partial(path) for path in paths]
        destination = root / "aggregated"
        aggregate_partials(config, partial_dir=partial_dir, output_dir=destination)
        loaded_after = [load_graph_partial(path) for path in paths]
        self.assertEqual(
            [partial["threshold_rows"] for partial in loaded_before],
            [partial["threshold_rows"] for partial in loaded_after],
        )
        with (destination / "degree_spectral_tail_by_graph.csv").open(newline="") as handle:
            tail_rows = list(csv.DictReader(handle))
        self.assertTrue(tail_rows)
        self.assertTrue(all(int(row["spectral_ccdf_denominator"]) == 79 for row in tail_rows))
        with (serial_destination / "degree_spectral_tail_by_graph.csv").open(newline="") as handle:
            serial_rows = list(csv.DictReader(handle))
        self.assertEqual(
            [
                (row["graph_index"], row["model_variant"], row["tail_lower_quantile"])
                for row in serial_rows
            ],
            [
                (row["graph_index"], row["model_variant"], row["tail_lower_quantile"])
                for row in tail_rows
            ],
        )
        for serial, partial in zip(serial_rows, tail_rows):
            self.assertAlmostEqual(
                float(serial["absolute_spectral_tau_hat"]),
                float(partial["absolute_spectral_tau_hat"]),
                places=9,
            )
        with (destination / "degree_preservation_checks.csv").open(newline="") as handle:
            checks = list(csv.DictReader(handle))
        self.assertTrue(all(row["degree_match"] == "True" for row in checks))


if __name__ == "__main__":
    unittest.main()
