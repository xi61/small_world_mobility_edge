import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from small_world_mobility_edge.preferential_attachment.multifractality import (
    analyze_case,
    aggregate_results,
    configuration_identity,
    controlled_validation,
    estimate_tau,
    generalized_log_ipr,
    multiplicity_cluster_sizes,
    q_grid,
    local_gap_diagnostic,
    spectral_window_indices,
)


class PAMultifractalityTests(unittest.TestCase):
    def test_local_gap_keeps_atom_but_uses_resolved_scale(self):
        result = local_gap_diagnostic(np.array([-2., 0., 0., 1., 3.]), 2, 1e-8, 3)
        self.assertEqual(result["minimum_neighbor_gap"], 0.0)
        self.assertEqual(result["local_positive_gap_median"], 2.0)
        self.assertEqual(result["minimum_gap_over_local_median"], 0.0)

    def test_aggregate_rejects_wrong_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory) / "N_100/alpha_2"
            case.mkdir(parents=True)
            (case / "case_manifest.json").write_text(json.dumps({"status": "COMPLETE", "configuration_sha256": "wrong"}))
            with self.assertRaisesRegex(ValueError, "configuration identity"):
                aggregate_results(output_root=Path(directory), report_dir=Path(directory) / "report",
                                  config={"sizes": [100], "alpha_grid": [2.]})

    def test_aggregate_rejects_wrong_case_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory) / "N_100/alpha_2"
            case.mkdir(parents=True)
            config = {"sizes": [100], "alpha_grid": [2.]}
            (case / "case_manifest.json").write_text(json.dumps({"status": "COMPLETE", "configuration_sha256": configuration_identity(config),
                                                                 "case": {"n": 200, "degree_ccdf_alpha": 2.}}))
            with self.assertRaisesRegex(ValueError, "unexpected"):
                aggregate_results(output_root=Path(directory), report_dir=Path(directory) / "report", config=config)

    def test_generalized_ipr_uniform_and_localized_limits(self):
        q = np.asarray([0.0, 0.5, 1.0, 2.0, 3.0])
        n = 100
        uniform = generalized_log_ipr(np.full(n, 1.0 / n), q)[:, 0]
        localized_mass = np.zeros(n)
        localized_mass[0] = 1.0
        localized = generalized_log_ipr(localized_mass, q)[:, 0]
        self.assertTrue(np.allclose(uniform, (1.0 - q) * math.log(n)))
        self.assertTrue(np.allclose(localized, np.asarray([math.log(n), 0.0, 0.0, 0.0, 0.0])))

    def test_q_grid_reproduces_paper_step(self):
        grid = q_grid({"q_min": 0.0, "q_max": 3.0, "q_step": 0.02})
        self.assertEqual(len(grid), 151)
        self.assertAlmostEqual(grid[1] - grid[0], 0.02)
        self.assertAlmostEqual(grid[-1], 3.0)

    def test_spectral_windows_have_fixed_counts_and_exclude_perron(self):
        quantile = spectral_window_indices(500, {"kind": "quantile", "center": 0.5, "count": 50})
        edge = spectral_window_indices(500, {"kind": "descending_rank", "first_rank": 2, "count": 50})
        self.assertEqual(len(quantile), 50)
        self.assertEqual(edge[0], 449)
        self.assertEqual(edge[-1], 498)
        self.assertNotIn(499, edge)

    def test_multiplicity_clusters_chain_only_within_tolerance(self):
        values = np.asarray([0.0, 1e-12, 2e-12, 1.0, 1.0 + 1e-8])
        result = multiplicity_cluster_sizes(values, 2e-12)
        self.assertEqual(result.tolist(), [3, 3, 3, 1, 1])

    def test_two_size_tau_sign_convention(self):
        q = 2.5
        values = {100: (1.0 - q) * math.log(100), 1000: (1.0 - q) * math.log(1000)}
        tau, r_squared = estimate_tau(values)
        self.assertAlmostEqual(tau, q - 1.0)
        self.assertAlmostEqual(r_squared, 1.0)

    def test_controlled_fixtures_pass(self):
        rows, passed = controlled_validation()
        self.assertTrue(passed, msg=[row for row in rows if not row["passed"]])

    def test_small_cached_eigensystem_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            arrays = source / "arrays"
            arrays.mkdir(parents=True)
            n = 12
            rng = np.random.default_rng(7)
            vectors, _ = np.linalg.qr(rng.normal(size=(n, n)))
            values = np.linspace(-2.0, 3.0, n)
            np.save(arrays / "full_eigenvalues_ascending.npy", values)
            np.save(arrays / "full_eigenvectors_ascending.npy", vectors)
            edge_u = np.arange(n - 1, dtype=int)
            edge_v = np.arange(1, n, dtype=int)
            degrees = np.asarray([1] + [2] * (n - 2) + [1])
            np.savez(
                arrays / "graph_edges.npz",
                n=n,
                edge_u=edge_u,
                edge_v=edge_v,
                degrees=degrees,
            )
            (source / "spectral_cache_manifest.json").write_text(
                json.dumps({"spectral_cache_identity": "fixture"})
            )
            config = {
                "analysis_id": "fixture",
                "moments": {"q_min": 0.0, "q_max": 2.0, "q_step": 0.5},
                "selection": {
                    "band_size": 4,
                    "multiplicity_tolerance_factor": 64.0,
                    "local_gap_radius": 2,
                    "top_r": 4,
                    "high_degree_fraction": 0.25,
                },
                "alpha_histogram": {"minimum": 0.0, "maximum": 8.0, "width": 0.25},
                "log10_scaled_mass_histogram": {"minimum": -16.0, "maximum": 4.0, "width": 0.25},
                "windows": [{"id": "middle", "kind": "quantile", "center": 0.5, "count": 4}],
            }
            output = root / "output"
            manifest = analyze_case(
                source_case_dir=source,
                output_case_dir=output,
                n=n,
                alpha=2.0,
                config=config,
            )
            self.assertEqual(manifest["status"], "COMPLETE")
            self.assertEqual(manifest["output_row_counts"]["eigenvector_diagnostics"], 4)
            moments = (output / "tables/typical_moments.csv").read_text()
            self.assertIn("typical_iq", moments)
            self.assertTrue((output / "tables/singularity_spectrum_single_size.csv").exists())


if __name__ == "__main__":
    unittest.main()
