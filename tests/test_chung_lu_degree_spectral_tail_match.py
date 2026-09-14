from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from small_world_mobility_edge.chung_lu_degree_spectral_tail_match import (
    MODEL_CHUNG_LU,
    TailMatchConfig,
    graph_tail_objects,
    pooled_ccdf_rows,
    rank_match_fit,
    sparse_tail_spectrum,
    threshold_rows_for_model,
    top_count_for_quantile,
)


class ChungLuDegreeSpectralTailMatchTests(unittest.TestCase):
    def test_rank_match_fit_can_explicitly_fit_a_top_ten_diagnostic(self) -> None:
        degrees = np.linspace(100.0, 10.0, 10)
        eigenvalues = 1.2 * np.sqrt(degrees)
        default = rank_match_fit(degrees, eigenvalues, max_rank=10)
        top_ten = rank_match_fit(
            degrees, eigenvalues, max_rank=10, minimum_observations=3
        )
        self.assertTrue(np.isnan(default["slope"]))
        self.assertAlmostEqual(top_ten["slope"], 0.5)
        self.assertAlmostEqual(top_ten["median_lambda_over_sqrt_degree"], 1.2)

    def test_sparse_edge_eigenvalues_match_dense_on_small_graph(self) -> None:
        graph = nx.expected_degree_graph(np.full(80, 8.0), seed=17, selfloops=False)
        result = sparse_tail_spectrum(
            graph,
            model_variant=MODEL_CHUNG_LU,
            k_per_side=12,
            eigsh_tol=1e-10,
            eigsh_maxiter=10000,
            eigsh_ncv_factor=3,
            validate_dense=True,
        )

        self.assertLess(result.dense_max_abs_error, 1e-7)
        self.assertLess(float(np.max(result.residuals_la)), 1e-7)
        self.assertLess(float(np.max(result.residuals_sa)), 1e-7)
        self.assertEqual(result.perron_eigenvalue, result.eigenvalues_la[-1])
        self.assertFalse(np.any(np.isclose(result.positive_nonperron, result.perron_eigenvalue)))

    def test_threshold_rows_use_full_nonperron_denominator_count(self) -> None:
        config = TailMatchConfig(n=400, tail_lower_quantiles=(0.95,), target_tail_count=20)
        degree_desc = np.linspace(100.0, 1.0, 400)
        absolute_desc = np.linspace(30.0, 1.0, 60)
        rows = threshold_rows_for_model(
            graph_index=0,
            model_variant=MODEL_CHUNG_LU,
            tau=1.5,
            config=config,
            degree_desc=degree_desc,
            absolute_desc=absolute_desc,
            lambda_squared_desc=absolute_desc**2,
        )

        self.assertEqual(top_count_for_quantile(config.n - 1, 0.95), 20)
        self.assertEqual(rows[0]["spectral_tail_count"], 20)
        self.assertTrue(np.isfinite(rows[0]["alpha_hill"]))
        self.assertTrue(np.isfinite(rows[0]["beta_hill"]))
        self.assertTrue(np.isfinite(rows[0]["gamma_hill"]))

    def test_graph_objects_preserve_fixed_degree_control_and_save_ccdf_denominator(self) -> None:
        config = TailMatchConfig(
            n=120,
            tau_values=(1.5,),
            target_mean_degree=8.0,
            num_graphs=1,
            k_per_side=24,
            target_tail_count=20,
            tail_lower_quantiles=(0.80,),
            run_fixed_degree_control=True,
            swaps_per_edge=1,
            validate_dense=True,
            output_dir=Path(tempfile.mkdtemp()),
        )
        partial = graph_tail_objects(
            graph_index=0,
            graph_seed=101,
            sequence_seed=201,
            rewire_seed=301,
            tau=1.5,
            config=config,
        )
        base, tail_rows, threshold_rows, rank_rows, cutoff_rows, arrays = partial

        self.assertEqual(base["tau_convention"], "experimental_ccdf_exponent")
        self.assertEqual(len(tail_rows), 2)
        self.assertEqual(len(threshold_rows), 2)
        self.assertEqual(len(rank_rows), 4)
        self.assertEqual(len(cutoff_rows), 2)
        self.assertEqual({row["model_variant"] for row in tail_rows}, {"Chung-Lu", "Fixed-degree configuration model"})
        self.assertGreaterEqual(len(arrays["cl_absolute_nonperron_tail"]), config.target_tail_count)
        ccdf = pooled_ccdf_rows(tau=1.5, config=config, partials=[partial])
        spectral_first = next(row for row in ccdf if row["curve"] == "absolute_spectral_ccdf")
        self.assertAlmostEqual(float(spectral_first["tail_probability"]), 1.0 / (config.n - 1))


if __name__ == "__main__":
    unittest.main()
