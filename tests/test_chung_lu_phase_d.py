from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
from scipy import integrate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from small_world_mobility_edge.chung_lu_baseline import (
    clipped_mean_from_scale,
    constant_weight_value,
    generate_power_law_chung_lu_ensemble_from_seeds,
    generate_power_law_expected_weights,
    power_law_quantile_sequence,
    sampled_pareto_sequence,
    solve_clipped_mean_scaling_constant,
)
from small_world_mobility_edge.empirical_spectral_density import estimate_empirical_spectral_density
from small_world_mobility_edge.regular_baseline import adjacent_gap_ratios_from_spacings, local_window_spacings


class ChungLuPhaseDTests(unittest.TestCase):
    def test_constant_weight_helper_remains_available_as_sanity_check(self) -> None:
        n = 500
        average_degree = 6.0
        expected_weight = average_degree * n / (n - 1)
        self.assertAlmostEqual(constant_weight_value(n, average_degree), expected_weight)

    def test_power_law_weight_sequence_hits_target_mean_and_admissibility(self) -> None:
        weights, diagnostics = generate_power_law_expected_weights(
            n=500,
            tau=3.0,
            target_mean_degree=10.0,
            w_min=1.0,
            w_max=None,
            cutoff_mode="structural",
            cutoff_coefficient=1.0,
            sequence_mode="quantile",
            sequence_seed=17,
        )
        self.assertLess(abs(float(np.mean(weights)) - 10.0), 1e-6)
        self.assertTrue(diagnostics.admissibility_holds)
        self.assertLessEqual(np.max(weights) ** 2, np.sum(weights) + 1e-10)
        self.assertAlmostEqual(diagnostics.final_mean, 10.0, places=8)

    def test_power_law_sequence_is_reproducible_for_fixed_seed(self) -> None:
        weights_a, diagnostics_a = generate_power_law_expected_weights(
            n=400,
            tau=3.0,
            target_mean_degree=8.0,
            w_min=1.0,
            w_max=40.0,
            cutoff_mode="fixed",
            cutoff_coefficient=1.0,
            sequence_mode="sampled",
            sequence_seed=1234,
        )
        weights_b, diagnostics_b = generate_power_law_expected_weights(
            n=400,
            tau=3.0,
            target_mean_degree=8.0,
            w_min=1.0,
            w_max=40.0,
            cutoff_mode="fixed",
            cutoff_coefficient=1.0,
            sequence_mode="sampled",
            sequence_seed=1234,
        )
        self.assertTrue(np.allclose(weights_a, weights_b))
        self.assertAlmostEqual(diagnostics_a.scaling_constant, diagnostics_b.scaling_constant, places=14)

    def test_power_law_density_integrates_to_one(self) -> None:
        spectra = generate_power_law_chung_lu_ensemble_from_seeds(
            n=300,
            tau=3.0,
            target_mean_degree=10.0,
            w_min=1.0,
            w_max=None,
            cutoff_mode="structural",
            cutoff_coefficient=1.0,
            sequence_mode="quantile",
            graph_seeds=[11, 13, 17, 19, 23, 29],
            sequence_seeds=[31, 37, 41, 43, 47, 53],
            component_mode="largest_connected_component",
        )
        density = estimate_empirical_spectral_density(
            eigenvalue_samples=[spectrum.bulk_eigenvalues for spectrum in spectra],
            num_graphs=len(spectra),
            per_graph_dimension=float(np.mean([spectrum.analyzed_dimension for spectrum in spectra])),
            method="kde",
            num_points=1201,
        )
        self.assertLess(abs(float(integrate.trapezoid(density.density, density.grid)) - 1.0), 0.05)

    def test_local_window_spacing_is_finite_for_power_law_chung_lu(self) -> None:
        spectra = generate_power_law_chung_lu_ensemble_from_seeds(
            n=400,
            tau=3.0,
            target_mean_degree=10.0,
            w_min=1.0,
            w_max=None,
            cutoff_mode="structural",
            cutoff_coefficient=1.0,
            sequence_mode="quantile",
            graph_seeds=[101, 103, 107, 109, 113, 127],
            sequence_seeds=[131, 137, 139, 149, 151, 157],
            component_mode="largest_connected_component",
        )
        density = estimate_empirical_spectral_density(
            eigenvalue_samples=[spectrum.bulk_eigenvalues for spectrum in spectra],
            num_graphs=len(spectra),
            per_graph_dimension=float(np.mean([spectrum.analyzed_dimension for spectrum in spectra])),
            method="local_count",
            local_count_half_width=0.25,
            num_points=1201,
        )
        result = local_window_spacings(
            eigenvalues=spectra[0].bulk_eigenvalues,
            E=0.0,
            window_half_width=0.25,
            density_at_E=float(density.evaluate(0.0)),
            density_method="empirical",
            analyzed_dimension=spectra[0].analyzed_dimension,
            min_eigenvalues=6,
        )
        gap_ratios = adjacent_gap_ratios_from_spacings(result.spacings)
        self.assertEqual(result.num_gaps, result.num_eigenvalues - 1)
        self.assertTrue(np.all(np.isfinite(result.spacings)))
        self.assertTrue(np.all(np.isfinite(gap_ratios)))

    def test_perron_outlier_is_excluded_and_tau_estimate_is_broadly_consistent(self) -> None:
        spectra = generate_power_law_chung_lu_ensemble_from_seeds(
            n=500,
            tau=3.5,
            target_mean_degree=10.0,
            w_min=1.0,
            w_max=None,
            cutoff_mode="structural",
            cutoff_coefficient=1.0,
            sequence_mode="quantile",
            graph_seeds=[211, 223, 227, 229],
            sequence_seeds=[233, 239, 241, 251],
            component_mode="largest_connected_component",
        )
        for spectrum in spectra:
            self.assertEqual(len(spectrum.bulk_eigenvalues), spectrum.analyzed_dimension - 1)
            self.assertLess(np.max(spectrum.bulk_eigenvalues), spectrum.perron_eigenvalue)
        tau_estimates = [
            spectrum.weight_diagnostics.estimated_tau_after_cutoff
            for spectrum in spectra
            if spectrum.weight_diagnostics.estimated_tau_after_cutoff is not None
        ]
        self.assertTrue(tau_estimates)
        self.assertLess(abs(float(np.mean(tau_estimates)) - 3.5), 1.0)

    def test_no_active_upper_cutoff_preserves_raw_ratios_above_lower_bound(self) -> None:
        raw = power_law_quantile_sequence(n=500, tau=3.0)
        weights, diagnostics = generate_power_law_expected_weights(
            n=500,
            tau=3.0,
            target_mean_degree=10.0,
            w_min=1.0,
            w_max=None,
            cutoff_mode="fixed",
            cutoff_coefficient=1.0,
            sequence_mode="quantile",
            sequence_seed=7,
        )
        free = weights > 1.0 + 1e-10
        self.assertGreater(np.sum(free), 10)
        ratio_raw = raw[free][0] / raw[free][-1]
        ratio_weights = weights[free][0] / weights[free][-1]
        self.assertAlmostEqual(ratio_raw, ratio_weights, places=10)
        self.assertAlmostEqual(diagnostics.final_mean, 10.0, places=8)

    def test_structural_cutoff_uses_sqrt_n_target_mean(self) -> None:
        weights, diagnostics = generate_power_law_expected_weights(
            n=2000,
            tau=3.0,
            target_mean_degree=10.0,
            w_min=1.0,
            w_max=None,
            cutoff_mode="structural",
            cutoff_coefficient=1.0,
            sequence_mode="quantile",
            sequence_seed=17,
        )
        self.assertAlmostEqual(diagnostics.structural_cap, np.sqrt(2000.0 * 10.0), places=8)
        self.assertLessEqual(np.max(weights), diagnostics.structural_cap + 1e-10)
        self.assertAlmostEqual(diagnostics.final_mean, 10.0, places=8)
        self.assertTrue(diagnostics.admissibility_holds)

    def test_regression_cap_for_n2000_tau3_mean10_is_sqrt_20000(self) -> None:
        _, diagnostics = generate_power_law_expected_weights(
            n=2000,
            tau=3.0,
            target_mean_degree=10.0,
            w_min=1.0,
            w_max=None,
            cutoff_mode="structural",
            cutoff_coefficient=1.0,
            sequence_mode="quantile",
            sequence_seed=99,
        )
        self.assertAlmostEqual(diagnostics.final_upper_cap, np.sqrt(20000.0), places=6)

    def test_quantile_and_sampled_raw_sequences_exist(self) -> None:
        quantile_raw = power_law_quantile_sequence(n=100, tau=3.0)
        sampled_raw = sampled_pareto_sequence(n=100, tau=3.0, seed=123)
        self.assertEqual(len(quantile_raw), 100)
        self.assertEqual(len(sampled_raw), 100)
        self.assertTrue(np.all(quantile_raw > 0))
        self.assertTrue(np.all(sampled_raw > 0))

    def test_infeasible_bounds_raise(self) -> None:
        with self.assertRaises(ValueError):
            generate_power_law_expected_weights(
                n=100,
                tau=3.0,
                target_mean_degree=10.0,
                w_min=12.0,
                w_max=8.0,
                cutoff_mode="fixed",
                cutoff_coefficient=1.0,
                sequence_mode="quantile",
                sequence_seed=5,
            )

    def test_clipped_mean_solver_matches_target(self) -> None:
        raw = power_law_quantile_sequence(n=200, tau=3.0)
        scale = solve_clipped_mean_scaling_constant(
            raw_weights=raw,
            target_mean_degree=8.0,
            lower_bound=1.0,
            upper_cap=np.sqrt(200 * 8.0),
        )
        mean_value = clipped_mean_from_scale(raw, scale, 1.0, np.sqrt(200 * 8.0))
        self.assertAlmostEqual(mean_value, 8.0, places=10)


if __name__ == "__main__":
    unittest.main()
