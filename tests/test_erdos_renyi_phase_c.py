from __future__ import annotations

from pathlib import Path
import sys
import unittest

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from small_world_mobility_edge.empirical_spectral_density import estimate_empirical_spectral_density
from small_world_mobility_edge.erdos_renyi_baseline import (
    ErdosRenyiSpectrum,
    analyze_erdos_renyi_graph,
    generate_erdos_renyi_ensemble,
)
from small_world_mobility_edge.erdos_renyi_spacing_by_energy import (
    CONTINUOUS_NONZERO,
    ZERO_TOLERANCE,
    build_empirical_density,
    compute_local_spacing_summary,
    continuous_bulk_eigenvalues,
)
from small_world_mobility_edge.regular_baseline import local_window_spacings


def fake_spectrum(graph_index: int, bulk_eigenvalues: np.ndarray) -> ErdosRenyiSpectrum:
    bulk_eigenvalues = np.sort(np.asarray(bulk_eigenvalues, dtype=float))
    eigenvalues = np.sort(np.append(bulk_eigenvalues, 3.0))
    return ErdosRenyiSpectrum(
        graph_index=graph_index,
        seed=1000 + graph_index,
        n_requested=len(eigenvalues),
        average_degree_requested=6.0,
        p=0.1,
        component_mode="full",
        analyzed_dimension=len(eigenvalues),
        realized_mean_degree=6.0,
        num_components=1,
        largest_component_size=len(eigenvalues),
        num_isolated_vertices=0,
        degrees=np.full(len(eigenvalues), 6.0),
        eigenvalues=eigenvalues,
        bulk_eigenvalues=bulk_eigenvalues,
        perron_eigenvalue=3.0,
    )


class ErdosRenyiPhaseCTests(unittest.TestCase):
    def test_perron_outlier_is_excluded(self) -> None:
        spectrum = analyze_erdos_renyi_graph(
            n=300,
            average_degree=6,
            seed=17,
            component_mode="largest_connected_component",
        )
        self.assertEqual(len(spectrum.bulk_eigenvalues), spectrum.analyzed_dimension - 1)
        self.assertLess(np.max(spectrum.bulk_eigenvalues), spectrum.perron_eigenvalue)

    def test_empirical_density_integrates_to_one(self) -> None:
        spectra = generate_erdos_renyi_ensemble(
            n=300,
            average_degree=6,
            num_graphs=6,
            seed=19,
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

    def test_selector_density_excludes_zero_atom_and_uses_retained_count(self) -> None:
        spectra = [
            fake_spectrum(0, np.array([-2.0, -1.0, 0.0, 0.0, 1.0])),
            fake_spectrum(1, np.array([-1.5, 0.0, 0.0, 0.5, 2.0])),
        ]

        raw_density = build_empirical_density(spectra, select_values=lambda spectrum: spectrum.bulk_eigenvalues)
        continuous_density = build_empirical_density(spectra, select_values=continuous_bulk_eigenvalues)

        self.assertEqual(raw_density.per_graph_dimension, 5.0)
        self.assertEqual(continuous_density.per_graph_dimension, 3.0)
        self.assertFalse(
            np.any(np.isclose(continuous_density.pooled_eigenvalues, 0.0, atol=ZERO_TOLERANCE, rtol=0.0))
        )
        self.assertLess(abs(float(integrate.trapezoid(continuous_density.density, continuous_density.grid)) - 1.0), 0.05)

    def test_selector_density_rejects_empty_continuous_sample(self) -> None:
        spectra = [fake_spectrum(0, np.array([0.0, 0.0, 0.0]))]

        with self.assertRaisesRegex(ValueError, "retained no eigenvalues"):
            build_empirical_density(spectra, select_values=continuous_bulk_eigenvalues)

    def test_continuous_spacing_summary_uses_retained_unfolding_dimension(self) -> None:
        base = np.linspace(-0.24, 0.24, 49)
        spectra = [
            fake_spectrum(0, base),
            fake_spectrum(1, base + 0.001),
        ]
        density = build_empirical_density(spectra, select_values=continuous_bulk_eigenvalues)

        fig_grid, fig_summary, rows = compute_local_spacing_summary(
            spectra=spectra,
            density=density,
            E_values=[0.0],
            average_degree=6.0,
            spectral_quantiles=[None],
            analysis_variant=CONTINUOUS_NONZERO,
            component_mode="full",
            run_meta={"code_version": None, "run_id": "test", "timestamp": "test"},
            select_values=continuous_bulk_eigenvalues,
            density_variant=CONTINUOUS_NONZERO,
            selected_spectrum_variant=CONTINUOUS_NONZERO,
            rng_seed=11,
        )
        try:
            row = rows[0]
            self.assertEqual(row["analysis_variant"], CONTINUOUS_NONZERO)
            self.assertTrue(row["unfolding_dimension_matches_retained_count"])
            self.assertEqual(row["mean_unfolding_dimension"], row["mean_retained_eigenvalue_count"])
            self.assertEqual(row["number_of_exact_zero_gaps_in_center_window"], 0.0)
            self.assertTrue(np.all(np.isfinite(np.asarray(row["spacings"], dtype=float))))
        finally:
            plt.close(fig_grid)
            plt.close(fig_summary)

    def test_empirical_local_window_spacing_is_finite(self) -> None:
        spectra = generate_erdos_renyi_ensemble(
            n=400,
            average_degree=6,
            num_graphs=8,
            seed=23,
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

        self.assertEqual(result.num_gaps, result.num_eigenvalues - 1)
        self.assertTrue(np.all(np.isfinite(result.spacings)))


if __name__ == "__main__":
    unittest.main()
