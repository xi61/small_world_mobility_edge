from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import networkx as nx

from small_world_mobility_edge.preferential_attachment.baseline import (
    MODEL_AFFINE_PA,
    MODEL_NONLINEAR_PA,
    generate_nonlinear_preferential_attachment_graph,
    generate_preferential_attachment_graph,
)
from small_world_mobility_edge.preferential_attachment.mobility_edge_postprocess import (
    adaptive_quantile_window,
    build_cross_fitted_models,
    build_gap_records,
    cross_fitted_unfolded_gaps,
    graph_level_quantile_rows,
    summarize_quantile_window,
)
from small_world_mobility_edge.preferential_attachment.full_spectrum_scan import (
    MobilityEdgeProductionConfig,
    run_mobility_chunk,
    run_structural_replay_chunk,
    validate_mobility_chunk,
    validate_structural_replay_chunk,
)


class PreferentialAttachmentMobilityEdgePostprocessTests(unittest.TestCase):
    @staticmethod
    def _legacy_affine_graph(n: int, m: int, seed: int, alpha: float) -> nx.Graph:
        graph = nx.complete_graph(m + 1)
        rng = np.random.default_rng(seed)
        delta = m * (alpha - 2.0)
        for source in range(m + 1, n):
            existing = np.arange(source, dtype=int)
            degrees = np.fromiter(
                (graph.degree(int(node)) for node in existing), dtype=float
            )
            targets = rng.choice(
                existing,
                size=m,
                replace=False,
                p=(degrees + delta) / np.sum(degrees + delta),
            )
            graph.add_node(source)
            graph.add_edges_from((source, int(target)) for target in targets)
        return graph

    def test_incremental_degree_generator_is_edge_exact_to_legacy(self) -> None:
        for alpha in (2.0, 1.8, 1.5, 1.1, 1.05):
            expected = self._legacy_affine_graph(180, 5, 8841, alpha)
            observed, _ = generate_preferential_attachment_graph(
                n=180,
                m=5,
                seed=8841,
                model=MODEL_AFFINE_PA,
                degree_ccdf_alpha=alpha,
            )
            self.assertEqual(set(expected.edges()), set(observed.edges()))

    def test_gap_first_locations_and_each_gap_assigned_once(self) -> None:
        values = np.asarray([-3.0, -1.0, 2.0, 7.0])
        record = build_gap_records(
            values,
            graph_index=4,
            graph_seed=9,
            excluded_atom_energies=(),
        )
        np.testing.assert_allclose(record.gap_value, [2.0, 3.0, 5.0])
        np.testing.assert_allclose(record.gap_q, [0.125, 0.375, 0.625])
        np.testing.assert_allclose(record.gap_energy, [-2.0, 0.5, 4.5])
        np.testing.assert_allclose(record.ratio_q, [0.25, 0.5])
        np.testing.assert_allclose(record.ratio_energy, [-1.0, 2.0])
        assigned = 0
        for index in range(4):
            low, high = index / 4, (index + 1) / 4
            assigned += int(np.sum((record.gap_q >= low) & (record.gap_q < high)))
        self.assertEqual(assigned, len(record.gap_value))

    def test_records_never_mix_graphs(self) -> None:
        first = build_gap_records(
            np.asarray([0.0, 1.0, 3.0]),
            graph_index=0,
            graph_seed=10,
            excluded_atom_energies=(),
        )
        second = build_gap_records(
            np.asarray([100.0, 104.0, 109.0]),
            graph_index=1,
            graph_seed=11,
            excluded_atom_energies=(),
        )
        self.assertEqual(first.gap_value.tolist(), [1.0, 2.0])
        self.assertEqual(second.gap_value.tolist(), [4.0, 5.0])
        self.assertNotIn(97.0, np.concatenate((first.gap_value, second.gap_value)))

    def test_raw_ratio_is_affine_rescaling_invariant(self) -> None:
        values = np.asarray([-4.0, -2.0, 1.0, 5.0, 11.0])
        original = build_gap_records(
            values,
            graph_index=0,
            graph_seed=1,
            excluded_atom_energies=(),
        )
        transformed = build_gap_records(
            7.25 * values - 19.0,
            graph_index=0,
            graph_seed=1,
            excluded_atom_energies=(),
        )
        np.testing.assert_allclose(original.ratio_value, transformed.ratio_value)

    def test_zero_positive_and_zero_zero_ratio_accounting(self) -> None:
        record = build_gap_records(
            np.asarray([1.0, 1.0, 1.0, 2.0, 4.0, 4.0]),
            graph_index=0,
            graph_seed=1,
            excluded_atom_energies=(),
        )
        self.assertEqual(
            record.ratio_category.tolist(),
            ["zero_zero", "one_zero", "positive_positive", "one_zero"],
        )
        self.assertTrue(math.isnan(record.ratio_value[0]))
        np.testing.assert_allclose(record.ratio_value[1:], [0.0, 0.5, 0.0])
        row, _ = summarize_quantile_window(
            [record],
            q_low=0.0,
            q_high=1.0,
            minimum_positive_gaps=1,
        )
        self.assertEqual(row["number_of_positive_positive_ratios"], 1)
        self.assertEqual(row["number_of_one_zero_ratios"], 2)
        self.assertEqual(row["number_of_zero_zero_ratios_omitted"], 1)
        self.assertAlmostEqual(row["graph_equal_mean_raw_r_inclusive"], 1.0 / 6.0)
        self.assertAlmostEqual(row["graph_equal_mean_raw_r_positive_conditioned"], 0.5)

    def test_excluded_atom_does_not_create_bridge_gap_or_ratio(self) -> None:
        record = build_gap_records(
            np.asarray([-2.0, -1.0, 0.0, 0.0, 1.0, 3.0]),
            graph_index=0,
            graph_seed=1,
            excluded_atom_energies=(0.0,),
        )
        np.testing.assert_allclose(record.continuous_eigenvalues, [-2.0, -1.0, 1.0, 3.0])
        np.testing.assert_allclose(record.gap_value, [1.0, 2.0])
        np.testing.assert_allclose(record.gap_q, [0.125, 0.625])
        self.assertEqual(len(record.ratio_value), 0)
        self.assertEqual(record.omitted_atom_boundary_gap_count, 1)
        self.assertEqual(record.omitted_atom_boundary_ratio_count, 2)

    def test_minimum_gap_gate_and_adaptive_bounds(self) -> None:
        records = [
            build_gap_records(
                np.linspace(-2.0, 2.0, 21),
                graph_index=index,
                graph_seed=index,
                excluded_atom_energies=(),
            )
            for index in range(5)
        ]
        fixed, _ = summarize_quantile_window(
            records,
            q_low=0.49,
            q_high=0.51,
            minimum_positive_gaps=20,
        )
        self.assertEqual(fixed["histogram_sample_status"], "INSUFFICIENT_POSITIVE_GAPS")
        adapted = adaptive_quantile_window(
            records,
            nominal_q_low=0.49,
            nominal_q_high=0.51,
            expansion_step=0.05,
            minimum_positive_gaps=20,
        )
        self.assertEqual(adapted.status, "ADAPTED")
        self.assertLess(adapted.actual_q_low, adapted.nominal_q_low)
        self.assertGreater(adapted.actual_q_high, adapted.nominal_q_high)
        self.assertGreaterEqual(adapted.positive_gap_count, 20)

    def test_crossfit_train_test_separation_and_cdf_normalization(self) -> None:
        records = [
            build_gap_records(
                np.linspace(-3.0, 3.0, 31) + 0.01 * index,
                graph_index=index,
                graph_seed=100 + index,
                excluded_atom_energies=(),
            )
            for index in range(10)
        ]
        models = build_cross_fitted_models(records, num_folds=5, num_points=101)
        self.assertEqual(len(models), 5)
        for model in models:
            self.assertFalse(set(model.training_graph_ids) & set(model.held_out_graph_ids))
            self.assertAlmostEqual(float(model.density_model.cdf[0]), 0.0)
            self.assertAlmostEqual(float(model.density_model.cdf[-1]), 1.0)
            self.assertTrue(np.all(np.diff(model.density_model.cdf) >= -1e-12))
            for graph_id in model.held_out_graph_ids:
                unfolded = cross_fitted_unfolded_gaps(records[graph_id], model)
                self.assertEqual(len(unfolded), len(records[graph_id].gap_value))

    def test_graph_level_rows_preserve_pairing_identifiers(self) -> None:
        records = [
            build_gap_records(
                np.asarray([-3.0, -1.0, 0.5, 2.0, 5.0]) + index,
                graph_index=index,
                graph_seed=100 + index,
                excluded_atom_energies=(),
            )
            for index in range(2)
        ]
        rows = graph_level_quantile_rows(records, width=0.5)
        self.assertEqual(len(rows), 4)
        self.assertEqual({row["graph_index"] for row in rows}, {0, 1})
        self.assertEqual({row["graph_seed"] for row in rows}, {100, 101})
        self.assertEqual(sum(int(row["total_gap_count"]) for row in rows), 8)

    def test_nonlinear_is_deterministic_and_zeta_one_matches_affine(self) -> None:
        first = generate_nonlinear_preferential_attachment_graph(
            n=100, m=4, seed=72, attachment_zeta=1.2
        )
        second = generate_nonlinear_preferential_attachment_graph(
            n=100, m=4, seed=72, attachment_zeta=1.2
        )
        self.assertEqual(set(first.edges()), set(second.edges()))
        self.assertTrue(all(first.degree(node) >= 4 for node in range(5, 100)))
        nonlinear_one = generate_nonlinear_preferential_attachment_graph(
            n=100, m=4, seed=73, attachment_zeta=1.0
        )
        affine_zero, _ = generate_preferential_attachment_graph(
            n=100,
            m=4,
            seed=73,
            model=MODEL_AFFINE_PA,
            degree_ccdf_alpha=2.0,
        )
        self.assertEqual(set(nonlinear_one.edges()), set(affine_zero.edges()))

    def test_nonlinear_zeta_zero_is_uniform_attachment(self) -> None:
        n, m, seed = 80, 3, 918
        observed = generate_nonlinear_preferential_attachment_graph(
            n=n, m=m, seed=seed, attachment_zeta=0.0
        )
        expected = nx.complete_graph(m + 1)
        rng = np.random.default_rng(seed)
        for source in range(m + 1, n):
            existing = np.arange(source, dtype=int)
            targets = rng.choice(
                existing,
                size=m,
                replace=False,
                p=np.full(source, 1.0 / source),
            )
            expected.add_node(source)
            expected.add_edges_from((source, int(target)) for target in targets)
        self.assertEqual(set(observed.edges()), set(expected.edges()))
        self.assertTrue(nx.is_connected(observed))
        self.assertEqual(min(dict(observed.degree()).values()), m)

    def test_sublinear_config_uses_separate_regime_directory(self) -> None:
        config = MobilityEdgeProductionConfig(
            n=40,
            m=2,
            num_graphs=5,
            graph_seed_base=1200,
            output_root=Path("outputs/preferential_attachment/mobility_edge"),
            model_family="nonlinear",
            attachment_zeta=0.5,
            quantile_width=0.1,
        )
        self.assertIn("nonlinear/sublinear/n_40_m_2", str(config.parameter_dir))
        self.assertEqual(config.metadata()["attachment_regime"], "sublinear")
        with self.assertRaisesRegex(ValueError, "attachment_zeta >= 0"):
            MobilityEdgeProductionConfig(
                n=40,
                m=2,
                num_graphs=5,
                graph_seed_base=1200,
                output_root=Path("outputs"),
                model_family="nonlinear",
                attachment_zeta=-0.1,
                quantile_width=0.1,
            )

    def test_canonical_parameter_names_reach_diagnostics(self) -> None:
        _, affine = generate_preferential_attachment_graph(
            n=80,
            m=3,
            seed=17,
            model=MODEL_AFFINE_PA,
            degree_ccdf_alpha=1.5,
        )
        self.assertEqual(affine.to_dict()["degree_ccdf_alpha"], 1.5)
        _, nonlinear = generate_preferential_attachment_graph(
            n=80,
            m=3,
            seed=17,
            model=MODEL_NONLINEAR_PA,
            attachment_zeta=1.2,
        )
        self.assertEqual(nonlinear.to_dict()["attachment_zeta"], 1.2)
        self.assertTrue(math.isnan(nonlinear.to_dict()["degree_ccdf_alpha"]))

    def test_future_cache_is_complete_and_resume_check_is_content_aware(self) -> None:
        root = Path(tempfile.mkdtemp())
        config = MobilityEdgeProductionConfig(
            n=30,
            m=2,
            num_graphs=2,
            graph_seed_base=901,
            output_root=root,
            model_family="affine_linear",
            degree_ccdf_alpha=2.0,
            quantile_width=0.1,
            localization_block_size=8,
        )
        run_mobility_chunk(config, 0, 2)
        valid, reason = validate_mobility_chunk(config, 0, 2)
        self.assertTrue(valid, reason)
        cache_path = next(config.partial_dir.glob("mobility_cache_*.npz"))
        with np.load(cache_path, allow_pickle=False) as data:
            self.assertEqual(data["nonperron_eigenvalues"].shape, (2, 29))
            self.assertEqual(data["degree_arrays"].shape, (2, 30))
            self.assertIn("eigenpair_top_200_mass", data.files)
            self.assertIn("eigenpair_r_95_fraction", data.files)
            self.assertEqual(int(data["eigenpair_offsets"][-1]), len(data["eigenpair_eigenvalue"]))
        changed = MobilityEdgeProductionConfig(
            **{**config.__dict__, "degree_ccdf_alpha": 1.5}
        )
        changed_valid, _ = validate_mobility_chunk(changed, 0, 2)
        self.assertFalse(changed_valid)
        run_structural_replay_chunk(config, 0, 2)
        structural_valid, structural_reason = validate_structural_replay_chunk(
            config, 0, 2
        )
        self.assertTrue(structural_valid, structural_reason)

    def test_rewired_future_cache_records_exact_degree_control(self) -> None:
        root = Path(tempfile.mkdtemp())
        config = MobilityEdgeProductionConfig(
            n=30,
            m=2,
            num_graphs=2,
            graph_seed_base=1301,
            output_root=root,
            model_family="affine_linear",
            degree_ccdf_alpha=2.0,
            graph_variant="rewired",
            rewire_seed_base=2301,
            quantile_width=0.1,
            localization_block_size=8,
        )
        run_mobility_chunk(config, 0, 2)
        valid, reason = validate_mobility_chunk(config, 0, 2)
        self.assertTrue(valid, reason)
        self.assertEqual(config.parameter_dir.name, "rewired")


if __name__ == "__main__":
    unittest.main()
