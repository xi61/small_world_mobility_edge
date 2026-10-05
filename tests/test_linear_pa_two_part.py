import json
import inspect
import math
import tempfile
import unittest
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import sparse

from small_world_mobility_edge.preferential_attachment.linear_pa_two_part import (
    _cumulative_and_disjoint_windows,
    _spacing_rows,
    alpha_slug,
    analyze_target_pair,
    actual_mode_count_with_degeneracy,
    controlled_validation_rows,
    evenly_spaced_integer_sample,
    greedy_mass_regions,
    grouped_selected_balls,
    mass_local_maximum_centers,
    ordered_residual_orthogonalization,
    overlap_coefficient,
    quantile_rank_windows,
    resolve_case,
    select_part2_targets,
    stable_graph_seed,
    symmetric_pair_basis,
)
from small_world_mobility_edge.preferential_attachment.linear_pa_two_part_reporting import (
    _plot_coverage,
    summarize_scientific_cases,
)


class LinearPATwoPartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.config_path = cls.project_root / "experiments/linear_pa_two_part_20260924/config.json"
        cls.config = json.loads(cls.config_path.read_text())

    def test_resolved_parameter_convention_and_stable_seeds(self):
        case = resolve_case(self.config, n=500, alpha=1.05)
        self.assertEqual(case["m"], 10)
        self.assertAlmostEqual(case["attachment_offset"], -9.5)
        self.assertEqual(case["expected_edge_count"], 10 * 11 // 2 + 10 * (500 - 11))
        self.assertEqual(
            case["graph_seed"],
            stable_graph_seed(self.config["base_seed"], 500, 0, 1.05),
        )
        self.assertNotEqual(
            stable_graph_seed(self.config["base_seed"], 500, 0, 1.05),
            stable_graph_seed(self.config["base_seed"], 10000, 0, 1.05),
        )

    def test_mass_local_maximum_tie_break_and_fallback(self):
        graph = nx.path_graph(3)
        vector = np.asarray([1.0, 1.0, 0.0]) / math.sqrt(2.0)
        centers, rows = mass_local_maximum_centers(graph, vector, 3)
        self.assertIn(0, centers)
        self.assertNotIn(1, centers)
        self.assertTrue(rows[0]["graph_neighbor_local_maximum"])

    def test_greedy_union_counts_overlap_once_and_records_failure(self):
        graph = nx.path_graph(6)
        left = np.asarray([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]) / math.sqrt(2.0)
        right = np.asarray([0.0, 1.0, 1.0, 0.0, 0.0, 0.0]) / math.sqrt(2.0)
        result = greedy_mass_regions(
            graph,
            left,
            right,
            [0, 2],
            radius=1,
            x=0.8,
            y=0.5,
            maximum_centers=8,
        )
        union = result["union_nodes"]
        self.assertEqual(len(union), len(set(union.tolist())))
        self.assertLessEqual(result["union_vertex_fraction"], 0.5)
        failed = greedy_mass_regions(
            graph,
            left,
            right,
            [0, 2],
            radius=2,
            x=0.95,
            y=0.1,
            maximum_centers=1,
        )
        self.assertFalse(failed["coverage_pass"])
        self.assertTrue(failed["rejections"])

    def test_grouped_regions_use_transitive_overlap_components(self):
        selected = [
            {"center": 0, "nodes": np.asarray([0, 1])},
            {"center": 1, "nodes": np.asarray([1, 2])},
            {"center": 2, "nodes": np.asarray([2, 3])},
        ]
        regions, rows = grouped_selected_balls(selected, 0.5)
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].tolist(), [0, 1, 2, 3])
        self.assertEqual(sum(bool(row["merged_by_rule"]) for row in rows), 2)

    def test_overlap_definitions_remain_distinct(self):
        left = np.asarray([math.sqrt(0.6), math.sqrt(0.4), 0.0])
        right = np.asarray([math.sqrt(0.2), math.sqrt(0.4), math.sqrt(0.4)])
        mass_overlap = float(np.sum(np.minimum(left * left, right * right)))
        top_set_overlap = len({0, 1} & {1, 2})
        structural_overlap = overlap_coefficient([0, 1], [1, 2])
        self.assertAlmostEqual(mass_overlap, 0.6)
        self.assertEqual(top_set_overlap, 1)
        self.assertAlmostEqual(structural_overlap, 0.5)

    def test_ordered_orthogonalization_is_nested_and_handles_duplicates(self):
        states = np.column_stack([np.eye(4)[:, 0], np.eye(4)[:, 1], np.eye(4)[:, 0]])
        first = ordered_residual_orthogonalization(states[:, :2], 1e-8)
        enlarged = ordered_residual_orthogonalization(states, 1e-8)
        self.assertEqual(first["states"].shape, (4, 2))
        self.assertEqual(enlarged["states"].shape, (4, 2))
        self.assertEqual(enlarged["discarded_indices"], [2])
        self.assertTrue(np.allclose(first["states"], enlarged["states"]))

    def test_symmetric_pair_basis_preserves_spatial_orientation(self):
        states = np.column_stack([
            np.asarray([1.0, 0.0, 0.1]),
            np.asarray([0.0, 1.0, 0.1]),
        ])
        result = symmetric_pair_basis(states, 1e-10)
        self.assertTrue(result["valid"])
        u = result["states"]
        self.assertLess(np.linalg.norm(u.T @ u - np.eye(2)), 1e-12)
        self.assertGreater(np.sum(u[[0, 2], 0] ** 2), 0.8)
        self.assertGreater(np.sum(u[[1, 2], 1] ** 2), 0.8)

    def test_rank_one_pair_returns_inadequate_status(self):
        duplicate = np.column_stack([np.eye(2)[:, 0], np.eye(2)[:, 0]])
        result = symmetric_pair_basis(duplicate, 1e-8)
        self.assertFalse(result["valid"])
        self.assertEqual(result["status"], "rank_one_or_duplicate_pair")

    def test_local_degenerate_boundary_is_kept(self):
        values = np.asarray([3.0, 2.0, 2.0 + 1e-12, 1.0])
        self.assertEqual(actual_mode_count_with_degeneracy(values, 2, 1e-9), 3)

    def test_zero_over_zero_gap_ratio_is_undefined_not_zero(self):
        _, ratios, _, _ = _spacing_rows(
            np.asarray([-1.0, 0.0, 0.0, 0.0, 1.0]),
            case_prefix={"case_id": "fixture"},
            tolerance_factor=64.0,
        )
        zero_zero = [row for row in ratios if row["left_gap"] == 0.0 and row["right_gap"] == 0.0]
        self.assertTrue(all(math.isnan(row["raw_gap_ratio_all_levels"]) for row in zero_zero))

    def test_n500_clipping_and_target_counts(self):
        windows = _cumulative_and_disjoint_windows(
            500, [10, 25, 50, 100, 200, 300, 400, 500]
        )
        clipped = [
            window
            for family, _, window in windows
            if family == "cumulative_nonperron" and window["requested_count"] == 500
        ][0]
        self.assertEqual(clipped["state_count"], 499)
        self.assertEqual(clipped["last_decreasing_rank"], 500)
        quantiles = quantile_rank_windows(500, self.config["part1"]["quantile_edges"])
        targets, coverage = select_part2_targets(
            500,
            quantiles,
            self.config["part1"]["cumulative_upper_counts"]["500"],
            self.config["part2"],
        )
        primary = [target for target in targets if "exhaustive_first_50_nonperron" in target["selection_roles"]]
        self.assertEqual(len(primary), 49)
        self.assertEqual(primary[0]["pair_start_decreasing_rank"], 2)
        self.assertEqual(primary[-1]["pair_start_decreasing_rank"], 50)
        self.assertTrue(coverage)

    def test_even_sampling_is_deterministic_and_bounded(self):
        first = evenly_spaced_integer_sample(np.arange(10, 101), 3)
        second = evenly_spaced_integer_sample(np.arange(10, 101), 3)
        self.assertEqual(first.tolist(), second.tolist())
        self.assertEqual(first.tolist(), [10, 55, 100])

    def test_disconnected_induced_vs_full_graph_distance(self):
        graph = nx.Graph([(0, 2), (2, 1)])
        induced = graph.subgraph([0, 1])
        self.assertFalse(nx.has_path(induced, 0, 1))
        self.assertEqual(nx.shortest_path_length(graph, 0, 1), 2)

    def test_all_controlled_validation_fixtures_pass(self):
        rows, passed = controlled_validation_rows()
        self.assertTrue(rows)
        self.assertTrue(passed, msg=[row for row in rows if not row["passed"]])

    def test_part2_acceptance_configuration_is_explicitly_propagated(self):
        parameters = inspect.signature(analyze_target_pair).parameters
        self.assertIn("config", parameters)
        self.assertIn("acceptance", parameters)
        self.assertNotIn("acceptance", self.config["part2"])
        self.assertIn("minimum_capture", self.config["acceptance"])

    def test_integer_alpha_case_slug_matches_scheduler_gate(self):
        self.assertEqual(alpha_slug(2.0), "2")

    def test_report_summary_counts_unique_targets_not_sensitivity_rows(self):
        cases = [{"case_id": "fixture", "n": 500, "degree_ccdf_alpha": 2.0}]
        localization = [
            {"case_id": "fixture", "decreasing_algebraic_rank": rank, "n_times_ipr": 4.0,
             "maximum_coordinate_mass": 0.1, "support_size_80": 20}
            for rank in range(1, 501)
        ]
        windows = [{
            "case_id": "fixture", "window_family": "cumulative_nonperron",
            "window_id": "nonperron_top_50", "mean_gap_ratio_filtered": 0.5,
            "valid_ratio_count_multiplicity_and_atom_excluded": 48,
        }]
        targets = [{"case_id": "fixture", "target_id": "a"}, {"case_id": "fixture", "target_id": "b"}]
        coverage = [{
            "case_id": "fixture", "target_id": "a", "construction_family": "mass_informed_greedy",
            "grouping_variant": "overlap_component_grouped", "radius": 2, "x_target_mass": 0.8,
            "y_union_fraction_cap": 0.2, "paired_coverage_and_union_cap_pass": True,
        }]
        dictionary = [
            {"case_id": "fixture", "target_id": "a", "construction_family": "mass_informed_greedy",
             "reference_setting_in_equivalents": True, "minimum_dictionary_capture": capture,
             "full_dictionary_quality_screen_pass": True, "two_mode_quality_screen_pass": False,
             "scientific_interpretation": "multimode_explanation_for_this_a_posteriori_reference_description"}
            for capture in (0.91, 0.96)
        ]
        summary = summarize_scientific_cases(cases, localization, windows, coverage, dictionary, [], targets)[0]
        self.assertEqual(summary["selected_target_count"], 2)
        self.assertEqual(summary["any_dictionary_target_count"], 1)
        self.assertEqual(summary["mass_informed_dictionary_target_count"], 1)
        self.assertEqual(summary["reference_dictionary_target_count"], 1)
        self.assertEqual(summary["full_dictionary_quality_pass_target_count"], 1)
        self.assertEqual(summary["mass_informed_full_dictionary_quality_pass_target_count"], 1)
        self.assertEqual(summary["reference_full_dictionary_quality_pass_target_count"], 1)
        self.assertEqual(summary["multimode_explanation_target_count"], 1)
        self.assertAlmostEqual(summary["median_best_reference_dictionary_capture"], 0.96)

    def test_coverage_plot_uses_hoffman_compatible_tick_api(self):
        rows = []
        for n in (500, 10000):
            rows.append({
                "n": n,
                "degree_ccdf_alpha": 2.0,
                "construction_family": "mass_informed_greedy",
                "grouping_variant": "overlap_component_grouped",
                "radius": 2,
                "x_target_mass": 0.8,
                "y_union_fraction_cap": 0.2,
                "paired_coverage_and_union_cap_pass": True,
            })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "coverage.png"
            _plot_coverage(rows, path)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
