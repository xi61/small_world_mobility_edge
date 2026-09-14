from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import json

import numpy as np

from small_world_mobility_edge.preferential_attachment.extreme_edge_quantiles import (
    REQUESTED_INTERVALS,
    _window_mask,
    add_ratio_graph_ids,
    config_from_manifest,
    map_edge_records,
    ModelSpecification,
    plot_adaptive_companion_histograms,
)
from small_world_mobility_edge.preferential_attachment.extreme_edge_augmentation import (
    run_augmentation_chunk,
    validate_augmentation_chunk,
)
from small_world_mobility_edge.preferential_attachment.full_spectrum_scan import (
    MobilityEdgeProductionConfig,
)
from small_world_mobility_edge.preferential_attachment.mobility_edge_postprocess import (
    build_cross_fitted_models,
    build_gap_records,
)


class PreferentialAttachmentExtremeEdgeTests(unittest.TestCase):
    def test_supported_adaptive_companion_writes_histogram_figure(self) -> None:
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "adaptive.png"
            row = {
                "histogram_status": "OK",
                "nominal_q_low": 0.9999,
                "nominal_q_high": 0.99995,
                "actual_q_low": 0.9995,
                "actual_q_high": 0.99995,
                "crossfit_unfolded_histogram_sample_count": 120,
                "number_of_contributing_graphs": 80,
            }
            values = np.linspace(0.01, 2.5, 120)
            plot_adaptive_companion_histograms(
                row,
                values / 100.0,
                values,
                specification=ModelSpecification(
                    "nonlinear", "attachment_zeta", 1.2
                ),
                output_path=output,
            )
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)

    def test_eigenvalue_only_augmentation_is_deterministic_and_validated(self) -> None:
        with TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            specification = ModelSpecification(
                "nonlinear", "attachment_zeta", 1.2
            )
            config = MobilityEdgeProductionConfig(
                n=60,
                m=3,
                num_graphs=2,
                graph_seed_base=711,
                output_root=output_root / "base",
                model_family="nonlinear",
                attachment_zeta=1.2,
            )
            run_augmentation_chunk(
                config,
                specification,
                output_root,
                base_graph_count=2,
                target_graph_count=4,
                start=2,
                stop=4,
            )
            valid, reason = validate_augmentation_chunk(
                config,
                specification,
                output_root,
                base_graph_count=2,
                target_graph_count=4,
                start=2,
                stop=4,
            )
            self.assertTrue(valid, reason)

    def test_authoritative_legacy_manifest_schema_is_accepted(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            specification = ModelSpecification(
                "affine_linear", "degree_ccdf_alpha", 1.5
            )
            directory = (
                root
                / "affine_linear"
                / "n_5000_m_10"
                / "degree_ccdf_alpha_1p5"
            )
            directory.mkdir(parents=True)
            (directory / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "analysis_version": "pa_mobility_edge_gap_first_v2_atom_segmented",
                        "canonical_parameter": {
                            "name": "degree_ccdf_alpha",
                            "value": 1.5,
                        },
                        "n": 5000,
                        "m": 10,
                        "num_graphs": 500,
                        "graph_seed_base": 27352,
                        "atom_filtering": {"atol": 1e-8},
                    }
                )
            )
            config = config_from_manifest(root, specification)
            self.assertEqual(config.quantile_width, 0.002)
            self.assertEqual(config.zero_eigenvalue_atol, 1e-8)

    def test_requested_intervals_are_exact_and_nonoverlapping(self) -> None:
        self.assertEqual(REQUESTED_INTERVALS[0], (0.995, 0.996))
        self.assertEqual(REQUESTED_INTERVALS[-1], (0.99995, 1.0))
        for left, right in zip(REQUESTED_INTERVALS[:-1], REQUESTED_INTERVALS[1:]):
            self.assertEqual(left[1], right[0])
        q = np.asarray([0.995, 0.996, 0.99995, 1.0])
        self.assertEqual(int(np.sum(_window_mask(q, 0.995, 0.996))), 1)
        self.assertEqual(int(np.sum(_window_mask(q, 0.99995, 1.0))), 2)

    def test_crossfit_edge_mapping_preserves_graph_and_atom_boundaries(self) -> None:
        records = []
        for graph_index in range(10):
            values = np.asarray([-4.0, -2.0, -1.0, 0.0, 0.5, 1.5, 3.0, 5.0])
            values = values + 0.01 * graph_index
            values[-2] = 5.0 + 2.0 * graph_index
            values[-1] = 5.1 + 2.0 * graph_index
            if graph_index == 0:
                values[3] = 0.0
            records.append(
                build_gap_records(
                    values,
                    graph_index=graph_index,
                    graph_seed=1000 + graph_index,
                    excluded_atom_energies=(0.0,),
                )
            )
        models = build_cross_fitted_models(records, num_folds=5, num_points=401)
        mapped = map_edge_records(records, models)
        add_ratio_graph_ids(mapped, records, models)
        self.assertGreater(len(mapped["raw_gap"]), 0)
        self.assertEqual(len(mapped["ratio_value"]), len(mapped["ratio_graph_index"]))
        self.assertTrue(np.all(mapped["atom_segment_identifier"] != 0))
        self.assertTrue(np.all(mapped["fold"] == mapped["graph_index"] % 5))
        self.assertTrue(np.all(mapped["ensemble_crossfit_cdf_quantile"] >= 0.995))


if __name__ == "__main__":
    unittest.main()
