import unittest

import networkx as nx
import numpy as np
from scipy import sparse

from small_world_mobility_edge.random_schrodinger.handoff_coupling import (
    apply_dictionary_enlargement_gate,
    controlled_validation_rows,
    fixed_partition_operator,
    merge_structural_regions,
    orthogonalize_with_truncation,
    overlap_coefficient,
    structural_vertices,
)
from small_world_mobility_edge.random_schrodinger.handoff_metrics import localization_metrics


class PACouplingHandoffTests(unittest.TestCase):
    def test_structural_vertices_degree_then_id(self):
        graph = nx.Graph([(0, 2), (1, 2), (2, 3), (3, 4)])
        assert structural_vertices(graph, 3).tolist() == [2, 3, 0]


    def test_overlap_coefficient_and_region_cap(self):
        graph = nx.path_graph(8)
        assert overlap_coefficient([0, 1], [1, 2]) == 0.5
        regions, centers, rows, _ = merge_structural_regions(
            graph, [1, 2], radius=1, overlap_threshold=0.5, maximum_fraction=0.25
        )
        assert regions == [] and centers == []
        assert rows[0]["rejection_reason"] == "merged_region_exceeds_local_fraction_cap"


    def test_gram_truncates_duplicate_directions(self):
        result = orthogonalize_with_truncation(
            np.column_stack([np.array([1.0, 0.0]), np.array([1.0, 0.0])]),
            relative_cutoff=1e-8,
        )
        assert result["retained_rank"] == 1
        assert result["orthonormality_error"] < 1e-12


    def test_fixed_partition_scales_only_interblock_entries(self):
        operator = sparse.csr_matrix(np.array([[0.0, 2.0, 3.0], [2.0, 0.0, 5.0], [3.0, 5.0, 0.0]]))
        result = fixed_partition_operator(operator, np.array([0, 0, 1]), 0.25).toarray()
        assert np.allclose(result, [[0.0, 2.0, 0.75], [2.0, 0.0, 1.25], [0.75, 1.25, 0.0]])


    def test_localization_metrics_distinguish_coordinate_and_mass(self):
        metrics = localization_metrics(np.array([3.0, 4.0]) / 5.0)
        assert metrics["maximum_absolute_coordinate"] == 0.8
        assert np.isclose(metrics["maximum_coordinate_mass"], 0.64)
        assert np.isclose(metrics["ipr"], 337.0 / 625.0)


    def test_enlargement_gate_requires_same_pair_and_stable_predictions(self):
        base = {
            "variant": "radius_1_modes_3", "left_region_id": 0, "right_region_id": 1,
            "left_local_mode_rank": 1, "right_local_mode_rank": 1,
            "left_upper_rank": 3, "right_upper_rank": 2, "predicted_gap": 1.0,
            "predicted_reference_weights": [[0.8, 0.2], [0.2, 0.8]],
            "accepted_before_stability_check": True,
        }
        enlarged = {**base, "variant": "radius_1_modes_5", "predicted_gap": 1.05}
        rows = apply_dictionary_enlargement_gate([base, enlarged], primary_variant="radius_1_modes_3", enlarged_variant="radius_1_modes_5")
        assert rows[0]["accepted_predictive_two_mode"] is True


    def test_all_controlled_fixtures_pass(self):
        rows, passed = controlled_validation_rows()
        assert passed
        assert rows and all(bool(row["passed"]) for row in rows)
