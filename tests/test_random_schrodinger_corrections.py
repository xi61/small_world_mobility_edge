from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import sparse

from small_world_mobility_edge.random_schrodinger.common import (
    analyzed_to_original_ids,
    graph_from_edge_cache,
    save_graph_cache,
)
from small_world_mobility_edge.random_schrodinger.disorder_correction import (
    _duplicate_neighborhood_groups,
    _gap_triplets,
    _projected_diagonal_summary,
    _window_ipr_summary,
)
from small_world_mobility_edge.random_schrodinger.tunneling_correction import (
    _reference_comparison,
    subspace_diagnostics,
)
from small_world_mobility_edge.random_schrodinger.zero_subspace_correction import (
    _projected_potential_spectrum,
    _projector_diagnostics,
)


class RandomSchrodingerCorrectionTests(unittest.TestCase):
    def test_graph_cache_recovers_nonidentity_original_ids(self) -> None:
        graph = nx.path_graph(4)
        original = np.asarray([2, 5, 7, 11], dtype=np.int64)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "graph.npz"
            save_graph_cache(path, graph, original, {"fixture": True})
            loaded, metadata = graph_from_edge_cache(path)
        np.testing.assert_array_equal(analyzed_to_original_ids(loaded), original)
        np.testing.assert_array_equal(analyzed_to_original_ids(loaded, [3, 0]), [11, 2])
        self.assertEqual(metadata["original_vertex_ids_nonidentity_count"], 4)
        self.assertEqual(metadata["vertex_mapping"], "explicit_analyzed_index_to_original_vertex_id")

    def test_subspace_separates_internal_coupling_from_leakage(self) -> None:
        coupling = 0.2
        operator = sparse.csr_matrix([[0.0, coupling], [coupling, 1.0]])
        result = subspace_diagnostics(operator, np.eye(2))
        self.assertTrue(result.valid)
        self.assertLess(float(result.summary["external_leakage_frobenius_norm"]), 1e-14)
        for row in result.state_rows:
            self.assertAlmostEqual(float(row["internal_coupling_norm"]), coupling, places=13)
            self.assertAlmostEqual(
                float(row["individual_approximate_eigenvector_residual_norm"]), coupling, places=13
            )
            self.assertLess(float(row["external_leakage_norm"]), 1e-14)
            self.assertLess(float(row["residual_squared_decomposition_error"]), 1e-14)

    def test_full_isolation_spectrum_includes_unretained_modes(self) -> None:
        operator = sparse.diags([0.0, 0.1, 2.0], format="csr")
        states = np.column_stack([np.eye(3)[:, 0], np.eye(3)[:, 2]])
        result = subspace_diagnostics(operator, states)
        comparison = _reference_comparison(
            result,
            np.asarray([0.0, 2.0]),
            states,
            True,
            isolation_values=np.asarray([0.0, 0.1, 2.0]),
        )
        self.assertEqual(comparison["isolation_spectrum_size"], 3)
        self.assertAlmostEqual(
            float(comparison["nearest_known_omitted_spectral_separation"]), 0.1
        )
        self.assertTrue(bool(comparison["full_spectrum_known"]))

    def test_zero_projector_and_projected_potential_are_basis_invariant(self) -> None:
        operator = sparse.csr_matrix(nx.to_numpy_array(nx.star_graph(3)))
        values, vectors = np.linalg.eigh(operator.toarray())
        mask = np.abs(values) < 1e-12
        summary, diagonal, _ = _projector_diagnostics(operator, values[mask], vectors[:, mask])
        self.assertEqual(summary["dimension"], 2)
        np.testing.assert_allclose(diagonal, [0.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0], atol=1e-12)
        xi = np.asarray([0.7, -0.5, 0.1, 0.9])
        spectrum, _ = _projected_potential_spectrum(vectors[:, mask], xi)
        rotation, _ = np.linalg.qr(np.asarray([[1.0, 1.0], [-1.0, 1.0]]))
        rotated_spectrum, _ = _projected_potential_spectrum(vectors[:, mask] @ rotation, xi)
        np.testing.assert_allclose(spectrum, rotated_spectrum, atol=1e-12)

    def test_atom_exclusion_never_bridges_removed_levels(self) -> None:
        values = np.asarray([-2.0, -1e-14, 0.0, 1e-14, 2.0])
        atom = np.asarray([False, True, True, True, False])
        rows, summary = _gap_triplets(values, atom, 0.0, "all", np.arange(5))
        self.assertEqual(len(rows), 3)
        self.assertEqual(summary["excluded_triplet_count"], 3)
        self.assertEqual(summary["retained_triplet_count"], 0)
        self.assertTrue(all(bool(row["excluded_due_to_atom"]) for row in rows))

    def test_duplicate_neighborhood_modes_and_projected_moments(self) -> None:
        graph = nx.star_graph(3)
        groups = _duplicate_neighborhood_groups(graph)
        self.assertIn([1, 2, 3], groups)
        values = np.asarray([-0.5, 0.1, 0.9])
        result = _projected_diagonal_summary(values)
        projector = np.eye(3) - np.ones((3, 3)) / 3.0
        projected = projector @ np.diag(values) @ projector
        self.assertAlmostEqual(result["projected_trace"], float(np.trace(projected)), places=13)
        self.assertAlmostEqual(
            result["projected_frobenius_squared"], float(np.sum(projected**2)), places=13
        )

    def test_degenerate_atom_ipr_is_retained_but_not_interpreted(self) -> None:
        ipr = np.asarray([0.1, 0.2, 0.3])
        atom = np.asarray([False, True, False])
        result = _window_ipr_summary(ipr, np.arange(3), atom)
        self.assertTrue(np.isnan(float(result["mean_ipr"])))
        self.assertTrue(np.isnan(float(result["median_ipr"])))
        self.assertAlmostEqual(float(result["cached_basis_dependent_mean_ipr"]), 0.2)
        self.assertFalse(bool(result["individual_mode_ipr_interpretable"]))

    def test_nonatom_ipr_remains_interpretable(self) -> None:
        ipr = np.asarray([0.1, 0.2, 0.3])
        atom = np.zeros(3, dtype=bool)
        result = _window_ipr_summary(ipr, np.arange(3), atom)
        self.assertAlmostEqual(float(result["mean_ipr"]), 0.2)
        self.assertTrue(bool(result["individual_mode_ipr_interpretable"]))


if __name__ == "__main__":
    unittest.main()
