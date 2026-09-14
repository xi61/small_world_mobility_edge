from __future__ import annotations

import unittest

import networkx as nx
import numpy as np

from small_world_mobility_edge import chung_lu_e1_projector_core_scan as scan


class ChungLuE1ProjectorCoreScanTests(unittest.TestCase):
    def test_k2_has_basis_invariant_two_vertex_projectors(self) -> None:
        graph = nx.Graph([(0, 1)])
        eigenvalues, modes, nodes = scan.eigendecomposition_projector_data(graph)
        np.testing.assert_allclose(eigenvalues, [-1.0, 1.0])
        self.assertEqual(nodes, [0, 1])
        for energy in scan.ATOM_ENERGIES:
            self.assertEqual(modes[energy]["multiplicity"], 1)
            np.testing.assert_allclose(modes[energy]["projector_diagonal"], [0.5, 0.5])

    def test_projector_diagonal_does_not_change_under_basis_rotation(self) -> None:
        rng = np.random.default_rng(41)
        raw = rng.normal(size=(8, 3))
        basis, _ = np.linalg.qr(raw)
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        original = np.sum(np.square(basis), axis=1)
        rotated = np.sum(np.square(basis @ rotation), axis=1)
        np.testing.assert_allclose(original, rotated, atol=1e-12)

    def test_two_disconnected_k2_projector_support_is_four(self) -> None:
        graph = nx.Graph([(0, 1), (2, 3)])
        eigenvalues, modes, nodes = scan.eigendecomposition_projector_data(graph)
        mode = modes[1.0]
        metrics = scan.projector_metrics(
            diagonal=np.asarray(mode["projector_diagonal"]),
            multiplicity=int(mode["multiplicity"]),
            nodes=nodes,
            graph_for_degrees=graph,
            largest_component_nodes={0, 1},
            basis_ipr=np.asarray(mode["basis_ipr"]),
        )
        self.assertEqual(mode["multiplicity"], 2)
        self.assertAlmostEqual(float(metrics["projector_effective_support"]), 4.0)
        self.assertAlmostEqual(float(metrics["projector_degree_1_mass_fraction"]), 1.0)

    def test_2_core_removes_isolated_edge_and_path(self) -> None:
        graph = nx.Graph([(0, 1), (2, 3), (3, 4)])
        core = nx.k_core(graph, k=2)
        self.assertEqual(core.number_of_nodes(), 0)

    def test_atom_quantile_removes_zero_and_perron(self) -> None:
        values = np.array([-1.0, 0.0, 1.0, 3.0])
        low, high, midpoint = scan.atom_quantile_interval(values, 1.0)
        self.assertAlmostEqual(low, 0.5)
        self.assertAlmostEqual(high, 1.0)
        self.assertAlmostEqual(midpoint, 0.75)


if __name__ == "__main__":
    unittest.main()
