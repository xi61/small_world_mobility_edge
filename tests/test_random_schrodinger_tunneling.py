import math
import unittest

import networkx as nx
import numpy as np
from scipy import sparse

from small_world_mobility_edge.random_schrodinger.tunneling import (
    effective_pair_from_centers,
    paper_alpha_star,
    projected_effective_pair,
    schur_effective_operator,
    symmetric_orthogonalization,
    two_hub_graph,
)
from small_world_mobility_edge.random_schrodinger.common import graph_to_csr


class RandomSchrodingerTunnelingTests(unittest.TestCase):
    def test_exact_two_state_splitting(self):
        matrix = sparse.csr_matrix([[0.2, 0.13], [0.13, 0.7]])
        expected = math.sqrt((0.2 - 0.7) ** 2 + 4 * 0.13**2)
        result = projected_effective_pair(
            matrix, np.array([1.0, 0.0]), np.array([0.0, 1.0]), expected
        )
        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["uncoupled_detuning"], 0.5, places=13)
        self.assertAlmostEqual(result["absolute_direct_coupling"], 0.13, places=13)
        self.assertAlmostEqual(result["predicted_two_state_splitting"], expected, places=13)

    def test_symmetric_orthogonalization(self):
        states = np.array([[1.0, 0.2], [0.0, 1.0], [0.0, 0.1]])
        result = symmetric_orthogonalization(states)
        self.assertTrue(result["valid"])
        orthonormal = result["states"]
        self.assertTrue(np.allclose(orthonormal.T @ orthonormal, np.eye(2), atol=1e-12))

    def test_schur_equals_full_operator_when_p_is_full_space(self):
        matrix = sparse.csr_matrix([[0.1, 0.2], [0.2, 0.5]])
        result = schur_effective_operator(matrix, np.eye(2), energy=0.3, eta=1e-3)
        self.assertTrue(np.allclose(result["effective_matrix"], matrix.toarray(), atol=1e-11))

    def test_disjoint_balls_can_have_zero_direct_coupling(self):
        graph, left, right = two_hub_graph(6, 6, separation=4)
        operator = graph_to_csr(graph, nodelist=range(graph.number_of_nodes()))
        eigenvalues = np.linalg.eigvalsh(operator.toarray())
        summary, schur = effective_pair_from_centers(
            graph,
            operator,
            (left, right),
            radius=1,
            observed_gap=float(eigenvalues[-1] - eigenvalues[-2]),
            eta_values=(1e-2,),
        )
        self.assertAlmostEqual(summary["absolute_direct_coupling"], 0.0, places=13)
        self.assertEqual(len(schur), 1)
        self.assertGreaterEqual(schur[0]["absolute_mediated_coupling"], 0.0)

    def test_paper_alpha_star_is_on_degree_grid_or_kappa_floor(self):
        result = paper_alpha_star(1000, 0.6 * math.log(1000), 0.02, 0.1)
        raw = result["raw_binomial_quantile_alpha"]
        self.assertAlmostEqual(raw * (0.6 * math.log(1000)), result["threshold_degree"])
        self.assertGreaterEqual(result["alpha_star"], 2.1)


if __name__ == "__main__":
    unittest.main()
