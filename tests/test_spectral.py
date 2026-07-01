from __future__ import annotations

import unittest

import numpy as np
from scipy import sparse

from small_world_mobility_edge.spectral import (
    GOE_GAP_RATIO,
    POISSON_GAP_RATIO,
    adjacent_gap_ratios,
    classify_gap_statistics,
    inverse_participation_ratio,
    scan_spectrum_windows,
    symmetric_eigpairs,
)


class SpectralTests(unittest.TestCase):
    def test_gap_ratios_basic(self) -> None:
        ratios = adjacent_gap_ratios(np.array([0.0, 1.0, 3.0, 6.0]))
        self.assertTrue(np.allclose(ratios, np.array([0.5, 2.0 / 3.0])))

    def test_classify_gap_statistics(self) -> None:
        self.assertEqual(classify_gap_statistics(GOE_GAP_RATIO), "goe_like")
        self.assertEqual(classify_gap_statistics(POISSON_GAP_RATIO), "poisson_like")

    def test_inverse_participation_ratio(self) -> None:
        vecs = np.array([[1.0, 1.0 / np.sqrt(2)], [0.0, 1.0 / np.sqrt(2)]])
        ipr = inverse_participation_ratio(vecs)
        self.assertTrue(np.allclose(ipr, np.array([1.0, 0.5])))

    def test_dense_eigensolver_path(self) -> None:
        adj = sparse.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
        evals, _ = symmetric_eigpairs(adj)
        self.assertTrue(np.allclose(evals, np.array([-1.0, 1.0])))

    def test_window_scan_returns_labels(self) -> None:
        adj = sparse.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
        evals, evecs = symmetric_eigpairs(adj)
        windows = scan_spectrum_windows(evals, evecs, windows=1)
        self.assertEqual(windows, [])


if __name__ == "__main__":
    unittest.main()
