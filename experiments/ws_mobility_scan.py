from __future__ import annotations

import numpy as np

from small_world_mobility_edge.models import watts_strogatz_graph
from small_world_mobility_edge.spectral import mobility_edge_candidates, scan_spectrum_windows, symmetric_eigpairs


def main() -> None:
    rng = np.random.default_rng(7)
    n = 180
    degree = 6

    for rewire_prob in (0.0, 0.02, 0.05, 0.1, 0.2):
        sample = watts_strogatz_graph(n=n, degree=degree, rewire_prob=rewire_prob, rng=rng)
        evals, evecs = symmetric_eigpairs(sample.adjacency)
        windows = scan_spectrum_windows(evals, evecs, windows=10)
        candidates = mobility_edge_candidates(windows)

        print(f"rewire_prob={rewire_prob:.3f}")
        for window in windows:
            print(
                "  "
                f"E={window.center:+.4f} "
                f"gap={window.mean_gap_ratio:.4f} "
                f"ipr={window.mean_ipr:.4e} "
                f"pr={window.mean_pr:.2f} "
                f"{window.label}"
            )
        print(f"  candidate_edges={candidates}")
        print()


if __name__ == "__main__":
    main()
