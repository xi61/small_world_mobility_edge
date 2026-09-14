from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

try:
    from .regular_baseline import (
        generate_random_regular_graph,
        adjacency_eigendecomposition,
        compare_unfolding_methods,
        wigner_surmise_goe,
    )
except ImportError:  # pragma: no cover - script-style fallback
    from regular_baseline import (
        generate_random_regular_graph,
        adjacency_eigendecomposition,
        compare_unfolding_methods,
        wigner_surmise_goe,
    )


def main() -> None:
    n = 2000
    degree = 3
    seed = 7

    graph = generate_random_regular_graph(n=n, degree=degree, seed=seed)
    eigenvalues, _ = adjacency_eigendecomposition(graph)

    comparison = compare_unfolding_methods(eigenvalues, degree=degree)
    exact_spacings = comparison.exact_spacings
    local_spacings = comparison.local_spacings
    spacing_midpoints = comparison.spacing_midpoints
    abs_diff = comparison.absolute_difference
    rel_diff = comparison.relative_difference

    print(f"n={n}, degree={degree}")
    print(f"mean exact unfolded spacing = {exact_spacings.mean():.6f}")
    print(f"mean spacing rescaled with local density = {local_spacings.mean():.6f}")
    print(f"mean abs difference                    = {abs_diff.mean():.6e}")
    print(f"median rel difference                  = {np.median(rel_diff):.6e}")
    print(f"corr(exact, local-density)             = {np.corrcoef(exact_spacings, local_spacings)[0, 1]:.6f}")
    ks_result = ks_2samp(exact_spacings, local_spacings)
    print(f"KS statistic                           = {ks_result.statistic:.6f}")
    print(f"KS p-value                             = {ks_result.pvalue:.6f}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    s_grid = np.linspace(0.0, 3.5, 400)

    ax = axes[0, 0]
    ax.hist(
        exact_spacings,
        bins=30,
        density=True,
        alpha=0.55,
        color="mediumseagreen",
        edgecolor="black",
        label="counting-function",
    )
    ax.hist(
        local_spacings,
        bins=30,
        density=True,
        alpha=0.40,
        color="goldenrod",
        edgecolor="black",
        label="rescaled with local density",
    )
    ax.plot(s_grid, wigner_surmise_goe(s_grid), color="black", lw=2, label="GOE Wigner surmise")
    ax.set_title("Unfolded Spacing Histograms")
    ax.set_xlabel("s")
    ax.set_ylabel("density")
    ax.legend()

    ax = axes[0, 1]
    lo = min(exact_spacings.min(), local_spacings.min())
    hi = max(exact_spacings.max(), local_spacings.max())
    ax.scatter(exact_spacings, local_spacings, s=8, alpha=0.35)
    ax.plot([lo, hi], [lo, hi], color="black", ls="--")
    ax.set_title("Counting-Function vs Local-Density")
    ax.set_xlabel("exact unfolded spacing")
    ax.set_ylabel("spacing rescaled with local density")

    ax = axes[1, 0]
    ax.scatter(spacing_midpoints, abs_diff, s=8, alpha=0.35)
    ax.set_title("Absolute Difference by Eigenvalue")
    ax.set_xlabel("gap midpoint eigenvalue")
    ax.set_ylabel("|s_exact - s_local|")

    ax = axes[1, 1]
    ax.scatter(spacing_midpoints, rel_diff, s=8, alpha=0.35)
    ax.set_title("Relative Difference by Eigenvalue")
    ax.set_xlabel("gap midpoint eigenvalue")
    ax.set_ylabel("relative difference")
    ax.set_yscale("log")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
