from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

from regular_baseline import (
    generate_random_regular_graph,
    adjacency_eigendecomposition,
    compare_unfolding_methods,
    inverse_participation_ratio,
    adjacent_gap_ratios,
    kesten_mckay_grid,
    wigner_surmise_goe,
    bin_average_by_eigenvalue,
)


def main() -> None:
    n = 2000
    degree = 3
    seed = 7

    graph = generate_random_regular_graph(n=n, degree=degree, seed=seed)
    eigenvalues, eigenvectors = adjacency_eigendecomposition(graph)

    # IPR for each eigenvector
    ipr = inverse_participation_ratio(eigenvectors)

    # Adjacent gap ratios and their spectral locations.
    gap_ratios, gap_ratio_locations = adjacent_gap_ratios(eigenvalues)

    # McKay law on a dense grid for overlay and comparison.
    mckay_x, mckay_density, _ = kesten_mckay_grid(degree=degree)

    # Compare two unfolding implementations:
    # 1. exact counting-function unfolding using McKay's law
    # 2. Cook-style local-density gap normalization
    unfolding_comparison = compare_unfolding_methods(eigenvalues, degree=degree)
    exact_spacings = unfolding_comparison.exact_spacings
    local_spacings = unfolding_comparison.local_spacings
    spacing_midpoints = unfolding_comparison.spacing_midpoints
    abs_diff = unfolding_comparison.absolute_difference
    rel_diff = unfolding_comparison.relative_difference

    # Bin adjacent gap ratios by eigenvalue location
    gap_bin_x, gap_bin_mean, _ = bin_average_by_eigenvalue(
        gap_ratio_locations,
        gap_ratios,
        num_bins=30,
    )

    # Binned IPR by eigenvalue
    ipr_bin_x, ipr_bin_mean, _ = bin_average_by_eigenvalue(
        eigenvalues,
        ipr,
        num_bins=30,
    )

    print(f"n={n}, degree={degree}")
    print(f"mean IPR = {ipr.mean():.6e}")
    print(f"n * mean(IPR) = {n * ipr.mean():.4f}")
    print(f"mean adjacent gap ratio = {gap_ratios.mean():.6f}")
    print(f"mean exact unfolded spacing = {exact_spacings.mean():.6f}")
    print(f"mean local unfolded spacing = {local_spacings.mean():.6f}")
    print(f"mean abs unfolding diff     = {abs_diff.mean():.6e}")
    print(f"median rel unfolding diff  = {np.median(rel_diff):.6e}")
    print(f"corr(exact, local)         = {np.corrcoef(exact_spacings, local_spacings)[0, 1]:.6f}")
    ks_result = ks_2samp(exact_spacings, local_spacings)
    print(f"KS statistic               = {ks_result.statistic:.6f}")
    print(f"KS p-value                 = {ks_result.pvalue:.6f}")

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))

    # 1. Eigenvalue histogram vs McKay law
    ax = axes[0, 0]
    ax.hist(eigenvalues, bins=40, density=True, alpha=0.75, color="tomato", edgecolor="black")
    ax.plot(mckay_x, mckay_density, color="black", lw=2, label="Kesten-McKay")
    ax.set_title(f"Adjacency Spectrum: {degree}-regular, n={n}")
    ax.set_xlabel("eigenvalue")
    ax.set_ylabel("density")
    ax.legend()

    # 2. Eigenvalue vs IPR scatter (not yet used)
    ax = axes[0, 1]
    ax.scatter(eigenvalues, ipr, s=10, alpha=0.6)
    ax.set_title("Eigenvalue vs IPR")
    ax.set_xlabel("eigenvalue")
    ax.set_ylabel("IPR")

    # 3. Binned IPR vs eigenvalue (not yet used)
    ax = axes[0, 2]
    ax.plot(ipr_bin_x, ipr_bin_mean, marker="o")
    ax.set_title("Binned IPR vs Eigenvalue")
    ax.set_xlabel("eigenvalue bin center")
    ax.set_ylabel("mean IPR")

    # 4. Gap-ratio histogram (not yet used)
    ax = axes[1, 0]
    ax.hist(gap_ratios, bins=30, density=True, alpha=0.75, color="steelblue", edgecolor="black")
    ax.axvline(gap_ratios.mean(), color="black", ls="--", label=f"mean={gap_ratios.mean():.3f}")
    ax.set_title("Adjacent Gap Ratio Histogram")
    ax.set_xlabel("r")
    ax.set_ylabel("density")
    ax.legend()

    # 5. Binned gap ratio vs eigenvalue 
    ax = axes[1, 1]
    ax.plot(gap_bin_x, gap_bin_mean, marker="o")
    ax.axhline(0.5307, color="black", ls="--", label="GOE benchmark")
    ax.set_title("Binned Gap Ratio vs Eigenvalue")
    ax.set_xlabel("eigenvalue bin center")
    ax.set_ylabel("mean gap ratio")
    ax.legend()

    # 6. Unfolded spacing histogram vs Wigner surmise
    ax = axes[1, 2]
    s_grid = np.linspace(0.0, 3.5, 400)
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
        label="Cook local-density",
    )
    ax.plot(s_grid, wigner_surmise_goe(s_grid), color="black", lw=2, label="GOE Wigner surmise")
    ax.set_title("Unfolded Spacing Histogram")
    ax.set_xlabel("s")
    ax.set_ylabel("density")
    ax.legend()

    # 7. Exact vs local unfolded spacings
    ax = axes[0, 3]
    lo = min(exact_spacings.min(), local_spacings.min())
    hi = max(exact_spacings.max(), local_spacings.max())
    ax.scatter(exact_spacings, local_spacings, s=8, alpha=0.35)
    ax.plot([lo, hi], [lo, hi], color="black", ls="--")
    ax.set_title("Cook vs Counting-Function")
    ax.set_xlabel("exact unfolded spacing")
    ax.set_ylabel("local-density spacing")

    # 8. Difference across the spectrum
    ax = axes[1, 3]
    ax.scatter(spacing_midpoints, abs_diff, s=8, alpha=0.35)
    ax.set_title("Unfolding Difference by Eigenvalue")
    ax.set_xlabel("gap midpoint eigenvalue")
    ax.set_ylabel("|s_exact - s_local|")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
