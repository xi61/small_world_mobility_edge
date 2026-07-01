from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from regular_baseline import (
    generate_random_regular_graph,
    adjacency_eigendecomposition,
    inverse_participation_ratio,
    adjacent_gap_ratios,
    kesten_mckay_grid,
    unfold_with_kesten_mckay,
    unfolded_spacings,
    wigner_surmise_goe,
    bin_average_by_eigenvalue,
)


def main() -> None:
    n = 2000
    degree = 3
    seed = 7

    graph = generate_random_regular_graph(n=n, degree=degree, seed=seed)
    eigenvalues, eigenvectors = adjacency_eigendecomposition(graph)

    # IPR for each eigenvector.
    ipr = inverse_participation_ratio(eigenvectors)

    # Adjacent gap ratios and their spectral locations.
    gap_ratios, gap_ratio_locations = adjacent_gap_ratios(eigenvalues)

    # McKay law on a dense grid for overlay.
    mckay_x, mckay_density, _ = kesten_mckay_grid(degree=degree)

    # Unfold, then compute unfolded spacing.
    unfolding = unfold_with_kesten_mckay(eigenvalues, degree=degree)
    unfolded = unfolding.unfolded
    spacings_unfolded = unfolded_spacings(unfolded)

    # Bin adjacent gap ratios by eigenvalue location.
    gap_bin_x, gap_bin_mean, _ = bin_average_by_eigenvalue(
        gap_ratio_locations,
        gap_ratios,
        num_bins=30,
    )

    # Bin IPR by eigenvalue.
    ipr_bin_x, ipr_bin_mean, _ = bin_average_by_eigenvalue(
        eigenvalues,
        ipr,
        num_bins=30,
    )

    print(f"n={n}, degree={degree}")
    print(f"mean IPR = {ipr.mean():.6e}")
    print(f"n * mean(IPR) = {n * ipr.mean():.4f}")
    print(f"mean adjacent gap ratio = {gap_ratios.mean():.6f}")
    print(f"mean unfolded spacing = {spacings_unfolded.mean():.6f}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # 1. Eigenvalue histogram vs McKay law.
    ax = axes[0, 0]
    ax.hist(eigenvalues, bins=40, density=True, alpha=0.75, color="tomato", edgecolor="black")
    ax.plot(mckay_x, mckay_density, color="black", lw=2, label="Kesten-McKay")
    ax.set_title(f"Adjacency Spectrum: {degree}-regular, n={n}")
    ax.set_xlabel("eigenvalue")
    ax.set_ylabel("density")
    ax.legend()

    # 2. Eigenvalue vs IPR scatter.
    ax = axes[0, 1]
    ax.scatter(eigenvalues, ipr, s=10, alpha=0.6)
    ax.set_title("Eigenvalue vs IPR")
    ax.set_xlabel("eigenvalue")
    ax.set_ylabel("IPR")

    # 3. Binned IPR vs eigenvalue.
    ax = axes[0, 2]
    ax.plot(ipr_bin_x, ipr_bin_mean, marker="o")
    ax.set_title("Binned IPR vs Eigenvalue")
    ax.set_xlabel("eigenvalue bin center")
    ax.set_ylabel("mean IPR")

    # 4. Gap-ratio histogram.
    ax = axes[1, 0]
    ax.hist(gap_ratios, bins=30, density=True, alpha=0.75, color="steelblue", edgecolor="black")
    ax.axvline(gap_ratios.mean(), color="black", ls="--", label=f"mean={gap_ratios.mean():.3f}")
    ax.set_title("Adjacent Gap Ratio Histogram")
    ax.set_xlabel("r")
    ax.set_ylabel("density")
    ax.legend()

    # 5. Binned gap ratio vs eigenvalue.
    ax = axes[1, 1]
    ax.plot(gap_bin_x, gap_bin_mean, marker="o")
    ax.axhline(0.5307, color="black", ls="--", label="GOE benchmark")
    ax.set_title("Binned Gap Ratio vs Eigenvalue")
    ax.set_xlabel("eigenvalue bin center")
    ax.set_ylabel("mean gap ratio")
    ax.legend()

    # 6. Unfolded spacing histogram vs Wigner surmise.
    ax = axes[1, 2]
    s_grid = np.linspace(0.0, 3.5, 400)
    ax.hist(
        spacings_unfolded,
        bins=30,
        density=True,
        alpha=0.75,
        color="mediumseagreen",
        edgecolor="black",
        label="empirical",
    )
    ax.plot(s_grid, wigner_surmise_goe(s_grid), color="black", lw=2, label="GOE Wigner surmise")
    ax.set_title("Unfolded Spacing Histogram")
    ax.set_xlabel("s")
    ax.set_ylabel("density")
    ax.legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
