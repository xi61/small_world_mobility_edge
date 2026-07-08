from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.stats import ks_2samp

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
    adjacent_gap_ratios_from_spacings,
    kesten_mckay_density,
    local_window_spacings,
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


def poisson_spacing_density(s: np.ndarray | float) -> np.ndarray:
    """Evaluate the Poisson nearest-neighbor spacing density P(s)=exp(-s)."""
    s = np.asarray(s, dtype=float)
    density = np.zeros_like(s, dtype=float)
    nonnegative = s >= 0
    density[nonnegative] = np.exp(-s[nonnegative])
    return density


def plot_density_with_energy_windows(
    degree: int,
    E_values: list[float] | np.ndarray,
    window_half_width: float,
) -> plt.Figure:
    """Plot the Kesten-McKay density and mark the chosen energy windows."""
    support_radius = 2.0 * np.sqrt(degree - 1.0)
    x_grid = np.linspace(-support_radius, support_radius, 1200)
    density = kesten_mckay_density(x_grid, degree)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(x_grid, density, color="black", lw=2, label="Kesten-McKay density")

    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(E_values)))
    for color, E in zip(colors, np.asarray(E_values, dtype=float)):
        rho_E = float(kesten_mckay_density(E, degree))
        left = E - window_half_width
        right = E + window_half_width
        ax.axvspan(left, right, color=color, alpha=0.15)
        ax.axvline(E, color=color, lw=1.8)
        ax.scatter([E], [rho_E], color=color, s=35, zorder=3)
        ax.text(E, rho_E + 0.008, f"E={E:.2f}\nrho={rho_E:.3f}", color=color, ha="center", va="bottom")

    ax.set_title(f"Kesten-McKay Density with Local Windows, degree={degree}")
    ax.set_xlabel("eigenvalue / energy E")
    ax.set_ylabel(r"$\rho(E)$")
    ax.set_xlim(-support_radius, support_radius)
    ax.legend()
    fig.tight_layout()
    return fig


def local_spacing_scan_plots(
    eigenvalues: np.ndarray,
    degree: int,
    E_values: list[float] | np.ndarray,
    window_half_width: float,
    num_bins: int = 30,
    min_eigenvalues: int = 6,
    reweight_within_window: bool = False,
    normalize_mean: bool = False,
    allow_edge_energies: bool = False,
    compute_ks_distances: bool = True,
    ks_sample_size: int = 4096,
    rng_seed: int = 12345,
) -> tuple[plt.Figure, plt.Figure, list[dict[str, object]]]:
    """Build local spacing histograms and a summary-vs-energy figure."""
    support_radius = 2.0 * np.sqrt(degree - 1.0)
    rng = np.random.default_rng(rng_seed)
    results: list[dict[str, object]] = []
    valid_spacings: list[np.ndarray] = []

    for E in np.asarray(E_values, dtype=float):
        rho_E = float(kesten_mckay_density(E, degree))
        if not allow_edge_energies and abs(E) >= support_radius:
            raise ValueError(
                f"E={E:.6f} lies outside the Kesten-McKay bulk [-{support_radius:.6f}, {support_radius:.6f}]."
            )
        if rho_E <= 0:
            message = f"E={E:.6f} has rho(E)={rho_E:.6e}; skipping this window."
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            results.append(
                {
                    "E": float(E),
                    "status": message,
                    "num_eigenvalues": 0,
                    "num_gaps": 0,
                    "rho_at_E": rho_E,
                    "mean_spacing": np.nan,
                    "mean_r": np.nan,
                    "ks_goe": np.nan,
                    "ks_poisson": np.nan,
                    "window_min": float(E - window_half_width),
                    "window_max": float(E + window_half_width),
                }
            )
            continue
        if abs(E) + window_half_width > support_radius:
            warnings.warn(
                f"Window around E={E:.6f} reaches the spectral edge; local density estimates may be unstable.",
                RuntimeWarning,
                stacklevel=2,
            )

        try:
            result = local_window_spacings(
                eigenvalues=eigenvalues,
                E=float(E),
                window_half_width=window_half_width,
                degree=degree,
                density_method="kesten_mckay",
                reweight_within_window=reweight_within_window,
                normalize_mean=normalize_mean,
                min_eigenvalues=min_eigenvalues,
            )
        except ValueError as exc:
            results.append(
                {
                    "E": float(E),
                    "status": str(exc),
                    "num_eigenvalues": 0,
                    "num_gaps": 0,
                    "rho_at_E": rho_E,
                    "mean_spacing": np.nan,
                    "mean_r": np.nan,
                    "ks_goe": np.nan,
                    "ks_poisson": np.nan,
                    "window_min": float(E - window_half_width),
                    "window_max": float(E + window_half_width),
                }
            )
            continue

        local_gap_ratios = adjacent_gap_ratios_from_spacings(result.spacings)
        mean_r = float(np.mean(local_gap_ratios)) if len(local_gap_ratios) > 0 else np.nan

        ks_goe = np.nan
        ks_poisson = np.nan
        if compute_ks_distances and len(result.spacings) > 0:
            sample_size = max(ks_sample_size, len(result.spacings))
            goe_sample = rng.rayleigh(scale=np.sqrt(2.0 / np.pi), size=sample_size)
            poisson_sample = rng.exponential(scale=1.0, size=sample_size)
            ks_goe = float(ks_2samp(result.spacings, goe_sample).statistic)
            ks_poisson = float(ks_2samp(result.spacings, poisson_sample).statistic)

        results.append(
            {
                "E": result.energy,
                "status": "ok",
                "result": result,
                "num_eigenvalues": result.num_eigenvalues,
                "num_gaps": result.num_gaps,
                "rho_at_E": result.rho_at_E,
                "mean_spacing": result.mean_spacing,
                "mean_r": mean_r,
                "ks_goe": ks_goe,
                "ks_poisson": ks_poisson,
                "window_min": result.window_min,
                "window_max": result.window_max,
            }
        )
        valid_spacings.append(result.spacings)

    if not valid_spacings:
        raise ValueError("no local windows produced enough eigenvalues for a histogram")

    all_spacings = np.concatenate(valid_spacings)
    positive_spacings = all_spacings[np.isfinite(all_spacings) & (all_spacings >= 0)]
    if len(positive_spacings) == 0:
        raise ValueError("local spacing scan produced no finite nonnegative spacings")

    hist_max = max(3.5, float(np.quantile(positive_spacings, 0.99)))
    bin_edges = np.linspace(0.0, hist_max, num_bins + 1)
    s_grid = np.linspace(0.0, hist_max, 400)

    num_panels = len(results)
    num_cols = min(3, num_panels)
    num_rows = int(np.ceil(num_panels / num_cols))
    fig_grid, axes = plt.subplots(num_rows, num_cols, figsize=(5.2 * num_cols, 3.9 * num_rows), squeeze=False)

    for ax, item in zip(axes.flat, results):
        if item["status"] != "ok":
            ax.text(0.5, 0.5, str(item["status"]), ha="center", va="center", wrap=True)
            ax.set_title(f"E={item['E']:.2f}")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        result = item["result"]
        ax.hist(
            result.spacings,
            bins=bin_edges,
            density=True,
            alpha=0.75,
            color="mediumseagreen",
            edgecolor="black",
            label="empirical",
        )
        ax.plot(s_grid, wigner_surmise_goe(s_grid), color="black", lw=2, label="GOE")
        ax.plot(s_grid, poisson_spacing_density(s_grid), color="firebrick", lw=2, ls="--", label="Poisson")
        ax.set_title(
            f"E={result.energy:.2f}, window=[{result.window_min:.2f}, {result.window_max:.2f}]\n"
            f"n_eigs={result.num_eigenvalues}, n_gaps={result.num_gaps}, mean s={result.mean_spacing:.3f}"
        )
        ax.set_xlabel("s")
        ax.set_ylabel("density")
        ax.set_xlim(bin_edges[0], bin_edges[-1])

    for ax in axes.flat[num_panels:]:
        ax.set_axis_off()

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig_grid.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig_grid.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    valid_items = [item for item in results if item["status"] == "ok"]
    E_plot = np.array([float(item["E"]) for item in valid_items], dtype=float)
    mean_r_plot = np.array([float(item["mean_r"]) for item in valid_items], dtype=float)
    ks_goe_plot = np.array([float(item["ks_goe"]) for item in valid_items], dtype=float)
    ks_poisson_plot = np.array([float(item["ks_poisson"]) for item in valid_items], dtype=float)
    counts_plot = np.array([int(item["num_eigenvalues"]) for item in valid_items], dtype=float)

    if compute_ks_distances:
        fig_summary, axes_summary = plt.subplots(1, 3, figsize=(15, 4.4))
    else:
        fig_summary, axes_summary = plt.subplots(1, 2, figsize=(10, 4.4))
        axes_summary = np.asarray(axes_summary, dtype=object)

    ax = axes_summary[0]
    ax.plot(E_plot, mean_r_plot, marker="o", color="steelblue")
    ax.axhline(0.53590, color="black", ls="--", label="GOE mean r")
    ax.axhline(0.38629, color="firebrick", ls="--", label="Poisson mean r")
    ax.set_title("Mean Local Gap Ratio vs E")
    ax.set_xlabel("E")
    ax.set_ylabel("mean local gap ratio")
    ax.legend()

    if compute_ks_distances:
        ax = axes_summary[1]
        ax.plot(E_plot, ks_goe_plot, marker="o", color="black", label="KS to GOE")
        ax.plot(E_plot, ks_poisson_plot, marker="s", color="firebrick", label="KS to Poisson")
        ax.set_title("KS Distance vs E")
        ax.set_xlabel("E")
        ax.set_ylabel("KS statistic")
        ax.legend()
        ax = axes_summary[2]
    else:
        ax = axes_summary[1]

    ax.plot(E_plot, counts_plot, marker="o", color="goldenrod")
    ax.set_title("Eigenvalues per Window")
    ax.set_xlabel("E")
    ax.set_ylabel("count")

    fig_summary.tight_layout()
    return fig_grid, fig_summary, results


if __name__ == "__main__":
    main()
