from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from regular_baseline import (
    adjacency_eigendecomposition,
    generate_random_regular_graph,
    local_window_spacings,
    remove_trivial_regular_eigenvalue,
)
from regular_baseline_experiments import local_spacing_scan_plots
from regular_baseline_experiments import plot_density_with_energy_windows


N = 10000
DEGREE = 3
SEED = 7
E_VALUES = [-2.5, -2.2, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.2, 2.5]
WINDOW_HALF_WIDTH = 0.30
NUM_BINS = 30
MIN_EIGENVALUES = 12
REWEIGHT_WITHIN_WINDOW = True
NORMALIZE_MEAN = False
ALLOW_EDGE_ENERGIES = False
COMPUTE_KS_DISTANCES = True
KS_SAMPLE_SIZE = 4096
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "local_spacing_by_energy"


def run_validation_checks(eigenvalues: np.ndarray, degree: int) -> None:
    trimmed = remove_trivial_regular_eigenvalue(eigenvalues, degree=degree)
    assert len(trimmed) == len(eigenvalues) - 1, "expected to remove the trivial lambda=degree mode"
    assert not np.any(np.isclose(trimmed, degree, atol=1e-10, rtol=0.0)), "trivial eigenvalue was not excluded"

    check_E = 0.0
    narrow_width = min(WINDOW_HALF_WIDTH, 0.15)
    if narrow_width <= 0:
        narrow_width = 0.15

    fixed = local_window_spacings(
        eigenvalues=eigenvalues,
        E=check_E,
        window_half_width=narrow_width,
        degree=degree,
        reweight_within_window=False,
        normalize_mean=False,
        min_eigenvalues=6,
    )
    midpoint = local_window_spacings(
        eigenvalues=eigenvalues,
        E=check_E,
        window_half_width=narrow_width,
        degree=degree,
        reweight_within_window=True,
        normalize_mean=False,
        min_eigenvalues=6,
    )

    assert np.all(fixed.selected_eigenvalues >= fixed.window_min)
    assert np.all(fixed.selected_eigenvalues <= fixed.window_max)
    assert fixed.num_gaps == fixed.num_eigenvalues - 1
    assert np.all(np.isfinite(fixed.spacings))
    assert np.all(np.isfinite(midpoint.spacings))

    scale = max(fixed.mean_spacing, 1e-12)
    mean_difference = float(np.mean(np.abs(fixed.spacings - midpoint.spacings)))
    assert mean_difference / scale < 0.10, "fixed and midpoint density methods diverge in a narrow bulk window"
    assert abs(fixed.mean_spacing - 1.0) < 0.35, "bulk mean spacing is not close to the expected unit scale"


def print_results_table(results: list[dict[str, object]]) -> None:
    print(f"{'E':>6}  {'window':<22}  {'num eigs':>8}  {'num gaps':>8}  {'rho(E)':>8}  {'mean s':>8}  {'mean r':>8}  status")
    for item in results:
        window = f"[{item['window_min']:.3f}, {item['window_max']:.3f}]"
        mean_spacing = "nan" if np.isnan(item["mean_spacing"]) else f"{item['mean_spacing']:.4f}"
        mean_r = "nan" if np.isnan(item["mean_r"]) else f"{item['mean_r']:.4f}"
        print(
            f"{item['E']:>6.2f}  {window:<22}  {int(item['num_eigenvalues']):>8d}  "
            f"{int(item['num_gaps']):>8d}  {item['rho_at_E']:>8.5f}  {mean_spacing:>8}  "
            f"{mean_r:>8}  {item['status']}"
        )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    graph = generate_random_regular_graph(n=N, degree=DEGREE, seed=SEED)
    eigenvalues, _ = adjacency_eigendecomposition(graph)

    run_validation_checks(eigenvalues=eigenvalues, degree=DEGREE)

    fig_grid, fig_summary, results = local_spacing_scan_plots(
        eigenvalues=eigenvalues,
        degree=DEGREE,
        E_values=E_VALUES,
        window_half_width=WINDOW_HALF_WIDTH,
        num_bins=NUM_BINS,
        min_eigenvalues=MIN_EIGENVALUES,
        reweight_within_window=REWEIGHT_WITHIN_WINDOW,
        normalize_mean=NORMALIZE_MEAN,
        allow_edge_energies=ALLOW_EDGE_ENERGIES,
        compute_ks_distances=COMPUTE_KS_DISTANCES,
        ks_sample_size=KS_SAMPLE_SIZE,
        rng_seed=SEED,
    )
    fig_density = plot_density_with_energy_windows(
        degree=DEGREE,
        E_values=E_VALUES,
        window_half_width=WINDOW_HALF_WIDTH,
    )

    mode_label = "midpoint_reweight" if REWEIGHT_WITHIN_WINDOW else "fixed_density"
    grid_path = OUTPUT_DIR / f"local_spacing_grid_{mode_label}.png"
    summary_path = OUTPUT_DIR / f"local_spacing_summary_{mode_label}.png"
    density_path = OUTPUT_DIR / f"local_density_windows_{mode_label}.png"
    fig_grid.savefig(grid_path, dpi=200, bbox_inches="tight")
    fig_summary.savefig(summary_path, dpi=200, bbox_inches="tight")
    fig_density.savefig(density_path, dpi=200, bbox_inches="tight")
    plt.close(fig_grid)
    plt.close(fig_summary)
    plt.close(fig_density)

    print(f"Saved histogram grid to {grid_path}")
    print(f"Saved summary figure to {summary_path}")
    print(f"Saved density-window figure to {density_path}")
    print_results_table(results)


if __name__ == "__main__":
    main()
