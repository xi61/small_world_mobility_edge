from __future__ import annotations

import csv
import json
import os
import tempfile
import subprocess
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "small-world-mobility-edge-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

try:
    from .chung_lu_baseline import ChungLuSpectrum, generate_power_law_chung_lu_ensemble_from_seeds
    from .empirical_spectral_density import EmpiricalSpectralDensity, estimate_empirical_spectral_density
except ImportError:  # pragma: no cover
    from chung_lu_baseline import ChungLuSpectrum, generate_power_law_chung_lu_ensemble_from_seeds
    from empirical_spectral_density import EmpiricalSpectralDensity, estimate_empirical_spectral_density


N = 2000
TAU_VALUES = [5.0, 10.0, 20.0]
TARGET_MEAN_DEGREES = [10.0, 20.0]
NUM_GRAPHS = 10
GRAPH_SEED = 515151
SEQUENCE_SEED = 616161
COMPONENT_MODE = "full"
SEQUENCE_MODE = "quantile"
W_MIN = 1.0
W_MAX = None
CUTOFF_MODE = "structural"
CUTOFF_COEFFICIENT = 1.0
DENSITY_METHOD = "local_count"
LOCAL_COUNT_HALF_WIDTH = 0.08
DENSITY_NUM_POINTS = 2001
DENSITY_NUM_BINS = 120
DENSITY_BANDWIDTH_SCALE = 1.0
ZERO_EIGENVALUE_ATOL = 1e-8
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "chung_lu" / "global_density_semicircle"


def code_version() -> str | None:
    repo = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def make_seeds(num_graphs: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(value) for value in rng.integers(0, 2**32 - 1, size=num_graphs, dtype=np.uint32)]


def parameter_seed_offset(tau: float, target_mean_degree: float) -> int:
    return int(100 * tau) + int(target_mean_degree)


def run_slug(tau: float, target_mean_degree: float) -> str:
    tau_slug = str(tau).replace(".", "p")
    degree_slug = str(target_mean_degree).replace(".", "p")
    return f"tau_{tau_slug}_avgdeg_{degree_slug}"


def standard_semicircle_density(x: np.ndarray | float) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    density = np.zeros_like(values, dtype=float)
    inside = np.abs(values) <= 2.0
    density[inside] = np.sqrt(np.maximum(4.0 - values[inside] ** 2, 0.0)) / (2.0 * np.pi)
    return density


def build_ensemble(tau: float, target_mean_degree: float) -> list[ChungLuSpectrum]:
    seed_offset = parameter_seed_offset(tau=tau, target_mean_degree=target_mean_degree)
    graph_seeds = make_seeds(NUM_GRAPHS, GRAPH_SEED + seed_offset)
    sequence_seeds = make_seeds(NUM_GRAPHS, SEQUENCE_SEED + seed_offset)
    return generate_power_law_chung_lu_ensemble_from_seeds(
        n=N,
        tau=tau,
        target_mean_degree=target_mean_degree,
        w_min=W_MIN,
        w_max=W_MAX,
        cutoff_mode=CUTOFF_MODE,
        cutoff_coefficient=CUTOFF_COEFFICIENT,
        sequence_mode=SEQUENCE_MODE,
        graph_seeds=graph_seeds,
        sequence_seeds=sequence_seeds,
        component_mode=COMPONENT_MODE,
    )


def normalized_bulk_samples(
    spectra: list[ChungLuSpectrum],
    target_mean_degree: float,
) -> list[np.ndarray]:
    scale = np.sqrt(target_mean_degree)
    if scale <= 0:
        raise ValueError("target_mean_degree must be positive")
    return [np.asarray(spectrum.bulk_eigenvalues, dtype=float) / scale for spectrum in spectra]


def estimate_normalized_density(samples: list[np.ndarray]) -> EmpiricalSpectralDensity:
    return estimate_empirical_spectral_density(
        eigenvalue_samples=samples,
        num_graphs=len(samples),
        per_graph_dimension=float(np.mean([len(sample) for sample in samples])),
        method=DENSITY_METHOD,
        num_points=DENSITY_NUM_POINTS,
        num_bins=DENSITY_NUM_BINS,
        bandwidth_scale=DENSITY_BANDWIDTH_SCALE,
        local_count_half_width=LOCAL_COUNT_HALF_WIDTH,
    )


def semicircle_comparison_metrics(density: EmpiricalSpectralDensity) -> dict[str, float]:
    semicircle = standard_semicircle_density(density.grid)
    empirical_mass_outside_support = float(
        integrate.trapezoid(density.density[np.abs(density.grid) > 2.0], density.grid[np.abs(density.grid) > 2.0])
    ) if np.any(np.abs(density.grid) > 2.0) else 0.0
    l1 = float(integrate.trapezoid(np.abs(density.density - semicircle), density.grid))
    l2 = float(np.sqrt(integrate.trapezoid((density.density - semicircle) ** 2, density.grid)))
    return {
        "semicircle_l1_distance": l1,
        "semicircle_l2_distance": l2,
        "empirical_mass_outside_semicircle_support": empirical_mass_outside_support,
    }


def summarize_spectra(
    spectra: list[ChungLuSpectrum],
    normalized_samples: list[np.ndarray],
    density: EmpiricalSpectralDensity,
    tau: float,
    target_mean_degree: float,
) -> dict[str, float | int | str]:
    pooled = np.concatenate(normalized_samples)
    metrics = semicircle_comparison_metrics(density)
    zero_counts = [int(np.sum(np.isclose(spectrum.bulk_eigenvalues, 0.0, atol=ZERO_EIGENVALUE_ATOL, rtol=0.0))) for spectrum in spectra]
    analyzed_dimensions = [spectrum.analyzed_dimension for spectrum in spectra]
    return {
        "tau": tau,
        "target_mean_degree": target_mean_degree,
        "n": N,
        "num_graphs": len(spectra),
        "mean_analyzed_dimension": float(np.mean(analyzed_dimensions)),
        "mean_realized_mean_degree": float(np.mean([spectrum.realized_mean_degree for spectrum in spectra])),
        "mean_num_components": float(np.mean([spectrum.num_components for spectrum in spectra])),
        "mean_isolated_vertices": float(np.mean([spectrum.num_isolated_vertices for spectrum in spectra])),
        "mean_isolated_vertex_fraction": float(np.mean([spectrum.num_isolated_vertices / spectrum.n_requested for spectrum in spectra])),
        "mean_zero_eigenvalue_count": float(np.mean(zero_counts)),
        "mean_zero_eigenvalue_fraction": float(np.mean([zero / max(dim - 1, 1) for zero, dim in zip(zero_counts, analyzed_dimensions)])),
        "mean_max_realized_degree": float(np.mean([spectrum.max_realized_degree for spectrum in spectra])),
        "mean_max_expected_weight": float(np.mean([spectrum.weight_diagnostics.final_max for spectrum in spectra])),
        "mean_fraction_at_upper_cap": float(np.mean([spectrum.weight_diagnostics.fraction_at_upper_cap for spectrum in spectra])),
        "normalized_mean": float(np.mean(pooled)),
        "normalized_variance": float(np.var(pooled)),
        "normalized_min": float(np.min(pooled)),
        "normalized_max": float(np.max(pooled)),
        **metrics,
    }


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "tau", "target_mean_degree", "n", "num_graphs", "mean_analyzed_dimension",
        "mean_realized_mean_degree", "mean_num_components", "mean_isolated_vertices",
        "mean_isolated_vertex_fraction", "mean_zero_eigenvalue_count",
        "mean_zero_eigenvalue_fraction", "mean_max_realized_degree", "mean_max_expected_weight",
        "mean_fraction_at_upper_cap", "normalized_mean", "normalized_variance",
        "normalized_min", "normalized_max", "semicircle_l1_distance",
        "semicircle_l2_distance", "empirical_mass_outside_semicircle_support",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def plot_density_grid(results: dict[tuple[float, float], dict[str, object]]) -> plt.Figure:
    fig, axes = plt.subplots(
        len(TAU_VALUES),
        len(TARGET_MEAN_DEGREES),
        figsize=(4.4 * len(TARGET_MEAN_DEGREES), 3.2 * len(TAU_VALUES)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    x_grid = np.linspace(-3.5, 3.5, 600)
    for row_index, tau in enumerate(TAU_VALUES):
        for col_index, degree in enumerate(TARGET_MEAN_DEGREES):
            ax = axes[row_index, col_index]
            result = results[(tau, degree)]
            density = result["density"]
            ax.plot(density.grid, density.density, color="steelblue", lw=2, label="empirical")
            ax.plot(x_grid, standard_semicircle_density(x_grid), color="black", lw=1.8, ls="--", label="semicircle")
            ax.set_title(
                f"tau={tau:.1f}, mean={degree:.0f}\n"
                f"L1={result['summary']['semicircle_l1_distance']:.3f}, "
                f"zero={result['summary']['mean_zero_eigenvalue_fraction']:.3f}"
            )
            ax.set_xlabel("lambda / sqrt(mean degree)")
            ax.set_ylabel("density")
            ax.set_xlim(-3.5, 3.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Power-Law Chung-Lu Global Density vs Semicircle", y=1.01)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return fig


def plot_summary_heatmaps(rows: list[dict[str, object]]) -> plt.Figure:
    metrics = [
        ("semicircle_l1_distance", "L1 distance to semicircle"),
        ("mean_zero_eigenvalue_fraction", "zero eigenvalue fraction"),
        ("mean_isolated_vertex_fraction", "isolated vertex fraction"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.0 * len(metrics), 4.2), squeeze=False)
    for ax, (field, title) in zip(axes.flat, metrics):
        matrix = np.array(
            [
                [float(next(row[field] for row in rows if row["tau"] == tau and row["target_mean_degree"] == degree)) for degree in TARGET_MEAN_DEGREES]
                for tau in TAU_VALUES
            ],
            dtype=float,
        )
        image = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis")
        ax.set_title(title)
        ax.set_xticks(range(len(TARGET_MEAN_DEGREES)), [f"{degree:.0f}" for degree in TARGET_MEAN_DEGREES])
        ax.set_yticks(range(len(TAU_VALUES)), [f"{tau:.1f}" for tau in TAU_VALUES])
        ax.set_xlabel("target mean degree")
        ax.set_ylabel("tau")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def run_parameter_pair(tau: float, target_mean_degree: float) -> tuple[dict[str, object], dict[str, object]]:
    spectra = build_ensemble(tau=tau, target_mean_degree=target_mean_degree)
    normalized_samples = normalized_bulk_samples(spectra=spectra, target_mean_degree=target_mean_degree)
    density = estimate_normalized_density(normalized_samples)
    summary = summarize_spectra(
        spectra=spectra,
        normalized_samples=normalized_samples,
        density=density,
        tau=tau,
        target_mean_degree=target_mean_degree,
    )
    return {"density": density, "summary": summary}, summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[tuple[float, float], dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    for tau in TAU_VALUES:
        for target_mean_degree in TARGET_MEAN_DEGREES:
            result, row = run_parameter_pair(tau=tau, target_mean_degree=target_mean_degree)
            results[(tau, target_mean_degree)] = result
            rows.append(row)
            print(
                f"tau={tau:.1f}, mean={target_mean_degree:.0f}: "
                f"L1={row['semicircle_l1_distance']:.3f}, "
                f"zero_frac={row['mean_zero_eigenvalue_fraction']:.3f}, "
                f"isolated_frac={row['mean_isolated_vertex_fraction']:.3f}"
            )

    write_csv(rows, OUTPUT_DIR / "semicircle_comparison_summary.csv")
    density_fig = plot_density_grid(results)
    density_fig.savefig(OUTPUT_DIR / "global_density_vs_semicircle_grid.png", dpi=200, bbox_inches="tight")
    plt.close(density_fig)
    heatmap_fig = plot_summary_heatmaps(rows)
    heatmap_fig.savefig(OUTPUT_DIR / "semicircle_diagnostics_heatmaps.png", dpi=200, bbox_inches="tight")
    plt.close(heatmap_fig)
    manifest = {
        "diagnostic": "chung_lu_global_density_semicircle",
        "n": N,
        "tau_values": TAU_VALUES,
        "target_mean_degrees": TARGET_MEAN_DEGREES,
        "num_graphs": NUM_GRAPHS,
        "component_mode": COMPONENT_MODE,
        "sequence_mode": SEQUENCE_MODE,
        "cutoff_mode": CUTOFF_MODE,
        "cutoff_coefficient": CUTOFF_COEFFICIENT,
        "w_min": W_MIN,
        "w_max": W_MAX,
        "density_method": DENSITY_METHOD,
        "local_count_half_width": LOCAL_COUNT_HALF_WIDTH,
        "normalization": "lambda / sqrt(target_mean_degree), Perron eigenvalue excluded",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "code_version": code_version(),
    }
    with (OUTPUT_DIR / "run_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
