from __future__ import annotations

import csv
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy.stats import ks_2samp

try:
    from .chung_lu_baseline import ChungLuSpectrum, generate_power_law_chung_lu_ensemble_from_seeds
    from .empirical_spectral_density import EmpiricalSpectralDensity, estimate_empirical_spectral_density
    from .regular_baseline import adjacent_gap_ratios_from_spacings, wigner_surmise_goe
    from .regular_baseline_experiments import poisson_spacing_density
except ImportError:  # pragma: no cover
    from chung_lu_baseline import ChungLuSpectrum, generate_power_law_chung_lu_ensemble_from_seeds
    from empirical_spectral_density import EmpiricalSpectralDensity, estimate_empirical_spectral_density
    from regular_baseline import adjacent_gap_ratios_from_spacings, wigner_surmise_goe
    from regular_baseline_experiments import poisson_spacing_density


N = 5000
ALL_TAU_VALUES = [3.0, 3.5, 4.0]
ALL_TARGET_MEAN_DEGREES = [6.0, 10.0, 20.0]
ACTIVE_PARAMETER_PAIRS = [(3.0, 6.0), (3.0, 10.0)]
RUN_FULL_GRID = True
NUM_DENSITY_GRAPHS = 30
NUM_SPACING_GRAPHS = 30
FULL_RECOMMENDED_NUM_GRAPHS_maybe= 50
DENSITY_GRAPH_SEED = 31415
SPACING_GRAPH_SEED = 27182
DENSITY_SEQUENCE_SEED = 16180
SPACING_SEQUENCE_SEED = 14142
COMPONENT_MODE = "full"
DENSITY_METHOD = "local_count"
LOCAL_COUNT_HALF_WIDTH = 0.20
DENSITY_NUM_POINTS = 2001
DENSITY_NUM_BINS = 120
DENSITY_BANDWIDTH_SCALE = 1.0
# BULK_QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
# EDGE_QUANTILES = [0.005, 0.01, 0.025, 0.05, 0.95, 0.975, 0.99, 0.995]
LOCAL_QUANTILES = [
    0.005, 0.01, 0.025, 0.05,
    0.10, 0.25, 0.50, 0.75, 0.90,
    0.95, 0.975, 0.99, 0.995,
]
ENABLE_EXTRA_EXTREME_EDGE_QUANTILES = False
EXTRA_EXTREME_EDGE_QUANTILES = [0.0025, 0.9975]
EXPLICIT_E_VALUES: list[float] = []
WINDOW_MODE = "adaptive_quantile_mass"
TARGET_EIGENVALUES_PER_GRAPH_BULK = 25
TARGET_EIGENVALUES_PER_GRAPH_EDGE = 20
LOCAL_TARGET_EIGENVALUES_PER_GRAPH = 25
BULK_UNFOLDING_METHOD = "fixed_density"
EDGE_UNFOLDING_METHOD = "cdf"
LOCAL_UNFOLDING_METHOD = "cdf"
NUM_BINS = 30
MIN_EIGENVALUES = 10
SMALL_SPACING_THRESHOLD = 0.20
KS_SAMPLE_SIZE = 4096
SEQUENCE_MODE = "quantile"
W_MIN = 1.0
W_MAX = None
CUTOFF_MODE = "structural"
CUTOFF_COEFFICIENT = 1.0
FIXED_RANK_EDGE_K_VALUES = [20]
OUTPUT_PROFILE = "local_spacing_only"
SUPPORTED_OUTPUT_PROFILES = {"local_spacing_only", "full_diagnostics"}
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "chung_lu" / "power_law"


def active_parameter_pairs() -> list[tuple[float, float]]:
    if RUN_FULL_GRID:
        return [(tau, target_mean_degree) for tau in ALL_TAU_VALUES for target_mean_degree in ALL_TARGET_MEAN_DEGREES]
    return list(ACTIVE_PARAMETER_PAIRS)


def code_version() -> str | None:
    repo = Path(__file__).resolve().parents[2]
    try:
        return (
            subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True)
            .strip()
        )
    except Exception:
        return None


def make_seeds(num_graphs: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(value) for value in rng.integers(0, 2**32 - 1, size=num_graphs, dtype=np.uint32)]


def parameter_seed_offset(tau: float, target_mean_degree: float) -> int:
    return int(100 * tau) + int(target_mean_degree)


def run_slug(tau: float, target_mean_degree: float) -> str:
    return f"tau_{str(tau).replace('.', 'p')}_avgdeg_{str(target_mean_degree).replace('.', 'p')}"


def parameter_output_dir(tau: float, target_mean_degree: float) -> Path:
    return OUTPUT_DIR / run_slug(tau=tau, target_mean_degree=target_mean_degree)


def clear_output_directory_files(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.iterdir():
        if path.is_file():
            path.unlink()


def build_parameter_pair_ensembles(
    tau: float,
    target_mean_degree: float,
) -> tuple[list[ChungLuSpectrum], list[ChungLuSpectrum]]:
    seed_offset = parameter_seed_offset(tau=tau, target_mean_degree=target_mean_degree)
    density_graph_seeds = make_seeds(NUM_DENSITY_GRAPHS, DENSITY_GRAPH_SEED + seed_offset)
    spacing_graph_seeds = make_seeds(NUM_SPACING_GRAPHS, SPACING_GRAPH_SEED + seed_offset)
    density_sequence_seeds = make_seeds(NUM_DENSITY_GRAPHS, DENSITY_SEQUENCE_SEED + seed_offset)
    spacing_sequence_seeds = make_seeds(NUM_SPACING_GRAPHS, SPACING_SEQUENCE_SEED + seed_offset)

    density_spectra = generate_power_law_chung_lu_ensemble_from_seeds(
        n=N,
        tau=tau,
        target_mean_degree=target_mean_degree,
        w_min=W_MIN,
        w_max=W_MAX,
        cutoff_mode=CUTOFF_MODE,
        cutoff_coefficient=CUTOFF_COEFFICIENT,
        sequence_mode=SEQUENCE_MODE,
        graph_seeds=density_graph_seeds,
        sequence_seeds=density_sequence_seeds,
        component_mode=COMPONENT_MODE,
    )
    spacing_spectra = generate_power_law_chung_lu_ensemble_from_seeds(
        n=N,
        tau=tau,
        target_mean_degree=target_mean_degree,
        w_min=W_MIN,
        w_max=W_MAX,
        cutoff_mode=CUTOFF_MODE,
        cutoff_coefficient=CUTOFF_COEFFICIENT,
        sequence_mode=SEQUENCE_MODE,
        graph_seeds=spacing_graph_seeds,
        sequence_seeds=spacing_sequence_seeds,
        component_mode=COMPONENT_MODE,
    )
    return density_spectra, spacing_spectra


def run_manifest(
    tau: float,
    target_mean_degree: float,
    output_profile: str,
) -> dict[str, object]:
    return {
        "model": "power_law_chung_lu",
        "output_profile": output_profile,
        "n": N,
        "tau": tau,
        "target_mean_degree": target_mean_degree,
        "sequence_mode": SEQUENCE_MODE,
        "cutoff_mode": CUTOFF_MODE,
        "cutoff_coefficient": CUTOFF_COEFFICIENT,
        "w_min": W_MIN,
        "w_max": W_MAX,
        "component_mode": COMPONENT_MODE,
        "density_method": DENSITY_METHOD,
        "density_local_count_half_width": LOCAL_COUNT_HALF_WIDTH,
        "density_num_points": DENSITY_NUM_POINTS,
        "density_num_bins": DENSITY_NUM_BINS,
        "density_bandwidth_scale": DENSITY_BANDWIDTH_SCALE,
        "local_quantiles": LOCAL_QUANTILES,
        "local_target_eigenvalues_per_graph": LOCAL_TARGET_EIGENVALUES_PER_GRAPH,
        "local_unfolding_method": LOCAL_UNFOLDING_METHOD,
        "window_mode": WINDOW_MODE,
        "min_eigenvalues": MIN_EIGENVALUES,
        "density_graph_seed": DENSITY_GRAPH_SEED + parameter_seed_offset(tau, target_mean_degree),
        "spacing_graph_seed": SPACING_GRAPH_SEED + parameter_seed_offset(tau, target_mean_degree),
        "density_sequence_seed": DENSITY_SEQUENCE_SEED + parameter_seed_offset(tau, target_mean_degree),
        "spacing_sequence_seed": SPACING_SEQUENCE_SEED + parameter_seed_offset(tau, target_mean_degree),
        "num_density_graphs": NUM_DENSITY_GRAPHS,
        "num_spacing_graphs": NUM_SPACING_GRAPHS,
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "code_version": code_version(),
    }


def build_empirical_density(
    spectra: list[ChungLuSpectrum],
) -> EmpiricalSpectralDensity:
    return estimate_empirical_spectral_density(
        eigenvalue_samples=[spectrum.bulk_eigenvalues for spectrum in spectra],
        num_graphs=len(spectra),
        per_graph_dimension=float(np.mean([spectrum.analyzed_dimension for spectrum in spectra])),
        method=DENSITY_METHOD,
        num_points=DENSITY_NUM_POINTS,
        num_bins=DENSITY_NUM_BINS,
        bandwidth_scale=DENSITY_BANDWIDTH_SCALE,
        local_count_half_width=LOCAL_COUNT_HALF_WIDTH,
    )


def extend_edge_quantiles() -> list[float]:
    values = list(EDGE_QUANTILES)
    if ENABLE_EXTRA_EXTREME_EDGE_QUANTILES:
        values.extend(EXTRA_EXTREME_EDGE_QUANTILES)
    return sorted(set(values))


def estimate_power_mass(
    target_eigenvalues_per_graph: int,
    spectra: list[ChungLuSpectrum],
) -> float:
    analyzed_dimension = float(np.mean([spectrum.analyzed_dimension for spectrum in spectra]))
    return float(min(0.15, max(2.0 / analyzed_dimension, target_eigenvalues_per_graph / analyzed_dimension)))


def quantile_window_bounds(
    q: float,
    mass: float,
    analyzed_dimension: float,
) -> tuple[float, float]:
    half = 0.5 * mass
    edge_buffer = 1.0 / max(analyzed_dimension, 10.0)
    lower = max(edge_buffer, q - half)
    upper = min(1.0 - edge_buffer, q + half)
    if upper - lower < mass:
        if q < 0.5:
            upper = min(1.0 - edge_buffer, lower + mass)
        else:
            lower = max(edge_buffer, upper - mass)
    return float(lower), float(upper)


def select_window_eigenvalues(
    eigenvalues: np.ndarray,
    energy_low: float,
    energy_high: float,
) -> np.ndarray:
    values = np.sort(np.asarray(eigenvalues, dtype=float))
    return values[(values >= energy_low) & (values <= energy_high)]


def unfold_selected_spacings(
    selected: np.ndarray,
    density: EmpiricalSpectralDensity,
    analyzed_dimension: int,
    unfolding_method: str,
    energy_center: float,
) -> np.ndarray:
    gaps = np.diff(selected)
    if unfolding_method == "fixed_density":
        rho_center = float(density.evaluate(energy_center))
        return analyzed_dimension * rho_center * gaps
    if unfolding_method == "midpoint_density":
        midpoints = 0.5 * (selected[:-1] + selected[1:])
        return analyzed_dimension * density.evaluate(midpoints) * gaps
    if unfolding_method == "cdf":
        unfolded = analyzed_dimension * np.interp(selected, density.grid, density.cdf)
        return np.diff(unfolded)
    raise ValueError(f"unsupported unfolding_method={unfolding_method!r}")


def summarize_graph_level_ratios(
    graph_gap_ratios: list[np.ndarray],
) -> tuple[float, float, float]:
    pooled = np.concatenate([ratios for ratios in graph_gap_ratios if len(ratios) > 0]) if any(len(r) > 0 for r in graph_gap_ratios) else np.array([], dtype=float)
    pooled_mean = float(np.mean(pooled)) if len(pooled) > 0 else np.nan
    graph_means = np.array([float(np.mean(ratios)) for ratios in graph_gap_ratios if len(ratios) > 0], dtype=float)
    equal_graph_mean = float(np.mean(graph_means)) if len(graph_means) > 0 else np.nan
    se = float(np.std(graph_means, ddof=1) / np.sqrt(len(graph_means))) if len(graph_means) >= 2 else np.nan
    return pooled_mean, equal_graph_mean, se


def compute_quantile_rows(
    spectra: list[ChungLuSpectrum],
    density: EmpiricalSpectralDensity,
    tau: float,
    target_mean_degree: float,
    quantiles: list[float],
    target_eigenvalues_per_graph: int,
    unfolding_method: str,
    region_label: str,
    rng_seed: int,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(rng_seed)
    analyzed_dimension = float(np.mean([spectrum.analyzed_dimension for spectrum in spectra]))
    window_mass = estimate_power_mass(target_eigenvalues_per_graph=target_eigenvalues_per_graph, spectra=spectra)
    rows: list[dict[str, object]] = []

    for q in quantiles:
        q_low, q_high = quantile_window_bounds(q=q, mass=window_mass, analyzed_dimension=analyzed_dimension)
        E_center = float(density.quantiles_to_energies([q])[0])
        E_low, E_high = density.quantiles_to_energies([q_low, q_high])

        graph_spacings: list[np.ndarray] = []
        graph_gap_ratios: list[np.ndarray] = []
        graph_eigenvalue_counts: list[int] = []
        skipped_graphs = 0

        for spectrum in spectra:
            selected = select_window_eigenvalues(spectrum.bulk_eigenvalues, float(E_low), float(E_high))
            if len(selected) < MIN_EIGENVALUES:
                skipped_graphs += 1
                continue
            gaps = np.diff(selected)
            if len(gaps) == 0:
                skipped_graphs += 1
                continue
            spacings = unfold_selected_spacings(
                selected=selected,
                density=density,
                analyzed_dimension=spectrum.analyzed_dimension,
                unfolding_method=unfolding_method,
                energy_center=E_center,
            )
            if not np.all(np.isfinite(spacings)):
                skipped_graphs += 1
                continue
            graph_spacings.append(spacings)
            graph_gap_ratios.append(adjacent_gap_ratios_from_spacings(gaps))
            graph_eigenvalue_counts.append(len(selected))

        if not graph_spacings:
            rows.append(
                {
                    "tau": tau,
                    "target_mean_degree": target_mean_degree,
                    "spectral_region": region_label,
                    "quantile": q,
                    "quantile_low": q_low,
                    "quantile_high": q_high,
                    "E": E_center,
                    "E_low": float(E_low),
                    "E_high": float(E_high),
                    "spacing_window_mode": WINDOW_MODE,
                    "spacing_window_half_width": np.nan,
                    "target_eigenvalues_per_graph": target_eigenvalues_per_graph,
                    "unfolding_method": unfolding_method,
                    "number_of_graphs": len(spectra),
                    "graphs_used": 0,
                    "graphs_skipped": len(spectra),
                    "number_of_eigenvalues": 0,
                    "number_of_gaps": 0,
                    "number_of_gap_ratios": 0,
                    "rho_hat_E": float(density.evaluate(E_center)),
                    "mean_spacing": np.nan,
                    "pooled_mean_gap_ratio": np.nan,
                    "graph_equal_mean_gap_ratio": np.nan,
                    "gap_ratio_standard_error": np.nan,
                    "small_gap_probability": np.nan,
                    "KS_to_GOE": np.nan,
                    "KS_to_Poisson": np.nan,
                    "status": "no_graph_has_enough_eigenvalues",
                    "spacings": np.array([], dtype=float),
                }
            )
            continue

        spacings = np.concatenate(graph_spacings)
        pooled_mean_r, graph_equal_mean_r, r_se = summarize_graph_level_ratios(graph_gap_ratios)
        gap_ratio_count = int(sum(len(ratios) for ratios in graph_gap_ratios))
        sample_size = max(KS_SAMPLE_SIZE, len(spacings))
        goe_sample = rng.rayleigh(scale=np.sqrt(2.0 / np.pi), size=sample_size)
        poisson_sample = rng.exponential(scale=1.0, size=sample_size)
        rows.append(
            {
                "tau": tau,
                "target_mean_degree": target_mean_degree,
                "spectral_region": region_label,
                "quantile": q,
                "quantile_low": q_low,
                "quantile_high": q_high,
                "E": E_center,
                "E_low": float(E_low),
                "E_high": float(E_high),
                "spacing_window_mode": WINDOW_MODE,
                "spacing_window_half_width": 0.5 * float(E_high - E_low),
                "target_eigenvalues_per_graph": target_eigenvalues_per_graph,
                "unfolding_method": unfolding_method,
                "number_of_graphs": len(spectra),
                "graphs_used": len(graph_spacings),
                "graphs_skipped": skipped_graphs,
                "number_of_eigenvalues": int(sum(graph_eigenvalue_counts)),
                "number_of_gaps": int(sum(len(values) for values in graph_spacings)),
                "number_of_gap_ratios": gap_ratio_count,
                "rho_hat_E": float(density.evaluate(E_center)),
                "mean_spacing": float(np.mean(spacings)),
                "pooled_mean_gap_ratio": pooled_mean_r,
                "graph_equal_mean_gap_ratio": graph_equal_mean_r,
                "gap_ratio_standard_error": r_se,
                "small_gap_probability": float(np.mean(spacings < SMALL_SPACING_THRESHOLD)),
                "KS_to_GOE": float(ks_2samp(spacings, goe_sample).statistic),
                "KS_to_Poisson": float(ks_2samp(spacings, poisson_sample).statistic),
                "status": "ok",
                "spacings": spacings,
            }
        )
    return rows


def compute_fixed_rank_edge_rows(
    spectra: list[ChungLuSpectrum],
    tau: float,
    target_mean_degree: float,
    side: str,
    rank_k: int,
) -> list[dict[str, object]]:
    gaps_by_rank: dict[int, list[float]] = {rank: [] for rank in range(1, rank_k)}
    rows: list[dict[str, object]] = []
    for spectrum in spectra:
        values = np.sort(np.asarray(spectrum.bulk_eigenvalues, dtype=float))
        if len(values) < rank_k:
            continue
        if side == "top":
            edge_values = values[-rank_k:]
        elif side == "bottom":
            edge_values = values[:rank_k]
        else:
            raise ValueError(f"unsupported side={side!r}")
        edge_values = np.sort(edge_values)
        raw_gaps = np.diff(edge_values)
        for rank, gap in enumerate(raw_gaps, start=1):
            gaps_by_rank[rank].append(float(gap))

    mean_gap_by_rank = {rank: float(np.mean(gaps)) for rank, gaps in gaps_by_rank.items() if gaps}

    for rank, gaps in gaps_by_rank.items():
        if not gaps:
            continue
        gaps = np.asarray(gaps, dtype=float)
        normalized = gaps / mean_gap_by_rank[rank]
        rows.append(
            {
                "tau": tau,
                "target_mean_degree": target_mean_degree,
                "side": side,
                "rank": rank,
                "number_of_graphs": len(gaps),
                "mean_raw_gap": float(np.mean(gaps)),
                "mean_normalized_gap": float(np.mean(normalized)),
                "std_normalized_gap": float(np.std(normalized, ddof=1)) if len(normalized) >= 2 else 0.0,
            }
        )
    return rows


def plot_weight_degree_tail_diagnostics(
    spectra: list[ChungLuSpectrum],
    tau: float,
    target_mean_degree: float,
) -> plt.Figure:
    expected_weights = np.concatenate([spectrum.expected_weights for spectrum in spectra])
    realized_degrees = np.concatenate([spectrum.degrees for spectrum in spectra])
    diagnostic = spectra[0].weight_diagnostics
    sorted_weights = np.sort(expected_weights)
    ccdf = 1.0 - (np.arange(len(sorted_weights), dtype=float) + 0.5) / len(sorted_weights)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].hist(expected_weights, bins=60, density=True, color="slateblue", alpha=0.7, edgecolor="black")
    axes[0].axvline(diagnostic.final_upper_cap, color="firebrick", ls="--", lw=2, label="final upper cap")
    axes[0].set_title(
        f"Expected Weights\nclip-high frac={diagnostic.fraction_at_upper_cap:.3f}"
    )
    axes[0].set_xlabel("expected weight")
    axes[0].set_ylabel("density")
    axes[0].legend()

    axes[1].hist(realized_degrees, bins=min(80, int(np.max(realized_degrees)) + 1), density=True, color="steelblue", alpha=0.7, edgecolor="black")
    axes[1].set_title("Realized Degree Distribution")
    axes[1].set_xlabel("realized degree")
    axes[1].set_ylabel("density")

    axes[2].loglog(sorted_weights, np.maximum(ccdf, 1e-6), color="black", lw=2, label="empirical expected-weight CCDF")
    reference_x = np.array([sorted_weights[0], sorted_weights[-1]])
    reference_y = ccdf[0] * (reference_x / reference_x[0]) ** (-(tau - 1.0))
    axes[2].loglog(reference_x, reference_y, color="firebrick", ls="--", lw=2, label=f"slope -(tau-1)")
    axes[2].axvline(diagnostic.final_upper_cap, color="darkorange", ls=":", lw=2, label="upper cap")
    axes[2].set_title("Log-Log Tail Diagnostic")
    axes[2].set_xlabel("expected weight")
    axes[2].set_ylabel("CCDF")
    axes[2].legend()

    fig.suptitle(
        f"Power-Law Chung-Lu Diagnostics, n={N}, tau={tau:.2f}, target mean degree={target_mean_degree:.1f}",
        y=1.02,
    )
    fig.tight_layout()
    return fig


def plot_global_density(
    spectra: list[ChungLuSpectrum],
    density: EmpiricalSpectralDensity,
    all_E_values: list[float],
    tau: float,
    target_mean_degree: float,
) -> plt.Figure:
    pooled = np.concatenate([spectrum.bulk_eigenvalues for spectrum in spectra])
    mean_perron = float(np.mean([spectrum.perron_eigenvalue for spectrum in spectra]))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.hist(pooled, bins=100, density=True, alpha=0.45, color="steelblue", edgecolor="black", label="pooled bulk eigenvalues")
    ax.plot(density.grid, density.density, color="black", lw=2, label=f"ensemble {density.method} density")
    for E in all_E_values:
        ax.axvline(E, color="firebrick", ls="--", lw=1.1)
    ax.axvline(mean_perron, color="darkorange", ls=":", lw=2, label="mean Perron outlier")
    ax.set_title(f"Power-Law Chung-Lu Bulk Density, n={N}, tau={tau:.2f}, mean degree={target_mean_degree:.1f}")
    ax.set_xlabel("eigenvalue")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_local_histogram_grid(
    rows: list[dict[str, object]],
    title: str,
) -> plt.Figure:
    valid_spacings = [
        np.asarray(row["spacings"], dtype=float)
        for row in rows
        if int(row["number_of_gaps"]) > 0 and len(np.asarray(row["spacings"], dtype=float)) > 0
    ]
    if not valid_spacings:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.text(0.5, 0.5, "No valid local spacing samples", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.suptitle(title, y=0.95)
        fig.tight_layout()
        return fig
    all_spacings = np.concatenate(valid_spacings)
    hist_max = max(3.5, float(np.quantile(all_spacings, 0.99)))
    bin_edges = np.linspace(0.0, hist_max, NUM_BINS + 1)
    s_grid = np.linspace(0.0, hist_max, 400)
    num_panels = len(rows)
    num_cols = min(4, num_panels)
    num_rows = int(np.ceil(num_panels / num_cols))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4.3 * num_cols, 3.7 * num_rows), squeeze=False, sharex=True, sharey=True)
    for ax, row in zip(axes.flat, rows):
        spacings = np.asarray(row["spacings"], dtype=float)
        ax.hist(spacings, bins=bin_edges, density=True, alpha=0.75, color="mediumseagreen", edgecolor="black", label="empirical")
        ax.plot(s_grid, wigner_surmise_goe(s_grid), color="black", lw=2, label="GOE")
        ax.plot(s_grid, poisson_spacing_density(s_grid), color="firebrick", lw=2, ls="--", label="Poisson")
        ax.set_title(
            f"q={row['quantile']:.4f}\n"
            f"E=[{row['E_low']:.2f},{row['E_high']:.2f}] n_gaps={row['number_of_gaps']}"
        )
        ax.set_xlabel("s")
        ax.set_ylabel("density")
    for ax in list(axes.flat)[len(rows):]:
        ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(title, y=1.02)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig


def plot_quantile_summary(rows: list[dict[str, object]], title: str) -> plt.Figure:
    quantiles = np.array([float(row["quantile"]) for row in rows], dtype=float)
    mean_r = np.array([float(row["graph_equal_mean_gap_ratio"]) for row in rows], dtype=float)
    mean_r_se = np.array([float(row["gap_ratio_standard_error"]) for row in rows], dtype=float)
    small_gap = np.array([float(row["small_gap_probability"]) for row in rows], dtype=float)
    gap_counts = np.array([int(row["number_of_gaps"]) for row in rows], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(quantiles, mean_r, marker="o", color="black")
    axes[0].fill_between(quantiles, mean_r - mean_r_se, mean_r + mean_r_se, color="gray", alpha=0.2)
    axes[0].axhline(0.53590, color="black", ls="--", label="GOE mean r")
    axes[0].axhline(0.38629, color="firebrick", ls="--", label="Poisson mean r")
    axes[0].set_title("Gap Ratio vs Quantile")
    axes[0].set_xlabel("quantile")
    axes[0].set_ylabel("graph-equal mean r")
    axes[0].legend()
    axes[1].plot(quantiles, small_gap, marker="o", color="seagreen")
    axes[1].set_title(f"P(s < {SMALL_SPACING_THRESHOLD:.2f})")
    axes[1].set_xlabel("quantile")
    axes[1].set_ylabel("probability")
    axes[2].plot(quantiles, gap_counts, marker="o", color="steelblue")
    axes[2].set_title("Sample Sizes")
    axes[2].set_xlabel("quantile")
    axes[2].set_ylabel("number of gaps")
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def plot_fixed_rank_summary(rows: list[dict[str, object]], title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for side, color in [("top", "firebrick"), ("bottom", "steelblue")]:
        side_rows = [row for row in rows if row["side"] == side]
        ranks = np.array([int(row["rank"]) for row in side_rows], dtype=float)
        means = np.array([float(row["mean_raw_gap"]) for row in side_rows], dtype=float)
        ax.plot(ranks, means, marker="o", color=color, label=side)
    ax.set_title(title)
    ax.set_xlabel("gap rank")
    ax.set_ylabel("ensemble mean raw gap")
    ax.legend()
    fig.tight_layout()
    return fig


def write_csv(rows: list[dict[str, object]], path: Path, fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def spacing_summary_fieldnames() -> list[str]:
    return [
        "tau", "target_mean_degree", "spectral_region", "quantile", "quantile_low", "quantile_high", "E", "E_low", "E_high",
        "spacing_window_mode", "spacing_window_half_width", "target_eigenvalues_per_graph", "unfolding_method",
        "number_of_graphs", "graphs_used", "graphs_skipped", "number_of_eigenvalues", "number_of_gaps", "number_of_gap_ratios",
        "rho_hat_E", "mean_spacing", "pooled_mean_gap_ratio", "graph_equal_mean_gap_ratio", "gap_ratio_standard_error",
        "small_gap_probability", "KS_to_GOE", "KS_to_Poisson", "status",
    ]


def validate_power_law_run(
    density_spectra: list[ChungLuSpectrum],
    density: EmpiricalSpectralDensity,
    bulk_rows: list[dict[str, object]],
    edge_rows: list[dict[str, object]],
    tau: float,
    target_mean_degree: float,
) -> None:
    density_mass = float(integrate.trapezoid(density.density, density.grid))
    assert abs(density_mass - 1.0) < 0.05
    diagnostics = [spectrum.weight_diagnostics for spectrum in density_spectra]
    assert all(abs(diagnostic.final_mean - target_mean_degree) <= 1e-8 for diagnostic in diagnostics)
    assert all(diagnostic.admissibility_holds for diagnostic in diagnostics)
    realized_mean_degree = float(np.mean([spectrum.realized_mean_degree for spectrum in density_spectra]))
    assert abs(realized_mean_degree - target_mean_degree) < max(0.5 * target_mean_degree, 1.0)
    for spectrum in density_spectra:
        assert len(spectrum.bulk_eigenvalues) == spectrum.analyzed_dimension - 1
        assert float(np.max(spectrum.bulk_eigenvalues)) < spectrum.perron_eigenvalue
    tau_before = [diagnostic.estimated_tau_before_cutoff for diagnostic in diagnostics if diagnostic.estimated_tau_before_cutoff is not None]
    tau_after = [diagnostic.estimated_tau_after_cutoff for diagnostic in diagnostics if diagnostic.estimated_tau_after_cutoff is not None]
    if tau_before:
        assert abs(float(np.mean(tau_before)) - tau) < 1.0
    if tau_after:
        assert abs(float(np.mean(tau_after)) - tau) < 1.0
    assert bulk_rows and edge_rows


def run_local_spacing_only_for_parameter_pair(
    tau: float,
    target_mean_degree: float,
) -> tuple[list[dict[str, object]], Path]:
    output_dir = parameter_output_dir(tau=tau, target_mean_degree=target_mean_degree)
    clear_output_directory_files(output_dir)

    density_spectra, spacing_spectra = build_parameter_pair_ensembles(
        tau=tau,
        target_mean_degree=target_mean_degree,
    )
    density = build_empirical_density(density_spectra)
    local_rows = compute_quantile_rows(
        spectra=spacing_spectra,
        density=density,
        tau=tau,
        target_mean_degree=target_mean_degree,
        quantiles=LOCAL_QUANTILES,
        target_eigenvalues_per_graph=LOCAL_TARGET_EIGENVALUES_PER_GRAPH,
        unfolding_method=LOCAL_UNFOLDING_METHOD,
        region_label="local_quantiles",
        rng_seed=SPACING_GRAPH_SEED + parameter_seed_offset(tau=tau, target_mean_degree=target_mean_degree),
    )

    all_E_values = [float(row["E"]) for row in local_rows]
    density_fig = plot_global_density(
        spacing_spectra,
        density=density,
        all_E_values=all_E_values,
        tau=tau,
        target_mean_degree=target_mean_degree,
    )
    spacing_grid_fig = plot_local_histogram_grid(
        local_rows,
        title=f"Power-Law Chung-Lu Local Quantiles, tau={tau:.2f}, mean degree={target_mean_degree:.1f}",
    )
    spacing_summary_fig = plot_quantile_summary(
        local_rows,
        title=f"Local Quantile Summary, tau={tau:.2f}, mean degree={target_mean_degree:.1f}",
    )

    density_fig.savefig(output_dir / "global_empirical_density.png", dpi=200, bbox_inches="tight")
    spacing_grid_fig.savefig(output_dir / "local_spacing_grid_quantiles.png", dpi=200, bbox_inches="tight")
    spacing_summary_fig.savefig(output_dir / "local_spacing_summary_quantiles.png", dpi=200, bbox_inches="tight")
    plt.close(density_fig)
    plt.close(spacing_grid_fig)
    plt.close(spacing_summary_fig)

    write_csv(local_rows, output_dir / "spacing_summary.csv", spacing_summary_fieldnames())
    with (output_dir / "run_manifest.json").open("w") as handle:
        json.dump(
            run_manifest(tau=tau, target_mean_degree=target_mean_degree, output_profile=OUTPUT_PROFILE),
            handle,
            indent=2,
            sort_keys=True,
        )

    print(
        f"tau={tau:.2f}, mean degree={target_mean_degree:.1f}: "
        f"saved local-spacing-only outputs to {output_dir}"
    )
    return local_rows, output_dir


def run_for_parameter_pair(tau: float, target_mean_degree: float) -> dict[str, object]:
    output_dir = parameter_output_dir(tau=tau, target_mean_degree=target_mean_degree)
    output_dir.mkdir(parents=True, exist_ok=True)
    density_spectra, spacing_spectra = build_parameter_pair_ensembles(
        tau=tau,
        target_mean_degree=target_mean_degree,
    )

    density = build_empirical_density(density_spectra)
    bulk_rows = compute_quantile_rows(
        spectra=spacing_spectra,
        density=density,
        tau=tau,
        target_mean_degree=target_mean_degree,
        quantiles=BULK_QUANTILES,
        target_eigenvalues_per_graph=TARGET_EIGENVALUES_PER_GRAPH_BULK,
        unfolding_method=BULK_UNFOLDING_METHOD,
        region_label="bulk_quantiles",
        rng_seed=SPACING_GRAPH_SEED + parameter_seed_offset(tau=tau, target_mean_degree=target_mean_degree),
    )
    edge_rows = compute_quantile_rows(
        spectra=spacing_spectra,
        density=density,
        tau=tau,
        target_mean_degree=target_mean_degree,
        quantiles=extend_edge_quantiles(),
        target_eigenvalues_per_graph=TARGET_EIGENVALUES_PER_GRAPH_EDGE,
        unfolding_method=EDGE_UNFOLDING_METHOD,
        region_label="near_edge_quantiles",
        rng_seed=SPACING_GRAPH_SEED + parameter_seed_offset(tau=tau, target_mean_degree=target_mean_degree) + 17,
    )
    fixed_rank_rows: list[dict[str, object]] = []
    for rank_k in FIXED_RANK_EDGE_K_VALUES:
        fixed_rank_rows.extend(compute_fixed_rank_edge_rows(spacing_spectra, tau=tau, target_mean_degree=target_mean_degree, side="top", rank_k=rank_k))
        fixed_rank_rows.extend(compute_fixed_rank_edge_rows(spacing_spectra, tau=tau, target_mean_degree=target_mean_degree, side="bottom", rank_k=rank_k))

    validate_power_law_run(density_spectra, density, bulk_rows, edge_rows, tau=tau, target_mean_degree=target_mean_degree)

    weight_fig = plot_weight_degree_tail_diagnostics(spacing_spectra, tau=tau, target_mean_degree=target_mean_degree)
    all_E_values = [float(row["E"]) for row in bulk_rows + edge_rows]
    density_fig = plot_global_density(spacing_spectra, density=density, all_E_values=all_E_values, tau=tau, target_mean_degree=target_mean_degree)
    bulk_hist_fig = plot_local_histogram_grid(bulk_rows, title=f"Power-Law Chung-Lu Bulk Quantiles, tau={tau:.2f}, mean degree={target_mean_degree:.1f}")
    edge_hist_fig = plot_local_histogram_grid(edge_rows, title=f"Power-Law Chung-Lu Near-Edge Quantiles, tau={tau:.2f}, mean degree={target_mean_degree:.1f}")
    bulk_summary_fig = plot_quantile_summary(bulk_rows, title=f"Bulk Quantile Summary, tau={tau:.2f}, mean degree={target_mean_degree:.1f}")
    edge_summary_fig = plot_quantile_summary(edge_rows, title=f"Near-Edge Quantile Summary, tau={tau:.2f}, mean degree={target_mean_degree:.1f}")
    fixed_rank_fig = plot_fixed_rank_summary(fixed_rank_rows, title=f"Fixed-Rank Edge Gaps, tau={tau:.2f}, mean degree={target_mean_degree:.1f}")

    weight_fig.savefig(output_dir / "expected_weight_degree_diagnostics.png", dpi=200, bbox_inches="tight")
    density_fig.savefig(output_dir / "global_empirical_density.png", dpi=200, bbox_inches="tight")
    bulk_hist_fig.savefig(output_dir / "bulk_quantiles_histogram_grid.png", dpi=200, bbox_inches="tight")
    edge_hist_fig.savefig(output_dir / "near_edge_quantiles_histogram_grid.png", dpi=200, bbox_inches="tight")
    bulk_summary_fig.savefig(output_dir / "bulk_quantiles_summary.png", dpi=200, bbox_inches="tight")
    edge_summary_fig.savefig(output_dir / "near_edge_quantiles_summary.png", dpi=200, bbox_inches="tight")
    fixed_rank_fig.savefig(output_dir / "fixed_rank_edge_summary.png", dpi=200, bbox_inches="tight")
    plt.close(weight_fig)
    plt.close(density_fig)
    plt.close(bulk_hist_fig)
    plt.close(edge_hist_fig)
    plt.close(bulk_summary_fig)
    plt.close(edge_summary_fig)
    plt.close(fixed_rank_fig)

    diagnostic = spacing_spectra[0].weight_diagnostics
    weight_fieldnames = [
        "sequence_mode", "cutoff_mode", "tau", "target_mean_degree", "requested_w_min", "requested_w_max",
        "cutoff_coefficient", "raw_min", "raw_max", "scaling_constant", "final_lower_bound", "final_upper_cap",
        "structural_cap", "power_of_n_cap", "target_mean_weight", "final_mean", "final_max", "total_weight",
        "max_weight_squared", "admissibility_margin", "fraction_at_lower_bound", "fraction_at_upper_cap",
        "number_clipped_low", "number_clipped_high", "estimated_tau_before_cutoff", "estimated_tau_after_cutoff",
    ]
    bulk_edge_fieldnames = spacing_summary_fieldnames()
    fixed_rank_fieldnames = [
        "tau", "target_mean_degree", "side", "rank", "number_of_graphs", "mean_raw_gap", "mean_normalized_gap", "std_normalized_gap",
    ]
    extreme_fieldnames = [
        "graph_index", "graph_seed", "sequence_seed", "tau", "target_mean_degree", "lambda_max", "lambda_2", "lambda_min",
        "perron_eigenvalue", "max_expected_weight", "max_realized_degree",
    ]

    write_csv([{name: getattr(diagnostic, name) for name in weight_fieldnames}], output_dir / "weight_sequence_summary.csv", weight_fieldnames)
    write_csv(bulk_rows, output_dir / "bulk_quantiles_summary.csv", bulk_edge_fieldnames)
    write_csv(edge_rows, output_dir / "near_edge_quantiles_summary.csv", bulk_edge_fieldnames)
    write_csv(fixed_rank_rows, output_dir / "fixed_rank_edge_summary.csv", fixed_rank_fieldnames)
    write_csv(
        [
            {
                "graph_index": spectrum.graph_index,
                "graph_seed": spectrum.graph_seed,
                "sequence_seed": spectrum.sequence_seed,
                "tau": spectrum.tau,
                "target_mean_degree": spectrum.target_mean_degree_requested,
                "lambda_max": spectrum.lambda_max,
                "lambda_2": spectrum.lambda_2,
                "lambda_min": spectrum.lambda_min,
                "perron_eigenvalue": spectrum.perron_eigenvalue,
                "max_expected_weight": spectrum.weight_diagnostics.final_max,
                "max_realized_degree": spectrum.max_realized_degree,
            }
            for spectrum in spacing_spectra
        ],
        output_dir / "extreme_eigenvalue_summary.csv",
        extreme_fieldnames,
    )

    manifest = {
        **run_manifest(tau=tau, target_mean_degree=target_mean_degree, output_profile="full_diagnostics"),
        "bulk_quantiles": BULK_QUANTILES,
        "edge_quantiles": extend_edge_quantiles(),
        "target_eigenvalues_per_graph_bulk": TARGET_EIGENVALUES_PER_GRAPH_BULK,
        "target_eigenvalues_per_graph_edge": TARGET_EIGENVALUES_PER_GRAPH_EDGE,
        "bulk_unfolding_method": BULK_UNFOLDING_METHOD,
        "edge_unfolding_method": EDGE_UNFOLDING_METHOD,
    }
    with (output_dir / "run_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    print(
        f"tau={tau:.2f}, mean degree={target_mean_degree:.1f}: "
        f"cap={diagnostic.final_upper_cap:.3f}, raw_max={diagnostic.raw_max:.3f}, "
        f"final_max={diagnostic.final_max:.3f}, tau_before={diagnostic.estimated_tau_before_cutoff:.3f}, "
        f"tau_after={diagnostic.estimated_tau_after_cutoff:.3f}"
    )
    return {
        "tau": tau,
        "target_mean_degree": target_mean_degree,
        "bulk_rows": bulk_rows,
        "edge_rows": edge_rows,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PROFILE == "local_spacing_only":
        for tau, target_mean_degree in active_parameter_pairs():
            run_local_spacing_only_for_parameter_pair(tau=tau, target_mean_degree=target_mean_degree)
        return
    if OUTPUT_PROFILE != "full_diagnostics":
        raise ValueError(
            f"unsupported OUTPUT_PROFILE={OUTPUT_PROFILE!r}; "
            f"supported values are {sorted(SUPPORTED_OUTPUT_PROFILES)}"
        )

    grid_rows: list[dict[str, object]] = []
    for tau, target_mean_degree in active_parameter_pairs():
        result = run_for_parameter_pair(tau=tau, target_mean_degree=target_mean_degree)
        for row in result["bulk_rows"] + result["edge_rows"]:
            grid_rows.append(
                {
                    "tau": tau,
                    "target_mean_degree": target_mean_degree,
                    "spectral_region": row["spectral_region"],
                    "quantile": row["quantile"],
                    "E": row["E"],
                    "graph_equal_mean_gap_ratio": row["graph_equal_mean_gap_ratio"],
                    "small_gap_probability": row["small_gap_probability"],
                    "KS_to_GOE": row["KS_to_GOE"],
                    "KS_to_Poisson": row["KS_to_Poisson"],
                    "number_of_gaps": row["number_of_gaps"],
                    "unfolding_method": row["unfolding_method"],
                }
            )
    write_csv(
        grid_rows,
        OUTPUT_DIR / "power_law_grid_summary_quantiles.csv",
        [
            "tau", "target_mean_degree", "spectral_region", "quantile", "E",
            "graph_equal_mean_gap_ratio", "small_gap_probability", "KS_to_GOE",
            "KS_to_Poisson", "number_of_gaps", "unfolding_method",
        ],
    )


if __name__ == "__main__":
    main()
