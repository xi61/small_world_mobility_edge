from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy.stats import ks_2samp

try:
    from .empirical_spectral_density import EmpiricalSpectralDensity, estimate_empirical_spectral_density
    from .erdos_renyi_baseline import ErdosRenyiSpectrum, generate_erdos_renyi_ensemble
    from .regular_baseline import adjacent_gap_ratios_from_spacings, local_window_spacings, wigner_surmise_goe
except ImportError:  # pragma: no cover - script-style fallback
    from empirical_spectral_density import EmpiricalSpectralDensity, estimate_empirical_spectral_density
    from erdos_renyi_baseline import ErdosRenyiSpectrum, generate_erdos_renyi_ensemble
    from regular_baseline import adjacent_gap_ratios_from_spacings, local_window_spacings, wigner_surmise_goe


MODEL = "erdos_renyi"
N = 3000
ACTIVE_AVERAGE_DEGREES = [3, 6, 10, 20]
ALL_AVERAGE_DEGREES = [3, 6, 10, 20]
ACTIVE_COMPONENT_MODES = ["largest_connected_component"]
NUM_DENSITY_GRAPHS = 30
NUM_SPACING_GRAPHS = 30
DENSITY_SEED = 31415
SPACING_SEED = 27182
DENSITY_METHOD = "local_count"
LOCAL_COUNT_HALF_WIDTH = 0.15
DENSITY_NUM_POINTS = 2001
DENSITY_NUM_BINS = 120
DENSITY_BANDWIDTH_SCALE = 1.0
EXPLICIT_E_VALUES = [-2.5, -1.5, 0.0, 1.5, 2.5]
BULK_QUANTILES = [0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95]
QUANTILES = BULK_QUANTILES
ER_EDGE_QUANTILES = [0.005, 0.01, 0.025, 0.05, 0.95, 0.975, 0.99, 0.995]
SPACING_WINDOW_MODE = "fixed_energy_half_width"
SPACING_WINDOW_HALF_WIDTH = 0.15
NUM_BINS = 30
MIN_EIGENVALUES = 10
MIN_TOTAL_GAPS = 100
SMALL_SPACING_THRESHOLD = 0.20
KS_SAMPLE_SIZE = 4096
ZERO_TOLERANCE = 1e-10
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "erdos_renyi"
RAW_ALL_EIGS = "raw_all_eigs"
CONTINUOUS_NONZERO = "continuous_nonzero"

SpectrumSelector = Callable[[ErdosRenyiSpectrum], np.ndarray]


def code_version() -> str | None:
    repo = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def timestamp_slug() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def config_hash(config: dict[str, object]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def run_manifest(
    average_degree: float,
    component_mode: str,
    run_id: str,
    timestamp: str,
) -> dict[str, object]:
    return {
        "model": MODEL,
        "n_requested": N,
        "average_degree": average_degree,
        "p": average_degree / (N - 1),
        "component_mode": component_mode,
        "density_method": DENSITY_METHOD,
        "density_local_count_half_width": LOCAL_COUNT_HALF_WIDTH,
        "spacing_window_mode": SPACING_WINDOW_MODE,
        "spacing_window_half_width": SPACING_WINDOW_HALF_WIDTH,
        "bulk_quantiles": BULK_QUANTILES,
        "edge_quantiles": ER_EDGE_QUANTILES,
        "explicit_E_values": EXPLICIT_E_VALUES,
        "density_seed": ensemble_seed(DENSITY_SEED, average_degree),
        "spacing_seed": ensemble_seed(SPACING_SEED, average_degree),
        "density_base_seed": DENSITY_SEED,
        "spacing_base_seed": SPACING_SEED,
        "num_density_graphs": NUM_DENSITY_GRAPHS,
        "num_spacing_graphs": NUM_SPACING_GRAPHS,
        "code_version": code_version(),
        "run_id": run_id,
        "timestamp": timestamp,
    }


def run_output_dir(manifest: dict[str, object]) -> Path:
    ts = str(manifest["timestamp"]).replace(":", "").replace("-", "")
    digest = config_hash({k: v for k, v in manifest.items() if k not in {"run_id", "timestamp"}})
    return (
        OUTPUT_DIR
        / f"n_{int(manifest['n_requested'])}"
        / f"avgdeg_{int(float(manifest['average_degree']))}"
        / f"component_{manifest['component_mode']}"
        / f"run_{ts}_{digest}"
    )


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def poisson_spacing_density(s: np.ndarray | float) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    density = np.zeros_like(s, dtype=float)
    nonnegative = s >= 0
    density[nonnegative] = np.exp(-s[nonnegative])
    return density


def ensemble_seed(base_seed: int, average_degree: float) -> int:
    return int(base_seed + int(100 * average_degree))


def _ensemble_dimension(spectra: list[ErdosRenyiSpectrum]) -> float:
    return float(np.mean([spectrum.analyzed_dimension for spectrum in spectra]))


def _pooled_bulk_eigenvalues(spectra: list[ErdosRenyiSpectrum]) -> list[np.ndarray]:
    return [spectrum.bulk_eigenvalues for spectrum in spectra]


def _spectral_quantiles_from_density(density: EmpiricalSpectralDensity, E_values: list[float] | np.ndarray) -> np.ndarray:
    return np.interp(np.asarray(E_values, dtype=float), density.grid, density.cdf)


# def build_empirical_density(spectra: list[ErdosRenyiSpectrum]) -> EmpiricalSpectralDensity:
#     return estimate_empirical_spectral_density(
#         eigenvalue_samples=_pooled_bulk_eigenvalues(spectra),
#         num_graphs=len(spectra),
#         per_graph_dimension=_ensemble_dimension(spectra),
#         method=DENSITY_METHOD,
#         num_points=DENSITY_NUM_POINTS,
#         num_bins=DENSITY_NUM_BINS,
#         bandwidth_scale=DENSITY_BANDWIDTH_SCALE,
#         local_count_half_width=LOCAL_COUNT_HALF_WIDTH,
#     )
def build_empirical_density(
    spectra: list[ErdosRenyiSpectrum],
    select_values: SpectrumSelector,
) -> EmpiricalSpectralDensity:
    """
    Estimate a density from exactly the eigenvalues that will later
    be used in the spacing analysis.

    If select_values removes the zero atom, the resulting density is
    the conditional continuous density, normalized to total mass one.
    """
    if not spectra:
        raise ValueError("spectra must be non-empty")
    eigenvalue_samples = []
    for spectrum in spectra:
        values = np.asarray(select_values(spectrum), dtype=float)
        if values.ndim != 1:
            raise ValueError(f"selector must return a one-dimensional array; got shape={values.shape}")
        if len(values) == 0:
            raise ValueError(
                f"selector retained no eigenvalues for graph_index={spectrum.graph_index} seed={spectrum.seed}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"selector returned non-finite eigenvalues for graph_index={spectrum.graph_index}")
        eigenvalue_samples.append(np.sort(values))

    retained_dimension = float(
        np.mean([len(values) for values in eigenvalue_samples])
    )

    return estimate_empirical_spectral_density(
        eigenvalue_samples=eigenvalue_samples,
        num_graphs=len(eigenvalue_samples),
        per_graph_dimension=retained_dimension,
        method=DENSITY_METHOD,
        num_points=DENSITY_NUM_POINTS,
        num_bins=DENSITY_NUM_BINS,
        bandwidth_scale=DENSITY_BANDWIDTH_SCALE,
        local_count_half_width=LOCAL_COUNT_HALF_WIDTH,
    )


def exact_zero_multiplicity(eigenvalues: np.ndarray) -> int:
    return int(np.sum(np.isclose(np.asarray(eigenvalues, dtype=float), 0.0, atol=ZERO_TOLERANCE, rtol=0.0)))


def zero_gap_count(values: np.ndarray) -> int:
    gaps = np.diff(np.sort(np.asarray(values, dtype=float)))
    return int(np.sum(np.isclose(gaps, 0.0, atol=ZERO_TOLERANCE, rtol=0.0)))


def continuous_bulk_eigenvalues(spectrum: ErdosRenyiSpectrum) -> np.ndarray:
    values = np.asarray(spectrum.bulk_eigenvalues, dtype=float)
    return values[~np.isclose(values, 0.0, atol=ZERO_TOLERANCE, rtol=0.0)]


def compute_local_spacing_summary(
    spectra: list[ErdosRenyiSpectrum],
    density: EmpiricalSpectralDensity,
    E_values: list[float] | np.ndarray,
    average_degree: float,
    spectral_quantiles: list[float | None],
    analysis_variant: str,
    component_mode: str,
    run_meta: dict[str, object],
    select_values: SpectrumSelector,
    density_variant: str,
    selected_spectrum_variant: str,
    rng_seed: int,
) -> tuple[plt.Figure, plt.Figure, list[dict[str, object]]]:
    if analysis_variant == CONTINUOUS_NONZERO:
        expected_pair = (CONTINUOUS_NONZERO, CONTINUOUS_NONZERO)
    elif analysis_variant == RAW_ALL_EIGS:
        expected_pair = (RAW_ALL_EIGS, RAW_ALL_EIGS)
    else:
        raise ValueError(f"unsupported analysis_variant={analysis_variant!r}")
    if (density_variant, selected_spectrum_variant) != expected_pair:
        raise ValueError(
            f"analysis_variant={analysis_variant!r} cannot use density_variant={density_variant!r} "
            f"and selected_spectrum_variant={selected_spectrum_variant!r}"
        )

    rng = np.random.default_rng(rng_seed)
    results: list[dict[str, object]] = []
    pooled_spacings: list[np.ndarray] = []

    for E, quantile in zip(np.asarray(E_values, dtype=float), spectral_quantiles):
        rho_E = float(density.evaluate(E))
        graph_results = []
        skipped_graphs = 0
        zero_multiplicities = []
        zero_gap_counts = []
        retained_counts = []
        retained_counts_used = []
        unfolding_dimensions = []
        for spectrum in spectra:
            eigenvalues = np.asarray(select_values(spectrum), dtype=float)
            if eigenvalues.ndim != 1:
                raise ValueError(f"selector must return a one-dimensional array; got shape={eigenvalues.shape}")
            if len(eigenvalues) == 0:
                raise ValueError(f"selector retained no eigenvalues for graph_index={spectrum.graph_index} seed={spectrum.seed}")
            if not np.all(np.isfinite(eigenvalues)):
                raise ValueError(f"selector returned non-finite eigenvalues for graph_index={spectrum.graph_index}")
            eigenvalues = np.sort(eigenvalues)
            retained_count = len(eigenvalues)
            retained_counts.append(retained_count)
            zero_multiplicities.append(exact_zero_multiplicity(spectrum.eigenvalues))
            # Zero removal leaves the negative-to-positive center gap intact;
            # split-sign center diagnostics can be added separately later.
            center_selected = eigenvalues[
                (eigenvalues >= float(E - SPACING_WINDOW_HALF_WIDTH))
                & (eigenvalues <= float(E + SPACING_WINDOW_HALF_WIDTH))
            ]
            zero_gap_counts.append(zero_gap_count(center_selected) if len(center_selected) >= 2 else 0)
            try:
                graph_result = local_window_spacings(
                    eigenvalues=eigenvalues,
                    E=float(E),
                    window_half_width=SPACING_WINDOW_HALF_WIDTH,
                    density_at_E=rho_E,
                    density_method="empirical",
                    analyzed_dimension=retained_count,
                    reweight_within_window=False,
                    normalize_mean=False,
                    min_eigenvalues=MIN_EIGENVALUES,
                )
            except ValueError:
                skipped_graphs += 1
                continue
            if graph_result.analyzed_dimension != retained_count:
                raise AssertionError("unfolding dimension must match the selected eigenvalue count")
            retained_counts_used.append(retained_count)
            unfolding_dimensions.append(graph_result.analyzed_dimension)
            graph_results.append(graph_result)

        base_row = {
            "model": MODEL,
            "n_requested": N,
            "average_degree": average_degree,
            "p": average_degree / (N - 1),
            "component_mode": component_mode,
            "analysis_variant": analysis_variant,
            "density_variant": density_variant,
            "selected_spectrum_variant": selected_spectrum_variant,
            "density_method": DENSITY_METHOD,
            "density_local_count_half_width": LOCAL_COUNT_HALF_WIDTH,
            "spacing_window_mode": SPACING_WINDOW_MODE,
            "spacing_window_half_width": SPACING_WINDOW_HALF_WIDTH,
            "quantile": quantile,
            "E": float(E),
            "density_seed": run_meta.get("density_seed", ensemble_seed(DENSITY_SEED, average_degree)),
            "spacing_seed": run_meta.get("spacing_seed", ensemble_seed(SPACING_SEED, average_degree)),
            "num_density_graphs": NUM_DENSITY_GRAPHS,
            "num_spacing_graphs": NUM_SPACING_GRAPHS,
            "code_version": run_meta["code_version"],
            "run_id": run_meta["run_id"],
            "timestamp": run_meta["timestamp"],
            "window_min": float(E - SPACING_WINDOW_HALF_WIDTH),
            "window_max": float(E + SPACING_WINDOW_HALF_WIDTH),
            "mean_component_size": float(np.mean([spectrum.analyzed_dimension for spectrum in spectra])),
            "mean_num_components": float(np.mean([spectrum.num_components for spectrum in spectra])),
            "mean_largest_component_size": float(np.mean([spectrum.largest_component_size for spectrum in spectra])),
            "mean_num_isolated_vertices": float(np.mean([spectrum.num_isolated_vertices for spectrum in spectra])),
            "exact_zero_multiplicity": float(np.mean(zero_multiplicities)) if zero_multiplicities else np.nan,
            "number_of_exact_zero_gaps_in_center_window": float(np.mean(zero_gap_counts)) if zero_gap_counts else np.nan,
            "rho_hat_E": rho_E,
            "mean_retained_eigenvalue_count": float(np.mean(retained_counts)) if retained_counts else np.nan,
        }

        if not graph_results:
            results.append(
                {
                    **base_row,
                    "number_of_graphs": len(spectra),
                    "graphs_used": 0,
                    "graphs_skipped": len(spectra),
                    "number_of_eigenvalues": 0,
                    "number_of_gaps": 0,
                    "mean_spacing": np.nan,
                    "mean_gap_ratio": np.nan,
                    "small_gap_probability": np.nan,
                    "ks_distance_to_goe": np.nan,
                    "ks_distance_to_poisson": np.nan,
                    "mean_unfolding_dimension": np.nan,
                    "unfolding_dimension_matches_retained_count": True,
                    "status": "no_graph_has_enough_eigenvalues",
                }
            )
            continue

        spacings = np.concatenate([result.spacings for result in graph_results])
        eigenvalue_count = int(sum(result.num_eigenvalues for result in graph_results))
        gap_count = int(sum(result.num_gaps for result in graph_results))
        assert np.all(np.isfinite(spacings))
        gap_ratios = np.concatenate(
            [adjacent_gap_ratios_from_spacings(result.raw_gaps) for result in graph_results if len(result.raw_gaps) >= 2]
        ) if any(len(result.raw_gaps) >= 2 for result in graph_results) else np.array([], dtype=float)

        status = "ok" if gap_count >= MIN_TOTAL_GAPS else f"few_gaps:{gap_count}"
        mean_spacing = float(np.mean(spacings))
        mean_gap_ratio = float(np.mean(gap_ratios)) if len(gap_ratios) > 0 else np.nan
        small_spacing_probability = float(np.mean(spacings < SMALL_SPACING_THRESHOLD))
        sample_size = max(KS_SAMPLE_SIZE, len(spacings))
        goe_sample = rng.rayleigh(scale=np.sqrt(2.0 / np.pi), size=sample_size)
        poisson_sample = rng.exponential(scale=1.0, size=sample_size)
        ks_goe = float(ks_2samp(spacings, goe_sample).statistic)
        ks_poisson = float(ks_2samp(spacings, poisson_sample).statistic)

        row = {
            **base_row,
            "number_of_graphs": len(spectra),
            "graphs_used": len(graph_results),
            "graphs_skipped": skipped_graphs,
            "number_of_eigenvalues": eigenvalue_count,
            "number_of_gaps": gap_count,
            "mean_spacing": mean_spacing,
            "mean_gap_ratio": mean_gap_ratio,
            "small_gap_probability": small_spacing_probability,
            "ks_distance_to_goe": ks_goe,
            "ks_distance_to_poisson": ks_poisson,
            "mean_unfolding_dimension": float(np.mean(unfolding_dimensions)),
            "unfolding_dimension_matches_retained_count": bool(
                np.array_equal(
                    np.asarray(unfolding_dimensions, dtype=int),
                    np.asarray(retained_counts_used, dtype=int),
                )
            ),
            "status": status,
            "spacings": spacings,
        }
        results.append(row)
        if gap_count >= MIN_TOTAL_GAPS:
            pooled_spacings.append(spacings)

    valid_items = [item for item in results if int(item.get("number_of_gaps", 0)) > 0]
    if not valid_items:
        raise ValueError("no local ER windows produced any gaps")

    spacing_source = pooled_spacings if pooled_spacings else [np.asarray(item["spacings"], dtype=float) for item in valid_items]
    all_spacings = np.concatenate(spacing_source)
    hist_max = max(3.5, float(np.quantile(all_spacings, 0.99)))
    bin_edges = np.linspace(0.0, hist_max, NUM_BINS + 1)
    s_grid = np.linspace(0.0, hist_max, 400)

    num_panels = len(results)
    num_cols = min(3, num_panels)
    num_rows = int(np.ceil(num_panels / num_cols))
    fig_grid, axes = plt.subplots(num_rows, num_cols, figsize=(5.2 * num_cols, 3.9 * num_rows), squeeze=False)

    for ax, item in zip(axes.flat, results):
        if int(item["number_of_gaps"]) == 0:
            ax.text(0.5, 0.5, str(item["status"]), ha="center", va="center", wrap=True)
            ax.set_title(f"E={float(item['E']):.2f}")
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        spacings = np.asarray(item["spacings"], dtype=float)
        ax.hist(spacings, bins=bin_edges, density=True, alpha=0.75, color="mediumseagreen", edgecolor="black", label="empirical")
        ax.plot(s_grid, wigner_surmise_goe(s_grid), color="black", lw=2, label="GOE")
        ax.plot(s_grid, poisson_spacing_density(s_grid), color="firebrick", lw=2, ls="--", label="Poisson")
        q_text = "n/a" if item["quantile"] is None else f"q={float(item['quantile']):.3f}, "
        ax.set_title(
            f"E={float(item['E']):.2f}, {q_text}window=[{float(item['window_min']):.2f}, {float(item['window_max']):.2f}]\n"
            f"variant={item['analysis_variant']}, graphs={int(item['graphs_used'])}, gaps={int(item['number_of_gaps'])}"
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

    E_plot = np.array([float(item["E"]) for item in valid_items], dtype=float)
    mean_r_plot = np.array([float(item["mean_gap_ratio"]) for item in valid_items], dtype=float)
    small_gap_plot = np.array([float(item["small_gap_probability"]) for item in valid_items], dtype=float)
    eigen_count_plot = np.array([int(item["number_of_eigenvalues"]) for item in valid_items], dtype=float)
    gap_count_plot = np.array([int(item["number_of_gaps"]) for item in valid_items], dtype=float)
    ks_goe_plot = np.array([float(item["ks_distance_to_goe"]) for item in valid_items], dtype=float)
    ks_poisson_plot = np.array([float(item["ks_distance_to_poisson"]) for item in valid_items], dtype=float)

    fig_summary, axes_summary = plt.subplots(2, 2, figsize=(13, 8))
    ax = axes_summary[0, 0]
    ax.plot(E_plot, mean_r_plot, marker="o", color="steelblue")
    ax.axhline(0.53590, color="black", ls="--", label="GOE mean r")
    ax.axhline(0.38629, color="firebrick", ls="--", label="Poisson mean r")
    ax.set_title(f"Mean Local Gap Ratio vs E ({analysis_variant})")
    ax.set_xlabel("E")
    ax.set_ylabel("mean local gap ratio")
    ax.legend()

    ax = axes_summary[0, 1]
    ax.plot(E_plot, small_gap_plot, marker="o", color="seagreen")
    ax.set_title(f"Small-Gap Probability P(s < {SMALL_SPACING_THRESHOLD:.2f})")
    ax.set_xlabel("E")
    ax.set_ylabel("probability")

    ax = axes_summary[1, 0]
    ax.plot(E_plot, ks_goe_plot, marker="o", color="black", label="KS to GOE")
    ax.plot(E_plot, ks_poisson_plot, marker="s", color="firebrick", label="KS to Poisson")
    ax.set_title("KS Distance vs E")
    ax.set_xlabel("E")
    ax.set_ylabel("KS statistic")
    ax.legend()

    ax = axes_summary[1, 1]
    ax.plot(E_plot, eigen_count_plot, marker="o", color="goldenrod", label="eigenvalues")
    ax.plot(E_plot, gap_count_plot, marker="s", color="mediumpurple", label="gaps")
    ax.set_title("Window Sample Sizes vs E")
    ax.set_xlabel("E")
    ax.set_ylabel("count")
    ax.legend()
    fig_summary.tight_layout()

    return fig_grid, fig_summary, results


def plot_global_empirical_density(
    density_spectra: list[ErdosRenyiSpectrum],
    spacing_spectra: list[ErdosRenyiSpectrum],
    density: EmpiricalSpectralDensity,
    explicit_E_values: list[float],
    quantile_E_values: list[float],
    average_degree: float,
    component_mode: str,
) -> plt.Figure:
    pooled_spacing_eigenvalues = np.concatenate([spectrum.bulk_eigenvalues for spectrum in spacing_spectra])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(pooled_spacing_eigenvalues, bins=100, density=True, alpha=0.45, color="steelblue", edgecolor="black", label="pooled bulk eigenvalues")
    ax.plot(density.grid, density.density, color="black", lw=2, label=f"ensemble {density.method} density")
    for E in explicit_E_values:
        ax.axvspan(E - SPACING_WINDOW_HALF_WIDTH, E + SPACING_WINDOW_HALF_WIDTH, color="mediumseagreen", alpha=0.10)
        ax.axvline(E, color="mediumseagreen", lw=1.4)
    for E in quantile_E_values:
        ax.axvline(E, color="firebrick", lw=1.2, ls="--")
    mean_perron = float(np.mean([spectrum.perron_eigenvalue for spectrum in density_spectra]))
    ax.axvline(mean_perron, color="darkorange", lw=1.8, ls=":", label="mean Perron outlier")
    ax.set_title(f"Erdos-Renyi Bulk Density, n={N}, average degree={average_degree}, mode={component_mode}")
    ax.set_xlabel("eigenvalue")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    return fig


def write_summary_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "model",
        "n_requested",
        "average_degree",
        "p",
        "component_mode",
        "analysis_variant",
        "density_variant",
        "selected_spectrum_variant",
        "density_method",
        "density_local_count_half_width",
        "spacing_window_mode",
        "spacing_window_half_width",
        "quantile",
        "E",
        "density_seed",
        "spacing_seed",
        "num_density_graphs",
        "num_spacing_graphs",
        "code_version",
        "run_id",
        "timestamp",
        "window_min",
        "window_max",
        "mean_component_size",
        "mean_num_components",
        "mean_largest_component_size",
        "mean_num_isolated_vertices",
        "exact_zero_multiplicity",
        "number_of_exact_zero_gaps_in_center_window",
        "number_of_graphs",
        "graphs_used",
        "graphs_skipped",
        "number_of_eigenvalues",
        "number_of_gaps",
        "rho_hat_E",
        "mean_retained_eigenvalue_count",
        "mean_unfolding_dimension",
        "unfolding_dimension_matches_retained_count",
        "mean_spacing",
        "mean_gap_ratio",
        "small_gap_probability",
        "ks_distance_to_goe",
        "ks_distance_to_poisson",
        "status",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def print_results_table(rows: list[dict[str, object]]) -> None:
    print(
        f"{'mode':>8}  {'variant':>18}  {'E':>6}  {'q':>6}  {'gaps':>8}  "
        f"{'mean s':>8}  {'mean r':>8}  {'zero mult':>9}  status"
    )
    for row in rows:
        q = "n/a" if row["quantile"] is None else f"{float(row['quantile']):.3f}"
        mean_s = "nan" if np.isnan(row["mean_spacing"]) else f"{float(row['mean_spacing']):.4f}"
        mean_r = "nan" if np.isnan(row["mean_gap_ratio"]) else f"{float(row['mean_gap_ratio']):.4f}"
        zero_mult = "nan" if np.isnan(row["exact_zero_multiplicity"]) else f"{float(row['exact_zero_multiplicity']):.2f}"
        print(
            f"{str(row['component_mode']):>8}  {str(row['analysis_variant']):>18}  {float(row['E']):>6.2f}  {q:>6}  "
            f"{int(row['number_of_gaps']):>8d}  {mean_s:>8}  {mean_r:>8}  {zero_mult:>9}  {row['status']}"
        )


def run_validation_checks(
    density_spectra: list[ErdosRenyiSpectrum],
    spacing_spectra: list[ErdosRenyiSpectrum],
    density: EmpiricalSpectralDensity,
    summary_rows: list[dict[str, object]],
    average_degree: float,
    component_mode: str,
    select_values: SpectrumSelector,
) -> None:
    mean_realized_degree = float(np.mean([spectrum.realized_mean_degree for spectrum in spacing_spectra]))
    assert abs(mean_realized_degree - average_degree) < max(0.35 * average_degree, 0.5)
    density_mass = float(integrate.trapezoid(density.density, density.grid))
    assert abs(density_mass - 1.0) < 0.05
    repro_a = generate_erdos_renyi_ensemble(n=N, average_degree=average_degree, num_graphs=1, seed=ensemble_seed(DENSITY_SEED, average_degree), component_mode=component_mode)[0]
    repro_b = generate_erdos_renyi_ensemble(n=N, average_degree=average_degree, num_graphs=1, seed=ensemble_seed(DENSITY_SEED, average_degree), component_mode=component_mode)[0]
    assert np.allclose(repro_a.eigenvalues, repro_b.eigenvalues)
    for spectrum in density_spectra + spacing_spectra:
        assert len(spectrum.bulk_eigenvalues) == spectrum.analyzed_dimension - 1
        assert float(np.max(spectrum.bulk_eigenvalues)) < spectrum.perron_eigenvalue
    ok_rows = [row for row in summary_rows if int(row["number_of_gaps"]) > 0]
    assert ok_rows
    for spectrum in density_spectra + spacing_spectra:
        selected = np.asarray(select_values(spectrum), dtype=float)
        assert selected.ndim == 1
        assert len(selected) > 0
        assert not np.any(np.isclose(selected, 0.0, atol=ZERO_TOLERANCE, rtol=0.0))
    for row in ok_rows:
        assert row["analysis_variant"] == CONTINUOUS_NONZERO
        assert row["density_variant"] == CONTINUOUS_NONZERO
        assert row["selected_spectrum_variant"] == CONTINUOUS_NONZERO
        assert bool(row["unfolding_dimension_matches_retained_count"])
        assert float(row["number_of_exact_zero_gaps_in_center_window"]) == 0.0
        for field in [
            "rho_hat_E",
            "mean_spacing",
            "mean_gap_ratio",
            "small_gap_probability",
            "ks_distance_to_goe",
            "ks_distance_to_poisson",
        ]:
            assert np.isfinite(float(row[field]))


def run_for_average_degree_and_mode(average_degree: float, component_mode: str) -> dict[str, object]:
    timestamp = timestamp_slug()
    provisional = run_manifest(average_degree=average_degree, component_mode=component_mode, run_id="", timestamp=timestamp)
    run_id = f"{timestamp}_{config_hash(provisional)}"
    manifest = run_manifest(average_degree=average_degree, component_mode=component_mode, run_id=run_id, timestamp=timestamp)
    output_dir = run_output_dir(manifest)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "run_manifest.json", manifest)

    density_spectra = generate_erdos_renyi_ensemble(
        n=N,
        average_degree=average_degree,
        num_graphs=NUM_DENSITY_GRAPHS,
        seed=ensemble_seed(DENSITY_SEED, average_degree),
        component_mode=component_mode,
    )
    spacing_spectra = generate_erdos_renyi_ensemble(
        n=N,
        average_degree=average_degree,
        num_graphs=NUM_SPACING_GRAPHS,
        seed=ensemble_seed(SPACING_SEED, average_degree),
        component_mode=component_mode,
    )
    raw_density = build_empirical_density(
        density_spectra,
        select_values=lambda spectrum: spectrum.bulk_eigenvalues,
    )
    continuous_density = build_empirical_density(
        density_spectra,
        select_values=continuous_bulk_eigenvalues,
    )
    quantile_E_values = continuous_density.quantiles_to_energies(BULK_QUANTILES)
    explicit_quantiles = [float(q) for q in _spectral_quantiles_from_density(continuous_density, EXPLICIT_E_VALUES)]

    fig_density = plot_global_empirical_density(
        density_spectra=density_spectra,
        spacing_spectra=spacing_spectra,
        density=raw_density,
        explicit_E_values=EXPLICIT_E_VALUES,
        quantile_E_values=list(np.asarray(quantile_E_values, dtype=float)),
        average_degree=average_degree,
        component_mode=component_mode,
    )
    fig_explicit_grid, fig_explicit_summary, explicit_rows = compute_local_spacing_summary(
        spectra=spacing_spectra,
        density=continuous_density,
        E_values=EXPLICIT_E_VALUES,
        average_degree=average_degree,
        spectral_quantiles=explicit_quantiles,
        analysis_variant=CONTINUOUS_NONZERO,
        component_mode=component_mode,
        run_meta=manifest,
        select_values=continuous_bulk_eigenvalues,
        density_variant=CONTINUOUS_NONZERO,
        selected_spectrum_variant=CONTINUOUS_NONZERO,
        rng_seed=SPACING_SEED + int(average_degree),
    )
    fig_quantile_grid, fig_quantile_summary, quantile_rows = compute_local_spacing_summary(
        spectra=spacing_spectra,
        density=continuous_density,
        E_values=quantile_E_values,
        average_degree=average_degree,
        spectral_quantiles=[float(q) for q in BULK_QUANTILES],
        analysis_variant=CONTINUOUS_NONZERO,
        component_mode=component_mode,
        run_meta=manifest,
        select_values=continuous_bulk_eigenvalues,
        density_variant=CONTINUOUS_NONZERO,
        selected_spectrum_variant=CONTINUOUS_NONZERO,
        rng_seed=SPACING_SEED + int(average_degree) + 17,
    )

    all_rows = explicit_rows + quantile_rows
    run_validation_checks(
        density_spectra=density_spectra,
        spacing_spectra=spacing_spectra,
        density=continuous_density,
        summary_rows=all_rows,
        average_degree=average_degree,
        component_mode=component_mode,
        select_values=continuous_bulk_eigenvalues,
    )

    fig_density.savefig(output_dir / "global_empirical_density.png", dpi=200, bbox_inches="tight")
    fig_explicit_grid.savefig(output_dir / "local_spacing_grid_explicit_E.png", dpi=200, bbox_inches="tight")
    fig_explicit_summary.savefig(output_dir / "local_spacing_summary_explicit_E.png", dpi=200, bbox_inches="tight")
    fig_quantile_grid.savefig(output_dir / "local_spacing_grid_quantiles.png", dpi=200, bbox_inches="tight")
    fig_quantile_summary.savefig(output_dir / "local_spacing_summary_quantiles.png", dpi=200, bbox_inches="tight")

    plt.close(fig_density)
    plt.close(fig_explicit_grid)
    plt.close(fig_explicit_summary)
    plt.close(fig_quantile_grid)
    plt.close(fig_quantile_summary)

    write_summary_csv(all_rows, output_dir / "spacing_summary.csv")
    print(f"\nAverage degree {average_degree:.2f}, component_mode={component_mode}")
    print(f"Saved run directory to {output_dir}")
    print_results_table(all_rows)
    return {"manifest": manifest, "output_dir": output_dir, "summary_rows": all_rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for average_degree in ACTIVE_AVERAGE_DEGREES:
        for component_mode in ACTIVE_COMPONENT_MODES:
            run_for_average_degree_and_mode(average_degree, component_mode)


if __name__ == "__main__":
    main()
