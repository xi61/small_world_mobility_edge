from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
import tempfile
from pathlib import Path
import subprocess
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "small-world-mobility-edge-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp

from .chung_lu_eigenvector_diagnostics import (
    COORDINATE_DECAY_MAX_RANK,
    coordinate_decay_profile_rows,
    eigenvector_localization_summary,
)
from .chung_lu_spacing_by_energy import (
    CUTOFF_COEFFICIENT,
    CUTOFF_MODE,
    DENSITY_BANDWIDTH_SCALE,
    DENSITY_METHOD,
    DENSITY_NUM_BINS,
    DENSITY_NUM_POINTS,
    LOCAL_COUNT_HALF_WIDTH,
    LOCAL_QUANTILE_BINS,
    N,
    NUM_SPACING_GRAPHS,
    REMOVE_EXACT_ZERO_EIGENVALUES,
    SEQUENCE_MODE,
    SMALL_SPACING_THRESHOLD,
    SPACING_GRAPH_SEED,
    SPACING_SEQUENCE_SEED,
    TAIL_DENSITY_QUANTILE_CUTOFF,
    TAIL_LOCAL_COUNT_HALF_WIDTH,
    W_MAX,
    W_MIN,
    ZERO_EIGENVALUE_ATOL,
    parameter_seed_offset,
    validate_quantile_bins,
)
from .empirical_spectral_density import estimate_empirical_spectral_density
from .fixed_degree_configuration_baseline import (
    MODEL_CHUNG_LU,
    MODEL_CONFIGURATION,
    TAU_CONVENTION_CCDF,
    TAU_CONVENTION_DENSITY,
    GraphSpectralResult,
    PairedConfigurationResult,
    generate_paired_configuration_ensemble_from_seeds,
    generate_paired_configuration_realization,
)
from .regular_baseline import adjacent_gap_ratios_from_spacings, wigner_surmise_goe
from .regular_baseline_experiments import poisson_spacing_density


GRAPH_SEED_BASE = SPACING_GRAPH_SEED
SEQUENCE_SEED_BASE = SPACING_SEQUENCE_SEED
REWIRE_SEED_BASE = 17320
DEFAULT_NUM_GRAPHS = NUM_SPACING_GRAPHS
DEFAULT_TAU = 1.5
DEFAULT_TARGET_MEAN_DEGREE = 20.0
COMPONENT_MODE = "full"
SWAPS_PER_EDGE = 10
MAX_TRIES_FACTOR = 20
SPACING_NORMALIZATION = "rank_gap_pooled_mean_separately_by_model"
SPACING_WINDOW_MODE = "rank_gap_nonoverlapping_quantile_bins"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "configuration_model" / "paired_chung_lu"

LOCALIZATION_METRICS = [
    "mean_ipr",
    "median_ipr",
    "mean_n_times_ipr",
    "median_n_times_ipr",
    "mean_participation_ratio",
    "mean_participation_fraction",
    "median_max_coordinate_squared",
    "q90_max_coordinate_squared",
    "mean_top_5_coordinate_mass",
    "mean_top_10_coordinate_mass",
]
DIFFERENCE_METRICS = [
    "graph_mean_gap_ratio",
    "mean_n_times_ipr",
    "mean_participation_fraction",
    "mean_top_10_coordinate_mass",
    "median_max_coordinate_squared",
]


@dataclass(frozen=True)
class PairedTailConfig:
    n: int = N
    tau: float = DEFAULT_TAU
    target_mean_degree: float = DEFAULT_TARGET_MEAN_DEGREE
    num_graphs: int = DEFAULT_NUM_GRAPHS
    component_mode: str = COMPONENT_MODE
    sequence_mode: str = SEQUENCE_MODE
    w_min: float = W_MIN
    w_max: float | None = W_MAX
    cutoff_mode: str = CUTOFF_MODE
    cutoff_coefficient: float = CUTOFF_COEFFICIENT
    tau_convention: str = TAU_CONVENTION_DENSITY
    swaps_per_edge: int = SWAPS_PER_EDGE
    max_tries_factor: int = MAX_TRIES_FACTOR
    graph_seed_base: int = GRAPH_SEED_BASE
    sequence_seed_base: int = SEQUENCE_SEED_BASE
    rewire_seed_base: int = REWIRE_SEED_BASE
    quantile_bins: tuple[tuple[float, float], ...] = tuple(LOCAL_QUANTILE_BINS)


def make_seeds(num_graphs: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(value) for value in rng.integers(0, 2**32 - 1, size=num_graphs, dtype=np.uint32)]


def _slug_number(value: float) -> str:
    return str(value).replace(".", "p")


def parameter_output_dir(config: PairedTailConfig) -> Path:
    return OUTPUT_DIR / (
        f"n_{config.n}_tau_{_slug_number(config.tau)}_"
        f"avgdeg_{_slug_number(config.target_mean_degree)}"
    )


def code_version() -> str | None:
    repo = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def encode_array(values: np.ndarray) -> str:
    return " ".join(f"{value:.12g}" for value in np.asarray(values, dtype=float))


def decode_array(value: object) -> np.ndarray:
    text = str(value).strip()
    return np.fromstring(text, sep=" ", dtype=float) if text else np.array([], dtype=float)


def _finite_stats(values: Iterable[float]) -> tuple[float, float, int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return np.nan, np.nan, 0
    mean = float(np.mean(array))
    se = float(np.std(array, ddof=1) / np.sqrt(len(array))) if len(array) >= 2 else np.nan
    return mean, se, int(len(array))


def _continuous_bulk(result: GraphSpectralResult) -> tuple[np.ndarray, np.ndarray, int]:
    if result.bulk_eigenvectors is None:
        raise ValueError("bulk eigenvectors are required for the joint tail analysis")
    values = np.asarray(result.bulk_eigenvalues, dtype=float)
    zero_mask = np.isclose(values, 0.0, atol=ZERO_EIGENVALUE_ATOL, rtol=0.0)
    keep = ~zero_mask if REMOVE_EXACT_ZERO_EIGENVALUES else np.ones(len(values), dtype=bool)
    return values[keep], np.asarray(result.bulk_eigenvectors[:, keep], dtype=float), int(np.sum(zero_mask))


def _rank_gap_indices(num_gaps: int, q_low: float, q_high: float) -> tuple[int, int]:
    start = int(np.floor(q_low * num_gaps))
    stop = int(np.ceil(q_high * num_gaps))
    start = max(0, min(start, num_gaps - 1))
    stop = max(start + 1, min(stop, num_gaps))
    return start, stop


def graph_variant_tail_rows(
    result: GraphSpectralResult,
    pair: PairedConfigurationResult,
    quantile_bins: tuple[tuple[float, float], ...],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    values, vectors, zero_count = _continuous_bulk(result)
    if len(values) < 3:
        raise ValueError("continuous bulk must contain at least three eigenvalues")
    all_gaps = np.diff(values)
    rows: list[dict[str, object]] = []
    decay_rows: list[dict[str, object]] = []
    for q_low, q_high in quantile_bins:
        start, stop = _rank_gap_indices(len(all_gaps), q_low, q_high)
        candidate_indices = np.arange(start, stop, dtype=int)
        candidate_gaps = all_gaps[candidate_indices]
        valid = np.isfinite(candidate_gaps) & (candidate_gaps > 0)
        gap_indices = candidate_indices[valid]
        raw_gaps = candidate_gaps[valid]
        ratios = adjacent_gap_ratios_from_spacings(raw_gaps)
        eigenvalue_indices = np.unique(np.concatenate((gap_indices, gap_indices + 1)))
        selected_values = values[eigenvalue_indices]
        selected_vectors = vectors[:, eigenvalue_indices]
        localization = eigenvector_localization_summary(
            selected_vectors,
            analyzed_dimension=result.analyzed_dimension,
        )
        rows.append(
            {
                "pair_index": result.graph_index,
                "model_variant": result.model_variant,
                "graph_seed": result.graph_seed,
                "sequence_seed": result.sequence_seed,
                "rewire_seed": result.rewire_seed,
                "tau": result.tau,
                "tau_convention": result.tau_convention,
                "target_mean_degree": result.target_mean_degree,
                "quantile_low": q_low,
                "quantile_high": q_high,
                "quantile": 0.5 * (q_low + q_high),
                "E_low": float(selected_values[0]) if len(selected_values) else np.nan,
                "E_high": float(selected_values[-1]) if len(selected_values) else np.nan,
                "E_center": float(np.median(selected_values)) if len(selected_values) else np.nan,
                "index_start": start,
                "index_stop": stop,
                "number_of_eigenvalues": int(len(eigenvalue_indices)),
                "number_of_gaps": int(len(raw_gaps)),
                "number_of_gap_ratios": int(len(ratios)),
                "graph_mean_gap_ratio": float(np.mean(ratios)) if len(ratios) else np.nan,
                "analyzed_dimension": result.analyzed_dimension,
                "num_components": result.num_components,
                "largest_component_size": result.largest_component_size,
                "num_isolated_vertices": result.num_isolated_vertices,
                "zero_eigenvalue_count": zero_count,
                "realized_mean_degree": result.realized_mean_degree,
                "max_realized_degree": result.max_realized_degree,
                "edge_overlap_fraction": pair.edge_overlap_fraction,
                "requested_swaps": pair.requested_swaps,
                "raw_gaps": encode_array(raw_gaps),
                "gap_ratios": encode_array(ratios),
                **localization,
            }
        )
        for decay in coordinate_decay_profile_rows(
            selected_vectors,
            max_rank=COORDINATE_DECAY_MAX_RANK,
        ):
            decay_rows.append(
                {
                    "pair_index": result.graph_index,
                    "model_variant": result.model_variant,
                    "graph_seed": result.graph_seed,
                    "sequence_seed": result.sequence_seed,
                    "rewire_seed": result.rewire_seed,
                    "tau": result.tau,
                    "target_mean_degree": result.target_mean_degree,
                    "window_mode": SPACING_WINDOW_MODE,
                    "quantile": 0.5 * (q_low + q_high),
                    "quantile_low": q_low,
                    "quantile_high": q_high,
                    "index_start": start,
                    "index_stop": stop,
                    "number_of_eigenvectors": int(len(eigenvalue_indices)),
                    "E_center": float(np.median(selected_values)) if len(selected_values) else np.nan,
                    "analyzed_dimension": result.analyzed_dimension,
                    **decay,
                }
            )
    return rows, decay_rows


def aggregate_coordinate_decay_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            str(row["model_variant"]),
            float(row["quantile"]),
            int(float(row["rank"])),
        )
        grouped.setdefault(key, []).append(row)
    summaries: list[dict[str, object]] = []
    for (variant, quantile, rank), group in sorted(grouped.items()):
        first = group[0]
        weights = np.asarray([float(row["number_of_eigenvectors"]) for row in group], dtype=float)
        if float(np.sum(weights)) <= 0.0:
            weights = np.ones(len(group), dtype=float)
        summaries.append(
            {
                "model_variant": variant,
                "tau": first["tau"],
                "target_mean_degree": first["target_mean_degree"],
                "window_mode": first["window_mode"],
                "quantile": quantile,
                "quantile_low": first["quantile_low"],
                "quantile_high": first["quantile_high"],
                "rank": rank,
                "number_of_graphs": len(group),
                "number_of_eigenvectors": int(np.sum(weights)),
                "mean_E_center": float(np.average([float(row["E_center"]) for row in group], weights=weights)),
                "mean_coordinate_mass": float(np.average([float(row["mean_coordinate_mass"]) for row in group], weights=weights)),
                "mean_cumulative_mass": float(np.average([float(row["mean_cumulative_mass"]) for row in group], weights=weights)),
            }
        )
    return summaries


def aggregate_joint_rows(rows: list[dict[str, object]], rng_seed: int = 8675309) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float, float], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["model_variant"]), float(row["quantile_low"]), float(row["quantile_high"]))
        grouped.setdefault(key, []).append(row)
    rng = np.random.default_rng(rng_seed)
    summaries: list[dict[str, object]] = []
    for (variant, q_low, q_high), group in sorted(grouped.items()):
        raw_gap_parts = [decode_array(row["raw_gaps"]) for row in group]
        raw_gaps = np.concatenate([part for part in raw_gap_parts if len(part)])
        normalized = raw_gaps / float(np.mean(raw_gaps)) if len(raw_gaps) else np.array([], dtype=float)
        ratio_parts = [decode_array(row["gap_ratios"]) for row in group]
        ratios = np.concatenate([part for part in ratio_parts if len(part)]) if any(len(part) for part in ratio_parts) else np.array([], dtype=float)
        graph_ratio_mean, graph_ratio_se, graphs_with_ratios = _finite_stats(
            float(row["graph_mean_gap_ratio"]) for row in group
        )
        sample_size = max(4096, len(normalized))
        goe_sample = rng.rayleigh(scale=np.sqrt(2.0 / np.pi), size=sample_size)
        poisson_sample = rng.exponential(scale=1.0, size=sample_size)
        summary: dict[str, object] = {
            "model_variant": variant,
            "quantile_low": q_low,
            "quantile_high": q_high,
            "quantile": 0.5 * (q_low + q_high),
            "number_of_graphs": len(group),
            "graphs_with_gaps": int(sum(int(row["number_of_gaps"]) > 0 for row in group)),
            "graphs_with_gap_ratios": graphs_with_ratios,
            "number_of_eigenvalues": int(sum(int(row["number_of_eigenvalues"]) for row in group)),
            "number_of_gaps": int(len(raw_gaps)),
            "number_of_gap_ratios": int(len(ratios)),
            "mean_normalized_spacing": float(np.mean(normalized)) if len(normalized) else np.nan,
            "pooled_mean_gap_ratio": float(np.mean(ratios)) if len(ratios) else np.nan,
            "graph_equal_mean_gap_ratio": graph_ratio_mean,
            "gap_ratio_standard_error": graph_ratio_se,
            "small_gap_probability": float(np.mean(normalized < SMALL_SPACING_THRESHOLD)) if len(normalized) else np.nan,
            "KS_to_GOE": float(ks_2samp(normalized, goe_sample).statistic) if len(normalized) else np.nan,
            "KS_to_Poisson": float(ks_2samp(normalized, poisson_sample).statistic) if len(normalized) else np.nan,
            "mean_E_low": float(np.mean([float(row["E_low"]) for row in group])),
            "mean_E_high": float(np.mean([float(row["E_high"]) for row in group])),
            "spacing_window_mode": SPACING_WINDOW_MODE,
            "spacing_normalization": SPACING_NORMALIZATION,
            "normalized_spacings": encode_array(normalized),
        }
        for metric in LOCALIZATION_METRICS:
            mean, se, count = _finite_stats(float(row[metric]) for row in group)
            summary[metric] = mean
            summary[f"{metric}_standard_error"] = se
            summary[f"{metric}_graph_count"] = count
        summaries.append(summary)
    return summaries


def paired_difference_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, float, float], dict[str, dict[str, object]]] = {}
    for row in rows:
        key = (int(row["pair_index"]), float(row["quantile_low"]), float(row["quantile_high"]))
        grouped.setdefault(key, {})[str(row["model_variant"])] = row
    differences: list[dict[str, object]] = []
    for (pair_index, q_low, q_high), variants in sorted(grouped.items()):
        if MODEL_CHUNG_LU not in variants or MODEL_CONFIGURATION not in variants:
            continue
        original = variants[MODEL_CHUNG_LU]
        configuration = variants[MODEL_CONFIGURATION]
        row: dict[str, object] = {
            "pair_index": pair_index,
            "graph_seed": original["graph_seed"],
            "sequence_seed": original["sequence_seed"],
            "rewire_seed": original["rewire_seed"],
            "quantile_low": q_low,
            "quantile_high": q_high,
            "quantile": 0.5 * (q_low + q_high),
            "edge_overlap_fraction": original["edge_overlap_fraction"],
        }
        for metric in DIFFERENCE_METRICS:
            left = float(original[metric])
            right = float(configuration[metric])
            row[f"delta_{metric}"] = right - left if np.isfinite(left) and np.isfinite(right) else np.nan
        differences.append(row)
    return differences


def aggregate_difference_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[float, float], list[dict[str, object]]] = {}
    for row in rows:
        key = (float(row["quantile_low"]), float(row["quantile_high"]))
        grouped.setdefault(key, []).append(row)
    summaries: list[dict[str, object]] = []
    for (q_low, q_high), group in sorted(grouped.items()):
        summary: dict[str, object] = {
            "quantile_low": q_low,
            "quantile_high": q_high,
            "quantile": 0.5 * (q_low + q_high),
            "number_of_pairs": len(group),
        }
        for metric in DIFFERENCE_METRICS:
            mean, se, count = _finite_stats(float(row[f"delta_{metric}"]) for row in group)
            summary[f"mean_delta_{metric}"] = mean
            summary[f"se_delta_{metric}"] = se
            summary[f"valid_pairs_{metric}"] = count
        summaries.append(summary)
    return summaries


def degree_check_row(pair: PairedConfigurationResult) -> dict[str, object]:
    original = pair.original
    configuration = pair.configuration
    return {
        "pair_index": original.graph_index,
        "graph_seed": original.graph_seed,
        "sequence_seed": original.sequence_seed,
        "rewire_seed": original.rewire_seed,
        "same_node_labels": bool(np.array_equal(original.node_labels, configuration.node_labels)),
        "same_number_of_vertices": original.n_requested == configuration.n_requested,
        "same_number_of_edges": original.num_edges == configuration.num_edges,
        "number_of_edges": original.num_edges,
        "degree_match": pair.degree_match,
        "requested_swaps": pair.requested_swaps,
        "swaps_per_edge": pair.swaps_per_edge,
        "max_tries": pair.max_tries,
        "edge_overlap_fraction": pair.edge_overlap_fraction,
        "original_num_components": original.num_components,
        "configuration_num_components": configuration.num_components,
        "original_max_degree": original.max_realized_degree,
        "configuration_max_degree": configuration.max_realized_degree,
    }


def write_csv(rows: list[dict[str, object]], path: Path, omit: set[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    omit = omit or set()
    fieldnames = [key for key in rows[0] if key not in omit]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_spacing_histograms(
    summaries: list[dict[str, object]], model_variant: str, title: str
) -> plt.Figure:
    selected = sorted(
        (row for row in summaries if row["model_variant"] == model_variant),
        key=lambda row: float(row["quantile"]),
    )
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), constrained_layout=True)
    s_grid = np.linspace(0.0, 4.0, 400)
    for ax, row in zip(axes.flat, selected):
        spacings = decode_array(row["normalized_spacings"])
        ax.hist(spacings, bins=30, density=True, alpha=0.55, color="steelblue")
        ax.plot(s_grid, wigner_surmise_goe(s_grid), color="black", lw=1.5, label="GOE")
        ax.plot(s_grid, poisson_spacing_density(s_grid), color="firebrick", ls="--", lw=1.5, label="Poisson")
        ax.set_title(f"q=[{row['quantile_low']:.4f}, {row['quantile_high']:.4f}]\n{row['number_of_gaps']} gaps")
        ax.set_xlim(0.0, 4.0)
        ax.set_xlabel("pooled-mean normalized gap")
    axes.flat[0].legend(frameon=False)
    fig.suptitle(title, y=1.02)
    return fig


def plot_model_spacing_grid(
    rows: list[dict[str, object]],
    model_variant: str,
    config: PairedTailConfig,
) -> plt.Figure:
    selected = _summary_by_variant(rows, model_variant)
    spacing_samples = [decode_array(row["normalized_spacings"]) for row in selected]
    nonempty = [values for values in spacing_samples if len(values)]
    if not nonempty:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.text(0.5, 0.5, "No valid local spacing samples", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig
    all_spacings = np.concatenate(nonempty)
    hist_max = max(3.5, float(np.quantile(all_spacings, 0.99)))
    bin_edges = np.linspace(0.0, hist_max, 31)
    s_grid = np.linspace(0.0, hist_max, 400)
    fig, axes = plt.subplots(2, 4, figsize=(17.2, 7.4), sharex=True, sharey=True, constrained_layout=True)
    for ax, row, spacings in zip(axes.flat, selected, spacing_samples):
        ax.hist(spacings, bins=bin_edges, density=True, alpha=0.75, color="mediumseagreen", edgecolor="black", label="empirical")
        ax.plot(s_grid, wigner_surmise_goe(s_grid), color="black", lw=2, label="GOE")
        ax.plot(s_grid, poisson_spacing_density(s_grid), color="firebrick", lw=2, ls="--", label="Poisson")
        ax.set_title(
            f"q=[{row['quantile_low']:.4f},{row['quantile_high']:.4f}]\n"
            f"mean E=[{row['mean_E_low']:.2f},{row['mean_E_high']:.2f}] "
            f"n_gaps={row['number_of_gaps']}"
        )
        ax.set_xlabel("s")
        ax.set_ylabel("density")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(
        f"{model_variant} Rank-Gap Tail Histograms, n={config.n}, tau={config.tau:.2f}, "
        f"mean degree={config.target_mean_degree:.1f}",
        y=1.04,
    )
    return fig


def plot_model_spacing_summary(
    rows: list[dict[str, object]],
    model_variant: str,
    config: PairedTailConfig,
) -> plt.Figure:
    selected = _summary_by_variant(rows, model_variant)
    quantiles = np.asarray([float(row["quantile"]) for row in selected], dtype=float)
    mean_r = np.asarray([float(row["graph_equal_mean_gap_ratio"]) for row in selected], dtype=float)
    mean_r_se = np.asarray([float(row["gap_ratio_standard_error"]) for row in selected], dtype=float)
    small_gap = np.asarray([float(row["small_gap_probability"]) for row in selected], dtype=float)
    gap_counts = np.asarray([int(row["number_of_gaps"]) for row in selected], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].plot(quantiles, mean_r, marker="o", color="black")
    axes[0].fill_between(quantiles, mean_r - mean_r_se, mean_r + mean_r_se, color="gray", alpha=0.2)
    axes[0].axhline(0.53590, color="black", ls="--", label="GOE mean r")
    axes[0].axhline(0.38629, color="firebrick", ls="--", label="Poisson mean r")
    axes[0].set_title("Gap Ratio vs Quantile Window")
    axes[0].set_ylabel("graph-equal mean r")
    axes[0].legend(frameon=False)
    axes[1].plot(quantiles, small_gap, marker="o", color="seagreen")
    axes[1].set_title(f"P(s < {SMALL_SPACING_THRESHOLD:.2f})")
    axes[1].set_ylabel("probability")
    axes[2].plot(quantiles, gap_counts, marker="o", color="steelblue")
    axes[2].set_title("Sample Sizes")
    axes[2].set_ylabel("number of gaps")
    for ax in axes:
        ax.set_xlabel("quantile midpoint")
        ax.grid(alpha=0.2)
    fig.suptitle(
        f"{model_variant} Rank-Gap Tail Summary, n={config.n}, tau={config.tau:.2f}, "
        f"mean degree={config.target_mean_degree:.1f}",
        y=1.05,
    )
    return fig


def plot_model_coordinate_decay_summary(
    rows: list[dict[str, object]],
    model_variant: str,
    config: PairedTailConfig,
) -> plt.Figure:
    selected = [row for row in rows if row["model_variant"] == model_variant]
    quantiles = sorted({float(row["quantile"]) for row in selected})
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(quantiles)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for color, quantile in zip(colors, quantiles):
        q_rows = sorted(
            (row for row in selected if float(row["quantile"]) == quantile),
            key=lambda row: int(row["rank"]),
        )
        ranks = np.asarray([int(row["rank"]) for row in q_rows], dtype=float)
        mass = np.asarray([float(row["mean_coordinate_mass"]) for row in q_rows], dtype=float)
        cumulative = np.asarray([float(row["mean_cumulative_mass"]) for row in q_rows], dtype=float)
        label = f"q={quantile:.4f}"
        axes[0].plot(ranks, mass, color=color, lw=1.8, label=label)
        axes[1].plot(ranks, cumulative, color=color, lw=1.8, label=label)
    axes[0].axhline(1.0 / max(float(config.n), 1.0), color="0.55", lw=1.0, ls="--", label="1 / n")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("coordinate rank after sorting |v_i|^2")
    axes[0].set_ylabel("mean sorted coordinate mass")
    axes[0].set_title("Sorted Coordinate Decay")
    axes[1].set_xlabel("top-r coordinate count")
    axes[1].set_ylabel("mean cumulative mass")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_title("Cumulative Top-r Mass")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.08, 0.5), frameon=False)
    fig.suptitle(
        f"{model_variant} Eigenvector Coordinate Profiles, n={config.n}, tau={config.tau:.2f}, "
        f"mean degree={config.target_mean_degree:.1f}",
        y=1.04,
    )
    return fig


def plot_global_density_comparison(
    eigenvalue_samples: dict[str, list[np.ndarray]], config: PairedTailConfig
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    colors = {MODEL_CHUNG_LU: "black", MODEL_CONFIGURATION: "darkorange"}
    for variant in (MODEL_CHUNG_LU, MODEL_CONFIGURATION):
        samples = eigenvalue_samples[variant]
        density = estimate_empirical_spectral_density(
            eigenvalue_samples=samples,
            num_graphs=len(samples),
            per_graph_dimension=float(np.mean([len(values) for values in samples])),
            method=DENSITY_METHOD,
            num_points=DENSITY_NUM_POINTS,
            num_bins=DENSITY_NUM_BINS,
            bandwidth_scale=DENSITY_BANDWIDTH_SCALE,
            local_count_half_width=LOCAL_COUNT_HALF_WIDTH,
            tail_local_count_half_width=TAIL_LOCAL_COUNT_HALF_WIDTH,
            tail_quantile_cutoff=TAIL_DENSITY_QUANTILE_CUTOFF,
        )
        ax.plot(density.grid, density.density, lw=1.8, color=colors[variant], label=variant)
    ax.set_xlabel("continuous bulk eigenvalue")
    ax.set_ylabel("empirical density")
    ax.set_title(
        f"Paired global density, n={config.n}, tau={config.tau:g}, "
        f"mean degree={config.target_mean_degree:g}"
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    return fig


def _summary_by_variant(
    summaries: list[dict[str, object]], variant: str
) -> list[dict[str, object]]:
    return sorted(
        (row for row in summaries if row["model_variant"] == variant),
        key=lambda row: float(row["quantile"]),
    )


def plot_paired_summary(
    summaries: list[dict[str, object]], differences: list[dict[str, object]], config: PairedTailConfig
) -> plt.Figure:
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 12.0), sharex=True, constrained_layout=True)
    styles = {MODEL_CHUNG_LU: ("black", "o"), MODEL_CONFIGURATION: ("darkorange", "s")}
    for variant in (MODEL_CHUNG_LU, MODEL_CONFIGURATION):
        rows = _summary_by_variant(summaries, variant)
        q = np.asarray([row["quantile"] for row in rows], dtype=float)
        color, marker = styles[variant]
        axes[0, 0].errorbar(q, [row["graph_equal_mean_gap_ratio"] for row in rows], yerr=[row["gap_ratio_standard_error"] for row in rows], color=color, marker=marker, label=variant)
        axes[1, 0].errorbar(q, [row["mean_n_times_ipr"] for row in rows], yerr=[row["mean_n_times_ipr_standard_error"] for row in rows], color=color, marker=marker, label=variant)
        axes[2, 0].errorbar(q, [row["mean_top_10_coordinate_mass"] for row in rows], yerr=[row["mean_top_10_coordinate_mass_standard_error"] for row in rows], color=color, marker=marker, label=variant)
        axes[2, 1].plot(q, [row["number_of_gaps"] for row in rows], color=color, marker=marker, label=f"{variant} gaps")
    q = np.asarray([row["quantile"] for row in differences], dtype=float)
    axes[0, 1].errorbar(q, [row["mean_delta_graph_mean_gap_ratio"] for row in differences], yerr=[row["se_delta_graph_mean_gap_ratio"] for row in differences], color="seagreen", marker="o")
    axes[1, 1].errorbar(q, [row["mean_delta_mean_n_times_ipr"] for row in differences], yerr=[row["se_delta_mean_n_times_ipr"] for row in differences], color="steelblue", marker="o")
    axes[2, 1].plot(q, [row["valid_pairs_graph_mean_gap_ratio"] for row in differences], color="purple", marker="o", label="pairs with gap ratio")
    axes[0, 0].axhline(0.53590, color="0.4", ls="--", lw=1, label="GOE")
    axes[0, 0].axhline(0.38629, color="firebrick", ls="--", lw=1, label="Poisson")
    axes[0, 1].axhline(0.0, color="0.5", lw=1)
    axes[1, 1].axhline(0.0, color="0.5", lw=1)
    titles = [
        "Mean adjacent gap ratio", "Paired difference in mean gap ratio",
        "Mean n * IPR", "Paired difference in n * IPR",
        "Mean top-10 coordinate mass", "Usable pairs and pooled gaps",
    ]
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title)
        ax.set_xlabel("spectral quantile midpoint")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=8)
    axes[2, 1].legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"Chung-Lu vs fixed-degree control, n={config.n}, tau={config.tau:g}, "
        f"mean degree={config.target_mean_degree:g}\n"
        "Differences are fixed-degree configuration minus Chung-Lu",
        y=1.04,
    )
    return fig


def run_manifest(
    config: PairedTailConfig,
    checks: list[dict[str, object]],
    weight_diagnostics: dict[str, object],
) -> dict[str, object]:
    overlaps = np.asarray([float(row["edge_overlap_fraction"]) for row in checks], dtype=float)
    return {
        "model": "fixed_degree_configuration_model",
        "comparison": "paired_with_realized_chung_lu",
        "implementation": "degree_preserving_double_edge_swap",
        "sampling_caveat": "finite double-edge-swap chains are not claimed to be provably uniform",
        "n": config.n,
        "tau": config.tau,
        "tau_convention": config.tau_convention,
        "target_mean_degree": config.target_mean_degree,
        "sequence_mode": config.sequence_mode,
        "cutoff_mode": config.cutoff_mode,
        "cutoff_coefficient": config.cutoff_coefficient,
        "w_min": config.w_min,
        "w_max": config.w_max,
        "component_mode": config.component_mode,
        "zero_removal_rule": "remove values isclose to zero" if REMOVE_EXACT_ZERO_EIGENVALUES else "retain zeros",
        "zero_eigenvalue_atol": ZERO_EIGENVALUE_ATOL,
        "quantile_bins": config.quantile_bins,
        "spacing_window_mode": SPACING_WINDOW_MODE,
        "spacing_normalization": SPACING_NORMALIZATION,
        "swaps_per_edge": config.swaps_per_edge,
        "max_tries_factor": config.max_tries_factor,
        "graph_seed_base": config.graph_seed_base,
        "sequence_seed_base": config.sequence_seed_base,
        "rewire_seed_base": config.rewire_seed_base,
        "number_of_graphs": config.num_graphs,
        "mean_edge_overlap_fraction": float(np.mean(overlaps)),
        "minimum_edge_overlap_fraction": float(np.min(overlaps)),
        "maximum_edge_overlap_fraction": float(np.max(overlaps)),
        "all_labeled_degrees_preserved": bool(all(bool(row["degree_match"]) for row in checks)),
        "weight_diagnostics": weight_diagnostics,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_commit": code_version(),
    }


def _write_aggregated_outputs(
    *,
    config: PairedTailConfig,
    rows: list[dict[str, object]],
    decay_rows: list[dict[str, object]],
    checks: list[dict[str, object]],
    density_samples: dict[str, list[np.ndarray]],
    weight_diagnostics: dict[str, object],
    destination: Path,
) -> Path:
    summaries = aggregate_joint_rows(rows)
    decay_summaries = aggregate_coordinate_decay_rows(decay_rows)
    differences = paired_difference_rows(rows)
    difference_summaries = aggregate_difference_rows(differences)
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(rows, destination / "paired_joint_tail_by_graph.csv")
    write_csv(summaries, destination / "paired_joint_tail_summary.csv", omit={"normalized_spacings"})
    write_csv(differences, destination / "paired_differences_by_graph.csv")
    write_csv(difference_summaries, destination / "paired_differences_summary.csv")
    write_csv(checks, destination / "degree_preservation_checks.csv")
    for variant, prefix in ((MODEL_CHUNG_LU, "chung_lu"), (MODEL_CONFIGURATION, "configuration")):
        write_csv(
            [row for row in summaries if row["model_variant"] == variant],
            destination / f"{prefix}_spacing_summary_rank_gap.csv",
            omit={"normalized_spacings"},
        )
        write_csv(
            [row for row in decay_summaries if row["model_variant"] == variant],
            destination / f"{prefix}_eigenvector_coordinate_decay_summary.csv",
        )
    figures = [
        (plot_global_density_comparison(density_samples, config), "global_density_comparison.png"),
        (plot_spacing_histograms(summaries, MODEL_CHUNG_LU, "Chung-Lu rank-gap tail histograms"), "chung_lu_rank_gap_histograms.png"),
        (plot_spacing_histograms(summaries, MODEL_CONFIGURATION, "Fixed-degree configuration rank-gap tail histograms"), "configuration_rank_gap_histograms.png"),
        (plot_paired_summary(summaries, difference_summaries, config), "paired_spacing_localization_summary.png"),
    ]
    for variant, prefix in ((MODEL_CHUNG_LU, "chung_lu"), (MODEL_CONFIGURATION, "configuration")):
        figures.extend(
            [
                (plot_model_spacing_grid(summaries, variant, config), f"{prefix}_local_spacing_grid_quantiles_rank_gap.png"),
                (plot_model_spacing_summary(summaries, variant, config), f"{prefix}_local_spacing_summary_quantiles_rank_gap.png"),
                (plot_model_coordinate_decay_summary(decay_summaries, variant, config), f"{prefix}_eigenvector_coordinate_decay_summary.png"),
            ]
        )
    for figure, name in figures:
        figure.savefig(destination / name, dpi=200, bbox_inches="tight")
        plt.close(figure)
    with (destination / "run_manifest.json").open("w") as handle:
        json.dump(run_manifest(config, checks, weight_diagnostics), handle, indent=2, sort_keys=True)
    print(f"saved paired configuration-model comparison to {destination}")
    return destination


def _seed_lists(config: PairedTailConfig) -> tuple[list[int], list[int], list[int]]:
    offset = parameter_seed_offset(config.tau, config.target_mean_degree)
    return (
        make_seeds(config.num_graphs, config.graph_seed_base + offset),
        make_seeds(config.num_graphs, config.sequence_seed_base + offset),
        make_seeds(config.num_graphs, config.rewire_seed_base + offset),
    )


def run_pair_partial(config: PairedTailConfig, pair_index: int, partial_dir: Path) -> Path:
    if not 0 <= pair_index < config.num_graphs:
        raise ValueError(f"pair_index must be in [0, {config.num_graphs})")
    graph_seeds, sequence_seeds, rewire_seeds = _seed_lists(config)
    pair = generate_paired_configuration_realization(
        n=config.n,
        tau=config.tau,
        target_mean_degree=config.target_mean_degree,
        w_min=config.w_min,
        w_max=config.w_max,
        cutoff_mode=config.cutoff_mode,
        cutoff_coefficient=config.cutoff_coefficient,
        sequence_mode=config.sequence_mode,
        graph_seed=graph_seeds[pair_index],
        sequence_seed=sequence_seeds[pair_index],
        rewire_seed=rewire_seeds[pair_index],
        component_mode=config.component_mode,
        graph_index=pair_index,
        swaps_per_edge=config.swaps_per_edge,
        max_tries_factor=config.max_tries_factor,
        tau_convention=config.tau_convention,
        retain_eigenvectors=True,
    )
    rows: list[dict[str, object]] = []
    decay_rows: list[dict[str, object]] = []
    continuous: dict[str, np.ndarray] = {}
    for result in (pair.original, pair.configuration):
        result_rows, result_decay_rows = graph_variant_tail_rows(result, pair, config.quantile_bins)
        rows.extend(result_rows)
        decay_rows.extend(result_decay_rows)
        values, _, _ = _continuous_bulk(result)
        continuous[result.model_variant] = values
    partial_dir.mkdir(parents=True, exist_ok=True)
    path = partial_dir / f"pair_{pair_index:06d}.npz"
    np.savez_compressed(
        path,
        rows_json=np.asarray(json.dumps(rows)),
        decay_rows_json=np.asarray(json.dumps(decay_rows)),
        check_json=np.asarray(json.dumps(degree_check_row(pair))),
        weight_diagnostics_json=np.asarray(json.dumps(asdict(pair.weight_diagnostics))),
        chung_lu_bulk=continuous[MODEL_CHUNG_LU],
        configuration_bulk=continuous[MODEL_CONFIGURATION],
    )
    print(f"wrote pair {pair_index} partial to {path}")
    return path


def aggregate_pair_partials(
    config: PairedTailConfig,
    partial_dir: Path,
    output_dir: Path | None = None,
) -> Path:
    paths = [partial_dir / f"pair_{index:06d}.npz" for index in range(config.num_graphs)]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} pair partials; first missing file is {missing[0]}")
    rows: list[dict[str, object]] = []
    decay_rows: list[dict[str, object]] = []
    checks: list[dict[str, object]] = []
    density_samples = {MODEL_CHUNG_LU: [], MODEL_CONFIGURATION: []}
    weight_diagnostics: dict[str, object] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            rows.extend(json.loads(str(data["rows_json"].item())))
            decay_rows.extend(json.loads(str(data["decay_rows_json"].item())))
            checks.append(json.loads(str(data["check_json"].item())))
            if not weight_diagnostics:
                weight_diagnostics = json.loads(str(data["weight_diagnostics_json"].item()))
            density_samples[MODEL_CHUNG_LU].append(np.asarray(data["chung_lu_bulk"], dtype=float))
            density_samples[MODEL_CONFIGURATION].append(np.asarray(data["configuration_bulk"], dtype=float))
    return _write_aggregated_outputs(
        config=config,
        rows=rows,
        decay_rows=decay_rows,
        checks=checks,
        density_samples=density_samples,
        weight_diagnostics=weight_diagnostics,
        destination=output_dir or parameter_output_dir(config),
    )


def run_paired_experiment(config: PairedTailConfig, output_dir: Path | None = None) -> Path:
    validate_quantile_bins(list(config.quantile_bins))
    if config.num_graphs < 1:
        raise ValueError("num_graphs must be positive")
    graph_seeds, sequence_seeds, rewire_seeds = _seed_lists(config)
    rows: list[dict[str, object]] = []
    decay_rows: list[dict[str, object]] = []
    checks: list[dict[str, object]] = []
    density_samples = {MODEL_CHUNG_LU: [], MODEL_CONFIGURATION: []}
    first_weight_diagnostics: dict[str, object] | None = None
    ensemble = generate_paired_configuration_ensemble_from_seeds(
        n=config.n,
        tau=config.tau,
        target_mean_degree=config.target_mean_degree,
        w_min=config.w_min,
        w_max=config.w_max,
        cutoff_mode=config.cutoff_mode,
        cutoff_coefficient=config.cutoff_coefficient,
        sequence_mode=config.sequence_mode,
        graph_seeds=graph_seeds,
        sequence_seeds=sequence_seeds,
        rewire_seeds=rewire_seeds,
        component_mode=config.component_mode,
        swaps_per_edge=config.swaps_per_edge,
        max_tries_factor=config.max_tries_factor,
        tau_convention=config.tau_convention,
        retain_eigenvectors=True,
    )
    for graph_number, pair in enumerate(ensemble, start=1):
        if first_weight_diagnostics is None:
            first_weight_diagnostics = asdict(pair.weight_diagnostics)
        checks.append(degree_check_row(pair))
        for result in (pair.original, pair.configuration):
            result_rows, result_decay_rows = graph_variant_tail_rows(result, pair, config.quantile_bins)
            rows.extend(result_rows)
            decay_rows.extend(result_decay_rows)
            values, _, _ = _continuous_bulk(result)
            density_samples[result.model_variant].append(values.copy())
        print(
            f"finished pair {graph_number}/{config.num_graphs}; "
            f"edge overlap={pair.edge_overlap_fraction:.4f}",
            flush=True,
        )

    return _write_aggregated_outputs(
        config=config,
        rows=rows,
        decay_rows=decay_rows,
        checks=checks,
        density_samples=density_samples,
        weight_diagnostics=first_weight_diagnostics or {},
        destination=output_dir or parameter_output_dir(config),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=N)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--degree", type=float, default=DEFAULT_TARGET_MEAN_DEGREE)
    parser.add_argument("--num-graphs", type=int, default=DEFAULT_NUM_GRAPHS)
    parser.add_argument("--swaps-per-edge", type=int, default=SWAPS_PER_EDGE)
    parser.add_argument("--max-tries-factor", type=int, default=MAX_TRIES_FACTOR)
    parser.add_argument(
        "--tau-convention",
        choices=[TAU_CONVENTION_DENSITY, TAU_CONVENTION_CCDF],
        default=TAU_CONVENTION_DENSITY,
    )
    parser.add_argument("--cutoff-coefficient", type=float, default=CUTOFF_COEFFICIENT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--partial-dir", type=Path)
    parser.add_argument("--pair-index", type=int)
    parser.add_argument("--aggregate-partials", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PairedTailConfig(
        n=args.n,
        tau=args.tau,
        target_mean_degree=args.degree,
        num_graphs=args.num_graphs,
        swaps_per_edge=args.swaps_per_edge,
        max_tries_factor=args.max_tries_factor,
        tau_convention=args.tau_convention,
        cutoff_coefficient=args.cutoff_coefficient,
    )
    if args.aggregate_partials:
        if args.partial_dir is None:
            raise SystemExit("--aggregate-partials requires --partial-dir")
        aggregate_pair_partials(config, partial_dir=args.partial_dir, output_dir=args.output_dir)
    elif args.pair_index is not None:
        if args.partial_dir is None:
            raise SystemExit("--pair-index requires --partial-dir")
        run_pair_partial(config, pair_index=args.pair_index, partial_dir=args.partial_dir)
    else:
        run_paired_experiment(config, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
