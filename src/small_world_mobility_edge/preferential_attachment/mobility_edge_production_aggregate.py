"""Aggregate parameterized PA mobility-edge production caches."""

from __future__ import annotations

import csv
import gzip
import json
import math
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from ..chung_lu_degree_spectral_tail_match import (
    hill_ccdf_exponent_from_top,
    log_ccdf_regression,
    rank_match_fit,
)
from .full_spectrum_scan import (
    COORDINATE_DECAY_MAX_RANK,
    MASS_LEVELS,
    TOP_R_VALUES,
    MobilityEdgeProductionConfig,
    mobility_chunk_paths,
    validate_mobility_chunk,
)
from .mobility_edge_cached_pipeline import (
    fine_candidate_scans,
    histogram_rows,
    plot_coarse,
    slug,
    write_global_spectrum,
)
from .mobility_edge_postprocess import (
    ANALYSIS_VERSION,
    build_cross_fitted_models,
    build_gap_records,
    detect_candidate_regions,
    graph_level_quantile_rows,
    join_localization_and_level_statistics,
    save_crossfit_models,
    summarize_quantile_grid,
    write_rows,
)


LOCALIZATION_METRICS = (
    "ipr",
    "n_times_ipr",
    "participation_ratio",
    "participation_fraction",
    "top_10_mass",
    "top_50_mass",
    "top_100_mass",
    "top_200_mass",
    "r_50_fraction",
    "r_80_fraction",
    "r_90_fraction",
    "r_95_fraction",
)

HUB_EDGE_FIXED_RANKS = (1, 2, 3, 5, 10, 20)
HUB_EDGE_FIT_RANKS = (10, 20, 50)


def _mean_se(values: Sequence[float]) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if not len(data):
        return math.nan, math.nan
    return (
        float(np.mean(data)),
        float(np.std(data, ddof=1) / math.sqrt(len(data))) if len(data) > 1 else math.nan,
    )


def _read_gzip_csv(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _linear_fit(x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[valid], dtype=float)
    y = np.asarray(y[valid], dtype=float)
    if len(x) < 3 or float(np.std(x)) <= 0.0:
        return {"observations": int(len(x)), "slope": math.nan, "intercept": math.nan, "r2": math.nan}
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    residual = float(np.sum(np.square(y - fitted)))
    total = float(np.sum(np.square(y - np.mean(y))))
    return {
        "observations": int(len(x)),
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(1.0 - residual / total) if total > 0.0 else math.nan,
    }


def write_sublinear_stretched_exponential_diagnostic(
    config: MobilityEdgeProductionConfig,
    graphs: Sequence[dict[str, object]],
    ccdf_rows: Sequence[dict[str, object]],
    structural_dir: Path,
) -> dict[str, object]:
    """Check log CCDF versus k**(1-zeta) without imposing a tail law."""
    if config.model_family != "nonlinear" or config.canonical_parameter_value >= 1.0:
        return {}
    zeta = config.canonical_parameter_value
    transformed_rows: list[dict[str, object]] = []
    for row in ccdf_rows:
        degree = float(row["degree"])
        ccdf = float(row["mean_ccdf"])
        transformed_rows.append(
            {
                **row,
                "attachment_zeta": zeta,
                "k_to_one_minus_zeta": degree ** (1.0 - zeta),
                "log_mean_ccdf": math.log(ccdf) if ccdf > 0.0 else math.nan,
                "primary_diagnostic_range": bool(0.01 <= ccdf <= 0.20 and degree > config.m),
            }
        )
    write_rows(structural_dir / "sublinear_stretched_exponential_diagnostic.csv", transformed_rows)

    fits: list[dict[str, object]] = []
    for graph in graphs:
        degrees = np.asarray(graph["degrees"], dtype=float)
        maximum = int(np.max(degrees))
        k = np.arange(config.m + 1, maximum + 1, dtype=float)
        ccdf = np.asarray([np.mean(degrees >= value) for value in k], dtype=float)
        mask = (ccdf >= 0.01) & (ccdf <= 0.20)
        range_rule = "0.01<=CCDF<=0.20"
        if int(np.sum(mask)) < 10:
            mask = (ccdf >= 0.005) & (ccdf <= 0.40)
            range_rule = "fallback_0.005<=CCDF<=0.40"
        result = _linear_fit(k[mask] ** (1.0 - zeta), np.log(ccdf[mask]))
        fits.append(
            {
                "graph_index": graph["graph_index"],
                "graph_seed": graph["graph_seed"],
                "attachment_zeta": zeta,
                "range_rule": range_rule,
                "minimum_fitted_degree": float(np.min(k[mask])) if np.any(mask) else math.nan,
                "maximum_fitted_degree": float(np.max(k[mask])) if np.any(mask) else math.nan,
                **result,
            }
        )
    write_rows(structural_dir / "sublinear_stretched_exponential_fits_by_graph.csv", fits)
    summary: dict[str, object] = {
        "diagnostic": "log empirical degree CCDF versus k**(1-attachment_zeta)",
        "interpretation": "validation diagnostic only; not a forced distributional fit",
        "attachment_zeta": zeta,
    }
    for field in ("slope", "intercept", "r2", "observations"):
        mean, se = _mean_se([float(row[field]) for row in fits])
        summary[f"mean_{field}"] = mean
        summary[f"se_{field}"] = se

    x = np.asarray([float(row["k_to_one_minus_zeta"]) for row in transformed_rows])
    y = np.asarray([float(row["log_mean_ccdf"]) for row in transformed_rows])
    mask = np.asarray([bool(row["primary_diagnostic_range"]) for row in transformed_rows])
    figure, axis = plt.subplots(figsize=(7.4, 5.2))
    axis.plot(x, y, color="tab:blue", label="ensemble mean CCDF")
    if np.sum(mask) >= 3:
        fit = _linear_fit(x[mask], y[mask])
        axis.plot(
            x[mask],
            float(fit["intercept"]) + float(fit["slope"]) * x[mask],
            color="firebrick",
            ls="--",
            label=f"diagnostic fit, $R^2$={float(fit['r2']):.3f}",
        )
    axis.set_xlabel(r"$k^{1-\zeta}$")
    axis.set_ylabel(r"$\log\,\widehat P(D\geq k)$")
    axis.set_title(f"Sublinear PA stretched-exponential diagnostic, zeta={zeta:g}")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(structural_dir / "sublinear_stretched_exponential_ccdf.png", dpi=180)
    plt.close(figure)
    return summary


def write_nonlinear_hub_spectral_edge_diagnostic(
    config: MobilityEdgeProductionConfig,
    graphs: Sequence[dict[str, object]],
    structural_rows: Sequence[dict[str, object]],
    output_dir: Path,
) -> dict[str, object]:
    """Test, rather than assume, lambda proportional to sqrt(realized degree)."""
    if config.model_family != "nonlinear":
        return {}
    structural_by_index = {int(row["graph_index"]): row for row in structural_rows}
    fixed_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    for graph in graphs:
        graph_index = int(graph["graph_index"])
        degrees = np.sort(np.asarray(graph["degrees"], dtype=float))[::-1]
        spectrum = np.asarray(graph["nonperron_eigenvalues"], dtype=float)
        positive = np.sort(spectrum[spectrum > 0.0])[::-1]
        negative = np.sort(np.abs(spectrum[spectrum < 0.0]))[::-1]
        perron = float(structural_by_index[graph_index]["perron_eigenvalue"])
        fixed_rows.append(
            {
                "graph_index": graph_index,
                "graph_seed": graph["graph_seed"],
                "attachment_zeta": config.canonical_parameter_value,
                "spectral_side": "perron",
                "rank": 1,
                "degree": degrees[0],
                "absolute_eigenvalue": perron,
                "lambda_over_sqrt_degree": perron / math.sqrt(degrees[0]),
            }
        )
        for side, eigenvalues in (("positive_nonperron", positive), ("negative_absolute", negative)):
            for rank in HUB_EDGE_FIXED_RANKS:
                if rank <= len(degrees) and rank <= len(eigenvalues):
                    fixed_rows.append(
                        {
                            "graph_index": graph_index,
                            "graph_seed": graph["graph_seed"],
                            "attachment_zeta": config.canonical_parameter_value,
                            "spectral_side": side,
                            "rank": rank,
                            "degree": degrees[rank - 1],
                            "absolute_eigenvalue": eigenvalues[rank - 1],
                            "lambda_over_sqrt_degree": eigenvalues[rank - 1] / math.sqrt(degrees[rank - 1]),
                        }
                    )
            for top_count in HUB_EDGE_FIT_RANKS:
                fit_rows.append(
                    {
                        "graph_index": graph_index,
                        "graph_seed": graph["graph_seed"],
                        "attachment_zeta": config.canonical_parameter_value,
                        "spectral_side": side,
                        "top_rank_count": top_count,
                        **rank_match_fit(
                            degrees,
                            eigenvalues,
                            max_rank=top_count,
                            minimum_observations=3,
                        ),
                    }
                )
    diagnostic_dir = output_dir / "global_spectrum" / "nonlinear_hub_edge"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    write_rows(diagnostic_dir / "fixed_rank_lambda_over_sqrt_degree_by_graph.csv", fixed_rows)
    write_rows(diagnostic_dir / "loglog_fits_by_graph.csv", fit_rows)
    summary_rows: list[dict[str, object]] = []
    for side in ("perron", "positive_nonperron", "negative_absolute"):
        for rank in HUB_EDGE_FIXED_RANKS:
            selected = [row for row in fixed_rows if row["spectral_side"] == side and int(row["rank"]) == rank]
            if not selected:
                continue
            row: dict[str, object] = {
                "record_type": "fixed_rank",
                "attachment_zeta": config.canonical_parameter_value,
                "spectral_side": side,
                "rank_or_top_count": rank,
                "graph_count": len(selected),
            }
            for field in ("degree", "absolute_eigenvalue", "lambda_over_sqrt_degree"):
                mean, se = _mean_se([float(item[field]) for item in selected])
                row[f"mean_{field}"] = mean
                row[f"se_{field}"] = se
                row[f"median_{field}"] = float(np.median([float(item[field]) for item in selected]))
            summary_rows.append(row)
    for side in ("positive_nonperron", "negative_absolute"):
        for top_count in HUB_EDGE_FIT_RANKS:
            selected = [row for row in fit_rows if row["spectral_side"] == side and int(row["top_rank_count"]) == top_count]
            row = {
                "record_type": "loglog_fit",
                "attachment_zeta": config.canonical_parameter_value,
                "spectral_side": side,
                "rank_or_top_count": top_count,
                "graph_count": len(selected),
            }
            for field in ("slope", "r2", "spearman_rank_correlation", "median_lambda_over_sqrt_degree"):
                mean, se = _mean_se([float(item[field]) for item in selected])
                row[f"mean_{field}"] = mean
                row[f"se_{field}"] = se
            summary_rows.append(row)
    write_rows(diagnostic_dir / "hub_edge_summary.csv", summary_rows)

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    for side, marker in (("positive_nonperron", "o"), ("negative_absolute", "s"), ("perron", "^")):
        selected = [row for row in summary_rows if row["record_type"] == "fixed_rank" and row["spectral_side"] == side]
        if selected:
            axes[0].plot(
                [float(row["mean_degree"]) for row in selected],
                [float(row["mean_absolute_eigenvalue"]) for row in selected],
                marker=marker,
                label=side,
            )
            axes[1].plot(
                [int(row["rank_or_top_count"]) for row in selected],
                [float(row["median_lambda_over_sqrt_degree"]) for row in selected],
                marker=marker,
                label=side,
            )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("mean realized degree at matched rank")
    axes[0].set_ylabel("mean absolute edge eigenvalue")
    axes[1].set_xlabel("matched rank")
    axes[1].set_ylabel(r"median $|\lambda_j|/\sqrt{d_{(j)}}$")
    for axis in axes:
        axis.grid(alpha=0.2, which="both")
        axis.legend(frameon=False)
    figure.suptitle(f"Nonlinear PA hub--spectral-edge diagnostic, zeta={config.canonical_parameter_value:g}")
    figure.tight_layout()
    figure.savefig(diagnostic_dir / "hub_spectral_edge_scaling.png", dpi=180)
    plt.close(figure)
    return {
        "fixed_rank_rows": len(fixed_rows),
        "fit_rows": len(fit_rows),
        "summary_file": str(diagnostic_dir / "hub_edge_summary.csv"),
        "interpretation": "empirical test; square-root scaling is not assumed",
    }


def load_production_cache(
    config: MobilityEdgeProductionConfig,
    *,
    chunk_size: int = 5,
) -> tuple[list[dict[str, object]], list[dict[str, str]], np.ndarray, np.ndarray]:
    graphs: list[dict[str, object]] = []
    structural: list[dict[str, str]] = []
    decay_sum = np.zeros(
        (int(round(1.0 / config.quantile_width)), COORDINATE_DECAY_MAX_RANK),
        dtype=float,
    )
    decay_squares = np.zeros_like(decay_sum)
    for start in range(0, config.num_graphs, chunk_size):
        stop = min(start + chunk_size, config.num_graphs)
        valid, reason = validate_mobility_chunk(config, start, stop)
        if not valid:
            raise ValueError(f"invalid chunk [{start},{stop}): {reason}")
        cache_path, structural_path = mobility_chunk_paths(config, start, stop)
        structural.extend(_read_gzip_csv(structural_path))
        with np.load(cache_path, allow_pickle=False) as data:
            indices = np.asarray(data["graph_indices"], dtype=int)
            seeds = np.asarray(data["graph_seeds"], dtype=np.uint64)
            spectra = np.asarray(data["nonperron_eigenvalues"], dtype=float)
            degrees = np.asarray(data["degree_arrays"], dtype=np.int32)
            degree_orders = np.asarray(data["degree_orderings"], dtype=np.int32)
            offsets = np.asarray(data["eigenpair_offsets"], dtype=int)
            flat = {
                name: np.asarray(data[f"eigenpair_{name}"])
                for name in ("eigenvalue", "continuous_rank", "continuous_quantile", *LOCALIZATION_METRICS)
            }
            decay_sum += np.asarray(data["coordinate_decay_sum"], dtype=float)
            decay_squares += np.asarray(data["coordinate_decay_sum_squares"], dtype=float)
        for local, graph_index in enumerate(indices):
            left, right = int(offsets[local]), int(offsets[local + 1])
            graphs.append(
                {
                    "graph_index": int(graph_index),
                    "graph_seed": int(seeds[local]),
                    "nonperron_eigenvalues": spectra[local],
                    "degrees": degrees[local],
                    "degree_order": degree_orders[local],
                    "eigenpairs": {name: values[left:right] for name, values in flat.items()},
                }
            )
    graphs.sort(key=lambda row: int(row["graph_index"]))
    structural.sort(key=lambda row: int(row["graph_index"]))
    if [row["graph_index"] for row in graphs] != list(range(config.num_graphs)):
        raise ValueError("production cache graph coverage is incomplete")
    return graphs, structural, decay_sum, decay_squares


def collect_chunk_execution_index(
    config: MobilityEdgeProductionConfig, *, chunk_size: int
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for start in range(0, config.num_graphs, chunk_size):
        stop = min(start + chunk_size, config.num_graphs)
        cache_path, _ = mobility_chunk_paths(config, start, stop)
        with np.load(cache_path, allow_pickle=False) as data:
            execution = (
                json.loads(str(data["execution_json"].item()))
                if "execution_json" in data.files
                else {}
            )
        rows.append(
            {
                "graph_index_start": start,
                "graph_index_stop_exclusive": stop,
                "cache_path": str(cache_path),
                "embedded_execution_metadata": bool(execution),
                **execution,
            }
        )
    return rows


def aggregate_localization(
    config: MobilityEdgeProductionConfig,
    graphs: Sequence[dict[str, object]],
    output_dir: Path,
) -> list[dict[str, object]]:
    num_bins = int(round(1.0 / config.quantile_width))
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    graph_rows_path = output_dir / "eigenvector_percentile_scan_by_graph.csv.gz"
    graph_fields = [
        "graph_index",
        "graph_seed",
        "quantile",
        "quantile_low",
        "quantile_high",
        "number_of_eigenvectors",
        "E_mean",
        "E_min",
        "E_max",
        *[f"mean_{name}" for name in LOCALIZATION_METRICS],
    ]
    accumulators: list[dict[str, object]] = [
        {
            "graph_count": 0,
            "eigenvector_count": 0,
            "E_min": math.inf,
            "E_max": -math.inf,
            "metrics": {name: [] for name in ("E_mean", *LOCALIZATION_METRICS)},
        }
        for _ in range(num_bins)
    ]
    graph_rows_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(graph_rows_path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=graph_fields)
        writer.writeheader()
        for graph in graphs:
            cache = graph["eigenpairs"]
            quantiles = np.asarray(cache["continuous_quantile"], dtype=float)
            eigenvalues = np.asarray(cache["eigenvalue"], dtype=float)
            for bin_index in range(num_bins):
                q_low, q_high = float(edges[bin_index]), float(edges[bin_index + 1])
                left = int(np.searchsorted(quantiles, q_low, side="left"))
                side = "right" if bin_index == num_bins - 1 else "left"
                right = int(np.searchsorted(quantiles, q_high, side=side))
                if left == right:
                    continue
                selected_values = eigenvalues[left:right]
                row: dict[str, object] = {
                    "graph_index": graph["graph_index"],
                    "graph_seed": graph["graph_seed"],
                    "quantile": 0.5 * (q_low + q_high),
                    "quantile_low": q_low,
                    "quantile_high": q_high,
                    "number_of_eigenvectors": right - left,
                    "E_mean": float(np.mean(selected_values)),
                    "E_min": float(np.min(selected_values)),
                    "E_max": float(np.max(selected_values)),
                }
                for name in LOCALIZATION_METRICS:
                    row[f"mean_{name}"] = float(np.mean(np.asarray(cache[name])[left:right]))
                writer.writerow(row)
                accumulator = accumulators[bin_index]
                accumulator["graph_count"] = int(accumulator["graph_count"]) + 1
                accumulator["eigenvector_count"] = int(accumulator["eigenvector_count"]) + right - left
                accumulator["E_min"] = min(float(accumulator["E_min"]), float(row["E_min"]))
                accumulator["E_max"] = max(float(accumulator["E_max"]), float(row["E_max"]))
                metrics = accumulator["metrics"]
                for name in ("E_mean", *LOCALIZATION_METRICS):
                    metrics[name].append(float(row[name if name == "E_mean" else f"mean_{name}"]))
    summaries: list[dict[str, object]] = []
    for bin_index, accumulator in enumerate(accumulators):
        q_low, q_high = float(edges[bin_index]), float(edges[bin_index + 1])
        row: dict[str, object] = {
            "degree_ccdf_alpha" if config.model_family == "affine_linear" else "attachment_zeta": config.canonical_parameter_value,
            "quantile": 0.5 * (q_low + q_high),
            "quantile_low": q_low,
            "quantile_high": q_high,
            "number_of_graphs": accumulator["graph_count"],
            "number_of_eigenvectors": accumulator["eigenvector_count"],
            "E_min": accumulator["E_min"],
            "E_max": accumulator["E_max"],
        }
        metrics = accumulator["metrics"]
        for name in ("E_mean", *LOCALIZATION_METRICS):
            mean, se = _mean_se(metrics[name])
            output_name = name if name == "E_mean" else f"mean_{name}"
            row[output_name] = mean
            row[f"{output_name}_standard_error"] = se
        summaries.append(row)
    write_rows(output_dir / "eigenvector_percentile_scan_summary.csv", summaries)
    return summaries


def write_coordinate_heatmap(
    config: MobilityEdgeProductionConfig,
    decay_sum: np.ndarray,
    graph_count: int,
    output_dir: Path,
) -> None:
    mean = decay_sum / graph_count
    cumulative = np.cumsum(mean, axis=1)
    np.savez_compressed(
        output_dir / "eigenvector_coordinate_mass_heatmap_data.npz",
        mean_sorted_coordinate_mass=mean,
        mean_cumulative_top_r_mass=cumulative,
        quantile_centers=(np.arange(len(mean)) + 0.5) * config.quantile_width,
        coordinate_ranks=np.arange(1, mean.shape[1] + 1),
    )
    figure, axis = plt.subplots(figsize=(11.5, 7.0))
    image = axis.imshow(
        cumulative,
        origin="lower",
        aspect="auto",
        extent=(1, mean.shape[1], 0.0, 1.0),
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )
    figure.colorbar(image, ax=axis, label="mean cumulative top-r mass")
    axis.set_xlabel("top-r coordinate count")
    axis.set_ylabel("continuous non-Perron spectral quantile")
    axis.set_title(
        f"PA coordinate-mass concentration: {config.canonical_parameter_name}={config.canonical_parameter_value:g}"
    )
    figure.tight_layout()
    figure.savefig(output_dir / "eigenvector_coordinate_mass_heatmap.png", dpi=180)
    plt.close(figure)


def degree_and_tail_validation(
    config: MobilityEdgeProductionConfig,
    graphs: Sequence[dict[str, object]],
    structural_rows: Sequence[dict[str, str]],
    output_dir: Path,
) -> dict[str, object]:
    structural_dir = output_dir / "structural"
    structural_dir.mkdir(parents=True, exist_ok=True)
    graph_by_index = {int(graph["graph_index"]): graph for graph in graphs}
    enriched_structural: list[dict[str, object]] = []
    for source in structural_rows:
        row: dict[str, object] = dict(source)
        graph = graph_by_index[int(row["graph_index"])]
        degrees = np.asarray(graph["degrees"], dtype=float)
        total_degree = float(np.sum(degrees))
        spectrum = np.asarray(graph["nonperron_eigenvalues"], dtype=float)
        row.setdefault(
            "vertices_with_at_least_0p1_percent_total_degree",
            int(np.sum(degrees / total_degree >= 0.001)),
        )
        row.setdefault(
            "vertices_with_at_least_0p5_percent_total_degree",
            int(np.sum(degrees / total_degree >= 0.005)),
        )
        row.setdefault(
            "vertices_with_at_least_1_percent_total_degree",
            int(np.sum(degrees / total_degree >= 0.01)),
        )
        row.setdefault("nonperron_spectral_mean", float(np.mean(spectrum)))
        row.setdefault(
            "nonperron_spectral_standard_deviation", float(np.std(spectrum))
        )
        row.setdefault(
            "nonperron_mean_absolute_energy", float(np.mean(np.abs(spectrum)))
        )
        row.setdefault(
            "nonperron_positive_fraction", float(np.mean(spectrum > 0.0))
        )
        enriched_structural.append(row)
    structural_rows = enriched_structural
    write_rows(structural_dir / "structural_by_graph.csv", structural_rows)
    max_degree = max(int(np.max(graph["degrees"])) for graph in graphs)
    ccdf_rows: list[dict[str, object]] = []
    pmf_rows: list[dict[str, object]] = []
    for degree in range(config.m, max_degree + 1):
        values = [float(np.mean(np.asarray(graph["degrees"]) >= degree)) for graph in graphs]
        mean, se = _mean_se(values)
        ccdf_rows.append({"degree": degree, "mean_ccdf": mean, "standard_error": se})
        pmf_values = [float(np.mean(np.asarray(graph["degrees"]) == degree)) for graph in graphs]
        pmf_mean, pmf_se = _mean_se(pmf_values)
        pmf_rows.append({"degree": degree, "mean_pmf": pmf_mean, "standard_error": pmf_se})
    write_rows(structural_dir / "degree_ccdf.csv", ccdf_rows)
    write_rows(structural_dir / "degree_pmf.csv", pmf_rows)
    stretched_exponential = write_sublinear_stretched_exponential_diagnostic(
        config, graphs, ccdf_rows, structural_dir
    )

    fit_rows: list[dict[str, object]] = []
    tail_quantiles = (0.95, 0.96, 0.97, 0.976, 0.98)
    rank_rows: list[dict[str, object]] = []
    for graph in graphs:
        degrees_desc = np.sort(np.asarray(graph["degrees"], dtype=float))[::-1]
        if config.model_family == "affine_linear":
            for q0 in tail_quantiles:
                count = max(20, int(math.ceil((1.0 - q0) * config.n)))
                hill, xmin, observations = hill_ccdf_exponent_from_top(degrees_desc, count)
                regression = log_ccdf_regression(degrees_desc, config.n, count)
                fit_rows.append(
                    {
                        "graph_index": graph["graph_index"],
                        "tail_lower_quantile": q0,
                        "tail_observations": observations,
                        "xmin": xmin,
                        "hill_degree_ccdf_alpha": hill,
                        **regression,
                    }
                )
        spectrum = np.asarray(graph["nonperron_eigenvalues"], dtype=float)
        positive = np.sort(spectrum[spectrum > 0.0])[::-1]
        negative = np.sort(np.abs(spectrum[spectrum < 0.0]))[::-1]
        for side, eigenvalues in (("positive", positive), ("negative", negative)):
            result = rank_match_fit(degrees_desc, eigenvalues, max_rank=180)
            rank_rows.append({"graph_index": graph["graph_index"], "spectral_side": side, **result})
    if fit_rows:
        write_rows(structural_dir / "degree_tail_fits_by_graph.csv", fit_rows)
    write_rows(output_dir / "global_spectrum" / "degree_spectral_edge_match_by_graph.csv", rank_rows)

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    axes[0].loglog(
        [row["degree"] for row in ccdf_rows],
        [row["mean_ccdf"] for row in ccdf_rows],
        color="black",
    )
    axes[0].set_xlabel("degree k")
    axes[0].set_ylabel(r"mean empirical $P(D\geq k)$")
    axes[0].set_title("CCDF")
    axes[1].loglog(
        [row["degree"] for row in pmf_rows],
        [row["mean_pmf"] for row in pmf_rows],
        color="tab:blue",
    )
    axes[1].set_xlabel("degree k")
    axes[1].set_ylabel(r"mean empirical $P(D=k)$")
    axes[1].set_title("PMF")
    for axis in axes:
        axis.grid(alpha=0.2, which="both")
    figure.suptitle(
        f"PA degree distribution: {config.canonical_parameter_name}={config.canonical_parameter_value:g}"
    )
    figure.tight_layout()
    figure.savefig(structural_dir / "degree_ccdf.png", dpi=180)
    plt.close(figure)

    summary: dict[str, object] = {}
    for field in (
        "num_edges",
        "realized_mean_degree",
        "minimum_degree",
        "maximum_degree",
        "degree_gini",
        "top_1_degree_share",
        "top_5_degree_share",
        "top_10_degree_share",
        "top_1_percent_degree_share",
        "maximum_degree_over_n",
        "maximum_degree_over_total_degree",
        "degree_m_fraction",
        "maximum_degree_vertex_birth_index",
        "vertices_with_at_least_0p1_percent_total_degree",
        "vertices_with_at_least_0p5_percent_total_degree",
        "vertices_with_at_least_1_percent_total_degree",
        "num_components",
    ):
        mean, se = _mean_se([float(row[field]) for row in structural_rows])
        summary[f"mean_{field}"] = mean
        summary[f"se_{field}"] = se
    connected_values = [
        1.0 if str(row.get("connected", "False")).lower() == "true" else 0.0
        for row in structural_rows
    ]
    summary["connected_graph_fraction"], summary["connected_graph_fraction_se"] = _mean_se(
        connected_values
    )
    if config.graph_variant == "rewired":
        overlap_values = [float(row["rewiring_edge_overlap_fraction"]) for row in structural_rows]
        summary["mean_rewiring_edge_overlap_fraction"], summary[
            "se_rewiring_edge_overlap_fraction"
        ] = _mean_se(overlap_values)
        summary["exact_labeled_degree_preservation_verified_for_all_graphs"] = all(
            str(row.get("exact_labeled_degree_preserved", "False")).lower() == "true"
            for row in structural_rows
        )
    if config.model_family == "affine_linear":
        for q0 in tail_quantiles:
            selected = [row for row in fit_rows if row["tail_lower_quantile"] == q0]
            for field in ("hill_degree_ccdf_alpha", "ccdf_regression_exponent"):
                mean, se = _mean_se([float(row[field]) for row in selected])
                summary[f"q{q0:g}_{field}_mean"] = mean
                summary[f"q{q0:g}_{field}_se"] = se
    else:
        summary["power_law_exponent_fit_performed"] = False
        summary["reason"] = (
            "attachment_zeta is a kernel exponent; nonlinear degrees are assessed by "
            "condensation metrics and, for zeta<1, a stretched-exponential diagnostic"
        )
        if stretched_exponential:
            summary["sublinear_stretched_exponential_diagnostic"] = stretched_exponential
    for side in ("positive", "negative"):
        selected = [row for row in rank_rows if row["spectral_side"] == side]
        for field in (
            "slope",
            "r2",
            "spearman_rank_correlation",
            "median_lambda_over_sqrt_degree",
            "iqr_lambda_over_sqrt_degree",
        ):
            mean, se = _mean_se([float(row[field]) for row in selected])
            summary[f"{side}_{field}_mean"] = mean
            summary[f"{side}_{field}_se"] = se
    nonlinear_hub_edge = write_nonlinear_hub_spectral_edge_diagnostic(
        config, graphs, structural_rows, output_dir
    )
    if nonlinear_hub_edge:
        summary["nonlinear_hub_spectral_edge_diagnostic"] = nonlinear_hub_edge
    (structural_dir / "structural_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def fine_localization_from_cache(
    graphs: Sequence[dict[str, object]],
    fine_spacing_rows: Sequence[dict[str, object]],
    output_dir: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spacing in fine_spacing_rows:
        q_low = float(spacing["actual_q_low"])
        q_high = float(spacing["actual_q_high"])
        graph_metrics = {name: [] for name in LOCALIZATION_METRICS}
        graph_count = 0
        eigenpair_count = 0
        for graph in graphs:
            cache = graph["eigenpairs"]
            quantiles = np.asarray(cache["continuous_quantile"], dtype=float)
            left = int(np.searchsorted(quantiles, q_low, side="left"))
            side = "right" if q_high >= 1.0 - 1e-15 else "left"
            right = int(np.searchsorted(quantiles, q_high, side=side))
            if left == right:
                continue
            graph_count += 1
            eigenpair_count += right - left
            for name in LOCALIZATION_METRICS:
                graph_metrics[name].append(float(np.mean(np.asarray(cache[name])[left:right])))
        row = dict(spacing)
        row["fine_localization_graph_count"] = graph_count
        row["fine_localization_eigenpair_count"] = eigenpair_count
        for name in LOCALIZATION_METRICS:
            mean, se = _mean_se(graph_metrics[name])
            row[f"fine_mean_{name}"] = mean
            row[f"fine_mean_{name}_standard_error"] = se
        rows.append(row)
    if rows:
        write_rows(output_dir / "fine" / "joined_fine_localization_spacing.csv", rows)
    return rows


def aggregate_production_parameter(
    config: MobilityEdgeProductionConfig,
    *,
    chunk_size: int = 5,
) -> Path:
    output_dir = config.parameter_dir
    graphs, structural, decay_sum, _ = load_production_cache(config, chunk_size=chunk_size)
    execution_index = collect_chunk_execution_index(config, chunk_size=chunk_size)
    write_rows(output_dir / "provenance" / "chunk_execution_index.csv", execution_index)
    localization = aggregate_localization(
        config,
        graphs,
        output_dir / "coarse" / "q0_1_dq_0p002",
    )
    write_coordinate_heatmap(config, decay_sum, config.num_graphs, output_dir / "coarse" / "q0_1_dq_0p002")
    records = [
        build_gap_records(
            np.asarray(graph["nonperron_eigenvalues"]),
            graph_index=int(graph["graph_index"]),
            graph_seed=int(graph["graph_seed"]),
            excluded_atom_energies=(0.0,),
            audited_atom_energies=(-1.0, 0.0, 1.0),
            atom_atol=config.zero_eigenvalue_atol,
        )
        for graph in graphs
    ]
    models = build_cross_fitted_models(records, num_folds=5)
    save_crossfit_models(
        output_dir / "unfolding" / "crossfit_k5",
        models,
        metadata={**config.metadata(), "source_cache": str(config.partial_dir)},
    )
    level_rows, histograms = summarize_quantile_grid(
        records,
        width=config.quantile_width,
        crossfit_models=models,
        minimum_positive_gaps=100,
    )
    coarse_dir = output_dir / "coarse" / "q0_1_dq_0p002"
    write_rows(coarse_dir / "raw_r_crossfit_spacing_summary.csv", level_rows)
    write_rows(
        coarse_dir / "raw_r_crossfit_spacing_by_graph.csv.gz",
        graph_level_quantile_rows(
            records,
            width=config.quantile_width,
            crossfit_models=models,
        ),
    )
    density_rows = histogram_rows(histograms, level_rows)
    if density_rows:
        write_rows(coarse_dir / "crossfit_spacing_histogram_density.csv", density_rows)
    joined = join_localization_and_level_statistics(localization, level_rows)
    write_rows(coarse_dir / "joined_localization_level_statistics.csv", joined)
    candidates = detect_candidate_regions(joined)
    (coarse_dir / "candidate_regions.json").write_text(
        json.dumps(candidates, indent=2, sort_keys=True) + "\n"
    )
    fine_spacing = fine_candidate_scans(records, models, candidates, output_dir)
    fine_localization_from_cache(graphs, fine_spacing, output_dir)
    global_summary = write_global_spectrum(
        [np.asarray(graph["nonperron_eigenvalues"]) for graph in graphs],
        [record.continuous_eigenvalues for record in records],
        output_dir,
        model_label=(
            "affine preferential attachment"
            if config.model_family == "affine_linear"
            else "nonlinear preferential attachment"
        ),
    )
    write_rows(
        output_dir / "global_spectrum" / "atom_multiplicity_by_graph.csv",
        [
            {
                "graph_index": int(row["graph_index"]),
                "graph_seed": int(row["graph_seed"]),
                "minus_one_multiplicity": int(float(row["minus_one_multiplicity"])),
                "zero_multiplicity": int(float(row["zero_multiplicity"])),
                "plus_one_multiplicity": int(float(row["plus_one_multiplicity"])),
                "continuous_nonperron_dimension": int(
                    float(row["continuous_nonperron_dimension"])
                ),
            }
            for row in structural
        ],
    )
    structural_summary = degree_and_tail_validation(
        config, graphs, structural, output_dir
    )
    plot_coarse(
        joined,
        config.canonical_parameter_value,
        output_dir,
        model_label=(
            "Affine PA" if config.model_family == "affine_linear" else "Nonlinear PA"
        ),
        parameter_name=config.canonical_parameter_name,
    )
    source_cache = {
        "status": "NEW_PRODUCTION_CACHE",
        "source_cache": str(config.partial_dir),
        "chunk_execution_index": str(
            output_dir / "provenance" / "chunk_execution_index.csv"
        ),
        "sge_job_ids_embedded_in_chunks": sorted(
            {
                str(row.get("sge_job_id", ""))
                for row in execution_index
                if str(row.get("sge_job_id", ""))
            }
        ),
        "analysis_version": "pa_mobility_edge_production_cache_v1",
        "canonical_parameter": {
            "name": config.canonical_parameter_name,
            "value": config.canonical_parameter_value,
        },
        "graph_indices": list(range(config.num_graphs)),
        "graph_seeds": config.graph_seeds(),
        "rewire_seeds": (
            config.rewire_seeds() if config.graph_variant == "rewired" else None
        ),
    }
    (output_dir / "source_cache.json").write_text(
        json.dumps(source_cache, indent=2, sort_keys=True) + "\n"
    )
    manifest = {
        **config.metadata(),
        "postprocess_analysis_version": ANALYSIS_VERSION,
        "result_origin": (
            "new eigendecomposition of exact-degree rewired PA graphs and per-eigenpair cache"
            if config.graph_variant == "rewired"
            else "new eigendecomposition and per-eigenpair cache"
        ),
        "source_cache": str(config.partial_dir),
        "chunk_execution_index": source_cache["chunk_execution_index"],
        "sge_job_ids_embedded_in_chunks": source_cache[
            "sge_job_ids_embedded_in_chunks"
        ],
        "gap_assignment": "gap-first midpoint",
        "ratio_assignment": "middle eigenvalue",
        "unfolding": "five-fold graph-level cross-fitted empirical CDF",
        "minimum_positive_gaps_per_histogram": 100,
        "candidate_regions": candidates,
        "global_spectrum_summary": global_summary,
        "structural_summary": structural_summary,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return output_dir
