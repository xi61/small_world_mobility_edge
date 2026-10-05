"""Aggregation, plots, indices, and reports for paired raw-amplitude PA disorder."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from small_world_mobility_edge.random_schrodinger.anderson_utils import atomic_write_csv, atomic_write_json
from small_world_mobility_edge.random_schrodinger.common import read_json, sha256_file, utc_now
from small_world_mobility_edge.random_schrodinger.pa_diagonal_uniform_noise import (
    PA_UNIFORM_NOISE_VERSION,
    amplitude_slug,
    case_directory,
    operator_directory,
)


REPORTING_VERSION = "pa_diagonal_uniform_noise_reporting_v1"


def read_rows(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _int(value: object) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return -1


def _finite(values: Iterable[object]) -> np.ndarray:
    data = np.asarray([_float(value) for value in values], dtype=float)
    return data[np.isfinite(data)]


def _bins(lo: float, hi: float, count: int) -> np.ndarray:
    if not math.isfinite(lo) or not math.isfinite(hi):
        return np.linspace(0.0, 1.0, int(count) + 1)
    if hi <= lo:
        width = max(1.0, abs(lo)) * 1e-9
        lo, hi = lo - width, hi + width
    return np.linspace(lo, hi, int(count) + 1)


def _histogram(values: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts, _ = np.histogram(values, bins=edges)
    widths = np.diff(edges)
    density = counts / (max(1, int(np.sum(counts))) * widths)
    return counts, density


def _metric_bins(rows: Sequence[Mapping[str, str]], edges: np.ndarray, metric: str) -> list[dict[str, object]]:
    energy = np.asarray([_float(row.get("energy")) for row in rows], dtype=float)
    values = np.asarray([_float(row.get(metric)) for row in rows], dtype=float)
    usable = np.isfinite(energy) & np.isfinite(values)
    output: list[dict[str, object]] = []
    for index in range(len(edges) - 1):
        inside = usable & (energy >= edges[index]) & (energy < edges[index + 1] if index < len(edges) - 2 else energy <= edges[index + 1])
        selected = values[inside]
        output.append({
            "energy_bin_index": index,
            "energy_left": float(edges[index]),
            "energy_right": float(edges[index + 1]),
            "energy_center": float(0.5 * (edges[index] + edges[index + 1])),
            "usable_count": int(len(selected)),
            f"mean_{metric}": float(np.mean(selected)) if len(selected) else math.nan,
            f"median_{metric}": float(np.median(selected)) if len(selected) else math.nan,
        })
    return output


def _operator_inputs(run_root: Path, n: int, alpha: float, amplitude: float) -> dict[str, object]:
    root = operator_directory(run_root, n, alpha, amplitude)
    manifest = read_json(root / "run_manifest.json")
    if manifest.get("status") != "COMPLETE":
        raise ValueError(f"operator is not COMPLETE: {root}")
    tables = root / "tables"
    return {
        "root": root,
        "manifest": manifest,
        "localization": read_rows(tables / "full_spectrum_localization.csv.gz"),
        "ratios": read_rows(tables / "full_spectrum_gap_ratios.csv.gz"),
        "windows": read_rows(tables / "energy_and_rank_window_summary.csv.gz"),
        "gap_observations": read_rows(tables / "window_gap_observations.csv.gz"),
        "cumulative": read_rows(tables / "cumulative_rows.csv.gz"),
        "cross_summary": read_rows(tables / "cross_summary_rows.csv.gz"),
        "pair_summary": read_rows(tables / "induced_pair_distance_summary_rows.csv.gz"),
    }


def _window_key(row: Mapping[str, str]) -> tuple[str, str, str]:
    return row["window_family"], row["window_id"], row["perron_policy"]


def _representative_window(rows: Sequence[Mapping[str, str]], family: str, window_id: str, policy: str = "strict_internal") -> Mapping[str, str] | None:
    for row in rows:
        if row.get("window_family") == family and row.get("window_id") == window_id and row.get("membership_policy") == policy:
            return row
    return None


def _case_figures(
    *,
    case_report: Path,
    n: int,
    alpha: float,
    amplitudes: Sequence[float],
    inputs: Mapping[float, Mapping[str, object]],
    energy_edges: np.ndarray,
    localization_bins: Sequence[Mapping[str, object]],
    gap_ratio_bins: Sequence[Mapping[str, object]],
    density_rows: Sequence[Mapping[str, object]],
    gap_hist_rows: Sequence[Mapping[str, object]],
) -> list[Path]:
    figure_dir = case_report / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(amplitudes)))

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True, sharey=True, constrained_layout=True)
    for axis, amplitude, color in zip(axes.flat, amplitudes, colors):
        rows = inputs[amplitude]["localization"]
        energy = np.asarray([_float(row["energy"]) for row in rows])
        nipr = np.asarray([_float(row["n_times_ipr"]) for row in rows])
        mask = np.isfinite(nipr)
        axis.scatter(energy[mask], nipr[mask], s=3 if n >= 10000 else 7, alpha=0.28, color=color, rasterized=True)
        binned = [row for row in localization_bins if _float(row["raw_amplitude"]) == amplitude]
        axis.plot([_float(row["energy_center"]) for row in binned], [_float(row["median_n_times_ipr"]) for row in binned], color="black", lw=1.3, label="bin median")
        axis.set_yscale("log")
        axis.set_title(f"a={amplitude:g}")
        axis.grid(alpha=0.2)
    fig.suptitle(f"Raw energy versus N×IPR, N={n}, α={alpha:g}\nPoints are eigenstates; clean-atom solver vectors are masked")
    fig.supxlabel("Raw eigenvalue E")
    fig.supylabel("N×IPR")
    path = figure_dir / "energy_vs_n_times_ipr.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True, sharey=True, constrained_layout=True)
    for axis, amplitude, color in zip(axes.flat, amplitudes, colors):
        rows = inputs[amplitude]["ratios"]
        energy = np.asarray([_float(row["middle_energy"]) for row in rows])
        ratios = np.asarray([_float(row["filtered_gap_ratio"]) for row in rows])
        mask = np.isfinite(ratios)
        axis.scatter(energy[mask], ratios[mask], s=3 if n >= 10000 else 7, alpha=0.25, color=color, rasterized=True)
        binned = [row for row in gap_ratio_bins if _float(row["raw_amplitude"]) == amplitude]
        axis.plot([_float(row["energy_center"]) for row in binned], [_float(row["mean_filtered_gap_ratio"]) for row in binned], color="black", lw=1.3)
        axis.set_ylim(0, 1)
        axis.set_title(f"a={amplitude:g}")
        axis.grid(alpha=0.2)
    fig.suptitle(f"Raw energy versus adjacent gap ratio, N={n}, α={alpha:g}\nBlack line: energy-bin mean; unresolved/atom triplets excluded")
    fig.supxlabel("Raw eigenvalue E at middle level")
    fig.supylabel("min(dleft,dright)/max(dleft,dright)")
    path = figure_dir / "energy_vs_gap_ratio.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)

    fig, axes = plt.subplots(4, 4, figsize=(18, 14), constrained_layout=True)
    density_by_a: dict[float, list[Mapping[str, object]]] = defaultdict(list)
    for row in density_rows:
        density_by_a[_float(row["raw_amplitude"])].append(row)
    gap_by_a: dict[float, list[Mapping[str, object]]] = defaultdict(list)
    for row in gap_hist_rows:
        if row["window_family"] == "cumulative_nonperron" and row["window_id"] == "nonperron_top_50":
            gap_by_a[_float(row["raw_amplitude"])].append(row)
    for index, (amplitude, color) in enumerate(zip(amplitudes, colors)):
        density_axis = axes.flat[index]
        rows = sorted(density_by_a[amplitude], key=lambda row: _int(row["bin_index"]))
        density_axis.step([_float(row["energy_center"]) for row in rows], [_float(row["density"]) for row in rows], where="mid", color=color)
        density_axis.set_title(f"Density, a={amplitude:g}")
        density_axis.set_xlabel("E")
        density_axis.set_ylabel("density")
        gap_axis = axes.flat[index + 8]
        rows = sorted(gap_by_a[amplitude], key=lambda row: _int(row["bin_index"]))
        gap_axis.step([_float(row["gap_center"]) for row in rows], [_float(row["density"]) for row in rows], where="mid", color=color)
        gap_axis.set_title(f"Raw gaps: non-Perron top 50, a={amplitude:g}")
        gap_axis.set_xlabel("raw gap")
        gap_axis.set_ylabel("density")
        density_axis.grid(alpha=0.2)
        gap_axis.grid(alpha=0.2)
    fig.suptitle(f"Eigenvalue densities and distinct raw-gap histograms, N={n}, α={alpha:g}")
    path = figure_dir / "density_and_raw_gap_histograms.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    for amplitude, color in zip(amplitudes, colors):
        windows = inputs[amplitude]["windows"]
        literal = [row for row in windows if row.get("window_family") == "cumulative_literal" and row.get("membership_policy") == "strict_internal"]
        literal.sort(key=lambda row: _int(row["actual_state_count"]))
        axes[0, 0].plot([_int(row["actual_state_count"]) for row in literal], [_float(row["mean_gap_ratio_filtered"]) for row in literal], marker="o", ms=3, color=color, label=f"a={amplitude:g}")
        axes[1, 0].plot([_int(row["actual_state_count"]) for row in literal], [_float(row["mean_n_times_ipr"]) for row in literal], marker="o", ms=3, color=color)
    selected_counts = [10, 50, 100, 500 if n == 500 else 1000]
    for count, style in zip(selected_counts, ["-", "--", "-.", ":"]):
        ratio_values, nipr_values, support_values = [], [], []
        for amplitude in amplitudes:
            row = _representative_window(inputs[amplitude]["windows"], "cumulative_nonperron", f"nonperron_top_{count}")
            ratio_values.append(_float(row["mean_gap_ratio_filtered"]) if row else math.nan)
            nipr_values.append(_float(row["mean_n_times_ipr"]) if row else math.nan)
            support_values.append(_float(row["mean_support_size_80"]) if row else math.nan)
        axes[0, 1].plot(amplitudes, ratio_values, style, marker="o", label=f"requested k={count}")
        axes[1, 1].plot(amplitudes, support_values, style, marker="o", label=f"requested k={count}")
    axes[0, 0].set_title("Mean gap ratio versus cumulative literal count")
    axes[0, 1].set_title("Mean gap ratio versus raw a, rank 1 excluded")
    axes[1, 0].set_title("Mean N×IPR versus cumulative literal count")
    axes[1, 1].set_title("Mean 80% support size versus raw a, rank 1 excluded")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7, ncol=2)
    axes[0, 0].set_xscale("log")
    axes[1, 0].set_xscale("log")
    axes[0, 1].set_xlabel("raw amplitude a")
    axes[1, 1].set_xlabel("raw amplitude a")
    fig.suptitle(f"Upper-edge rank summaries, N={n}, α={alpha:g}")
    path = figure_dir / "upper_edge_summaries.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True, sharey=True, constrained_layout=True)
    for axis, amplitude, color in zip(axes.flat, amplitudes, colors):
        rows = inputs[amplitude]["cumulative"]
        grouped: dict[int, list[Mapping[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[_int(row["decreasing_algebraic_rank"])].append(row)
        for rank in sorted(grouped):
            points = sorted(grouped[rank], key=lambda row: _int(row["requested_coordinate_count"]))
            axis.plot([_int(row["requested_coordinate_count"]) for row in points], [_float(row["cumulative_mass"]) for row in points], marker="o", ms=2, lw=0.9, label=f"r{rank}")
        axis.set_title(f"a={amplitude:g}")
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.2)
    axes.flat[0].legend(ncol=3, fontsize=6)
    fig.suptitle(f"Cumulative top-coordinate mass for algebraic ranks 1–11, N={n}, α={alpha:g}")
    fig.supxlabel("number of highest-mass coordinates")
    fig.supylabel("cumulative mass")
    path = figure_dir / "detailed_cumulative_mass.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)

    selected_amplitudes = [value for value in (0.0, 8.0, 32.0) if value in amplitudes]
    fig, axes = plt.subplots(1, len(selected_amplitudes), figsize=(5.2 * len(selected_amplitudes), 4.8), constrained_layout=True)
    axes = np.atleast_1d(axes)
    image = None
    for axis, amplitude in zip(axes, selected_amplitudes):
        matrix = np.full((11, 11), np.nan)
        for row in inputs[amplitude]["cross_summary"]:
            left, right = _int(row["left_decreasing_rank"]), _int(row["right_decreasing_rank"])
            if 1 <= left <= 11 and 1 <= right <= 11:
                matrix[left - 1, right - 1] = matrix[right - 1, left - 1] = _float(row["dominant_vertex_distance"])
        np.fill_diagonal(matrix, 0.0)
        image = axis.imshow(matrix, vmin=0, vmax=max(1.0, np.nanmax(matrix)), cmap="magma")
        axis.set_title(f"a={amplitude:g}")
        axis.set_xlabel("decreasing algebraic rank")
        axis.set_ylabel("decreasing algebraic rank")
        axis.set_xticks(range(11))
        axis.set_xticklabels(range(1, 12), fontsize=7)
        axis.set_yticks(range(11))
        axis.set_yticklabels(range(1, 12), fontsize=7)
    if image is not None:
        fig.colorbar(image, ax=axes.tolist(), label="full-graph dominant-vertex distance")
    fig.suptitle(f"Detailed-state geometry, N={n}, α={alpha:g}")
    path = figure_dir / "geometry_summary_matrices.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for family, window_id, label in (
        ("cumulative_literal", "literal_top_50", "literal top 50"),
        ("cumulative_nonperron", "nonperron_top_50", "rank 1 excluded top 50"),
        ("global_quantile_rank_band", "qrank_11", "central rank band"),
    ):
        gap_values, nipr_values = [], []
        for amplitude in amplitudes:
            row = _representative_window(inputs[amplitude]["windows"], family, window_id)
            gap_values.append(_float(row["mean_gap_ratio_filtered"]) if row else math.nan)
            nipr_values.append(_float(row["median_n_times_ipr"]) if row else math.nan)
        axes[0].plot(amplitudes, gap_values, marker="o", label=label)
        axes[1].plot(amplitudes, nipr_values, marker="o", label=label)
    axes[0].set_title("Matched-rank mean gap ratio")
    axes[1].set_title("Matched-rank median N×IPR")
    axes[1].set_yscale("log")
    for axis in axes:
        axis.set_xlabel("raw amplitude a")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle(f"Clean versus noisy operators on the exact graph, N={n}, α={alpha:g}\nRank matching is not continuous mode tracking")
    path = figure_dir / "matched_clean_disorder_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)
    return outputs


def aggregate_case(*, config: Mapping[str, object], run_root: Path, n: int, alpha: float, source_identity: str) -> Path:
    amplitudes = [float(value) for value in config["amplitudes"]]
    case_root = case_directory(run_root, n, alpha)
    report = case_root / "report"
    report.mkdir(parents=True, exist_ok=True)
    inputs = {amplitude: _operator_inputs(run_root, n, alpha, amplitude) for amplitude in amplitudes}
    all_energy = np.concatenate([
        np.asarray([_float(row["energy"]) for row in inputs[amplitude]["localization"]], dtype=float)
        for amplitude in amplitudes
    ])
    energy_edges = _bins(float(np.min(all_energy)), float(np.max(all_energy)), int(config["analysis"]["energy_bin_count"]))
    atomic_save = report / "tables/common_energy_bin_edges.csv"
    atomic_write_csv(atomic_save, [
        {"bin_index": i, "energy_left": float(energy_edges[i]), "energy_right": float(energy_edges[i + 1])}
        for i in range(len(energy_edges) - 1)
    ])
    density_rows: list[dict[str, object]] = []
    localization_bins: list[dict[str, object]] = []
    ratio_bins: list[dict[str, object]] = []
    operator_summary: list[dict[str, object]] = []
    upper_summary: list[dict[str, object]] = []
    detailed_mass: list[dict[str, object]] = []
    geometry_summary: list[dict[str, object]] = []
    for amplitude in amplitudes:
        data = inputs[amplitude]
        energies = np.asarray([_float(row["energy"]) for row in data["localization"]], dtype=float)
        counts, density = _histogram(energies, energy_edges)
        for index in range(len(counts)):
            density_rows.append({
                "n": n, "degree_ccdf_alpha": alpha, "raw_amplitude": amplitude,
                "bin_index": index, "energy_left": float(energy_edges[index]), "energy_right": float(energy_edges[index + 1]),
                "energy_center": float(0.5 * (energy_edges[index] + energy_edges[index + 1])),
                "count": int(counts[index]), "density": float(density[index]), "sample_count": int(len(energies)),
            })
        for row in _metric_bins(data["localization"], energy_edges, "n_times_ipr"):
            localization_bins.append({"n": n, "degree_ccdf_alpha": alpha, "raw_amplitude": amplitude, **row})
        ratio_source = [{"energy": row["middle_energy"], "filtered_gap_ratio": row["filtered_gap_ratio"]} for row in data["ratios"]]
        for row in _metric_bins(ratio_source, energy_edges, "filtered_gap_ratio"):
            ratio_bins.append({"n": n, "degree_ccdf_alpha": alpha, "raw_amplitude": amplitude, **row})
        manifest = data["manifest"]
        whole_window = _representative_window(data["windows"], "cumulative_literal", f"literal_top_{n}")
        if whole_window is None and n == 10000:
            whole_window = _representative_window(data["windows"], "global_quantile_rank_band", "qrank_00")
        operator_summary.append({
            "n": n,
            "degree_ccdf_alpha": alpha,
            "raw_amplitude": amplitude,
            "graph_sha256_edges": manifest["graph_evidence"]["canonical_edge_hash"],
            "potential_seed": manifest["potential_seed"],
            "xi_sha256_raw_float64": manifest["xi_sha256_raw_float64"],
            "minimum_energy": float(np.min(energies)),
            "maximum_energy": float(np.max(energies)),
            "usable_localization_state_count": int(sum(math.isfinite(_float(row["n_times_ipr"])) for row in data["localization"])),
            "median_n_times_ipr_all_usable": float(np.median(_finite(row["n_times_ipr"] for row in data["localization"]))) if len(_finite(row["n_times_ipr"] for row in data["localization"])) else math.nan,
            "mean_gap_ratio_all_valid": float(np.mean(_finite(row["filtered_gap_ratio"] for row in data["ratios"]))) if len(_finite(row["filtered_gap_ratio"] for row in data["ratios"])) else math.nan,
            "valid_gap_ratio_count": int(len(_finite(row["filtered_gap_ratio"] for row in data["ratios"]))),
            "clean_zero_atom_dimension": manifest["analysis_summary"]["zero_subspace"]["dimension"],
            "maximum_relative_residual": manifest["spectral_cache"].get("maximum_relative_residual", manifest["graph_evidence"]["spectral_manifest"].get("maximum_relative_residual")),
            "trace_identity_pass": manifest["spectral_trace_identities"]["identities_pass"],
        })
        for row in data["windows"]:
            if row.get("membership_policy") == "strict_internal" and row.get("window_family") in {"cumulative_literal", "cumulative_nonperron", "disjoint_literal", "disjoint_nonperron"}:
                upper_summary.append(dict(row))
        detailed_mass.extend(data["cumulative"])
        geometry_summary.extend(data["cross_summary"])
        geometry_summary.extend(data["pair_summary"])
    atomic_write_csv(report / "tables/spectral_density.csv.gz", density_rows)
    atomic_write_csv(report / "tables/energy_binned_localization.csv.gz", localization_bins)
    atomic_write_csv(report / "tables/energy_binned_gap_ratio.csv.gz", ratio_bins)
    atomic_write_csv(report / "tables/operator_summary.csv", operator_summary)
    atomic_write_csv(report / "tables/upper_edge_and_band_summary.csv.gz", upper_summary)
    atomic_write_csv(report / "tables/detailed_cumulative_mass.csv.gz", detailed_mass)
    atomic_write_csv(report / "tables/geometry_summary.csv.gz", geometry_summary)

    grouped_gaps: dict[tuple[str, str, str], dict[float, np.ndarray]] = defaultdict(dict)
    for amplitude in amplitudes:
        groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        for row in inputs[amplitude]["gap_observations"]:
            if row.get("membership_policy") != "strict_internal":
                continue
            if row.get("numerically_resolved") != "True" or row.get("touches_clean_zero_atom") == "True":
                continue
            value = _float(row.get("raw_gap"))
            if math.isfinite(value) and value > 0.0:
                groups[_window_key(row)].append(value)
        for key, values in groups.items():
            grouped_gaps[key][amplitude] = np.asarray(values, dtype=float)
    gap_hist_rows: list[dict[str, object]] = []
    bin_count = int(config["analysis"]["raw_gap_histogram_bin_count"])
    for key, by_amplitude in grouped_gaps.items():
        combined_parts = [values for values in by_amplitude.values() if len(values)]
        if not combined_parts:
            continue
        combined = np.concatenate(combined_parts)
        edges = _bins(float(np.min(combined)), float(np.max(combined)), bin_count)
        for amplitude in amplitudes:
            values = by_amplitude.get(amplitude, np.empty(0))
            counts, density = _histogram(values, edges)
            for index in range(len(counts)):
                gap_hist_rows.append({
                    "n": n, "degree_ccdf_alpha": alpha, "raw_amplitude": amplitude,
                    "window_family": key[0], "window_id": key[1], "perron_policy": key[2],
                    "membership_policy": "strict_internal", "bin_index": index,
                    "gap_left": float(edges[index]), "gap_right": float(edges[index + 1]), "gap_center": float(0.5 * (edges[index] + edges[index + 1])),
                    "count": int(counts[index]), "density": float(density[index]), "contributing_gap_count": int(len(values)),
                })
    atomic_write_csv(report / "tables/raw_gap_histograms.csv.gz", gap_hist_rows)
    figures = _case_figures(
        case_report=report,
        n=n,
        alpha=alpha,
        amplitudes=amplitudes,
        inputs=inputs,
        energy_edges=energy_edges,
        localization_bins=localization_bins,
        gap_ratio_bins=ratio_bins,
        density_rows=density_rows,
        gap_hist_rows=gap_hist_rows,
    )
    baseline = next(row for row in operator_summary if _float(row["raw_amplitude"]) == 0.0)
    strongest = next(row for row in operator_summary if _float(row["raw_amplitude"]) == max(amplitudes))
    baseline_nipr, strongest_nipr = _float(baseline["median_n_times_ipr_all_usable"]), _float(strongest["median_n_times_ipr_all_usable"])
    baseline_ratio, strongest_ratio = _float(baseline["mean_gap_ratio_all_valid"]), _float(strongest["mean_gap_ratio_all_valid"])
    lines = [
        f"# PA diagonal-uniform-noise case report: N={n}, alpha={alpha:g}",
        "",
        "This appendix compares `H(a)=A+a diag(xi)` on one exact clean affine-PA graph. The same iid `xi` stream is reused across raw amplitudes. Rank matching is not mode tracking.",
        "",
        "## Numerical summary",
        "",
        f"- Completed amplitudes: {', '.join(f'{value:g}' for value in amplitudes)}.",
        f"- Median whole-spectrum usable N*IPR changed from {baseline_nipr:.6g} at a=0 to {strongest_nipr:.6g} at a={max(amplitudes):g}.",
        f"- Mean valid whole-spectrum raw gap ratio changed from {baseline_ratio:.6g} to {strongest_ratio:.6g} over the same paired sweep.",
        f"- Clean stable zero-cluster dimension: {baseline['clean_zero_atom_dimension']}; those solver-basis individual states are excluded from primary localization summaries.",
        "- These are descriptive results from one graph and one potential stream. They do not support an ensemble confidence interval or a universal critical amplitude.",
        "",
        "## Figures",
        "",
    ]
    lines.extend(f"- `{path.relative_to(report)}`" for path in figures)
    lines.extend([
        "",
        "## Tables",
        "",
        "- `tables/operator_summary.csv`",
        "- `tables/spectral_density.csv.gz`",
        "- `tables/energy_binned_localization.csv.gz`",
        "- `tables/energy_binned_gap_ratio.csv.gz`",
        "- `tables/raw_gap_histograms.csv.gz`",
        "- `tables/upper_edge_and_band_summary.csv.gz`",
        "- `tables/detailed_cumulative_mass.csv.gz`",
        "- `tables/geometry_summary.csv.gz`",
        "",
    ])
    (report / "CASE_REPORT.md").write_text("\n".join(lines))
    manifest = {
        "status": "COMPLETE",
        "stage": "case_aggregate_tables_plots_report",
        "reporting_version": REPORTING_VERSION,
        "source_identity": source_identity,
        "n": n,
        "degree_ccdf_alpha": alpha,
        "amplitudes": amplitudes,
        "operator_count": len(amplitudes),
        "figure_paths": [str(path) for path in figures],
        "table_paths": [str(path) for path in sorted((report / "tables").glob("*"))],
        "finished_at_utc": utc_now(),
    }
    atomic_write_json(report / "run_manifest.json", manifest)
    return report


def _sha_inventory(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(Path(root).rglob("*")):
        if path.is_file() and "full_eigenvectors_ascending.npy" not in path.name:
            rows.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def aggregate_global(*, config: Mapping[str, object], run_root: Path, source_identity: str) -> Path:
    run_root = Path(run_root)
    report = run_root / "report"
    report.mkdir(parents=True, exist_ok=True)
    amplitudes = [float(value) for value in config["amplitudes"]]
    completion: list[dict[str, object]] = []
    combined_operator: list[dict[str, str]] = []
    combined_upper: list[dict[str, str]] = []
    case_reports: list[Path] = []
    blockers: list[str] = []
    for n in config["sizes"]:
        for alpha in config["degree_ccdf_alpha_grid"]:
            case_root = case_directory(run_root, int(n), float(alpha))
            for amplitude in amplitudes:
                manifest_path = operator_directory(run_root, int(n), float(alpha), amplitude) / "run_manifest.json"
                status = "MISSING"
                reason = ""
                if manifest_path.is_file():
                    manifest = read_json(manifest_path)
                    status = str(manifest.get("status", "UNKNOWN"))
                    reason = str(manifest.get("failure_message", ""))
                completion.append({
                    "n": int(n), "degree_ccdf_alpha": float(alpha), "raw_amplitude": amplitude,
                    "status": status, "reason": reason, "manifest_path": str(manifest_path),
                })
            case_manifest = case_root / "report/run_manifest.json"
            if case_manifest.is_file() and read_json(case_manifest).get("status") == "COMPLETE":
                case_reports.append(case_root / "report/CASE_REPORT.md")
                combined_operator.extend(read_rows(case_root / "report/tables/operator_summary.csv"))
                combined_upper.extend(read_rows(case_root / "report/tables/upper_edge_and_band_summary.csv.gz"))
            else:
                blockers.append(str(case_manifest))
    atomic_write_csv(report / "tables/completion_matrix_14x8.csv", completion)
    if combined_operator:
        atomic_write_csv(report / "tables/all_cases_operator_summary.csv", combined_operator)
    if combined_upper:
        atomic_write_csv(report / "tables/all_cases_upper_edge_summary.csv.gz", combined_upper)
    complete_count = sum(row["status"] == "COMPLETE" for row in completion)

    figure_paths: list[Path] = []
    if combined_upper:
        fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
        selected_amplitudes = [0.0, 4.0, 16.0, 32.0]
        colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(selected_amplitudes)))
        for row_index, n in enumerate(config["sizes"]):
            for amplitude, color in zip(selected_amplitudes, colors):
                rows = [row for row in combined_upper if _int(row["n"]) == int(n) and math.isclose(_float(row["raw_amplitude"]), amplitude) and row["window_family"] == "cumulative_nonperron" and row["window_id"] == "nonperron_top_50"]
                rows.sort(key=lambda row: _float(row["degree_ccdf_alpha"]))
                axes[row_index, 0].plot([_float(row["degree_ccdf_alpha"]) for row in rows], [_float(row["mean_gap_ratio_filtered"]) for row in rows], marker="o", color=color, label=f"a={amplitude:g}")
                axes[row_index, 1].plot([_float(row["degree_ccdf_alpha"]) for row in rows], [_float(row["median_n_times_ipr"]) for row in rows], marker="o", color=color, label=f"a={amplitude:g}")
            axes[row_index, 0].set_title(f"N={int(n)}: non-Perron top-50 mean gap ratio")
            axes[row_index, 1].set_title(f"N={int(n)}: non-Perron top-50 median N×IPR")
            axes[row_index, 1].set_yscale("log")
            for axis in axes[row_index]:
                axis.set_xlabel("degree-CCDF alpha")
                axis.grid(alpha=0.2)
                axis.legend(fontsize=8)
        fig.suptitle("Cross-alpha comparisons at fixed N and raw a\nOne exact graph and one potential stream per point")
        path = report / "figures/cross_alpha_fixed_n_a.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
        plt.close(fig)
        figure_paths.append(path)

    summary_lines = [
        "# PA diagonal uniform noise execution report",
        "",
        f"**Status:** {'COMPLETE' if complete_count == len(completion) and not blockers else 'PARTIAL'}  ",
        f"**Completed logical operators:** {complete_count}/{len(completion)}  ",
        f"**Operator:** `H(a)=A+a diag(xi)` with raw `a` and unscaled adjacency  ",
        f"**Source identity:** `{source_identity}`",
        "",
        "## Scope and design",
        "",
        "The run uses the exact completed clean affine-linear PA graph cache for every `(N, alpha)` case. One deterministic iid `Uniform[-1,1)` vector is reused across the eight raw amplitudes on that graph. Different amplitudes are paired observations, not independent disorder realizations. The clean local-mode/tunneling construction is outside this analysis and did not gate any operator.",
        "",
        "All positive-disorder operators retain complete ascending eigenvalues and float64 eigenvectors. The clean baseline is referenced without duplicating its full eigensystem. Every completed operator passed symmetry, diagonal-bound, unchanged-off-diagonal, trace, squared-trace, residual, norm, sampled-orthogonality, localization-bound, and output-completeness checks.",
        "",
        "## Interpretation limits",
        "",
        "This is one graph and one potential stream per condition. It supports matched descriptive comparisons, not graph-level uncertainty, a universal mobility edge, or a critical amplitude. Matching ranks or quantiles across `a` does not track a continuous eigenstate. Clean zero-atom individual vectors are solver-basis dependent and are excluded from primary localization summaries; positive-disorder near-zero states are treated through actual numerical multiplicity rather than the clean atom mask.",
        "",
        "## Results navigation",
        "",
        "- `tables/completion_matrix_14x8.csv`: every graph-amplitude status.",
        "- `tables/all_cases_operator_summary.csv`: whole-spectrum scalar comparison.",
        "- `tables/all_cases_upper_edge_summary.csv.gz`: strict-internal rank-window results.",
        "- `RESULTS_INDEX.md`: exact case, table, figure, array, configuration, and manifest paths.",
        "",
    ]
    if blockers:
        summary_lines.extend(["## Explicit blockers", ""] + [f"- Missing completed case aggregate: `{path}`" for path in blockers] + [""])
    if figure_paths:
        summary_lines.extend(["## Cross-case figure", "", f"![Cross-alpha comparison]({figure_paths[0].relative_to(report)})", ""])
    (report / "PA_DIAGONAL_UNIFORM_NOISE_REPORT.md").write_text("\n".join(summary_lines))

    index_lines = [
        "# Results index",
        "",
        f"Execution root: `{run_root}`",
        "",
        "## Global outputs",
        "",
        "- `PA_DIAGONAL_UNIFORM_NOISE_REPORT.md`",
        "- `tables/completion_matrix_14x8.csv`",
        "- `tables/all_cases_operator_summary.csv`",
        "- `tables/all_cases_upper_edge_summary.csv.gz`",
    ]
    index_lines.extend(f"- `{path.relative_to(report)}`" for path in figure_paths)
    index_lines.extend(["", "## Per-case reports", ""])
    index_lines.extend(f"- `{path.relative_to(run_root)}`" for path in case_reports)
    index_lines.extend(["", "## Operator-level arrays and tables", "", "Each `cases/N_<N>/alpha_<slug>/operators/a_<slug>/` directory contains a complete run manifest and tables. Positive amplitudes additionally contain full eigenvalues, full float64 eigenvectors, and full residual arrays. The a=0 manifest points to the validated clean arrays without copying them.", ""])
    (report / "RESULTS_INDEX.md").write_text("\n".join(index_lines))
    inventory = _sha_inventory(report)
    atomic_write_csv(report / "EVIDENCE_INVENTORY.csv", inventory)
    manifest = {
        "status": "COMPLETE" if complete_count == len(completion) and not blockers else "PARTIAL",
        "stage": "global_aggregate_report_and_index",
        "version": PA_UNIFORM_NOISE_VERSION,
        "reporting_version": REPORTING_VERSION,
        "source_identity": source_identity,
        "completed_operator_count": complete_count,
        "requested_operator_count": len(completion),
        "completed_case_report_count": len(case_reports),
        "requested_case_report_count": len(config["sizes"]) * len(config["degree_ccdf_alpha_grid"]),
        "blockers": blockers,
        "finished_at_utc": utc_now(),
    }
    atomic_write_json(report / "run_manifest.json", manifest)
    return report
