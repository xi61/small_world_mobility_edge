"""Reprocess legacy affine-PA eigenspectra without eigendecomposition."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
import tempfile
from pathlib import Path
import shutil

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "small-world-mobility-edge-pa-mobility"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..empirical_spectral_density import estimate_variable_local_count_density
from .mobility_edge_postprocess import (
    ANALYSIS_VERSION,
    GOE_MEAN_GAP_RATIO,
    POISSON_MEAN_GAP_RATIO,
    build_cross_fitted_models,
    build_gap_records,
    build_unfolded_gap_cache,
    detect_candidate_regions,
    graph_level_quantile_rows,
    join_localization_and_level_statistics,
    save_crossfit_models,
    summarize_adaptive_window,
    summarize_quantile_grid,
    write_rows,
)


LEGACY_PARAMETERS = {
    1.8: ("n_5000_m_10_preference_tau_1p8_avgdeg_19p978", 27282),
    1.5: ("n_5000_m_10_preference_tau_1p5_avgdeg_19p978", 27352),
    1.05: ("n_5000_m_10_preference_tau_1p05_avgdeg_19p978", 27307),
}
N = 5000
M = 10
NUM_GRAPHS = 500
Q_WIDTH = 0.002
ZERO_ATOL = 1e-8


def slug(value: float) -> str:
    return str(value).replace(".", "p")


def make_seeds(base_seed: int) -> list[int]:
    rng = np.random.default_rng(base_seed)
    values = rng.integers(0, 2**32 - 1, size=NUM_GRAPHS, dtype=np.uint32)
    return [int(value) for value in values]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_legacy_spectra(
    source_root: Path, alpha: float
) -> tuple[list[np.ndarray], list[int], Path]:
    if alpha not in LEGACY_PARAMETERS:
        raise ValueError(f"no legacy cache mapping for alpha={alpha}")
    dirname, seed_base = LEGACY_PARAMETERS[alpha]
    parameter_dir = source_root / dirname
    paths = sorted((parameter_dir / "partials").glob("spectral_arrays_*.npz"))
    if len(paths) != 100:
        raise FileNotFoundError(f"expected 100 spectral chunks, found {len(paths)}")
    expected_seeds = make_seeds(seed_base)
    by_index: dict[int, np.ndarray] = {}
    seeds_by_index: dict[int, int] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            indices = np.asarray(data["graph_indices"], dtype=int)
            seeds = np.asarray(data["graph_seeds"], dtype=np.uint64)
            values = np.asarray(data["nonperron_eigenvalues"], dtype=float)
        if values.shape != (len(indices), N - 1):
            raise ValueError(f"invalid eigenspectrum shape in {path}: {values.shape}")
        for graph_index, graph_seed, spectrum in zip(indices, seeds, values):
            index = int(graph_index)
            if index in by_index:
                raise ValueError(f"duplicate graph index {index}")
            if int(graph_seed) != expected_seeds[index]:
                raise ValueError(f"seed mismatch for graph {index}")
            if np.any(np.diff(spectrum) < -1e-12):
                raise ValueError(f"unsorted spectrum for graph {index}")
            by_index[index] = spectrum
            seeds_by_index[index] = int(graph_seed)
    if sorted(by_index) != list(range(NUM_GRAPHS)):
        raise ValueError("legacy cache does not cover graph indices 0..499")
    return (
        [by_index[index] for index in range(NUM_GRAPHS)],
        [seeds_by_index[index] for index in range(NUM_GRAPHS)],
        parameter_dir,
    )


def write_global_spectrum(
    raw_samples: list[np.ndarray],
    continuous_samples: list[np.ndarray],
    output_dir: Path,
    *,
    model_label: str = "affine preferential attachment",
) -> dict[str, object]:
    global_dir = output_dir / "global_spectrum"
    global_dir.mkdir(parents=True, exist_ok=True)
    pooled_raw = np.concatenate(raw_samples)
    pooled_continuous = np.concatenate(continuous_samples)
    density = estimate_variable_local_count_density(
        continuous_samples,
        num_graphs=len(continuous_samples),
        per_graph_dimension=float(np.mean([len(values) for values in continuous_samples])),
        bulk_local_count_half_width=0.20,
        tail_local_count_half_width=0.08,
        tail_quantile_cutoff=0.95,
        num_points=2001,
    )
    np.savez_compressed(
        global_dir / "empirical_density_cdf.npz",
        grid=density.grid,
        density=density.density,
        cdf=density.cdf,
    )
    raw_hist, raw_edges = np.histogram(pooled_raw, bins=240, density=False)
    continuous_hist, continuous_edges = np.histogram(
        pooled_continuous, bins=240, density=True
    )
    np.savez_compressed(
        global_dir / "global_esd_histograms.npz",
        raw_counts=raw_hist,
        raw_edges=raw_edges,
        normalized_continuous_density=continuous_hist,
        continuous_edges=continuous_edges,
    )
    figure, axis = plt.subplots(figsize=(9.0, 5.4))
    axis.plot(density.grid, density.density, color="black", lw=1.3)
    axis.set_xlabel("energy E")
    axis.set_ylabel("normalized continuous density")
    axis.set_title(f"{model_label} continuous non-Perron empirical spectral density")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(global_dir / "global_continuous_esd.png", dpi=180)
    plt.close(figure)
    positive = pooled_continuous[pooled_continuous > 0.0]
    negative_abs = np.abs(pooled_continuous[pooled_continuous < 0.0])
    return {
        "raw_spectral_min": float(np.min(pooled_raw)),
        "raw_spectral_max": float(np.max(pooled_raw)),
        "continuous_spectral_min": float(np.min(pooled_continuous)),
        "continuous_spectral_max": float(np.max(pooled_continuous)),
        "mean_continuous_dimension": float(
            np.mean([len(values) for values in continuous_samples])
        ),
        "cdf_start": float(density.cdf[0]),
        "cdf_end": float(density.cdf[-1]),
        "continuous_positive_count": int(len(positive)),
        "continuous_negative_count": int(len(negative_abs)),
        "continuous_positive_fraction": float(
            len(positive) / max(len(pooled_continuous), 1)
        ),
        "positive_to_negative_edge_magnitude_ratio": float(
            np.max(positive) / np.max(negative_abs)
        ) if len(positive) and len(negative_abs) else math.nan,
        "positive_to_negative_median_magnitude_ratio": float(
            np.median(positive) / np.median(negative_abs)
        ) if len(positive) and len(negative_abs) else math.nan,
    }


def histogram_rows(
    histograms: dict[int, np.ndarray], rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    edges = np.linspace(0.0, 3.5, 48)
    output: list[dict[str, object]] = []
    for index, values in sorted(histograms.items()):
        counts, _ = np.histogram(values, bins=edges)
        density = counts / (np.sum(counts) * np.diff(edges)) if np.sum(counts) else counts
        for bin_index, value in enumerate(density):
            output.append(
                {
                    "coarse_bin_index": index,
                    "q_low": rows[index]["actual_q_low"],
                    "q_high": rows[index]["actual_q_high"],
                    "spacing_low": edges[bin_index],
                    "spacing_high": edges[bin_index + 1],
                    "density": float(value),
                    "sample_count": len(values),
                }
            )
    return output


def plot_coarse(
    joined: list[dict[str, object]],
    parameter_value: float,
    output_dir: Path,
    *,
    model_label: str = "Affine PA",
    parameter_name: str = "degree CCDF alpha",
) -> None:
    q = np.asarray([float(row["q_center"]) for row in joined])
    energy = np.asarray([float(row["E_mean"]) for row in joined])
    nipr = np.asarray(
        [float(row.get("eigenvector_mean_n_times_ipr", math.nan)) for row in joined]
    )
    raw_r = np.asarray(
        [float(row["graph_equal_mean_raw_r_positive_conditioned"]) for row in joined]
    )
    zero = np.asarray([float(row["zero_gap_fraction"]) for row in joined])
    coarse_dir = output_dir / "coarse" / "q0_1_dq_0p002"
    coarse_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 1, figsize=(10.0, 8.0), sharex=True)
    axes[0].plot(q, nipr, color="black", lw=1.2)
    axes[0].set_ylabel(r"mean $N\,\mathrm{IPR}$")
    axes[0].set_yscale("log")
    axes[1].plot(q, raw_r, color="tab:blue", lw=1.2, label="positive-gap raw r")
    axes[1].axhline(GOE_MEAN_GAP_RATIO, color="black", ls="--", label="GOE")
    axes[1].axhline(POISSON_MEAN_GAP_RATIO, color="firebrick", ls=":", label="Poisson")
    axes[1].set_xlabel("continuous non-Perron spectral quantile q")
    axes[1].set_ylabel("graph-equal mean raw r")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle(
        f"{model_label} localization and raw gaps: {parameter_name}={parameter_value:g}"
    )
    figure.tight_layout()
    figure.savefig(coarse_dir / "localization_and_raw_r_vs_quantile.png", dpi=180)
    plt.close(figure)

    finite = np.isfinite(energy)
    figure, axes = plt.subplots(2, 1, figsize=(10.0, 8.0), sharex=True)
    axes[0].plot(energy[finite], nipr[finite], color="black", lw=1.2)
    axes[0].set_ylabel(r"mean $N\,\mathrm{IPR}$")
    axes[0].set_yscale("log")
    axes[1].plot(energy[finite], raw_r[finite], color="tab:blue", lw=1.2)
    axes[1].axhline(GOE_MEAN_GAP_RATIO, color="black", ls="--")
    axes[1].axhline(POISSON_MEAN_GAP_RATIO, color="firebrick", ls=":")
    axes[1].set_xlabel("gap-midpoint energy E")
    axes[1].set_ylabel("graph-equal mean raw r")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle(
        f"{model_label} localization and raw gaps in energy: {parameter_name}={parameter_value:g}"
    )
    figure.tight_layout()
    figure.savefig(coarse_dir / "localization_and_raw_r_vs_energy.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10.0, 4.8))
    axis.plot(q, zero, color="tab:red")
    axis.set_xlabel("continuous non-Perron spectral quantile q")
    axis.set_ylabel("zero-gap fraction")
    axis.set_title(f"Exact-degeneracy audit: {parameter_name}={parameter_value:g}")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(coarse_dir / "zero_gap_fraction_vs_quantile.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.8), sharex=True)
    axes[0, 0].plot(q, nipr, color="black")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel(r"mean $N\,\mathrm{IPR}$")
    axes[0, 0].set_title("Inverse participation ratio")
    axes[0, 1].plot(
        q,
        [float(row.get("eigenvector_mean_participation_fraction", math.nan)) for row in joined],
        color="tab:blue",
    )
    axes[0, 1].set_ylabel("PR/N")
    axes[0, 1].set_title("Participation fraction")
    for top_r in (10, 50, 100, 200):
        axes[1, 0].plot(
            q,
            [float(row.get(f"eigenvector_mean_top_{top_r}_mass", math.nan)) for row in joined],
            label=f"top {top_r}",
        )
    axes[1, 0].set_ylabel("cumulative coordinate mass")
    axes[1, 0].set_title("Top-r mass")
    axes[1, 0].legend(frameon=False, ncol=2)
    for mass in (50, 80, 90, 95):
        axes[1, 1].plot(
            q,
            [float(row.get(f"eigenvector_mean_r_{mass}_fraction", math.nan)) for row in joined],
            label=rf"$r_{{{mass}}}/N$",
        )
    axes[1, 1].set_ylabel("minimum coordinate fraction")
    axes[1, 1].set_title("Coordinates needed for target mass")
    axes[1, 1].legend(frameon=False, ncol=2)
    for axis in axes.flat:
        axis.set_xlabel("continuous non-Perron spectral quantile q")
        axis.grid(alpha=0.2)
    figure.suptitle(
        f"{model_label} eigenvector localization: {parameter_name}={parameter_value:g}"
    )
    figure.tight_layout()
    figure.savefig(coarse_dir / "eigenvector_localization_metrics.png", dpi=180)
    plt.close(figure)


def fine_candidate_scans(
    records,
    models,
    candidates: list[dict[str, object]],
    output_dir: Path,
) -> list[dict[str, object]]:
    fine_root = output_dir / "fine"
    unfolded = build_unfolded_gap_cache(records, models)
    all_rows: list[dict[str, object]] = []
    plotted: dict[tuple[int, float], list[tuple[dict[str, object], np.ndarray]]] = {}
    for candidate_index, candidate in enumerate(candidates):
        if candidate["category"] in {"NO_SUSTAINED_SIGNAL", "ATOM_CONTAMINATED"}:
            continue
        region_low = float(candidate["q_low"])
        region_high = float(candidate["q_high"])
        for width in (0.001, 0.0005, 0.0002):
            count = max(1, int(math.ceil((region_high - region_low) / width)))
            for cell in range(count):
                q_low = region_low + cell * width
                q_high = min(q_low + width, region_high)
                if q_low >= q_high:
                    continue
                row, histogram = summarize_adaptive_window(
                    records,
                    nominal_q_low=q_low,
                    nominal_q_high=q_high,
                    expansion_step=width,
                    crossfit_models=models,
                    minimum_positive_gaps=100,
                    unfolded_gap_cache=unfolded,
                )
                row["candidate_index"] = candidate_index
                row["nominal_width"] = width
                row["candidate_category"] = candidate["category"]
                row["histogram_values_file"] = ""
                if row["histogram_sample_status"] == "OK":
                    destination = (
                        fine_root
                        / f"candidate_{candidate_index:02d}"
                        / f"dq_{slug(width)}"
                    )
                    destination.mkdir(parents=True, exist_ok=True)
                    filename = f"spacing_q_{slug(q_low)}_{slug(q_high)}.npz"
                    np.savez_compressed(
                        destination / filename,
                        normalized_positive_crossfit_spacings=histogram,
                        nominal_q_bounds=np.asarray([q_low, q_high]),
                        actual_q_bounds=np.asarray(
                            [row["actual_q_low"], row["actual_q_high"]]
                        ),
                    )
                    row["histogram_values_file"] = str(
                        (destination / filename).relative_to(output_dir)
                    )
                    plotted.setdefault((candidate_index, width), []).append(
                        (row, np.asarray(histogram, dtype=float))
                    )
                all_rows.append(row)
    if all_rows:
        write_rows(fine_root / "fine_window_summary.csv", all_rows)
    x = np.linspace(0.0, 3.5, 500)
    goe = 0.5 * math.pi * x * np.exp(-math.pi * np.square(x) / 4.0)
    poisson = np.exp(-x)
    for (candidate_index, width), entries in plotted.items():
        take = np.linspace(0, len(entries) - 1, min(12, len(entries)), dtype=int)
        figure, axes = plt.subplots(3, 4, figsize=(14.5, 10.0), sharex=True, sharey=True)
        for axis in axes.flat:
            axis.set_visible(False)
        for axis, selected_index in zip(axes.flat, take):
            row, values = entries[int(selected_index)]
            axis.set_visible(True)
            displayed = values[(values >= 0.0) & (values <= 3.5)]
            axis.hist(
                displayed,
                bins=np.linspace(0.0, 3.5, 30),
                density=True,
                color="#52b788",
                alpha=0.75,
            )
            axis.plot(x, goe, color="black", lw=1.1, label="GOE")
            axis.plot(x, poisson, color="firebrick", ls="--", lw=1.1, label="Poisson")
            axis.set_title(
                "nominal q=[%.4f, %.4f]\nactual q=[%.4f, %.4f], n=%d"
                % (
                    row["nominal_q_low"],
                    row["nominal_q_high"],
                    row["actual_q_low"],
                    row["actual_q_high"],
                    row["crossfit_unfolded_histogram_sample_count"],
                ),
                fontsize=8,
            )
            axis.grid(alpha=0.15)
        visible = [axis for axis in axes.flat if axis.get_visible()]
        if visible:
            visible[0].legend(frameon=False, fontsize=8)
        figure.suptitle(
            f"Fine cross-fitted spacings: candidate {candidate_index}, nominal dq={width:g}"
        )
        figure.tight_layout()
        figure.savefig(
            fine_root
            / f"fine_candidate_{candidate_index:02d}_dq_{slug(width)}_histograms.png",
            dpi=180,
        )
        plt.close(figure)
    return all_rows


def reprocess_parameter(
    *,
    source_root: Path,
    output_root: Path,
    alpha: float,
    provenance_root: Path | None = None,
) -> Path:
    raw_samples, seeds, legacy_dir = load_legacy_spectra(source_root, alpha)
    records = [
        build_gap_records(
            values,
            graph_index=index,
            graph_seed=seeds[index],
            excluded_atom_energies=(0.0,),
            audited_atom_energies=(-1.0, 0.0, 1.0),
            atom_atol=ZERO_ATOL,
        )
        for index, values in enumerate(raw_samples)
    ]
    parameter_dir = (
        output_root
        / "affine_linear"
        / "n_5000_m_10"
        / f"degree_ccdf_alpha_{slug(alpha)}"
    )
    parameter_dir.mkdir(parents=True, exist_ok=True)
    models = build_cross_fitted_models(records, num_folds=5)
    save_crossfit_models(
        parameter_dir / "unfolding" / "crossfit_k5",
        models,
        metadata={
            "degree_ccdf_alpha": alpha,
            "source_cache": str(legacy_dir),
            "bulk_half_width": 0.20,
            "tail_half_width": 0.08,
            "tail_quantile_cutoff": 0.95,
            "excluded_atoms": [0.0],
        },
    )
    level_rows, histograms = summarize_quantile_grid(
        records,
        width=Q_WIDTH,
        crossfit_models=models,
        minimum_positive_gaps=100,
    )
    coarse_dir = parameter_dir / "coarse" / "q0_1_dq_0p002"
    write_rows(coarse_dir / "raw_r_crossfit_spacing_summary.csv", level_rows)
    write_rows(
        coarse_dir / "raw_r_crossfit_spacing_by_graph.csv.gz",
        graph_level_quantile_rows(
            records, width=Q_WIDTH, crossfit_models=models
        ),
    )
    histogram_table = histogram_rows(histograms, level_rows)
    if histogram_table:
        write_rows(coarse_dir / "crossfit_spacing_histogram_density.csv", histogram_table)

    localization_path = legacy_dir / "eigenvector_percentile_scan_summary.csv"
    if not localization_path.is_file():
        raise FileNotFoundError(localization_path)
    localization_rows = read_csv(localization_path)
    legacy_graph_localization = legacy_dir / "eigenvector_percentile_scan_by_graph.csv.gz"
    if legacy_graph_localization.is_file():
        shutil.copyfile(
            legacy_graph_localization,
            coarse_dir / "eigenvector_percentile_scan_by_graph.csv.gz",
        )
    legacy_heatmap = legacy_dir / "eigenvector_coordinate_mass_heatmap.png"
    if legacy_heatmap.is_file():
        shutil.copyfile(
            legacy_heatmap,
            coarse_dir / "eigenvector_coordinate_mass_heatmap_legacy_reused.png",
        )
    joined = join_localization_and_level_statistics(localization_rows, level_rows)
    write_rows(coarse_dir / "joined_localization_level_statistics.csv", joined)
    candidates = detect_candidate_regions(joined)
    (coarse_dir / "candidate_regions.json").write_text(
        json.dumps(candidates, indent=2, sort_keys=True) + "\n"
    )
    fine_rows = fine_candidate_scans(records, models, candidates, parameter_dir)

    atom_rows = []
    for record in records:
        atom_rows.append(
            {
                "graph_index": record.graph_index,
                "graph_seed": record.graph_seed,
                "minus_one_multiplicity": record.audited_atom_counts["-1"],
                "zero_multiplicity": record.audited_atom_counts["+0"],
                "plus_one_multiplicity": record.audited_atom_counts["+1"],
                "continuous_nonperron_dimension": record.continuous_dimension,
            }
        )
    write_rows(parameter_dir / "global_spectrum" / "atom_multiplicity_by_graph.csv", atom_rows)
    global_summary = write_global_spectrum(
        raw_samples,
        [record.continuous_eigenvalues for record in records],
        parameter_dir,
    )
    plot_coarse(joined, alpha, parameter_dir)

    source_cache_destination = parameter_dir / "source_cache.json"
    if provenance_root is not None:
        candidate = provenance_root / f"degree_ccdf_alpha_{slug(alpha)}" / "source_cache.json"
        if candidate.is_file():
            shutil.copy2(candidate, source_cache_destination)
    if not source_cache_destination.is_file():
        source_cache_destination.write_text(
            json.dumps(
                {
                    "status": "VALIDATED_BY_REPROCESSOR",
                    "source_cache": str(legacy_dir),
                    "degree_ccdf_alpha": alpha,
                    "graph_indices": [0, NUM_GRAPHS - 1],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    existing_manifest_path = parameter_dir / "run_manifest.json"
    existing_manifest = (
        json.loads(existing_manifest_path.read_text())
        if existing_manifest_path.is_file()
        else {}
    )
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "result_origin": "legacy eigenspectrum cache; no eigendecomposition",
        "model": "affine linear preferential attachment",
        "canonical_parameter": {"name": "degree_ccdf_alpha", "value": alpha},
        "legacy_parameter": {"name": "attachment_tail_exponent", "value": alpha},
        "attachment_offset_delta": M * (alpha - 2.0),
        "n": N,
        "m": M,
        "num_graphs": NUM_GRAPHS,
        "graph_seed_base": LEGACY_PARAMETERS[alpha][1],
        "source_cache": str(legacy_dir),
        "gap_assignment": "gap-first midpoint q=(i+1/2)/M and midpoint energy",
        "ratio_assignment": "middle eigenvalue q=(i+1)/M and E=lambda_i+1",
        "raw_ratio_primary": True,
        "atom_filtering": {"excluded_from_continuous": [0.0], "atol": ZERO_ATOL},
        "unfolding": "five-fold graph-level cross-fitted variable local-count empirical CDF",
        "crossfit_fold_rule": "graph_index % 5",
        "minimum_positive_gaps_per_displayed_histogram": 100,
        "legacy_localization_reused": True,
        "fine_localization_available": False,
        "candidate_regions": candidates,
        "fine_spacing_window_count": len(fine_rows),
        "global_spectrum_summary": global_summary,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "postprocess_sge_job_id": os.environ.get("JOB_ID", ""),
        "postprocess_sge_task_id": os.environ.get("SGE_TASK_ID", ""),
    }
    if "structural_replay" in existing_manifest:
        manifest["structural_replay"] = existing_manifest["structural_replay"]
    (parameter_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return parameter_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--provenance-root", type=Path)
    parser.add_argument("--degree-ccdf-alpha", type=float, action="append")
    args = parser.parse_args()
    alphas = args.degree_ccdf_alpha or [1.8, 1.5, 1.05]
    for alpha in alphas:
        destination = reprocess_parameter(
            source_root=args.source_root,
            output_root=args.output_root,
            alpha=float(alpha),
            provenance_root=args.provenance_root,
        )
        print(f"reprocessed alpha={alpha:g} into {destination}", flush=True)


if __name__ == "__main__":
    main()
