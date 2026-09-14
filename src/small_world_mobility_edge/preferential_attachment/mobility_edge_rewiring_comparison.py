"""Paired original-versus-exact-degree-rewired mobility-edge summaries."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp

from .mobility_edge_postprocess import (
    GOE_MEAN_GAP_RATIO,
    POISSON_MEAN_GAP_RATIO,
    write_rows,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _mean_se(values: list[float]) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if not len(data):
        return math.nan, math.nan
    return (
        float(np.mean(data)),
        float(np.std(data, ddof=1) / math.sqrt(len(data))) if len(data) > 1 else math.nan,
    )


def _q_key(row: dict[str, str]) -> tuple[int, float, float]:
    low = row.get("quantile_low", row.get("q_low", "nan"))
    high = row.get("quantile_high", row.get("q_high", "nan"))
    return int(row["graph_index"]), round(float(low), 12), round(float(high), 12)


def _paired_rows(
    original_dir: Path, rewired_dir: Path
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    coarse_rel = Path("coarse/q0_1_dq_0p002")
    original_loc = {
        _q_key(row): row
        for row in _read_csv(original_dir / coarse_rel / "eigenvector_percentile_scan_by_graph.csv.gz")
    }
    rewired_loc = {
        _q_key(row): row
        for row in _read_csv(rewired_dir / coarse_rel / "eigenvector_percentile_scan_by_graph.csv.gz")
    }
    original_level = {
        _q_key(row): row
        for row in _read_csv(original_dir / coarse_rel / "raw_r_crossfit_spacing_by_graph.csv.gz")
    }
    rewired_level = {
        _q_key(row): row
        for row in _read_csv(rewired_dir / coarse_rel / "raw_r_crossfit_spacing_by_graph.csv.gz")
    }
    common = sorted(set(original_loc) & set(rewired_loc) & set(original_level) & set(rewired_level))
    if not common:
        raise ValueError("no matched graph/bin rows between original and rewired results")
    detailed: list[dict[str, object]] = []
    for graph_index, q_low, q_high in common:
        left_loc, right_loc = original_loc[(graph_index, q_low, q_high)], rewired_loc[(graph_index, q_low, q_high)]
        left_level, right_level = original_level[(graph_index, q_low, q_high)], rewired_level[(graph_index, q_low, q_high)]
        source = {
            "n_times_ipr": (left_loc, right_loc, "mean_n_times_ipr"),
            "top_50_mass": (left_loc, right_loc, "mean_top_50_mass"),
            "top_200_mass": (left_loc, right_loc, "mean_top_200_mass"),
            "raw_r_positive_conditioned": (
                left_level,
                right_level,
                "mean_raw_r_positive_conditioned",
            ),
            "crossfit_unfolded_positive_spacing": (
                left_level,
                right_level,
                "mean_crossfit_unfolded_positive_spacing",
            ),
        }
        row: dict[str, object] = {
            "graph_index": graph_index,
            "graph_seed": int(left_loc["graph_seed"]),
            "q_low": q_low,
            "q_high": q_high,
            "q_center": 0.5 * (q_low + q_high),
            "difference_definition": "rewired_minus_original",
        }
        for output_name, (left, right, field) in source.items():
            original = float(left.get(field, "nan"))
            rewired = float(right.get(field, "nan"))
            row[f"original_{output_name}"] = original
            row[f"rewired_{output_name}"] = rewired
            row[f"delta_{output_name}"] = rewired - original
        detailed.append(row)
    summaries: list[dict[str, object]] = []
    for q_low, q_high in sorted({(row["q_low"], row["q_high"]) for row in detailed}):
        selected = [row for row in detailed if row["q_low"] == q_low and row["q_high"] == q_high]
        summary: dict[str, object] = {
            "q_low": q_low,
            "q_high": q_high,
            "q_center": 0.5 * (q_low + q_high),
            "paired_graph_count": len(selected),
            "difference_definition": "rewired_minus_original",
        }
        for metric in (
            "n_times_ipr",
            "top_50_mass",
            "top_200_mass",
            "raw_r_positive_conditioned",
            "crossfit_unfolded_positive_spacing",
        ):
            for prefix in ("original", "rewired", "delta"):
                mean, se = _mean_se([float(row[f"{prefix}_{metric}"]) for row in selected])
                summary[f"mean_{prefix}_{metric}"] = mean
                summary[f"se_{prefix}_{metric}"] = se
        summaries.append(summary)
    return detailed, summaries


def _load_chunk_cache(partial_dir: Path) -> dict[int, dict[str, object]]:
    output: dict[int, dict[str, object]] = {}
    mobility = sorted(partial_dir.glob("mobility_cache_*.npz"))
    legacy = sorted(partial_dir.glob("spectral_arrays_*.npz"))
    paths = mobility or legacy
    if not paths:
        raise FileNotFoundError(f"no spectral chunks below {partial_dir}")
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            indices = np.asarray(data["graph_indices"], dtype=int)
            seeds = np.asarray(data["graph_seeds"], dtype=np.uint64)
            spectra = np.asarray(data["nonperron_eigenvalues"], dtype=float)
            degree_arrays = (
                np.asarray(data["degree_arrays"], dtype=np.int32)
                if "degree_arrays" in data.files
                else None
            )
        for local, graph_index in enumerate(indices):
            output[int(graph_index)] = {
                "graph_seed": int(seeds[local]),
                "spectrum": spectra[local],
                "degrees": degree_arrays[local] if degree_arrays is not None else None,
            }
    return output


def _load_replay_degrees(parameter_dir: Path) -> dict[int, np.ndarray]:
    output: dict[int, np.ndarray] = {}
    for path in sorted(
        (parameter_dir / "structural" / "replay_partials").glob("degree_arrays_*.npz")
    ):
        with np.load(path, allow_pickle=False) as data:
            indices = np.asarray(data["graph_indices"], dtype=int)
            degrees = np.asarray(data["degree_arrays"], dtype=np.int32)
        for local, graph_index in enumerate(indices):
            output[int(graph_index)] = degrees[local]
    return output


def _source_partial_dir(parameter_dir: Path) -> Path:
    source = json.loads((parameter_dir / "source_cache.json").read_text())
    root = Path(source["source_cache"])
    return root if root.name == "partials" else root / "partials"


def _paired_spectrum_rows(
    original_dir: Path, rewired_dir: Path
) -> tuple[list[dict[str, object]], dict[str, object]]:
    original = _load_chunk_cache(_source_partial_dir(original_dir))
    rewired = _load_chunk_cache(rewired_dir / "partials")
    replay_degrees = _load_replay_degrees(original_dir)
    rows: list[dict[str, object]] = []
    for graph_index in sorted(set(original) & set(rewired)):
        left = original[graph_index]
        right = rewired[graph_index]
        if int(left["graph_seed"]) != int(right["graph_seed"]):
            raise ValueError(f"graph seed mismatch for paired graph {graph_index}")
        left_spectrum = np.sort(np.asarray(left["spectrum"], dtype=float))
        right_spectrum = np.sort(np.asarray(right["spectrum"], dtype=float))
        if len(left_spectrum) != len(right_spectrum):
            raise ValueError(f"spectrum length mismatch for graph {graph_index}")
        left_degrees = left["degrees"]
        if left_degrees is None:
            left_degrees = replay_degrees.get(graph_index)
        right_degrees = right["degrees"]
        degree_match = (
            left_degrees is not None
            and right_degrees is not None
            and np.array_equal(left_degrees, right_degrees)
        )
        rank_difference = right_spectrum - left_spectrum
        ks_result = ks_2samp(left_spectrum, right_spectrum)
        rows.append(
            {
                "graph_index": graph_index,
                "graph_seed": int(left["graph_seed"]),
                "exact_labeled_degree_match": degree_match,
                "nonperron_esd_KS_distance": float(ks_result.statistic),
                "mean_absolute_rank_matched_eigenvalue_difference": float(
                    np.mean(np.abs(rank_difference))
                ),
                "rms_rank_matched_eigenvalue_difference": float(
                    np.sqrt(np.mean(np.square(rank_difference)))
                ),
                "original_lower_edge": float(left_spectrum[0]),
                "rewired_lower_edge": float(right_spectrum[0]),
                "delta_lower_edge": float(right_spectrum[0] - left_spectrum[0]),
                "original_upper_nonperron_edge": float(left_spectrum[-1]),
                "rewired_upper_nonperron_edge": float(right_spectrum[-1]),
                "delta_upper_nonperron_edge": float(
                    right_spectrum[-1] - left_spectrum[-1]
                ),
                "original_zero_multiplicity": int(
                    np.sum(np.isclose(left_spectrum, 0.0, atol=1e-8, rtol=0.0))
                ),
                "rewired_zero_multiplicity": int(
                    np.sum(np.isclose(right_spectrum, 0.0, atol=1e-8, rtol=0.0))
                ),
                "original_plus_one_multiplicity": int(
                    np.sum(np.isclose(left_spectrum, 1.0, atol=1e-8, rtol=0.0))
                ),
                "rewired_plus_one_multiplicity": int(
                    np.sum(np.isclose(right_spectrum, 1.0, atol=1e-8, rtol=0.0))
                ),
                "original_minus_one_multiplicity": int(
                    np.sum(np.isclose(left_spectrum, -1.0, atol=1e-8, rtol=0.0))
                ),
                "rewired_minus_one_multiplicity": int(
                    np.sum(np.isclose(right_spectrum, -1.0, atol=1e-8, rtol=0.0))
                ),
            }
        )
    if len(rows) != len(original) or len(rows) != len(rewired):
        raise ValueError("original/rewired graph coverage is not identical")
    summary: dict[str, object] = {
        "paired_graph_count": len(rows),
        "exact_labeled_degree_match_for_all_graphs": all(
            bool(row["exact_labeled_degree_match"]) for row in rows
        ),
    }
    for field in (
        "nonperron_esd_KS_distance",
        "mean_absolute_rank_matched_eigenvalue_difference",
        "rms_rank_matched_eigenvalue_difference",
        "delta_lower_edge",
        "delta_upper_nonperron_edge",
        "original_zero_multiplicity",
        "rewired_zero_multiplicity",
        "original_plus_one_multiplicity",
        "rewired_plus_one_multiplicity",
        "original_minus_one_multiplicity",
        "rewired_minus_one_multiplicity",
    ):
        mean, se = _mean_se([float(row[field]) for row in rows])
        summary[f"mean_{field}"] = mean
        summary[f"se_{field}"] = se
    return rows, summary


def build_rewiring_comparison(parameter_dir: Path) -> Path:
    original_dir = parameter_dir
    rewired_dir = parameter_dir / "rewired"
    output_dir = parameter_dir / "rewiring_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    original_manifest = json.loads((original_dir / "run_manifest.json").read_text())
    rewired_manifest = json.loads((rewired_dir / "run_manifest.json").read_text())
    if original_manifest["canonical_parameter"] != rewired_manifest["canonical_parameter"]:
        raise ValueError("original and rewired canonical parameters differ")
    if original_manifest["graph_seed_base"] != rewired_manifest["graph_seed_base"]:
        raise ValueError("original and rewired graph seed bases differ")

    detailed, summaries = _paired_rows(original_dir, rewired_dir)
    write_rows(output_dir / "paired_coarse_by_graph.csv.gz", detailed)
    write_rows(output_dir / "paired_coarse_summary.csv", summaries)
    spectrum_rows, spectrum_summary = _paired_spectrum_rows(original_dir, rewired_dir)
    write_rows(output_dir / "paired_global_spectrum_by_graph.csv", spectrum_rows)
    (output_dir / "paired_global_spectrum_summary.json").write_text(
        json.dumps(spectrum_summary, indent=2, sort_keys=True) + "\n"
    )

    original_density = np.load(original_dir / "global_spectrum/empirical_density_cdf.npz")
    rewired_density = np.load(rewired_dir / "global_spectrum/empirical_density_cdf.npz")
    figure, axes = plt.subplots(3, 1, figsize=(9.5, 11.0))
    axes[0].plot(original_density["grid"], original_density["density"], label="original")
    axes[0].plot(rewired_density["grid"], rewired_density["density"], label="exact-degree rewired")
    axes[0].set_xlabel("energy E")
    axes[0].set_ylabel("normalized continuous density")
    axes[0].legend(frameon=False)
    q = [float(row["q_center"]) for row in summaries]
    axes[1].plot(q, [float(row["mean_original_n_times_ipr"]) for row in summaries], label="original")
    axes[1].plot(q, [float(row["mean_rewired_n_times_ipr"]) for row in summaries], label="exact-degree rewired")
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"mean $N\,\mathrm{IPR}$")
    axes[1].legend(frameon=False)
    axes[2].plot(q, [float(row["mean_original_raw_r_positive_conditioned"]) for row in summaries], label="original")
    axes[2].plot(q, [float(row["mean_rewired_raw_r_positive_conditioned"]) for row in summaries], label="exact-degree rewired")
    axes[2].axhline(GOE_MEAN_GAP_RATIO, color="black", ls="--", label="GOE")
    axes[2].axhline(POISSON_MEAN_GAP_RATIO, color="firebrick", ls=":", label="Poisson")
    axes[2].set_xlabel("continuous non-Perron spectral quantile q")
    axes[2].set_ylabel("graph-equal mean raw r")
    axes[2].legend(frameon=False, ncol=2)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "original_vs_exact_degree_rewired.png", dpi=180)
    plt.close(figure)

    structural_rows = _read_csv(rewired_dir / "structural/structural_by_graph.csv")
    degree_preserved = all(
        row.get("exact_labeled_degree_preserved", "False").lower() == "true"
        for row in structural_rows
    )
    manifest = {
        "canonical_parameter": original_manifest["canonical_parameter"],
        "n": original_manifest["n"],
        "m": original_manifest["m"],
        "num_graphs": original_manifest["num_graphs"],
        "graph_seed_base": original_manifest["graph_seed_base"],
        "rewire_seed_base": rewired_manifest["rewire_seed_base"],
        "paired_by_graph_index_and_graph_seed": True,
        "difference_definition": "rewired_minus_original",
        "exact_labeled_degree_preservation_verified_for_all_graphs": degree_preserved,
        "paired_global_spectrum_summary": spectrum_summary,
        "rewiring": rewired_manifest["rewiring"],
        "original_candidate_regions": original_manifest.get("candidate_regions", []),
        "rewired_candidate_regions": rewired_manifest.get("candidate_regions", []),
        "source_directories": [str(original_dir), str(rewired_dir)],
        "interpretation": "rewiring preserves the realized labeled degree sequence; changes isolate organization beyond degrees, subject to finite-swap-chain mixing",
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameter-dir", type=Path, required=True)
    args = parser.parse_args()
    print(build_rewiring_comparison(args.parameter_dir))


if __name__ == "__main__":
    main()
