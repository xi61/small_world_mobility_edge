#!/usr/bin/env python3
"""Aggregate the four abnormal-band microscope models and write the report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from small_world_mobility_edge.preferential_attachment.abnormal_edge_followup import (
    GOE_MEAN_GAP_RATIO,
)
from small_world_mobility_edge.preferential_attachment.mobility_edge_postprocess import (
    write_rows,
)


VERSION = "pa_abnormal_band_microscope_aggregate_v1"
MODELS = (
    ("alpha_2p0", "affine alpha=2.00"),
    ("alpha_1p25", "affine alpha=1.25"),
    ("alpha_1p05", "affine alpha=1.05"),
    ("zeta_1p50", "superlinear zeta=1.50"),
)
CACHE_KEYS = {
    "alpha_2p0": "affine_alpha_2p0",
    "alpha_1p25": "affine_alpha_1p25",
    "alpha_1p05": "affine_alpha_1p05",
    "zeta_1p50": "superlinear_zeta_1p5",
}


def scalar(value: str):
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if value == "":
        return ""
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def read_rows(path: Path):
    with path.open(newline="") as handle:
        return [{key: scalar(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def heatmap(root: Path, field: str, title: str, path: Path, *, vmin=None, vmax=None, cmap="viridis"):
    rows_by_model = [
        read_rows(root / model_dir / "dq_0p0001" / "standard_summary.csv")
        for model_dir, _ in MODELS
    ]
    q = np.asarray([row["q_center"] for row in rows_by_model[0]], dtype=float)
    values = np.full((len(MODELS), len(q)), np.nan)
    for model_index, rows in enumerate(rows_by_model):
        for column, row in enumerate(rows):
            if bool(row["ratio_support"]):
                if field == "ks_preference":
                    values[model_index, column] = (
                        float(row["unfolded_spacing_KS_to_Poisson"])
                        - float(row["unfolded_spacing_KS_to_GOE"])
                    )
                else:
                    values[model_index, column] = float(row[field])
    masked = np.ma.masked_invalid(values)
    figure, axis = plt.subplots(figsize=(14.0, 4.4))
    image = axis.imshow(
        masked,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=(q[0] - 0.00005, q[-1] + 0.00005, len(MODELS) - 0.5, -0.5),
    )
    image.cmap.set_bad("#d1d5db")
    axis.set_yticks(range(len(MODELS)))
    axis.set_yticklabels([label for _, label in MODELS])
    ticks = np.linspace(q[0], q[-1], 9)
    axis.set_xticks(ticks)
    axis.set_xlabel("ensemble-CDF quantile q; dq=0.0001")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, shrink=.82)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=190)
    plt.close(figure)


def candidate_band(rows):
    supported = [row for row in rows if bool(row["ratio_support"])]
    minimum = min(supported, key=lambda row: float(row["raw_r"]))
    index = rows.index(minimum)
    left = right = index
    threshold = GOE_MEAN_GAP_RATIO - .01
    while left and bool(rows[left - 1]["ratio_support"]) and float(rows[left - 1]["raw_r"]) < threshold:
        left -= 1
    while right + 1 < len(rows) and bool(rows[right + 1]["ratio_support"]) and float(rows[right + 1]["raw_r"]) < threshold:
        right += 1
    return rows[left], rows[right], minimum


def overlapping_band(standard, shifted):
    std_left, std_right, _ = candidate_band(standard)
    sh_left, sh_right, _ = candidate_band(shifted)
    overlap_low = max(float(std_left["q_low"]), float(sh_left["q_low"]))
    overlap_high = min(float(std_right["q_high"]), float(sh_right["q_high"]))
    return (
        float(std_left["q_low"]),
        float(std_right["q_high"]),
        float(sh_left["q_low"]),
        float(sh_right["q_high"]),
        max(0.0, overlap_high - overlap_low),
    )


def support_recommendations(root: Path):
    output = []
    all_supported = True
    for model_dir, label in MODELS:
        cache_manifest = json.loads(
            (root / "cached_n5000_reanalysis" / CACHE_KEYS[model_dir] / "run_manifest.json").read_text()
        )
        graph_count = int(cache_manifest["total_graph_count"])
        levels = ["dq_0p0002", "dq_0p0001"]
        if (root / model_dir / "dq_0p00005").is_dir():
            levels.append("dq_0p00005")
        for level in levels:
            for grid_file, grid in (
                ("standard_summary.csv", "standard"),
                ("half_shifted_summary.csv", "half_shifted"),
            ):
                for row in read_rows(root / model_dir / level / grid_file):
                    ratio_supported = bool(row["ratio_support"])
                    histogram_supported = bool(row["histogram_support"])
                    ratio_count = int(row["valid_ratio_count"])
                    yield_per_graph = ratio_count / graph_count
                    target = (
                        int(math.ceil(100.0 / yield_per_graph * 1.25))
                        if yield_per_graph > 0
                        else 10000
                    )
                    target = min(target, 10000)
                    additional = max(0, target - graph_count)
                    if ratio_supported and histogram_supported:
                        decision = "NO_NEW_GRAPHS"
                    elif histogram_supported and not ratio_supported and additional > 0:
                        decision = (
                            "TARGETED_AUGMENTATION_CANDIDATE"
                            if target < 10000
                            else "DO_NOT_CHASE_FINE_CELL"
                        )
                        all_supported = False
                    else:
                        decision = "DO_NOT_CHASE_FINE_CELL"
                        all_supported = False
                    output.append({
                        "aggregate_version": VERSION,
                        "model": label,
                        "model_directory": model_dir,
                        "resolution_level": level,
                        "grid": grid,
                        "q_low": row["q_low"],
                        "q_high": row["q_high"],
                        "current_graphs": graph_count,
                        "positive_gaps": row["positive_gap_count"],
                        "gap_graphs": row["gap_contributing_graphs"],
                        "valid_ratios": ratio_count,
                        "ratio_graphs": row["ratio_contributing_graphs"],
                        "histogram_support": histogram_supported,
                        "ratio_support": ratio_supported,
                        "observed_ratio_yield_per_graph": yield_per_graph,
                        "estimated_target_graphs_with_25pct_cushion": target,
                        "additional_graphs": additional,
                        "decision": decision,
                    })
    decision = {
        "aggregate_version": VERSION,
        "all_requested_main_region_cells_supported": all_supported,
        "automatic_augmentation_submitted": False,
        "decision": (
            "NO_AUGMENTATION_REQUIRED"
            if all_supported
            else "REVIEW_TARGETED_CANDIDATES_BEFORE_SUBMISSION"
        ),
        "upper_cap_graphs": 10000,
        "rule": "ceil(100 / observed_ratio_yield_per_graph * 1.25)",
    }
    return output, decision


def block_range(root: Path, model_dir: str, level: str, minimum):
    rows = read_rows(root / model_dir / level / "independent_graph_blocks.csv")
    selected = [
        row for row in rows
        if math.isclose(float(row["q_low"]), float(minimum["q_low"]), abs_tol=1e-10)
        and math.isclose(float(row["q_high"]), float(minimum["q_high"]), abs_tol=1e-10)
    ]
    values = [float(row["raw_r"]) for row in selected if np.isfinite(float(row["raw_r"]))]
    return (min(values), max(values), len(values)) if values else (math.nan, math.nan, 0)


def model_digest(root: Path, model_dir: str, label: str):
    level = "dq_0p00005" if (root / model_dir / "dq_0p00005").is_dir() else "dq_0p0001"
    standard = read_rows(root / model_dir / level / "standard_summary.csv")
    shifted = read_rows(root / model_dir / level / "half_shifted_summary.csv")
    left, right, minimum = candidate_band(standard)
    overlap = overlapping_band(standard, shifted)
    sliding = read_rows(root / model_dir / "sliding_windows" / "sliding_window_summary.csv")
    fine_sliding = [row for row in sliding if math.isclose(float(row["window_width"]), .0002)]
    sliding_min_raw = min(fine_sliding, key=lambda row: float(row["raw_r"]))
    sliding_min_unfolded = min(fine_sliding, key=lambda row: float(row["unfolded_r"]))
    fixed = [
        row for row in read_rows(root / model_dir / "fixed_rank" / "fixed_rank_summary.csv")
        if row["side"] == "positive"
    ]
    fixed_min = min(fixed, key=lambda row: float(row["raw_r"]))
    block_low, block_high, block_count = block_range(root, model_dir, level, minimum)
    raw = float(minimum["raw_r"])
    unfolded = float(minimum["unfolded_r"])
    if unfolded - raw > .015:
        density = "material"
    elif unfolded - raw > .005:
        density = "modest"
    else:
        density = "small"
    classifications = []
    if overlap[4] <= 0:
        classifications.append("H1 bin-boundary sensitive")
    if density in {"material", "modest"}:
        classifications.append("H2 density-gradient contribution")
    if int(minimum["edge_rank_p90"]) - int(minimum["edge_rank_p10"]) <= 25:
        classifications.append("H3 finite-rank hub-mode band")
    if unfolded < GOE_MEAN_GAP_RATIO - .01 and overlap[4] > 0 and block_count >= 3:
        classifications.append("H4 robust intermediate non-GOE band")
    if not classifications:
        classifications.append("H5 or unresolved")
    return {
        "model": label,
        "level": level,
        "band_low": float(left["q_low"]),
        "band_high": float(right["q_high"]),
        "shifted_band_low": overlap[2],
        "shifted_band_high": overlap[3],
        "band_overlap_width": overlap[4],
        "minimum_q_low": float(minimum["q_low"]),
        "minimum_q_high": float(minimum["q_high"]),
        "minimum_raw_r": raw,
        "minimum_raw_se": float(minimum["raw_r_se"]),
        "minimum_raw_ci_low": float(minimum["raw_r_bootstrap_ci_low"]),
        "minimum_raw_ci_high": float(minimum["raw_r_bootstrap_ci_high"]),
        "minimum_unfolded_r": unfolded,
        "minimum_unfolded_ci_low": float(minimum["unfolded_r_bootstrap_ci_low"]),
        "minimum_unfolded_ci_high": float(minimum["unfolded_r_bootstrap_ci_high"]),
        "minimum_ratios": int(minimum["valid_ratio_count"]),
        "minimum_ratio_graphs": int(minimum["ratio_contributing_graphs"]),
        "rank_p10": float(minimum["edge_rank_p10"]),
        "rank_median": float(minimum["edge_rank_median"]),
        "rank_p90": float(minimum["edge_rank_p90"]),
        "NIPR": float(minimum["NIPR"]),
        "sliding_min_raw_r": float(sliding_min_raw["raw_r"]),
        "sliding_min_raw_q": float(sliding_min_raw["q_center"]),
        "sliding_min_unfolded_r": float(sliding_min_unfolded["unfolded_r"]),
        "sliding_min_unfolded_q": float(sliding_min_unfolded["q_center"]),
        "fixed_rank_min_window": f"{fixed_min['rank_low']}-{fixed_min['rank_high']}",
        "fixed_rank_min_raw_r": float(fixed_min["raw_r"]),
        "fixed_rank_min_cdf_unfolded_r": float(fixed_min["cdf_unfolded_r"]),
        "block_min": block_low,
        "block_max": block_high,
        "block_count": block_count,
        "density_effect": density,
        "classification": "; ".join(classifications),
    }


def format_float(value, digits=4):
    return "NA" if not np.isfinite(float(value)) else f"{float(value):.{digits}f}"


def write_report(root: Path, digests, decision, aggregate_job):
    by_name = {row["model"]: row for row in digests}
    alpha105 = by_name["affine alpha=1.05"]
    zeta15 = by_name["superlinear zeta=1.50"]
    alpha125 = by_name["affine alpha=1.25"]
    control = by_name["affine alpha=2.00"]
    cache_jobs = sorted({
        json.loads((root / "cached_n5000_reanalysis" / key / "run_manifest.json").read_text()).get("sge_job_id", "")
        for key in CACHE_KEYS.values()
    })
    microscope_jobs = sorted({
        json.loads((root / model_dir / "run_manifest.json").read_text()).get("sge_job_id", "")
        for model_dir, _ in MODELS
    })
    lines = [
        "# Abnormal-band microscope report",
        "",
        f"Analysis version: {VERSION}. Authoritative outputs were not modified.",
        "",
        "## Executive result",
        "",
        (
            f"Affine alpha=1.05 has a supported fine-grid low-r band spanning approximately "
            f"q={alpha105['band_low']:.5f}-{alpha105['band_high']:.5f} on the standard "
            f"{alpha105['level']} grid. The lowest cell is "
            f"[{alpha105['minimum_q_low']:.5f},{alpha105['minimum_q_high']:.5f}) with "
            f"raw r={alpha105['minimum_raw_r']:.4f} and same-triplet unfolded "
            f"r={alpha105['minimum_unfolded_r']:.4f} from {alpha105['minimum_ratios']} ratios."
        ),
        "",
        (
            f"Superlinear zeta=1.50 has a corresponding band around "
            f"q={zeta15['band_low']:.5f}-{zeta15['band_high']:.5f}; its lowest fine cell "
            f"has raw/unfolded r={zeta15['minimum_raw_r']:.4f}/"
            f"{zeta15['minimum_unfolded_r']:.4f}."
        ),
        "",
        f"Graph augmentation decision: {decision['decision']}.",
        "",
        "## Fine-grid model digest",
        "",
        "| Model | Grid | Candidate band | Lowest cell raw / unfolded r | Ratios / graphs | Rank p10 / median / p90 | Classification |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in digests:
        lines.append(
            f"| {row['model']} | {row['level']} | "
            f"{row['band_low']:.5f}-{row['band_high']:.5f} | "
            f"{row['minimum_raw_r']:.4f} / {row['minimum_unfolded_r']:.4f} | "
            f"{row['minimum_ratios']} / {row['minimum_ratio_graphs']} | "
            f"{row['rank_p10']:.1f} / {row['rank_median']:.1f} / {row['rank_p90']:.1f} | "
            f"{row['classification']} |"
        )
    lines += [
        "",
        "## Required scientific questions",
        "",
        "1. Affine alpha=1.05 low-r band: yes if the standard/shifted overlap and independent blocks above remain nonempty; the exact supported interval is reported in the digest.",
        f"2. Estimated start/end: q={alpha105['band_low']:.5f}-{alpha105['band_high']:.5f} on the finest requested standard grid.",
        "3. Resolution stability: see each model's dq_0p0002, dq_0p0001, and, for the two primary models, dq_0p00005 standard summaries and histogram panels.",
        f"4. Half-bin shift: alpha=1.05 overlap width={alpha105['band_overlap_width']:.5g}; zeta=1.50 overlap width={zeta15['band_overlap_width']:.5g}.",
        f"5. Sliding windows: alpha=1.05 minimum raw r={alpha105['sliding_min_raw_r']:.4f} at q={alpha105['sliding_min_raw_q']:.5f}; zeta=1.50 minimum={zeta15['sliding_min_raw_r']:.4f} at q={zeta15['sliding_min_raw_q']:.5f}.",
        f"6. Same-triplet unfolding: alpha=1.05 lowest-cell unfolded r={alpha105['minimum_unfolded_r']:.4f}; density correction is {alpha105['density_effect']}. Zeta=1.50 unfolded r={zeta15['minimum_unfolded_r']:.4f}.",
        "7. Full P(r): candidate-band panels compare observed raw and unfolded distributions with GOE and Poisson references under each primary model directory.",
        f"8. Consecutive spacings: correlations, confidence intervals, scatter, and log spacing-ratio plots are saved; these distinguish broad imbalance from simple negative alternation.",
        f"9. Fixed edge ranks: alpha=1.05 minimum fixed window is {alpha105['fixed_rank_min_window']}; zeta=1.50 minimum is {zeta15['fixed_rank_min_window']}.",
        f"10. Zeta=1.50: classification is {zeta15['classification']}.",
        f"11. Alpha=1.25: lowest fine raw/unfolded r={alpha125['minimum_raw_r']:.4f}/{alpha125['minimum_unfolded_r']:.4f}; classification is {alpha125['classification']}.",
        f"12. Alpha=2.00 control: lowest fine raw/unfolded r={control['minimum_raw_r']:.4f}/{control['minimum_unfolded_r']:.4f}; classification is {control['classification']}.",
        f"13. Primary classifications: alpha=1.05: {alpha105['classification']}; zeta=1.50: {zeta15['classification']}.",
        "",
        "## Statistical stability",
        "",
    ]
    for row in digests:
        lines.append(
            f"- {row['model']}: at its minimum fine cell the independent 500-graph block raw-r range is "
            f"{format_float(row['block_min'])}-{format_float(row['block_max'])} across "
            f"{row['block_count']} blocks; bootstrap CI is "
            f"[{row['minimum_raw_ci_low']:.4f},{row['minimum_raw_ci_high']:.4f}]."
        )
    lines += [
        "",
        "## Support and augmentation",
        "",
        "Separate histogram (100 positive gaps, 50 graphs) and mean-r (100 ratios, 50 graphs) gates were applied to every cell. Unsupported cells are masked in comparison heatmaps. The complete yield-based decision table is model_comparison/support_and_augmentation_recommendations.csv.",
        "",
        "## Provenance and paths",
        "",
        f"- Cache-only Hoffman array job IDs: {', '.join(filter(None, cache_jobs))}.",
        f"- Fine microscope Hoffman array job IDs: {', '.join(filter(None, microscope_jobs))}.",
        f"- Comparison/report Hoffman job ID: {aggregate_job}.",
        f"- Remote result root: {root}.",
        "- Local synchronized result root: outputs/preferential_attachment/mobility_edge/diagnostics/abnormal_band_microscope/.",
        "- Upstream base/augmentation cache checksums and job metadata are recorded under cached_n5000_reanalysis/MODEL/source_cache_checksums.csv.",
        "",
        "## Main output map",
        "",
        "- Per-model nested summaries, shifted grids, histograms, rank maps, block replication, negative-edge mirror, fixed-rank results, and sliding windows live under alpha_2p0, alpha_1p25, alpha_1p05, and zeta_1p50.",
        "- Cross-model raw-r, unfolded-r, and signed KS heatmaps live under model_comparison.",
        "- Cells failing ratio support are masked; no outermost q>0.9999 computation was launched.",
        "",
    ]
    (root / "abnormal_band_microscope_report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root
    comparison = root / "model_comparison"
    heatmap(root, "raw_r", "Graph-equal raw r; unsupported cells masked", comparison / "dq_0p0001_raw_r_heatmap.png", vmin=.44, vmax=.56, cmap="viridis")
    heatmap(root, "unfolded_r", "Same-triplet unfolded r; unsupported cells masked", comparison / "dq_0p0001_unfolded_r_heatmap.png", vmin=.44, vmax=.56, cmap="viridis")
    heatmap(root, "ks_preference", "KS(Poisson) - KS(GOE); positive favors GOE", comparison / "dq_0p0001_signed_ks_heatmap.png", vmin=-.15, vmax=.20, cmap="coolwarm")
    recommendations, decision = support_recommendations(root)
    write_rows(comparison / "support_and_augmentation_recommendations.csv", recommendations)
    (comparison / "augmentation_decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    digests = [model_digest(root, model_dir, label) for model_dir, label in MODELS]
    write_rows(comparison / "model_digest.csv", digests)
    write_report(root, digests, decision, os.environ.get("JOB_ID", ""))
    manifest = {
        "aggregate_version": VERSION,
        "sge_job_id": os.environ.get("JOB_ID", ""),
        "authoritative_results_modified": False,
        "augmentation_decision": decision,
        "files": sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()),
    }
    (comparison / "aggregate_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(root / "abnormal_band_microscope_report.md", flush=True)


if __name__ == "__main__":
    main()
