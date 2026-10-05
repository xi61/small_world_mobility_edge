"""Aggregate completed random-Schrodinger stage manifests on Hoffman."""

from __future__ import annotations

import argparse
import csv
import gzip
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from small_world_mobility_edge.random_schrodinger.common import (
    read_json,
    utc_now,
    write_csv,
    write_json,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _finite_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _manifest_row(stage: str, case: str, path: Path) -> dict[str, object]:
    if not path.is_file():
        return {
            "stage": stage,
            "case": case,
            "status": "NOT_FOUND",
            "job_id": None,
            "runtime_seconds": math.nan,
            "manifest_path": str(path),
        }
    manifest = read_json(path)
    return {
        "stage": stage,
        "case": case,
        "status": manifest.get("status", "UNKNOWN"),
        "job_id": manifest.get("job_id"),
        "sge_task_id": manifest.get("sge_task_id"),
        "runtime_seconds": manifest.get("runtime_seconds", math.nan),
        "manifest_path": str(path),
        "failure_type": manifest.get("failure_type"),
        "failure_message": manifest.get("failure_message"),
    }


def aggregate(root: Path, output_dir: Path) -> Path:
    root = Path(root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    statuses: list[dict[str, object]] = []
    stage1_models = ("affine_alpha_1p05", "nonlinear_zeta_1p5", "affine_alpha_2p0")
    for model in stage1_models:
        statuses.append(
            _manifest_row("stage1", model, root / "stage1" / model / "run_manifest.json")
        )
    for label in ("pilot_N1000_seed21901001", "published_size_N10000_seed21901001"):
        statuses.append(
            _manifest_row(
                "stage2_er",
                label,
                root / "outputs" / "er_benchmark" / label / "run_manifest.json",
            )
        )
    statuses.append(
        _manifest_row(
            "stage2_validation",
            "controlled_fixtures",
            root / "outputs" / "tunneling_validation" / "run_manifest.json",
        )
    )
    for case in ("er", "pa", "cl"):
        statuses.append(
            _manifest_row(
                "stage2_application",
                case,
                root / "outputs" / "tunneling_applications" / case / "run_manifest.json",
            )
        )
    potential_seeds = (931701, 931702)
    for model in stage1_models:
        for seed in potential_seeds:
            statuses.append(
                _manifest_row(
                    "stage3_disorder",
                    f"{model}/potential_seed_{seed}",
                    root
                    / "outputs"
                    / "pa_disorder_pilot"
                    / model
                    / f"potential_seed_{seed}"
                    / "run_manifest.json",
                )
            )
    write_csv(output_dir / "status_table.csv", statuses)

    stage1_rows: list[dict[str, object]] = []
    for model in stage1_models:
        path = root / "stage1" / model / "tables" / "eigenpair_geometry.csv"
        if not path.is_file():
            continue
        for row in _read_csv(path):
            rank = int(row["upper_edge_rank"])
            if 2 <= rank <= 11:
                stage1_rows.append(
                    {
                        "model": model,
                        "upper_edge_rank": rank,
                        "eigenvalue": float(row["eigenvalue"]),
                        "ipr": float(row["ipr"]),
                        "geometry_label": row["geometry_label"],
                        "center_count": int(row["center_count"]),
                        "top10_mass": float(row["top10_mass"]),
                        "top10_finite_distance_mean": float(row["finite_distance_mean"]),
                        "degree_matched_distance_zscore": float(row["degree_matched_distance_zscore"]),
                    }
                )
    write_csv(output_dir / "stage1_nonperron_geometry.csv", stage1_rows)
    stage1_group_rows: list[dict[str, object]] = []
    for model in stage1_models:
        group = [row for row in stage1_rows if row["model"] == model]
        if not group:
            continue
        labels = sorted({str(row["geometry_label"]) for row in group})
        summary: dict[str, object] = {
            "model": model,
            "nonperron_eigenpair_count": len(group),
            "median_ipr": float(np.median([float(row["ipr"]) for row in group])),
            "median_top10_mass": float(
                np.median([float(row["top10_mass"]) for row in group])
            ),
            "median_top10_finite_distance_mean": float(
                np.median(
                    [float(row["top10_finite_distance_mean"]) for row in group]
                )
            ),
            "median_degree_matched_distance_zscore": float(
                np.median(
                    [float(row["degree_matched_distance_zscore"]) for row in group]
                )
            ),
        }
        for label in labels:
            summary[f"geometry_label_{label}_count"] = sum(
                str(row["geometry_label"]) == label for row in group
            )
        stage1_group_rows.append(summary)
    write_csv(output_dir / "stage1_model_summary.csv", stage1_group_rows)

    er_rows: list[dict[str, object]] = []
    for label in ("pilot_N1000_seed21901001", "published_size_N10000_seed21901001"):
        benchmark_dir = root / "outputs" / "er_benchmark" / label
        manifest_path = benchmark_dir / "run_manifest.json"
        spectrum_path = benchmark_dir / "full_spectrum_diagnostics.npz"
        if not manifest_path.is_file() or not spectrum_path.is_file():
            continue
        benchmark_manifest = read_json(manifest_path)
        if benchmark_manifest.get("status") != "COMPLETE":
            continue
        with np.load(spectrum_path, allow_pickle=False) as data:
            values = np.asarray(data["eigenvalues"], dtype=float)
            infinity_norms = np.asarray(data["infinity_norms"], dtype=float)
            ipr = np.asarray(data["ipr"], dtype=float)
        n_eff = len(values)
        bulk_half_width = max(5, int(round(0.05 * n_eff)))
        bulk_center = n_eff // 2
        bulk_slice = slice(
            max(0, bulk_center - bulk_half_width),
            min(n_eff, bulk_center + bulk_half_width),
        )
        nonperron_edge = infinity_norms[max(0, n_eff - 11) : n_eff - 1]
        er_rows.append(
            {
                "run_label": label,
                "original_n": int(benchmark_manifest["original_n"]),
                "analyzed_n": int(benchmark_manifest["analyzed_n"]),
                "d": float(benchmark_manifest["publication_setup"]["d"]),
                "minimum_energy": float(values[0]),
                "maximum_energy_perron": float(values[-1]),
                "perron_infinity_norm": float(infinity_norms[-1]),
                "perron_ipr": float(ipr[-1]),
                "top10_nonperron_infinity_norm_median": float(np.median(nonperron_edge)),
                "top10_nonperron_infinity_norm_maximum": float(np.max(nonperron_edge)),
                "central_10_percent_infinity_norm_median": float(
                    np.median(infinity_norms[bulk_slice])
                ),
                "central_10_percent_ipr_median": float(np.median(ipr[bulk_slice])),
                "full_spectrum_coverage": bool(benchmark_manifest["full_spectrum_coverage"]),
                "maximum_sampled_relative_residual": float(
                    benchmark_manifest["sampled_maximum_relative_residual_norm"]
                ),
            }
        )
    write_csv(output_dir / "er_benchmark_summary.csv", er_rows)

    validation_summary: list[dict[str, object]] = []
    validation_manifest_path = (
        root / "outputs" / "tunneling_validation" / "run_manifest.json"
    )
    if validation_manifest_path.is_file():
        validation_manifest = read_json(validation_manifest_path)
        validation_summary.append(
            {
                "status": validation_manifest.get("status"),
                "validation_passed": validation_manifest.get("validation_passed"),
                "case_count": validation_manifest.get("case_count"),
                "schur_case_count": validation_manifest.get("schur_case_count"),
                "maximum_exact_two_state_splitting_error": validation_manifest.get(
                    "maximum_exact_two_state_splitting_error"
                ),
                "runtime_seconds": validation_manifest.get("runtime_seconds"),
                "job_id": validation_manifest.get("job_id"),
            }
        )
    write_csv(output_dir / "stage2_validation_summary.csv", validation_summary)

    application_summary: list[dict[str, object]] = []
    for case in ("er", "pa", "cl"):
        case_dir = root / "outputs" / "tunneling_applications" / case
        path = case_dir / "adjacent_pair_effective_coupling.csv.gz"
        if not path.is_file():
            continue
        case_manifest = read_json(case_dir / "run_manifest.json")
        rows = _read_csv(path)
        primary = [
            row
            for row in rows
            if int(row["radius"]) == 2
            and row["nonperron_top10_window"] == "True"
            and row["valid"] == "True"
        ]
        coupling = np.asarray([float(row["absolute_direct_coupling"]) for row in primary])
        detuning = np.asarray([float(row["uncoupled_detuning"]) for row in primary])
        residual_failures = sum(
            row["poor_local_state_residual_flag"] == "True" for row in primary
        )
        quality_passing = [
            row for row in primary if row["poor_local_state_residual_flag"] != "True"
        ]
        application_summary.append(
            {
                "case": case,
                "primary_pair_count": len(primary),
                "poor_local_state_residual_count": residual_failures,
                "quality_passing_pair_count": len(quality_passing),
                "exploratory_median_direct_coupling_all_projection_valid": (
                    float(np.median(coupling)) if len(coupling) else math.nan
                ),
                "exploratory_median_uncoupled_detuning_all_projection_valid": (
                    float(np.median(detuning)) if len(detuning) else math.nan
                ),
                "exploratory_median_coupling_over_detuning_all_projection_valid": (
                    float(np.median(coupling[detuning > 0.0] / detuning[detuning > 0.0]))
                    if np.any(detuning > 0.0)
                    else math.nan
                ),
                "paper_approximate_state_count": case_manifest.get(
                    "paper_approximate_state_count"
                ),
                "paper_approximate_pair_count": case_manifest.get(
                    "paper_approximate_pair_count"
                ),
                "maximum_source_relative_residual": case_manifest.get(
                    "maximum_source_relative_residual"
                ),
                "graph_id": case_manifest.get("graph_id"),
                "operator_id": case_manifest.get("operator_id"),
            }
        )
    write_csv(output_dir / "stage2_application_summary.csv", application_summary)

    disorder_rows: list[dict[str, object]] = []
    center_attribution_rows: list[dict[str, object]] = []
    disorder_tunneling_rows: list[dict[str, object]] = []
    for model in stage1_models:
        for seed in potential_seeds:
            sweep_dir = (
                root
                / "outputs"
                / "pa_disorder_pilot"
                / model
                / f"potential_seed_{seed}"
            )
            path = sweep_dir / "sweep_summary.csv"
            if not path.is_file():
                continue
            sweep_manifest = read_json(path.parent / "run_manifest.json")
            source_dir = Path(str(sweep_manifest["source_dir"]))
            for row in _read_csv(path):
                ratio = float(row["a_over_s_ref"])
                repaired_baseline_interior = False
                if ratio == 0.0 and not math.isfinite(float(row["interior_mean_ipr"])):
                    with np.load(
                        source_dir / "spectrum" / "single_graph_spectral_arrays.npz",
                        allow_pickle=False,
                    ) as source_spectrum:
                        source_ipr = np.asarray(source_spectrum["ipr"], dtype=float)
                        source_bulk = np.asarray(
                            source_spectrum["bulk50_indices"], dtype=int
                        )
                    row["interior_mean_ipr"] = float(np.mean(source_ipr[source_bulk]))
                    repaired_baseline_interior = True
                disorder_rows.append(
                    {
                        "model": model,
                        "graph_role": "one fixed cached graph for this model",
                        "potential_seed": seed,
                        "interior_window_definition": (
                            "historical atom-aware bulk50"
                            if ratio == 0.0
                            else "ascending-rank q50 width51"
                        ),
                        "baseline_interior_summary_recovered_from_source_cache": repaired_baseline_interior,
                        **row,
                    }
                )
            for ratio_dir in sorted(sweep_dir.glob("ratio_*")):
                selected_path = ratio_dir / "selected_eigenvector_diagnostics.csv.gz"
                if not selected_path.is_file():
                    continue
                selected_rows = _read_csv(selected_path)
                ratio = float(selected_rows[0]["a_over_s_ref"])
                for region in ("upper_edge", "lower_edge", "interior"):
                    region_rows = [row for row in selected_rows if row["region"] == region]
                    if region in ("upper_edge", "lower_edge"):
                        region_rows = [
                            row
                            for row in region_rows
                            if float(row["edge_rank_1based"]) <= 10.0
                        ]
                    if not region_rows:
                        continue
                    center_attribution_rows.append(
                        {
                            "model": model,
                            "potential_seed": seed,
                            "a_over_s_ref": ratio,
                            "region": region,
                            "rank_window": (
                                "edge ranks 1-10"
                                if region != "interior"
                                else (
                                    "historical atom-aware bulk50"
                                    if ratio == 0.0
                                    else "all selected q25/q50/q75 windows"
                                )
                            ),
                            "eigenvector_count": len(region_rows),
                            "mean_dominant_center_standardized_degree": float(
                                np.mean(
                                    [
                                        float(row["dominant_center_standardized_degree"])
                                        for row in region_rows
                                    ]
                                )
                            ),
                            "mean_dominant_center_xi": float(
                                np.mean(
                                    [float(row["dominant_center_xi"]) for row in region_rows]
                                )
                            ),
                            "mean_dominant_center_diagonal_potential": float(
                                np.mean(
                                    [
                                        float(row["dominant_center_diagonal_potential"])
                                        for row in region_rows
                                    ]
                                )
                            ),
                            "fraction_dominant_center_xi_positive": float(
                                np.mean(
                                    [
                                        float(row["dominant_center_xi"]) > 0.0
                                        for row in region_rows
                                    ]
                                )
                            ),
                            "fraction_dominant_center_xi_negative": float(
                                np.mean(
                                    [
                                        float(row["dominant_center_xi"]) < 0.0
                                        for row in region_rows
                                    ]
                                )
                            ),
                            "mean_top10_xi": float(
                                np.mean([float(row["top10_mean_xi"]) for row in region_rows])
                            ),
                        }
                    )
                coupling_path = ratio_dir / "upper_adjacent_effective_coupling.csv"
                if coupling_path.is_file():
                    coupling_rows = [
                        row
                        for row in _read_csv(coupling_path)
                        if int(row["radius"]) == 2
                    ]
                    poor = []
                    quality = []
                    numeric_projection = []
                    for row in coupling_rows:
                        left_residual = _finite_float(row["left_residual_norm"])
                        right_residual = _finite_float(row["right_residual_norm"])
                        direct = _finite_float(row["absolute_direct_coupling"])
                        detuning = _finite_float(row["uncoupled_detuning"])
                        if math.isfinite(direct) and math.isfinite(detuning):
                            numeric_projection.append((direct, detuning))
                        if (
                            row.get("valid") == "True"
                            and math.isfinite(left_residual)
                            and math.isfinite(right_residual)
                            and max(left_residual, right_residual) <= 0.25
                        ):
                            quality.append(row)
                        else:
                            poor.append(row)
                    disorder_tunneling_rows.append(
                        {
                            "model": model,
                            "potential_seed": seed,
                            "a_over_s_ref": ratio,
                            "radius": 2,
                            "pair_count": len(coupling_rows),
                            "poor_local_state_residual_count": len(poor),
                            "quality_passing_pair_count": len(quality),
                            "finite_projection_pair_count": len(numeric_projection),
                            "exploratory_median_direct_coupling_all_pairs": (
                                float(np.median([item[0] for item in numeric_projection]))
                                if numeric_projection
                                else math.nan
                            ),
                            "exploratory_median_uncoupled_detuning_all_pairs": (
                                float(np.median([item[1] for item in numeric_projection]))
                                if numeric_projection
                                else math.nan
                            ),
                        }
                    )
    write_csv(output_dir / "stage3_nested_raw_summary.csv", disorder_rows)
    write_csv(output_dir / "stage3_center_attribution_nested.csv", center_attribution_rows)
    write_csv(output_dir / "stage3_tunneling_quality_nested.csv", disorder_tunneling_rows)
    grouped: dict[tuple[str, float], list[dict[str, object]]] = defaultdict(list)
    for row in disorder_rows:
        grouped[(str(row["model"]), float(row["a_over_s_ref"]))].append(row)
    disorder_group_rows: list[dict[str, object]] = []
    for (model, ratio), group in sorted(grouped.items()):
        result: dict[str, object] = {
            "model": model,
            "a_over_s_ref": ratio,
            "fixed_graph_count": 1,
            "potential_realization_count": len(group),
            "uncertainty_scope": "between-potential only; no within-model graph-level uncertainty",
        }
        for metric in ("upper_edge_mean_ipr", "lower_edge_mean_ipr", "interior_mean_ipr"):
            values = np.asarray([float(row[metric]) for row in group], dtype=float)
            result[metric + "_mean_across_potentials"] = float(np.mean(values))
            result[metric + "_sd_across_potentials"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
            )
        disorder_group_rows.append(result)
    write_csv(output_dir / "stage3_potential_level_summary.csv", disorder_group_rows)

    center_groups: dict[tuple[str, float, str], list[dict[str, object]]] = defaultdict(list)
    for row in center_attribution_rows:
        center_groups[
            (str(row["model"]), float(row["a_over_s_ref"]), str(row["region"]))
        ].append(row)
    center_group_rows: list[dict[str, object]] = []
    for (model, ratio, region), group in sorted(center_groups.items()):
        item: dict[str, object] = {
            "model": model,
            "a_over_s_ref": ratio,
            "region": region,
            "fixed_graph_count": 1,
            "potential_realization_count": len(group),
            "uncertainty_scope": "between-potential only; no within-model graph-level uncertainty",
        }
        for metric in (
            "mean_dominant_center_standardized_degree",
            "mean_dominant_center_xi",
            "mean_dominant_center_diagonal_potential",
            "fraction_dominant_center_xi_positive",
            "fraction_dominant_center_xi_negative",
            "mean_top10_xi",
        ):
            values = np.asarray([float(row[metric]) for row in group], dtype=float)
            item[metric + "_mean_across_potentials"] = float(np.mean(values))
            item[metric + "_sd_across_potentials"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
            )
        center_group_rows.append(item)
    write_csv(output_dir / "stage3_center_attribution_summary.csv", center_group_rows)

    completed = sum(row["status"] == "COMPLETE" for row in statuses)
    report_lines = [
        "# Random Schrödinger extension: automated Hoffman status",
        "",
        f"Generated at {utc_now()}. This report only summarizes artifacts present under {root}.",
        "",
        f"- Complete manifests: {completed}/{len(statuses)}",
        "- NOT_FOUND means unsubmitted, queued without a manifest, or not yet started; it does not mean scientific failure.",
        "- Stage 3 uncertainty is between two potentials on one cached graph per PA model. It does not establish graph-level or finite-size uncertainty.",
        "- No distinct SW result is reported because the repository contains no explicit SW generator.",
        "",
        "Raw status is in status_table.csv; consolidated Stage 1/2/3 tables are stored beside this file.",
    ]
    (output_dir / "AUTOMATED_STATUS.md").write_text("\n".join(report_lines) + "\n")
    write_json(
        output_dir / "aggregate_manifest.json",
        {
            "status": "COMPLETE",
            "generated_at_utc": utc_now(),
            "source_root": str(root),
            "status_row_count": len(statuses),
            "complete_manifest_count": completed,
            "stage1_nonperron_row_count": len(stage1_rows),
            "stage1_model_summary_row_count": len(stage1_group_rows),
            "er_benchmark_run_count": len(er_rows),
            "stage2_validation_summary_row_count": len(validation_summary),
            "stage2_application_case_count": len(application_summary),
            "stage3_raw_sweep_row_count": len(disorder_rows),
            "stage3_center_attribution_nested_row_count": len(center_attribution_rows),
            "stage3_tunneling_quality_nested_row_count": len(disorder_tunneling_rows),
        },
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Run root containing the completed stage outputs.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    aggregate(args.root, args.output_dir)


if __name__ == "__main__":
    main()
