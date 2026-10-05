"""Read-only verification of completed random-Schrodinger artifacts on Hoffman.

This script deliberately writes only below the supplied audit output directory.
It does not import or call production analysis entry points and does not mutate
the completed experiment tree.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


MODELS = ("affine_alpha_1p05", "nonlinear_zeta_1p5", "affine_alpha_2p0")
POTENTIAL_SEEDS = (931701, 931702)
RATIOS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finite(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def safe_max(values: Iterable[float]) -> float:
    kept = [float(value) for value in values if math.isfinite(float(value))]
    return max(kept) if kept else math.nan


def safe_median(values: Iterable[float]) -> float:
    kept = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.median(kept)) if kept else math.nan


def ratio_slug(value: float) -> str:
    return ("%g" % value).replace(".", "p")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_inventory(root: Path) -> list[dict]:
    expected: list[tuple[str, str, Path]] = []
    stage1_files = (
        "run_manifest.json",
        "source_cache_provenance.json",
        "tables/eigenpair_geometry.csv",
        "tables/top10_pair_distances.csv",
        "tables/top10_distance_matrices.csv.gz",
        "tables/mass_set_sensitivity.csv",
        "tables/center_rule_sensitivity.csv",
        "tables/center_shell_profiles.csv",
        "tables/center_decay_fits.csv",
        "tables/structural_baselines.csv.gz",
        "tables/adjacent_edge_pairs.csv",
        "tables/adjacent_cross_distances.csv.gz",
    )
    for model in MODELS:
        base = root / "stage1" / model
        expected.extend(("stage1", model, base / name) for name in stage1_files)
    for label in ("pilot_N1000_seed21901001", "published_size_N10000_seed21901001"):
        base = root / "outputs" / "er_benchmark" / label
        for name in (
            "run_manifest.json",
            "graph_realization.npz",
            "full_spectrum_diagnostics.npz",
            "selected_eigenvectors.npz",
            "sampled_eigenpair_residuals.csv",
            "figure10_replication_energy_vs_infinity_norm.png",
        ):
            expected.append(("stage2_er", label, base / name))
    base = root / "outputs" / "tunneling_validation"
    for name in ("run_manifest.json", "validation_cases.csv", "schur_sensitivity.csv"):
        expected.append(("stage2_validation", "controlled_fixtures", base / name))
    for case in ("er", "pa", "cl"):
        base = root / "outputs" / "tunneling_applications" / case
        for name in (
            "run_manifest.json",
            "adjacent_pair_effective_coupling.csv.gz",
            "source_eigenpair_residuals.csv",
            "schur_complement_sensitivity.csv",
        ):
            expected.append(("stage2_application", case, base / name))
    for model in MODELS:
        for seed in POTENTIAL_SEEDS:
            case = f"{model}/potential_seed_{seed}"
            sweep = root / "outputs" / "pa_disorder_pilot" / model / f"potential_seed_{seed}"
            for name in ("run_manifest.json", "paired_potential.npz", "sweep_summary.csv"):
                expected.append(("stage3_disorder", case, sweep / name))
            for ratio in RATIOS:
                ratio_dir = sweep / f"ratio_{ratio_slug(ratio)}"
                names = [
                    "run_manifest.json",
                    "spectrum_diagnostics.npz",
                    "selected_eigenvectors.npz",
                    "selected_eigenvector_diagnostics.csv.gz",
                    "upper_adjacent_effective_coupling.csv",
                ]
                # Production evaluates Schur rows only at these ratios;
                # write_csv creates no file for an empty table.
                if ratio in (0.0, 1.0, 8.0):
                    names.append("schur_complement_sensitivity.csv")
                for name in names:
                    expected.append(("stage3_ratio", f"{case}/ratio_{ratio_slug(ratio)}", ratio_dir / name))
    rows = []
    for stage, case, path in expected:
        exists = path.is_file()
        rows.append(
            {
                "stage": stage,
                "case": case,
                "path": str(path),
                "exists": exists,
                "bytes": path.stat().st_size if exists else 0,
                "nonempty": bool(exists and path.stat().st_size > 0),
            }
        )
    return rows


def source_revisions(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("run_manifest.json")):
        manifest = read_json(path)
        revision = manifest.get("code_revision") or {}
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "stage": manifest.get("stage"),
                "stage_version": manifest.get("stage_version"),
                "status": manifest.get("status"),
                "job_id": manifest.get("job_id"),
                "sge_task_id": manifest.get("sge_task_id"),
                "code_revision_sha256": revision.get("sha256"),
                "code_revision_file_count": revision.get("file_count"),
                "cache_id": manifest.get("cache_id"),
                "graph_id": manifest.get("graph_id"),
                "potential_id": manifest.get("potential_id"),
            }
        )
    return rows


def verify_stage1(root: Path) -> list[dict]:
    results = []
    for model in MODELS:
        table_dir = root / "stage1" / model / "tables"
        geometry = read_csv(table_dir / "eigenpair_geometry.csv")
        pairs = read_csv(table_dir / "top10_pair_distances.csv")
        sensitivity = read_csv(table_dir / "center_rule_sensitivity.csv")
        shells = read_csv(table_dir / "center_shell_profiles.csv")
        by_rank: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in pairs:
            by_rank[int(row["upper_edge_rank"])].append(row)
        geometry_by_rank = {int(row["upper_edge_rank"]): row for row in geometry}
        mean_errors = []
        weighted_errors = []
        all_pairs_valid = True
        for rank, rank_rows in by_rank.items():
            distances = [finite(row["distance"]) for row in rank_rows if row["unreachable"] != "True"]
            weights = [finite(row["pair_mass_weight"]) for row in rank_rows if row["unreachable"] != "True"]
            observed = finite(geometry_by_rank[rank]["finite_distance_mean"])
            observed_weighted = finite(geometry_by_rank[rank]["finite_mass_weighted_mean_distance"])
            recomputed = float(np.mean(distances)) if distances else math.nan
            recomputed_weighted = float(np.dot(distances, weights) / np.sum(weights)) if weights else math.nan
            mean_errors.append(abs(observed - recomputed))
            weighted_errors.append(abs(observed_weighted - recomputed_weighted))
            all_pairs_valid = all_pairs_valid and len(rank_rows) == 45
        primary = [
            row
            for row in sensitivity
            if int(row["radius"]) == 2
            and int(row["peak_separation"]) == 3
            and abs(float(row["target_mass"]) - 0.8) < 1e-12
        ]
        nonperron_primary = [row for row in primary if 2 <= int(row["upper_edge_rank"]) <= 11]
        shell_mass_errors = []
        rms_errors = []
        for row in shells:
            size = int(row["shell_size"])
            if size:
                shell_mass_errors.append(
                    abs(float(row["mean_mass_per_shell_vertex"]) - float(row["shell_mass"]) / size)
                )
                rms_errors.append(
                    abs(float(row["rms_component"]) ** 2 - float(row["mean_mass_per_shell_vertex"]))
                )
        result = {
            "model": model,
            "geometry_rows": len(geometry),
            "rank_set": ",".join(str(value) for value in sorted(geometry_by_rank)),
            "pair_rows": len(pairs),
            "all_ranks_have_45_pairs": all_pairs_valid,
            "unreachable_pair_rows": sum(row["unreachable"] == "True" for row in pairs),
            "maximum_mean_distance_recompute_error": max(mean_errors),
            "maximum_mass_weighted_distance_recompute_error": max(weighted_errors),
            "nonperron_primary_rows": len(nonperron_primary),
            "nonperron_primary_one_center": sum(row["geometry_label"] == "one_center" for row in nonperron_primary),
            "all_sensitivity_rows": len(sensitivity),
            "all_sensitivity_label_counts": json.dumps(Counter(row["geometry_label"] for row in sensitivity), sort_keys=True),
            "all_sensitivity_resolved_fraction": sum(row["resolved"] == "True" for row in sensitivity) / len(sensitivity),
            "maximum_shell_mean_mass_identity_error": max(shell_mass_errors),
            "maximum_shell_rms_identity_error": max(rms_errors),
        }
        results.append(result)
    return results


def verify_stage2(root: Path) -> list[dict]:
    results = []
    for case in ("er", "pa", "cl"):
        path = root / "outputs" / "tunneling_applications" / case / "adjacent_pair_effective_coupling.csv.gz"
        rows = read_csv(path)
        primary_all = [
            row
            for row in rows
            if int(row["radius"]) == 2 and row["nonperron_top10_window"] == "True"
        ]
        valid = [row for row in primary_all if row["valid"] == "True"]
        residual_gap_ratios = []
        residuals = []
        for row in valid:
            residual = max(finite(row["left_residual_norm"]), finite(row["right_residual_norm"]))
            gap = finite(row["observed_adjacent_full_operator_gap"])
            residuals.append(residual)
            if gap > 0.0:
                residual_gap_ratios.append(residual / gap)
        results.append(
            {
                "case": case,
                "all_pair_radius_rows": len(rows),
                "expected_nonperron_top10_radius2_pairs": len(primary_all),
                "projection_valid_pairs": len(valid),
                "projection_invalid_pairs": len(primary_all) - len(valid),
                "same_exact_dominant_center_pairs": sum(row["same_dominant_center"] == "True" for row in primary_all),
                "absolute_threshold_0p25_pass_count": sum(value <= 0.25 for value in residuals),
                "residual_over_observed_gap_lt_1_count": sum(value < 1.0 for value in residual_gap_ratios),
                "residual_over_observed_gap_lt_0p1_count": sum(value < 0.1 for value in residual_gap_ratios),
                "median_max_local_residual": safe_median(residuals),
                "median_residual_over_observed_gap_proxy": safe_median(residual_gap_ratios),
                "minimum_residual_over_observed_gap_proxy": min(residual_gap_ratios) if residual_gap_ratios else math.nan,
                "maximum_residual_over_observed_gap_proxy": max(residual_gap_ratios) if residual_gap_ratios else math.nan,
                "note": "Observed exact adjacent gap is only a proxy, not the 2024 approximate-energy isolation Delta.",
            }
        )
    return results


def verify_er(root: Path) -> list[dict]:
    results = []
    for label in ("pilot_N1000_seed21901001", "published_size_N10000_seed21901001"):
        base = root / "outputs" / "er_benchmark" / label
        manifest = read_json(base / "run_manifest.json")
        with np.load(base / "full_spectrum_diagnostics.npz", allow_pickle=False) as data:
            values = np.asarray(data["eigenvalues"], dtype=float)
            ipr = np.asarray(data["ipr"], dtype=float)
            infinity = np.asarray(data["infinity_norms"], dtype=float)
        with np.load(base / "graph_realization.npz", allow_pickle=False) as data:
            graph_keys = sorted(data.files)
            original_ids = np.asarray(data["original_vertex_ids"], dtype=int)
        results.append(
            {
                "label": label,
                "original_n": manifest["original_n"],
                "analyzed_n": manifest["analyzed_n"],
                "spectrum_length": len(values),
                "ipr_length": len(ipr),
                "infinity_norm_length": len(infinity),
                "all_finite": bool(np.all(np.isfinite(values)) and np.all(np.isfinite(ipr)) and np.all(np.isfinite(infinity))),
                "eigenvalues_nondecreasing": bool(np.all(np.diff(values) >= -1e-12)),
                "original_vertex_id_count": len(original_ids),
                "original_vertex_ids_nonidentity": bool(np.any(original_ids != np.arange(len(original_ids)))),
                "graph_cache_keys": ",".join(graph_keys),
                "sampled_max_relative_residual": manifest["sampled_maximum_relative_residual_norm"],
                "sampled_orthogonality_error": manifest[
                    "sampled_orthogonality_spectral_norm_error"
                ],
            }
        )
    return results


def verify_schur(root: Path) -> list[dict]:
    sources: list[tuple[str, Path]] = []
    sources.append(
        (
            "validation",
            root / "outputs" / "tunneling_validation" / "schur_sensitivity.csv",
        )
    )
    for case in ("er", "pa", "cl"):
        sources.append(
            (
                f"application/{case}",
                root
                / "outputs"
                / "tunneling_applications"
                / case
                / "schur_complement_sensitivity.csv",
            )
        )
    for model in MODELS:
        for seed in POTENTIAL_SEEDS:
            for ratio in (0.0, 1.0, 8.0):
                sources.append(
                    (
                        f"disorder/{model}/{seed}/{ratio_slug(ratio)}",
                        root
                        / "outputs"
                        / "pa_disorder_pilot"
                        / model
                        / f"potential_seed_{seed}"
                        / f"ratio_{ratio_slug(ratio)}"
                        / "schur_complement_sensitivity.csv",
                    )
                )
    results = []
    for label, path in sources:
        if not path.is_file():
            results.append({"source": label, "file_exists": False, "row_count": 0})
            continue
        rows = read_csv(path)
        grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
        for row in rows:
            key = tuple(
                str(row.get(field, ""))
                for field in (
                    "fixture",
                    "case_index",
                    "left_rank_largest_algebraic",
                    "right_rank_largest_algebraic",
                    "radius",
                    "energy",
                )
            )
            value = finite(row.get("absolute_mediated_coupling"))
            if math.isfinite(value):
                grouped[key].append(value)
        factors = []
        for values in grouped.values():
            if len(values) >= 2 and min(values) > 0.0:
                factors.append(max(values) / min(values))
        results.append(
            {
                "source": label,
                "file_exists": True,
                "row_count": len(rows),
                "eta_values": ",".join(sorted({str(row.get("eta")) for row in rows})),
                "near_pole_count": sum(row.get("near_pole_flag") == "True" for row in rows),
                "maximum_linear_solve_residual_norm": safe_max(
                    finite(row.get("linear_solve_residual_norm")) for row in rows
                ),
                "eta_paired_group_count": len(factors),
                "maximum_eta_coupling_factor": max(factors) if factors else math.nan,
                "median_eta_coupling_factor": safe_median(factors),
            }
        )
    return results


def verify_stage3(root: Path) -> tuple[list[dict], list[dict], list[dict]]:
    baseline_rows = []
    quality_rows = []
    pairing_rows = []
    for model in MODELS:
        source_dir = Path(
            read_json(
                root / "outputs" / "pa_disorder_pilot" / model / "potential_seed_931701" / "run_manifest.json"
            )["source_dir"]
        )
        with np.load(source_dir / "spectrum" / "single_graph_spectral_arrays.npz", allow_pickle=False) as data:
            source_values = np.asarray(data["eigenvalues"], dtype=float)
            source_ipr = np.asarray(data["ipr"], dtype=float)
            bulk = np.asarray(data["bulk50_indices"], dtype=int)
            source_keys = sorted(data.files)
        n = len(source_values)
        center = int(round((n - 1) * 0.5))
        q50 = np.arange(center - 25, center + 26, dtype=int)
        baseline_rows.append(
            {
                "model": model,
                "n": n,
                "source_array_keys": ",".join(source_keys),
                "exact_zero_eigenvalue_count": int(np.sum(source_values == 0.0)),
                "near_zero_abs_le_1e_12_count": int(np.sum(np.abs(source_values) <= 1e-12)),
                "historical_bulk_count": len(bulk),
                "historical_bulk_index_min": int(np.min(bulk)),
                "historical_bulk_index_max": int(np.max(bulk)),
                "historical_bulk_energy_min": float(np.min(source_values[bulk])),
                "historical_bulk_energy_max": float(np.max(source_values[bulk])),
                "historical_bulk_mean_ipr": float(np.mean(source_ipr[bulk])),
                "ascending_q50_count": len(q50),
                "ascending_q50_index_min": int(np.min(q50)),
                "ascending_q50_index_max": int(np.max(q50)),
                "ascending_q50_energy_min": float(np.min(source_values[q50])),
                "ascending_q50_energy_max": float(np.max(source_values[q50])),
                "ascending_q50_mean_ipr": float(np.mean(source_ipr[q50])),
                "window_index_overlap_count": len(np.intersect1d(bulk, q50)),
                "q50_to_historical_mean_ipr_ratio": float(np.mean(source_ipr[q50]) / np.mean(source_ipr[bulk])),
            }
        )
        potentials = []
        ratio0_values = []
        ratio0_ipr = []
        for seed in POTENTIAL_SEEDS:
            sweep = root / "outputs" / "pa_disorder_pilot" / model / f"potential_seed_{seed}"
            with np.load(sweep / "paired_potential.npz", allow_pickle=False) as data:
                xi = np.asarray(data["xi"], dtype=float)
            potentials.append(xi)
            with np.load(sweep / "ratio_0" / "spectrum_diagnostics.npz", allow_pickle=False) as data:
                values0 = np.asarray(data["eigenvalues"], dtype=float)
                ipr0 = np.asarray(data["ipr"], dtype=float)
                gaps0 = np.asarray(data["adjacent_raw_gaps"], dtype=float)
            ratio0_values.append(values0)
            ratio0_ipr.append(ipr0)
            pairing_rows.append(
                {
                    "model": model,
                    "potential_seed": seed,
                    "xi_sha256_raw_float64": hashlib.sha256(xi.tobytes()).hexdigest(),
                    "xi_min": float(np.min(xi)),
                    "xi_max": float(np.max(xi)),
                    "xi_mean": float(np.mean(xi)),
                    "a0_max_eigenvalue_difference_from_source": float(np.max(np.abs(values0 - source_values))),
                    "a0_max_ipr_difference_from_source": float(np.max(np.abs(ipr0 - source_ipr))),
                    "a0_max_saved_gap_difference_from_consecutive_diff": float(np.max(np.abs(gaps0 - np.diff(values0)))),
                }
            )
            for ratio in RATIOS:
                ratio_dir = sweep / f"ratio_{ratio_slug(ratio)}"
                rows = read_csv(ratio_dir / "upper_adjacent_effective_coupling.csv")
                radius2 = [row for row in rows if int(row["radius"]) == 2]
                valid = [row for row in radius2 if row["valid"] == "True"]
                residuals = []
                residual_gap_ratios = []
                for row in valid:
                    residual = max(finite(row["left_residual_norm"]), finite(row["right_residual_norm"]))
                    gap = finite(row["observed_adjacent_full_operator_gap"])
                    residuals.append(residual)
                    if gap > 0.0:
                        residual_gap_ratios.append(residual / gap)
                quality_rows.append(
                    {
                        "model": model,
                        "potential_seed": seed,
                        "a_over_s_ref": ratio,
                        "radius2_pair_count": len(radius2),
                        "projection_valid_count": len(valid),
                        "projection_invalid_count": len(radius2) - len(valid),
                        "absolute_threshold_0p25_pass_count": sum(value <= 0.25 for value in residuals),
                        "residual_over_observed_gap_lt_1_count": sum(value < 1.0 for value in residual_gap_ratios),
                        "residual_over_observed_gap_lt_0p1_count": sum(value < 0.1 for value in residual_gap_ratios),
                        "median_max_local_residual": safe_median(residuals),
                        "median_residual_over_observed_gap_proxy": safe_median(residual_gap_ratios),
                    }
                )
        pairing_rows.append(
            {
                "model": model,
                "potential_seed": "cross_seed_check",
                "xi_arrays_equal_between_seeds": bool(np.array_equal(potentials[0], potentials[1])),
                "a0_spectra_equal_between_seeds": bool(np.array_equal(ratio0_values[0], ratio0_values[1])),
                "a0_ipr_equal_between_seeds": bool(np.array_equal(ratio0_ipr[0], ratio0_ipr[1])),
            }
        )
    return baseline_rows, quality_rows, pairing_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory = artifact_inventory(args.root)
    revisions = source_revisions(args.root)
    stage1 = verify_stage1(args.root)
    stage2 = verify_stage2(args.root)
    er = verify_er(args.root)
    schur = verify_schur(args.root)
    baseline, quality, pairing = verify_stage3(args.root)
    write_csv(args.output_dir / "artifact_inventory.csv", inventory)
    write_csv(args.output_dir / "source_revisions.csv", revisions)
    write_csv(args.output_dir / "stage1_distance_integrity.csv", stage1)
    write_csv(args.output_dir / "er_integrity.csv", er)
    write_csv(args.output_dir / "stage2_residual_spacing_proxy.csv", stage2)
    write_csv(args.output_dir / "schur_integrity.csv", schur)
    write_csv(args.output_dir / "stage3_baseline_window_comparison.csv", baseline)
    write_csv(args.output_dir / "stage3_tunneling_quality_proxy.csv", quality)
    write_csv(args.output_dir / "stage3_pairing_and_a0_recovery.csv", pairing)
    code_hash_counts = Counter(
        str(row["code_revision_sha256"])
        for row in revisions
        if row["code_revision_sha256"]
    )
    summary = {
        "status": "COMPLETE",
        "generated_at_utc": utc_now(),
        "read_only_source_root": str(args.root),
        "audit_output_dir": str(args.output_dir),
        "expected_artifact_count": len(inventory),
        "missing_artifact_count": sum(not row["exists"] for row in inventory),
        "empty_artifact_count": sum(row["exists"] and not row["nonempty"] for row in inventory),
        "run_manifest_count": len(revisions),
        "run_manifest_complete_count": sum(row["status"] == "COMPLETE" for row in revisions),
        "code_revision_counts": dict(code_hash_counts),
        "stage1_all_distance_recomputations_exact_within_1e_12": all(
            float(row["maximum_mean_distance_recompute_error"]) <= 1e-12
            and float(row["maximum_mass_weighted_distance_recompute_error"]) <= 1e-12
            for row in stage1
        ),
        "stage2_absolute_threshold_pass_total": sum(int(row["absolute_threshold_0p25_pass_count"]) for row in stage2),
        "stage2_residual_over_observed_gap_lt_1_total": sum(int(row["residual_over_observed_gap_lt_1_count"]) for row in stage2),
        "schur_near_pole_total": sum(int(row.get("near_pole_count", 0)) for row in schur),
        "stage3_absolute_threshold_pass_total": sum(int(row["absolute_threshold_0p25_pass_count"]) for row in quality),
        "stage3_residual_over_observed_gap_lt_1_total": sum(int(row["residual_over_observed_gap_lt_1_count"]) for row in quality),
        "scope_note": "Residual/observed-gap is a diagnostic proxy, not the paper's approximate-energy isolation Delta.",
    }
    with (args.output_dir / "audit_verification_manifest.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
