#!/usr/bin/env python3
"""Structural and support validation for abnormal-band microscope outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


MODELS = {
    "alpha_2p0": {"dq_0p0002": 25, "dq_0p0001": 40},
    "alpha_1p25": {"dq_0p0002": 25, "dq_0p0001": 40},
    "alpha_1p05": {"dq_0p0002": 25, "dq_0p0001": 40, "dq_0p00005": 50},
    "zeta_1p50": {"dq_0p0002": 25, "dq_0p0001": 40, "dq_0p00005": 70},
}
REQUIRED_COLUMNS = {
    "positive_gap_count",
    "gap_contributing_graphs",
    "valid_ratio_count",
    "ratio_contributing_graphs",
    "histogram_support",
    "ratio_support",
    "raw_r",
    "raw_r_se",
    "raw_r_bootstrap_ci_low",
    "raw_r_bootstrap_ci_high",
    "unfolded_r",
    "unfolded_r_bootstrap_ci_low",
    "unfolded_r_bootstrap_ci_high",
    "edge_rank_p10",
    "edge_rank_median",
    "edge_rank_p90",
    "mean_energy",
    "energy_min",
    "energy_max",
}


def read(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def truth(value: str) -> bool:
    return value.lower() == "true"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root
    issues = []
    checks = []
    for model, levels in MODELS.items():
        manifest = root / model / "run_manifest.json"
        if not manifest.is_file():
            issues.append(f"missing {manifest}")
            continue
        metadata = json.loads(manifest.read_text())
        if not metadata.get("sge_job_id"):
            issues.append(f"missing microscope job id for {model}")
        for level, expected in levels.items():
            directory = root / model / level
            standard_path = directory / "standard_summary.csv"
            shifted_path = directory / "half_shifted_summary.csv"
            if not standard_path.is_file() or not shifted_path.is_file():
                issues.append(f"missing grid summary for {model}/{level}")
                continue
            standard = read(standard_path)
            shifted = read(shifted_path)
            if len(standard) != expected:
                issues.append(f"{model}/{level}: expected {expected} standard rows, got {len(standard)}")
            if len(shifted) != expected - 1:
                issues.append(f"{model}/{level}: expected {expected-1} shifted rows, got {len(shifted)}")
            if standard and not REQUIRED_COLUMNS.issubset(standard[0]):
                issues.append(f"{model}/{level}: missing required summary columns")
            for left, right in zip(standard[:-1], standard[1:]):
                if not math.isclose(float(left["q_high"]), float(right["q_low"]), abs_tol=1e-10):
                    issues.append(f"{model}/{level}: non-contiguous standard bins")
                    break
            supported_histograms = sum(truth(row["histogram_support"]) for row in standard)
            histogram_files = list((directory / "histograms").glob("*.png"))
            if len(histogram_files) != supported_histograms:
                issues.append(
                    f"{model}/{level}: {supported_histograms} supported histograms but "
                    f"{len(histogram_files)} files"
                )
            for figure in (
                "standard_vs_half_shifted_r.png",
                "q_to_edge_rank.png",
                "localization_r_ks_overlay.png",
                "unfolded_spacing_histogram_panel.png",
                "independent_graph_block_replication.png",
            ):
                path = directory / figure
                if not path.is_file() or path.stat().st_size == 0:
                    issues.append(f"missing/empty {path}")
            checks.append({
                "model": model,
                "level": level,
                "standard_rows": len(standard),
                "shifted_rows": len(shifted),
                "histogram_supported_cells": supported_histograms,
                "ratio_supported_cells": sum(truth(row["ratio_support"]) for row in standard),
            })
        for required in (
            root / model / "negative_edge" / "dq_0p0002_standard_summary.csv",
            root / model / "fixed_rank" / "fixed_rank_summary.csv",
            root / model / "sliding_windows" / "sliding_window_summary.csv",
        ):
            if not required.is_file():
                issues.append(f"missing {required}")
    for required in (
        root / "model_comparison" / "dq_0p0001_raw_r_heatmap.png",
        root / "model_comparison" / "dq_0p0001_unfolded_r_heatmap.png",
        root / "model_comparison" / "dq_0p0001_signed_ks_heatmap.png",
        root / "model_comparison" / "support_and_augmentation_recommendations.csv",
        root / "model_comparison" / "augmentation_decision.json",
        root / "model_comparison" / "model_digest.csv",
        root / "abnormal_band_microscope_report.md",
    ):
        if not required.is_file() or required.stat().st_size == 0:
            issues.append(f"missing/empty {required}")
    decision_path = root / "model_comparison" / "augmentation_decision.json"
    decision = json.loads(decision_path.read_text()) if decision_path.is_file() else {}
    report = {
        "validation_version": "pa_abnormal_band_microscope_validation_v1",
        "status": "PASS" if not issues else "FAIL",
        "issue_count": len(issues),
        "issues": issues,
        "grid_checks": checks,
        "augmentation_decision": decision,
        "file_count": sum(1 for path in root.rglob("*") if path.is_file()),
    }
    destination = root / "model_comparison" / "validation_report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(destination)
    if issues:
        raise SystemExit("\n".join(issues))


if __name__ == "__main__":
    main()
