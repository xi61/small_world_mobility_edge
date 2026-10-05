#!/usr/bin/env python3
"""Package every figure used by the combined clean-PA/disorder report."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CLEAN_FIGURES = (
    "localization_energy_nipr.png",
    "gap_ratio_windows.png",
    "gap_histograms_mean_rescaled.png",
    "first_ten_nonperron_cumulative_mass.png",
    "region_coverage_xy.png",
    "capture_and_gap_prediction.png",
    "localization_interaction_case_linkage.png",
)


def copy_checked(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def build_overview(figure_dir: Path) -> Path:
    panels = (
        ("localization_energy_nipr.png", "Clean spectra and localization"),
        ("gap_ratio_windows.png", "Clean adjacent-gap ratios"),
        ("cross_alpha_fixed_n_a.png", "Disorder comparison across α"),
        ("matched_clean_disorder_N10000_alpha3.png", "Matched clean/noisy ranks"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for axis, (name, title) in zip(axes.flat, panels):
        axis.imshow(plt.imread(figure_dir / name))
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle("Clean affine-PA localization and paired diagonal disorder")
    output = figure_dir / "comparison_overview.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-report", type=Path, required=True)
    parser.add_argument("--disorder-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name in CLEAN_FIGURES:
        copy_checked(args.clean_report / "figures" / name, args.output_dir / name)
    copy_checked(
        args.disorder_run / "report/figures/cross_alpha_fixed_n_a.png",
        args.output_dir / "cross_alpha_fixed_n_a.png",
    )
    for alpha, packaged in (
        ("1p05", "matched_clean_disorder_N10000_alpha1p05.png"),
        ("3", "matched_clean_disorder_N10000_alpha3.png"),
    ):
        source = (
            args.disorder_run
            / f"cases/N_10000/alpha_{alpha}/report/figures/matched_clean_disorder_comparison.png"
        )
        copy_checked(source, args.output_dir / packaged)
    copy_checked(
        args.output_dir / "matched_clean_disorder_N10000_alpha3.png",
        args.output_dir / "matched_clean_disorder_comparison.png",
    )
    build_overview(args.output_dir)
    print(args.output_dir)


if __name__ == "__main__":
    main()
