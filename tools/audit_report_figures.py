#!/usr/bin/env python3
"""Validate and map every packaged report figure to code and inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PA_REPORT = ROOT / "docs/reports/pa_three_model_edge_bulk_conclusive_report"
COMBINED_REPORT = ROOT / "docs/reports/PA_and_Anderson_report_20260924"
CSV_PATH = ROOT / "docs/figure_reproduction.csv"
MARKDOWN_PATH = ROOT / "docs/figure_reproduction.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pa_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with (PA_REPORT / "figure_manifest.csv").open(newline="") as handle:
        manifest = list(csv.DictReader(handle))
    if len(manifest) != 84:
        raise ValueError(f"expected 84 PA manifest rows, found {len(manifest)}")
    command = (
        "python tools/hoffman_pa_single_graph_edge_bulk.py --n 5000 --m 10 "
        "--model-family <family> --parameter-value <value> --graph-seed <seed> "
        "--reference-cdf <held-out-cdf.npz> --output-root <run-root>"
    )
    for entry in manifest:
        artifact = PA_REPORT / entry["packaged_path"]
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        actual = sha256(artifact)
        if entry.get("sha256") and actual != entry["sha256"]:
            raise ValueError(f"checksum mismatch: {artifact}")
        rows.append(
            {
                "report": "pa_three_model_edge_bulk_conclusive_report",
                "figure": entry["tex_relative_path"],
                "used_in_report": "true",
                "generator": "src/small_world_mobility_edge/preferential_attachment/single_graph_edge_bulk.py",
                "entry_point": "tools/hoffman_pa_single_graph_edge_bulk.py",
                "inputs": entry["original_source"],
                "configuration": "experiments/single_graph_edge_bulk/production_models.json",
                "command": command,
                "packaging_step": "scripts/revise_conclusive_report.py",
                "sha256": actual,
                "status": "recorded artifact present and checksum verified",
            }
        )
    return rows


def combined_rows() -> list[dict[str, str]]:
    tex = (COMBINED_REPORT / "PA_and_Anderson_report.tex").read_text()
    referenced = set(re.findall(r"\\figpath\{([^}]+\.png)\}", tex))
    packaged = {path.name for path in (COMBINED_REPORT / "figures").glob("*.png")}
    if len(referenced) != 11:
        raise ValueError(f"expected 11 TeX-referenced combined-report figures, found {len(referenced)}")
    if len(packaged) != 12:
        raise ValueError(f"expected 12 packaged combined-report figures, found {len(packaged)}")
    if not referenced <= packaged:
        raise FileNotFoundError(sorted(referenced - packaged))

    clean = set(CLEAN_NAMES)
    global_disorder = {"cross_alpha_fixed_n_a.png"}
    case_disorder = {
        "matched_clean_disorder_N10000_alpha1p05.png",
        "matched_clean_disorder_N10000_alpha3.png",
        "matched_clean_disorder_comparison.png",
    }
    rows: list[dict[str, str]] = []
    for name in sorted(packaged):
        if name in clean:
            generator = "src/small_world_mobility_edge/preferential_attachment/linear_pa_two_part_reporting.py"
            entry_point = "tools/hoffman_linear_pa_two_part.py"
            inputs = "<clean-run-root>/cases/*/{part1,part2}/tables"
            configuration = "experiments/linear_pa_two_part_20260924/config.json"
            command = (
                "python tools/hoffman_linear_pa_two_part.py --project-root . --config "
                "experiments/linear_pa_two_part_20260924/config.json --freeze-manifest "
                "<freeze.json> aggregate --output-root <clean-run-root> --report-dir <clean-report>"
            )
        elif name in global_disorder:
            generator = "src/small_world_mobility_edge/random_schrodinger/pa_diagonal_uniform_noise_reporting.py"
            entry_point = "tools/hoffman_pa_diagonal_uniform_noise.py"
            inputs = "<disorder-run-root>/cases/*/report/tables"
            configuration = "experiments/pa_diagonal_uniform_noise/production.json"
            command = (
                "python tools/hoffman_pa_diagonal_uniform_noise.py --project-root . --config "
                "experiments/pa_diagonal_uniform_noise/production.json --freeze-manifest "
                "<freeze.json> --run-root <disorder-run-root> global-aggregate"
            )
        elif name in case_disorder:
            generator = "src/small_world_mobility_edge/random_schrodinger/pa_diagonal_uniform_noise_reporting.py"
            entry_point = "tools/hoffman_pa_diagonal_uniform_noise.py"
            alpha = "1.05" if "alpha1p05" in name else "3.0"
            inputs = f"<disorder-run-root>/cases/N_10000/alpha_{alpha.replace('.', 'p')}/operators"
            configuration = "experiments/pa_diagonal_uniform_noise/production.json"
            command = (
                "python tools/hoffman_pa_diagonal_uniform_noise.py --project-root . --config "
                "experiments/pa_diagonal_uniform_noise/production.json --freeze-manifest "
                f"<freeze.json> --run-root <disorder-run-root> case-aggregate --n 10000 --alpha {alpha}"
            )
        elif name == "comparison_overview.png":
            generator = "tools/package_combined_report_figures.py"
            entry_point = generator
            inputs = "clean aggregate figures plus diagonal-disorder aggregate/case figures"
            configuration = "experiments/linear_pa_two_part_20260924/config.json; experiments/pa_diagonal_uniform_noise/production.json"
            command = (
                "python tools/package_combined_report_figures.py --clean-report <clean-report> "
                "--disorder-run <disorder-run-root> --output-dir <combined-report>/figures"
            )
        else:
            raise ValueError(f"unmapped combined-report figure: {name}")
        artifact = COMBINED_REPORT / "figures" / name
        rows.append(
            {
                "report": "PA_and_Anderson_report_20260924",
                "figure": f"figures/{name}",
                "used_in_report": str(name in referenced).lower(),
                "generator": generator,
                "entry_point": entry_point,
                "inputs": inputs,
                "configuration": configuration,
                "command": command,
                "packaging_step": "tools/package_combined_report_figures.py",
                "sha256": sha256(artifact),
                "status": "recorded artifact present",
            }
        )
    return rows


CLEAN_NAMES = (
    "localization_energy_nipr.png",
    "gap_ratio_windows.png",
    "gap_histograms_mean_rescaled.png",
    "first_ten_nonperron_cumulative_mass.png",
    "region_coverage_xy.png",
    "capture_and_gap_prediction.png",
    "localization_interaction_case_linkage.png",
)


def write_outputs(rows: list[dict[str, str]]) -> None:
    fieldnames = list(rows[0])
    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["report"]] = counts.get(row["report"], 0) + 1
    lines = [
        "# Figure reproduction map",
        "",
        "`figure_reproduction.csv` maps each packaged figure to its generating module, entry point, inputs, configuration, command template, checksum, and packaging step.",
        "",
        "| Report package | Packaged figures |",
        "|---|---:|",
        *[f"| `{name}` | {count} |" for name, count in sorted(counts.items())],
        "",
        "The PA conclusive report contains 84 TeX-referenced figures. The combined report contains 11 TeX-referenced figures plus one retained alternate disorder comparison. Commands containing angle-bracket fields require the corresponding production run roots or immutable caches; the repository intentionally does not commit those multi-gigabyte intermediates.",
        "",
        "Run `python tools/audit_report_figures.py --write` to verify all recorded assets and refresh the map.",
    ]
    MARKDOWN_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="refresh CSV and Markdown maps")
    args = parser.parse_args()
    rows = pa_rows() + combined_rows()
    if args.write:
        write_outputs(rows)
    print(f"verified {len(rows)} packaged figures across 2 report packages")


if __name__ == "__main__":
    main()
