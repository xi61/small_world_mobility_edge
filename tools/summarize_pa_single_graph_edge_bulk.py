#!/usr/bin/env python3
"""Validate and summarize transferred fixed-graph edge/bulk production outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Sequence

EXPECTED_MODELS = {
    "affine_alpha_1p05": ("degree_ccdf_alpha", 1.05, 4011773341),
    "nonlinear_zeta_1p5": ("attachment_zeta", 1.5, 1912200482),
    "affine_alpha_2p0": ("degree_ccdf_alpha", 2.0, 3479052237),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def truth(value: str) -> bool:
    return value.strip().lower() == "true"


def finite_values(rows: Sequence[dict[str, str]], field: str) -> list[float]:
    output = []
    for row in rows:
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            output.append(value)
    return output


def mean(rows: Sequence[dict[str, str]], field: str) -> float:
    values = finite_values(rows, field)
    return sum(values) / len(values) if values else math.nan


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def selected_window(
    rows: Sequence[dict[str, str]], region: str, size: int
) -> list[dict[str, str]]:
    region_rows = [row for row in rows if row["region"] == region]
    if size == 50:
        return region_rows
    return [row for row in region_rows if truth(row[f"in_window_{size}"])]


def validate_model(model_dir: Path) -> tuple[dict[str, object], list[dict[str, str]]]:
    model_key = model_dir.name
    parameter_name, parameter_value, graph_seed = EXPECTED_MODELS[model_key]
    manifest = json.loads((model_dir / "run_manifest.json").read_text())
    if manifest.get("status") != "COMPLETE":
        raise ValueError(f"{model_key}: manifest is not COMPLETE")
    if int(manifest["n"]) != 5000 or int(manifest["m"]) != 10:
        raise ValueError(f"{model_key}: unexpected N/m")
    if int(manifest["graph_seed"]) != graph_seed:
        raise ValueError(f"{model_key}: unexpected graph seed")
    if not math.isclose(float(manifest["parameter_value"]), parameter_value):
        raise ValueError(f"{model_key}: unexpected model parameter")
    rows = read_csv(model_dir / "tables" / "edge_bulk_eigenpairs.csv")
    support = read_csv(model_dir / "tables" / "top10_support_vertices.csv")
    residual = read_csv(model_dir / "tables" / "eigenpair_residual_audit.csv")
    figures = read_csv(model_dir / "figure_index.csv")
    if len(rows) != 100 or len(support) != 1000:
        raise ValueError(f"{model_key}: selected/support row count mismatch")
    if len([row for row in rows if row["region"] == "edge"]) != 50:
        raise ValueError(f"{model_key}: edge row count mismatch")
    if len([row for row in rows if row["region"] == "bulk"]) != 50:
        raise ValueError(f"{model_key}: bulk row count mismatch")
    perron = [row for row in rows if row["region"] == "edge" and row["edge_rank"] == "1"]
    if len(perron) != 1 or math.isfinite(float(perron[0]["adjacent_gap_ratio"])):
        raise ValueError(f"{model_key}: Perron ratio must be NaN")
    maximum_residual = max(finite_values(residual, "relative_residual_norm"))
    if maximum_residual >= 1e-8:
        raise ValueError(f"{model_key}: eigenpair residual check failed")
    summary: dict[str, object] = {
        "model": model_key,
        "parameter_name": parameter_name,
        "parameter_value": parameter_value,
        "graph_seed": graph_seed,
        "job_id": manifest.get("job_id"),
        "task_id": manifest.get("sge_task_id"),
        "status": manifest["status"],
        "runtime_seconds": manifest["runtime_seconds"],
        "eigendecomposition_seconds": manifest["eigendecomposition_seconds"],
        "peak_rss_platform_units": manifest["peak_rss_platform_units"],
        "zero_eigenvalue_count": manifest["zero_eigenvalue_count"],
        "bulk_center_energy": manifest["bulk_center_energy"],
        "maximum_checked_relative_residual": maximum_residual,
        "figure_count": len(figures),
    }
    for region in ("edge", "bulk"):
        for size in (10, 25, 50):
            subset = selected_window(rows, region, size)
            prefix = f"{region}{size}"
            summary[f"{prefix}_mean_NIPR"] = mean(subset, "NIPR")
            summary[f"{prefix}_mean_top10_mass"] = mean(subset, "top10_mass")
            summary[f"{prefix}_mean_local_r"] = mean(subset, "adjacent_gap_ratio")
            summary[f"{prefix}_finite_r_count"] = len(finite_values(subset, "adjacent_gap_ratio"))
            summary[f"{prefix}_mean_support_degree"] = mean(subset, "top10_mean_degree")
            summary[f"{prefix}_mean_induced_density"] = mean(subset, "induced_density")
            summary[f"{prefix}_mean_pairwise_distance"] = mean(subset, "mean_pairwise_distance")
            summary[f"{prefix}_mean_random_distance_zscore"] = mean(subset, "random_distance_zscore")
            summary[f"{prefix}_mean_random_induced_density_zscore"] = mean(subset, "random_induced_density_zscore")
            summary[f"{prefix}_mean_random_max_degree_zscore"] = mean(subset, "random_max_degree_zscore")
            summary[f"{prefix}_mean_random_mean_degree_zscore"] = mean(subset, "random_mean_degree_zscore")
    return summary, figures


def format_number(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.6g}" if math.isfinite(number) else "NA"


def report_text(output_root: Path, rows: Sequence[dict[str, object]], figure_count: int) -> str:
    result_lines = [
        "| model | seed | zero atom | edge-50 mean NIPR | bulk-50 mean NIPR | edge-50 mean top-10 mass | bulk-50 mean top-10 mass | edge-50 mean local r | bulk-50 mean local r |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        result_lines.append(
            "| {model} | {seed} | {zero} | {en} | {bn} | {em} | {bm} | {er} | {br} |".format(
                model=row["model"],
                seed=row["graph_seed"],
                zero=row["zero_eigenvalue_count"],
                en=format_number(row["edge50_mean_NIPR"]),
                bn=format_number(row["bulk50_mean_NIPR"]),
                em=format_number(row["edge50_mean_top10_mass"]),
                bm=format_number(row["bulk50_mean_top10_mass"]),
                er=format_number(row["edge50_mean_local_r"]),
                br=format_number(row["bulk50_mean_local_r"]),
            )
        )
    structural_lines = [
        "| model | edge/bulk mean support degree | edge/bulk induced density | edge/bulk mean distance | edge distance z | edge density z | edge mean-degree z |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        structural_lines.append(
            "| {model} | {ed}/{bd} | {ei}/{bi} | {ex}/{bx} | {dz} | {iz} | {gz} |".format(
                model=row["model"],
                ed=format_number(row["edge50_mean_support_degree"]),
                bd=format_number(row["bulk50_mean_support_degree"]),
                ei=format_number(row["edge50_mean_induced_density"]),
                bi=format_number(row["bulk50_mean_induced_density"]),
                ex=format_number(row["edge50_mean_pairwise_distance"]),
                bx=format_number(row["bulk50_mean_pairwise_distance"]),
                dz=format_number(row["edge50_mean_random_distance_zscore"]),
                iz=format_number(row["edge50_mean_random_induced_density_zscore"]),
                gz=format_number(row["edge50_mean_random_mean_degree_zscore"]),
            )
        )
    job_ids = sorted({str(row["job_id"]) for row in rows})
    return """# Fixed single-graph edge/bulk implementation report

## Completion

All three `N=5000`, `m=10` production realizations completed on Hoffman job array `{jobs}`. The transferred local output root is `{local_root}`; the authoritative Hoffman root is `<HOFFMAN_SCRATCH>/small_world_mobility_edge_single_graph_20260901/outputs`. No production task failed or remained incomplete. The combined figure index contains {figure_count} PNG files.

## Files added and existing code reused

Added: `single_graph_edge_bulk.py`, the Hoffman CLI/array/sync tools, `experiments/single_graph_edge_bulk/` inventory, focused tests, and this summarizer. No established source file was rewritten or modified.

The batch reuses the canonical `generate_preferential_attachment_graph`, the established dense symmetric adjacency eigensolver convention, `build_gap_records` atom/gap bookkeeping, the historical histogram quantile centers/bins and GOE/Poisson references, and each model's existing graph-fold-0 held-out CDF.

## Exact conventions

Eigenvalues are sorted algebraically ascending and the same permutation is applied to eigenvector columns. Sampled selected/quartile eigenpair residuals must be below `1e-8`. Edge top 10/25/50 are nested largest-algebraic windows and include Perron; Perron's two-sided ratio is `NA`. Bulk 50 is a 50-level window inside one atom-free segment, centered as close as feasible to the median valid rank while retaining one external level on either side; bulk 10/25 are nested at the same center.

Exact zero eigenvalues are retained in the saved spectrum/density but removed from continuous spacing statistics and split the spectrum; no gap or ratio crosses the atom. For each interior valid eigenvalue, `r_i=min(lambda_i-lambda_(i-1),lambda_(i+1)-lambda_i)/max(...)`. Every `r_i` is only an individual local-spacing diagnostic.

The local one-graph histograms preserve the established twelve centers, display bins, unfolding, and references. To obtain more support than a width-0.002 single-graph cell, the maximum width is 0.05; boundary cells are symmetrically clipped so their requested center remains exact. Every gap remains within one graph and is normalized by the mean positive gap of that same one-graph window.

Top support means the ten largest `|v_i(j)|^2` coordinates, where coordinate `j` is original vertex `j`. Induced-subgraph connectivity and full-graph shortest-path distance are stored separately. Comparisons use 1000 fixed-seed random ten-vertex sets. Random diagonal disorder was not implemented.

## Parameters and compact descriptive results

{results}

### Edge-50 structural support versus bulk-50

Negative distance z-scores mean closer support than random ten-vertex sets; positive density/degree z-scores mean denser or higher-degree support.

{structures}

These values are descriptive comparisons of fixed windows from one realization per model. They do not establish a GOE or Poisson distribution for a top-10/25/50 set.

## Output index

- `batch_summary.csv`: complete numerical batch summary, including all nested window sizes.
- `batch_figure_index.csv`: every generated figure and exact path.
- Each model: `run_manifest.json`, `IMPLEMENTATION_REPORT.md`, `figure_index.csv`, graph/spectral arrays, selected eigenvectors, edge/bulk eigenpair table, normalized support table, random baseline, residual audit, representative comparison table, and plots.
- Hoffman logs are copied into `{local_root}/logs`.
""".format(
        jobs=", ".join(job_ids),
        local_root=output_root.resolve(),
        figure_count=figure_count,
        results="\n".join(result_lines),
        structures="\n".join(structural_lines),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/preferential_attachment/single_graph_edge_bulk"),
    )
    args = parser.parse_args()
    output_root = args.output_root
    summaries = []
    figure_rows: list[dict[str, object]] = []
    for model_key in EXPECTED_MODELS:
        summary, figures = validate_model(output_root / model_key)
        summaries.append(summary)
        for row in figures:
            relative = Path(model_key) / row["figure"]
            path = output_root / relative
            figure_rows.append(
                {
                    "model": model_key,
                    "figure": str(relative),
                    "format": row["format"],
                    "size_bytes": int(row["size_bytes"]),
                    "exists": path.is_file(),
                }
            )
    expected_pngs = sorted(path.relative_to(output_root) for path in output_root.rglob("*.png"))
    indexed_pngs = sorted(Path(str(row["figure"])) for row in figure_rows)
    if expected_pngs != indexed_pngs or not all(bool(row["exists"]) for row in figure_rows):
        raise ValueError("combined figure index does not exactly match transferred PNG files")
    write_csv(output_root / "batch_summary.csv", summaries)
    write_csv(output_root / "batch_figure_index.csv", figure_rows)
    (output_root / "BATCH_IMPLEMENTATION_REPORT.md").write_text(
        report_text(output_root, summaries, len(figure_rows))
    )
    print(output_root / "BATCH_IMPLEMENTATION_REPORT.md")


if __name__ == "__main__":
    main()
