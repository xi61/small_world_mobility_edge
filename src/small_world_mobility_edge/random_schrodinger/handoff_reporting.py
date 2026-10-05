"""Aggregate retained clean-PA coupling evidence and produce review figures."""
from __future__ import annotations

import csv
import gzip
import math
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import read_json, sha256_file, source_configuration_identity, utc_now, write_csv, write_json

REPORTING_VERSION = "clean_pa_coupling_reporting_v1"


def read_rows(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def finite(row: Mapping[str, str], key: str) -> float:
    try:
        value = float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return math.nan
    return value


def truth(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def model_label(model_key: str) -> str:
    return {
        "affine_alpha_1p05": r"affine $\alpha=1.05$",
        "affine_alpha_2p0": r"affine $\alpha=2$",
        "nonlinear_zeta_1p5": r"nonlinear $\zeta=1.5$",
    }.get(model_key, model_key)


def run_clean_pa_aggregation(
    *,
    clean_pa_root: Path,
    output_dir: Path,
    project_root: Path,
    freeze_manifest: Path,
    config: Mapping[str, object],
) -> Path:
    clean_pa_root, output_dir = Path(clean_pa_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = output_dir / "figures"
    tables = output_dir / "tables"
    figures.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)
    freeze = read_json(freeze_manifest)
    source_id = source_configuration_identity(project_root)["sha256"]
    if freeze.get("status") != "COMPLETE" or freeze.get("source_configuration_id") != source_id:
        raise ValueError("freeze manifest does not match aggregation source/configuration")

    baseline_all: list[dict[str, object]] = []
    geometry_all: list[dict[str, object]] = []
    candidate_all: list[dict[str, object]] = []
    validation_all: list[dict[str, object]] = []
    schur_all: list[dict[str, object]] = []
    continuation_all: list[dict[str, object]] = []
    model_summary: list[dict[str, object]] = []
    input_files: list[Path] = []
    for model in config["models"]:
        key = str(model["model_key"])
        root = clean_pa_root / key
        manifest = read_json(root / "run_manifest.json")
        if manifest.get("status") != "COMPLETE":
            raise ValueError(f"clean-PA case is not COMPLETE: {key}")
        paths = {
            "baseline": root / "tables/baseline_selected_vector_metrics.csv.gz",
            "geometry": root / "tables/adjacent_pair_geometry_ranks_2_to_50.csv.gz",
            "candidate": root / "tables/structural_candidate_pairs.csv.gz",
            "validation": root / "tables/predictive_pair_validation.csv.gz",
            "schur": root / "tables/schur_feshbach_diagnostics.csv.gz",
            "continuation": root / "tables/continuation_tracked_modes.csv.gz",
        }
        rows = {name: read_rows(p) if p.is_file() else [] for name, p in paths.items()}
        input_files.extend(p for p in paths.values() if p.is_file())
        baseline_all.extend(rows["baseline"])
        geometry_all.extend(rows["geometry"])
        candidate_all.extend(rows["candidate"])
        validation_all.extend(rows["validation"])
        schur_all.extend(rows["schur"])
        continuation_all.extend(rows["continuation"])
        primary = [r for r in rows["validation"] if r.get("variant") == "radius_1_modes_3"]
        accepted = [r for r in primary if truth(r.get("accepted_predictive_two_mode"))]
        captures = [finite(r, "minimum_exact_captured_mass") for r in primary]
        leakages = [finite(r, "external_leakage_over_separation") for r in primary]
        gap_errors = [finite(r, "relative_gap_prediction_error") for r in primary]
        model_summary.append({
            "model_key": key,
            "n": model["n"],
            "m": model["m"],
            "graph_seed": model["graph_seed"],
            "source_cache_id": manifest.get("source_cache_id"),
            "source_configuration_id": manifest.get("source_configuration_id"),
            "primary_candidate_count": len(primary),
            "accepted_predictive_two_mode_count": len(accepted),
            "maximum_minimum_captured_mass": max((x for x in captures if math.isfinite(x)), default=math.nan),
            "minimum_leakage_over_external_separation": min((x for x in leakages if math.isfinite(x)), default=math.nan),
            "minimum_relative_gap_prediction_error": min((x for x in gap_errors if math.isfinite(x)), default=math.nan),
            "schur_row_count": len(rows["schur"]),
            "continuation_row_count": len(rows["continuation"]),
        })
    write_csv(tables / "model_summary.csv", model_summary)
    write_csv(tables / "all_models_primary_pair_validation.csv.gz", [r for r in validation_all if r.get("variant") == "radius_1_modes_3"])
    write_csv(tables / "all_models_schur_diagnostics.csv.gz", schur_all)
    write_csv(tables / "all_models_continuation.csv.gz", continuation_all)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), sharex="col")
    for column, model in enumerate(config["models"]):
        key = str(model["model_key"])
        rows = [r for r in baseline_all if r.get("model_key") == key and 2 <= int(r["upper_algebraic_rank"]) <= 50]
        rows.sort(key=lambda r: int(r["upper_algebraic_rank"]))
        x = [int(r["upper_algebraic_rank"]) for r in rows]
        axes[0, column].plot(x, [finite(r, "ipr") for r in rows], marker=".", lw=1)
        axes[1, column].plot(x, [finite(r, "maximum_absolute_coordinate") for r in rows], marker=".", lw=1, color="#b44e3b")
        axes[0, column].set_title(model_label(key))
        axes[0, column].set_ylabel("IPR")
        axes[1, column].set_ylabel(r"$\|v\|_\infty$")
        axes[1, column].set_xlabel("upper algebraic rank (Perron excluded)")
        axes[0, column].grid(alpha=0.25)
        axes[1, column].grid(alpha=0.25)
    fig.suptitle("Retained clean-PA edge localization diagnostics")
    fig.tight_layout()
    fig.savefig(figures / "clean_pa_edge_ipr_and_max_coordinate.png", dpi=180)
    plt.close(fig)

    geometry_colors = [finite(r, "full_mass_overlap") for r in geometry_all]
    geometry_finite = [value for value in geometry_colors if math.isfinite(value)]
    geometry_norm = matplotlib.colors.Normalize(
        vmin=min(geometry_finite, default=0.0), vmax=max(geometry_finite, default=1.0)
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True)
    for column, model in enumerate(config["models"]):
        key = str(model["model_key"])
        rows = [r for r in geometry_all if r.get("model_key") == key]
        x = [finite(r, "observed_adjacent_gap") for r in rows]
        y = [finite(r, "dominant_vertex_distance") for r in rows]
        color = [finite(r, "full_mass_overlap") for r in rows]
        scatter = axes[column].scatter(x, y, c=color, cmap="viridis", norm=geometry_norm, s=25)
        axes[column].set_xscale("log")
        axes[column].set_title(model_label(key))
        axes[column].set_xlabel("observed adjacent gap")
        axes[column].grid(alpha=0.25)
    axes[0].set_ylabel("distance between dominant vertices")
    fig.colorbar(scatter, ax=axes, label="full mass overlap", shrink=0.85)
    fig.suptitle("Adjacent non-Perron edge pairs: geometry versus observed gap")
    fig.savefig(figures / "clean_pa_adjacent_pair_geometry.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    primary_validation = [r for r in validation_all if r.get("variant") == "radius_1_modes_3"]
    leakage_colors = [finite(r, "external_leakage_over_separation") for r in primary_validation]
    positive_leakage = [value for value in leakage_colors if value > 0.0 and math.isfinite(value)]
    leakage_norm = matplotlib.colors.LogNorm(
        vmin=min(positive_leakage, default=1e-3),
        vmax=max(positive_leakage, default=1.0),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for column, model in enumerate(config["models"]):
        key = str(model["model_key"])
        rows = [r for r in validation_all if r.get("model_key") == key and r.get("variant") == "radius_1_modes_3"]
        x = [finite(r, "minimum_exact_captured_mass") for r in rows]
        y = [finite(r, "relative_gap_prediction_error") for r in rows]
        c = [finite(r, "external_leakage_over_separation") for r in rows]
        scatter = axes[column].scatter(x, y, c=c, norm=leakage_norm, cmap="magma", s=32)
        axes[column].axvline(0.90, color="black", ls="--", lw=0.8)
        axes[column].axhline(0.10, color="black", ls="--", lw=0.8)
        axes[column].set_yscale("log")
        axes[column].set_title(model_label(key))
        axes[column].set_xlabel("minimum captured mass")
        axes[column].grid(alpha=0.25)
    axes[0].set_ylabel("relative gap error")
    fig.colorbar(scatter, ax=axes, label=r"$\|R\|/g_{ext}$", shrink=0.85)
    fig.suptitle("Predictive two-mode gates (radius 1, three modes per region)")
    fig.savefig(figures / "clean_pa_predictive_gate_diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=False)
    for column, model in enumerate(config["models"]):
        key = str(model["model_key"])
        rows = [r for r in continuation_all if r.get("model_key") == key and int(r.get("tracked_state", 0)) == 1]
        for case in sorted(set(int(r["case_order"]) for r in rows)):
            case_rows = sorted((r for r in rows if int(r["case_order"]) == case), key=lambda r: finite(r, "eta_couple"))
            axes[column].plot([finite(r, "eta_couple") for r in case_rows], [finite(r, "tracked_pair_gap") for r in case_rows], marker="o", ms=3, label=f"case {case}")
        axes[column].set_title(model_label(key))
        axes[column].set_xlabel(r"$\eta_{couple}$")
        axes[column].set_ylabel("tracked pair gap")
        axes[column].grid(alpha=0.25)
        if rows:
            axes[column].legend(fontsize=8)
    fig.suptitle("Fixed-partition coupling continuation")
    fig.tight_layout()
    fig.savefig(figures / "clean_pa_coupling_continuation.png", dpi=180)
    plt.close(fig)

    evidence_files = sorted(set(input_files + list(tables.glob("*")) + list(figures.glob("*"))))
    evidence_rows = [{"path": str(p), "size_bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in evidence_files]
    write_csv(output_dir / "EVIDENCE_INVENTORY.csv", evidence_rows)
    total_accepted = sum(int(row["accepted_predictive_two_mode_count"]) for row in model_summary)
    report_lines = [
        "# Clean-PA geometry and coupling execution report", "",
        f"Generated: {utc_now()}", "",
        "## Scope", "",
        "This report covers only the retained clean preferential-attachment graphs. New diagonal-disorder and random 3-regular Anderson work remained paused. The operator is the unscaled simple-undirected adjacency matrix in original vertex order. Perron rank 1 is reported in baseline metrics but excluded from adjacent-pair matching.", "",
        "## Completion", "",
        "All three retained cases completed baseline geometry. The two affine cases also completed structural dictionaries, predictive validation, Schur diagnostics, and fixed-partition continuation. For nonlinear zeta=1.5, every candidate structural construction was rejected by the predeclared 20% locality cap, so no Schur or continuation case was forced. Failed and inadequate constructions remain in the tables.", "",
        "| model | seed | primary candidates | accepted predictive pairs | best captured mass | best leakage/g_ext | best relative gap error |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in model_summary:
        report_lines.append(f"| {row['model_key']} | {row['graph_seed']} | {row['primary_candidate_count']} | {row['accepted_predictive_two_mode_count']} | {row['maximum_minimum_captured_mass']:.4g} | {row['minimum_leakage_over_external_separation']:.4g} | {row['minimum_relative_gap_prediction_error']:.4g} |")
    report_lines.extend([
        "", "## Interpretation", "",
        f"The complete predeclared gate accepted {total_accepted} primary two-mode candidates across the three graphs. Acceptance requires capture, leakage/isolation, gap, conditional weight, and dictionary-enlargement stability simultaneously. Schur rows are conditional background-mediated diagnostics and are not promoted to accepted explanations when the reference subspace itself fails those gates.", "",
        "The continuation uses one fixed graph partition for every eta value, scales only inter-block edges, tracks states by overlap assignment, records projector overlap near degeneracy, and automatically refines ambiguous intervals up to the configured 25-point cap. It does not use off-diagonal matrix elements between exact eigenvectors as tunneling estimates and does not identify observed gaps with uncoupled detuning.", "",
        "These are three N=5000 single-graph retained cases. They can establish finite-graph behavior and failure/success of the proposed mechanism on those cases, but cannot establish a localized phase, a mobility edge, or distinct asymptotic scaling regimes.", "",
        "## Files", "",
        "- `tables/model_summary.csv`: compact case-level outcomes.",
        "- `tables/all_models_primary_pair_validation.csv.gz`: complete primary predictive gates.",
        "- `tables/all_models_schur_diagnostics.csv.gz`: epsilon-resolvent sensitivity and direct-versus-mediated terms.",
        "- `tables/all_models_continuation.csv.gz`: complete tracked continuation rows.",
        "- `EVIDENCE_INVENTORY.csv`: paths, sizes, and SHA-256 hashes.",
        "- `figures/`: four PNG review figures.", "",
    ])
    (output_dir / "CLEAN_PA_EXECUTION_REPORT.md").write_text("\n".join(report_lines))
    index_lines = ["# Clean-PA results index", "", "Recommended review order:", "", "1. `CLEAN_PA_EXECUTION_REPORT.md`", "2. `tables/model_summary.csv`", "3. `tables/all_models_primary_pair_validation.csv.gz`", "4. `tables/all_models_schur_diagnostics.csv.gz`", "5. `tables/all_models_continuation.csv.gz`", "6. `figures/`", "7. `EVIDENCE_INVENTORY.csv`", "", "The complete per-model geometry, dictionaries, rejection/merge records, arrays, and path rows remain under the sibling `clean_pa/<model_key>/` directories.", ""]
    (output_dir / "RESULTS_INDEX.md").write_text("\n".join(index_lines))
    write_json(output_dir / "run_manifest.json", {
        "status": "COMPLETE", "stage_version": REPORTING_VERSION,
        "source_configuration_id": source_id, "finished_at_utc": utc_now(),
        "model_count": len(model_summary), "accepted_predictive_two_mode_count": total_accepted,
        "figure_count": len(list(figures.glob("*.png"))),
    })
    return output_dir
