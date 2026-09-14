from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


SUBLINEAR = (
    (0.0, "0p0"),
    (0.25, "0p25"),
    (0.40, "0p4"),
    (0.50, "0p5"),
    (0.60, "0p6"),
    (0.75, "0p75"),
    (0.90, "0p9"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def finite_mean(values) -> float:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.mean(values) if values else math.nan


def fmt(value, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}g}" if math.isfinite(number) else "NA"


def markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    output = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    output.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return output


def one(rows: list[dict[str, str]], **criteria: object) -> dict[str, str]:
    return next(
        row
        for row in rows
        if all(str(row[key]) == str(value) for key, value in criteria.items())
    )


def sublinear_rows(mobility_root: Path) -> tuple[list[list[object]], list[str]]:
    table: list[list[object]] = []
    details: list[str] = []
    for zeta, slug in SUBLINEAR:
        directory = (
            mobility_root
            / "nonlinear"
            / "sublinear"
            / "n_5000_m_10"
            / f"attachment_zeta_{slug}"
        )
        structural = json.loads((directory / "structural" / "structural_summary.json").read_text())
        manifest = json.loads((directory / "run_manifest.json").read_text())
        coarse = read_csv(
            directory
            / "coarse"
            / "q0_1_dq_0p002"
            / "joined_localization_level_statistics.csv"
        )
        atoms = read_csv(directory / "global_spectrum" / "atom_multiplicity_by_graph.csv")
        structural_by_graph = read_csv(directory / "structural" / "structural_by_graph.csv")
        hub = read_csv(
            directory
            / "global_spectrum"
            / "nonlinear_hub_edge"
            / "hub_edge_summary.csv"
        )
        q = [float(row["q_center"]) for row in coarse]
        nipr = [float(row["eigenvector_mean_n_times_ipr"]) for row in coarse]
        raw_r = [float(row["graph_equal_mean_raw_r_positive_conditioned"]) for row in coarse]
        bulk = [index for index, value in enumerate(q) if 0.10 <= value <= 0.90]
        negative = [index for index, value in enumerate(q) if value <= 0.10]
        positive = [index for index, value in enumerate(q) if value >= 0.90]
        maximum_index = max(range(len(nipr)), key=nipr.__getitem__)
        positive_rank_one = one(
            hub,
            record_type="fixed_rank",
            spectral_side="positive_nonperron",
            rank_or_top_count="1",
        )
        negative_rank_one = one(
            hub,
            record_type="fixed_rank",
            spectral_side="negative_absolute",
            rank_or_top_count="1",
        )
        positive_fit = one(
            hub,
            record_type="loglog_fit",
            spectral_side="positive_nonperron",
            rank_or_top_count="50",
        )
        negative_fit = one(
            hub,
            record_type="loglog_fit",
            spectral_side="negative_absolute",
            rank_or_top_count="50",
        )
        global_summary = manifest["global_spectrum_summary"]
        zero_mean = finite_mean(row["zero_multiplicity"] for row in atoms)
        plus_one_mean = finite_mean(row["plus_one_multiplicity"] for row in atoms)
        minus_one_mean = finite_mean(row["minus_one_multiplicity"] for row in atoms)
        candidate_text = "; ".join(
            f"{candidate['category']} [{fmt(candidate.get('q_low'), 5)},{fmt(candidate.get('q_high'), 5)}]"
            for candidate in manifest.get("candidate_regions", [])
        )
        table.append(
            [
                fmt(zeta, 3),
                fmt(structural["mean_maximum_degree"]),
                fmt(structural["mean_degree_gini"]),
                fmt(structural["mean_top_1_percent_degree_share"]),
                fmt(zero_mean),
                fmt(max(nipr)),
                fmt(q[maximum_index], 4),
                fmt(finite_mean(raw_r[index] for index in bulk)),
                fmt(finite_mean(raw_r[index] for index in negative)),
                fmt(finite_mean(raw_r[index] for index in positive)),
                fmt(positive_rank_one["median_lambda_over_sqrt_degree"]),
                fmt(negative_rank_one["median_lambda_over_sqrt_degree"]),
                fmt(positive_fit["mean_slope"]),
                fmt(negative_fit["mean_slope"]),
                candidate_text,
            ]
        )
        stretched = structural.get("sublinear_stretched_exponential_diagnostic", {})
        details.extend(
            [
                f"### zeta={zeta:g}",
                "",
                f"- Degree structure: mean d_max={fmt(structural['mean_maximum_degree'])}, "
                f"d_max/N={fmt(structural['mean_maximum_degree_over_n'])}, "
                f"d_max/sum(d)={fmt(structural['mean_maximum_degree_over_total_degree'])}, "
                f"mean degree={fmt(structural['mean_realized_mean_degree'])}, minimum degree={fmt(structural['mean_minimum_degree'])}, "
                f"Gini={fmt(structural['mean_degree_gini'])}, "
                f"top-1/top-5/top-10/top-1% shares={fmt(structural['mean_top_1_degree_share'])}/"
                f"{fmt(structural['mean_top_5_degree_share'])}/{fmt(structural['mean_top_10_degree_share'])}/"
                f"{fmt(structural['mean_top_1_percent_degree_share'])}, "
                f"degree-m fraction={fmt(structural['mean_degree_m_fraction'])}, "
                f"highest-degree birth rank={fmt(structural['mean_maximum_degree_vertex_birth_index'])}, "
                f"mean component count={fmt(structural['mean_num_components'])}. All graphs connected={structural['connected_graph_fraction'] == 1.0}.",
                f"- Stretched-exponential diagnostic: mean slope={fmt(stretched.get('mean_slope'))}, "
                f"mean R2={fmt(stretched.get('mean_r2'))}; this is a validation diagnostic, not a forced fit.",
                f"- Spectrum: continuous range [{fmt(global_summary['continuous_spectral_min'])}, {fmt(global_summary['continuous_spectral_max'])}], "
                f"mean Perron={fmt(finite_mean(row['perron_eigenvalue'] for row in structural_by_graph))}, "
                f"mean negative edge={fmt(finite_mean(row['lower_spectral_edge'] for row in structural_by_graph))}, "
                f"mean positive non-Perron edge={fmt(finite_mean(row['upper_nonperron_spectral_edge'] for row in structural_by_graph))}; "
                f"mean zero multiplicity={fmt(zero_mean)}, mean -1 multiplicity={fmt(minus_one_mean)}, mean +1 multiplicity={fmt(plus_one_mean)}.",
                f"- Localization/spacing: max coarse NIPR={fmt(max(nipr))} at q={fmt(q[maximum_index], 4)}; "
                f"bulk raw r={fmt(finite_mean(raw_r[index] for index in bulk))}; "
                f"negative-edge raw r={fmt(finite_mean(raw_r[index] for index in negative))}; "
                f"positive-edge raw r={fmt(finite_mean(raw_r[index] for index in positive))}; "
                f"positive-edge inclusive raw r={fmt(finite_mean(coarse[index]['graph_equal_mean_raw_r_inclusive'] for index in positive))}; "
                f"mean positive-edge KS(GOE)/KS(Poisson)="
                f"{fmt(finite_mean(coarse[index]['crossfit_unfolded_KS_to_GOE'] for index in positive))}/"
                f"{fmt(finite_mean(coarse[index]['crossfit_unfolded_KS_to_Poisson'] for index in positive))}; "
                f"maximum coarse zero-gap fraction={fmt(max(float(row['zero_gap_fraction']) for row in coarse))}.",
                f"- Hub-edge test: rank-1 R+={fmt(positive_rank_one['median_lambda_over_sqrt_degree'])}, "
                f"R-={fmt(negative_rank_one['median_lambda_over_sqrt_degree'])}; "
                f"top-50 b+={fmt(positive_fit['mean_slope'])} (R2={fmt(positive_fit['mean_r2'])}), "
                f"b-={fmt(negative_fit['mean_slope'])} (R2={fmt(negative_fit['mean_r2'])}).",
                f"- Candidate screen: {candidate_text or 'none'}.",
                "",
            ]
        )
    return table, details


def extreme_rows(mobility_root: Path) -> tuple[list[list[object]], list[dict[str, str]]]:
    comparison_root = mobility_root / "extreme_positive_edge_quantiles"
    path = comparison_root / "model_comparison" / "extreme_edge_model_by_interval_summary.csv"
    graph_counts: dict[tuple[str, float], int] = {}
    for manifest_path in comparison_root.glob("*/*/run_manifest.json"):
        manifest = json.loads(manifest_path.read_text())
        model = manifest.get("model", {})
        if "model_family" in model and "parameter_value" in model and "graph_count" in manifest:
            graph_counts[(str(model["model_family"]), float(model["parameter_value"]))] = int(
                manifest["graph_count"]
            )
    exact = [
        row
        for row in read_csv(path)
        if row["analysis_kind"] == "EXACT_REQUESTED_INTERVAL"
    ]
    rows: list[list[object]] = []
    for row in exact:
        histogram = row.get("unfolded_histogram_panel_relative_path", row["unfolded_histogram_panel_path"])
        ensemble_graph_count = row.get("ensemble_graph_count") or graph_counts[
            (row["model_family"], float(row["parameter_value"]))
        ]
        rows.append(
            [
                row["model_label"],
                f"[{fmt(row['nominal_q_low'], 6)},{fmt(row['nominal_q_high'], 6)})",
                ensemble_graph_count,
                row["number_of_contributing_graphs"],
                row["crossfit_unfolded_histogram_sample_count"],
                row["number_of_zero_gaps"],
                row["number_of_valid_raw_ratios"],
                f"{fmt(row['graph_equal_mean_raw_r_positive_conditioned'])} +/- {fmt(row['raw_r_positive_conditioned_standard_error'], 3)}",
                fmt(row["absolute_distance_raw_r_positive_conditioned_to_GOE"]),
                fmt(row["absolute_distance_raw_r_positive_conditioned_to_Poisson"]),
                fmt(row["crossfit_unfolded_KS_to_GOE"]),
                fmt(row["crossfit_unfolded_KS_to_Poisson"]),
                f"[{fmt(row['E_min'])},{fmt(row['E_max'])}]",
                row["histogram_status"],
                row["atom_warning"],
                histogram,
            ]
        )
    return rows, exact


def figure_paths(mobility_root: Path) -> list[str]:
    patterns = (
        "nonlinear/sublinear/n_5000_m_10/controlled_family_summary/*.png",
        "nonlinear/sublinear/n_5000_m_10/attachment_zeta_*/structural/*.png",
        "nonlinear/sublinear/n_5000_m_10/attachment_zeta_*/global_spectrum/*.png",
        "nonlinear/sublinear/n_5000_m_10/attachment_zeta_*/global_spectrum/nonlinear_hub_edge/*.png",
        "nonlinear/n_5000_m_10/nonlinear_hub_edge_cross_regime_summary/*.png",
        "extreme_positive_edge_quantiles/*/*/*.png",
        "extreme_positive_edge_quantiles/model_comparison/*.png",
    )
    paths: list[str] = []
    for pattern in patterns:
        paths.extend(str(path.relative_to(mobility_root)) for path in sorted(mobility_root.glob(pattern)))
    return sorted(set(paths))


def build_report(project_root: Path, job_ledger: Path | None, output: Path) -> Path:
    mobility_root = project_root / "outputs" / "preferential_attachment" / "mobility_edge"
    sublinear, details = sublinear_rows(mobility_root)
    extreme, exact = extreme_rows(mobility_root)
    lines = [
        "# Preferential-attachment sublinear and extreme-edge execution report",
        "",
        "Authoritative methodology: `pa_mobility_edge_gap_first_v2_atom_segmented`.",
        "",
        "## Code and execution scope",
        "",
        "The existing nonlinear generator was generalized to finite zeta >= 0; no duplicate generator was introduced. Existing affine/superlinear spectra were postprocessed, while only graph indices beyond 499 were added where the two narrowest ensemble-CDF edge cells required more support.",
        "",
        "Changed implementation points:",
        "",
        "- `preferential_attachment/baseline.py`: nonlinear kernel validation and the explicit zeta=0 uniform-weight boundary.",
        "- `preferential_attachment/full_spectrum_scan.py`: sublinear regime validation, metadata, and separate canonical output hierarchy.",
        "- `preferential_attachment/mobility_edge_production_aggregate.py`: stretched-exponential and nonlinear hub-edge diagnostics.",
        "- `preferential_attachment/mobility_edge_family_summary.py`: sublinear family and cross-regime hub-edge comparisons.",
        "- `preferential_attachment/extreme_edge_quantiles.py`: v2 gap-first cross-fitted ensemble-CDF edge analysis, support audit, adaptive companions, and comparison figures.",
        "- `preferential_attachment/extreme_edge_augmentation.py`: deterministic, eigenvalues-only graph augmentation with content validation.",
        "- Hoffman wrappers and focused tests under `tools/` and `tests/`.",
        "",
        "## Reuse audit",
        "",
        "- Sublinear zeta=0,0.25,0.40,0.50,0.60,0.75,0.90: `NEW_EIGENDECOMPOSITION_REQUIRED` (500 new graphs per point).",
        "- Affine alpha=2.00,1.50,1.25,1.05 and superlinear zeta=1.05,1.20,1.50 base ensembles: `REPOSTPROCESS_ONLY`; their original 500 spectra were not recomputed.",
        "- Narrow resolvable edge cells: `AUGMENT_GRAPHS_ONLY`; deterministic graph indices began at 500 and stopped at the recorded per-model target/G_max.",
        "- Superlinear hub-edge comparison: eigenspectra were reused; degree arrays were reconstructed by deterministic graph replay where absent.",
        "",
        "## Gap, quantile, and inference conventions",
        "",
        "Every gap is formed inside one graph and one atom-free segment after positive-edge Perron removal. Raw adjacent-gap ratios are computed before unfolding. Requested edge cells use the historical Chung--Lu-aligned ensemble empirical-CDF energy coordinate, implemented as a held-out five-fold training-only CDF evaluated at each within-graph gap midpoint. The graph-local rank coordinate is retained alongside it. Consequently, the two width-5e-5 cells are resolvable in the ensemble-CDF coordinate even though they are finer than the single-graph rank lattice. A supported displayed histogram requires at least 100 positive within-graph gaps and 50 contributing graphs; this is pooled support across independent graphs, not 100 gaps from every graph.",
        "",
        "The principal spacing histogram is the unit-mean five-fold cross-fitted unfolded spacing distribution. Raw-gap histograms are scale diagnostics only. Exact atoms, zero gaps, positive-conditioned ratios, and inclusive ratios remain separate products.",
        "",
    ]
    if job_ledger and job_ledger.is_file():
        jobs = read_csv(job_ledger)
        lines.extend(["## Hoffman SGE jobs", ""])
        lines.extend(
            markdown_table(
                list(jobs[0]) if jobs else [],
                [[row[key] for key in row] for row in jobs],
            )
        )
        lines.append("")
    lines.extend(
        [
            "## Sublinear family summary",
            "",
            *markdown_table(
                [
                    "zeta",
                    "mean dmax",
                    "Gini",
                    "top-1% share",
                    "zero mult.",
                    "max NIPR",
                    "q(max)",
                    "bulk r",
                    "negative r",
                    "positive r",
                    "R+ rank1",
                    "R- rank1",
                    "b+ top50",
                    "b- top50",
                    "candidates",
                ],
                sublinear,
            ),
            "",
            *details,
            "## Exact extreme positive-edge intervals",
            "",
            *markdown_table(
                [
                    "model",
                    "q interval",
                    "G",
                    "graphs",
                    "+ gaps",
                    "zero",
                    "ratios",
                    "raw r +/- SE",
                    "|r-GOE|",
                    "|r-Pois|",
                    "KS GOE",
                    "KS Pois",
                    "energy range",
                    "status",
                    "atom",
                    "histogram panel",
                ],
                extreme,
            ),
            "",
            "## Resolution audit",
            "",
        ]
    )
    for q_low, q_high in ((0.9999, 0.99995), (0.99995, 1.0)):
        selected = [
            row
            for row in exact
            if math.isclose(float(row["nominal_q_low"]), q_low)
            and math.isclose(float(row["nominal_q_high"]), q_high)
        ]
        lines.append(
            f"- [{q_low},{q_high}): ensemble/CDF coordinate is mathematically resolvable; "
            f"final support statuses: "
            + ", ".join(
                f"{row['model_label']}={row['histogram_status']} ({row['crossfit_unfolded_histogram_sample_count']} gaps)"
                for row in selected
            )
            + "."
        )
    lines.extend(["", "## Figures", ""])
    lines.extend(f"- `{path}`" for path in figure_paths(mobility_root))
    lines.extend(
        [
            "",
            "## Portable numerical sources",
            "",
            "- `nonlinear/sublinear/n_5000_m_10/controlled_family_summary/family_summary.csv`",
            "- `nonlinear/n_5000_m_10/nonlinear_hub_edge_cross_regime_summary/hub_edge_cross_regime_summary.csv`",
            "- `extreme_positive_edge_quantiles/model_comparison/extreme_edge_model_by_interval_summary.csv`",
            "- `extreme_positive_edge_quantiles/model_comparison/augmentation_requirement_audit.csv`",
            "- `pa_sublinear_extreme_edge_sge_job_ledger.csv`",
            "- `pa_sublinear_extreme_edge_sync_checksum_audit.csv`",
            "",
            "## Validation and interpretation",
            "",
            "The production validators enforce consecutive graph indices, deterministic seeds, model/parameter/N/m/version agreement, readable chunks, no graph-boundary gap, no atom bridge, Perron removal, pre-unfolding raw r, and fold exclusion. Unsupported exact cells are labeled rather than plotted inferentially; adaptive companion intervals never replace their nominal requested rows.",
            "",
            "Across the sublinear family, hub strength and edge NIPR increase smoothly toward zeta=1, while positive-conditioned raw r remains close to the GOE reference and cross-fitted unfolded KS is much smaller to GOE than to Poisson. The zeta=1/2 structural landmark is not accompanied by a demonstrated spectral transition. The hub-edge regressions retain strong rank association but do not support one universal lambda/sqrt(d) prefactor or exponent across nonlinear regimes, especially under superlinear condensation. High IPR with GOE-like positive spacings is therefore reported as edge localization/prelocalization, not a clean Anderson-style mobility edge.",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--job-ledger", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build_report(args.project_root, args.job_ledger, args.output))


if __name__ == "__main__":
    main()
