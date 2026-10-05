"""Aggregation, figures, and report for the two-part affine-PA experiment."""

from __future__ import annotations

import csv
import gzip
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from .linear_pa_two_part import alpha_slug, case_directory, resolve_case
from ..random_schrodinger.common import read_json, sha256_file, utc_now, write_csv, write_json


def _read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value: object, default: float = math.nan) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _boolean(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _json_value(value: object) -> object:
    if isinstance(value, (list, dict)):
        return value
    if value in (None, ""):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return value


def _finite_mean(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else math.nan


def discover_cases(output_root: Path, config: Mapping[str, object]) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for n in config["sizes"]:
        for alpha in config["alpha_grid"]:
            case_dir = case_directory(output_root, int(n), float(alpha))
            part1_path = case_dir / "part1_manifest.json"
            part2_path = case_dir / "part2/manifest.json"
            part1 = read_json(part1_path) if part1_path.exists() else {"status": "MISSING"}
            part2 = read_json(part2_path) if part2_path.exists() else {"status": "MISSING"}
            cases.append({
                "case_id": resolve_case(config, n=int(n), alpha=float(alpha))["case_id"],
                "n": int(n),
                "degree_ccdf_alpha": float(alpha),
                "case_dir": str(case_dir),
                "part1_status": part1.get("status"),
                "part2_status": part2.get("status"),
                "part1_source_configuration_id": part1.get("source_configuration_id"),
                "part2_source_configuration_id": part2.get("source_configuration_id"),
                "part1_job_id": part1.get("job_id"),
                "part1_sge_task_id": part1.get("sge_task_id"),
                "part2_job_id": part2.get("job_id"),
                "part2_sge_task_id": part2.get("sge_task_id"),
                "graph_seed": part1.get("resolved_case", {}).get("graph_seed"),
                "graph_sha256_edges": part1.get("graph_summary", {}).get("graph_sha256_edges"),
                "num_edges": part1.get("graph_summary", {}).get("num_edges"),
                "maximum_degree": part1.get("graph_summary", {}).get("maximum_degree"),
                "empirical_degree_ccdf_alpha_hill": part1.get("graph_summary", {}).get("degree_ccdf_alpha_hill"),
                "maximum_relative_eigenpair_residual": part1.get("spectral_summary", {}).get("maximum_relative_residual"),
                "part2_target_count": part2.get("target_count"),
                "dictionary_mode_row_count": part2.get("dictionary_mode_row_count"),
                "pair_coupling_row_count": part2.get("pair_coupling_row_count"),
                "schur_row_count": part2.get("schur_row_count"),
            })
    return cases


def _collect_case_table(cases: Sequence[Mapping[str, object]], relative: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for case in cases:
        path = Path(str(case["case_dir"])) / relative
        if not path.exists():
            continue
        for row in _read_csv(path):
            row.setdefault("case_id", str(case["case_id"]))
            row.setdefault("n", str(case["n"]))
            row.setdefault("degree_ccdf_alpha", str(case["degree_ccdf_alpha"]))
            rows.append(row)
    return rows


def _case_rows(rows: Sequence[Mapping[str, object]], case_id: str) -> list[Mapping[str, object]]:
    return [row for row in rows if str(row.get("case_id")) == case_id]


def _target_ids(rows: Sequence[Mapping[str, object]], predicate) -> set[str]:
    return {
        str(row.get("target_id"))
        for row in rows
        if row.get("target_id") not in (None, "") and predicate(row)
    }


def summarize_scientific_cases(
    cases: Sequence[Mapping[str, object]],
    localization: Sequence[Mapping[str, object]],
    windows: Sequence[Mapping[str, object]],
    coverage: Sequence[Mapping[str, object]],
    dictionary: Sequence[Mapping[str, object]],
    pairs: Sequence[Mapping[str, object]],
    targets: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Make one non-inferential localization/interaction summary per fixed graph."""
    result: list[dict[str, object]] = []
    for case in cases:
        case_id = str(case["case_id"])
        n = int(case["n"])
        loc = _case_rows(localization, case_id)
        upper = [row for row in loc if 2 <= int(_number(row.get("decreasing_algebraic_rank"), -1)) <= min(51, n)]
        bulk_lo = max(1, int(math.floor(0.4 * n)) + 1)
        bulk_hi = min(n, int(math.ceil(0.6 * n)))
        bulk = [row for row in loc if bulk_lo <= int(_number(row.get("decreasing_algebraic_rank"), -1)) <= bulk_hi]
        lower = [row for row in loc if max(1, n - 49) <= int(_number(row.get("decreasing_algebraic_rank"), -1)) <= n]
        case_windows = _case_rows(windows, case_id)
        top50 = next((
            row for row in case_windows
            if row.get("window_family") == "cumulative_nonperron"
            and row.get("window_id") == "nonperron_top_50"
        ), {})
        case_targets = _case_rows(targets, case_id)
        case_coverage = _case_rows(coverage, case_id)
        case_dictionary = _case_rows(dictionary, case_id)
        case_pairs = _case_rows(pairs, case_id)
        target_ids = _target_ids(case_targets, lambda row: True)
        reference_coverage = [
            row for row in case_coverage
            if row.get("construction_family") == "mass_informed_greedy"
            and row.get("grouping_variant") == "overlap_component_grouped"
            and int(_number(row.get("radius"), -1)) == 2
            and math.isclose(_number(row.get("x_target_mass")), 0.8)
            and math.isclose(_number(row.get("y_union_fraction_cap")), 0.2)
        ]
        mass_dictionary = [
            row for row in case_dictionary
            if row.get("construction_family") == "mass_informed_greedy"
        ]
        reference_dictionary = [row for row in mass_dictionary if _boolean(row.get("reference_setting_in_equivalents"))]
        any_dictionary_targets = _target_ids(case_dictionary, lambda row: True)
        mass_dictionary_targets = _target_ids(mass_dictionary, lambda row: True)
        reference_dictionary_targets = _target_ids(reference_dictionary, lambda row: True)
        full_pass_targets = _target_ids(case_dictionary, lambda row: _boolean(row.get("full_dictionary_quality_screen_pass")))
        mass_full_pass_targets = _target_ids(mass_dictionary, lambda row: _boolean(row.get("full_dictionary_quality_screen_pass")))
        reference_full_pass_targets = _target_ids(reference_dictionary, lambda row: _boolean(row.get("full_dictionary_quality_screen_pass")))
        two_mode_pass_targets = _target_ids(case_dictionary, lambda row: _boolean(row.get("two_mode_quality_screen_pass")))
        mass_two_mode_pass_targets = _target_ids(mass_dictionary, lambda row: _boolean(row.get("two_mode_quality_screen_pass")))
        multimode_targets = _target_ids(
            case_dictionary,
            lambda row: row.get("scientific_interpretation") == "multimode_explanation_for_this_a_posteriori_reference_description",
        )
        inadequate_targets = target_ids - full_pass_targets - two_mode_pass_targets - multimode_targets
        spatial_pair_targets = _target_ids(case_pairs, lambda row: _boolean(row.get("spatial_pair_interpretable")))
        pass_coverage_targets = _target_ids(reference_coverage, lambda row: _boolean(row.get("paired_coverage_and_union_cap_pass")))
        best_capture_by_target: dict[str, float] = {}
        for row in case_dictionary:
            target_id = str(row.get("target_id"))
            capture = _number(row.get("minimum_dictionary_capture"))
            if math.isfinite(capture):
                best_capture_by_target[target_id] = max(capture, best_capture_by_target.get(target_id, -math.inf))
        result.append({
            "case_id": case_id,
            "n": n,
            "degree_ccdf_alpha": case["degree_ccdf_alpha"],
            "graph_seed": case.get("graph_seed"),
            "graph_sha256_edges": case.get("graph_sha256_edges"),
            "upper_nonperron_rank_range": f"2-{min(51, n)}",
            "upper_nonperron_state_count": len(upper),
            "upper_nonperron_mean_n_times_ipr": _finite_mean([_number(row.get("n_times_ipr")) for row in upper]),
            "upper_nonperron_mean_max_coordinate_mass": _finite_mean([_number(row.get("maximum_coordinate_mass")) for row in upper]),
            "upper_nonperron_mean_support_size_80": _finite_mean([_number(row.get("support_size_80")) for row in upper]),
            "bulk_rank_range": f"{bulk_lo}-{bulk_hi}",
            "bulk_state_count": len(bulk),
            "bulk_mean_n_times_ipr": _finite_mean([_number(row.get("n_times_ipr")) for row in bulk]),
            "bulk_mean_max_coordinate_mass": _finite_mean([_number(row.get("maximum_coordinate_mass")) for row in bulk]),
            "lower_edge_rank_range": f"{max(1, n - 49)}-{n}",
            "lower_edge_mean_n_times_ipr": _finite_mean([_number(row.get("n_times_ipr")) for row in lower]),
            "nonperron_top50_filtered_mean_gap_ratio": _number(top50.get("mean_gap_ratio_filtered")),
            "nonperron_top50_valid_gap_ratio_count": int(_number(top50.get("valid_ratio_count_multiplicity_and_atom_excluded"), 0)),
            "selected_target_count": len(target_ids),
            "reference_coverage_target_count": len(pass_coverage_targets),
            "any_dictionary_target_count": len(any_dictionary_targets),
            "mass_informed_dictionary_target_count": len(mass_dictionary_targets),
            "reference_dictionary_target_count": len(reference_dictionary_targets),
            "spatial_pair_interpretable_target_count": len(spatial_pair_targets),
            "full_dictionary_quality_pass_target_count": len(full_pass_targets),
            "mass_informed_full_dictionary_quality_pass_target_count": len(mass_full_pass_targets),
            "reference_full_dictionary_quality_pass_target_count": len(reference_full_pass_targets),
            "two_mode_quality_pass_target_count": len(two_mode_pass_targets),
            "mass_informed_two_mode_quality_pass_target_count": len(mass_two_mode_pass_targets),
            "multimode_explanation_target_count": len(multimode_targets),
            "inadequate_or_unresolved_target_count": len(inadequate_targets),
            "median_best_reference_dictionary_capture": (
                float(np.median(list(best_capture_by_target.values()))) if best_capture_by_target else math.nan
            ),
        })
    return result


def representative_reference_rows(
    cases: Sequence[Mapping[str, object]],
    dictionary: Sequence[Mapping[str, object]],
    pairs: Sequence[Mapping[str, object]],
    targets: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Select deterministic best-available and unresolved examples without hiding failures."""
    output: list[dict[str, object]] = []
    for case in cases:
        case_id = str(case["case_id"])
        case_dictionary = _case_rows(dictionary, case_id)
        case_pairs = _case_rows(pairs, case_id)
        case_targets = _case_rows(targets, case_id)
        best_by_target: dict[str, Mapping[str, object]] = {}
        for row in case_dictionary:
            target_id = str(row.get("target_id"))
            score = (
                int(_boolean(row.get("full_dictionary_quality_screen_pass"))),
                int(_boolean(row.get("two_mode_quality_screen_pass"))),
                _number(row.get("minimum_dictionary_capture"), -math.inf),
                -_number(row.get("relative_gap_error"), math.inf),
                int(_number(row.get("requested_modes_per_region"), 0)),
            )
            current = best_by_target.get(target_id)
            if current is None:
                best_by_target[target_id] = row
                continue
            current_score = (
                int(_boolean(current.get("full_dictionary_quality_screen_pass"))),
                int(_boolean(current.get("two_mode_quality_screen_pass"))),
                _number(current.get("minimum_dictionary_capture"), -math.inf),
                -_number(current.get("relative_gap_error"), math.inf),
                int(_number(current.get("requested_modes_per_region"), 0)),
            )
            if score > current_score:
                best_by_target[target_id] = row

        def emit(role: str, target_id: str, row: Mapping[str, object] | None) -> None:
            target = next((item for item in case_targets if str(item.get("target_id")) == target_id), {})
            pair = None if row is None else next((
                item for item in case_pairs
                if str(item.get("dictionary_id")) == str(row.get("dictionary_id"))
                and int(_number(item.get("requested_modes_per_region"), -1)) == int(_number(row.get("requested_modes_per_region"), -1))
            ), None)
            output.append({
                "case_id": case_id,
                "n": case["n"],
                "degree_ccdf_alpha": case["degree_ccdf_alpha"],
                "example_role": role,
                "target_id": target_id,
                "pair_start_decreasing_rank": target.get("pair_start_decreasing_rank"),
                "observed_adjacent_gap": target.get("observed_adjacent_gap"),
                "descriptive_concentration_relationship": target.get("descriptive_concentration_relationship"),
                "dominant_vertex_distance": target.get("dominant_vertex_distance"),
                "whole_vector_mass_overlap": target.get("whole_vector_mass_overlap"),
                "reference_description_available": row is not None,
                "construction_family": None if row is None else row.get("construction_family"),
                "reference_setting_in_equivalents": False if row is None else _boolean(row.get("reference_setting_in_equivalents")),
                "selection_method": None if row is None else row.get("selection_method"),
                "requested_modes_per_region": None if row is None else row.get("requested_modes_per_region"),
                "minimum_dictionary_capture": None if row is None else row.get("minimum_dictionary_capture"),
                "predicted_adjacent_gap": None if row is None else row.get("predicted_adjacent_gap"),
                "relative_gap_error": None if row is None else row.get("relative_gap_error"),
                "maximum_region_mass_absolute_error": None if row is None else row.get("maximum_region_mass_absolute_error"),
                "pair_only_minimum_capture": None if pair is None else pair.get("minimum_pair_only_capture"),
                "absolute_direct_coupling": None if pair is None else pair.get("absolute_direct_coupling"),
                "reference_detuning": None if pair is None else pair.get("reference_detuning"),
                "direct_coupling_over_reference_detuning": None if pair is None else pair.get("direct_coupling_over_reference_detuning"),
                "direct_coupling_over_observed_gap": None if pair is None else pair.get("direct_coupling_over_observed_gap"),
                "two_mode_predicted_gap": None if pair is None else pair.get("two_mode_predicted_gap"),
                "pair_relative_gap_error": None if pair is None else pair.get("pair_relative_gap_error"),
                "full_dictionary_quality_screen_pass": False if row is None else _boolean(row.get("full_dictionary_quality_screen_pass")),
                "two_mode_quality_screen_pass": False if row is None else _boolean(row.get("two_mode_quality_screen_pass")),
                "scientific_interpretation": "no_reference_dictionary_at_reference_setting" if row is None else row.get("scientific_interpretation"),
            })

        ranked = sorted(
            best_by_target.items(),
            key=lambda item: (
                int(_boolean(item[1].get("full_dictionary_quality_screen_pass"))),
                int(_boolean(item[1].get("two_mode_quality_screen_pass"))),
                _number(item[1].get("minimum_dictionary_capture"), -math.inf),
                -_number(item[1].get("relative_gap_error"), math.inf),
                item[0],
            ),
            reverse=True,
        )
        if ranked:
            emit("best_available_reference_description", ranked[0][0], ranked[0][1])
        elif case_targets:
            emit("best_available_reference_description", str(case_targets[0].get("target_id")), None)
        unresolved = sorted(
            (
                str(row.get("target_id"))
                for row in case_targets
                if str(row.get("target_id")) not in best_by_target
                or not _boolean(best_by_target[str(row.get("target_id"))].get("full_dictionary_quality_screen_pass"))
            ),
        )
        if unresolved:
            emit("representative_unresolved_target", unresolved[0], best_by_target.get(unresolved[0]))
    return output


def _plot_localization(localization: Sequence[Mapping[str, object]], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, n in zip(axes, (500, 10000)):
        subset = [row for row in localization if int(float(str(row["n"]))) == n]
        for alpha in sorted({_number(row["degree_ccdf_alpha"]) for row in subset}):
            rows = [row for row in subset if math.isclose(_number(row["degree_ccdf_alpha"]), alpha)]
            rows.sort(key=lambda row: _number(row["energy"]))
            axis.plot(
                [_number(row["energy"]) for row in rows],
                [_number(row["n_times_ipr"]) for row in rows],
                marker=".",
                markersize=1.2 if n == 10000 else 2.5,
                linewidth=0.45,
                alpha=0.75,
                label=f"alpha={alpha:g}",
            )
        axis.set_yscale("log")
        axis.set_xlabel("raw adjacency eigenvalue")
        axis.set_ylabel("N x IPR")
        axis.set_title(f"Affine PA, N={n}, m=10; Perron retained")
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=7, ncol=2)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_gap_ratio_windows(windows: Sequence[Mapping[str, object]], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, n in zip(axes, (500, 10000)):
        subset = [
            row for row in windows
            if int(float(str(row["n"]))) == n
            and row.get("window_family") == "global_quantile_rank_band"
        ]
        for alpha in sorted({_number(row["degree_ccdf_alpha"]) for row in subset}):
            rows = [row for row in subset if math.isclose(_number(row["degree_ccdf_alpha"]), alpha)]
            rows.sort(key=lambda row: _number(row["quantile_left"]))
            x = [0.5 * (_number(row["quantile_left"]) + _number(row["quantile_right"])) for row in rows]
            y = [_number(row["mean_gap_ratio_filtered"]) for row in rows]
            axis.plot(x, y, marker="o", markersize=2.2, linewidth=0.75, label=f"alpha={alpha:g}")
        axis.axhline(2.0 * math.log(2.0) - 1.0, color="0.4", linestyle=":", linewidth=0.8, label="Poisson mean")
        axis.axhline(0.5307, color="0.2", linestyle="--", linewidth=0.8, label="GOE reference")
        axis.set_xlabel("decreasing-rank quantile midpoint")
        axis.set_ylabel("mean raw adjacent-gap ratio")
        axis.set_title(f"N={n}; multiplicities/zero atom excluded")
        axis.set_ylim(0.0, 0.75)
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=6.5, ncol=2)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_gap_histograms(gaps: Sequence[Mapping[str, object]], path: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    selected_alphas = (1.05, 2.0, 3.0)
    for row_index, n in enumerate((500, 10000)):
        for column, alpha in enumerate(selected_alphas):
            axis = axes[row_index, column]
            values = np.asarray([
                _number(row["window_mean_rescaled_gap"])
                for row in gaps
                if int(float(str(row["n"]))) == n
                and math.isclose(_number(row["degree_ccdf_alpha"]), alpha)
                and row.get("window_family") == "cumulative_nonperron"
                and row.get("window_id") == "nonperron_top_100"
                and _boolean(row.get("numerically_resolved"))
                and not _boolean(row.get("touches_zero_atom"))
            ])
            values = values[np.isfinite(values)]
            if len(values):
                axis.hist(values, bins=min(30, max(8, int(math.sqrt(len(values))))), density=True, alpha=0.55, color="#4C78A8", label=f"data n={len(values)}")
            x = np.linspace(0.0, 3.5, 300)
            axis.plot(x, np.exp(-x), color="#F58518", linewidth=1.0, label="Poisson")
            axis.plot(x, 0.5 * math.pi * x * np.exp(-math.pi * x * x / 4.0), color="#54A24B", linewidth=1.0, label="GOE Wigner")
            axis.set_xlim(0.0, 3.5)
            axis.set_title(f"N={n}, alpha={alpha:g}, non-Perron top 100")
            axis.set_xlabel("gap / window mean")
            axis.set_ylabel("density")
            axis.grid(alpha=0.15)
    axes[0, 0].legend(fontsize=7)
    figure.suptitle("Single-window mean rescaling (not ensemble unfolding)")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_cumulative_mass(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, n in zip(axes, (500, 10000)):
        subset = [
            row for row in rows
            if int(float(str(row["n"]))) == n
            and 2 <= int(float(str(row["decreasing_algebraic_rank"]))) <= 11
        ]
        grouped: dict[tuple[float, int], list[Mapping[str, object]]] = {}
        for row in subset:
            grouped.setdefault((_number(row["degree_ccdf_alpha"]), int(float(str(row["requested_coordinate_count"])))), []).append(row)
        for alpha in sorted({key[0] for key in grouped}):
            counts = sorted({key[1] for key in grouped if math.isclose(key[0], alpha)})
            means = [np.mean([_number(item["cumulative_mass"]) for item in grouped[(alpha, count)]]) for count in counts]
            axis.plot(counts, means, marker="o", linewidth=0.8, label=f"alpha={alpha:g}")
        axis.set_xscale("log")
        axis.set_ylim(0.0, 1.02)
        axis.set_xlabel("retained coordinate count")
        axis.set_ylabel("mean cumulative mass, ranks 2-11")
        axis.set_title(f"N={n}; non-Perron upper edge")
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=7, ncol=2)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_coverage(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    for row_index, n in enumerate((500, 10000)):
        alphas = sorted({_number(row["degree_ccdf_alpha"]) for row in rows if int(float(str(row["n"]))) == n})
        for column, x in enumerate((0.7, 0.8, 0.9, 0.95)):
            axis = axes[row_index, column]
            matrix = np.full((len(alphas), 4), np.nan)
            for alpha_index, alpha in enumerate(alphas):
                for y_index, y in enumerate((0.05, 0.1, 0.2, 0.3)):
                    selected = [
                        row for row in rows
                        if int(float(str(row["n"]))) == n
                        and math.isclose(_number(row["degree_ccdf_alpha"]), alpha)
                        and row.get("construction_family") == "mass_informed_greedy"
                        and row.get("grouping_variant") == "overlap_component_grouped"
                        and int(float(str(row["radius"]))) == 2
                        and math.isclose(_number(row["x_target_mass"]), x)
                        and math.isclose(_number(row["y_union_fraction_cap"]), y)
                    ]
                    if selected:
                        matrix[alpha_index, y_index] = np.mean([_boolean(row["paired_coverage_and_union_cap_pass"]) for row in selected])
            image = axis.imshow(matrix, vmin=0.0, vmax=1.0, aspect="auto", origin="lower", cmap="viridis")
            axis.set_xticks(range(4))
            axis.set_xticklabels([".05", ".10", ".20", ".30"])
            axis.set_yticks(range(len(alphas)))
            axis.set_yticklabels([f"{alpha:g}" for alpha in alphas])
            axis.set_xlabel("y union cap")
            axis.set_ylabel("nominal CCDF alpha")
            axis.set_title(f"N={n}, x={x:g}, r=2")
    figure.colorbar(image, ax=axes, label="fraction of selected target pairs passing coverage/locality")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_capture_and_gap(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    descriptions = list(rows)
    for n in (500, 10000):
        subset = [row for row in descriptions if int(float(str(row["n"]))) == n]
        counts = sorted({int(float(str(row["requested_modes_per_region"]))) for row in subset})
        medians = []
        for count in counts:
            values = [_number(row["minimum_dictionary_capture"]) for row in subset if int(float(str(row["requested_modes_per_region"]))) == count]
            values = [value for value in values if math.isfinite(value)]
            medians.append(float(np.median(values)) if values else math.nan)
        axes[0].plot(counts, medians, marker="o", label=f"N={n}")
    axes[0].axhline(0.9, color="0.3", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("requested local modes per region")
    axes[0].set_ylabel("median minimum full-dictionary capture")
    axes[0].set_title("All predeclared region and mode sensitivities")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].grid(alpha=0.2)
    axes[0].legend()
    qualified = [
        row for row in descriptions
        if _boolean(row.get("full_dictionary_quality_screen_pass"))
        and math.isfinite(_number(row.get("observed_adjacent_gap")))
        and math.isfinite(_number(row.get("predicted_adjacent_gap")))
    ]
    for n, marker in ((500, "o"), (10000, "^")):
        subset = [row for row in qualified if int(float(str(row["n"]))) == n]
        axes[1].scatter(
            [_number(row["observed_adjacent_gap"]) for row in subset],
            [_number(row["predicted_adjacent_gap"]) for row in subset],
            s=18,
            marker=marker,
            alpha=0.7,
            label=f"N={n}, count={len(subset)}",
        )
    limits = axes[1].get_xlim()
    maximum = max(limits[1], axes[1].get_ylim()[1], 1e-12)
    axes[1].plot([0.0, maximum], [0.0, maximum], color="0.3", linestyle="--", linewidth=0.8)
    axes[1].set_xlim(0.0, maximum)
    axes[1].set_ylim(0.0, maximum)
    axes[1].set_xlabel("observed adjacent gap")
    axes[1].set_ylabel("reduced-model gap")
    axes[1].set_title("Only rows passing every full-dictionary screen")
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=7)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_region_mass_validation(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for axis, n in zip(axes, (500, 10000)):
        exact_values: list[float] = []
        predicted_values: list[float] = []
        state_rows = 0
        for row in rows:
            if int(_number(row.get("n"), -1)) != n:
                continue
            if not _boolean(row.get("full_dictionary_quality_screen_pass")):
                continue
            exact = _json_value(row.get("exact_region_masses"))
            predicted = _json_value(row.get("predicted_region_masses_including_interference"))
            if not isinstance(exact, list) or not isinstance(predicted, list) or len(exact) != len(predicted):
                continue
            added = False
            for exact_state, predicted_state in zip(exact, predicted):
                if not isinstance(exact_state, list) or not isinstance(predicted_state, list):
                    continue
                if len(exact_state) != len(predicted_state):
                    continue
                for exact_mass, predicted_mass in zip(exact_state, predicted_state):
                    exact_number = _number(exact_mass)
                    predicted_number = _number(predicted_mass)
                    if math.isfinite(exact_number) and math.isfinite(predicted_number):
                        exact_values.append(exact_number)
                        predicted_values.append(predicted_number)
                        added = True
            state_rows += int(added)
        if exact_values:
            axis.scatter(exact_values, predicted_values, s=9, alpha=0.45)
            maximum = max(max(exact_values), max(predicted_values), 1e-12)
            axis.plot([0.0, maximum], [0.0, maximum], color="0.25", linestyle="--", linewidth=0.8)
            axis.set_xlim(0.0, maximum)
            axis.set_ylim(0.0, maximum)
        else:
            axis.text(0.5, 0.5, "No full-dictionary rows passed all screens", ha="center", va="center", transform=axis.transAxes)
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("exact mass in declared region")
        axis.set_ylabel("reconstructed mass including interference")
        axis.set_title(f"N={n}; qualified descriptions={state_rows}, entries={len(exact_values)}")
        axis.grid(alpha=0.2)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_localization_interaction_summary(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for axis, n in zip(axes, (500, 10000)):
        subset = [row for row in rows if int(row["n"]) == n]
        for row in subset:
            total = max(int(row["selected_target_count"]), 1)
            x = _number(row["upper_nonperron_mean_n_times_ipr"])
            y = int(row["full_dictionary_quality_pass_target_count"]) / total
            alpha = _number(row["degree_ccdf_alpha"])
            axis.scatter([x], [y], s=42)
            axis.annotate(f"{alpha:g}", (x, y), xytext=(3, 3), textcoords="offset points", fontsize=7)
        axis.set_xscale("log")
        axis.set_ylim(-0.02, 1.02)
        axis.set_xlabel("mean N x IPR, non-Perron ranks 2-51")
        axis.set_ylabel("fraction of fixed targets with any qualified full-dictionary row")
        axis.set_title(f"N={n}; descriptive case linkage, one graph per alpha")
        axis.grid(alpha=0.2)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _markdown_float(value: object, digits: int = 4) -> str:
    number = _number(value)
    return f"{number:.{digits}g}" if math.isfinite(number) else "NA"


def _markdown_case_table(rows: Sequence[Mapping[str, object]]) -> list[str]:
    lines = [
        "| N | alpha | upper mean N*IPR | bulk mean N*IPR | lower mean N*IPR | top-50 mean r | targets | dictionaries any/ref | full-dict pass | two-mode pass |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {n} | {alpha} | {upper} | {bulk} | {lower} | {ratio} ({ratio_n}) | {targets} | {dictionaries} | {full_pass} | {two_mode} |".format(
                n=row["n"],
                alpha=_markdown_float(row["degree_ccdf_alpha"], 3),
                upper=_markdown_float(row["upper_nonperron_mean_n_times_ipr"]),
                bulk=_markdown_float(row["bulk_mean_n_times_ipr"]),
                lower=_markdown_float(row["lower_edge_mean_n_times_ipr"]),
                ratio=_markdown_float(row["nonperron_top50_filtered_mean_gap_ratio"]),
                ratio_n=row["nonperron_top50_valid_gap_ratio_count"],
                targets=row["selected_target_count"],
                dictionaries=f"{row['any_dictionary_target_count']}/{row['reference_dictionary_target_count']}",
                full_pass=row["full_dictionary_quality_pass_target_count"],
                two_mode=row["two_mode_quality_pass_target_count"],
            )
        )
    return lines


def _markdown_representative_table(rows: Sequence[Mapping[str, object]]) -> list[str]:
    lines = [
        "| N | alpha | role | construction | target | full capture | full gap err. | abs(t)/detuning | abs(t)/observed gap | full pass | two-mode pass | interpretation |",
        "|---:|---:|:---|:---|:---|---:|---:|---:|---:|:---:|:---:|:---|",
    ]
    for row in rows:
        lines.append(
            "| {n} | {alpha} | {role} | {construction} | {target} | {capture} | {gap_error} | {t_detuning} | {t_gap} | {full_pass} | {two_mode} | {interpretation} |".format(
                n=row["n"],
                alpha=_markdown_float(row["degree_ccdf_alpha"], 3),
                role=row["example_role"],
                construction=(
                    f"{row.get('construction_family')} (reference)"
                    if _boolean(row.get("reference_setting_in_equivalents"))
                    else row.get("construction_family")
                ),
                target=row["target_id"],
                capture=_markdown_float(row.get("minimum_dictionary_capture")),
                gap_error=_markdown_float(row.get("relative_gap_error")),
                t_detuning=_markdown_float(row.get("direct_coupling_over_reference_detuning")),
                t_gap=_markdown_float(row.get("direct_coupling_over_observed_gap")),
                full_pass="yes" if _boolean(row.get("full_dictionary_quality_screen_pass")) else "no",
                two_mode="yes" if _boolean(row.get("two_mode_quality_screen_pass")) else "no",
                interpretation=row.get("scientific_interpretation"),
            )
        )
    return lines


def _summary_page(pdf: PdfPages, lines: Sequence[str], title: str) -> None:
    figure = plt.figure(figsize=(8.5, 11))
    figure.text(0.07, 0.96, title, fontsize=15, weight="bold", va="top")
    figure.text(0.07, 0.92, "\n".join(lines), fontsize=9, family="monospace", va="top", wrap=True)
    pdf.savefig(figure)
    plt.close(figure)


def _case_summary_pdf_page(pdf: PdfPages, rows: Sequence[Mapping[str, object]]) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(11, 8.5), constrained_layout=True)
    columns = ("alpha", "upper N*IPR", "bulk N*IPR", "top-50 mean r", "dict any/ref/targets", "full pass", "two-mode")
    for axis, n in zip(axes, (500, 10000)):
        subset = [row for row in rows if int(row["n"]) == n]
        cells = [
            [
                _markdown_float(row["degree_ccdf_alpha"], 3),
                _markdown_float(row["upper_nonperron_mean_n_times_ipr"]),
                _markdown_float(row["bulk_mean_n_times_ipr"]),
                _markdown_float(row["nonperron_top50_filtered_mean_gap_ratio"]),
                f"{row['any_dictionary_target_count']} / {row['reference_dictionary_target_count']} / {row['selected_target_count']}",
                str(row["full_dictionary_quality_pass_target_count"]),
                str(row["two_mode_quality_pass_target_count"]),
            ]
            for row in subset
        ]
        axis.axis("off")
        table = axis.table(cellText=cells, colLabels=columns, cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.25)
        axis.set_title(f"N={n}: fixed-graph numerical summary", fontsize=12)
    figure.suptitle("Upper non-Perron ranks 2-51; one graph per (N, alpha)", fontsize=13)
    pdf.savefig(figure)
    plt.close(figure)


def run_aggregate(
    *,
    config: Mapping[str, object],
    output_root: Path,
    report_dir: Path,
    freeze_manifest: Path,
) -> Path:
    output_root = Path(output_root)
    report_dir = Path(report_dir)
    figures = report_dir / "figures"
    tables = report_dir / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    cases = discover_cases(output_root, config)
    write_csv(tables / "case_status.csv", cases)
    localization = _collect_case_table(cases, "tables/full_spectrum_localization.csv.gz")
    windows = _collect_case_table(cases, "tables/energy_and_rank_window_summary.csv.gz")
    gaps = _collect_case_table(cases, "tables/window_gap_observations.csv.gz")
    cumulative = _collect_case_table(cases, "tables/cumulative_rows.csv.gz")
    if not cumulative:
        cumulative = _collect_case_table(cases, "tables/cumulative_rows.csv")
    coverage = _collect_case_table(cases, "part2/tables/all_targets_region_sensitivity.csv.gz")
    dictionary = _collect_case_table(cases, "part2/tables/all_targets_dictionary_mode_sensitivity.csv.gz")
    pairs = _collect_case_table(cases, "part2/tables/all_targets_spatial_pair_coupling.csv.gz")
    schur = _collect_case_table(cases, "part2/tables/schur_effective_coupling.csv.gz")
    targets = _collect_case_table(cases, "part2/tables/target_summary.csv")
    write_csv(tables / "all_cases_window_summary.csv.gz", windows)
    write_csv(tables / "all_cases_target_summary.csv.gz", targets)
    write_csv(tables / "all_cases_dictionary_mode_sensitivity.csv.gz", dictionary)
    write_csv(tables / "all_cases_spatial_pair_coupling.csv.gz", pairs)
    write_csv(tables / "all_cases_schur_effective_coupling.csv.gz", schur)
    case_summary = summarize_scientific_cases(cases, localization, windows, coverage, dictionary, pairs, targets)
    representative_rows = representative_reference_rows(cases, dictionary, pairs, targets)
    write_csv(tables / "scientific_case_summary.csv", case_summary)
    write_csv(tables / "representative_reference_descriptions.csv", representative_rows)
    localization_path = figures / "localization_energy_nipr.png"
    gap_ratio_path = figures / "gap_ratio_windows.png"
    histogram_path = figures / "gap_histograms_mean_rescaled.png"
    cumulative_path = figures / "first_ten_nonperron_cumulative_mass.png"
    coverage_path = figures / "region_coverage_xy.png"
    prediction_path = figures / "capture_and_gap_prediction.png"
    region_mass_path = figures / "validated_region_mass_prediction.png"
    linkage_path = figures / "localization_interaction_case_linkage.png"
    _plot_localization(localization, localization_path)
    _plot_gap_ratio_windows(windows, gap_ratio_path)
    _plot_gap_histograms(gaps, histogram_path)
    if cumulative:
        _plot_cumulative_mass(cumulative, cumulative_path)
    _plot_coverage(coverage, coverage_path)
    _plot_capture_and_gap(dictionary, prediction_path)
    _plot_region_mass_validation(dictionary, region_mass_path)
    _plot_localization_interaction_summary(case_summary, linkage_path)
    complete_part1 = sum(row["part1_status"] == "COMPLETE" for row in cases)
    complete_part2 = sum(row["part2_status"] == "COMPLETE" for row in cases)
    dictionary_pass = sum(_boolean(row.get("full_dictionary_quality_screen_pass")) for row in dictionary)
    two_mode_pass = sum(_boolean(row.get("two_mode_quality_screen_pass")) for row in dictionary)
    multimode = sum(row.get("scientific_interpretation") == "multimode_explanation_for_this_a_posteriori_reference_description" for row in dictionary)
    inadequate = sum(row.get("scientific_interpretation") == "reference_model_inadequate_or_unresolved" for row in dictionary)
    schur_complete = sum(row.get("status") == "COMPLETE" for row in schur)
    total_selected_targets = sum(int(row["selected_target_count"]) for row in case_summary)
    any_dictionary_targets = sum(int(row["any_dictionary_target_count"]) for row in case_summary)
    mass_dictionary_targets = sum(int(row["mass_informed_dictionary_target_count"]) for row in case_summary)
    reference_dictionary_targets = sum(int(row["reference_dictionary_target_count"]) for row in case_summary)
    full_pass_targets = sum(int(row["full_dictionary_quality_pass_target_count"]) for row in case_summary)
    mass_full_pass_targets = sum(int(row["mass_informed_full_dictionary_quality_pass_target_count"]) for row in case_summary)
    reference_full_pass_targets = sum(int(row["reference_full_dictionary_quality_pass_target_count"]) for row in case_summary)
    two_mode_pass_targets = sum(int(row["two_mode_quality_pass_target_count"]) for row in case_summary)
    mass_two_mode_pass_targets = sum(int(row["mass_informed_two_mode_quality_pass_target_count"]) for row in case_summary)
    multimode_targets = sum(int(row["multimode_explanation_target_count"]) for row in case_summary)
    unresolved_targets = sum(int(row["inadequate_or_unresolved_target_count"]) for row in case_summary)
    target_level_conclusion = (
        f"Across all predeclared region and mode sensitivity descriptions, {any_dictionary_targets}/{total_selected_targets} selected adjacent pairs admitted at least one local dictionary; "
        f"{mass_dictionary_targets} did so under a mass-informed construction, and {reference_dictionary_targets} did so at the strict mass-informed reference setting (x,y,r)=(0.80,0.20,2) with overlap-component grouping. "
        f"At least one full-dictionary row passed every provisional capture, overlap, gap, and regional-mass screen for {full_pass_targets} targets "
        f"({mass_full_pass_targets} mass-informed; {reference_full_pass_targets} strict-reference); the spatial two-mode screen passed for {two_mode_pass_targets} targets ({mass_two_mode_pass_targets} mass-informed). "
        f"Multimode explanations occurred for {multimode_targets} targets, while {unresolved_targets} targets remained inadequate or unresolved under this reference construction. "
        "These are a-posteriori explanation counts, not independent predictions or phase estimates."
    )
    report_lines = [
        "# Linear affine-PA localization and local-mode interaction report",
        "",
        f"Generated: {utc_now()}",
        "",
        "## Scope and model",
        "",
        "This run uses one clean affine linear preferential-attachment graph for each configured (N, alpha) pair. Attachment weights are degree plus delta, with delta = m(alpha-2), m=10, a complete graph on m+1 seed vertices, and sequential weighted sampling without replacement. The diagonalized operator is the unscaled simple undirected adjacency in saved vertex order 0..N-1.",
        "",
        f"The configured alpha grid {config['alpha_grid']} is the implementation default supplied by the task specification, not an empirically fitted grid. N values are {config['sizes']}. Each condition has one realization; cross-alpha patterns are exploratory case comparisons rather than ensemble phase-transition estimates.",
        "",
        "## Execution status",
        "",
        f"- Part 1 complete cases: {complete_part1}/{len(cases)}.",
        f"- Part 2 complete cases: {complete_part2}/{len(cases)}.",
        f"- Fixed Part 2 targets: {total_selected_targets}; any sensitivity setting supplied a dictionary for {any_dictionary_targets} targets ({mass_dictionary_targets} mass-informed; {reference_dictionary_targets} strict-reference).",
        f"- Unique targets with any qualified full-dictionary row: {full_pass_targets} ({mass_full_pass_targets} mass-informed; {reference_full_pass_targets} strict-reference); with any qualified spatial two-mode row: {two_mode_pass_targets} ({mass_two_mode_pass_targets} mass-informed).",
        f"- Full-dictionary diagnostic rows passing all provisional quality screens: {dictionary_pass}/{len(dictionary)}.",
        f"- Spatial two-mode rows passing all provisional quality screens: {two_mode_pass}/{len(dictionary)}.",
        f"- Rows classified as multimode explanations: {multimode}.",
        f"- Rows with an inadequate or unresolved reference description: {inadequate}.",
        f"- Completed Schur regularization rows: {schur_complete}/{len(schur)}.",
        "",
        "Completion of a numerical row is not scientific acceptance. Every failed coverage, rank-deficient pair, solver failure, and quality-screen failure is retained in the detailed tables.",
        "",
        "## Numerical case summary",
        "",
        "Upper-edge entries use non-Perron decreasing ranks 2-51; bulk entries use the central 40%-60% rank band; the lower-edge entries use the last 50 algebraic ranks. The reported r is the raw adjacent-gap ratio for the non-Perron top-50 window after configured multiplicity and zero-atom exclusions. Counts in parentheses are valid ratios. Dictionary counts are unique targets with a dictionary under any predeclared sensitivity setting / the strict mass-informed reference setting.",
        "",
        *_markdown_case_table(case_summary),
        "",
        "## Local-mode interaction assessment",
        "",
        target_level_conclusion,
        "",
        "A full-dictionary pass supports only the declared a-posteriori local reference span. A two-mode pass additionally requires a spatially interpretable pair and its separate screens. Direct coupling, observed gap, and reference detuning remain distinct quantities. A target without a passing row is preserved as an unsuccessful reference approximation, not evidence that localization or interaction is absent.",
        "",
        "### Deterministic representative descriptions",
        "",
        *_markdown_representative_table(representative_rows),
        "",
        "## Literature context",
        "",
        "Alt, Ducatez, and Knowles (2021), `https://doi.org/10.1007/s00220-021-04167-y`, motivates examining eigenvector mass on small graph balls and interactions among local concentration centers. That paper treats a critical Erdos-Renyi ensemble and a differently normalized operator. Its theorems, pruning construction, and spectral boundaries are not transferred to the present unscaled affine-PA adjacency matrices.",
        "",
        "## Interpretation guardrails",
        "",
        "Part 2 is explicitly a-posteriori whenever exact target eigenvectors determine centers, regions, or mode selection. Full-dictionary capture and reconstruction error are representation diagnostics, not tunneling amplitudes. Direct coupling, the mediated Schur correction, and the total effective coupling are stored under distinct names. The finite-imaginary-part Schur operator is not inserted into the real two-state splitting formula. Larger edge IPR, low mean gap ratio, or multi-center mass sharing alone is not labelled a localized phase, a mobility edge, or graph-wide delocalization.",
        "",
        "## Figures",
        "",
        "1. `figures/localization_energy_nipr.png`: full-spectrum raw energy versus N x IPR, with Perron retained and labelled by convention.",
        "2. `figures/gap_ratio_windows.png`: multiplicity/zero-atom-aware raw adjacent-gap-ratio means in nonoverlapping rank-quantile windows.",
        "3. `figures/gap_histograms_mean_rescaled.png`: raw gaps divided by their own window mean; this is not ensemble unfolding.",
        "4. `figures/first_ten_nonperron_cumulative_mass.png`: mass captured by the requested coordinate counts for ranks 2-11.",
        "5. `figures/region_coverage_xy.png`: fraction of examined targets meeting both per-vector mass coverage and the union-volume cap.",
        "6. `figures/capture_and_gap_prediction.png`: nested full-dictionary capture and gap comparison across all predeclared sensitivity descriptions, restricted to rows passing every provisional full-dictionary screen.",
        "7. `figures/validated_region_mass_prediction.png`: exact versus reconstructed masses in declared regions, including interference, for qualified rows across all predeclared sensitivity descriptions.",
        "8. `figures/localization_interaction_case_linkage.png`: descriptive per-graph linkage between upper-edge N x IPR and the fraction of fixed targets with a qualified full-dictionary explanation; this is not a correlation or phase-transition estimate.",
        "",
        "## Machine-readable evidence",
        "",
        "The report tables contain graph identities and seeds; full eigenpair localization and residuals; raw gaps and ratios; cumulative/disjoint windows; detailed coordinate masses; induced/full/cross distances; complete target coverage; region selections/rejections; raw local-mode cache identities; basis transforms and Gram diagnostics; full-dictionary and pair-only captures; reduced-state correspondence; direct coupling; Schur corrections; and solver/resource statuses.",
        "",
        "See `tables/case_status.csv` for case completion and the per-case directories for complete non-aggregated evidence.",
    ]
    report_path = report_dir / "LINEAR_PA_TWO_PART_REPORT.md"
    report_path.write_text("\n".join(report_lines) + "\n")
    pdf_path = report_dir / "LINEAR_PA_TWO_PART_REPORT.pdf"
    with PdfPages(pdf_path) as pdf:
        _summary_page(pdf, [
            f"Generated: {utc_now()}",
            "Clean affine linear PA; unscaled simple adjacency; m=10; one graph per (N, alpha).",
            f"Part 1 complete: {complete_part1}/{len(cases)} cases.",
            f"Part 2 complete: {complete_part2}/{len(cases)} cases.",
            f"Fixed adjacent-pair targets: {total_selected_targets}.",
            f"Any dictionary / mass-informed / strict-reference: {any_dictionary_targets} / {mass_dictionary_targets} / {reference_dictionary_targets} targets.",
            f"Qualified full dictionary all / mass / strict-reference: {full_pass_targets} / {mass_full_pass_targets} / {reference_full_pass_targets} targets.",
            f"Qualified spatial two-mode all / mass-informed: {two_mode_pass_targets} / {mass_two_mode_pass_targets} targets.",
            f"Multimode explanation flags: {multimode_targets} targets.",
            f"Inadequate or unresolved under the reference construction: {unresolved_targets} targets.",
            "All interaction analyses are a-posteriori and target-informed.",
            "Counts are descriptive case evidence, not ensemble or phase-transition estimates.",
            "See the Markdown report and machine-readable tables for complete values and failed cases.",
        ], "Linear affine-PA experiment")
        _case_summary_pdf_page(pdf, case_summary)
        for figure_path in (localization_path, gap_ratio_path, histogram_path, cumulative_path, coverage_path, prediction_path, region_mass_path, linkage_path):
            if not figure_path.exists():
                continue
            image = plt.imread(figure_path)
            figure = plt.figure(figsize=(11, 8.5))
            axis = figure.add_axes([0.03, 0.03, 0.94, 0.94])
            axis.imshow(image)
            axis.axis("off")
            pdf.savefig(figure)
            plt.close(figure)
    freeze = read_json(freeze_manifest)
    scientific_source_configuration_ids = sorted({
        str(row[key])
        for row in cases
        for key in ("part1_source_configuration_id", "part2_source_configuration_id")
        if row.get(key)
    })
    manifest = {
        "status": "COMPLETE" if complete_part1 == len(cases) and complete_part2 == len(cases) else "INCOMPLETE_CASES",
        "generated_at_utc": utc_now(),
        "reporting_source_configuration_id": freeze.get("source_configuration_id"),
        "scientific_source_configuration_ids": scientific_source_configuration_ids,
        "case_count": len(cases),
        "part1_complete_count": complete_part1,
        "part2_complete_count": complete_part2,
        "report_markdown": str(report_path),
        "report_markdown_sha256": sha256_file(report_path),
        "report_pdf": str(pdf_path),
        "report_pdf_sha256": sha256_file(pdf_path),
        "figure_paths": [str(path) for path in (localization_path, gap_ratio_path, histogram_path, cumulative_path, coverage_path, prediction_path, region_mass_path, linkage_path) if path.exists()],
    }
    write_json(report_dir / "run_manifest.json", manifest)
    inventory = []
    for path in sorted(report_dir.rglob("*")):
        if path.is_file() and path.name != "EVIDENCE_INVENTORY.csv":
            inventory.append({"relative_path": str(path.relative_to(report_dir)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_csv(report_dir / "EVIDENCE_INVENTORY.csv", inventory)
    return report_dir
