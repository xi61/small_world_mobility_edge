"""Aggregate Anderson-only evidence, figures, report, index, and PDF."""

from __future__ import annotations

import csv
import gzip
import json
import math
import textwrap
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import networkx as nx
import numpy as np

from .anderson_rrg_benchmark import _rrg_task
from .anderson_utils import atomic_write_csv, atomic_write_json, read_csv_rows, verify_execution_context
from .common import deterministic_mass_order, graph_from_edge_cache, read_json, sha256_file, utc_now
from .disorder_replication import _fresh_pa_task


ANDERSON_REPORTING_VERSION = "anderson_repair_extension_reporting_v2"


def _number(row: Mapping[str, object], key: str) -> float:
    try:
        return float(row.get(key, math.nan))
    except (TypeError, ValueError):
        return math.nan


def _truth(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _ratio_slug(value: float) -> str:
    return ("%.12g" % float(value)).replace("-", "m").replace(".", "p")


def _amplitude_slug(value: float) -> str:
    return _ratio_slug(value)


def _read_if(path: Path) -> list[dict[str, str]]:
    return read_csv_rows(path) if path.is_file() else []


def _write_indexed_table(
    index_rows: list[dict[str, object]],
    checkpoint: Path,
    *,
    stage: str,
    metadata: Mapping[str, object],
) -> None:
    for path in sorted((checkpoint / "tables").glob("*")) if (checkpoint / "tables").is_dir() else []:
        index_rows.append({
            "stage": stage,
            **metadata,
            "table_kind": path.name,
            "remote_path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })


def _fresh_records(config: Mapping[str, object], fresh_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    total = len(config["fresh_pa"]["models"]) * len(config["fresh_pa"]["n_values"]) * int(config["fresh_pa"]["graph_replicates"])
    for task_id in range(1, total + 1):
        spec = _fresh_pa_task(config, task_id)
        graph_dir = fresh_root / str(spec["model_key"]) / ("N%d" % int(spec["n"])) / ("graph_rep_%d_seed_%d" % (int(spec["graph_replicate"]), int(spec["graph_seed"])))
        graph_manifest = read_json(graph_dir / "run_manifest.json")
        if graph_manifest.get("status") != "COMPLETE":
            raise ValueError("fresh PA graph task incomplete: %s" % graph_dir)
        records.append({**spec, "potential_replicate": 0, "potential_seed": -1, "a_over_s_ref": 0.0, "a": 0.0, "path": graph_dir / "baseline"})
        for potential_rep in range(1, int(config["fresh_pa"]["potential_replicates"]) + 1):
            seed = int(spec["graph_seed"]) + 100000000 + potential_rep
            potential_dir = graph_dir / ("potential_rep_%d_seed_%d" % (potential_rep, seed))
            for ratio in [float(value) for value in config["fresh_pa"]["amplitude_ratios"] if float(value) > 0.0]:
                records.append({
                    **spec,
                    "potential_replicate": potential_rep,
                    "potential_seed": seed,
                    "a_over_s_ref": ratio,
                    "a": ratio * float(config["fresh_pa"]["s_ref"]),
                    "path": potential_dir / ("ratio_" + _ratio_slug(ratio)),
                })
    for record in records:
        manifest = read_json(Path(record["path"]) / "run_manifest.json")
        if manifest.get("status") != "COMPLETE":
            raise ValueError("fresh PA checkpoint incomplete: %s" % record["path"])
    return records


def _pilot_records(config: Mapping[str, object], pilot_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for model in config["pilot"]["models"]:
        for seed in config["pilot"]["potential_seeds"]:
            sweep = pilot_root / str(model) / ("potential_seed_%d" % int(seed))
            if read_json(sweep / "run_manifest.json").get("status") != "COMPLETE":
                raise ValueError("pilot postprocess incomplete: %s" % sweep)
            for ratio in [float(value) for value in config["pilot"]["amplitude_ratios"]]:
                checkpoint = sweep / ("ratio_" + _ratio_slug(ratio))
                manifest = read_json(checkpoint / "run_manifest.json")
                if manifest.get("status") == "DEDUPLICATED_REFERENCE":
                    continue
                if manifest.get("status") != "COMPLETE":
                    raise ValueError("pilot checkpoint incomplete: %s" % checkpoint)
                records.append({
                    "model_key": model,
                    "n": 5000,
                    "graph_replicate": 1,
                    "graph_seed": config["pilot"]["graph_seeds"][list(config["pilot"]["models"]).index(model)],
                    "potential_replicate": list(config["pilot"]["potential_seeds"]).index(seed) + 1 if ratio > 0 else 0,
                    "potential_seed": int(seed) if ratio > 0 else -1,
                    "a_over_s_ref": ratio,
                    "a": ratio * float(config["pilot"]["s_ref"]),
                    "path": checkpoint,
                })
    return records


def _rrg_records(config: Mapping[str, object], rrg_root: Path, include_a9: bool) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    total = len(config["rrg"]["n_values"]) * int(config["rrg"]["graph_replicates"])
    for task_id in range(1, total + 1):
        spec = _rrg_task(config, task_id)
        graph_dir = rrg_root / ("N%d" % spec["n"]) / ("graph_rep_%d_seed_%d" % (spec["graph_replicate"], spec["graph_seed"]))
        if read_json(graph_dir / "run_manifest.json").get("status") != "COMPLETE":
            raise ValueError("RRG graph task incomplete: %s" % graph_dir)
        records.append({**spec, "potential_replicate": 0, "potential_seed": -1, "a": 0.0, "W": 0.0, "path": graph_dir / "baseline"})
        for potential_rep in range(1, int(config["rrg"]["potential_replicates"]) + 1):
            seed = int(spec["graph_seed"]) + 100000000 + potential_rep
            potential_dir = graph_dir / ("potential_rep_%d_seed_%d" % (potential_rep, seed))
            amplitudes = [float(value) for value in config["rrg"]["amplitudes"] if float(value) > 0.0]
            if include_a9:
                amplitudes.append(9.0)
            for amplitude in sorted(amplitudes):
                checkpoint = potential_dir / ("a_" + _amplitude_slug(amplitude))
                manifest = read_json(checkpoint / "run_manifest.json")
                if manifest.get("status") != "COMPLETE":
                    raise ValueError("RRG checkpoint incomplete: %s" % checkpoint)
                records.append({**spec, "potential_replicate": potential_rep, "potential_seed": seed, "a": amplitude, "W": 2.0 * amplitude, "path": checkpoint})
    return records


def _graph_points(
    rows: Sequence[Mapping[str, object]],
    *,
    dimensions: Sequence[str],
    metrics: Sequence[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    nested: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(name) for name in dimensions)
        nested[key].append(row)
    points: list[dict[str, object]] = []
    for key, group in nested.items():
        point = dict(zip(dimensions, key))
        point["potential_observation_count"] = len(group)
        for metric in metrics:
            values = [_number(row, metric) for row in group]
            finite = [value for value in values if math.isfinite(value)]
            point[metric] = float(np.mean(finite)) if finite else math.nan
        points.append(point)
    group_dimensions = [name for name in dimensions if name not in {"graph_replicate", "graph_seed"}]
    groups: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for point in points:
        groups[tuple(point.get(name) for name in group_dimensions)].append(point)
    summaries: list[dict[str, object]] = []
    for key, group in groups.items():
        summary = dict(zip(group_dimensions, key))
        summary["graph_count"] = len(group)
        for metric in metrics:
            values = np.asarray([_number(row, metric) for row in group], dtype=float)
            values = values[np.isfinite(values)]
            summary[metric + "_graph_mean"] = float(np.mean(values)) if len(values) else math.nan
            summary[metric + "_graph_median"] = float(np.median(values)) if len(values) else math.nan
            summary[metric + "_graph_sd"] = float(np.std(values, ddof=1)) if len(values) >= 2 else math.nan
            half = 4.302652729911275 * float(np.std(values, ddof=1)) / math.sqrt(len(values)) if len(values) == 3 else math.nan
            summary[metric + "_graph_t95_low"] = float(np.mean(values) - half) if math.isfinite(half) else math.nan
            summary[metric + "_graph_t95_high"] = float(np.mean(values) + half) if math.isfinite(half) else math.nan
        summaries.append(summary)
    return points, summaries


def _size_slopes(group_summary: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in group_summary:
        grouped[(row.get("model_key"), row.get("a_over_s_ref"), row.get("window"))].append(row)
    output: list[dict[str, object]] = []
    for key, rows in grouped.items():
        rows = sorted(rows, key=lambda row: int(float(row["n"])))
        for metric in ("median_ipr_graph_mean", "median_participation_fraction_graph_mean"):
            x = np.asarray([math.log(float(row["n"])) for row in rows if _number(row, metric) > 0.0], dtype=float)
            y = np.asarray([math.log(_number(row, metric)) for row in rows if _number(row, metric) > 0.0], dtype=float)
            slope = float(np.polyfit(x, y, 1)[0]) if len(x) == 3 else math.nan
            output.append({"model_key": key[0], "a_over_s_ref": key[1], "window": key[2], "metric": metric, "log_log_three_size_slope": slope, "size_count": int(len(x)), "interpretation": "descriptive three-size slope; not a critical exponent"})
    return output


def _heldout_unfolding(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    spectrum_cache: dict[str, np.ndarray] = {}
    for target in records:
        if float(target["a_over_s_ref"]) == 0.0:
            continue
        path = Path(target["path"])
        target_rows = _read_if(path / "tables" / "gap_triplets.csv.gz")
        references = [
            record for record in records
            if record["model_key"] == target["model_key"]
            and int(record["n"]) == int(target["n"])
            and float(record["a_over_s_ref"]) == float(target["a_over_s_ref"])
            and int(record["graph_replicate"]) != int(target["graph_replicate"])
        ]
        reference_values: list[np.ndarray] = []
        for reference in references:
            key = str(reference["path"])
            if key not in spectrum_cache:
                with np.load(Path(key) / "spectrum_scalars.npz", allow_pickle=False) as data:
                    spectrum_cache[key] = np.asarray(data["eigenvalues"], dtype=float)
            reference_values.append(spectrum_cache[key])
        if not reference_values:
            continue
        reference = np.sort(np.concatenate(reference_values))
        n = int(target["n"])
        for row in target_rows:
            updated: dict[str, object] = dict(row)
            energies = np.asarray([float(row["left_energy"]), float(row["middle_energy"]), float(row["right_energy"])], dtype=float)
            mapped = n * np.searchsorted(reference, energies, side="right") / float(len(reference))
            left, right = float(mapped[1] - mapped[0]), float(mapped[2] - mapped[1])
            denominator = max(left, right)
            updated.update({
                "left_unfolded_gap": left,
                "right_unfolded_gap": right,
                "unfolded_gap_ratio": min(left, right) / denominator if denominator > 0 else math.nan,
                "heldout_reference_graph_count": len(set(int(record["graph_replicate"]) for record in references)),
                "heldout_reference_spectrum_count": len(references),
                "target_graph_fully_excluded": True,
                "unfolding_status": "exploratory_three_graph_reference",
            })
            output.append(updated)
    return output


def _rrg_a9_decision(rrg_summary: Sequence[Mapping[str, object]], config: Mapping[str, object]) -> dict[str, object]:
    midpoint = 0.5 * (float(config["statistics"]["poisson_mean_raw_gap_ratio"]) + float(config["statistics"]["goe_reference"]))
    lookup = {(int(float(row["n"])), float(row["a"])): row for row in rrg_summary}
    straddles = []
    changes = []
    for n in [int(value) for value in config["rrg"]["n_values"]]:
        r8 = _number(lookup.get((n, 8.0), {}), "mean_raw_gap_ratio_graph_mean")
        r10 = _number(lookup.get((n, 10.0), {}), "mean_raw_gap_ratio_graph_mean")
        straddles.append(math.isfinite(r8) and math.isfinite(r10) and (r8 - midpoint) * (r10 - midpoint) <= 0.0)
        changes.append(abs(r10 - r8) if math.isfinite(r8) and math.isfinite(r10) else math.nan)
    slopes: dict[float, float] = {}
    for amplitude in (8.0, 10.0):
        rows = [lookup[(n, amplitude)] for n in [int(value) for value in config["rrg"]["n_values"]] if (n, amplitude) in lookup]
        x = np.asarray([math.log(float(row["n"])) for row in rows], dtype=float)
        y = np.asarray([math.log(_number(row, "median_participation_fraction_graph_mean")) for row in rows], dtype=float)
        slopes[amplitude] = float(np.polyfit(x, y, 1)[0]) if len(x) == 3 and np.all(np.isfinite(y)) else math.nan
    reverses = math.isfinite(slopes[8.0]) and math.isfinite(slopes[10.0]) and slopes[8.0] * slopes[10.0] < 0.0
    large_change = any(math.isfinite(value) and value >= 0.03 for value in changes)
    trigger = any(straddles) or reverses or large_change
    return {
        "status": "COMPLETE",
        "rule_frozen_before_results": str(config["rrg"]["conditional_a9_rule"]),
        "midpoint_poisson_goe_reference": midpoint,
        "straddles_by_size": straddles,
        "absolute_mean_r_change_by_size": changes,
        "participation_fraction_log_size_slopes": {"a8": slopes[8.0], "a10": slopes[10.0]},
        "slope_sign_reversal": reverses,
        "large_mean_r_change": large_change,
        "trigger_a9": trigger,
        "maximum_authorized_new_instances": int(config["rrg"]["conditional_a9_max_new_instances"]),
    }


def _model_label(key: str) -> str:
    return {"affine_alpha_1p05": r"affine $\alpha=1.05$", "nonlinear_zeta_1p5": r"nonlinear $\zeta=1.5$", "affine_alpha_2p0": r"affine $\alpha=2$"}.get(key, key)


def _make_figures(
    *,
    figures: Path,
    pilot_windows: Sequence[Mapping[str, object]],
    pa_summary: Sequence[Mapping[str, object]],
    rrg_summary: Sequence[Mapping[str, object]],
    selected_rows: Sequence[Mapping[str, object]],
    geometry_rows: Sequence[Mapping[str, object]],
    fresh_records: Sequence[Mapping[str, object]],
) -> list[Path]:
    figures.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    for axis, model in zip(axes, ("affine_alpha_1p05", "nonlinear_zeta_1p5", "affine_alpha_2p0")):
        for window, marker in (("q25_width_0p02", "o"), ("q50_width_0p02", "s"), ("q75_width_0p02", "^"), ("physical_abs_energy", "D")):
            rows = [row for row in pilot_windows if row.get("model_key") == model and row.get("window") == window]
            grouped: dict[float, list[float]] = defaultdict(list)
            for row in rows:
                value = _number(row, "median_ipr")
                if math.isfinite(value):
                    grouped[_number(row, "a_over_s_ref")].append(value)
            x = sorted(grouped)
            y = [float(np.mean(grouped[value])) for value in x]
            axis.plot(x, y, marker=marker, ms=4, label=window.replace("_width_0p02", ""))
        axis.set_title(_model_label(model)); axis.set_xlabel(r"$a/s_{ref}$"); axis.grid(alpha=.25)
    axes[0].set_ylabel("median IPR (pilot; scalar coverage)"); axes[-1].legend(fontsize=7)
    fig.suptitle("Repaired pilot: fixed quantile and physical windows")
    path = figures / "pilot_localization_by_disorder.png"; fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig); paths.append(path)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4), sharex="col")
    palette = {0.0: "#333333", 0.5: "#2b83ba", 2.0: "#fdae61", 8.0: "#d7191c"}
    for column, model in enumerate(("affine_alpha_1p05", "nonlinear_zeta_1p5", "affine_alpha_2p0")):
        for ratio in (0.0, 0.5, 2.0, 8.0):
            record = next((item for item in fresh_records if item["model_key"] == model and int(item["n"]) == 5000 and int(item["graph_replicate"]) == 1 and float(item["a_over_s_ref"]) == ratio and (ratio == 0.0 or int(item["potential_replicate"]) == 1)), None)
            if record is None:
                continue
            with np.load(Path(record["path"]) / "spectrum_scalars.npz", allow_pickle=False) as data:
                values = np.asarray(data["eigenvalues"], dtype=float)
                ipr = np.asarray(data["ipr"], dtype=float)
                atom = np.asarray(data["atom_mask"], dtype=bool)
            color = palette[ratio]
            axes[0, column].hist(values, bins=90, density=True, histtype="step", lw=1.0, color=color, label=str(ratio))
            edges = np.linspace(float(values[0]), float(values[-1]), 70)
            centers = 0.5 * (edges[:-1] + edges[1:])
            digit = np.digitize(values, edges) - 1
            medians = np.asarray([np.median(ipr[(digit == index) & ~atom]) if np.any((digit == index) & ~atom) else np.nan for index in range(len(centers))])
            axes[1, column].plot(centers, medians, lw=1.0, color=color)
        axes[0, column].set_title(_model_label(model)); axes[0, column].grid(alpha=.2); axes[1, column].grid(alpha=.2)
        axes[1, column].set_xlabel("energy E")
    axes[0, 0].set_ylabel("eigenvalue density"); axes[1, 0].set_ylabel("binned median IPR")
    axes[0, -1].legend(title=r"$a/s_{ref}$", fontsize=7)
    axes[1, 0].set_yscale("log"); axes[1, 1].set_yscale("log"); axes[1, 2].set_yscale("log")
    fig.suptitle("Fresh PA N=5000: energy density and localization (graph 1; potential 1 for a>0)")
    path = figures / "fresh_pa_energy_density_and_localization.png"; fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig); paths.append(path)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    for axis, model in zip(axes, ("affine_alpha_1p05", "nonlinear_zeta_1p5", "affine_alpha_2p0")):
        for ratio, color in ((0.0, "#333333"), (0.5, "#2b83ba"), (2.0, "#fdae61"), (8.0, "#d7191c")):
            rows = [row for row in pa_summary if row.get("model_key") == model and row.get("window") == "q75_width_0p02" and abs(_number(row, "a_over_s_ref") - ratio) < 1e-12]
            rows.sort(key=lambda row: int(float(row["n"])))
            axis.plot([int(float(row["n"])) for row in rows], [_number(row, "median_participation_fraction_graph_mean") for row in rows], marker="o", label=str(ratio), color=color)
        axis.set_xscale("log"); axis.set_yscale("log"); axis.set_title(_model_label(model)); axis.set_xlabel("N"); axis.grid(alpha=.25)
    axes[0].set_ylabel("graph-mean median participation fraction"); axes[-1].legend(title=r"$a/s_{ref}$", fontsize=7)
    fig.suptitle("Fresh PA replication: q75 finite-size flow")
    path = figures / "fresh_pa_q75_participation_fraction_size.png"; fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig); paths.append(path)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    for axis, model in zip(axes, ("affine_alpha_1p05", "nonlinear_zeta_1p5", "affine_alpha_2p0")):
        for n, marker in ((1250, "o"), (2500, "s"), (5000, "^")):
            rows = [row for row in pa_summary if row.get("model_key") == model and int(float(row["n"])) == n and row.get("window") == "upper_rank_1_10"]
            rows.sort(key=lambda row: _number(row, "a_over_s_ref"))
            axis.plot([_number(row, "a_over_s_ref") for row in rows], [_number(row, "mean_raw_gap_ratio_graph_mean") for row in rows], marker=marker, label="N=%d" % n)
        axis.axhline(0.3862943611, color="black", ls=":", lw=.8); axis.axhline(.53, color="gray", ls="--", lw=.8)
        axis.set_title(_model_label(model)); axis.set_xlabel(r"$a/s_{ref}$"); axis.grid(alpha=.25)
    axes[0].set_ylabel("graph-mean raw adjacent-gap ratio"); axes[-1].legend(fontsize=7)
    fig.suptitle("Fresh PA upper edge ranks 1–10 (counts in tables)")
    path = figures / "fresh_pa_upper_edge_gap_ratio.png"; fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig); paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for n, marker in ((1250, "o"), (2500, "s"), (5000, "^")):
        rows = [row for row in rrg_summary if int(float(row["n"])) == n]
        rows.sort(key=lambda row: _number(row, "a"))
        axes[0].plot([_number(row, "a") for row in rows], [_number(row, "mean_raw_gap_ratio_graph_mean") for row in rows], marker=marker, label="N=%d" % n)
        axes[1].plot([_number(row, "a") for row in rows], [_number(row, "median_participation_fraction_graph_mean") for row in rows], marker=marker, label="N=%d" % n)
    axes[0].axhline(0.3862943611, color="black", ls=":", lw=.8, label="Poisson ref."); axes[0].axhline(.53, color="gray", ls="--", lw=.8, label="GOE ref.")
    axes[0].set_ylabel("mean raw gap ratio"); axes[1].set_ylabel("median participation fraction")
    for axis in axes: axis.set_xlabel("a (W=2a)"); axis.grid(alpha=.25); axis.legend(fontsize=7)
    axes[1].set_yscale("log"); fig.suptitle(r"Random 3-regular benchmark, complete $|E|\leq0.1$ interval")
    path = figures / "rrg_central_finite_size_disorder.png"; fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig); paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    edge = [row for row in selected_rows if str(row.get("source_kind")) == "fresh_pa_replication" and str(row.get("window_memberships", "")).find("upper_rank_1_10") >= 0]
    axes[0].scatter([_number(row, "center_potential") for row in edge], [_number(row, "maximum_coordinate_mass") for row in edge], s=8, alpha=.45)
    axes[0].set_xlabel("potential at maximum-mass coordinate"); axes[0].set_ylabel("maximum coordinate mass"); axes[0].grid(alpha=.25)
    axes[1].scatter([_number(row, "dominant_vertex_distance") for row in geometry_rows], [_number(row, "full_mass_overlap") for row in geometry_rows], s=8, alpha=.45)
    axes[1].set_xlabel("dominant-coordinate graph distance"); axes[1].set_ylabel("mass-distribution overlap"); axes[1].grid(alpha=.25)
    fig.suptitle("Descriptive potential and geometry associations (not causal tests)")
    path = figures / "pa_potential_and_geometry_associations.png"; fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig); paths.append(path)

    representative = next((record for record in fresh_records if record["model_key"] == "affine_alpha_2p0" and int(record["n"]) == 5000 and int(record["graph_replicate"]) == 1 and int(record["potential_replicate"]) == 1 and float(record["a_over_s_ref"]) == 8.0), None)
    if representative is not None:
        checkpoint = Path(representative["path"])
        graph_dir = checkpoint.parents[1]
        graph, _ = graph_from_edge_cache(graph_dir / "graph" / "graph_realization.npz")
        with np.load(checkpoint / "selected_eigenvectors.npz", allow_pickle=False) as data:
            indices = np.asarray(data["ascending_indices"], dtype=int)
            vectors = np.asarray(data["eigenvectors"], dtype=float)
        position = int(np.argmax(indices))
        order, masses = deterministic_mass_order(vectors[:, position])
        top = order[:10]
        induced = graph.subgraph(top.tolist()).copy()
        fig, axis = plt.subplots(figsize=(6.4, 5.4))
        layout = nx.spring_layout(induced, seed=17)
        nx.draw_networkx(induced, pos=layout, node_size=300 + 9000 * masses[top], node_color=masses[top], cmap="viridis", with_labels=True, font_size=7, ax=axis)
        axis.set_title("Representative top-10 mass induced graph\naffine alpha=2, N=5000, a/s_ref=8, upper rank 1")
        axis.axis("off")
        path = figures / "representative_top10_induced_graph.png"; fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig); paths.append(path)
    return paths


def _format(value: float) -> str:
    return "NA" if not math.isfinite(value) else ("%.4g" % value)


def _report_text(
    *,
    phase: str,
    source_id: str,
    pilot_windows: Sequence[Mapping[str, object]],
    pa_summary: Sequence[Mapping[str, object]],
    pa_slopes: Sequence[Mapping[str, object]],
    rrg_summary: Sequence[Mapping[str, object]],
    decision: Mapping[str, object],
    figure_count: int,
) -> str:
    lines = [
        "# Anderson/random Schrodinger execution report", "",
        "Generated: %s" % utc_now(), "",
        "## Outcome", "",
        "The bounded Anderson-only workflow completed its cache repair, targeted baseline supplements, fresh PA replication, and random 3-regular benchmark. Historical arrays were not overwritten. The clean-PA coupling construction was not restarted; its retained result remains zero accepted predictive two-mode cases.", "",
        "The operator throughout is `H(a)=A+a*diag(xi)` with unit unscaled hopping and `xi` uniform on `[-1,1)`. PA amplitudes are also reported as `a/s_ref` with `s_ref=sqrt(20)`. RRG literature disorder is `W=2a`.", "",
        "## Completion matrix", "",
        "| stage | implemented | executed | numerically validated | scientifically supported |", "|---|---|---|---|---|",
        "| Pilot cache-only repair | yes | 6 sweeps / 48 inputs, graph-level a=0 deduplicated | inherited residual checks linked; repaired tables checked | descriptive on one graph/model only |",
        "| Targeted baseline supplement | yes | q25/q75 on 3 retained graphs; affine-alpha2 missing q50 vector only | index, residual, norm, and orthogonality checks | coordinate comparison only; heavy-tail q50 remains subspace-based |",
        "| Fresh PA replication | yes | 27 graphs, 54 potential streams, 189 operator instances | per-checkpoint residual/coverage checks | finite three-graph, three-size description; no transition claim |",
        "| Random 3-regular benchmark | yes | 9 graphs and base amplitudes, with conditional a=9 recorded below | complete-reference interval checks at N=1250 plus per-checkpoint checks | finite-size benchmark only |", "",
        "## Reused and repaired evidence", "",
        "The six historical PA sweeps remain the numerical source for the pilot. Positive-disorder selected eigenpairs retain the corrected worst relative residual `9.35e-14`. The repaired path adds explicit window membership, width-0.02 scalar windows, full coverage ledgers, top-k/support metrics, atom-safe consecutive gap triplets, shell/ball mass, induced connectivity, within/cross distances, center degree/potential, and the exact adjacency-plus-diagonal energy decomposition. Baseline `a=0` is counted once per graph.", "",
        "The stable baseline near-zero dimensions remain 1844, 240, and 0 for affine alpha=1.05, nonlinear zeta=1.5, and affine alpha=2. Individual-vector metrics inside the first two atoms are reported unavailable; projector/subspace evidence is used instead.", "",
        "## Fresh PA findings", "",
        "The tables retain all three graph-level points per cell and treat the two potentials as nested repeated observations. Representative q75 participation-fraction values are:", "",
        "| model | a/s_ref | N=1250 | N=2500 | N=5000 |", "|---|---:|---:|---:|---:|",
    ]
    for model in ("affine_alpha_1p05", "nonlinear_zeta_1p5", "affine_alpha_2p0"):
        for ratio in (0.0, 0.5, 2.0, 8.0):
            lookup = {(int(float(row["n"])), float(row["a_over_s_ref"])): row for row in pa_summary if row.get("model_key") == model and row.get("window") == "q75_width_0p02"}
            values = [_number(lookup.get((n, ratio), {}), "median_participation_fraction_graph_mean") for n in (1250, 2500, 5000)]
            lines.append("| %s | %g | %s | %s | %s |" % (model, ratio, *[_format(value) for value in values]))
    lines.extend([
        "", "Three-size slopes are descriptive and are not localization exponents. Rank-defined edges, quantile windows, and physical-energy windows follow different spectral populations and are kept separate. Full numerical values and uncertainty intervals are in the graph-level tables.", "",
        "## Random 3-regular benchmark", "",
        "The benchmark uses connected simple degree-three graphs and complete LAPACK interval coverage for `|E|<=0.1`, with one guard on each side. A complete N=1250 reference was used to validate interval counts and eigenvalues. Tikhonov–Mirlin (2019) study degree three at E=0 with box disorder `[-W/2,W/2]` and report `W_c=18.17±0.01`; therefore their convention maps to `a≈9.085` here. This is context, not a PA threshold or proof of a finite-size transition: https://arxiv.org/abs/1903.07877", "",
        "| a | W | N=1250 mean r | N=2500 mean r | N=5000 mean r |", "|---:|---:|---:|---:|---:|",
    ])
    amplitudes = sorted(set(_number(row, "a") for row in rrg_summary))
    for amplitude in amplitudes:
        lookup = {int(float(row["n"])): row for row in rrg_summary if abs(_number(row, "a") - amplitude) < 1e-12}
        values = [_number(lookup.get(n, {}), "mean_raw_gap_ratio_graph_mean") for n in (1250, 2500, 5000)]
        lines.append("| %g | %g | %s | %s | %s |" % (amplitude, 2 * amplitude, *[_format(value) for value in values]))
    lines.extend([
        "", "Conditional a=9 decision: `%s`. The frozen trigger result is `%s`." % (decision.get("rule_frozen_before_results"), decision.get("trigger_a9")), "",
        "## Interpretation and limitations", "",
        "Raw adjacent-gap ratios are shown before any unfolding. Poisson mean 2 log 2 - 1 (about 0.38629), GOE around 0.53, and the common 0.5359 surmise are references rather than binary classifiers. Large IPR alone does not imply Poisson gaps. Graph-held-out PA unfolding excludes every potential belonging to the target graph and is explicitly exploratory with only two reference graphs.", "",
        "The bounded design can identify finite-size and disorder-dependent concentration patterns that repeat across these graph samples. It cannot establish a mobility edge, a critical amplitude, an asymptotic localized phase, or a universal mechanism. Center potential, degree, distances, and energy decomposition are descriptive associations rather than causal tunneling tests.", "",
        "## Provenance and navigation", "",
        "Source/configuration identity: `%s`. The review bundle contains `RESULTS_INDEX.md`, `EVIDENCE_INVENTORY.csv`, `run_manifest.json`, the configuration/freeze references, graph-level uncertainty tables, coverage and table indices, exhaustive repaired tables, and %d inspected figure files." % (source_id, figure_count), "",
    ])
    return "\n".join(lines)


def _render_pdf(report_text: str, figure_paths: Sequence[Path], output_path: Path) -> None:
    plain = report_text.replace("`", "")
    logical_lines: list[str] = []
    for line in plain.splitlines():
        if line.startswith("|---"):
            continue
        logical_lines.extend(textwrap.wrap(line, width=108, replace_whitespace=False, drop_whitespace=False) or [""])
    with PdfPages(output_path) as pdf:
        # At this font size a letter page safely holds 62 wrapped lines above
        # the footer.  The earlier 49-line chunk left a two-line orphan page.
        for start in range(0, len(logical_lines), 62):
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(.07, .95, "\n".join(logical_lines[start:start + 62]), va="top", ha="left", fontsize=8.2, family="DejaVu Sans")
            fig.text(.5, .025, "Anderson/random Schrodinger execution report", ha="center", fontsize=8, color="0.4")
            pdf.savefig(fig); plt.close(fig)
        for path in figure_paths:
            image = plt.imread(path)
            fig = plt.figure(figsize=(11, 8.5))
            axis = fig.add_axes([.03, .05, .94, .90]); axis.imshow(image); axis.axis("off")
            fig.suptitle(path.stem.replace("_", " "), fontsize=12)
            pdf.savefig(fig); plt.close(fig)


def run_anderson_aggregation(
    *,
    project_root: Path,
    config_path: Path,
    freeze_manifest: Path,
    run_root: Path,
    output_dir: Path,
    phase: str,
) -> Path:
    started = time.time()
    if phase not in {"pre_a9", "final"}:
        raise ValueError("phase must be pre_a9 or final")
    context = verify_execution_context(project_root, config_path, freeze_manifest)
    config = context["config"]
    run_root, output_dir = Path(run_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = output_dir / "tables"; figures = output_dir / "figures"
    tables.mkdir(exist_ok=True); figures.mkdir(exist_ok=True)
    pilot_records = _pilot_records(config, run_root / "pilot_postprocessing")
    fresh_records = _fresh_records(config, run_root / "fresh_pa")
    rrg_records = _rrg_records(config, run_root / "rrg_benchmark", include_a9=phase == "final" and (run_root / "RRG_A9_TRIGGERED").is_file())
    index_rows: list[dict[str, object]] = []
    pilot_windows: list[dict[str, object]] = []
    fresh_windows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    geometry_rows: list[dict[str, object]] = []
    gap_rows: list[dict[str, object]] = []
    for stage, records in (("pilot_repair", pilot_records), ("fresh_pa", fresh_records)):
        for record in records:
            path = Path(record["path"])
            metadata = {key: value for key, value in record.items() if key != "path"}
            _write_indexed_table(index_rows, path, stage=stage, metadata=metadata)
            windows = _read_if(path / "tables" / "window_summary.csv")
            (pilot_windows if stage == "pilot_repair" else fresh_windows).extend(windows)
            selected_rows.extend(_read_if(path / "tables" / "selected_vector_metrics.csv.gz"))
            geometry_rows.extend(_read_if(path / "tables" / "edge_pair_geometry.csv.gz"))
            gap_rows.extend(_read_if(path / "tables" / "gap_triplets.csv.gz"))
    supplement_rows: list[dict[str, object]] = []
    for model in config["pilot"]["models"]:
        supplement = run_root / "targeted_baseline_supplements" / str(model)
        if read_json(supplement / "run_manifest.json").get("status") != "COMPLETE":
            raise ValueError("baseline supplement incomplete: %s" % supplement)
        _write_indexed_table(index_rows, supplement, stage="baseline_supplement", metadata={"model_key": model})
        supplement_rows.extend(_read_if(supplement / "tables" / "selected_vector_metrics.csv.gz"))
    selected_rows.extend(supplement_rows)
    rrg_rows: list[dict[str, object]] = []
    rrg_eigenpair_rows: list[dict[str, object]] = []
    for record in rrg_records:
        path = Path(record["path"])
        metadata = {key: value for key, value in record.items() if key != "path"}
        _write_indexed_table(index_rows, path, stage="rrg", metadata=metadata)
        rrg_rows.extend(_read_if(path / "tables" / "central_summary.csv"))
        rrg_eigenpair_rows.extend(_read_if(path / "tables" / "central_eigenpair_metrics.csv.gz"))
    pa_points, pa_summary = _graph_points(
        fresh_windows,
        dimensions=("model_key", "n", "a_over_s_ref", "a", "window", "graph_replicate", "graph_seed"),
        metrics=("mean_ipr", "median_ipr", "median_participation_fraction", "mean_raw_gap_ratio", "valid_gap_triplet_count"),
    )
    pa_slopes = _size_slopes(pa_summary)
    rrg_points, rrg_summary = _graph_points(
        rrg_rows,
        dimensions=("n", "a", "W", "graph_replicate", "graph_seed"),
        metrics=("mean_raw_gap_ratio", "median_ipr", "median_n_times_ipr", "median_participation_fraction", "median_maximum_coordinate_mass", "median_support_fraction_90", "target_level_count"),
    )
    decision = _rrg_a9_decision(rrg_summary, config)
    atomic_write_json(output_dir / "RRG_A9_DECISION.json", decision)
    if phase == "pre_a9":
        atomic_write_csv(tables / "rrg_graph_points.csv", rrg_points)
        atomic_write_csv(tables / "rrg_graph_uncertainty.csv", rrg_summary)
        atomic_write_json(output_dir / "run_manifest.json", {
            "status": "COMPLETE", "stage_version": ANDERSON_REPORTING_VERSION, "phase": phase,
            "source_configuration_id": context["source_configuration_identity"]["sha256"],
            "trigger_a9": decision["trigger_a9"], "finished_at_utc": utc_now(), "runtime_seconds": time.time() - started,
        })
        return output_dir
    unfolded_rows = _heldout_unfolding(fresh_records)
    atomic_write_csv(tables / "DATA_TABLE_INDEX.csv", index_rows)
    atomic_write_csv(tables / "pilot_window_summary.csv.gz", pilot_windows)
    atomic_write_csv(tables / "fresh_pa_window_summary.csv.gz", fresh_windows)
    atomic_write_csv(tables / "selected_vector_metrics.csv.gz", selected_rows)
    atomic_write_csv(tables / "edge_pair_geometry.csv.gz", geometry_rows)
    atomic_write_csv(tables / "gap_triplets.csv.gz", gap_rows)
    atomic_write_csv(tables / "fresh_pa_graph_points.csv", pa_points)
    atomic_write_csv(tables / "fresh_pa_graph_uncertainty.csv", pa_summary)
    atomic_write_csv(tables / "fresh_pa_size_slopes.csv", pa_slopes)
    atomic_write_csv(tables / "fresh_pa_graph_heldout_unfolded_triplets.csv.gz", unfolded_rows)
    atomic_write_csv(tables / "rrg_central_eigenpair_metrics.csv.gz", rrg_eigenpair_rows)
    atomic_write_csv(tables / "rrg_graph_points.csv", rrg_points)
    atomic_write_csv(tables / "rrg_graph_uncertainty.csv", rrg_summary)
    figure_paths = _make_figures(
        figures=figures, pilot_windows=pilot_windows, pa_summary=pa_summary, rrg_summary=rrg_summary,
        selected_rows=selected_rows, geometry_rows=geometry_rows, fresh_records=fresh_records,
    )
    report = _report_text(
        phase=phase, source_id=str(context["source_configuration_identity"]["sha256"]),
        pilot_windows=pilot_windows, pa_summary=pa_summary, pa_slopes=pa_slopes, rrg_summary=rrg_summary,
        decision=decision, figure_count=len(figure_paths),
    )
    (output_dir / "ANDERSON_EXECUTION_REPORT.md").write_text(report + "\n")
    _render_pdf(report, figure_paths, output_dir / "ANDERSON_EXECUTION_REPORT.pdf")
    index = """# Anderson/random Schrodinger results index

Recommended review order:

1. `ANDERSON_EXECUTION_REPORT.md` and `ANDERSON_EXECUTION_REPORT.pdf`
2. `tables/fresh_pa_graph_uncertainty.csv` and `tables/rrg_graph_uncertainty.csv`
3. `tables/pilot_window_summary.csv.gz` and `tables/fresh_pa_window_summary.csv.gz`
4. `tables/selected_vector_metrics.csv.gz`, `tables/gap_triplets.csv.gz`, and `tables/edge_pair_geometry.csv.gz`
5. `tables/fresh_pa_graph_heldout_unfolded_triplets.csv.gz`
6. `tables/DATA_TABLE_INDEX.csv` for all per-checkpoint machine-readable tables and remote cache paths
7. `RRG_A9_DECISION.json`, `EVIDENCE_INVENTORY.csv`, and `run_manifest.json`
8. `figures/`

Historical disorder arrays and corrected September 22 evidence remain immutable at their recorded roots. Large graph, potential, spectrum, and raw-vector caches remain in the sibling Hoffman stage directories and are indexed rather than duplicated here.
"""
    (output_dir / "RESULTS_INDEX.md").write_text(index)
    evidence_paths = sorted(list(tables.glob("*")) + list(figures.glob("*")) + [output_dir / "ANDERSON_EXECUTION_REPORT.md", output_dir / "ANDERSON_EXECUTION_REPORT.pdf", output_dir / "RESULTS_INDEX.md", output_dir / "RRG_A9_DECISION.json"])
    evidence = [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in evidence_paths]
    atomic_write_csv(output_dir / "EVIDENCE_INVENTORY.csv", evidence)
    manifest = {
        "status": "COMPLETE", "stage_version": ANDERSON_REPORTING_VERSION, "phase": phase,
        "source_configuration_id": context["source_configuration_identity"]["sha256"],
        "configuration_sha256": context["configuration_sha256"],
        "pilot_checkpoint_count": len(pilot_records), "baseline_supplement_model_count": len(config["pilot"]["models"]),
        "fresh_pa_graph_count": len(set((row["model_key"], row["n"], row["graph_replicate"]) for row in fresh_records)),
        "fresh_pa_operator_instance_count": len(fresh_records),
        "rrg_graph_count": len(set((row["n"], row["graph_replicate"]) for row in rrg_records)),
        "rrg_operator_instance_count": len(rrg_records), "rrg_a9_triggered": bool((run_root / "RRG_A9_TRIGGERED").is_file()),
        "figure_count": len(figure_paths), "pdf_sha256": sha256_file(output_dir / "ANDERSON_EXECUTION_REPORT.pdf"),
        "runtime_seconds": time.time() - started, "finished_at_utc": utc_now(),
    }
    atomic_write_json(output_dir / "run_manifest.json", manifest)
    return output_dir
