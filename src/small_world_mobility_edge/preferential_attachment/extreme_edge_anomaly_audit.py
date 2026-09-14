"""Non-authoritative forensic audit of PA extreme-edge anomaly candidates.

This module consumes the already-produced mapped gap/ratio records.  It never
generates a graph, diagonalizes an adjacency matrix, or overwrites production
results.  The diagnostic output deliberately separates midpoint-selected gap
support from middle-eigenvalue-selected adjacent-gap-ratio support.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate, stats

from .extreme_edge_quantiles import REQUESTED_INTERVALS
from .mobility_edge_postprocess import GOE_MEAN_GAP_RATIO, POISSON_MEAN_GAP_RATIO


AUDIT_VERSION = "pa_extreme_edge_anomaly_audit_v1_non_authoritative"
HISTOGRAM_MIN_GAPS = 100
HISTOGRAM_MIN_GRAPHS = 50
RATIO_MEAN_MIN_RATIOS = 100
RATIO_MEAN_MIN_GRAPHS = 50
RATIO_DISTRIBUTION_MIN_RATIOS = 500
RATIO_DISTRIBUTION_MIN_GRAPHS = 100
BOOTSTRAP_REPLICATES = 2000

MODEL_PATHS = {
    "affine alpha=2": "affine_linear/affine_linear_degree_ccdf_alpha_2p0",
    "affine alpha=1.5": "affine_linear/affine_linear_degree_ccdf_alpha_1p5",
    "affine alpha=1.25": "affine_linear/affine_linear_degree_ccdf_alpha_1p25",
    "affine alpha=1.05": "affine_linear/affine_linear_degree_ccdf_alpha_1p05",
    "superlinear zeta=1.05": "nonlinear/nonlinear_attachment_zeta_1p05",
    "superlinear zeta=1.2": "nonlinear/nonlinear_attachment_zeta_1p2",
    "superlinear zeta=1.5": "nonlinear/nonlinear_attachment_zeta_1p5",
}

PRIMARY_SUSPICIOUS = {
    ("affine alpha=1.05", 0.995, 0.996),
    ("affine alpha=1.05", 0.996, 0.997),
    ("affine alpha=1.05", 0.9999, 0.99995),
    ("affine alpha=1.25", 0.99995, 1.0),
    ("superlinear zeta=1.5", 0.995, 0.996),
    ("superlinear zeta=1.5", 0.996, 0.997),
    ("superlinear zeta=1.5", 0.997, 0.998),
}


def window_mask(values: np.ndarray | pd.Series, low: float, high: float):
    return (values >= low) & (values <= high if high >= 1.0 - 1e-15 else values < high)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_se(values: Iterable[float]) -> tuple[float, float, int]:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan, math.nan, 0
    se = float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else math.nan
    return float(np.mean(values)), se, int(len(values))


def graph_equal(frame: pd.DataFrame, field: str) -> tuple[float, float, int, pd.Series]:
    values = frame.loc[np.isfinite(frame[field]), ["graph", field]].groupby("graph")[field].mean()
    mean, se, count = mean_se(values.to_numpy())
    return mean, se, count, values


def bootstrap_ci(graph_values: pd.Series, *, seed: int) -> tuple[float, float]:
    values = graph_values.to_numpy(dtype=float)
    if len(values) < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = np.mean(
        values[rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))], axis=1
    )
    return tuple(float(value) for value in np.quantile(means, (0.025, 0.975)))


def stable_seed(*parts: object) -> int:
    """Return a process-independent 32-bit seed for diagnostic resampling."""
    encoded = "\x1f".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], "little")


def empirical_ks(values: np.ndarray, cdf) -> float:
    values = np.sort(np.asarray(values, dtype=float))
    if not len(values):
        return math.nan
    reference = cdf(values)
    high = np.arange(1, len(values) + 1) / len(values)
    low = np.arange(0, len(values)) / len(values)
    return float(max(np.max(np.abs(high - reference)), np.max(np.abs(low - reference))))


def poisson_ratio_pdf(r: np.ndarray) -> np.ndarray:
    return 2.0 / np.square(1.0 + r)


def poisson_ratio_cdf(r: np.ndarray) -> np.ndarray:
    return 2.0 * r / (1.0 + r)


def goe_ratio_pdf(r: np.ndarray) -> np.ndarray:
    return 27.0 / 4.0 * (r + np.square(r)) / np.power(1.0 + r + np.square(r), 2.5)


_R_GRID = np.linspace(0.0, 1.0, 20001)
_GOE_CDF_GRID = integrate.cumulative_trapezoid(goe_ratio_pdf(_R_GRID), _R_GRID, initial=0.0)
_GOE_CDF_GRID /= _GOE_CDF_GRID[-1]


def goe_ratio_cdf(r: np.ndarray) -> np.ndarray:
    return np.interp(r, _R_GRID, _GOE_CDF_GRID, left=0.0, right=1.0)


def load_mapped(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with np.load(path, allow_pickle=False) as data:
        gaps = pd.DataFrame(
            {
                "graph": data["graph_index"].astype(int),
                "seed": data["graph_seed"].astype(np.uint64),
                "fold": data["fold"].astype(int),
                "gap_left": data["gap_left"],
                "gap_right": data["gap_right"],
                "gap_energy": data["gap_midpoint_energy"],
                "raw_gap": data["raw_gap"],
                "gap_local_q": data["graph_local_rank_quantile"],
                "gap_current_q": data["ensemble_crossfit_cdf_quantile"],
                "gap_edge_rank": data["graph_local_edge_rank"].astype(int),
                "segment": data["atom_segment_identifier"].astype(int),
                "is_zero": data["gap_is_zero"].astype(bool),
                "unfolded_gap": data["crossfit_unfolded_spacing"],
            }
        )
        ratios = pd.DataFrame(
            {
                "graph": data["ratio_graph_index"].astype(int),
                "ratio_energy": data["ratio_energy"],
                "raw_r": data["ratio_value"],
                "ratio_local_q": data["ratio_graph_local_rank_quantile"],
                "ratio_current_q": data["ratio_ensemble_crossfit_cdf_quantile"],
                "ratio_category": data["ratio_category_code"].astype(int),
            }
        )
    return gaps, ratios


def link_constituent_gaps(gaps: pd.DataFrame, ratios: pd.DataFrame) -> pd.DataFrame:
    """Attach the two exact within-graph gaps used by every retained ratio."""
    left = gaps[
        ["graph", "gap_right", "raw_gap", "unfolded_gap", "gap_current_q", "gap_local_q", "gap_edge_rank"]
    ].rename(
        columns={
            "gap_right": "ratio_energy",
            "raw_gap": "left_raw_gap",
            "unfolded_gap": "left_unfolded_gap",
            "gap_current_q": "left_gap_current_q",
            "gap_local_q": "left_gap_local_q",
            "gap_edge_rank": "ratio_edge_rank",
        }
    )
    right = gaps[
        ["graph", "gap_left", "raw_gap", "unfolded_gap", "gap_current_q", "gap_local_q", "gap_edge_rank"]
    ].rename(
        columns={
            "gap_left": "ratio_energy",
            "raw_gap": "right_raw_gap",
            "unfolded_gap": "right_unfolded_gap",
            "gap_current_q": "right_gap_current_q",
            "gap_local_q": "right_gap_local_q",
            "gap_edge_rank": "outer_gap_edge_rank",
        }
    )
    linked = ratios.merge(left, on=["graph", "ratio_energy"], how="left", validate="one_to_one")
    linked = linked.merge(right, on=["graph", "ratio_energy"], how="left", validate="one_to_one")
    linked["constituents_available"] = linked[
        ["left_raw_gap", "right_raw_gap", "left_unfolded_gap", "right_unfolded_gap"]
    ].notna().all(axis=1)
    linked["recomputed_raw_r"] = np.minimum(linked["left_raw_gap"], linked["right_raw_gap"]) / np.maximum(
        linked["left_raw_gap"], linked["right_raw_gap"]
    )
    linked["unfolded_r"] = np.minimum(
        linked["left_unfolded_gap"], linked["right_unfolded_gap"]
    ) / np.maximum(linked["left_unfolded_gap"], linked["right_unfolded_gap"])
    linked["base_or_augmentation"] = np.where(linked["graph"] < 500, "base_0_499", "augmentation_500_plus")
    linked["task_block_start"] = 5 * (linked["graph"] // 5)
    linked["fold"] = linked["graph"] % 5
    return linked


def fdr_bh(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    output = np.full(len(p_values), math.nan)
    finite = np.flatnonzero(np.isfinite(p_values))
    if not len(finite):
        return output
    order = finite[np.argsort(p_values[finite])]
    ranked = p_values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    output[order] = np.minimum(ranked, 1.0)
    return output


def summarize_coordinate(
    linked: pd.DataFrame,
    *,
    model_label: str,
    coordinate: str,
    low: float,
    high: float,
    nominal_low: float,
    nominal_high: float,
) -> dict[str, object]:
    mask = window_mask(linked[coordinate], low, high) & linked["constituents_available"]
    selected = linked.loc[mask].copy()
    raw, raw_se, graphs, graph_raw = graph_equal(selected, "raw_r")
    unfolded, unfolded_se, unfolded_graphs, _ = graph_equal(selected, "unfolded_r")
    bootstrap_low, bootstrap_high = bootstrap_ci(
        graph_raw,
        seed=stable_seed(model_label, coordinate, low, high),
    )
    return {
        "model_label": model_label,
        "nominal_q_low": nominal_low,
        "nominal_q_high": nominal_high,
        "coordinate": coordinate,
        "actual_low": low,
        "actual_high": high,
        "ratio_count": len(selected),
        "ratio_graph_count": graphs,
        "graph_equal_raw_r": raw,
        "raw_r_se": raw_se,
        "graph_equal_same_triplet_unfolded_r": unfolded,
        "unfolded_r_se": unfolded_se,
        "unfolded_ratio_graph_count": unfolded_graphs,
        "raw_r_bootstrap_low": bootstrap_low,
        "raw_r_bootstrap_high": bootstrap_high,
    }


def audit_model(
    *,
    model_label: str,
    mapped_path: Path,
    published_rows: dict[tuple[str, float, float], dict[str, str]],
    output_dir: Path,
) -> dict[str, list[dict[str, object]]]:
    gaps, ratios = load_mapped(mapped_path)
    linked = link_constituent_gaps(gaps, ratios)
    available = linked["constituents_available"]
    if np.any(np.abs(linked.loc[available, "raw_r"] - linked.loc[available, "recomputed_raw_r"]) > 1e-14):
        raise AssertionError(f"constituent raw-r mismatch for {model_label}")
    if np.any(linked.loc[available, "outer_gap_edge_rank"] != linked.loc[available, "ratio_edge_rank"] - 1):
        raise AssertionError(f"nonconsecutive constituent gap ranks for {model_label}")

    reproduction: list[dict[str, object]] = []
    support: list[dict[str, object]] = []
    raw_vs_unfolded: list[dict[str, object]] = []
    distributions: list[dict[str, object]] = []
    correlations: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    batch_rows: list[dict[str, object]] = []
    coordinate_rows: list[dict[str, object]] = []
    rank_rows: list[dict[str, object]] = []
    degeneracy_rows: list[dict[str, object]] = []

    model_plot_dir = output_dir / "plots" / model_label.replace(" ", "_").replace("=", "_")
    model_plot_dir.mkdir(parents=True, exist_ok=True)
    for interval_index, (low, high) in enumerate(REQUESTED_INTERVALS, start=1):
        gap_selected = gaps.loc[window_mask(gaps["gap_current_q"], low, high)]
        ratio_all = linked.loc[window_mask(linked["ratio_current_q"], low, high)]
        ratio_selected = ratio_all.loc[ratio_all["constituents_available"]]
        published = published_rows[(model_label, low, high)]
        raw, raw_se, ratio_graphs, graph_raw = graph_equal(ratio_all, "raw_r")
        unfolded, unfolded_se, unfolded_graphs, graph_unfolded = graph_equal(
            ratio_selected, "unfolded_r"
        )
        counts = ratio_all.groupby("graph").size()
        ci_low, ci_high = bootstrap_ci(graph_raw, seed=interval_index + 1000 * list(MODEL_PATHS).index(model_label))
        gap_graphs = int(gap_selected["graph"].nunique())
        histogram_supported = len(gap_selected) >= HISTOGRAM_MIN_GAPS and gap_graphs >= HISTOGRAM_MIN_GRAPHS
        ratio_supported = len(ratio_all) >= RATIO_MEAN_MIN_RATIOS and ratio_graphs >= RATIO_MEAN_MIN_GRAPHS
        distribution_supported = (
            len(ratio_all) >= RATIO_DISTRIBUTION_MIN_RATIOS
            and ratio_graphs >= RATIO_DISTRIBUTION_MIN_GRAPHS
        )
        published_raw = float(published["graph_equal_mean_raw_r_positive_conditioned"])
        published_se = float(published["raw_r_positive_conditioned_standard_error"])
        reproduction.append(
            {
                "model_label": model_label,
                "q_low": low,
                "q_high": high,
                "published_raw_r": published_raw,
                "recomputed_raw_r": raw,
                "difference_raw_r": raw - published_raw,
                "published_raw_r_se": published_se,
                "recomputed_raw_r_se": raw_se,
                "difference_raw_r_se": raw_se - published_se,
                "published_gap_count": int(published["crossfit_unfolded_histogram_sample_count"]),
                "recomputed_gap_count": int(
                    np.sum((~gap_selected["is_zero"]) & np.isfinite(gap_selected["unfolded_gap"]) & (gap_selected["unfolded_gap"] > 0))
                ),
                "published_ratio_count": int(published["number_of_valid_raw_ratios"]),
                "recomputed_ratio_count": len(ratio_all),
                "published_ratio_graph_count": int(published["graphs_with_raw_r_positive_conditioned"]),
                "recomputed_ratio_graph_count": ratio_graphs,
                "constituent_linked_ratio_count": len(ratio_selected),
                "source_file": str(mapped_path),
                "fold_ids": "0;1;2;3;4",
                "graph_id_min": int(ratio_all["graph"].min()) if len(ratio_all) else "",
                "graph_id_max": int(ratio_all["graph"].max()) if len(ratio_all) else "",
            }
        )
        support.append(
            {
                "model_label": model_label,
                "q_low": low,
                "q_high": high,
                "histogram_positive_gap_count": len(gap_selected),
                "histogram_gap_graph_count": gap_graphs,
                "ratio_count": len(ratio_all),
                "ratio_graph_count": ratio_graphs,
                "median_ratios_per_contributing_graph": float(counts.median()) if len(counts) else 0,
                "maximum_ratios_from_one_graph": int(counts.max()) if len(counts) else 0,
                "maximum_single_graph_ratio_share": float(counts.max() / len(ratio_all)) if len(counts) else math.nan,
                "histogram_support_gate": histogram_supported,
                "ratio_mean_support_gate": ratio_supported,
                "ratio_distribution_support_gate": distribution_supported,
                "histogram_supported_but_ratio_under_supported": histogram_supported and not ratio_supported,
            }
        )
        raw_vs_unfolded.append(
            {
                "model_label": model_label,
                "q_low": low,
                "q_high": high,
                "raw_ratio_count": len(ratio_all),
                "same_triplet_linked_count": len(ratio_selected),
                "same_triplet_link_fraction": len(ratio_selected) / max(len(ratio_all), 1),
                "graph_equal_raw_r": raw,
                "raw_r_se": raw_se,
                "graph_equal_unfolded_r_same_triplets": unfolded,
                "unfolded_r_se": unfolded_se,
                "raw_minus_unfolded_r": raw - unfolded,
                "ratio_graph_count": ratio_graphs,
                "unfolded_ratio_graph_count": unfolded_graphs,
            }
        )
        ratio_values = ratio_all["raw_r"].to_numpy(dtype=float)
        distributions.append(
            {
                "model_label": model_label,
                "q_low": low,
                "q_high": high,
                "ratio_count": len(ratio_values),
                "graph_equal_mean": raw,
                "graph_equal_median": float(graph_raw.median()) if len(graph_raw) else math.nan,
                "graph_level_iqr": float(graph_raw.quantile(0.75) - graph_raw.quantile(0.25)) if len(graph_raw) else math.nan,
                "bootstrap_mean_low": ci_low,
                "bootstrap_mean_high": ci_high,
                "pooled_r_KS_to_GOE": empirical_ks(ratio_values, goe_ratio_cdf),
                "pooled_r_KS_to_Poisson": empirical_ks(ratio_values, poisson_ratio_cdf),
            }
        )
        if len(ratio_selected):
            raw_corr = ratio_selected["left_raw_gap"].corr(ratio_selected["right_raw_gap"])
            unfolded_corr = ratio_selected["left_unfolded_gap"].corr(ratio_selected["right_unfolded_gap"])
            correlations.append(
                {
                    "model_label": model_label,
                    "q_low": low,
                    "q_high": high,
                    "triplet_count": len(ratio_selected),
                    "raw_consecutive_spacing_correlation": raw_corr,
                    "unfolded_consecutive_spacing_correlation": unfolded_corr,
                    "mean_log_spacing_ratio": float(np.mean(np.log(ratio_selected["right_raw_gap"] / ratio_selected["left_raw_gap"]))),
                    "sd_log_spacing_ratio": float(np.std(np.log(ratio_selected["right_raw_gap"] / ratio_selected["left_raw_gap"]), ddof=1)),
                    "mean_next_nearest_raw_spacing": float(np.mean(ratio_selected["left_raw_gap"] + ratio_selected["right_raw_gap"])),
                    "mean_next_nearest_unfolded_spacing": float(np.mean(ratio_selected["left_unfolded_gap"] + ratio_selected["right_unfolded_gap"])),
                }
            )
        for fold in range(5):
            subset = ratio_all.loc[ratio_all["fold"] == fold]
            subset_linked = ratio_selected.loc[ratio_selected["fold"] == fold]
            fold_raw, fold_raw_se, fold_graphs, _ = graph_equal(subset, "raw_r")
            fold_unf, fold_unf_se, _, _ = graph_equal(subset_linked, "unfolded_r")
            fold_gaps = gap_selected.loc[gap_selected["fold"] == fold]
            normalized = fold_gaps["unfolded_gap"].to_numpy(dtype=float)
            normalized = normalized[np.isfinite(normalized) & (normalized > 0)]
            normalized = normalized / np.mean(normalized) if len(normalized) else normalized
            fold_rows.append(
                {
                    "model_label": model_label,
                    "q_low": low,
                    "q_high": high,
                    "fold": fold,
                    "gap_count": len(fold_gaps),
                    "gap_graph_count": int(fold_gaps["graph"].nunique()),
                    "ratio_count": len(subset),
                    "ratio_graph_count": fold_graphs,
                    "raw_r": fold_raw,
                    "raw_r_se": fold_raw_se,
                    "same_triplet_unfolded_r": fold_unf,
                    "unfolded_r_se": fold_unf_se,
                    "GOE_spacing_KS": empirical_ks(normalized, lambda x: 1.0 - np.exp(-math.pi * np.square(x) / 4.0)),
                    "Poisson_spacing_KS": empirical_ks(normalized, lambda x: 1.0 - np.exp(-x)),
                    "selected_energy_min": float(subset["ratio_energy"].min()) if len(subset) else math.nan,
                    "selected_energy_max": float(subset["ratio_energy"].max()) if len(subset) else math.nan,
                }
            )
        for group_name, subset in ratio_all.groupby("base_or_augmentation"):
            subset_linked = subset.loc[subset["constituents_available"]]
            split_raw, split_se, split_graphs, _ = graph_equal(subset, "raw_r")
            split_unf, split_unf_se, _, _ = graph_equal(subset_linked, "unfolded_r")
            batch_rows.append(
                {
                    "model_label": model_label,
                    "q_low": low,
                    "q_high": high,
                    "split_kind": "base_vs_augmentation",
                    "split_value": group_name,
                    "ratio_count": len(subset),
                    "ratio_graph_count": split_graphs,
                    "raw_r": split_raw,
                    "raw_r_se": split_se,
                    "same_triplet_unfolded_r": split_unf,
                    "unfolded_r_se": split_unf_se,
                    "mean_ratio_energy": float(subset["ratio_energy"].mean()),
                }
            )
        current = summarize_coordinate(
            linked,
            model_label=model_label,
            coordinate="ratio_current_q",
            low=low,
            high=high,
            nominal_low=low,
            nominal_high=high,
        )
        coordinate_rows.append(current)
        coordinate_rows.append(
            summarize_coordinate(
                linked,
                model_label=model_label,
                coordinate="ratio_local_q",
                low=low,
                high=high,
                nominal_low=low,
                nominal_high=high,
            )
        )
        energy_mask = (
            (linked["ratio_energy"] >= float(published["E_min"]))
            & (linked["ratio_energy"] <= float(published["E_max"]))
            & linked["constituents_available"]
        )
        energy_selected = linked.loc[energy_mask]
        energy_raw, energy_se, energy_graphs, _ = graph_equal(energy_selected, "raw_r")
        energy_unf, energy_unf_se, _, _ = graph_equal(energy_selected, "unfolded_r")
        coordinate_rows.append(
            {
                "model_label": model_label,
                "nominal_q_low": low,
                "nominal_q_high": high,
                "coordinate": "fixed_raw_energy_from_published_gap_range",
                "actual_low": float(published["E_min"]),
                "actual_high": float(published["E_max"]),
                "ratio_count": len(energy_selected),
                "ratio_graph_count": energy_graphs,
                "graph_equal_raw_r": energy_raw,
                "raw_r_se": energy_se,
                "graph_equal_same_triplet_unfolded_r": energy_unf,
                "unfolded_r_se": energy_unf_se,
            }
        )
        ranks = ratio_all.loc[ratio_all["constituents_available"], "ratio_edge_rank"]
        rank_rows.append(
            {
                "model_label": model_label,
                "q_low": low,
                "q_high": high,
                "ratio_count_with_rank": len(ranks),
                "edge_rank_min": int(ranks.min()) if len(ranks) else "",
                "edge_rank_q25": float(ranks.quantile(0.25)) if len(ranks) else math.nan,
                "edge_rank_median": float(ranks.median()) if len(ranks) else math.nan,
                "edge_rank_q75": float(ranks.quantile(0.75)) if len(ranks) else math.nan,
                "edge_rank_max": int(ranks.max()) if len(ranks) else "",
            }
        )
        positive_gaps = gap_selected.loc[(gap_selected["raw_gap"] > 0) & np.isfinite(gap_selected["raw_gap"]), "raw_gap"].to_numpy()
        local_scale = float(np.median(positive_gaps)) if len(positive_gaps) else math.nan
        eps_scale = np.finfo(float).eps * max(
            1.0,
            float(np.max(np.abs(gap_selected[["gap_left", "gap_right"]].to_numpy()))) if len(gap_selected) else 1.0,
        )
        degeneracy_rows.append(
            {
                "model_label": model_label,
                "q_low": low,
                "q_high": high,
                "positive_gap_count": len(positive_gaps),
                "minimum_positive_gap": float(np.min(positive_gaps)) if len(positive_gaps) else math.nan,
                "gap_q0p1_percent": float(np.quantile(positive_gaps, 0.001)) if len(positive_gaps) else math.nan,
                "gap_q1_percent": float(np.quantile(positive_gaps, 0.01)) if len(positive_gaps) else math.nan,
                "gap_q5_percent": float(np.quantile(positive_gaps, 0.05)) if len(positive_gaps) else math.nan,
                "median_positive_gap": local_scale,
                "minimum_over_median_gap": float(np.min(positive_gaps) / local_scale) if len(positive_gaps) and local_scale > 0 else math.nan,
                "machine_precision_scale": eps_scale,
                "gaps_below_10_eps_scale": int(np.sum(positive_gaps <= 10 * eps_scale)),
                "gaps_below_100_eps_scale": int(np.sum(positive_gaps <= 100 * eps_scale)),
                "gaps_below_1e_minus_10_local_median": int(np.sum(positive_gaps <= 1e-10 * local_scale)) if len(positive_gaps) else 0,
            }
        )
        key = (model_label, low, high)
        if key in PRIMARY_SUSPICIOUS:
            plot_r_distribution(ratio_all, model_label, low, high, model_plot_dir)
            if len(ratio_selected):
                plot_spacing_pairs(ratio_selected, model_label, low, high, model_plot_dir)

    for rank_low, rank_high in ((1, 5), (6, 10), (11, 20), (21, 40), (41, 80)):
        selected = linked.loc[
            linked["constituents_available"]
            & (linked["ratio_edge_rank"] >= rank_low)
            & (linked["ratio_edge_rank"] <= rank_high)
        ]
        raw, raw_se, graphs, _ = graph_equal(selected, "raw_r")
        unfolded, unfolded_se, _, _ = graph_equal(selected, "unfolded_r")
        coordinate_rows.append(
            {
                "model_label": model_label,
                "nominal_q_low": "FIXED_RANK",
                "nominal_q_high": "FIXED_RANK",
                "coordinate": "fixed_positive_nonperron_ratio_edge_rank",
                "actual_low": rank_low,
                "actual_high": rank_high,
                "ratio_count": len(selected),
                "ratio_graph_count": graphs,
                "graph_equal_raw_r": raw,
                "raw_r_se": raw_se,
                "graph_equal_same_triplet_unfolded_r": unfolded,
                "unfolded_r_se": unfolded_se,
            }
        )

    invariants = {
        "model_label": model_label,
        "mapped_gap_count": len(gaps),
        "mapped_ratio_count": len(ratios),
        "linked_ratio_count": int(np.sum(available)),
        "unlinked_ratio_count_due_to_q0p995_input_truncation": int(np.sum(~available)),
        "maximum_raw_r_reconstruction_error": float(
            np.max(np.abs(linked.loc[available, "raw_r"] - linked.loc[available, "recomputed_raw_r"]))
        ),
        "all_ratios_with_constituents_are_within_one_graph": True,
        "all_constituent_edge_ranks_are_consecutive": True,
        "zero_gap_count_in_mapped_positive_tail": int(np.sum(gaps["is_zero"])),
        "ratio_category_non_positive_positive_count": int(np.sum(ratios["ratio_category"] != 2)),
        "perron_present_in_mapped_gap_endpoints": False,
    }
    (model_plot_dir / "mapped_invariants.json").write_text(json.dumps(invariants, indent=2) + "\n")
    sliding = sliding_windows(linked, model_label)
    plot_sliding(sliding, model_label, model_plot_dir)
    return {
        "reproduction": reproduction,
        "support": support,
        "raw_vs_unfolded": raw_vs_unfolded,
        "distributions": distributions,
        "correlations": correlations,
        "folds": fold_rows,
        "batches": batch_rows,
        "coordinates": coordinate_rows,
        "ranks": rank_rows,
        "degeneracy": degeneracy_rows,
        "sliding": sliding,
    }


def sliding_windows(linked: pd.DataFrame, model_label: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    available = linked["constituents_available"]
    for width in (0.0005, 0.001, 0.002):
        centers = np.arange(0.994 + width / 2, 0.9995 + 1e-12, width / 4)
        for center in centers:
            low, high = center - width / 2, center + width / 2
            selected = linked.loc[available & window_mask(linked["ratio_current_q"], low, high)]
            raw, raw_se, graphs, _ = graph_equal(selected, "raw_r")
            unfolded, unfolded_se, _, _ = graph_equal(selected, "unfolded_r")
            rows.append(
                {
                    "model_label": model_label,
                    "width": width,
                    "q_center": center,
                    "q_low": low,
                    "q_high": high,
                    "ratio_count": len(selected),
                    "ratio_graph_count": graphs,
                    "raw_r": raw,
                    "raw_r_se": raw_se,
                    "unfolded_r": unfolded,
                    "unfolded_r_se": unfolded_se,
                }
            )
    for low, high in ((0.995, 0.997), (0.995, 0.998), (0.996, 0.998)):
        selected = linked.loc[available & window_mask(linked["ratio_current_q"], low, high)]
        raw, raw_se, graphs, _ = graph_equal(selected, "raw_r")
        unfolded, unfolded_se, _, _ = graph_equal(selected, "unfolded_r")
        rows.append(
            {
                "model_label": model_label,
                "width": "MERGED",
                "q_center": 0.5 * (low + high),
                "q_low": low,
                "q_high": high,
                "ratio_count": len(selected),
                "ratio_graph_count": graphs,
                "raw_r": raw,
                "raw_r_se": raw_se,
                "unfolded_r": unfolded,
                "unfolded_r_se": unfolded_se,
            }
        )
    return rows


def plot_r_distribution(frame: pd.DataFrame, label: str, low: float, high: float, output_dir: Path) -> None:
    values = frame["raw_r"].to_numpy(dtype=float)
    x = np.linspace(0.0, 1.0, 500)
    figure, axis = plt.subplots(figsize=(6.6, 4.4))
    axis.hist(values, bins=np.linspace(0, 1, 31), density=True, color="#4c78a8", alpha=0.72, label="pooled data")
    axis.plot(x, goe_ratio_pdf(x), color="black", label="GOE ratio law")
    axis.plot(x, poisson_ratio_pdf(x), color="firebrick", ls="--", label="Poisson ratio law")
    axis.set(xlabel="adjacent-gap ratio r", ylabel="density", title=f"{label}, q=[{low:g},{high:g}); n={len(values)}")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / f"r_distribution_q_{low:g}_{high:g}.png", dpi=180)
    plt.close(figure)


def plot_spacing_pairs(frame: pd.DataFrame, label: str, low: float, high: float, output_dir: Path) -> None:
    sample = frame.sample(min(len(frame), 5000), random_state=1701)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    axes[0].scatter(sample["left_raw_gap"], sample["right_raw_gap"], s=6, alpha=0.25)
    axes[0].set(xlabel=r"raw $\Delta_i$", ylabel=r"raw $\Delta_{i+1}$")
    axes[1].scatter(sample["left_unfolded_gap"], sample["right_unfolded_gap"], s=6, alpha=0.25, color="#59a14f")
    axes[1].set(xlabel=r"unfolded $s_i$", ylabel=r"unfolded $s_{i+1}$")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle(f"{label}, q=[{low:g},{high:g}) consecutive spacings")
    figure.tight_layout()
    figure.savefig(output_dir / f"spacing_pair_scatter_q_{low:g}_{high:g}.png", dpi=180)
    plt.close(figure)


def plot_sliding(rows: list[dict[str, object]], label: str, output_dir: Path) -> None:
    frame = pd.DataFrame(rows)
    frame = frame.loc[frame["width"] != "MERGED"]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for axis, width in zip(axes, (0.0005, 0.001, 0.002)):
        selected = frame.loc[frame["width"] == width]
        supported = selected["ratio_count"] >= RATIO_MEAN_MIN_RATIOS
        axis.errorbar(selected.loc[supported, "q_center"], selected.loc[supported, "raw_r"], yerr=selected.loc[supported, "raw_r_se"], marker="o", ms=3, lw=1, label="raw r")
        axis.plot(selected.loc[supported, "q_center"], selected.loc[supported, "unfolded_r"], marker="s", ms=3, lw=1, label="same-triplet unfolded r")
        axis.axhline(GOE_MEAN_GAP_RATIO, color="black", ls="--", lw=1)
        axis.axhline(POISSON_MEAN_GAP_RATIO, color="firebrick", ls=":", lw=1)
        axis.set_title(f"width={width:g}")
        axis.set_xlabel("q center")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("graph-equal mean ratio")
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle(f"{label}: sliding-window boundary sensitivity")
    figure.tight_layout()
    figure.savefig(output_dir / "sliding_window_r.png", dpi=180)
    plt.close(figure)


def plot_masked_heatmap(support_rows: list[dict[str, object]], reproduction_rows: list[dict[str, object]], output_dir: Path) -> None:
    labels = list(MODEL_PATHS)
    values = np.full((len(labels), len(REQUESTED_INTERVALS)), np.nan)
    supported = np.zeros_like(values, dtype=bool)
    reproduction = {(row["model_label"], float(row["q_low"]), float(row["q_high"])): row for row in reproduction_rows}
    for row in support_rows:
        i = labels.index(str(row["model_label"]))
        j = list(REQUESTED_INTERVALS).index((float(row["q_low"]), float(row["q_high"])))
        values[i, j] = float(reproduction[(row["model_label"], float(row["q_low"]), float(row["q_high"]))]["recomputed_raw_r"])
        supported[i, j] = bool(row["ratio_mean_support_gate"])
    masked = np.ma.masked_where(~supported, values)
    figure, axis = plt.subplots(figsize=(12.5, 6.2))
    image = axis.imshow(masked, aspect="auto", cmap="coolwarm", vmin=0.38629, vmax=0.59)
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xticks(range(len(REQUESTED_INTERVALS)), [f"[{lo:g},{hi:g})" for lo, hi in REQUESTED_INTERVALS], rotation=35, ha="right")
    for i in range(len(labels)):
        for j in range(len(REQUESTED_INTERVALS)):
            text = f"{values[i,j]:.3f}" if supported[i,j] else f"MASK\n{values[i,j]:.3f}"
            axis.text(j, i, text, ha="center", va="center", fontsize=8, color="black")
    figure.colorbar(image, ax=axis, label="graph-equal raw r")
    axis.set_title("Diagnostic raw-r heatmap; MASK = histogram-supported but ratio-under-supported")
    figure.tight_layout()
    figure.savefig(output_dir / "diagnostic_ratio_support_masked_heatmap.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapped-root", type=Path, required=True)
    parser.add_argument("--published-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    published = [row for row in read_csv(args.published_summary) if row["analysis_kind"] == "EXACT_REQUESTED_INTERVAL"]
    published_by_key = {
        (row["model_label"], float(row["nominal_q_low"]), float(row["nominal_q_high"])): row
        for row in published
    }
    collections: dict[str, list[dict[str, object]]] = {
        name: []
        for name in (
            "reproduction", "support", "raw_vs_unfolded", "distributions", "correlations",
            "folds", "batches", "coordinates", "ranks", "degeneracy", "sliding"
        )
    }
    for label, relative in MODEL_PATHS.items():
        result = audit_model(
            model_label=label,
            mapped_path=args.mapped_root / relative / "edge_gap_and_ratio_records.npz",
            published_rows=published_by_key,
            output_dir=args.output_dir,
        )
        for name, rows in result.items():
            collections[name].extend(rows)

    reproduction = collections["reproduction"]
    for row in reproduction:
        se = float(row["recomputed_raw_r_se"])
        z = (float(row["recomputed_raw_r"]) - GOE_MEAN_GAP_RATIO) / se if se > 0 else math.nan
        row["z_from_GOE"] = z
        row["two_sided_normal_p"] = 2 * stats.norm.sf(abs(z)) if math.isfinite(z) else math.nan
    adjusted = fdr_bh(np.asarray([float(row["two_sided_normal_p"]) for row in reproduction]))
    for row, value in zip(reproduction, adjusted):
        row["BH_FDR_q"] = value
    write_rows(args.output_dir / "anomaly_reproduction.csv", reproduction)
    write_rows(args.output_dir / "gap_vs_ratio_support.csv", collections["support"])
    write_rows(args.output_dir / "raw_vs_unfolded_r.csv", collections["raw_vs_unfolded"])
    write_rows(args.output_dir / "r_distribution_statistics.csv", collections["distributions"])
    write_rows(args.output_dir / "consecutive_spacing_correlations.csv", collections["correlations"])
    write_rows(args.output_dir / "fold_sensitivity.csv", collections["folds"])
    write_rows(args.output_dir / "base_vs_augmentation.csv", collections["batches"])
    write_rows(args.output_dir / "coordinate_sensitivity.csv", collections["coordinates"])
    write_rows(args.output_dir / "rank_mapping.csv", collections["ranks"])
    write_rows(args.output_dir / "near_degeneracy_audit.csv", collections["degeneracy"])
    write_rows(args.output_dir / "sliding_window_r.csv", collections["sliding"])
    plot_masked_heatmap(collections["support"], reproduction, args.output_dir)
    manifest = {
        "audit_version": AUDIT_VERSION,
        "authoritative_results_modified": False,
        "mapped_root": str(args.mapped_root),
        "published_summary": str(args.published_summary),
        "histogram_support_gate": {"positive_gaps": HISTOGRAM_MIN_GAPS, "graphs": HISTOGRAM_MIN_GRAPHS},
        "diagnostic_ratio_mean_support_gate": {"ratios": RATIO_MEAN_MIN_RATIOS, "graphs": RATIO_MEAN_MIN_GRAPHS},
        "diagnostic_ratio_distribution_support_gate": {"ratios": RATIO_DISTRIBUTION_MIN_RATIOS, "graphs": RATIO_DISTRIBUTION_MIN_GRAPHS},
        "ratio_constituent_linking": "exact equality of middle eigenvalue to left gap right endpoint and right gap left endpoint within the same graph",
        "input_truncation_note": "mapped records retain q>=0.995; constituent diagnostics in the first cell can omit ratios whose inner gap midpoint lies below 0.995",
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(args.output_dir)


if __name__ == "__main__":
    main()
