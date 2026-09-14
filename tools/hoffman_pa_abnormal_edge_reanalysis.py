#!/usr/bin/env python3
"""Reanalyze all cached N=5000 priority-model spectra in an isolated namespace."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from small_world_mobility_edge.preferential_attachment.abnormal_edge_followup import (
    FOLLOWUP_ANALYSIS_VERSION,
    GOE_MEAN_GAP_RATIO,
    MODEL_BY_KEY,
    POISSON_MEAN_GAP_RATIO,
    FollowupModel,
    build_positive_edge_gap_records,
    build_positive_edge_same_triplet_records,
    goe_ratio_cdf,
    graph_bootstrap_ci,
    graph_equal_summary,
    ks_distance_unit_mean_spacing,
    poisson_ratio_cdf,
    stable_seed,
    summarize_fixed_rank_dataset,
    summarize_q_window,
)
from small_world_mobility_edge.preferential_attachment.extreme_edge_quantiles import (
    ModelSpecification,
    config_from_manifest,
    load_verified_augmentation_records,
    load_verified_records,
)
from small_world_mobility_edge.preferential_attachment.mobility_edge_postprocess import (
    build_cross_fitted_models,
    write_rows,
)


STANDARD_WINDOWS = (
    (0.995, 0.996),
    (0.996, 0.997),
    (0.997, 0.998),
    (0.995, 0.997),
    (0.995, 0.998),
)
ISOLATED_WINDOWS = {
    "affine_alpha_1p05": ((0.9999, 0.99995),),
    "affine_alpha_1p25": ((0.99995, 1.0),),
}
SLIDING_WIDTHS = (0.0005, 0.001, 0.002)
EDGE_ARRAY_COUNT = 121


def _model_specification(model: FollowupModel) -> ModelSpecification:
    return ModelSpecification(
        model.model_family,
        model.parameter_name,
        model.parameter_value,
    )


def _subset(table: Mapping[str, np.ndarray], keep: np.ndarray) -> dict[str, np.ndarray]:
    return {name: np.asarray(values)[keep] for name, values in table.items()}


def _csv_gz(path: Path, table: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(table)
    lengths = {len(np.asarray(table[name])) for name in names}
    if len(lengths) != 1:
        raise ValueError(f"inconsistent record lengths for {path}")
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(names)
        for values in zip(*(np.asarray(table[name]) for name in names)):
            writer.writerow(values)


def _plot_ratio_distribution(
    triplets: Mapping[str, np.ndarray],
    *,
    model: FollowupModel,
    q_low: float,
    q_high: float,
    output_path: Path,
) -> None:
    mask = (triplets["ensemble_q"] >= q_low) & (
        triplets["ensemble_q"] <= q_high
        if q_high >= 1.0 - 1e-15
        else triplets["ensemble_q"] < q_high
    )
    raw = triplets["raw_r"][mask]
    unfolded = triplets["unfolded_r"][mask]
    raw = raw[np.isfinite(raw)]
    unfolded = unfolded[np.isfinite(unfolded)]
    x = np.linspace(0.0, 1.0, 1001)
    goe_pdf = (
        27.0 / 4.0 * (x + x * x) / np.power(1.0 + x + x * x, 2.5)
    )
    poisson_pdf = 2.0 / np.square(1.0 + x)
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), sharex=True, sharey=True)
    for axis, values, name, color in (
        (axes[0], raw, "raw", "#457b9d"),
        (axes[1], unfolded, "same-triplet CDF unfolded", "#52b788"),
    ):
        if len(values):
            axis.hist(
                values,
                bins=np.linspace(0.0, 1.0, 31),
                density=True,
                color=color,
                alpha=0.76,
            )
        axis.plot(x, goe_pdf, color="black", lw=1.2, label="GOE ratio")
        axis.plot(x, poisson_pdf, color="firebrick", ls="--", lw=1.2, label="Poisson ratio")
        axis.set_title(f"{name}; n={len(values):,}")
        axis.set_xlabel("adjacent-gap ratio r")
        axis.grid(alpha=0.18)
    axes[0].set_ylabel("density")
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle(f"{model.label}, q=[{q_low:g}, {q_high:g})")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_spacing_pairs(
    triplets: Mapping[str, np.ndarray],
    *,
    model: FollowupModel,
    q_low: float,
    q_high: float,
    output_path: Path,
) -> None:
    mask = (triplets["ensemble_q"] >= q_low) & (
        triplets["ensemble_q"] <= q_high
        if q_high >= 1.0 - 1e-15
        else triplets["ensemble_q"] < q_high
    )
    indices = np.flatnonzero(mask)
    if len(indices) > 20000:
        generator = np.random.default_rng(
            stable_seed(model.key, q_low, q_high, "spacing_scatter")
        )
        indices = np.sort(generator.choice(indices, size=20000, replace=False))
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    for axis, left_name, right_name, label, color in (
        (
            axes[0],
            "raw_left_gap",
            "raw_right_gap",
            "raw gaps",
            "#457b9d",
        ),
        (
            axes[1],
            "unfolded_left_gap",
            "unfolded_right_gap",
            "CDF-unfolded gaps",
            "#52b788",
        ),
    ):
        left = triplets[left_name][indices]
        right = triplets[right_name][indices]
        finite = np.isfinite(left) & np.isfinite(right)
        axis.scatter(left[finite], right[finite], s=4, alpha=0.18, color=color, rasterized=True)
        if np.any(finite):
            limit = float(np.quantile(np.concatenate((left[finite], right[finite])), 0.995))
            axis.plot([0.0, limit], [0.0, limit], color="black", lw=1.0, ls="--")
            axis.set_xlim(0.0, limit)
            axis.set_ylim(0.0, limit)
        axis.set_xlabel(r"$s_i$")
        axis.set_ylabel(r"$s_{i+1}$")
        axis.set_title(label)
        axis.grid(alpha=0.18)
    figure.suptitle(f"{model.label}, q=[{q_low:g}, {q_high:g})")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_rank_distribution(
    triplets: Mapping[str, np.ndarray],
    *,
    model: FollowupModel,
    q_low: float,
    q_high: float,
    output_path: Path,
) -> None:
    mask = (triplets["ensemble_q"] >= q_low) & (
        triplets["ensemble_q"] <= q_high
        if q_high >= 1.0 - 1e-15
        else triplets["ensemble_q"] < q_high
    )
    ranks = triplets["edge_rank"][mask]
    figure, axis = plt.subplots(figsize=(7.4, 4.2))
    if len(ranks):
        low = max(1, int(np.min(ranks)))
        high = int(np.max(ranks))
        axis.hist(ranks, bins=np.arange(low - 0.5, high + 1.5), color="#6a4c93", alpha=0.8)
    axis.set_xlabel("graph-local positive-edge rank of triplet middle eigenvalue")
    axis.set_ylabel("ratio count")
    axis.set_title(f"{model.label}, q=[{q_low:g}, {q_high:g})")
    axis.grid(alpha=0.18)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _sliding_summary(
    triplets: Mapping[str, np.ndarray],
    gaps: Mapping[str, np.ndarray],
    *,
    model: FollowupModel,
    width: float,
    center: float,
) -> dict[str, object]:
    q_low = center - 0.5 * width
    q_high = center + 0.5 * width
    ratio_mask = (triplets["ensemble_q"] >= q_low) & (triplets["ensemble_q"] < q_high)
    graph_ids = triplets["graph_index"][ratio_mask].astype(int)
    raw = triplets["raw_r"][ratio_mask]
    unfolded = triplets["unfolded_r"][ratio_mask]
    valid = np.isfinite(raw) & np.isfinite(unfolded)
    graph_ids = graph_ids[valid]
    raw = raw[valid]
    unfolded = unfolded[valid]
    raw_mean, raw_se, raw_graphs, _, _ = graph_equal_summary(graph_ids, raw)
    unfolded_mean, unfolded_se, unfolded_graphs, _, _ = graph_equal_summary(
        graph_ids, unfolded
    )
    raw_ci = graph_bootstrap_ci(
        graph_ids,
        raw,
        seed=stable_seed(model.key, width, center, "sliding_raw"),
        replicates=500,
    )
    unfolded_ci = graph_bootstrap_ci(
        graph_ids,
        unfolded,
        seed=stable_seed(model.key, width, center, "sliding_unfolded"),
        replicates=500,
    )
    gap_mask = (gaps["ensemble_q"] >= q_low) & (gaps["ensemble_q"] < q_high)
    raw_spacing = gaps["raw_gap"][gap_mask]
    unfolded_spacing = gaps["unfolded_gap"][gap_mask]
    raw_spacing = raw_spacing[np.isfinite(raw_spacing) & (raw_spacing > 1e-14)]
    unfolded_spacing = unfolded_spacing[
        np.isfinite(unfolded_spacing) & (unfolded_spacing > 1e-14)
    ]
    raw_spacing = raw_spacing / np.mean(raw_spacing) if len(raw_spacing) else raw_spacing
    unfolded_spacing = (
        unfolded_spacing / np.mean(unfolded_spacing)
        if len(unfolded_spacing)
        else unfolded_spacing
    )
    support_ok = len(raw) >= 100 and raw_graphs >= 50
    return {
        "analysis_version": FOLLOWUP_ANALYSIS_VERSION,
        "model_key": model.key,
        "model": model.label,
        "window_width": width,
        "q_center": center,
        "q_low": q_low,
        "q_high": q_high,
        "positive_gaps": int(len(raw_spacing)),
        "valid_ratios": int(len(raw)),
        "ratio_contributing_graphs": raw_graphs,
        "raw_r": raw_mean,
        "raw_r_se": raw_se,
        "raw_r_bootstrap_ci_low": raw_ci[0],
        "raw_r_bootstrap_ci_high": raw_ci[1],
        "unfolded_r": unfolded_mean,
        "unfolded_r_se": unfolded_se,
        "unfolded_r_bootstrap_ci_low": unfolded_ci[0],
        "unfolded_r_bootstrap_ci_high": unfolded_ci[1],
        "unfolded_ratio_contributing_graphs": unfolded_graphs,
        "raw_spacing_KS_to_GOE": ks_distance_unit_mean_spacing(raw_spacing, "goe"),
        "raw_spacing_KS_to_Poisson": ks_distance_unit_mean_spacing(raw_spacing, "poisson"),
        "unfolded_spacing_KS_to_GOE": ks_distance_unit_mean_spacing(
            unfolded_spacing, "goe"
        ),
        "unfolded_spacing_KS_to_Poisson": ks_distance_unit_mean_spacing(
            unfolded_spacing, "poisson"
        ),
        "support_status": "OK" if support_ok else "LOW_RATIO_SUPPORT",
    }


def _plot_sliding(rows: Sequence[dict[str, object]], model: FollowupModel, path: Path) -> None:
    figure, axes = plt.subplots(len(SLIDING_WIDTHS), 1, figsize=(9.0, 9.3), sharex=True)
    for axis, width in zip(axes, SLIDING_WIDTHS):
        selected = [row for row in rows if math.isclose(float(row["window_width"]), width)]
        q = np.asarray([row["q_center"] for row in selected], dtype=float)
        raw = np.asarray([row["raw_r"] for row in selected], dtype=float)
        raw_low = np.asarray([row["raw_r_bootstrap_ci_low"] for row in selected], dtype=float)
        raw_high = np.asarray([row["raw_r_bootstrap_ci_high"] for row in selected], dtype=float)
        unfolded = np.asarray([row["unfolded_r"] for row in selected], dtype=float)
        unfolded_low = np.asarray(
            [row["unfolded_r_bootstrap_ci_low"] for row in selected], dtype=float
        )
        unfolded_high = np.asarray(
            [row["unfolded_r_bootstrap_ci_high"] for row in selected], dtype=float
        )
        ok = np.asarray([row["support_status"] == "OK" for row in selected])
        axis.plot(q[ok], raw[ok], color="#457b9d", label="raw r")
        axis.fill_between(q[ok], raw_low[ok], raw_high[ok], color="#457b9d", alpha=0.16)
        axis.plot(q[ok], unfolded[ok], color="#2a9d8f", label="same-triplet unfolded r")
        axis.fill_between(
            q[ok], unfolded_low[ok], unfolded_high[ok], color="#2a9d8f", alpha=0.16
        )
        axis.axhline(GOE_MEAN_GAP_RATIO, color="black", ls="--", lw=1.0, label="GOE")
        axis.axhline(POISSON_MEAN_GAP_RATIO, color="firebrick", ls=":", lw=1.0, label="Poisson")
        axis.set_ylabel("graph-equal r")
        axis.set_title(f"width={width:g}")
        axis.grid(alpha=0.18)
    axes[0].legend(frameon=False, ncol=4, fontsize=8)
    axes[-1].set_xlabel("held-out ensemble CDF quantile q")
    figure.suptitle(f"{model.label}: overlapping positive-edge windows")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _fixed_rank_dataset(records, models, cache_paths: Sequence[Path], config) -> dict[str, np.ndarray]:
    graph_indices = np.asarray([record.graph_index for record in records], dtype=np.int32)
    graph_seeds = np.asarray([record.graph_seed for record in records], dtype=np.uint64)
    position = {int(graph_id): index for index, graph_id in enumerate(graph_indices)}
    shape = (len(records), EDGE_ARRAY_COUNT)
    output = {
        "graph_indices": graph_indices,
        "graph_seeds": graph_seeds,
        "positive_eigenvalue": np.full(shape, np.nan),
        "negative_eigenvalue": np.full(shape, np.nan),
        "positive_ensemble_q": np.full(shape, np.nan),
        "negative_ensemble_q": np.full(shape, np.nan),
        "positive_n_times_ipr": np.full(shape, np.nan),
        "negative_n_times_ipr": np.full(shape, np.nan),
        "positive_participation_fraction": np.full(shape, np.nan),
        "negative_participation_fraction": np.full(shape, np.nan),
        "positive_c10": np.full(shape, np.nan),
        "negative_c10": np.full(shape, np.nan),
        "positive_c50": np.full(shape, np.nan),
        "negative_c50": np.full(shape, np.nan),
        "positive_c200": np.full(shape, np.nan),
        "negative_c200": np.full(shape, np.nan),
        "degree_rank": np.full(shape, np.nan),
        "maximum_degree": np.full(len(records), np.nan),
        "positive_cdf_unfolded_gap": np.full((len(records), EDGE_ARRAY_COUNT - 1), np.nan),
        "negative_cdf_unfolded_gap": np.full((len(records), EDGE_ARRAY_COUNT - 1), np.nan),
    }
    model_mapping = {}
    for fold_model in models:
        for graph_id in fold_model.held_out_graph_ids:
            model_mapping[int(graph_id)] = fold_model
    for row, record in enumerate(records):
        positive = record.continuous_eigenvalues[record.continuous_eigenvalues > 0.0][::-1]
        negative = record.continuous_eigenvalues[record.continuous_eigenvalues < 0.0]
        if len(positive) < EDGE_ARRAY_COUNT or len(negative) < EDGE_ARRAY_COUNT:
            raise ValueError(f"graph {record.graph_index} lacks 121 modes on an edge")
        output["positive_eigenvalue"][row] = positive[:EDGE_ARRAY_COUNT]
        output["negative_eigenvalue"][row] = negative[:EDGE_ARRAY_COUNT]
        density = model_mapping[record.graph_index].density_model
        positive_cdf = np.interp(
            positive[:EDGE_ARRAY_COUNT], density.grid, density.cdf, left=0.0, right=1.0
        )
        negative_cdf = np.interp(
            negative[:EDGE_ARRAY_COUNT], density.grid, density.cdf, left=0.0, right=1.0
        )
        output["positive_ensemble_q"][row] = positive_cdf
        output["negative_ensemble_q"][row] = 1.0 - negative_cdf
        output["positive_cdf_unfolded_gap"][row] = record.continuous_dimension * (
            positive_cdf[:-1] - positive_cdf[1:]
        )
        output["negative_cdf_unfolded_gap"][row] = record.continuous_dimension * np.diff(
            negative_cdf
        )
    for cache_path in cache_paths:
        with np.load(cache_path, allow_pickle=False) as data:
            indices = np.asarray(data["graph_indices"], dtype=int)
            if "degree_arrays" in data.files:
                degrees = np.asarray(data["degree_arrays"], dtype=float)
                for local, graph_id in enumerate(indices):
                    row = position[int(graph_id)]
                    ordered = np.sort(degrees[local])[::-1]
                    output["degree_rank"][row] = ordered[:EDGE_ARRAY_COUNT]
                    output["maximum_degree"][row] = ordered[0]
            required = {
                "eigenpair_offsets",
                "eigenpair_eigenvalue",
                "eigenpair_n_times_ipr",
                "eigenpair_participation_fraction",
                "eigenpair_top_10_mass",
                "eigenpair_top_50_mass",
                "eigenpair_top_200_mass",
            }
            if not required.issubset(data.files):
                continue
            offsets = np.asarray(data["eigenpair_offsets"], dtype=int)
            energies = np.asarray(data["eigenpair_eigenvalue"], dtype=float)
            nipr = np.asarray(data["eigenpair_n_times_ipr"], dtype=float)
            participation = np.asarray(
                data["eigenpair_participation_fraction"], dtype=float
            )
            c10 = np.asarray(data["eigenpair_top_10_mass"], dtype=float)
            c50 = np.asarray(data["eigenpair_top_50_mass"], dtype=float)
            c200 = np.asarray(data["eigenpair_top_200_mass"], dtype=float)
            for local, graph_id in enumerate(indices):
                row = position[int(graph_id)]
                left, right = offsets[local : local + 2]
                graph_energy = energies[left:right]
                graph_nipr = nipr[left:right]
                graph_participation = participation[left:right]
                graph_c10 = c10[left:right]
                graph_c50 = c50[left:right]
                graph_c200 = c200[left:right]
                positive = np.flatnonzero(graph_energy > 0.0)[::-1][:EDGE_ARRAY_COUNT]
                negative = np.flatnonzero(graph_energy < 0.0)[:EDGE_ARRAY_COUNT]
                output["positive_n_times_ipr"][row] = graph_nipr[positive]
                output["negative_n_times_ipr"][row] = graph_nipr[negative]
                output["positive_participation_fraction"][row] = graph_participation[
                    positive
                ]
                output["negative_participation_fraction"][row] = graph_participation[
                    negative
                ]
                output["positive_c10"][row] = graph_c10[positive]
                output["negative_c10"][row] = graph_c10[negative]
                output["positive_c50"][row] = graph_c50[positive]
                output["negative_c50"][row] = graph_c50[negative]
                output["positive_c200"][row] = graph_c200[positive]
                output["negative_c200"][row] = graph_c200[negative]
    replay_dir = config.parameter_dir / "structural" / "replay_partials"
    if replay_dir.is_dir():
        for replay_path in sorted(replay_dir.glob("degree_arrays_*.npz")):
            with np.load(replay_path, allow_pickle=False) as data:
                indices = np.asarray(data["graph_indices"], dtype=int)
                degrees = np.asarray(data["degree_arrays"], dtype=float)
            for local, graph_id in enumerate(indices):
                if int(graph_id) not in position:
                    continue
                row = position[int(graph_id)]
                ordered = np.sort(degrees[local])[::-1]
                output["degree_rank"][row] = ordered[:EDGE_ARRAY_COUNT]
                output["maximum_degree"][row] = ordered[0]
    return output


def _localization_by_q(dataset: Mapping[str, np.ndarray], triplets, q_low, q_high) -> tuple[float, int]:
    ranks = triplets["edge_rank"]
    graph_ids = triplets["graph_index"].astype(int)
    q = triplets["ensemble_q"]
    keep = (q >= q_low) & (q <= q_high if q_high >= 1.0 - 1e-15 else q < q_high)
    position = {int(value): index for index, value in enumerate(dataset["graph_indices"])}
    values = []
    ids = []
    for graph_id, rank in zip(graph_ids[keep], ranks[keep]):
        if graph_id not in position or not 1 <= rank <= EDGE_ARRAY_COUNT:
            continue
        value = dataset["positive_n_times_ipr"][position[graph_id], int(rank) - 1]
        if np.isfinite(value):
            ids.append(graph_id)
            values.append(value)
    if not values:
        return math.nan, 0
    mean, _, count, _, _ = graph_equal_summary(np.asarray(ids), np.asarray(values))
    return mean, count


def _provisional_classification(row: Mapping[str, object]) -> str:
    ratios = int(row["ratios"])
    graphs = int(row["ratio_contributing_graphs"])
    if ratios < 100 or graphs < 50:
        return "LOW_RATIO_SUPPORT"
    raw = float(row["raw_r"])
    unfolded = float(row["unfolded_r"])
    tags = []
    if raw < GOE_MEAN_GAP_RATIO - 0.01 and unfolded - raw > 0.015:
        tags.append("CDF_OR_DENSITY_EFFECT")
    if unfolded < GOE_MEAN_GAP_RATIO - 0.01:
        tags.append("ROBUST_INTERMEDIATE_NON_GOE_BAND")
    if not tags:
        tags.append("NOT_REPRODUCED")
    tags.append("UNRESOLVED_N_SCALING")
    return ";".join(tags)


def run_reanalysis(source_root: Path, output_root: Path, model: FollowupModel) -> Path:
    specification = _model_specification(model)
    config = config_from_manifest(source_root, specification)
    base_records, base_cache_paths, base_sources = load_verified_records(config)
    augmentation_root = source_root / "extreme_positive_edge_quantiles"
    augmentation_records, augmentation_paths, augmentation_sources = (
        load_verified_augmentation_records(
            config,
            specification,
            augmentation_root,
            base_graph_count=config.num_graphs,
        )
    )
    records = [*base_records, *augmentation_records]
    graph_indices = [record.graph_index for record in records]
    if graph_indices != list(range(len(records))):
        raise ValueError("combined base/augmentation records are not contiguous")
    print(f"{model.key}: fitting five training-only CDF folds for {len(records)} spectra", flush=True)
    crossfit_models = build_cross_fitted_models(records, num_folds=5)
    triplets = build_positive_edge_same_triplet_records(records, crossfit_models)
    gaps = build_positive_edge_gap_records(records, crossfit_models)
    fixed_dataset = _fixed_rank_dataset(records, crossfit_models, base_cache_paths, config)

    cached_dir = output_root / "cached_n5000_reanalysis" / model.key
    raw_dir = output_root / "raw_vs_unfolded_r"
    distribution_dir = output_root / "p_r_distributions" / model.key
    pair_dir = output_root / "spacing_pair_correlations" / model.key
    sliding_dir = output_root / "sliding_windows"
    fixed_dir = output_root / "fixed_rank"
    rank_dir = output_root / "quantile_to_rank" / model.key
    replication_dir = output_root / "base_augmentation_replication"
    near_dir = output_root / "solver_validation"
    for directory in (
        cached_dir,
        raw_dir,
        distribution_dir,
        pair_dir,
        sliding_dir,
        fixed_dir,
        rank_dir,
        replication_dir,
        near_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(cached_dir / "positive_edge_triplets.npz", **triplets)
    np.savez_compressed(cached_dir / "positive_edge_gaps.npz", **gaps)
    np.savez_compressed(cached_dir / "fixed_rank_dataset.npz", **fixed_dataset)
    _csv_gz(raw_dir / f"{model.key}_same_triplet_records.csv.gz", triplets)

    windows = [*STANDARD_WINDOWS, *ISOLATED_WINDOWS.get(model.key, ())]
    summaries = []
    for q_low, q_high in windows:
        row = summarize_q_window(
            triplets,
            gaps,
            model=model,
            q_low=q_low,
            q_high=q_high,
        )
        nipr, nipr_graphs = _localization_by_q(
            fixed_dataset, triplets, q_low, q_high
        )
        row["NIPR"] = nipr
        row["NIPR_graphs"] = nipr_graphs
        row["classification"] = _provisional_classification(row)
        summaries.append(row)
        slug = f"q_{q_low:g}_{q_high:g}".replace(".", "p")
        _plot_ratio_distribution(
            triplets,
            model=model,
            q_low=q_low,
            q_high=q_high,
            output_path=distribution_dir / f"p_r_{slug}.png",
        )
        _plot_spacing_pairs(
            triplets,
            model=model,
            q_low=q_low,
            q_high=q_high,
            output_path=pair_dir / f"spacing_pairs_{slug}.png",
        )
        _plot_rank_distribution(
            triplets,
            model=model,
            q_low=q_low,
            q_high=q_high,
            output_path=rank_dir / f"rank_distribution_{slug}.png",
        )
    write_rows(cached_dir / "historical_band_table.csv", summaries)

    sliding_rows = []
    for width in SLIDING_WIDTHS:
        step = width / 5.0
        centers = np.arange(0.992 + width / 2.0, 1.0 - width / 2.0 + step / 2.0, step)
        for center in centers:
            sliding_rows.append(
                _sliding_summary(
                    triplets,
                    gaps,
                    model=model,
                    width=width,
                    center=float(center),
                )
            )
    write_rows(sliding_dir / f"{model.key}_sliding_windows.csv", sliding_rows)
    _plot_sliding(sliding_rows, model, sliding_dir / f"{model.key}_r_q.png")

    fixed_rows = summarize_fixed_rank_dataset(
        fixed_dataset,
        model=model,
        n=5000,
    )
    write_rows(fixed_dir / f"{model.key}_n5000_fixed_rank.csv", fixed_rows)

    replication_rows = []
    block_size = 500
    for block_start in range(0, len(records), block_size):
        block_stop = min(block_start + block_size, len(records))
        triplet_keep = (triplets["graph_index"] >= block_start) & (
            triplets["graph_index"] < block_stop
        )
        gap_keep = (gaps["graph_index"] >= block_start) & (
            gaps["graph_index"] < block_stop
        )
        for q_low, q_high in windows:
            row = summarize_q_window(
                _subset(triplets, triplet_keep),
                _subset(gaps, gap_keep),
                model=model,
                q_low=q_low,
                q_high=q_high,
                bootstrap_replicates=1000,
            )
            row["block_start"] = block_start
            row["block_stop_exclusive"] = block_stop
            row["block_kind"] = "BASE" if block_start == 0 else "AUGMENTATION"
            replication_rows.append(row)
    write_rows(replication_dir / f"{model.key}_base_augmentation.csv", replication_rows)

    near_rows = []
    for row in summaries:
        near_rows.append(
            {
                key: row[key]
                for key in (
                    "analysis_version",
                    "model_key",
                    "model",
                    "q_low",
                    "q_high",
                    "gaps",
                    "ratios",
                    "smallest_positive_gap",
                    "relative_gap_below_1e_minus_8",
                    "relative_gap_below_1e_minus_10",
                    "relative_gap_below_1e_minus_12",
                )
            }
        )
    write_rows(near_dir / f"{model.key}_n5000_near_degeneracy.csv", near_rows)

    source_rows = [
        {"source_kind": "BASE", **row} for row in base_sources
    ] + [{"source_kind": "AUGMENTATION", **row} for row in augmentation_sources]
    write_rows(cached_dir / "source_cache_checksums.csv", source_rows)
    manifest = {
        "analysis_version": FOLLOWUP_ANALYSIS_VERSION,
        "authoritative_results_modified": False,
        "model_key": model.key,
        "model_label": model.label,
        "N": 5000,
        "m": 10,
        "base_graph_count": len(base_records),
        "augmentation_graph_count": len(augmentation_records),
        "total_graph_count": len(records),
        "crossfit_folds": 5,
        "same_triplet_rule": "raw and unfolded ratios use the exact same three eigenvalues",
        "fixed_rank_ratio_rule": "ratio assigned to middle eigenvalue edge rank; rank 1 undefined",
        "sge_job_id": os.environ.get("JOB_ID", ""),
        "sge_task_id": os.environ.get("SGE_TASK_ID", ""),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "files": sorted(
            str(path.relative_to(output_root))
            for path in output_root.rglob(f"*{model.key}*")
            if path.is_file()
        ),
    }
    (cached_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"{model.key}: reanalysis complete at {cached_dir}", flush=True)
    return cached_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-key", choices=sorted(MODEL_BY_KEY), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_reanalysis(args.source_root, args.output_root, MODEL_BY_KEY[args.model_key])


if __name__ == "__main__":
    main()
