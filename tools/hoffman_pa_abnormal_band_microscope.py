#!/usr/bin/env python3
"""Fine-grid microscope for the cached preferential-attachment edge anomaly.

This isolated diagnostic postprocessor consumes cache-only products written by
the abnormal-edge reanalysis and never writes inside authoritative model paths.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from small_world_mobility_edge.preferential_attachment.abnormal_edge_followup import (
    GOE_MEAN_GAP_RATIO,
    MODEL_BY_KEY,
    POISSON_MEAN_GAP_RATIO,
    FollowupModel,
    graph_equal_summary,
    stable_seed,
    summarize_fixed_rank_dataset,
    summarize_q_window,
)
from small_world_mobility_edge.preferential_attachment.mobility_edge_postprocess import (
    write_rows,
)

VERSION = "pa_abnormal_band_microscope_v1"
MODEL_DIR = {
    "affine_alpha_2p0": "alpha_2p0",
    "affine_alpha_1p25": "alpha_1p25",
    "affine_alpha_1p05": "alpha_1p05",
    "superlinear_zeta_1p5": "zeta_1p50",
}
BASE_LEVELS = {
    "dq_0p0002": (0.9940, 0.9990, 0.0002),
    "dq_0p0001": (0.9945, 0.9985, 0.0001),
}
FOCUSED = {
    "affine_alpha_1p05": {"dq_0p00005": (0.9950, 0.9975, 0.00005)},
    "superlinear_zeta_1p5": {"dq_0p00005": (0.9950, 0.9985, 0.00005)},
}
SLIDING_WIDTHS = (0.0002, 0.0005, 0.001)
SLIDING_STEP = 0.00005


def mask(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return (values >= low) & (values < high)


def make_bins(low: float, high: float, width: float, shifted: bool = False):
    start = low + (width / 2 if shifted else 0.0)
    count = int(math.floor((high - start) / width + 1e-9))
    bins = []
    for index in range(count):
        left = start + index * width
        right = left + width
        if right <= high + 1e-12:
            bins.append((round(left, 10), round(right, 10)))
    if not shifted:
        expected = int(round((high - low) / width))
        if len(bins) != expected or not math.isclose(bins[-1][1], high, abs_tol=1e-10):
            raise ValueError("invalid standard grid")
    return bins


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def subset(table: Mapping[str, np.ndarray], keep: np.ndarray):
    return {name: np.asarray(values)[keep] for name, values in table.items()}


def ratio(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    maximum = np.maximum(left, right)
    output = np.full(np.broadcast_shapes(left.shape, right.shape), np.nan)
    valid = np.isfinite(left) & np.isfinite(right) & (left > 1e-14) & (right > 1e-14)
    output[valid] = np.minimum(left[valid], right[valid]) / maximum[valid]
    return output


def localization(
    dataset: Mapping[str, np.ndarray],
    triplets: Mapping[str, np.ndarray],
    low: float,
    high: float,
    side: str,
):
    keep = mask(triplets["ensemble_q"], low, high)
    graph_ids = triplets["graph_index"][keep].astype(int)
    ranks = triplets["edge_rank"][keep].astype(int)
    positions = {int(value): index for index, value in enumerate(dataset["graph_indices"])}
    field_map = {
        "NIPR": f"{side}_n_times_ipr",
        "PR_over_N": f"{side}_participation_fraction",
        "C10": f"{side}_c10",
        "C50": f"{side}_c50",
        "C200": f"{side}_c200",
    }
    output = {}
    for label, key in field_map.items():
        matrix = np.asarray(dataset.get(key, np.empty((0, 0))), dtype=float)
        ids, values = [], []
        if matrix.ndim == 2:
            for graph_id, rank_value in zip(graph_ids, ranks):
                row = positions.get(int(graph_id))
                column = int(rank_value) - 1
                if row is None or not 0 <= column < matrix.shape[1]:
                    continue
                value = matrix[row, column]
                if np.isfinite(value):
                    ids.append(graph_id)
                    values.append(value)
        if values:
            mean, se, count, _, _ = graph_equal_summary(np.asarray(ids), np.asarray(values))
        else:
            mean, se, count = math.nan, math.nan, 0
        output[label] = mean
        output[f"{label}_se"] = se
        output[f"{label}_graphs"] = count
    return output


def summarize_bins(triplets, gaps, dataset, model, bins, level, grid, side="positive", reps=1000):
    rows = []
    for low, high in bins:
        row = summarize_q_window(
            triplets, gaps, model=model, q_low=low, q_high=high,
            bootstrap_replicates=reps,
        )
        row.update({
            "microscope_version": VERSION,
            "resolution_level": level,
            "grid": grid,
            "edge_side": side,
            "q_center": (low + high) / 2,
            "dq": high - low,
        })
        row.update(localization(dataset, triplets, low, high, side))
        rows.append(row)
    return rows


def references(x):
    return (
        (math.pi / 2) * x * np.exp(-math.pi * x * x / 4),
        np.exp(-x),
    )


def spacing_histogram(gaps, low, high, title, path):
    keep = mask(gaps["ensemble_q"], low, high)
    raw = gaps["raw_gap"][keep]
    unfolded = gaps["unfolded_gap"][keep]
    raw = raw[np.isfinite(raw) & (raw > 1e-14)]
    unfolded = unfolded[np.isfinite(unfolded) & (unfolded > 1e-14)]
    raw = raw / np.mean(raw) if len(raw) else raw
    unfolded = unfolded / np.mean(unfolded) if len(unfolded) else unfolded
    x = np.linspace(0, 3.5, 501)
    goe, poisson = references(x)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharex=True, sharey=True)
    for axis, values, label, color in (
        (axes[0], raw, "raw spacing", "#457b9d"),
        (axes[1], unfolded, "cross-fitted unfolded spacing", "#2a9d8f"),
    ):
        axis.hist(values, bins=np.linspace(0, 3.5, 29), density=True, color=color, alpha=.74)
        axis.plot(x, goe, color="black", lw=1.2, label="GOE")
        axis.plot(x, poisson, color="firebrick", lw=1.2, ls="--", label="Poisson")
        axis.set_title(f"{label}; n={len(values):,}")
        axis.set_xlabel("unit-mean spacing")
        axis.grid(alpha=.16)
    axes[0].set_ylabel("density")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def histogram_panel(gaps, rows, model, level, path):
    columns = 5
    row_count = int(math.ceil(len(rows) / columns))
    fig, axes = plt.subplots(
        row_count, columns, figsize=(15, max(6.5, 2.15 * row_count)),
        sharex=True, sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()
    x = np.linspace(0, 3.5, 301)
    goe, poisson = references(x)
    for axis, row in zip(axes, rows):
        low, high = float(row["q_low"]), float(row["q_high"])
        keep = mask(gaps["ensemble_q"], low, high)
        values = gaps["unfolded_gap"][keep]
        values = values[np.isfinite(values) & (values > 1e-14)]
        values = values / np.mean(values) if len(values) else values
        if bool(row["histogram_support"]):
            axis.hist(
                values, bins=np.linspace(0, 3.5, 22), density=True,
                color="#2a9d8f", alpha=.72,
            )
        else:
            axis.set_facecolor("#eeeeee")
        axis.plot(x, goe, color="black", lw=.8)
        axis.plot(x, poisson, color="firebrick", ls="--", lw=.7)
        axis.set_title(
            f"{low:.5f}-{high:.5f}\nr={float(row['unfolded_r']):.3f}; "
            f"n={int(row['positive_gap_count'])}",
            fontsize=7,
        )
        axis.tick_params(labelsize=6)
        axis.grid(alpha=.12)
    for axis in axes[len(rows):]:
        axis.axis("off")
    fig.supxlabel("unit-mean unfolded spacing")
    fig.supylabel("density")
    fig.suptitle(f"{model.label}: {level} unfolded-spacing histogram panel")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def grid_curves(standard, shifted, model, level, path):
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 7), sharex=True)
    for axis, field, label in (
        (axes[0], "raw_r", "raw adjacent-gap ratio"),
        (axes[1], "unfolded_r", "same-triplet unfolded ratio"),
    ):
        for rows, name, color, marker in (
            (standard, "standard", "#1d4ed8", "o"),
            (shifted, "half-bin shifted", "#d97706", "s"),
        ):
            q = np.asarray([row["q_center"] for row in rows])
            values = np.asarray([row[field] for row in rows])
            ok = np.asarray([bool(row["ratio_support"]) for row in rows])
            axis.plot(q[ok], values[ok], marker=marker, ms=3, lw=1.1, color=color, label=name)
        axis.axhline(GOE_MEAN_GAP_RATIO, color="black", ls="--", lw=1, label="GOE")
        axis.axhline(POISSON_MEAN_GAP_RATIO, color="firebrick", ls=":", lw=1, label="Poisson")
        axis.set_ylabel(label)
        axis.grid(alpha=.18)
    axes[0].legend(frameon=False, ncol=4, fontsize=8)
    axes[-1].set_xlabel("held-out ensemble-CDF q")
    fig.suptitle(f"{model.label}: {level} standard vs half-bin shifted")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def rank_map(rows, model, path):
    q = np.asarray([row["q_center"] for row in rows])
    median = np.asarray([row["edge_rank_median"] for row in rows])
    low = np.asarray([row["edge_rank_p10"] for row in rows])
    high = np.asarray([row["edge_rank_p90"] for row in rows])
    fig, axis = plt.subplots(figsize=(8.8, 4.2))
    axis.plot(q, median, color="#6a4c93", label="median")
    axis.fill_between(q, low, high, color="#6a4c93", alpha=.2, label="10th-90th percentile")
    axis.invert_yaxis()
    axis.set(xlabel="ensemble-CDF q", ylabel="positive non-Perron edge rank")
    axis.set_title(f"{model.label}: q-to-edge-rank map")
    axis.grid(alpha=.18)
    axis.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def localization_overlay(rows, model, path):
    q = np.asarray([row["q_center"] for row in rows])
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(q, [row["NIPR"] for row in rows], color="#6a4c93")
    axes[0].set_ylabel("NIPR")
    axes[1].plot(q, [row["raw_r"] for row in rows], color="#457b9d", label="raw r")
    axes[1].plot(q, [row["unfolded_r"] for row in rows], color="#2a9d8f", label="unfolded r")
    axes[1].axhline(GOE_MEAN_GAP_RATIO, color="black", ls="--", lw=1)
    axes[1].set_ylabel("graph-equal r")
    axes[1].legend(frameon=False, fontsize=8)
    preference = [
        row["unfolded_spacing_KS_to_Poisson"] - row["unfolded_spacing_KS_to_GOE"]
        for row in rows
    ]
    axes[2].plot(q, preference, color="#bc6c25")
    axes[2].axhline(0, color="black", lw=1)
    axes[2].set(ylabel="KS(Poisson) - KS(GOE)", xlabel="ensemble-CDF q")
    for axis in axes:
        axis.grid(alpha=.18)
    fig.suptitle(f"{model.label}: localization and edge statistics")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def negative_records(dataset):
    graph_ids = np.asarray(dataset["graph_indices"], dtype=int)
    seeds = np.asarray(dataset["graph_seeds"], dtype=np.uint64)
    eigenvalues = np.asarray(dataset["negative_eigenvalue"], dtype=float)
    q = np.asarray(dataset["negative_ensemble_q"], dtype=float)
    unfolded_gaps = np.asarray(dataset["negative_cdf_unfolded_gap"], dtype=float)
    raw_gaps = np.diff(eigenvalues, axis=1)
    gap_shape = raw_gaps.shape
    gap_graphs = np.broadcast_to(graph_ids[:, None], gap_shape)
    gap_ranks = np.broadcast_to(np.arange(1, gap_shape[1] + 1), gap_shape)
    gaps = {
        "graph_index": gap_graphs.ravel(),
        "ensemble_q": (0.5 * (q[:, :-1] + q[:, 1:])).ravel(),
        "edge_rank": gap_ranks.ravel(),
        "left_energy": eigenvalues[:, :-1].ravel(),
        "right_energy": eigenvalues[:, 1:].ravel(),
        "raw_gap": raw_gaps.ravel(),
        "unfolded_gap": unfolded_gaps.ravel(),
    }
    left, right = raw_gaps[:, :-1], raw_gaps[:, 1:]
    uleft, uright = unfolded_gaps[:, :-1], unfolded_gaps[:, 1:]
    shape = left.shape
    graphs = np.broadcast_to(graph_ids[:, None], shape)
    triplets = {
        "graph_index": graphs.ravel(),
        "graph_seed": np.broadcast_to(seeds[:, None], shape).ravel(),
        "fold": (graphs % 5).ravel(),
        "middle_energy": eigenvalues[:, 1:-1].ravel(),
        "ensemble_q": q[:, 1:-1].ravel(),
        "edge_rank": np.broadcast_to(np.arange(2, eigenvalues.shape[1]), shape).ravel(),
        "raw_left_gap": left.ravel(),
        "raw_right_gap": right.ravel(),
        "raw_r": ratio(left, right).ravel(),
        "unfolded_left_gap": uleft.ravel(),
        "unfolded_right_gap": uright.ravel(),
        "unfolded_r": ratio(uleft, uright).ravel(),
        "relative_min_gap": (
            np.minimum(left, right) / np.maximum(np.abs(eigenvalues[:, 1:-1]), 1)
        ).ravel(),
    }
    return triplets, gaps


def negative_plot(positive, negative, model, path):
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.8), sharex=True)
    for axis, field in zip(axes, ("raw_r", "unfolded_r")):
        for rows, name, color in (
            (positive, "positive", "#2a9d8f"),
            (negative, "negative mirror", "#6a4c93"),
        ):
            q = np.asarray([row["q_center"] for row in rows])
            values = np.asarray([row[field] for row in rows])
            ok = np.asarray([bool(row["ratio_support"]) for row in rows])
            axis.plot(q[ok], values[ok], marker="o", ms=3, lw=1.1, color=color, label=name)
        axis.axhline(GOE_MEAN_GAP_RATIO, color="black", ls="--", lw=1)
        axis.set_ylabel(field)
        axis.grid(alpha=.18)
    axes[0].legend(frameon=False)
    axes[-1].set_xlabel("mirrored outer-edge ensemble quantile")
    fig.suptitle(f"{model.label}: positive/negative edge")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def ratio_distribution(triplets, low, high, title, path):
    keep = mask(triplets["ensemble_q"], low, high)
    raw, unfolded = triplets["raw_r"][keep], triplets["unfolded_r"][keep]
    x = np.linspace(0, 1, 1001)
    goe = 27 / 4 * (x + x * x) / np.power(1 + x + x * x, 2.5)
    poisson = 2 / np.square(1 + x)
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9), sharex=True, sharey=True)
    for axis, values, label, color in (
        (axes[0], raw, "raw", "#457b9d"),
        (axes[1], unfolded, "same-triplet unfolded", "#2a9d8f"),
    ):
        values = values[np.isfinite(values)]
        axis.hist(values, bins=np.linspace(0, 1, 31), density=True, color=color, alpha=.72)
        axis.plot(x, goe, color="black", lw=1.2, label="GOE")
        axis.plot(x, poisson, color="firebrick", ls="--", lw=1.2, label="Poisson")
        axis.set_title(f"{label}; n={len(values):,}")
        axis.set_xlabel("r")
        axis.grid(alpha=.18)
    axes[0].set_ylabel("density")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def spacing_pairs(triplets, low, high, title, path):
    indices = np.flatnonzero(mask(triplets["ensemble_q"], low, high))
    if len(indices) > 20000:
        rng = np.random.default_rng(stable_seed(title, low, high, "pairs"))
        indices = np.sort(rng.choice(indices, 20000, replace=False))
    left = triplets["unfolded_left_gap"][indices]
    right = triplets["unfolded_right_gap"][indices]
    finite = np.isfinite(left) & np.isfinite(right) & (left > 0) & (right > 0)
    left, right = left[finite], right[finite]
    directional = right / left
    limit = float(np.quantile(np.concatenate((left, right)), .995)) if len(left) else 1
    corr = float(np.corrcoef(left, right)[0, 1]) if len(left) > 1 else math.nan
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4))
    axes[0].scatter(left, right, s=4, alpha=.18, color="#2a9d8f", rasterized=True)
    axes[0].plot([0, limit], [0, limit], color="black", ls="--", lw=1)
    axes[0].set(xlim=(0, limit), ylim=(0, limit), xlabel="s_i", ylabel="s_(i+1)")
    axes[0].set_title(f"unfolded spacing pairs; corr={corr:.3f}")
    axes[1].hist(np.log10(directional), bins=45, color="#bc6c25", alpha=.76)
    axes[1].axvline(0, color="black", ls="--", lw=1)
    axes[1].set(xlabel="log10(s_(i+1)/s_i)", ylabel="count")
    axes[1].set_title("directional spacing ratio")
    for axis in axes:
        axis.grid(alpha=.16)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def sliding_plot(rows, model, path):
    fig, axes = plt.subplots(3, 1, figsize=(9, 8.6), sharex=True)
    for axis, width in zip(axes, SLIDING_WIDTHS):
        selected = [row for row in rows if math.isclose(float(row["window_width"]), width)]
        q = np.asarray([row["q_center"] for row in selected])
        ok = np.asarray([bool(row["ratio_support"]) for row in selected])
        for field, low_field, high_field, color, label in (
            ("raw_r", "raw_r_bootstrap_ci_low", "raw_r_bootstrap_ci_high", "#457b9d", "raw"),
            ("unfolded_r", "unfolded_r_bootstrap_ci_low", "unfolded_r_bootstrap_ci_high", "#2a9d8f", "unfolded"),
        ):
            values = np.asarray([row[field] for row in selected])
            low = np.asarray([row[low_field] for row in selected])
            high = np.asarray([row[high_field] for row in selected])
            axis.plot(q[ok], values[ok], color=color, label=label)
            axis.fill_between(q[ok], low[ok], high[ok], color=color, alpha=.14)
        axis.axhline(GOE_MEAN_GAP_RATIO, color="black", ls="--", lw=1)
        axis.set_ylabel("r")
        axis.set_title(f"width={width:g}")
        axis.grid(alpha=.18)
    axes[0].legend(frameon=False)
    axes[-1].set_xlabel("window-center q")
    fig.suptitle(f"{model.label}: sliding-window microscope")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def block_rows(triplets, gaps, rows, model, level):
    graph_count = int(np.max(triplets["graph_index"])) + 1
    output = []
    for start in range(0, graph_count, 500):
        stop = min(start + 500, graph_count)
        tkeep = (triplets["graph_index"] >= start) & (triplets["graph_index"] < stop)
        gkeep = (gaps["graph_index"] >= start) & (gaps["graph_index"] < stop)
        selected_triplets, selected_gaps = subset(triplets, tkeep), subset(gaps, gkeep)
        for source in rows:
            if not bool(source["ratio_support"]):
                continue
            row = summarize_q_window(
                selected_triplets, selected_gaps, model=model,
                q_low=float(source["q_low"]), q_high=float(source["q_high"]),
                bootstrap_replicates=500,
            )
            row.update({
                "microscope_version": VERSION,
                "resolution_level": level,
                "block_start": start,
                "block_stop_exclusive": stop,
            })
            output.append(row)
    return output


def block_plot(rows, model, path):
    fig, axis = plt.subplots(figsize=(9, 4.5))
    blocks = sorted({(int(row["block_start"]), int(row["block_stop_exclusive"])) for row in rows})
    colors = plt.cm.viridis(np.linspace(.08, .92, max(len(blocks), 1)))
    for color, (start, stop) in zip(colors, blocks):
        selected = [row for row in rows if int(row["block_start"]) == start]
        q = np.asarray([(float(row["q_low"]) + float(row["q_high"])) / 2 for row in selected])
        values = np.asarray([row["raw_r"] for row in selected])
        axis.plot(q, values, marker="o", ms=2.5, lw=1, color=color, label=f"{start}-{stop-1}")
    axis.axhline(GOE_MEAN_GAP_RATIO, color="black", ls="--", lw=1)
    axis.set(xlabel="ensemble-CDF q", ylabel="block graph-equal raw r")
    axis.set_title(f"{model.label}: independent 500-graph blocks")
    axis.grid(alpha=.18)
    axis.legend(frameon=False, fontsize=8, ncol=4)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def low_band(rows):
    supported = [row for row in rows if bool(row["ratio_support"])]
    if not supported:
        return math.nan, math.nan
    minimum = min(supported, key=lambda row: float(row["raw_r"]))
    index = rows.index(minimum)
    left = right = index
    threshold = GOE_MEAN_GAP_RATIO - .01
    while left and bool(rows[left - 1]["ratio_support"]) and float(rows[left - 1]["raw_r"]) < threshold:
        left -= 1
    while right + 1 < len(rows) and bool(rows[right + 1]["ratio_support"]) and float(rows[right + 1]["raw_r"]) < threshold:
        right += 1
    return float(rows[left]["q_low"]), float(rows[right]["q_high"])


def run(output_root: Path, model: FollowupModel):
    cache = output_root / "cached_n5000_reanalysis" / model.key
    triplets = load_npz(cache / "positive_edge_triplets.npz")
    gaps = load_npz(cache / "positive_edge_gaps.npz")
    dataset = load_npz(cache / "fixed_rank_dataset.npz")
    destination = output_root / MODEL_DIR[model.key]
    levels = dict(BASE_LEVELS)
    levels.update(FOCUSED.get(model.key, {}))
    all_rows = {}

    for level, (low, high, width) in levels.items():
        directory = destination / level
        standard = summarize_bins(
            triplets, gaps, dataset, model,
            make_bins(low, high, width), level, "standard",
        )
        shifted = summarize_bins(
            triplets, gaps, dataset, model,
            make_bins(low, high, width, True), level, "half_shifted",
        )
        all_rows[level] = standard
        write_rows(directory / "standard_summary.csv", standard)
        write_rows(directory / "half_shifted_summary.csv", shifted)
        grid_curves(standard, shifted, model, level, directory / "standard_vs_half_shifted_r.png")
        rank_map(standard, model, directory / "q_to_edge_rank.png")
        localization_overlay(standard, model, directory / "localization_r_ks_overlay.png")
        for row in standard:
            if not bool(row["histogram_support"]):
                continue
            low_value, high_value = float(row["q_low"]), float(row["q_high"])
            slug = f"q_{low_value:.5f}_{high_value:.5f}".replace(".", "p")
            spacing_histogram(
                gaps, low_value, high_value,
                f"{model.label}, q=[{low_value:.5f}, {high_value:.5f})",
                directory / "histograms" / f"{slug}.png",
            )
        histogram_panel(
            gaps, standard, model, level,
            directory / "unfolded_spacing_histogram_panel.png",
        )
        blocks = block_rows(triplets, gaps, standard, model, level)
        write_rows(directory / "independent_graph_blocks.csv", blocks)
        block_plot(blocks, model, directory / "independent_graph_block_replication.png")

    negative_triplets, negative_gaps = negative_records(dataset)
    low, high, width = BASE_LEVELS["dq_0p0002"]
    negative_standard = summarize_bins(
        negative_triplets, negative_gaps, dataset, model,
        make_bins(low, high, width), "dq_0p0002", "standard", side="negative",
    )
    negative_shifted = summarize_bins(
        negative_triplets, negative_gaps, dataset, model,
        make_bins(low, high, width, True), "dq_0p0002", "half_shifted", side="negative",
    )
    negative_dir = destination / "negative_edge"
    write_rows(negative_dir / "dq_0p0002_standard_summary.csv", negative_standard)
    write_rows(negative_dir / "dq_0p0002_half_shifted_summary.csv", negative_shifted)
    negative_plot(all_rows["dq_0p0002"], negative_standard, model, negative_dir / "positive_vs_negative.png")

    fixed = summarize_fixed_rank_dataset(dataset, model=model, n=5000)
    fixed_dir = destination / "fixed_rank"
    write_rows(fixed_dir / "fixed_rank_summary.csv", fixed)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for side, color in (("positive", "#2a9d8f"), ("negative", "#6a4c93")):
        selected = [row for row in fixed if row["side"] == side]
        ranks = np.asarray([(row["rank_low"] + row["rank_high"]) / 2 for row in selected])
        axes[0].plot(ranks, [row["raw_r"] for row in selected], marker="o", color=color, label=f"{side} raw")
        axes[0].plot(ranks, [row["cdf_unfolded_r"] for row in selected], marker="s", ls="--", color=color, label=f"{side} CDF-unfolded")
        axes[1].plot(ranks, [row["NIPR"] for row in selected], marker="o", color=color, label=side)
    axes[0].axhline(GOE_MEAN_GAP_RATIO, color="black", ls="--", lw=1)
    axes[0].set_ylabel("graph-equal r")
    axes[1].set_ylabel("NIPR")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("edge-rank window midpoint")
        axis.grid(alpha=.18)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle(f"{model.label}: fixed-rank companion")
    fig.tight_layout()
    fixed_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fixed_dir / "fixed_rank_r_and_localization.png", dpi=180)
    plt.close(fig)

    sliding = []
    for width in SLIDING_WIDTHS:
        centers = np.arange(.994 + width / 2, .999 - width / 2 + SLIDING_STEP / 2, SLIDING_STEP)
        rows = summarize_bins(
            triplets, gaps, dataset, model,
            [(float(c - width / 2), float(c + width / 2)) for c in centers],
            "sliding", "moving", reps=500,
        )
        for row, center in zip(rows, centers):
            row["window_width"] = width
            row["q_center"] = float(center)
        sliding.extend(rows)
    sliding_dir = destination / "sliding_windows"
    write_rows(sliding_dir / "sliding_window_summary.csv", sliding)
    sliding_plot(sliding, model, sliding_dir / "sliding_window_r.png")

    diagnostic_level = "dq_0p00005" if "dq_0p00005" in all_rows else "dq_0p0001"
    band_low, band_high = low_band(all_rows[diagnostic_level])
    if np.isfinite(band_low):
        ratio_distribution(
            triplets, band_low, band_high,
            f"{model.label}: candidate low-r band q=[{band_low:g}, {band_high:g})",
            destination / "ratio_histograms" / "candidate_low_band_p_r.png",
        )
        spacing_pairs(
            triplets, band_low, band_high,
            f"{model.label}: candidate low-r band q=[{band_low:g}, {band_high:g})",
            destination / "spacing_pair_correlations" / "candidate_low_band_pairs.png",
        )
        ranks = triplets["edge_rank"][mask(triplets["ensemble_q"], band_low, band_high)]
        fig, axis = plt.subplots(figsize=(7.8, 4))
        if len(ranks):
            axis.hist(ranks, bins=np.arange(int(np.min(ranks)) - .5, int(np.max(ranks)) + 1.5), color="#6a4c93", alpha=.8)
        axis.set(xlabel="positive non-Perron edge rank", ylabel="ratio count")
        axis.set_title(f"{model.label}: candidate-band rank distribution")
        axis.grid(alpha=.18)
        fig.tight_layout()
        rank_dir = destination / "quantile_to_rank"
        rank_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(rank_dir / "candidate_low_band_rank_histogram.png", dpi=180)
        plt.close(fig)

    manifest = {
        "microscope_version": VERSION,
        "authoritative_results_modified": False,
        "model_key": model.key,
        "model_label": model.label,
        "N": 5000,
        "m": 10,
        "source_cache_directory": str(cache),
        "output_directory": str(destination),
        "resolution_levels": levels,
        "standard_and_half_shifted": True,
        "sliding_widths": SLIDING_WIDTHS,
        "sliding_step": SLIDING_STEP,
        "histogram_support_gate": {"positive_gaps": 100, "graphs": 50},
        "ratio_support_gate": {"valid_ratios": 100, "graphs": 50},
        "candidate_band": [band_low, band_high],
        "sge_job_id": os.environ.get("JOB_ID", ""),
        "sge_task_id": os.environ.get("SGE_TASK_ID", ""),
        "files": sorted(str(path.relative_to(output_root)) for path in destination.rglob("*") if path.is_file()),
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(destination, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-key", choices=sorted(MODEL_BY_KEY), required=True)
    args = parser.parse_args()
    run(args.output_root, MODEL_BY_KEY[args.model_key])


if __name__ == "__main__":
    main()
