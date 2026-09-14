#!/usr/bin/env python3
"""Aggregate sparse edge chunks and produce fixed-rank summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from small_world_mobility_edge.preferential_attachment.abnormal_edge_followup import (
    GOE_MEAN_GAP_RATIO,
    MODEL_BY_KEY,
    POISSON_MEAN_GAP_RATIO,
    SparseEdgeConfig,
    aggregate_sparse_chunks,
    load_sparse_dataset,
    second_solver_validation,
    summarize_fixed_rank_dataset,
)
from small_world_mobility_edge.preferential_attachment.mobility_edge_postprocess import (
    write_rows,
)


def config_from_args(args: argparse.Namespace) -> SparseEdgeConfig:
    return SparseEdgeConfig(
        model=MODEL_BY_KEY[args.model_key],
        n=args.n,
        m=args.m,
        num_graphs=args.num_graphs,
        output_root=args.output_root,
        rank_count=args.rank_count,
        eigsh_tolerance=args.eigsh_tolerance,
        eigsh_maxiter_factor=args.eigsh_maxiter_factor,
    )


def plot_alignment(rows, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharex=True)
    colors = {"positive": "#2a9d8f", "negative": "#6a4c93"}
    for side in ("positive", "negative"):
        selected = [row for row in rows if row["side"] == side]
        rank = np.asarray(
            [0.5 * (row["rank_low"] + row["rank_high"]) for row in selected],
            dtype=float,
        )
        axes[0].plot(
            rank,
            [row["raw_r"] for row in selected],
            marker="o",
            color=colors[side],
            label=f"{side} raw r",
        )
        axes[0].plot(
            rank,
            [row["unfolded_r"] for row in selected],
            marker="s",
            ls="--",
            color=colors[side],
            alpha=0.8,
            label=f"{side} rank-unfolded r",
        )
        axes[1].plot(
            rank,
            [row["NIPR"] for row in selected],
            marker="o",
            color=colors[side],
            label=side,
        )
    axes[0].axhline(GOE_MEAN_GAP_RATIO, color="black", ls="--", lw=1, label="GOE")
    axes[0].axhline(
        POISSON_MEAN_GAP_RATIO, color="firebrick", ls=":", lw=1, label="Poisson"
    )
    axes[0].set_ylabel("graph-equal adjacent-gap ratio")
    axes[1].set_ylabel("mean N x IPR")
    for axis in axes:
        axis.set_xlabel("edge-rank window midpoint")
        axis.set_xscale("log")
        axis.grid(alpha=0.18)
        axis.legend(frameon=False, fontsize=8)
    first = rows[0]
    figure.suptitle(f"{first['model']}, N={first['N']}: localization and level statistics")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("aggregate", "validate"))
    parser.add_argument("--model-key", choices=sorted(MODEL_BY_KEY), required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--m", type=int, default=10)
    parser.add_argument("--num-graphs", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rank-count", type=int, default=120)
    parser.add_argument("--eigsh-tolerance", type=float, default=1e-9)
    parser.add_argument("--eigsh-maxiter-factor", type=int, default=30)
    parser.add_argument("--validation-graphs", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    if args.mode == "validate":
        rows = second_solver_validation(
            config, range(min(args.validation_graphs, config.num_graphs))
        )
        path = (
            config.output_root
            / "solver_validation"
            / f"{config.model.key}_n{config.n}_second_solver.csv"
        )
        write_rows(path, rows)
        print(path, flush=True)
        return
    aggregate_path = aggregate_sparse_chunks(config, chunk_size=args.chunk_size)
    dataset = load_sparse_dataset(aggregate_path)
    rows = summarize_fixed_rank_dataset(
        dataset,
        model=config.model,
        n=config.n,
    )
    summary_path = config.parameter_dir / "fixed_rank_summary.csv"
    write_rows(summary_path, rows)
    plot_alignment(rows, config.parameter_dir / "localization_level_statistics_alignment.png")
    print(summary_path, flush=True)


if __name__ == "__main__":
    main()
