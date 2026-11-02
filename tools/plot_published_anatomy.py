#!/usr/bin/env python3
"""Plot individual-edge and near-zero comparisons from committed scalar tables."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "docs/results/latest_summary_tables")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    levels = pd.read_csv(args.data_dir / "n10000_individual_levels.csv").sort_values("descending_rank_1based")
    if len(levels) != 10000 or levels["descending_rank_1based"].tolist() != list(range(1, 10001)):
        raise ValueError("expected the complete ranked N=10000 spectrum")
    selected = levels.iloc[1:51]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    axes[0].plot(selected["descending_rank_1based"], selected["n_times_ipr"], "o-", ms=3, color="#176b87")
    axes[0].set_ylabel("N × IPR")
    axes[0].set_title("Individual non-Perron eigenvectors")
    gaps = -np.diff(levels["energy"].to_numpy())
    axes[1].semilogy(np.arange(2, 51), gaps[1:50], "o-", ms=3, color="#a44a3f")
    axes[1].set_ylabel("E(rank) − E(rank+1), raw units")
    axes[1].set_title("Consecutive gaps; no monotonicity assumed")
    for ax in axes:
        ax.set_xlabel("Descending algebraic rank (Perron is rank 1)")
        ax.grid(alpha=0.2)
    fig.suptitle("One saved affine-PA graph: N=10000, m=10, degree-CCDF α=1.05")
    fig.savefig(args.output_dir / "individual_edge.png", dpi=180)
    plt.close(fig)

    summary = pd.read_csv(args.data_dir / "nearzero_structure.csv")
    regions = ["negative peak", "near zero", "positive peak"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), constrained_layout=True)
    for case, color, shift in [("N=5000", "#176b87", -0.07), ("N=10000", "#a44a3f", 0.07)]:
        for ax, metric in zip(axes, ["n_times_ipr", "top20_mass"]):
            rows = summary[(summary["case"] == case) & (summary["metric"] == metric)].set_index("region").loc[regions]
            median = rows["median"].to_numpy()
            errors = np.vstack([median - rows["p25"].to_numpy(), rows["p75"].to_numpy() - median])
            ax.errorbar(np.arange(3) + shift, median, yerr=errors, fmt="o", capsize=4, color=color, label=case)
            ax.set_xticks(np.arange(3), ["Negative shoulder", "Near zero (nonzero)", "Positive shoulder"], rotation=12)
            ax.grid(axis="y", alpha=0.2)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("N × IPR")
    axes[1].set_ylabel("Top-20 cumulative mass")
    axes[1].set_ylim(0, 1)
    axes[0].legend(frameon=False)
    fig.suptitle("One graph per N: medians and interquartile ranges over eigenvectors, not graph uncertainty")
    fig.savefig(args.output_dir / "nearzero_support.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
