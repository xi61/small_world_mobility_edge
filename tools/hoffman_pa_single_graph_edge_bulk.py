#!/usr/bin/env python3
"""Run one fixed-graph PA edge/bulk spectral microscope."""

from __future__ import annotations

import argparse
from pathlib import Path

from small_world_mobility_edge.preferential_attachment.single_graph_edge_bulk import (
    SingleGraphConfig,
    run_single_graph_experiment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--m", type=int, default=10)
    parser.add_argument(
        "--model-family", required=True, choices=("affine_linear", "nonlinear")
    )
    parser.add_argument("--parameter-value", required=True, type=float)
    parser.add_argument("--graph-seed", required=True, type=int)
    parser.add_argument("--graph-index", type=int, default=0)
    parser.add_argument("--reference-cdf", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--histogram-window-width", type=float, default=0.05)
    parser.add_argument("--random-baseline-seed", type=int, default=945001)
    parser.add_argument("--random-baseline-samples", type=int, default=1000)
    parser.add_argument(
        "--self-fit-cdf",
        action="store_true",
        help="smoke tests only; production must supply a held-out CDF",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = SingleGraphConfig(
        n=args.n,
        m=args.m,
        model_family=args.model_family,
        parameter_value=args.parameter_value,
        graph_seed=args.graph_seed,
        graph_index=args.graph_index,
        reference_cdf_path=args.reference_cdf,
        output_root=args.output_root,
        histogram_window_width=args.histogram_window_width,
        random_baseline_seed=args.random_baseline_seed,
        random_baseline_samples=args.random_baseline_samples,
        smoke_only_self_fitted_cdf=args.self_fit_cdf,
    )
    output = run_single_graph_experiment(config)
    print(output)


if __name__ == "__main__":
    main()
