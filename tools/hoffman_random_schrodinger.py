"""Command-line entry point for the staged random-Schrödinger extension."""

from __future__ import annotations

import argparse
from pathlib import Path

from small_world_mobility_edge.random_schrodinger.applications import run_tunneling_application
from small_world_mobility_edge.random_schrodinger.disorder import run_pa_disorder_sweep
from small_world_mobility_edge.random_schrodinger.er_benchmark import run_er_figure10_benchmark
from small_world_mobility_edge.random_schrodinger.geometry import run_stage1_cached_geometry
from small_world_mobility_edge.random_schrodinger.tunneling import run_tunneling_validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    stage1 = subparsers.add_parser("stage1", help="Reanalyse saved PA graph/eigenpair caches.")
    stage1.add_argument("--source-dir", type=Path, required=True)
    stage1.add_argument("--output-dir", type=Path, required=True)
    stage1.add_argument("--project-root", type=Path, default=Path.cwd())
    stage1.add_argument("--baseline-samples", type=int, default=100)
    stage1.add_argument("--baseline-seed", type=int, default=945101)
    er = subparsers.add_parser("er-benchmark", help="Run the full-spectrum ER Figure 10 setup.")
    er.add_argument("--n", type=int, required=True)
    er.add_argument("--graph-seed", type=int, required=True)
    er.add_argument("--output-dir", type=Path, required=True)
    er.add_argument("--project-root", type=Path, default=Path.cwd())
    er.add_argument("--run-label", required=True)
    validate = subparsers.add_parser("validate-tunneling", help="Run controlled tunneling fixtures.")
    validate.add_argument("--output-dir", type=Path, required=True)
    validate.add_argument("--project-root", type=Path, default=Path.cwd())
    application = subparsers.add_parser("apply-tunneling", help="Apply diagnostics to ER, PA, or CL.")
    application.add_argument("--case", choices=("er", "pa", "cl"), required=True)
    application.add_argument("--input-dir", type=Path)
    application.add_argument("--output-dir", type=Path, required=True)
    application.add_argument("--project-root", type=Path, default=Path.cwd())
    disorder = subparsers.add_parser("pa-disorder", help="Run a paired PA diagonal-disorder sweep.")
    disorder.add_argument("--source-dir", type=Path, required=True)
    disorder.add_argument("--output-dir", type=Path, required=True)
    disorder.add_argument("--potential-seed", type=int, required=True)
    disorder.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "stage1":
        run_stage1_cached_geometry(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            project_root=args.project_root,
            baseline_samples=args.baseline_samples,
            baseline_seed=args.baseline_seed,
        )
        return
    if args.command == "er-benchmark":
        run_er_figure10_benchmark(
            n_requested=args.n,
            graph_seed=args.graph_seed,
            output_dir=args.output_dir,
            project_root=args.project_root,
            run_label=args.run_label,
        )
        return
    if args.command == "validate-tunneling":
        run_tunneling_validation(args.output_dir, args.project_root)
        return
    if args.command == "apply-tunneling":
        run_tunneling_application(
            case=args.case,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            project_root=args.project_root,
        )
        return
    if args.command == "pa-disorder":
        run_pa_disorder_sweep(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            project_root=args.project_root,
            potential_seed=args.potential_seed,
        )
        return
    raise AssertionError("unhandled command")


if __name__ == "__main__":
    main()
