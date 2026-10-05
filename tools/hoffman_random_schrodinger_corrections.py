"""CLI for the independently reviewed random-Schrödinger correction workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from small_world_mobility_edge.random_schrodinger.common import source_configuration_identity
from small_world_mobility_edge.random_schrodinger.disorder_correction import run_disorder_correction
from small_world_mobility_edge.random_schrodinger.stage1_correction import run_stage1_correction
from small_world_mobility_edge.random_schrodinger.tunneling_correction import (
    run_corrected_tunneling_application,
    run_corrected_tunneling_validation,
    run_paper_er_nonempty,
)
from small_world_mobility_edge.random_schrodinger.zero_subspace_correction import (
    run_zero_subspace_correction,
)


def _verify_freeze(project_root: Path, freeze_manifest: Path) -> str:
    payload = json.loads(Path(freeze_manifest).read_text())
    if payload.get("status") != "COMPLETE":
        raise ValueError("freeze manifest is not COMPLETE")
    current = source_configuration_identity(project_root)["sha256"]
    if payload.get("source_configuration_id") != current:
        raise ValueError(
            "frozen source/configuration identity does not match current executable tree: "
            + str(payload.get("source_configuration_id"))
            + " != "
            + str(current)
        )
    return str(current)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    stage1 = subparsers.add_parser("stage1")
    stage1.add_argument("--source-dir", type=Path, required=True)
    stage1.add_argument("--original-stage1-dir", type=Path, required=True)
    stage1.add_argument("--output-dir", type=Path, required=True)

    disorder = subparsers.add_parser("disorder")
    disorder.add_argument("--source-dir", type=Path, required=True)
    disorder.add_argument("--original-sweep-dir", type=Path, required=True)
    disorder.add_argument("--output-dir", type=Path, required=True)

    zero = subparsers.add_parser("zero-subspace")
    zero.add_argument("--source-dir", type=Path, required=True)
    zero.add_argument("--sweep-dir", type=Path, action="append", required=True)
    zero.add_argument("--output-dir", type=Path, required=True)

    validation = subparsers.add_parser("tunneling-validation")
    validation.add_argument("--output-dir", type=Path, required=True)

    application = subparsers.add_parser("tunneling-application")
    application.add_argument("--case", choices=("er", "pa", "cl"), required=True)
    application.add_argument("--input-dir", type=Path, required=True)
    application.add_argument("--output-dir", type=Path, required=True)

    paper = subparsers.add_parser("paper-er")
    paper.add_argument("--input-dir", type=Path, required=True)
    paper.add_argument("--old-application-dir", type=Path, required=True)
    paper.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _verify_freeze(args.project_root, args.freeze_manifest)
    if args.command == "stage1":
        run_stage1_correction(
            source_dir=args.source_dir,
            original_stage1_dir=args.original_stage1_dir,
            output_dir=args.output_dir,
            project_root=args.project_root,
        )
    elif args.command == "disorder":
        run_disorder_correction(
            source_dir=args.source_dir,
            original_sweep_dir=args.original_sweep_dir,
            output_dir=args.output_dir,
            project_root=args.project_root,
        )
    elif args.command == "zero-subspace":
        run_zero_subspace_correction(
            source_dir=args.source_dir,
            sweep_dirs=args.sweep_dir,
            output_dir=args.output_dir,
            project_root=args.project_root,
        )
    elif args.command == "tunneling-validation":
        run_corrected_tunneling_validation(args.output_dir, args.project_root)
    elif args.command == "tunneling-application":
        run_corrected_tunneling_application(
            case=args.case,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            project_root=args.project_root,
        )
    elif args.command == "paper-er":
        run_paper_er_nonempty(
            input_dir=args.input_dir,
            old_application_dir=args.old_application_dir,
            output_dir=args.output_dir,
            project_root=args.project_root,
        )
    else:
        raise AssertionError("unhandled command")


if __name__ == "__main__":
    main()
