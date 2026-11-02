#!/usr/bin/env python3
"""Hoffman entry point for cached affine-PA multifractality analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from small_world_mobility_edge.preferential_attachment.multifractality import (
    aggregate_results,
    alpha_slug,
    analyze_case,
    controlled_validation,
    write_csv,
    write_json,
)


def load_config(path: Path) -> dict[str, object]:
    with Path(path).open() as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--output-dir", type=Path, required=True)

    case = subparsers.add_parser("case")
    case.add_argument("--alpha", type=float, required=True)
    case.add_argument("--source-root", type=Path, required=True)
    case.add_argument("--output-root", type=Path, required=True)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--output-root", type=Path, required=True)
    aggregate.add_argument("--report-dir", type=Path, required=True)

    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "validate":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        rows, passed = controlled_validation()
        write_csv(args.output_dir / "controlled_validation.csv", rows)
        write_json(args.output_dir / "validation_manifest.json", {
            "status": "COMPLETE" if passed else "FAILED",
            "validation_passed": passed,
            "fixture_count": len(rows),
        })
        if not passed:
            raise SystemExit("controlled multifractality validation failed")
        return

    if args.command == "case":
        configured_alphas = [float(value) for value in config["alpha_grid"]]
        matches = [value for value in configured_alphas if abs(value - args.alpha) < 1e-12]
        if len(matches) != 1:
            raise ValueError("requested alpha is not uniquely configured")
        for n in [int(value) for value in config["sizes"]]:
            relative = Path("N_%d" % n) / ("alpha_%s" % alpha_slug(args.alpha))
            manifest = analyze_case(
                source_case_dir=args.source_root / relative,
                output_case_dir=args.output_root / relative,
                n=n,
                alpha=args.alpha,
                config=config,
            )
            print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    if args.command == "aggregate":
        manifest = aggregate_results(
            output_root=args.output_root,
            report_dir=args.report_dir,
            config=config,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
