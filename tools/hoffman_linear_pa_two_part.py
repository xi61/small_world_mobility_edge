#!/usr/bin/env python3
"""Hoffman entry point for the two-part affine-PA experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from small_world_mobility_edge.preferential_attachment.linear_pa_two_part import (
    controlled_validation_rows,
    resolve_case,
    run_part1,
    run_part2,
)
from small_world_mobility_edge.preferential_attachment.linear_pa_two_part_reporting import run_aggregate
from small_world_mobility_edge.random_schrodinger.common import (
    read_json,
    source_configuration_identity,
    utc_now,
    write_csv,
    write_json,
)


def _load_config(path: Path) -> dict[str, object]:
    with Path(path).open() as handle:
        return json.load(handle)


def _verify_freeze(project_root: Path, freeze_manifest: Path) -> dict[str, object]:
    freeze = read_json(freeze_manifest)
    identity = source_configuration_identity(project_root)
    if freeze.get("status") != "COMPLETE":
        raise ValueError("freeze manifest is not complete")
    if freeze.get("source_configuration_id") != identity["sha256"]:
        raise ValueError("freeze manifest does not match the current source/configuration identity")
    return freeze


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--output", type=Path)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--output-dir", type=Path, required=True)

    for name in ("part1", "part2"):
        command = subparsers.add_parser(name)
        command.add_argument("--n", type=int, required=True)
        command.add_argument("--alpha", type=float, required=True)
        command.add_argument("--output-root", type=Path, required=True)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--output-root", type=Path, required=True)
    aggregate.add_argument("--report-dir", type=Path, required=True)

    args = parser.parse_args()
    project_root = args.project_root.resolve()
    config = _load_config(args.config)
    freeze = _verify_freeze(project_root, args.freeze_manifest)

    if args.command == "plan":
        cases = [
            resolve_case(config, n=int(n), alpha=float(alpha))
            for n in config["sizes"]
            for alpha in config["alpha_grid"]
        ]
        payload = {
            "experiment_id": config["experiment_id"],
            "generated_at_utc": utc_now(),
            "source_configuration_id": freeze["source_configuration_id"],
            "case_count": len(cases),
            "cases": cases,
        }
        if args.output:
            write_json(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if args.command == "validate":
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        rows, passed = controlled_validation_rows()
        configuration_pass = bool(
            config["sizes"] == [500, 10000]
            and config["alpha_grid"] == [1.05, 1.1, 1.25, 1.5, 2.0, 2.5, 3.0]
            and config["part2"]["x_grid"] == [0.7, 0.8, 0.9, 0.95]
            and config["part2"]["y_grid"] == [0.05, 0.1, 0.2, 0.3]
            and config["part2"]["gram_cutoff_sensitivity"] == [1e-6, 1e-8, 1e-10]
        )
        rows.append({"fixture": "configuration_propagation", "passed": configuration_pass})
        passed = bool(passed and configuration_pass)
        write_csv(output_dir / "controlled_validation.csv", rows)
        write_json(output_dir / "run_manifest.json", {
            "status": "COMPLETE" if passed else "FAILED",
            "validation_passed": passed,
            "fixture_count": len(rows),
            "source_configuration_id": freeze["source_configuration_id"],
            "finished_at_utc": utc_now(),
        })
        if not passed:
            raise SystemExit("controlled validation failed")
        return

    if args.command == "part1":
        result = run_part1(
            project_root=project_root,
            config=config,
            n=args.n,
            alpha=args.alpha,
            output_root=args.output_root,
            freeze_manifest=args.freeze_manifest,
        )
        print(result)
        return

    if args.command == "part2":
        result = run_part2(
            project_root=project_root,
            config=config,
            n=args.n,
            alpha=args.alpha,
            output_root=args.output_root,
            freeze_manifest=args.freeze_manifest,
        )
        print(result)
        return

    if args.command == "aggregate":
        result = run_aggregate(
            config=config,
            output_root=args.output_root,
            report_dir=args.report_dir,
            freeze_manifest=args.freeze_manifest,
        )
        print(result)


if __name__ == "__main__":
    main()
