#!/usr/bin/env python3
"""Hoffman entry point for the raw-amplitude PA diagonal-disorder study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from small_world_mobility_edge.random_schrodinger.common import (
    read_json,
    sha256_file,
    source_configuration_identity,
    utc_now,
    write_json,
)
from small_world_mobility_edge.random_schrodinger.pa_diagonal_uniform_noise import (
    load_configuration,
    prepare_case,
    run_controlled_pilot,
    run_operator,
)
from small_world_mobility_edge.random_schrodinger.pa_diagonal_uniform_noise_reporting import (
    aggregate_case,
    aggregate_global,
)


def verify_freeze(project_root: Path, config_path: Path, freeze_manifest: Path) -> tuple[dict[str, object], str]:
    freeze = read_json(freeze_manifest)
    current = source_configuration_identity(project_root)
    if freeze.get("status") != "COMPLETE":
        raise ValueError("freeze manifest is not COMPLETE")
    if freeze.get("source_configuration_id") != current["sha256"]:
        raise ValueError("live isolated source differs from immutable freeze")
    if freeze.get("configuration_sha256") != sha256_file(config_path):
        raise ValueError("live configuration differs from immutable freeze")
    return freeze, str(current["sha256"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    subparsers.add_parser("pilot").add_argument("--output", type=Path, required=True)
    for name in ("prepare", "case-aggregate"):
        command = subparsers.add_parser(name)
        command.add_argument("--n", type=int, required=True)
        command.add_argument("--alpha", type=float, required=True)
    operator = subparsers.add_parser("operator")
    operator.add_argument("--n", type=int, required=True)
    operator.add_argument("--alpha", type=float, required=True)
    operator.add_argument("--amplitude", type=float, required=True)
    subparsers.add_parser("global-aggregate")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    config_path = args.config.resolve()
    _, source_identity = verify_freeze(project_root, config_path, args.freeze_manifest.resolve())
    config = load_configuration(config_path)
    run_root = args.run_root.resolve()
    if args.command == "plan":
        cases = [
            {
                "n": int(n),
                "degree_ccdf_alpha": float(alpha),
                "raw_amplitude": float(amplitude),
                "clean_case_path": str(Path(config["clean_source_root"]) / f"N_{int(n)}"),
            }
            for n in config["sizes"]
            for alpha in config["degree_ccdf_alpha_grid"]
            for amplitude in config["amplitudes"]
        ]
        print(json.dumps({
            "experiment_id": config["experiment_id"],
            "source_identity": source_identity,
            "logical_operator_count": len(cases),
            "positive_disorder_operator_count": sum(float(row["raw_amplitude"]) > 0 for row in cases),
            "cases": cases,
        }, indent=2, sort_keys=True))
    elif args.command == "pilot":
        print(run_controlled_pilot(config=config, output=args.output))
    elif args.command == "prepare":
        print(prepare_case(
            config=config,
            run_root=run_root,
            clean_root=Path(config["clean_source_root"]),
            n=args.n,
            alpha=args.alpha,
            source_identity=source_identity,
        ))
    elif args.command == "operator":
        print(run_operator(
            config=config,
            run_root=run_root,
            n=args.n,
            alpha=args.alpha,
            amplitude=args.amplitude,
            source_identity=source_identity,
        ))
    elif args.command == "case-aggregate":
        print(aggregate_case(
            config=config,
            run_root=run_root,
            n=args.n,
            alpha=args.alpha,
            source_identity=source_identity,
        ))
    elif args.command == "global-aggregate":
        print(aggregate_global(config=config, run_root=run_root, source_identity=source_identity))


if __name__ == "__main__":
    main()
