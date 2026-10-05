"""Hoffman CLI for the clean-PA coupling handoff stage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from small_world_mobility_edge.random_schrodinger.common import (
    read_json,
    source_configuration_identity,
    utc_now,
    write_csv,
    write_json,
)
from small_world_mobility_edge.random_schrodinger.handoff_coupling import (
    COUPLING_HANDOFF_VERSION,
    controlled_validation_rows,
    run_retained_clean_pa,
)
from small_world_mobility_edge.random_schrodinger.handoff_reporting import run_clean_pa_aggregation


def verify_freeze(project_root: Path, freeze_manifest: Path) -> str:
    payload = read_json(freeze_manifest)
    current = source_configuration_identity(project_root)["sha256"]
    if payload.get("status") != "COMPLETE" or payload.get("source_configuration_id") != current:
        raise ValueError("freeze manifest does not match current executable source/configuration")
    return str(current)


def load_config(path: Path) -> dict[str, object]:
    return json.loads(Path(path).read_text())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--output-dir", type=Path, required=True)
    retained = sub.add_parser("retained")
    retained.add_argument("--task-id", type=int, required=True)
    retained.add_argument("--source-root", type=Path, required=True)
    retained.add_argument("--output-root", type=Path, required=True)
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--clean-pa-root", type=Path, required=True)
    aggregate.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    source_id = verify_freeze(args.project_root, args.freeze_manifest)
    config = load_config(args.config)
    if args.command == "validate":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        rows, passed = controlled_validation_rows()
        write_csv(args.output_dir / "controlled_validation.csv", rows)
        write_json(args.output_dir / "run_manifest.json", {
            "status": "COMPLETE" if passed else "FAILED",
            "validation_passed": bool(passed),
            "stage_version": COUPLING_HANDOFF_VERSION,
            "source_configuration_id": source_id,
            "finished_at_utc": utc_now(),
            "fixture_row_count": len(rows),
            "acceptance_rule": "every controlled-fixture row must pass its predeclared exact/behavioral threshold",
        })
        if not passed:
            raise RuntimeError("one or more controlled coupling fixtures failed")
        return
    if args.command == "aggregate":
        run_clean_pa_aggregation(
            clean_pa_root=args.clean_pa_root,
            output_dir=args.output_dir,
            project_root=args.project_root,
            freeze_manifest=args.freeze_manifest,
            config=config,
        )
        return
    models = {int(row["task_id"]): row for row in config["models"]}
    if args.task_id not in models:
        raise ValueError(f"unknown task ID {args.task_id}")
    model = models[args.task_id]
    run_retained_clean_pa(
        source_dir=args.source_root / str(model["model_key"]),
        output_dir=args.output_root / str(model["model_key"]),
        project_root=args.project_root,
        freeze_manifest=args.freeze_manifest,
        config=config,
    )


if __name__ == "__main__":
    main()
