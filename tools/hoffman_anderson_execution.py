#!/usr/bin/env python3
"""Command-line entry point for the isolated Anderson execution."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from small_world_mobility_edge.random_schrodinger.anderson_reporting import run_anderson_aggregation
from small_world_mobility_edge.random_schrodinger.anderson_rrg_benchmark import run_rrg_a9_stream, run_rrg_task
from small_world_mobility_edge.random_schrodinger.disorder_postprocess_v2 import run_pilot_postprocess
from small_world_mobility_edge.random_schrodinger.disorder_replication import run_baseline_supplement, run_pa_replication_task


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--project-root", type=Path, required=True)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--freeze-manifest", type=Path, required=True)
    commands = result.add_subparsers(dest="command", required=True)

    pilot = commands.add_parser("pilot-postprocess")
    pilot.add_argument("--model-key", required=True)
    pilot.add_argument("--potential-seed", type=int, required=True)
    pilot.add_argument("--output-dir", type=Path, required=True)

    supplement = commands.add_parser("baseline-supplement")
    supplement.add_argument("--model-key", required=True)
    supplement.add_argument("--output-dir", type=Path, required=True)

    pa = commands.add_parser("pa-replication")
    pa.add_argument("--task-id", type=int, default=None)
    pa.add_argument("--output-root", type=Path, required=True)

    rrg = commands.add_parser("rrg")
    rrg.add_argument("--task-id", type=int, default=None)
    rrg.add_argument("--output-root", type=Path, required=True)

    a9 = commands.add_parser("rrg-a9")
    a9.add_argument("--task-id", type=int, default=None)
    a9.add_argument("--output-root", type=Path, required=True)

    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--run-root", type=Path, required=True)
    aggregate.add_argument("--output-dir", type=Path, required=True)
    aggregate.add_argument("--phase", choices=("pre_a9", "final"), required=True)
    return result


def _task_id(argument: int | None) -> int:
    value = argument if argument is not None else os.environ.get("SGE_TASK_ID")
    if value is None:
        raise ValueError("task ID is required")
    return int(value)


def main() -> None:
    args = parser().parse_args()
    common = {"project_root": args.project_root, "config_path": args.config, "freeze_manifest": args.freeze_manifest}
    if args.command == "pilot-postprocess":
        run_pilot_postprocess(**common, model_key=args.model_key, potential_seed=args.potential_seed, output_dir=args.output_dir)
    elif args.command == "baseline-supplement":
        run_baseline_supplement(**common, model_key=args.model_key, output_dir=args.output_dir)
    elif args.command == "pa-replication":
        run_pa_replication_task(**common, task_id=_task_id(args.task_id), output_root=args.output_root)
    elif args.command == "rrg":
        run_rrg_task(**common, task_id=_task_id(args.task_id), output_root=args.output_root)
    elif args.command == "rrg-a9":
        run_rrg_a9_stream(**common, task_id=_task_id(args.task_id), output_root=args.output_root)
    elif args.command == "aggregate":
        run_anderson_aggregation(**common, run_root=args.run_root, output_dir=args.output_dir, phase=args.phase)
    else:
        raise AssertionError("unreachable command")


if __name__ == "__main__":
    main()
