#!/usr/bin/env python3
"""SGE task entry point for focused sparse PA edge eigenpairs."""

from __future__ import annotations

import os
from pathlib import Path

from small_world_mobility_edge.preferential_attachment.abnormal_edge_followup import (
    MODEL_BY_KEY,
    SparseEdgeConfig,
    run_sparse_chunk,
)


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"required environment variable {name} is missing")
    return value


def config_from_environment() -> SparseEdgeConfig:
    return SparseEdgeConfig(
        model=MODEL_BY_KEY[required("PA_MODEL_KEY")],
        n=int(required("PA_N")),
        m=int(os.environ.get("PA_M", "10")),
        num_graphs=int(required("PA_NUM_GRAPHS")),
        output_root=Path(required("PA_FOLLOWUP_OUTPUT_ROOT")),
        rank_count=int(os.environ.get("PA_EDGE_RANK_COUNT", "120")),
        eigsh_tolerance=float(os.environ.get("PA_EIGSH_TOLERANCE", "1e-9")),
        eigsh_maxiter_factor=int(os.environ.get("PA_EIGSH_MAXITER_FACTOR", "30")),
    )


def main() -> None:
    config = config_from_environment()
    task_id = int(os.environ.get("SGE_TASK_ID", "1"))
    chunk_size = int(os.environ.get("PA_CHUNK_SIZE", "1"))
    start_index = int(os.environ.get("PA_START_GRAPH_INDEX", "0"))
    start = start_index + (task_id - 1) * chunk_size
    stop = min(start + chunk_size, config.num_graphs)
    if start >= stop:
        print(f"task {task_id}: empty graph range", flush=True)
        return
    print(
        f"job={os.environ.get('JOB_ID', '')} task={task_id} model={config.model.key} "
        f"N={config.n} graphs=[{start},{stop})",
        flush=True,
    )
    run_sparse_chunk(config, start, stop, skip_valid=True)


if __name__ == "__main__":
    main()
