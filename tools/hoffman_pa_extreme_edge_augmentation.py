from __future__ import annotations

import os
from pathlib import Path

from small_world_mobility_edge.preferential_attachment.extreme_edge_augmentation import (
    run_augmentation_chunk,
)
from small_world_mobility_edge.preferential_attachment.extreme_edge_quantiles import (
    ModelSpecification,
    config_from_manifest,
)


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"required environment variable {name} is missing")
    return value


def main() -> None:
    pa_root = Path(required("PA_OUTPUT_ROOT"))
    output_root = Path(required("PA_EXTREME_EDGE_OUTPUT_ROOT"))
    specification = ModelSpecification(
        required("PA_MODEL_FAMILY"),
        required("PA_PARAMETER_NAME"),
        float(required("PA_PARAMETER_VALUE")),
    )
    config = config_from_manifest(pa_root, specification)
    base = int(os.environ.get("PA_BASE_GRAPH_COUNT", "500"))
    start_index = int(os.environ.get("PA_START_GRAPH_INDEX", str(base)))
    target = int(required("PA_TARGET_GRAPH_COUNT"))
    chunk_size = int(os.environ.get("PA_CHUNK_SIZE", "5"))
    task_id = int(os.environ.get("SGE_TASK_ID", "1"))
    start = start_index + (task_id - 1) * chunk_size
    stop = min(start + chunk_size, target)
    if start >= stop:
        print(f"task {task_id}: empty augmentation graph range", flush=True)
        return
    print(
        f"job={os.environ.get('JOB_ID', '')} task={task_id} "
        f"{specification.parameter_name}={specification.parameter_value:g} "
        f"graphs=[{start},{stop}) target={target}",
        flush=True,
    )
    run_augmentation_chunk(
        config,
        specification,
        output_root,
        base_graph_count=base,
        target_graph_count=target,
        start=start,
        stop=stop,
        skip_valid=True,
    )


if __name__ == "__main__":
    main()
