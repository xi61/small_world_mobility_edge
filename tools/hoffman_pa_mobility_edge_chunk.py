from __future__ import annotations

import os
from pathlib import Path

from small_world_mobility_edge.preferential_attachment.full_spectrum_scan import (
    MobilityEdgeProductionConfig,
    run_mobility_chunk,
    run_structural_replay_chunk,
)


def required(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise SystemExit(f"required environment variable {name} is missing")
    return value


def config_from_environment() -> MobilityEdgeProductionConfig:
    family = required("PA_MODEL_FAMILY")
    common = {
        "n": int(os.environ.get("PA_N", "5000")),
        "m": int(os.environ.get("PA_M", "10")),
        "num_graphs": int(os.environ.get("PA_NUM_GRAPHS", "500")),
        "graph_seed_base": int(required("PA_GRAPH_SEED_BASE")),
        "output_root": Path(required("PA_OUTPUT_ROOT")),
        "model_family": family,
        "graph_variant": os.environ.get("PA_GRAPH_VARIANT", "original"),
        "rewire_seed_base": (
            int(os.environ["PA_REWIRE_SEED_BASE"])
            if os.environ.get("PA_REWIRE_SEED_BASE")
            else None
        ),
        "swaps_per_edge": int(os.environ.get("PA_SWAPS_PER_EDGE", "10")),
        "max_tries_factor": int(os.environ.get("PA_MAX_TRIES_FACTOR", "20")),
    }
    if family == "affine_linear":
        return MobilityEdgeProductionConfig(
            **common, degree_ccdf_alpha=float(required("PA_DEGREE_CCDF_ALPHA"))
        )
    return MobilityEdgeProductionConfig(
        **common, attachment_zeta=float(required("PA_ATTACHMENT_ZETA"))
    )


def main() -> None:
    config = config_from_environment()
    task_id = int(os.environ.get("SGE_TASK_ID", "1"))
    chunk_size = int(os.environ.get("PA_CHUNK_SIZE", "5"))
    start = (task_id - 1) * chunk_size
    stop = min(start + chunk_size, config.num_graphs)
    if start >= stop:
        print(f"task {task_id}: empty graph range", flush=True)
        return
    print(
        f"job={os.environ.get('JOB_ID', '')} task={task_id} "
        f"{config.canonical_parameter_name}={config.canonical_parameter_value:g} "
        f"graphs=[{start},{stop})",
        flush=True,
    )
    if os.environ.get("PA_STRUCTURE_ONLY", "0") == "1":
        run_structural_replay_chunk(config, start, stop, skip_valid=True)
    else:
        run_mobility_chunk(config, start, stop, skip_valid=True)


if __name__ == "__main__":
    main()
