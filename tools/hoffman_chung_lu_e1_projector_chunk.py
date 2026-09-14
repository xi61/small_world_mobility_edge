from __future__ import annotations

import os

from small_world_mobility_edge import chung_lu_e1_projector_core_scan as scan


def main() -> None:
    task_id = int(os.environ.get("SGE_TASK_ID", "1"))
    chunk_size = int(os.environ.get("CHUNK_SIZE", "2"))
    start = (task_id - 1) * chunk_size
    stop = min(start + chunk_size, scan.NUM_GRAPHS)
    if start >= scan.NUM_GRAPHS:
        print(f"task {task_id}: no graphs assigned", flush=True)
        return
    graph_seeds, sequence_seeds = scan.reference_scan.graph_and_sequence_seeds(scan.DISPLAYED_TAU)
    rows = []
    for graph_index in range(start, stop):
        print(f"task {task_id}: starting graph {graph_index}", flush=True)
        rows.extend(
            scan.structure_rows(
                graph_index=graph_index,
                graph_seed=graph_seeds[graph_index],
                sequence_seed=sequence_seeds[graph_index],
            )
        )
        print(f"task {task_id}: finished graph {graph_index}", flush=True)
    partial_dir = scan.OUTPUT_DIR / "partials"
    partial_dir.mkdir(parents=True, exist_ok=True)
    scan.write_csv(rows, partial_dir / f"e1_projector_graphs_{start:04d}_{stop:04d}.csv.gz")


if __name__ == "__main__":
    main()
