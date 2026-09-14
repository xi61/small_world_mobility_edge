from __future__ import annotations

import os
from pathlib import Path

from small_world_mobility_edge import chung_lu_eigenvector_diagnostics as d


def main() -> None:
    task_id = int(os.environ.get("SGE_TASK_ID", "1"))
    chunk_size = int(os.environ.get("CHUNK_SIZE", "5"))
    start = (task_id - 1) * chunk_size
    stop = min(start + chunk_size, d.NUM_GRAPHS)
    if start >= d.NUM_GRAPHS:
        print(f"task {task_id}: no graph indices for start={start}, num_graphs={d.NUM_GRAPHS}")
        return

    for tau, target_mean_degree in d.active_parameter_pairs():
        output_dir = d.parameter_output_dir(tau=tau, target_mean_degree=target_mean_degree)
        partial_dir = output_dir / "partials"
        partial_dir.mkdir(parents=True, exist_ok=True)
        offset = d.parameter_seed_offset(tau=tau, target_mean_degree=target_mean_degree)
        graph_seeds = d.make_seeds(d.NUM_GRAPHS, d.GRAPH_SEED + offset)
        sequence_seeds = d.make_seeds(d.NUM_GRAPHS, d.SEQUENCE_SEED + offset)

        rows = []
        decay_rows = []
        print(
            f"task {task_id}: tau={tau}, degree={target_mean_degree}, "
            f"graphs [{start}, {stop}) of {d.NUM_GRAPHS}",
            flush=True,
        )
        for graph_index in range(start, stop):
            graph_rows, graph_decay_rows, _ = d.graph_localization_and_decay_rows(
                graph_index=graph_index,
                graph_seed=graph_seeds[graph_index],
                sequence_seed=sequence_seeds[graph_index],
                tau=tau,
                target_mean_degree=target_mean_degree,
            )
            rows.extend(graph_rows)
            decay_rows.extend(graph_decay_rows)
            print(f"task {task_id}: finished graph_index={graph_index}", flush=True)

        partial_path = partial_dir / f"eigenvector_localization_graphs_{start:04d}_{stop:04d}.csv"
        d.write_csv(rows, partial_path, d.graph_row_fieldnames())
        decay_partial_path = partial_dir / f"eigenvector_coordinate_decay_graphs_{start:04d}_{stop:04d}.csv"
        d.write_csv(decay_rows, decay_partial_path, d.coordinate_decay_row_fieldnames())
        print(f"task {task_id}: wrote {partial_path}", flush=True)
        print(f"task {task_id}: wrote {decay_partial_path}", flush=True)


if __name__ == "__main__":
    main()
