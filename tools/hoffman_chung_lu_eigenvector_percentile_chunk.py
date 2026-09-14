from __future__ import annotations

import os

import numpy as np

from small_world_mobility_edge import chung_lu_eigenvector_percentile_scan as scan


def main() -> None:
    task_id = int(os.environ.get("SGE_TASK_ID", "1"))
    chunk_size = int(os.environ.get("CHUNK_SIZE", "5"))
    tasks_per_tau = (scan.NUM_GRAPHS + chunk_size - 1) // chunk_size
    zero_based_task = task_id - 1
    tau_index = zero_based_task // tasks_per_tau
    local_task = zero_based_task % tasks_per_tau
    if tau_index >= len(scan.TAU_SPECIFICATIONS):
        print(f"task {task_id}: no tau assigned", flush=True)
        return

    specification = scan.TAU_SPECIFICATIONS[tau_index]
    start = local_task * chunk_size
    stop = min(start + chunk_size, scan.NUM_GRAPHS)
    if start >= scan.NUM_GRAPHS:
        print(f"task {task_id}: no graph indices assigned", flush=True)
        return

    graph_seeds, sequence_seeds = scan.graph_and_sequence_seeds(specification.displayed_tau)
    output_dir = scan.parameter_output_dir(specification.displayed_tau)
    partial_dir = output_dir / "partials"
    partial_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    decay_sum = np.zeros(
        (len(scan.LOCAL_QUANTILE_BINS), scan.COORDINATE_DECAY_MAX_RANK),
        dtype=float,
    )
    decay_sum_squares = np.zeros_like(decay_sum)
    graph_counts = np.zeros(len(scan.LOCAL_QUANTILE_BINS), dtype=int)
    print(
        f"task {task_id}: displayed_tau={specification.displayed_tau}, "
        f"effective_density_exponent={specification.effective_density_exponent}, "
        f"graphs [{start}, {stop})",
        flush=True,
    )
    for graph_index in range(start, stop):
        graph_rows, coordinate_decay, _ = scan.graph_scan_rows(
            graph_index=graph_index,
            graph_seed=graph_seeds[graph_index],
            sequence_seed=sequence_seeds[graph_index],
            displayed_tau=specification.displayed_tau,
        )
        rows.extend(graph_rows)
        decay_sum += coordinate_decay
        decay_sum_squares += np.square(coordinate_decay)
        graph_counts += 1
        print(f"task {task_id}: finished graph_index={graph_index}", flush=True)

    suffix = f"{start:04d}_{stop:04d}"
    scan.write_csv(
        rows,
        partial_dir / f"eigenvector_scan_graphs_{suffix}.csv.gz",
        scan.graph_row_fieldnames(),
    )
    np.savez_compressed(
        partial_dir / f"eigenvector_scan_decay_{suffix}.npz",
        decay_sum=decay_sum,
        decay_sum_squares=decay_sum_squares,
        graph_counts=graph_counts,
    )
    print(f"task {task_id}: wrote partials for graphs [{start}, {stop})", flush=True)


if __name__ == "__main__":
    main()
