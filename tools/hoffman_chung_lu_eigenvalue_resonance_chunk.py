from __future__ import annotations

import os

from small_world_mobility_edge import chung_lu_eigenvalue_resonance_scan as scan


def main() -> None:
    task_id = int(os.environ.get("SGE_TASK_ID", "1"))
    chunk_size = int(os.environ.get("CHUNK_SIZE", "5"))
    tasks_per_tau = (scan.NUM_GRAPHS + chunk_size - 1) // chunk_size
    zero_based_task = task_id - 1
    tau_index = zero_based_task // tasks_per_tau
    local_task = zero_based_task % tasks_per_tau
    if tau_index >= len(scan.eigenvector_scan.TAU_SPECIFICATIONS):
        print(f"task {task_id}: no tau assigned", flush=True)
        return

    specification = scan.eigenvector_scan.TAU_SPECIFICATIONS[tau_index]
    start = local_task * chunk_size
    stop = min(start + chunk_size, scan.NUM_GRAPHS)
    if start >= scan.NUM_GRAPHS:
        print(f"task {task_id}: no graph indices assigned", flush=True)
        return
    graph_seeds, sequence_seeds = scan.eigenvector_scan.graph_and_sequence_seeds(
        specification.displayed_tau
    )
    output_dir = scan.parameter_output_dir(specification.displayed_tau)
    partial_dir = output_dir / "partials"
    partial_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    print(
        f"task {task_id}: displayed_tau={specification.displayed_tau}, graphs [{start}, {stop})",
        flush=True,
    )
    for graph_index in range(start, stop):
        rows.extend(
            scan.graph_scan_rows(
                graph_index=graph_index,
                graph_seed=graph_seeds[graph_index],
                sequence_seed=sequence_seeds[graph_index],
                displayed_tau=specification.displayed_tau,
            )
        )
        print(f"task {task_id}: finished graph_index={graph_index}", flush=True)
    suffix = f"{start:04d}_{stop:04d}"
    scan.write_csv(
        rows,
        partial_dir / f"eigenvalue_resonance_graphs_{suffix}.csv.gz",
        scan.graph_row_fieldnames(),
    )
    print(f"task {task_id}: wrote partial for graphs [{start}, {stop})", flush=True)


if __name__ == "__main__":
    main()
