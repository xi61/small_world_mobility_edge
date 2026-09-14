from __future__ import annotations

import os

from small_world_mobility_edge.preferential_attachment import full_spectrum_scan as scan


def main() -> None:
    task_id = int(os.environ.get("SGE_TASK_ID", "1"))
    chunk_size = int(os.environ.get("CHUNK_SIZE", "5"))
    tasks_per_parameter = (scan.NUM_GRAPHS + chunk_size - 1) // chunk_size
    zero_based = task_id - 1
    parameter_index = zero_based // tasks_per_parameter
    local_task = zero_based % tasks_per_parameter
    if parameter_index >= len(scan.PREFERENCE_SPECIFICATIONS):
        print("task %d: no attachment parameter assigned" % task_id, flush=True)
        return
    item = scan.PREFERENCE_SPECIFICATIONS[parameter_index]
    start = local_task * chunk_size
    stop = min(start + chunk_size, scan.NUM_GRAPHS)
    if start >= stop:
        print("task %d: empty graph range" % task_id, flush=True)
        return
    print(
        "task %d: alpha=%g delta=%g graphs=[%d,%d)" % (
            task_id,
            item.attachment_tail_exponent,
            item.delta,
            start,
            stop,
        ),
        flush=True,
    )
    destination = scan.run_chunk(item.attachment_tail_exponent, start, stop)
    print("task %d: wrote %s" % (task_id, destination), flush=True)


if __name__ == "__main__":
    main()
