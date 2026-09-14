from __future__ import annotations

import os

from small_world_mobility_edge import chung_lu_eigenvector_lcc_percentile_scan as scan


def main() -> None:
    task_id = int(os.environ.get("SGE_TASK_ID", "1"))
    chunk_size = int(os.environ.get("CHUNK_SIZE", "5"))
    start = (task_id - 1) * chunk_size
    stop = min(start + chunk_size, scan.NUM_GRAPHS)
    if start >= scan.NUM_GRAPHS:
        print(f"task {task_id}: no graphs assigned", flush=True)
        return
    csv_path, npz_path = scan.run_chunk(start, stop)
    print(f"task {task_id}: wrote {csv_path} and {npz_path}", flush=True)


if __name__ == "__main__":
    main()
