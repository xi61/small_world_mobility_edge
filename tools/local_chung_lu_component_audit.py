"""Resumable local runner for the additive Chung-Lu component audits."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path


def ranges(num_graphs: int, chunk_size: int) -> list[tuple[int, int]]:
    if num_graphs < 1 or chunk_size < 1:
        raise ValueError("num_graphs and chunk_size must be positive")
    return [
        (start, min(start + chunk_size, num_graphs))
        for start in range(0, num_graphs, chunk_size)
    ]


def projector_chunk(start: int, stop: int, output_dir: str) -> str:
    from small_world_mobility_edge import chung_lu_e1_projector_core_scan as scan

    destination = Path(output_dir)
    scan.OUTPUT_DIR = destination
    partial = destination / "partials" / f"e1_projector_graphs_{start:04d}_{stop:04d}.csv.gz"
    if partial.exists():
        return f"projector [{start}, {stop}) already complete"
    graph_seeds, sequence_seeds = scan.reference_scan.graph_and_sequence_seeds(scan.DISPLAYED_TAU)
    rows: list[dict[str, object]] = []
    for graph_index in range(start, stop):
        rows.extend(
            scan.structure_rows(
                graph_index=graph_index,
                graph_seed=graph_seeds[graph_index],
                sequence_seed=sequence_seeds[graph_index],
            )
        )
    scan.write_csv(rows, partial)
    return f"projector [{start}, {stop}) complete"


def lcc_chunk(start: int, stop: int, output_dir: str) -> str:
    from small_world_mobility_edge import chung_lu_eigenvector_lcc_percentile_scan as scan

    destination = Path(output_dir)
    scan.OUTPUT_DIR = destination
    partial = destination / "partials" / f"eigenvector_scan_graphs_{start:04d}_{stop:04d}.csv.gz"
    decay = destination / "partials" / f"eigenvector_scan_decay_{start:04d}_{stop:04d}.npz"
    if partial.exists() and decay.exists():
        return f"LCC [{start}, {stop}) already complete"
    scan.run_chunk(start, stop)
    return f"LCC [{start}, {stop}) complete"


def aggregate(mode: str, output_dir: Path, full_baseline_dir: Path | None) -> Path:
    if mode == "projector":
        from small_world_mobility_edge import chung_lu_e1_projector_core_scan as scan

        scan.OUTPUT_DIR = output_dir
        return scan.aggregate_parameter()

    from small_world_mobility_edge import chung_lu_eigenvector_lcc_percentile_scan as scan

    scan.OUTPUT_DIR = output_dir
    if full_baseline_dir is not None:
        scan.FULL_GRAPH_BASELINE_DIR = full_baseline_dir
    return scan.aggregate_parameter()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("projector", "lcc"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--full-baseline-dir", type=Path)
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--chunk-size", type=int, default=2)
    parser.add_argument("--num-graphs", type=int, default=500)
    args = parser.parse_args()
    worker = projector_chunk if args.mode == "projector" else lcc_chunk
    work = ranges(args.num_graphs, args.chunk_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(worker, start, stop, str(args.output_dir)): (start, stop)
            for start, stop in work
        }
        for future in as_completed(futures):
            print(future.result(), flush=True)
    if args.num_graphs == 500:
        destination = aggregate(args.mode, args.output_dir, args.full_baseline_dir)
        print(f"aggregated {args.mode} audit at {destination}", flush=True)


if __name__ == "__main__":
    main()
