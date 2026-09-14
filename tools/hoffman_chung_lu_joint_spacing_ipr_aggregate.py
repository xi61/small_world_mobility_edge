from __future__ import annotations

import csv

from small_world_mobility_edge import chung_lu_joint_spacing_ipr as d


def read_rows(paths):
    rows = []
    for path in paths:
        with path.open(newline="") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    return rows


def main() -> None:
    for tau, target_mean_degree in d.active_parameter_pairs():
        output_dir = d.parameter_output_dir(tau=tau, target_mean_degree=target_mean_degree)
        partial_dir = output_dir / "partials"
        paths = sorted(partial_dir.glob("joint_spacing_ipr_graphs_*.csv"))
        if not paths:
            raise SystemExit(f"no partial CSVs found in {partial_dir}")

        rows = read_rows(paths)
        graph_ids = sorted({int(row["graph_index"]) for row in rows})
        expected = set(range(d.NUM_GRAPHS))
        missing = sorted(expected.difference(graph_ids))
        if missing:
            raise SystemExit(f"missing graph indices: {missing[:20]} (total missing={len(missing)})")

        summary_rows = d.write_outputs(
            rows,
            tau=tau,
            target_mean_degree=target_mean_degree,
            output_dir=output_dir,
            num_partial_files=len(paths),
        )
        print(
            f"aggregated {len(rows)} graph-window rows and {len(summary_rows)} summary rows "
            f"from {len(paths)} partial files into {output_dir}",
            flush=True,
        )


if __name__ == "__main__":
    main()
