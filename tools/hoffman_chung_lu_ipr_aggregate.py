from __future__ import annotations

import csv
import json
from datetime import datetime, timezone

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from small_world_mobility_edge import chung_lu_eigenvector_diagnostics as d


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
        paths = sorted(partial_dir.glob("eigenvector_localization_graphs_*.csv"))
        decay_paths = sorted(partial_dir.glob("eigenvector_coordinate_decay_graphs_*.csv"))
        if not paths:
            raise SystemExit(f"no partial CSVs found in {partial_dir}")
        rows = read_rows(paths)
        graph_ids = sorted({int(row["graph_index"]) for row in rows})
        expected = set(range(d.NUM_GRAPHS))
        missing = sorted(expected.difference(graph_ids))
        if missing:
            raise SystemExit(f"missing graph indices: {missing[:20]} (total missing={len(missing)})")

        summary_rows = d.aggregate_quantile_rows(rows)
        d.write_csv(rows, output_dir / "eigenvector_localization_by_graph.csv", d.graph_row_fieldnames())
        d.write_csv(summary_rows, output_dir / "eigenvector_localization_summary.csv", d.summary_row_fieldnames())
        fig = d.plot_localization_summary(summary_rows, tau=tau, target_mean_degree=target_mean_degree)
        fig.savefig(output_dir / "eigenvector_localization_summary.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        decay_summary_rows = []
        if decay_paths:
            decay_rows = read_rows(decay_paths)
            decay_summary_rows = d.aggregate_coordinate_decay_rows(decay_rows)
            d.write_csv(
                decay_rows,
                output_dir / "eigenvector_coordinate_decay_by_graph.csv",
                d.coordinate_decay_row_fieldnames(),
            )
            d.write_csv(
                decay_summary_rows,
                output_dir / "eigenvector_coordinate_decay_summary.csv",
                d.coordinate_decay_summary_fieldnames(),
            )
            decay_fig = d.plot_coordinate_decay_summary(
                decay_summary_rows,
                tau=tau,
                target_mean_degree=target_mean_degree,
            )
            decay_fig.savefig(output_dir / "eigenvector_coordinate_decay_summary.png", dpi=200, bbox_inches="tight")
            plt.close(decay_fig)

        manifest = {
            "model": "power_law_chung_lu",
            "diagnostic": "eigenvector_localization_chunked_aggregate",
            "n": d.N,
            "tau": tau,
            "target_mean_degree": target_mean_degree,
            "num_graphs": d.NUM_GRAPHS,
            "window_mode": d.WINDOW_MODE,
            "local_quantile_bins": d.LOCAL_QUANTILE_BINS,
            "coordinate_decay_max_rank": d.COORDINATE_DECAY_MAX_RANK,
            "active_parameter_pairs": d.ACTIVE_PARAMETER_PAIRS,
            "run_full_grid": d.RUN_FULL_GRID,
            "num_partial_files": len(paths),
            "num_coordinate_decay_partial_files": len(decay_paths),
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "code_version": d.code_version(),
        }
        with (output_dir / "run_manifest.json").open("w") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        print(
            f"aggregated {len(rows)} localization rows and {len(decay_summary_rows)} decay summary rows "
            f"from {len(paths)} partial files into {output_dir}"
        )


if __name__ == "__main__":
    main()
