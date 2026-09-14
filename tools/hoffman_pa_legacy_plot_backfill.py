from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path

from small_world_mobility_edge.preferential_attachment.mobility_edge_cached_pipeline import (
    plot_coarse,
    slug,
)


def main() -> None:
    root = Path(
        os.environ.get(
            "PA_OUTPUT_ROOT",
            "outputs/preferential_attachment/mobility_edge",
        )
    )
    for alpha in (1.8, 1.5, 1.05):
        parameter_dir = (
            root
            / "affine_linear"
            / "n_5000_m_10"
            / f"degree_ccdf_alpha_{slug(alpha)}"
        )
        joined_path = (
            parameter_dir
            / "coarse"
            / "q0_1_dq_0p002"
            / "joined_localization_level_statistics.csv"
        )
        with joined_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        plot_coarse(rows, alpha, parameter_dir)
        source = json.loads((parameter_dir / "source_cache.json").read_text())
        source_dir = Path(source["source_cache"])
        legacy_heatmap = source_dir / "eigenvector_coordinate_mass_heatmap.png"
        if legacy_heatmap.is_file():
            shutil.copyfile(
                legacy_heatmap,
                parameter_dir
                / "coarse"
                / "q0_1_dq_0p002"
                / "eigenvector_coordinate_mass_heatmap_legacy_reused.png",
            )
        print(f"backfilled localization plots for alpha={alpha:g}", flush=True)


if __name__ == "__main__":
    main()
