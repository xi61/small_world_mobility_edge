from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from small_world_mobility_edge.preferential_attachment.full_spectrum_scan import (
    structural_replay_paths,
    validate_structural_replay_chunk,
)
from small_world_mobility_edge.preferential_attachment.mobility_edge_cached_pipeline import (
    load_legacy_spectra,
)
from small_world_mobility_edge.preferential_attachment.mobility_edge_production_aggregate import (
    _read_gzip_csv,
    degree_and_tail_validation,
)
from hoffman_pa_mobility_edge_chunk import config_from_environment


def main() -> None:
    config = config_from_environment()
    chunk_size = int(os.environ.get("PA_CHUNK_SIZE", "5"))
    source_root = Path(os.environ["PA_LEGACY_SOURCE_ROOT"])
    spectra, seeds, _ = load_legacy_spectra(
        source_root, config.canonical_parameter_value
    )
    graphs = []
    structural = []
    for start in range(0, config.num_graphs, chunk_size):
        stop = min(start + chunk_size, config.num_graphs)
        valid, reason = validate_structural_replay_chunk(config, start, stop)
        if not valid:
            raise SystemExit(f"invalid structural replay [{start},{stop}): {reason}")
        degree_path, structural_path = structural_replay_paths(config, start, stop)
        structural.extend(_read_gzip_csv(structural_path))
        with np.load(degree_path, allow_pickle=False) as data:
            indices = np.asarray(data["graph_indices"], dtype=int)
            degree_arrays = np.asarray(data["degree_arrays"], dtype=np.int32)
            degree_orders = np.asarray(data["degree_orderings"], dtype=np.int32)
        for local, graph_index in enumerate(indices):
            index = int(graph_index)
            graphs.append(
                {
                    "graph_index": index,
                    "graph_seed": seeds[index],
                    "nonperron_eigenvalues": spectra[index],
                    "degrees": degree_arrays[local],
                    "degree_order": degree_orders[local],
                }
            )
    graphs.sort(key=lambda row: int(row["graph_index"]))
    structural.sort(key=lambda row: int(row["graph_index"]))
    summary = degree_and_tail_validation(
        config, graphs, structural, config.parameter_dir
    )
    manifest_path = config.parameter_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    manifest["structural_replay"] = {
        "completed": True,
        "without_eigendecomposition": True,
        "source_cache": str(source_root),
        "partial_dir": str(config.parameter_dir / "structural" / "replay_partials"),
        "summary": summary,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"aggregated structural replay into {config.parameter_dir}", flush=True)


if __name__ == "__main__":
    main()
