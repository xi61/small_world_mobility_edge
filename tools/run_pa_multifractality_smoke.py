#!/usr/bin/env python3
"""Exercise cached multifractal analysis on two small, real affine-PA graphs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np

from small_world_mobility_edge.preferential_attachment.baseline import generate_affine_preferential_attachment_graph
from small_world_mobility_edge.preferential_attachment.multifractality import analyze_case, aggregate_results


def run(output: Path) -> dict:
    config = {
        "analysis_id": "pa_multifractality_smoke",
        "sizes": [64, 128], "alpha_grid": [1.25],
        "moments": {"q_min": 0.0, "q_max": 3.0, "q_step": 0.1},
        "selection": {"band_size": 8, "multiplicity_tolerance_factor": 64.0,
                      "local_gap_radius": 8, "top_r": 8, "high_degree_fraction": 0.05},
        "alpha_histogram": {"minimum": 0.0, "maximum": 6.0, "width": 0.1},
        "log10_scaled_mass_histogram": {"minimum": -16.0, "maximum": 4.5, "width": 0.25},
        "windows": [
            {"id": "central_bulk_q50", "kind": "quantile", "center": 0.5, "count": 8},
            {"id": "positive_edge_nonperron", "kind": "descending_rank", "first_rank": 2, "count": 8},
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "smoke_config.json").write_text(json.dumps(config, indent=2) + "\n")
    for n in config["sizes"]:
        graph, _ = generate_affine_preferential_attachment_graph(n=n, m=3, seed=n + 901, degree_ccdf_alpha=1.25)
        adjacency = nx.to_numpy_array(graph, nodelist=range(n))
        values, vectors = np.linalg.eigh(adjacency)
        residual = float(np.max(np.linalg.norm(adjacency @ vectors - vectors * values, axis=0)))
        if residual > 1e-10:
            raise AssertionError(f"smoke eigenpair residual too large: {residual}")
        source = output / "source" / f"N_{n}" / "alpha_1p25"
        arrays = source / "arrays"
        arrays.mkdir(parents=True, exist_ok=True)
        np.save(arrays / "full_eigenvalues_ascending.npy", values)
        np.save(arrays / "full_eigenvectors_ascending.npy", vectors)
        edges = np.asarray(sorted(graph.edges()), dtype=np.int64)
        np.savez(arrays / "graph_edges.npz", n=n, edge_u=edges[:, 0], edge_v=edges[:, 1],
                 degrees=np.asarray([graph.degree(i) for i in range(n)]))
        (source / "spectral_cache_manifest.json").write_text(json.dumps({
            "spectral_cache_identity": hashlib.sha256(edges.tobytes()).hexdigest(),
            "max_absolute_residual": residual, "graph_seed": n + 901,
        }))
        analyze_case(source_case_dir=source, output_case_dir=output / "cases" / f"N_{n}" / "alpha_1p25",
                     n=n, alpha=1.25, config=config)
    manifest = aggregate_results(output_root=output / "cases", report_dir=output / "report", config=config)
    if manifest["complete_case_count"] != 2 or manifest["tau_row_count"] != 62:
        raise AssertionError("unexpected smoke output dimensions")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output_dir), indent=2))
