"""Additive largest-component repeat of the tau=1.5 eigenvector percentile scan.

This module deliberately writes outside the original percentile-scan directory.
It reuses the original graph and expected-weight seed streams, so every LCC row
has a same-seed full-graph counterpart in the preserved production results.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import chung_lu_eigenvector_percentile_scan as reference_scan


DISPLAYED_TAU = 1.5
COMPONENT_MODE = "largest_connected_component"
NUM_GRAPHS = reference_scan.NUM_GRAPHS
OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "outputs" / "chung_lu" / "power_law" / "eigenvectors"
    / "e1_resonance_lcc_percentile_scan_q0p5_1_width_0p002"
    / "n_5000_tau_1p5_avgdeg_20p0"
)
FULL_GRAPH_BASELINE_DIR = reference_scan.parameter_output_dir(DISPLAYED_TAU)
EXACT_PROJECTOR_AUDIT_DIR = (
    Path(__file__).resolve().parents[2]
    / "outputs" / "chung_lu" / "power_law" / "eigenvectors"
    / "e1_projector_component_scan" / "n_5000_tau_1p5_avgdeg_20p0"
)


def run_chunk(start: int, stop: int) -> tuple[Path, Path]:
    """Run an LCC-only chunk with exactly the production tau=1.5 seeds."""
    if not 0 <= start < stop <= NUM_GRAPHS:
        raise ValueError(f"require 0 <= start < stop <= {NUM_GRAPHS}")
    graph_seeds, sequence_seeds = reference_scan.graph_and_sequence_seeds(DISPLAYED_TAU)
    rows: list[dict[str, object]] = []
    decay_sum = np.zeros(
        (len(reference_scan.LOCAL_QUANTILE_BINS), reference_scan.COORDINATE_DECAY_MAX_RANK),
        dtype=float,
    )
    decay_sum_squares = np.zeros_like(decay_sum)
    graph_counts = np.zeros(len(reference_scan.LOCAL_QUANTILE_BINS), dtype=int)
    for graph_index in range(start, stop):
        graph_rows, coordinate_decay, _ = reference_scan.graph_scan_rows(
            graph_index=graph_index,
            graph_seed=graph_seeds[graph_index],
            sequence_seed=sequence_seeds[graph_index],
            displayed_tau=DISPLAYED_TAU,
            component_mode=COMPONENT_MODE,
        )
        rows.extend(graph_rows)
        decay_sum += coordinate_decay
        decay_sum_squares += np.square(coordinate_decay)
        graph_counts += 1
        print(f"finished LCC graph_index={graph_index}", flush=True)

    partial_dir = OUTPUT_DIR / "partials"
    partial_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{start:04d}_{stop:04d}"
    csv_path = partial_dir / f"eigenvector_scan_graphs_{suffix}.csv.gz"
    npz_path = partial_dir / f"eigenvector_scan_decay_{suffix}.npz"
    reference_scan.write_csv(rows, csv_path, reference_scan.graph_row_fieldnames())
    np.savez_compressed(
        npz_path,
        decay_sum=decay_sum,
        decay_sum_squares=decay_sum_squares,
        graph_counts=graph_counts,
    )
    return csv_path, npz_path


def aggregate_parameter() -> Path:
    """Aggregate the LCC scan and document its preserved full-graph comparator."""
    destination = reference_scan.aggregate_parameter(
        DISPLAYED_TAU,
        output_dir=OUTPUT_DIR,
        component_mode=COMPONENT_MODE,
    )
    manifest_path = destination / "run_manifest.json"
    with manifest_path.open() as handle:
        manifest = json.load(handle)
    manifest.update(
        {
            "output_is_additive": True,
            "audit_question": (
                "Does the q approximately 0.683 / E approximately 1 localization spike "
                "survive after discarding every non-giant component?"
            ),
            "same_seed_full_graph_baseline": str(FULL_GRAPH_BASELINE_DIR),
            "basis_invariant_exact_e1_projector_audit": str(EXACT_PROJECTOR_AUDIT_DIR),
            "paired_graph_seed_stream": True,
            "paired_expected_weight_sequence_seed_stream": True,
            "interpretation": (
                "The percentile scan reproduces the familiar heatmap and four-panel plots. "
                "The separate exact-projector audit is required because ordinary eigenvector "
                "IPR is basis-dependent inside a degenerate E=+/-1 eigenspace."
            ),
        }
    )
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return destination
