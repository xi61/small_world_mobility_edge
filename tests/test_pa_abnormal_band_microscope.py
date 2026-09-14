from __future__ import annotations

import numpy as np

from small_world_mobility_edge.preferential_attachment.abnormal_edge_followup import (
    MODEL_BY_KEY,
    summarize_q_window,
)
from tools.hoffman_pa_abnormal_band_microscope import make_bins, negative_records


def test_requested_grid_sizes_and_shift() -> None:
    assert len(make_bins(0.994, 0.999, 0.0002)) == 25
    assert len(make_bins(0.9945, 0.9985, 0.0001)) == 40
    assert len(make_bins(0.995, 0.9975, 0.00005)) == 50
    assert len(make_bins(0.995, 0.9985, 0.00005)) == 70
    assert make_bins(0.996, 0.9963, 0.0001, True)[0] == (0.99605, 0.99615)


def test_separate_histogram_and_ratio_support_gates() -> None:
    graph_ids = np.arange(120)
    triplets = {
        "graph_index": graph_ids,
        "ensemble_q": np.full(120, 0.99605),
        "raw_r": np.full(120, 0.5),
        "unfolded_r": np.full(120, 0.51),
        "raw_left_gap": np.ones(120),
        "raw_right_gap": np.full(120, 2.0),
        "unfolded_left_gap": np.ones(120),
        "unfolded_right_gap": np.full(120, 1.8),
        "edge_rank": np.full(120, 12),
        "middle_energy": np.full(120, 30.0),
        "relative_min_gap": np.full(120, 1e-3),
    }
    gaps = {
        "graph_index": graph_ids,
        "ensemble_q": np.full(120, 0.99605),
        "raw_gap": np.ones(120),
        "unfolded_gap": np.ones(120),
        "left_energy": np.full(120, 29.5),
        "right_energy": np.full(120, 30.5),
    }
    row = summarize_q_window(
        triplets,
        gaps,
        model=MODEL_BY_KEY["affine_alpha_1p05"],
        q_low=0.996,
        q_high=0.9961,
        bootstrap_replicates=20,
    )
    assert row["histogram_support"] is True
    assert row["ratio_support"] is True
    assert row["gap_contributing_graphs"] == 120
    assert row["ratio_contributing_graphs"] == 120
    assert row["edge_rank_p10"] == 12
    assert row["edge_rank_p90"] == 12


def test_negative_edge_records_use_mirrored_q_and_same_triplets() -> None:
    graph_ids = np.arange(2)
    eigenvalues = np.tile(np.linspace(-40.0, -10.0, 121), (2, 1))
    q = np.tile(np.linspace(0.999, 0.98, 121), (2, 1))
    dataset = {
        "graph_indices": graph_ids,
        "graph_seeds": np.asarray([11, 12], dtype=np.uint64),
        "negative_eigenvalue": eigenvalues,
        "negative_ensemble_q": q,
        "negative_cdf_unfolded_gap": np.ones((2, 120)),
    }
    triplets, gaps = negative_records(dataset)
    assert len(gaps["raw_gap"]) == 240
    assert len(triplets["raw_r"]) == 238
    assert np.all(triplets["edge_rank"].reshape(2, -1)[:, 0] == 2)
    assert np.allclose(triplets["raw_r"], 1.0)
    assert np.allclose(triplets["unfolded_r"], 1.0)
