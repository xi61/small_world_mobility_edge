from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from small_world_mobility_edge.preferential_attachment.abnormal_edge_followup import (
    MODEL_BY_KEY,
    SparseEdgeConfig,
    build_positive_edge_same_triplet_records,
    goe_ratio_cdf,
    poisson_ratio_cdf,
    rank_crossfit_unfolded_gaps,
    run_sparse_chunk,
    solve_sparse_graph_edges,
    validate_sparse_chunk,
)
from small_world_mobility_edge.preferential_attachment.mobility_edge_postprocess import (
    build_gap_records,
)


def _linear_crossfit_model(graph_id: int) -> SimpleNamespace:
    density = SimpleNamespace(
        grid=np.asarray([-10.0, 0.0, 10.0]),
        cdf=np.asarray([0.0, 0.2, 1.0]),
    )
    return SimpleNamespace(
        fold=graph_id % 2,
        training_graph_ids=(1 - graph_id,),
        held_out_graph_ids=(graph_id,),
        density_model=density,
    )


def test_same_triplet_raw_and_unfolded_are_linked() -> None:
    record = build_gap_records(
        np.asarray([-3.0, -2.0, 1.0, 2.0, 4.0, 7.0]),
        graph_index=0,
        graph_seed=17,
    )
    mapped = build_positive_edge_same_triplet_records(
        [record], [_linear_crossfit_model(0)], q_min=0.0
    )
    assert mapped["graph_index"].tolist() == [0, 0]
    assert np.allclose(mapped["raw_left_gap"], [1.0, 2.0])
    assert np.allclose(mapped["raw_right_gap"], [2.0, 3.0])
    assert np.allclose(mapped["raw_r"], [0.5, 2.0 / 3.0])
    # The CDF is linear on the positive interval, so unfolding preserves r.
    assert np.allclose(mapped["unfolded_r"], mapped["raw_r"])
    assert mapped["edge_rank"].tolist() == [3, 2]


def test_rank_crossfit_uses_training_folds_only() -> None:
    graph_ids = np.arange(10)
    gaps = np.column_stack((graph_ids + 1.0, 2.0 * graph_ids + 2.0))
    unfolded = rank_crossfit_unfolded_gaps(gaps, graph_ids, num_folds=5)
    for fold in range(5):
        held = graph_ids % 5 == fold
        training = ~held
        expected_scale = np.mean(gaps[training], axis=0)
        assert np.allclose(unfolded[held], gaps[held] / expected_scale)


def test_ratio_reference_cdfs_are_normalized_and_monotone() -> None:
    grid = np.linspace(0.0, 1.0, 101)
    for cdf in (goe_ratio_cdf(grid), poisson_ratio_cdf(grid)):
        assert np.isclose(cdf[0], 0.0)
        assert np.isclose(cdf[-1], 1.0)
        assert np.all(np.diff(cdf) >= 0.0)


def test_sparse_edge_solver_and_content_validation(tmp_path: Path) -> None:
    config = SparseEdgeConfig(
        model=MODEL_BY_KEY["affine_alpha_2p0"],
        n=80,
        m=3,
        num_graphs=1,
        output_root=tmp_path,
        rank_count=6,
        eigsh_tolerance=1e-10,
    )
    result = solve_sparse_graph_edges(config, 0)
    assert result["positive_eigenvalue"].shape == (7,)
    assert result["negative_eigenvalue"].shape == (7,)
    assert np.all(np.diff(result["positive_eigenvalue"]) <= 0.0)
    assert np.all(np.diff(result["negative_eigenvalue"]) >= 0.0)
    assert np.max(result["positive_relative_residual"]) < 1e-7
    assert np.max(result["negative_relative_residual"]) < 1e-7
    path = run_sparse_chunk(config, 0, 1)
    assert path.is_file()
    assert validate_sparse_chunk(config, 0, 1) == (True, "VALID")
    # A valid cache is resume-safe and remains unchanged on a second call.
    before = path.stat().st_mtime_ns
    run_sparse_chunk(config, 0, 1)
    assert path.stat().st_mtime_ns == before
