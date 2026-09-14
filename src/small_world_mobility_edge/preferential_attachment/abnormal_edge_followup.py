"""Focused diagnostics for abnormal preferential-attachment edge statistics.

This module is deliberately isolated from the authoritative mobility-edge v2
pipeline.  It reuses that pipeline's graph generator, atom handling, density
estimator, and deterministic seed construction, but writes only diagnostic
products.  The two principal data paths are:

* reanalysis of complete cached spectra at N=5000; and
* sparse extreme-eigenpair caches for finite-size fixed-rank comparisons.

Every adjacent-gap ratio is formed from two gaps in one graph and one spectral
segment.  Rank-window ratios are assigned to the middle eigenvalue of the
triplet.  Consequently edge rank 1 has no ratio of its own.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from .baseline import (
    MODEL_AFFINE_PA,
    MODEL_NONLINEAR_PA,
    generate_preferential_attachment_graph,
)
from .full_spectrum_scan import make_seeds, slug_number
from .mobility_edge_postprocess import (
    GOE_MEAN_GAP_RATIO,
    POISSON_MEAN_GAP_RATIO,
    GraphGapRecords,
    ks_distance_unit_mean_spacing,
)


FOLLOWUP_ANALYSIS_VERSION = "pa_abnormal_edge_followup_v1"
SPARSE_CACHE_VERSION = "pa_abnormal_edge_sparse_rank_cache_v1"
RANK_WINDOWS = ((1, 5), (6, 10), (11, 20), (21, 40), (41, 80), (81, 120))
DEFAULT_EDGE_RANK_COUNT = 120
DEFAULT_RELATIVE_GAP_TOLERANCES = (1e-8, 1e-10, 1e-12)
ZERO_GAP_ATOL = 1e-14


@dataclass(frozen=True)
class FollowupModel:
    key: str
    label: str
    model_family: str
    parameter_name: str
    parameter_value: float
    finite_size_seed_base: int

    @property
    def parameter_slug(self) -> str:
        return f"{self.parameter_name}_{slug_number(self.parameter_value)}"


PRIORITY_MODELS = (
    FollowupModel(
        "affine_alpha_2p0",
        "affine alpha=2.00",
        "affine_linear",
        "degree_ccdf_alpha",
        2.0,
        63100,
    ),
    FollowupModel(
        "affine_alpha_1p25",
        "affine alpha=1.25",
        "affine_linear",
        "degree_ccdf_alpha",
        1.25,
        63110,
    ),
    FollowupModel(
        "affine_alpha_1p05",
        "affine alpha=1.05",
        "affine_linear",
        "degree_ccdf_alpha",
        1.05,
        63120,
    ),
    FollowupModel(
        "superlinear_zeta_1p5",
        "superlinear zeta=1.50",
        "nonlinear",
        "attachment_zeta",
        1.5,
        63130,
    ),
)
MODEL_BY_KEY = {model.key: model for model in PRIORITY_MODELS}


@dataclass(frozen=True)
class SparseEdgeConfig:
    model: FollowupModel
    n: int
    m: int
    num_graphs: int
    output_root: Path
    rank_count: int = DEFAULT_EDGE_RANK_COUNT
    eigsh_tolerance: float = 1e-9
    eigsh_maxiter_factor: int = 30

    def __post_init__(self) -> None:
        if not 1 <= self.m < self.n:
            raise ValueError("requires 1 <= m < n")
        if self.num_graphs < 1:
            raise ValueError("num_graphs must be positive")
        if not 3 <= self.rank_count < self.n - 2:
            raise ValueError("rank_count is incompatible with n")
        if self.eigsh_tolerance <= 0.0 or self.eigsh_maxiter_factor < 1:
            raise ValueError("invalid eigsh controls")

    @property
    def parameter_dir(self) -> Path:
        return (
            Path(self.output_root)
            / "finite_size"
            / f"n_{self.n}"
            / self.model.model_family
            / self.model.parameter_slug
        )

    @property
    def partial_dir(self) -> Path:
        return self.parameter_dir / "partials"

    def graph_seeds(self) -> list[int]:
        return make_seeds(self.num_graphs, self.model.finite_size_seed_base)

    def metadata(self) -> dict[str, object]:
        return {
            "analysis_version": SPARSE_CACHE_VERSION,
            "model_key": self.model.key,
            "model_label": self.model.label,
            "model_family": self.model.model_family,
            "parameter_name": self.model.parameter_name,
            "parameter_value": self.model.parameter_value,
            "n": self.n,
            "m": self.m,
            "num_graphs": self.num_graphs,
            "graph_seed_base": self.model.finite_size_seed_base,
            "rank_count": self.rank_count,
            "stored_eigenpairs_per_side": self.rank_count + 1,
            "positive_perron_removed": True,
            "negative_edge_order": "most negative to inner",
            "positive_edge_order": "largest non-Perron to inner",
            "eigenvectors_retained_as_scalars_only": True,
            "eigsh_tolerance": self.eigsh_tolerance,
            "eigsh_maxiter_factor": self.eigsh_maxiter_factor,
        }


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _window_mask(values: np.ndarray, low: float, high: float) -> np.ndarray:
    if high >= 1.0 - 1e-15:
        return (values >= low) & (values <= high)
    return (values >= low) & (values < high)


def _ratio(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    maximum = np.maximum(left, right)
    output = np.full(np.broadcast_shapes(left.shape, right.shape), np.nan, dtype=float)
    valid = (left > ZERO_GAP_ATOL) & (right > ZERO_GAP_ATOL) & (maximum > ZERO_GAP_ATOL)
    output[valid] = np.minimum(left[valid], right[valid]) / maximum[valid]
    return output


def graph_equal_summary(
    graph_ids: np.ndarray,
    values: np.ndarray,
) -> tuple[float, float, int, np.ndarray, np.ndarray]:
    graph_ids = np.asarray(graph_ids, dtype=int)
    values = np.asarray(values, dtype=float)
    means: list[float] = []
    ids: list[int] = []
    for graph_id in np.unique(graph_ids):
        selected = (graph_ids == graph_id) & np.isfinite(values)
        if np.any(selected):
            ids.append(int(graph_id))
            means.append(float(np.mean(values[selected])))
    array = np.asarray(means, dtype=float)
    mean = float(np.mean(array)) if len(array) else math.nan
    se = (
        float(np.std(array, ddof=1) / math.sqrt(len(array)))
        if len(array) > 1
        else math.nan
    )
    return mean, se, len(array), np.asarray(ids, dtype=int), array


def graph_bootstrap_ci(
    graph_ids: np.ndarray,
    values: np.ndarray,
    *,
    seed: int,
    replicates: int = 2000,
) -> tuple[float, float]:
    _, _, _, _, graph_means = graph_equal_summary(graph_ids, values)
    if not len(graph_means):
        return math.nan, math.nan
    if len(graph_means) == 1:
        return float(graph_means[0]), float(graph_means[0])
    generator = np.random.default_rng(seed)
    sample_indices = generator.integers(
        0, len(graph_means), size=(replicates, len(graph_means))
    )
    means = np.mean(graph_means[sample_indices], axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def poisson_ratio_cdf(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    return 2.0 * values / (1.0 + values)


_GOE_RATIO_GRID = np.linspace(0.0, 1.0, 20001)
_GOE_RATIO_PDF = (
    27.0
    / 4.0
    * (_GOE_RATIO_GRID + np.square(_GOE_RATIO_GRID))
    / np.power(1.0 + _GOE_RATIO_GRID + np.square(_GOE_RATIO_GRID), 2.5)
)
_GOE_RATIO_CDF = np.concatenate(
    (
        [0.0],
        np.cumsum(
            0.5
            * (_GOE_RATIO_PDF[:-1] + _GOE_RATIO_PDF[1:])
            * np.diff(_GOE_RATIO_GRID)
        ),
    )
)
_GOE_RATIO_CDF /= _GOE_RATIO_CDF[-1]


def goe_ratio_cdf(values: np.ndarray) -> np.ndarray:
    return np.interp(
        np.clip(np.asarray(values, dtype=float), 0.0, 1.0),
        _GOE_RATIO_GRID,
        _GOE_RATIO_CDF,
    )


def ks_distance(values: np.ndarray, cdf: Callable[[np.ndarray], np.ndarray]) -> float:
    sample = np.sort(np.asarray(values, dtype=float))
    sample = sample[np.isfinite(sample)]
    if not len(sample):
        return math.nan
    reference = cdf(sample)
    high = np.arange(1, len(sample) + 1, dtype=float) / len(sample)
    low = np.arange(0, len(sample), dtype=float) / len(sample)
    return float(max(np.max(np.abs(high - reference)), np.max(np.abs(low - reference))))


def correlation_from_pairs(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    if np.sum(valid) < 2:
        return math.nan
    x = left[valid]
    y = right[valid]
    if np.std(x) <= 0.0 or np.std(y) <= 0.0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def _spacing_pair_sufficient_statistics(
    graph_ids: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    ids: list[int] = []
    for graph_id in np.unique(graph_ids):
        mask = (graph_ids == graph_id) & np.isfinite(left) & np.isfinite(right)
        x = left[mask]
        y = right[mask]
        if len(x):
            ids.append(int(graph_id))
            rows.append(
                [
                    len(x),
                    float(np.sum(x)),
                    float(np.sum(y)),
                    float(np.dot(x, x)),
                    float(np.dot(y, y)),
                    float(np.dot(x, y)),
                ]
            )
    return np.asarray(ids, dtype=int), np.asarray(rows, dtype=float)


def _correlation_from_sufficient(statistics: np.ndarray) -> float:
    if statistics.ndim != 2 or statistics.shape[1] != 6 or not len(statistics):
        return math.nan
    n, sx, sy, sxx, syy, sxy = np.sum(statistics, axis=0)
    if n < 2:
        return math.nan
    covariance = sxy - sx * sy / n
    variance_x = sxx - sx * sx / n
    variance_y = syy - sy * sy / n
    if variance_x <= 0.0 or variance_y <= 0.0:
        return math.nan
    return float(covariance / math.sqrt(variance_x * variance_y))


def correlation_bootstrap_ci(
    graph_ids: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    seed: int,
    replicates: int = 1000,
) -> tuple[float, float]:
    _, statistics = _spacing_pair_sufficient_statistics(graph_ids, left, right)
    if not len(statistics):
        return math.nan, math.nan
    if len(statistics) == 1:
        value = _correlation_from_sufficient(statistics)
        return value, value
    generator = np.random.default_rng(seed)
    output = np.empty(replicates, dtype=float)
    for index in range(replicates):
        selected = generator.integers(0, len(statistics), size=len(statistics))
        output[index] = _correlation_from_sufficient(statistics[selected])
    output = output[np.isfinite(output)]
    if not len(output):
        return math.nan, math.nan
    low, high = np.quantile(output, [0.025, 0.975])
    return float(low), float(high)


def crossfit_model_mapping(models: Sequence[object]) -> dict[int, object]:
    mapping: dict[int, object] = {}
    for model in models:
        training = set(model.training_graph_ids)
        held_out = set(model.held_out_graph_ids)
        if training & held_out:
            raise ValueError(f"cross-fit leakage in fold {model.fold}")
        for graph_id in held_out:
            if graph_id in mapping:
                raise ValueError(f"graph {graph_id} is held out in multiple folds")
            mapping[int(graph_id)] = model
    return mapping


def build_positive_edge_same_triplet_records(
    records: Sequence[GraphGapRecords],
    crossfit_models: Sequence[object],
    *,
    q_min: float = 0.9915,
) -> dict[str, np.ndarray]:
    """Build raw and unfolded ratios from the exact same eigenvalue triplet."""
    mapping = crossfit_model_mapping(crossfit_models)
    output: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "graph_index",
            "graph_seed",
            "fold",
            "middle_energy",
            "ensemble_q",
            "edge_rank",
            "raw_left_gap",
            "raw_right_gap",
            "raw_r",
            "unfolded_left_gap",
            "unfolded_right_gap",
            "unfolded_r",
            "relative_min_gap",
        )
    }
    for record in records:
        positive = record.continuous_eigenvalues[record.continuous_eigenvalues > 0.0]
        if len(positive) < 3:
            continue
        density = mapping[record.graph_index].density_model
        cdf = np.interp(positive, density.grid, density.cdf, left=0.0, right=1.0)
        middle_q = cdf[1:-1]
        keep = middle_q >= q_min
        if not np.any(keep):
            continue
        raw_left = np.diff(positive)[:-1]
        raw_right = np.diff(positive)[1:]
        unfolded_spacing = record.continuous_dimension * np.diff(cdf)
        unfolded_left = unfolded_spacing[:-1]
        unfolded_right = unfolded_spacing[1:]
        raw_r = _ratio(raw_left, raw_right)
        unfolded_r = _ratio(unfolded_left, unfolded_right)
        middle_energy = positive[1:-1]
        relative = np.minimum(raw_left, raw_right) / np.maximum(
            np.maximum(np.abs(middle_energy), np.abs(positive[2:])), 1.0
        )
        count = int(np.sum(keep))
        output["graph_index"].append(
            np.full(count, record.graph_index, dtype=np.int32)
        )
        output["graph_seed"].append(
            np.full(count, record.graph_seed, dtype=np.uint64)
        )
        output["fold"].append(
            np.full(count, mapping[record.graph_index].fold, dtype=np.int8)
        )
        output["middle_energy"].append(middle_energy[keep])
        output["ensemble_q"].append(middle_q[keep])
        edge_rank = len(positive) - np.arange(1, len(positive) - 1)
        output["edge_rank"].append(edge_rank[keep].astype(np.int32))
        output["raw_left_gap"].append(raw_left[keep])
        output["raw_right_gap"].append(raw_right[keep])
        output["raw_r"].append(raw_r[keep])
        output["unfolded_left_gap"].append(unfolded_left[keep])
        output["unfolded_right_gap"].append(unfolded_right[keep])
        output["unfolded_r"].append(unfolded_r[keep])
        output["relative_min_gap"].append(relative[keep])
    return {
        name: np.concatenate(chunks) if chunks else np.array([], dtype=float)
        for name, chunks in output.items()
    }


def build_positive_edge_gap_records(
    records: Sequence[GraphGapRecords],
    crossfit_models: Sequence[object],
    *,
    q_min: float = 0.9915,
) -> dict[str, np.ndarray]:
    mapping = crossfit_model_mapping(crossfit_models)
    output: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "graph_index",
            "ensemble_q",
            "edge_rank",
            "left_energy",
            "right_energy",
            "raw_gap",
            "unfolded_gap",
        )
    }
    for record in records:
        positive = record.continuous_eigenvalues[record.continuous_eigenvalues > 0.0]
        if len(positive) < 2:
            continue
        density = mapping[record.graph_index].density_model
        left = positive[:-1]
        right = positive[1:]
        raw = right - left
        midpoint = 0.5 * (left + right)
        q = np.interp(midpoint, density.grid, density.cdf, left=0.0, right=1.0)
        unfolded = record.continuous_dimension * (
            np.interp(right, density.grid, density.cdf, left=0.0, right=1.0)
            - np.interp(left, density.grid, density.cdf, left=0.0, right=1.0)
        )
        keep = q >= q_min
        if not np.any(keep):
            continue
        count = int(np.sum(keep))
        output["graph_index"].append(
            np.full(count, record.graph_index, dtype=np.int32)
        )
        output["ensemble_q"].append(q[keep])
        output["edge_rank"].append(
            (len(positive) - np.arange(len(raw)) - 1)[keep].astype(np.int32)
        )
        output["left_energy"].append(left[keep])
        output["right_energy"].append(right[keep])
        output["raw_gap"].append(raw[keep])
        output["unfolded_gap"].append(unfolded[keep])
    return {
        name: np.concatenate(chunks) if chunks else np.array([], dtype=float)
        for name, chunks in output.items()
    }


def summarize_q_window(
    triplets: Mapping[str, np.ndarray],
    gaps: Mapping[str, np.ndarray],
    *,
    model: FollowupModel,
    q_low: float,
    q_high: float,
    bootstrap_replicates: int = 2000,
) -> dict[str, object]:
    ratio_mask = _window_mask(triplets["ensemble_q"], q_low, q_high)
    gap_mask = _window_mask(gaps["ensemble_q"], q_low, q_high)
    graph_ids = triplets["graph_index"][ratio_mask].astype(int)
    raw = triplets["raw_r"][ratio_mask]
    unfolded = triplets["unfolded_r"][ratio_mask]
    finite_pair = np.isfinite(raw) & np.isfinite(unfolded)
    graph_ids = graph_ids[finite_pair]
    raw = raw[finite_pair]
    unfolded = unfolded[finite_pair]
    raw_mean, raw_se, graph_count, _, _ = graph_equal_summary(graph_ids, raw)
    unfolded_mean, unfolded_se, unfolded_graphs, _, _ = graph_equal_summary(
        graph_ids, unfolded
    )
    raw_ci = graph_bootstrap_ci(
        graph_ids,
        raw,
        seed=stable_seed(model.key, q_low, q_high, "raw"),
        replicates=bootstrap_replicates,
    )
    unfolded_ci = graph_bootstrap_ci(
        graph_ids,
        unfolded,
        seed=stable_seed(model.key, q_low, q_high, "unfolded"),
        replicates=bootstrap_replicates,
    )
    raw_left = triplets["raw_left_gap"][ratio_mask][finite_pair]
    raw_right = triplets["raw_right_gap"][ratio_mask][finite_pair]
    unfolded_left = triplets["unfolded_left_gap"][ratio_mask][finite_pair]
    unfolded_right = triplets["unfolded_right_gap"][ratio_mask][finite_pair]
    raw_corr = correlation_from_pairs(raw_left, raw_right)
    unfolded_corr = correlation_from_pairs(unfolded_left, unfolded_right)
    raw_corr_ci = correlation_bootstrap_ci(
        graph_ids,
        raw_left,
        raw_right,
        seed=stable_seed(model.key, q_low, q_high, "raw_corr"),
    )
    unfolded_corr_ci = correlation_bootstrap_ci(
        graph_ids,
        unfolded_left,
        unfolded_right,
        seed=stable_seed(model.key, q_low, q_high, "unfolded_corr"),
    )
    selected_gap_graphs = gaps["graph_index"][gap_mask].astype(int)
    selected_raw_gap = gaps["raw_gap"][gap_mask]
    selected_unfolded_gap = gaps["unfolded_gap"][gap_mask]
    finite_gap = np.isfinite(selected_raw_gap) & np.isfinite(selected_unfolded_gap)
    positive_gap = finite_gap & (selected_raw_gap > ZERO_GAP_ATOL) & (
        selected_unfolded_gap > ZERO_GAP_ATOL
    )
    raw_gap = selected_raw_gap[positive_gap]
    unfolded_gap = selected_unfolded_gap[positive_gap]
    positive_gap_graphs = selected_gap_graphs[positive_gap]
    raw_normalized = raw_gap / np.mean(raw_gap) if len(raw_gap) else raw_gap
    unfolded_normalized = (
        unfolded_gap / np.mean(unfolded_gap) if len(unfolded_gap) else unfolded_gap
    )
    raw_q25, raw_median, raw_q75 = (
        np.quantile(raw, [0.25, 0.5, 0.75]) if len(raw) else (math.nan,) * 3
    )
    unfolded_q25, unfolded_median, unfolded_q75 = (
        np.quantile(unfolded, [0.25, 0.5, 0.75])
        if len(unfolded)
        else (math.nan,) * 3
    )
    raw_gap_ratio = raw_right / np.maximum(raw_left, 1e-300)
    unfolded_gap_ratio = unfolded_right / np.maximum(unfolded_left, 1e-300)
    edge_ranks = triplets["edge_rank"][ratio_mask][finite_pair]
    energies = triplets["middle_energy"][ratio_mask][finite_pair]
    gap_left_energy = gaps.get("left_energy", np.full(len(gaps["graph_index"]), np.nan))[
        gap_mask
    ][positive_gap]
    gap_right_energy = gaps.get(
        "right_energy", np.full(len(gaps["graph_index"]), np.nan)
    )[gap_mask][positive_gap]
    gap_graph_count = int(len(np.unique(positive_gap_graphs)))
    histogram_support = len(unfolded_gap) >= 100 and gap_graph_count >= 50
    ratio_support = len(raw) >= 100 and graph_count >= 50
    return {
        "analysis_version": FOLLOWUP_ANALYSIS_VERSION,
        "model_key": model.key,
        "model": model.label,
        "q_low": q_low,
        "q_high": q_high,
        "gaps": int(len(unfolded_gap)),
        "positive_gap_count": int(len(unfolded_gap)),
        "gap_contributing_graphs": gap_graph_count,
        "ratios": int(len(raw)),
        "valid_ratio_count": int(len(raw)),
        "ratio_contributing_graphs": graph_count,
        "histogram_support": histogram_support,
        "ratio_support": ratio_support,
        "support_status": (
            "SUPPORTED"
            if histogram_support and ratio_support
            else "HISTOGRAM_ONLY"
            if histogram_support
            else "RATIO_ONLY"
            if ratio_support
            else "UNDER_SUPPORTED"
        ),
        "zero_gap_fraction": (
            float(np.mean(selected_raw_gap[finite_gap] <= ZERO_GAP_ATOL))
            if np.any(finite_gap)
            else math.nan
        ),
        "raw_r": raw_mean,
        "raw_r_se": raw_se,
        "raw_r_bootstrap_ci_low": raw_ci[0],
        "raw_r_bootstrap_ci_high": raw_ci[1],
        "unfolded_r": unfolded_mean,
        "unfolded_r_se": unfolded_se,
        "unfolded_r_bootstrap_ci_low": unfolded_ci[0],
        "unfolded_r_bootstrap_ci_high": unfolded_ci[1],
        "unfolded_ratio_contributing_graphs": unfolded_graphs,
        "raw_r_median": float(raw_median),
        "raw_r_iqr": float(raw_q75 - raw_q25),
        "unfolded_r_median": float(unfolded_median),
        "unfolded_r_iqr": float(unfolded_q75 - unfolded_q25),
        "raw_r_KS_to_GOE_ratio": ks_distance(raw, goe_ratio_cdf),
        "raw_r_KS_to_Poisson_ratio": ks_distance(raw, poisson_ratio_cdf),
        "unfolded_r_KS_to_GOE_ratio": ks_distance(unfolded, goe_ratio_cdf),
        "unfolded_r_KS_to_Poisson_ratio": ks_distance(
            unfolded, poisson_ratio_cdf
        ),
        "raw_spacing_KS_to_GOE": ks_distance_unit_mean_spacing(
            raw_normalized, "goe"
        ),
        "raw_spacing_KS_to_Poisson": ks_distance_unit_mean_spacing(
            raw_normalized, "poisson"
        ),
        "unfolded_spacing_KS_to_GOE": ks_distance_unit_mean_spacing(
            unfolded_normalized, "goe"
        ),
        "unfolded_spacing_KS_to_Poisson": ks_distance_unit_mean_spacing(
            unfolded_normalized, "poisson"
        ),
        "raw_consecutive_spacing_correlation": raw_corr,
        "raw_consecutive_spacing_correlation_ci_low": raw_corr_ci[0],
        "raw_consecutive_spacing_correlation_ci_high": raw_corr_ci[1],
        "unfolded_consecutive_spacing_correlation": unfolded_corr,
        "unfolded_consecutive_spacing_correlation_ci_low": unfolded_corr_ci[0],
        "unfolded_consecutive_spacing_correlation_ci_high": unfolded_corr_ci[1],
        "raw_spacing_ratio_median": (
            float(np.median(raw_gap_ratio)) if len(raw_gap_ratio) else math.nan
        ),
        "unfolded_spacing_ratio_median": (
            float(np.median(unfolded_gap_ratio))
            if len(unfolded_gap_ratio)
            else math.nan
        ),
        "raw_alternation_fraction_ratio_outside_0p5_2": (
            float(np.mean((raw_gap_ratio < 0.5) | (raw_gap_ratio > 2.0)))
            if len(raw_gap_ratio)
            else math.nan
        ),
        "unfolded_alternation_fraction_ratio_outside_0p5_2": (
            float(
                np.mean(
                    (unfolded_gap_ratio < 0.5) | (unfolded_gap_ratio > 2.0)
                )
            )
            if len(unfolded_gap_ratio)
            else math.nan
        ),
        "edge_rank_min": int(np.min(edge_ranks)) if len(edge_ranks) else "",
        "edge_rank_q25": (
            float(np.quantile(edge_ranks, 0.25)) if len(edge_ranks) else math.nan
        ),
        "edge_rank_p10": (
            float(np.quantile(edge_ranks, 0.10)) if len(edge_ranks) else math.nan
        ),
        "edge_rank_median": (
            float(np.median(edge_ranks)) if len(edge_ranks) else math.nan
        ),
        "edge_rank_p90": (
            float(np.quantile(edge_ranks, 0.90)) if len(edge_ranks) else math.nan
        ),
        "edge_rank_q75": (
            float(np.quantile(edge_ranks, 0.75)) if len(edge_ranks) else math.nan
        ),
        "edge_rank_max": int(np.max(edge_ranks)) if len(edge_ranks) else "",
        "mean_energy": float(np.mean(energies)) if len(energies) else math.nan,
        "energy_min": float(np.min(energies)) if len(energies) else math.nan,
        "energy_max": float(np.max(energies)) if len(energies) else math.nan,
        "gap_energy_min": (
            float(np.min(gap_left_energy)) if len(gap_left_energy) else math.nan
        ),
        "gap_energy_max": (
            float(np.max(gap_right_energy)) if len(gap_right_energy) else math.nan
        ),
        "smallest_positive_gap": (
            float(np.min(raw_gap)) if len(raw_gap) else math.nan
        ),
        "relative_gap_below_1e_minus_8": int(
            np.sum(triplets["relative_min_gap"][ratio_mask][finite_pair] < 1e-8)
        ),
        "relative_gap_below_1e_minus_10": int(
            np.sum(triplets["relative_min_gap"][ratio_mask][finite_pair] < 1e-10)
        ),
        "relative_gap_below_1e_minus_12": int(
            np.sum(triplets["relative_min_gap"][ratio_mask][finite_pair] < 1e-12)
        ),
    }


def rank_crossfit_unfolded_gaps(
    gaps: np.ndarray,
    graph_indices: np.ndarray,
    *,
    num_folds: int = 5,
) -> np.ndarray:
    gaps = np.asarray(gaps, dtype=float)
    graph_indices = np.asarray(graph_indices, dtype=int)
    if gaps.ndim != 2 or gaps.shape[0] != len(graph_indices):
        raise ValueError("gap matrix and graph indices are inconsistent")
    output = np.full_like(gaps, np.nan)
    for fold in range(num_folds):
        held = graph_indices % num_folds == fold
        training = ~held
        if not np.any(held) or not np.any(training):
            raise ValueError(f"empty cross-fit partition for fold {fold}")
        scale = np.nanmean(gaps[training], axis=0)
        valid_scale = np.isfinite(scale) & (scale > ZERO_GAP_ATOL)
        output[np.ix_(held, valid_scale)] = gaps[np.ix_(held, valid_scale)] / scale[
            valid_scale
        ]
    return output


def _edge_gap_matrix(eigenvalues: np.ndarray, side: str) -> np.ndarray:
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    if side == "positive":
        return eigenvalues[:, :-1] - eigenvalues[:, 1:]
    if side == "negative":
        return eigenvalues[:, 1:] - eigenvalues[:, :-1]
    raise ValueError(f"unknown edge side {side!r}")


def summarize_fixed_rank_dataset(
    dataset: Mapping[str, np.ndarray],
    *,
    model: FollowupModel,
    n: int,
    rank_windows: Sequence[tuple[int, int]] = RANK_WINDOWS,
    bootstrap_replicates: int = 2000,
) -> list[dict[str, object]]:
    graph_indices = np.asarray(dataset["graph_indices"], dtype=int)
    rows: list[dict[str, object]] = []
    for side in ("positive", "negative"):
        eigenvalues = np.asarray(dataset[f"{side}_eigenvalue"], dtype=float)
        raw_gaps = _edge_gap_matrix(eigenvalues, side)
        unfolded_gaps = rank_crossfit_unfolded_gaps(raw_gaps, graph_indices)
        cdf_unfolded_gaps = np.asarray(
            dataset.get(
                f"{side}_cdf_unfolded_gap",
                np.full_like(raw_gaps, np.nan),
            ),
            dtype=float,
        )
        if cdf_unfolded_gaps.shape != raw_gaps.shape:
            raise ValueError(f"{side} CDF-unfolded gap shape mismatch")
        raw_ratios = _ratio(raw_gaps[:, :-1], raw_gaps[:, 1:])
        unfolded_ratios = _ratio(unfolded_gaps[:, :-1], unfolded_gaps[:, 1:])
        cdf_unfolded_ratios = _ratio(
            cdf_unfolded_gaps[:, :-1], cdf_unfolded_gaps[:, 1:]
        )
        middle_ranks = np.arange(2, eigenvalues.shape[1], dtype=int)
        gap_ranks = np.arange(1, eigenvalues.shape[1], dtype=int)
        nipr = np.asarray(
            dataset.get(f"{side}_n_times_ipr", np.full_like(eigenvalues, np.nan)),
            dtype=float,
        )
        c50 = np.asarray(
            dataset.get(f"{side}_c50", np.full_like(eigenvalues, np.nan)),
            dtype=float,
        )
        c10 = np.asarray(
            dataset.get(f"{side}_c10", np.full_like(eigenvalues, np.nan)),
            dtype=float,
        )
        c200 = np.asarray(
            dataset.get(f"{side}_c200", np.full_like(eigenvalues, np.nan)),
            dtype=float,
        )
        participation_fraction = np.asarray(
            dataset.get(
                f"{side}_participation_fraction",
                np.full_like(eigenvalues, np.nan),
            ),
            dtype=float,
        )
        ensemble_q = np.asarray(
            dataset.get(f"{side}_ensemble_q", np.full_like(eigenvalues, np.nan)),
            dtype=float,
        )
        degrees = np.asarray(
            dataset.get("degree_rank", np.full_like(eigenvalues, np.nan)),
            dtype=float,
        )
        dmax = np.asarray(
            dataset.get("maximum_degree", np.full(len(graph_indices), np.nan)),
            dtype=float,
        )
        for rank_low, rank_high in rank_windows:
            ratio_columns = (middle_ranks >= rank_low) & (middle_ranks <= rank_high)
            gap_columns = (gap_ranks >= rank_low) & (gap_ranks <= rank_high)
            eigen_columns = (
                (np.arange(1, eigenvalues.shape[1] + 1) >= rank_low)
                & (np.arange(1, eigenvalues.shape[1] + 1) <= rank_high)
            )
            raw_selected = raw_ratios[:, ratio_columns]
            unfolded_selected = unfolded_ratios[:, ratio_columns]
            ratio_graph_grid = np.broadcast_to(
                graph_indices[:, None], raw_selected.shape
            )
            valid_pair = np.isfinite(raw_selected) & np.isfinite(unfolded_selected)
            selected_graphs = ratio_graph_grid[valid_pair]
            selected_raw = raw_selected[valid_pair]
            selected_unfolded = unfolded_selected[valid_pair]
            raw_mean, raw_se, graph_count, _, _ = graph_equal_summary(
                selected_graphs, selected_raw
            )
            unfolded_mean, unfolded_se, unfolded_graphs, _, _ = graph_equal_summary(
                selected_graphs, selected_unfolded
            )
            raw_ci = graph_bootstrap_ci(
                selected_graphs,
                selected_raw,
                seed=stable_seed(model.key, n, side, rank_low, rank_high, "raw"),
                replicates=bootstrap_replicates,
            )
            unfolded_ci = graph_bootstrap_ci(
                selected_graphs,
                selected_unfolded,
                seed=stable_seed(
                    model.key, n, side, rank_low, rank_high, "rank_unfolded"
                ),
                replicates=bootstrap_replicates,
            )
            cdf_selected = cdf_unfolded_ratios[:, ratio_columns]
            cdf_graph_grid = np.broadcast_to(graph_indices[:, None], cdf_selected.shape)
            cdf_valid = np.isfinite(cdf_selected)
            cdf_graphs = cdf_graph_grid[cdf_valid]
            cdf_values = cdf_selected[cdf_valid]
            cdf_mean, cdf_se, cdf_graph_count, _, _ = graph_equal_summary(
                cdf_graphs, cdf_values
            )
            raw_spacing = raw_gaps[:, gap_columns].ravel()
            raw_spacing = raw_spacing[
                np.isfinite(raw_spacing) & (raw_spacing > ZERO_GAP_ATOL)
            ]
            unfolded_spacing = unfolded_gaps[:, gap_columns].ravel()
            unfolded_spacing = unfolded_spacing[
                np.isfinite(unfolded_spacing)
                & (unfolded_spacing > ZERO_GAP_ATOL)
            ]
            raw_normalized = (
                raw_spacing / np.mean(raw_spacing) if len(raw_spacing) else raw_spacing
            )
            unfolded_normalized = (
                unfolded_spacing / np.mean(unfolded_spacing)
                if len(unfolded_spacing)
                else unfolded_spacing
            )
            selected_eigenvalues = eigenvalues[:, eigen_columns]
            selected_nipr = nipr[:, eigen_columns]
            selected_c50 = c50[:, eigen_columns]
            selected_c10 = c10[:, eigen_columns]
            selected_c200 = c200[:, eigen_columns]
            selected_participation = participation_fraction[:, eigen_columns]
            selected_q = ensemble_q[:, eigen_columns]
            selected_degrees = degrees[:, eigen_columns]
            lambda_scale = np.abs(selected_eigenvalues) / np.sqrt(selected_degrees)
            mean_gap_by_rank = np.nanmean(raw_gaps[:, gap_columns], axis=0)
            selected_gap_ranks = gap_ranks[gap_columns]
            gradient = math.nan
            if len(mean_gap_by_rank) >= 2 and np.all(mean_gap_by_rank > 0.0):
                gradient = float(
                    -np.polyfit(selected_gap_ranks, np.log(mean_gap_by_rank), 1)[0]
                )
            residual = np.asarray(
                dataset.get(
                    f"{side}_relative_residual",
                    np.full_like(eigenvalues, np.nan),
                ),
                dtype=float,
            )[:, eigen_columns]
            rows.append(
                {
                    "analysis_version": FOLLOWUP_ANALYSIS_VERSION,
                    "model_key": model.key,
                    "model": model.label,
                    "N": n,
                    "side": side,
                    "rank_low": rank_low,
                    "rank_high": rank_high,
                    "ratio_rank_convention": "middle eigenvalue edge rank; rank 1 has no ratio",
                    "graph_count": graph_count,
                    "valid_ratios": int(len(selected_raw)),
                    "raw_r": raw_mean,
                    "raw_r_se": raw_se,
                    "raw_r_bootstrap_ci_low": raw_ci[0],
                    "raw_r_bootstrap_ci_high": raw_ci[1],
                    "unfolded_r": unfolded_mean,
                    "unfolded_r_se": unfolded_se,
                    "unfolded_r_bootstrap_ci_low": unfolded_ci[0],
                    "unfolded_r_bootstrap_ci_high": unfolded_ci[1],
                    "unfolded_graph_count": unfolded_graphs,
                    "unfolding_method": "five-fold training-only mean gap at each edge rank",
                    "cdf_unfolded_r": cdf_mean,
                    "cdf_unfolded_r_se": cdf_se,
                    "cdf_unfolded_graph_count": cdf_graph_count,
                    "cdf_unfolding_method": (
                        "five-fold training-only full-spectrum empirical CDF"
                        if cdf_graph_count
                        else "NOT_AVAILABLE_FOR_EDGE_ONLY_SPECTRA"
                    ),
                    "positive_gaps": int(len(raw_spacing)),
                    "raw_spacing_KS_to_GOE": ks_distance_unit_mean_spacing(
                        raw_normalized, "goe"
                    ),
                    "raw_spacing_KS_to_Poisson": ks_distance_unit_mean_spacing(
                        raw_normalized, "poisson"
                    ),
                    "unfolded_spacing_KS_to_GOE": ks_distance_unit_mean_spacing(
                        unfolded_normalized, "goe"
                    ),
                    "unfolded_spacing_KS_to_Poisson": ks_distance_unit_mean_spacing(
                        unfolded_normalized, "poisson"
                    ),
                    "NIPR": (
                        float(np.nanmean(selected_nipr))
                        if np.any(np.isfinite(selected_nipr))
                        else math.nan
                    ),
                    "NIPR_graphs": int(
                        np.sum(np.any(np.isfinite(selected_nipr), axis=1))
                    ),
                    "C50": (
                        float(np.nanmean(selected_c50))
                        if np.any(np.isfinite(selected_c50))
                        else math.nan
                    ),
                    "C10": (
                        float(np.nanmean(selected_c10))
                        if np.any(np.isfinite(selected_c10))
                        else math.nan
                    ),
                    "C200": (
                        float(np.nanmean(selected_c200))
                        if np.any(np.isfinite(selected_c200))
                        else math.nan
                    ),
                    "PR_over_N": (
                        float(np.nanmean(selected_participation))
                        if np.any(np.isfinite(selected_participation))
                        else math.nan
                    ),
                    "mean_q": (
                        float(np.nanmean(selected_q))
                        if np.any(np.isfinite(selected_q))
                        else math.nan
                    ),
                    "mean_energy": (
                        float(np.nanmean(selected_eigenvalues))
                        if np.any(np.isfinite(selected_eigenvalues))
                        else math.nan
                    ),
                    "mean_absolute_energy": (
                        float(np.nanmean(np.abs(selected_eigenvalues)))
                        if np.any(np.isfinite(selected_eigenvalues))
                        else math.nan
                    ),
                    "dmax": (
                        float(np.nanmean(dmax))
                        if np.any(np.isfinite(dmax))
                        else math.nan
                    ),
                    "degree_graphs": int(np.sum(np.isfinite(dmax))),
                    "lambda_over_sqrt_d_rank": (
                        float(np.nanmean(lambda_scale))
                        if np.any(np.isfinite(lambda_scale))
                        else math.nan
                    ),
                    "local_empirical_density_log_gradient": gradient,
                    "maximum_relative_eigenpair_residual": (
                        float(np.nanmax(residual))
                        if np.any(np.isfinite(residual))
                        else math.nan
                    ),
                }
            )
    return rows


def _generate_graph(config: SparseEdgeConfig, graph_index: int) -> nx.Graph:
    graph_seed = config.graph_seeds()[graph_index]
    if config.model.model_family == "affine_linear":
        graph, _ = generate_preferential_attachment_graph(
            n=config.n,
            m=config.m,
            seed=graph_seed,
            model=MODEL_AFFINE_PA,
            degree_ccdf_alpha=config.model.parameter_value,
        )
    else:
        graph, _ = generate_preferential_attachment_graph(
            n=config.n,
            m=config.m,
            seed=graph_seed,
            model=MODEL_NONLINEAR_PA,
            attachment_zeta=config.model.parameter_value,
        )
    return graph


def _relative_residuals(
    adjacency: csr_matrix,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
) -> np.ndarray:
    residual = adjacency @ eigenvectors - eigenvectors * eigenvalues[None, :]
    numerator = np.linalg.norm(residual, axis=0)
    denominator = np.maximum(np.abs(eigenvalues), 1.0)
    return numerator / denominator


def _localization_scalars(eigenvectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    masses = np.square(eigenvectors)
    nipr = eigenvectors.shape[0] * np.sum(np.square(masses), axis=0)
    top_count = min(50, eigenvectors.shape[0])
    c50 = np.sum(np.partition(masses, -top_count, axis=0)[-top_count:, :], axis=0)
    return nipr, c50


def solve_sparse_graph_edges(
    config: SparseEdgeConfig,
    graph_index: int,
    *,
    v0_salt: str = "primary",
) -> dict[str, np.ndarray | float | int]:
    graph = _generate_graph(config, graph_index)
    if not nx.is_connected(graph):
        raise ValueError(f"generated graph {graph_index} is disconnected")
    if hasattr(nx, "to_scipy_sparse_array"):
        sparse_converter = nx.to_scipy_sparse_array
    else:
        sparse_converter = nx.to_scipy_sparse_matrix
    adjacency = sparse_converter(
        graph,
        nodelist=range(config.n),
        dtype=float,
        format="csr",
    )
    count = config.rank_count + 1
    positive_k = count + 1
    if positive_k >= config.n:
        raise ValueError("not enough dimension for Perron plus edge ranks")
    generator = np.random.default_rng(
        stable_seed(config.model.key, config.n, graph_index, v0_salt)
    )
    positive_values, positive_vectors = eigsh(
        adjacency,
        k=positive_k,
        which="LA",
        tol=config.eigsh_tolerance,
        maxiter=config.eigsh_maxiter_factor * config.n,
        ncv=min(config.n, max(2 * positive_k + 1, 64)),
        v0=generator.normal(size=config.n),
    )
    negative_values, negative_vectors = eigsh(
        adjacency,
        k=count,
        which="SA",
        tol=config.eigsh_tolerance,
        maxiter=config.eigsh_maxiter_factor * config.n,
        ncv=min(config.n, max(2 * count + 1, 64)),
        v0=generator.normal(size=config.n),
    )
    positive_order = np.argsort(positive_values)[::-1]
    positive_values = positive_values[positive_order]
    positive_vectors = positive_vectors[:, positive_order]
    perron = float(positive_values[0])
    perron_vector = positive_vectors[:, 0]
    positive_values = positive_values[1 : count + 1]
    positive_vectors = positive_vectors[:, 1 : count + 1]
    negative_order = np.argsort(negative_values)
    negative_values = negative_values[negative_order]
    negative_vectors = negative_vectors[:, negative_order]
    positive_residual = _relative_residuals(
        adjacency, positive_values, positive_vectors
    )
    negative_residual = _relative_residuals(
        adjacency, negative_values, negative_vectors
    )
    perron_residual = float(
        _relative_residuals(
            adjacency,
            np.asarray([perron]),
            perron_vector[:, None],
        )[0]
    )
    positive_nipr, positive_c50 = _localization_scalars(positive_vectors)
    negative_nipr, negative_c50 = _localization_scalars(negative_vectors)
    degrees = np.asarray([graph.degree(node) for node in range(config.n)], dtype=int)
    degree_rank = np.sort(degrees)[::-1][:count]
    return {
        "graph_index": graph_index,
        "graph_seed": config.graph_seeds()[graph_index],
        "positive_eigenvalue": positive_values,
        "negative_eigenvalue": negative_values,
        "positive_n_times_ipr": positive_nipr,
        "negative_n_times_ipr": negative_nipr,
        "positive_c50": positive_c50,
        "negative_c50": negative_c50,
        "positive_relative_residual": positive_residual,
        "negative_relative_residual": negative_residual,
        "perron_eigenvalue": perron,
        "perron_relative_residual": perron_residual,
        "degree_rank": degree_rank,
        "maximum_degree": int(degree_rank[0]),
        "num_edges": graph.number_of_edges(),
        "connected": True,
    }


def sparse_chunk_path(config: SparseEdgeConfig, start: int, stop: int) -> Path:
    return config.partial_dir / f"edge_rank_cache_{start:04d}_{stop:04d}.npz"


def execution_metadata(start: int, stop: int) -> dict[str, object]:
    return {
        "sge_job_id": os.environ.get("JOB_ID", ""),
        "sge_task_id": os.environ.get("SGE_TASK_ID", ""),
        "graph_index_start": start,
        "graph_index_stop_exclusive": stop,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "networkx_version": nx.__version__,
    }


def validate_sparse_chunk(
    config: SparseEdgeConfig,
    start: int,
    stop: int,
) -> tuple[bool, str]:
    path = sparse_chunk_path(config, start, stop)
    if not path.is_file():
        return False, "missing cache"
    count = config.rank_count + 1
    try:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            execution = json.loads(str(data["execution_json"].item()))
            indices = np.asarray(data["graph_indices"], dtype=int)
            seeds = np.asarray(data["graph_seeds"], dtype=np.uint64)
            positive = np.asarray(data["positive_eigenvalue"], dtype=float)
            negative = np.asarray(data["negative_eigenvalue"], dtype=float)
            degrees = np.asarray(data["degree_rank"], dtype=int)
            required_matrices = (
                "positive_n_times_ipr",
                "negative_n_times_ipr",
                "positive_c50",
                "negative_c50",
                "positive_relative_residual",
                "negative_relative_residual",
            )
            for name in required_matrices:
                if np.asarray(data[name]).shape != (stop - start, count):
                    return False, f"shape mismatch for {name}"
            perron = np.asarray(data["perron_eigenvalue"], dtype=float)
            perron_residual = np.asarray(
                data["perron_relative_residual"], dtype=float
            )
            maximum_degree = np.asarray(data["maximum_degree"], dtype=int)
            connected = np.asarray(data["connected"], dtype=bool)
    except Exception as error:
        return False, f"unreadable cache: {error}"
    if metadata != config.metadata():
        return False, "metadata mismatch"
    if not np.array_equal(indices, np.arange(start, stop)):
        return False, "graph-index mismatch"
    if not np.array_equal(
        seeds, np.asarray(config.graph_seeds()[start:stop], dtype=np.uint64)
    ):
        return False, "graph-seed mismatch"
    if int(execution.get("graph_index_start", -1)) != start or int(
        execution.get("graph_index_stop_exclusive", -1)
    ) != stop:
        return False, "execution range mismatch"
    expected_shape = (stop - start, count)
    if positive.shape != expected_shape or negative.shape != expected_shape:
        return False, "eigenvalue shape mismatch"
    if degrees.shape != expected_shape:
        return False, "degree-rank shape mismatch"
    if np.any(np.diff(positive, axis=1) > 1e-10):
        return False, "positive edge is not outer-to-inner ordered"
    if np.any(np.diff(negative, axis=1) < -1e-10):
        return False, "negative edge is not outer-to-inner ordered"
    if not np.all(np.isfinite(positive)) or not np.all(np.isfinite(negative)):
        return False, "non-finite eigenvalue"
    if not np.all(perron[:, None] >= positive - 1e-9):
        return False, "Perron ordering failure"
    if not np.all(connected):
        return False, "disconnected graph"
    if np.any(maximum_degree != degrees[:, 0]):
        return False, "maximum-degree mismatch"
    if not np.all(np.isfinite(perron_residual)):
        return False, "non-finite Perron residual"
    return True, "VALID"


def run_sparse_chunk(
    config: SparseEdgeConfig,
    start: int,
    stop: int,
    *,
    skip_valid: bool = True,
) -> Path:
    if not 0 <= start < stop <= config.num_graphs:
        raise ValueError("invalid graph range")
    valid, reason = validate_sparse_chunk(config, start, stop)
    if skip_valid and valid:
        print(f"skip valid sparse chunk [{start},{stop}): {reason}", flush=True)
        return sparse_chunk_path(config, start, stop)
    config.partial_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for graph_index in range(start, stop):
        results.append(solve_sparse_graph_edges(config, graph_index))
        print(
            f"finished {config.model.key} N={config.n} graph={graph_index}",
            flush=True,
        )
    array_fields = (
        "positive_eigenvalue",
        "negative_eigenvalue",
        "positive_n_times_ipr",
        "negative_n_times_ipr",
        "positive_c50",
        "negative_c50",
        "positive_relative_residual",
        "negative_relative_residual",
        "degree_rank",
    )
    scalar_fields = (
        "perron_eigenvalue",
        "perron_relative_residual",
        "maximum_degree",
        "num_edges",
        "connected",
    )
    payload: dict[str, object] = {
        "graph_indices": np.asarray(
            [result["graph_index"] for result in results], dtype=np.int32
        ),
        "graph_seeds": np.asarray(
            [result["graph_seed"] for result in results], dtype=np.uint64
        ),
        "metadata_json": np.asarray(json.dumps(config.metadata(), sort_keys=True)),
        "execution_json": np.asarray(
            json.dumps(execution_metadata(start, stop), sort_keys=True)
        ),
    }
    for name in array_fields:
        payload[name] = np.asarray([result[name] for result in results])
    for name in scalar_fields:
        payload[name] = np.asarray([result[name] for result in results])
    path = sparse_chunk_path(config, start, stop)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)
    valid, reason = validate_sparse_chunk(config, start, stop)
    if not valid:
        raise RuntimeError(f"new sparse chunk failed validation: {reason}")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_sparse_chunks(
    config: SparseEdgeConfig,
    *,
    chunk_size: int,
) -> Path:
    paths: list[Path] = []
    source_rows: list[dict[str, object]] = []
    for start in range(0, config.num_graphs, chunk_size):
        stop = min(start + chunk_size, config.num_graphs)
        valid, reason = validate_sparse_chunk(config, start, stop)
        if not valid:
            raise ValueError(f"invalid sparse chunk [{start},{stop}): {reason}")
        path = sparse_chunk_path(config, start, stop)
        paths.append(path)
        with np.load(path, allow_pickle=False) as data:
            execution = json.loads(str(data["execution_json"].item()))
        source_rows.append(
            {
                "start": start,
                "stop": stop,
                "path": str(path),
                "sha256": sha256(path),
                "sge_job_id": execution.get("sge_job_id", ""),
                "sge_task_id": execution.get("sge_task_id", ""),
            }
        )
    concatenated: dict[str, list[np.ndarray]] = {}
    fields: list[str] | None = None
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if fields is None:
                fields = [
                    name
                    for name in data.files
                    if name not in {"metadata_json", "execution_json"}
                ]
                concatenated = {name: [] for name in fields}
            for name in fields:
                concatenated[name].append(np.asarray(data[name]))
    if fields is None:
        raise ValueError("no sparse chunks found")
    payload = {name: np.concatenate(chunks, axis=0) for name, chunks in concatenated.items()}
    payload["metadata_json"] = np.asarray(json.dumps(config.metadata(), sort_keys=True))
    destination = config.parameter_dir / "edge_rank_spectra.npz"
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, destination)
    manifest = {
        **config.metadata(),
        "aggregate_path": str(destination),
        "aggregate_sha256": sha256(destination),
        "source_chunks": source_rows,
    }
    (config.parameter_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return destination


def load_sparse_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {
            name: np.asarray(data[name])
            for name in data.files
            if name not in {"metadata_json", "execution_json"}
        }


def second_solver_validation(
    config: SparseEdgeConfig,
    graph_indices: Iterable[int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for graph_index in graph_indices:
        primary = solve_sparse_graph_edges(config, graph_index, v0_salt="primary")
        secondary = solve_sparse_graph_edges(config, graph_index, v0_salt="secondary")
        for side in ("positive", "negative"):
            first = np.asarray(primary[f"{side}_eigenvalue"], dtype=float)
            second = np.asarray(secondary[f"{side}_eigenvalue"], dtype=float)
            difference = np.abs(first - second)
            relative = difference / np.maximum(np.abs(first), 1.0)
            rows.append(
                {
                    "analysis_version": FOLLOWUP_ANALYSIS_VERSION,
                    "model_key": config.model.key,
                    "model": config.model.label,
                    "N": config.n,
                    "graph_index": graph_index,
                    "graph_seed": config.graph_seeds()[graph_index],
                    "side": side,
                    "rank_count": len(first),
                    "maximum_absolute_eigenvalue_difference": float(
                        np.max(difference)
                    ),
                    "maximum_relative_eigenvalue_difference": float(
                        np.max(relative)
                    ),
                    "primary_maximum_relative_residual": float(
                        np.max(primary[f"{side}_relative_residual"])
                    ),
                    "secondary_maximum_relative_residual": float(
                        np.max(secondary[f"{side}_relative_residual"])
                    ),
                    "validation_solver": "independent eigsh run with distinct deterministic start vector",
                }
            )
    return rows
