"""Gap-first mobility-edge post-processing for full PA adjacency spectra.

This module never generates a graph or diagonalizes an adjacency matrix.  It
turns already-sorted, per-graph non-Perron spectra into raw gaps, raw adjacent
gap ratios, graph-level cross-fitted unfolded spacings, adaptive windows, and
joined coarse summaries.  Every gap and ratio is formed inside one graph.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import gzip
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ..empirical_spectral_density import (
    CrossFittedEmpiricalSpectralDensity,
    estimate_cross_fitted_variable_local_count_densities,
)


ANALYSIS_VERSION = "pa_mobility_edge_gap_first_v2_atom_segmented"
GOE_MEAN_GAP_RATIO = 0.53590
POISSON_MEAN_GAP_RATIO = 2.0 * math.log(2.0) - 1.0
DEFAULT_ZERO_ATOL = 1e-14
DEFAULT_ATOM_ATOL = 1e-8
DEFAULT_EXCLUDED_ATOMS = (0.0,)
DEFAULT_AUDITED_ATOMS = (-1.0, 0.0, 1.0)
DEFAULT_MINIMUM_POSITIVE_GAPS = 100


@dataclass(frozen=True)
class GraphGapRecords:
    """All gap-first records for one continuous non-Perron spectrum."""

    graph_index: int
    graph_seed: int
    original_nonperron_dimension: int
    continuous_eigenvalues: np.ndarray
    excluded_atom_counts: dict[str, int]
    audited_atom_counts: dict[str, int]
    gap_left: np.ndarray
    gap_right: np.ndarray
    gap_value: np.ndarray
    gap_q: np.ndarray
    gap_energy: np.ndarray
    gap_is_zero: np.ndarray
    ratio_value: np.ndarray
    ratio_q: np.ndarray
    ratio_energy: np.ndarray
    ratio_category: np.ndarray
    omitted_atom_boundary_gap_count: int
    omitted_atom_boundary_ratio_count: int

    @property
    def continuous_dimension(self) -> int:
        return int(len(self.continuous_eigenvalues))


@dataclass(frozen=True)
class AdaptiveWindow:
    nominal_q_low: float
    nominal_q_high: float
    actual_q_low: float
    actual_q_high: float
    positive_gap_count: int
    total_gap_count: int
    graph_count: int
    status: str


def _atom_key(energy: float) -> str:
    return f"{float(energy):+.12g}"


def separate_exact_atoms(
    nonperron_eigenvalues: np.ndarray,
    *,
    excluded_atom_energies: Sequence[float] = DEFAULT_EXCLUDED_ATOMS,
    audited_atom_energies: Sequence[float] = DEFAULT_AUDITED_ATOMS,
    atom_atol: float = DEFAULT_ATOM_ATOL,
) -> tuple[np.ndarray, dict[str, int], dict[str, int]]:
    """Sort a spectrum, count requested atoms, and remove excluded atoms."""
    values = np.sort(np.asarray(nonperron_eigenvalues, dtype=float))
    if values.ndim != 1:
        raise ValueError("one-dimensional eigenvalues are required")
    if not np.all(np.isfinite(values)):
        raise ValueError("eigenvalues must be finite")
    audited = {
        _atom_key(energy): int(
            np.sum(np.isclose(values, energy, atol=atom_atol, rtol=0.0))
        )
        for energy in audited_atom_energies
    }
    keep = np.ones(len(values), dtype=bool)
    excluded: dict[str, int] = {}
    for energy in excluded_atom_energies:
        mask = np.isclose(values, energy, atol=atom_atol, rtol=0.0)
        excluded[_atom_key(energy)] = int(np.sum(mask))
        keep &= ~mask
    return values[keep], excluded, audited


def build_gap_records(
    nonperron_eigenvalues: np.ndarray,
    *,
    graph_index: int,
    graph_seed: int,
    excluded_atom_energies: Sequence[float] = DEFAULT_EXCLUDED_ATOMS,
    audited_atom_energies: Sequence[float] = DEFAULT_AUDITED_ATOMS,
    atom_atol: float = DEFAULT_ATOM_ATOL,
    zero_gap_atol: float = DEFAULT_ZERO_ATOL,
) -> GraphGapRecords:
    """Form every gap once and every adjacent-gap ratio inside one graph.

    Gap ``i`` is located at ``(i+1/2)/M`` and its energy midpoint.  Ratio
    ``i`` compares gaps ``i`` and ``i+1`` and is located at the intervening
    eigenvalue, with quantile ``(i+1)/M``.
    """
    raw = np.asarray(nonperron_eigenvalues, dtype=float)
    continuous, excluded, audited = separate_exact_atoms(
        raw,
        excluded_atom_energies=excluded_atom_energies,
        audited_atom_energies=audited_atom_energies,
        atom_atol=atom_atol,
    )
    dimension = len(continuous)
    if dimension < 2:
        gap_left = np.array([], dtype=float)
        gap_right = np.array([], dtype=float)
    else:
        gap_left = continuous[:-1]
        gap_right = continuous[1:]
    full_gaps = np.maximum(gap_right - gap_left, 0.0)
    atom_boundary = np.zeros(len(full_gaps), dtype=bool)
    for energy in excluded_atom_energies:
        if excluded.get(_atom_key(energy), 0) > 0:
            atom_boundary |= (gap_left < energy - atom_atol) & (
                gap_right > energy + atom_atol
            )
    keep_gap = ~atom_boundary
    full_gap_q = (np.arange(len(full_gaps), dtype=float) + 0.5) / max(dimension, 1)
    full_gap_energy = 0.5 * (gap_left + gap_right)
    gaps = full_gaps[keep_gap]
    gap_left_kept = gap_left[keep_gap]
    gap_right_kept = gap_right[keep_gap]
    gap_q = full_gap_q[keep_gap]
    gap_energy = full_gap_energy[keep_gap]
    gap_is_zero = gaps <= zero_gap_atol

    omitted_ratio_count = 0
    if len(full_gaps) >= 2:
        left = full_gaps[:-1]
        right = full_gaps[1:]
        maximum = np.maximum(left, right)
        both_positive = (left > zero_gap_atol) & (right > zero_gap_atol)
        one_zero = (maximum > zero_gap_atol) & ~both_positive
        both_zero = maximum <= zero_gap_atol
        ratios = np.full(len(left), np.nan, dtype=float)
        valid = ~both_zero
        ratios[valid] = np.minimum(left[valid], right[valid]) / maximum[valid]
        categories = np.full(len(left), "zero_zero", dtype="U18")
        categories[one_zero] = "one_zero"
        categories[both_positive] = "positive_positive"
        ratio_q = (np.arange(len(left), dtype=float) + 1.0) / dimension
        ratio_energy = continuous[1:-1]
        keep_ratio = keep_gap[:-1] & keep_gap[1:]
        omitted_ratio_count = int(np.sum(~keep_ratio))
        ratios = ratios[keep_ratio]
        ratio_q = ratio_q[keep_ratio]
        ratio_energy = ratio_energy[keep_ratio]
        categories = categories[keep_ratio]
    else:
        ratios = np.array([], dtype=float)
        ratio_q = np.array([], dtype=float)
        ratio_energy = np.array([], dtype=float)
        categories = np.array([], dtype="U18")
    return GraphGapRecords(
        graph_index=int(graph_index),
        graph_seed=int(graph_seed),
        original_nonperron_dimension=int(len(raw)),
        continuous_eigenvalues=continuous,
        excluded_atom_counts=excluded,
        audited_atom_counts=audited,
        gap_left=gap_left_kept,
        gap_right=gap_right_kept,
        gap_value=gaps,
        gap_q=gap_q,
        gap_energy=gap_energy,
        gap_is_zero=gap_is_zero,
        ratio_value=ratios,
        ratio_q=ratio_q,
        ratio_energy=ratio_energy,
        ratio_category=categories,
        omitted_atom_boundary_gap_count=int(np.sum(atom_boundary)),
        omitted_atom_boundary_ratio_count=omitted_ratio_count,
    )


def coarse_quantile_windows(width: float = 0.002) -> list[tuple[float, float]]:
    if not 0.0 < width <= 1.0:
        raise ValueError("width must lie in (0, 1]")
    count = int(round(1.0 / width))
    if not np.isclose(count * width, 1.0, atol=1e-12):
        raise ValueError("width must evenly divide [0, 1]")
    edges = np.linspace(0.0, 1.0, count + 1)
    return [(float(edges[index]), float(edges[index + 1])) for index in range(count)]


def _window_mask(values: np.ndarray, low: float, high: float) -> np.ndarray:
    if high >= 1.0 - 1e-15:
        return (values >= low) & (values <= high)
    return (values >= low) & (values < high)


def _window_slice(values: np.ndarray, low: float, high: float) -> slice:
    """Return the unique sorted-record slice assigned to ``[low, high)``."""
    left = int(np.searchsorted(values, low, side="left"))
    side = "right" if high >= 1.0 - 1e-15 else "left"
    right = int(np.searchsorted(values, high, side=side))
    return slice(left, right)


def _counts_in_window(
    records: Sequence[GraphGapRecords], q_low: float, q_high: float
) -> tuple[int, int, int]:
    positive = 0
    total = 0
    graphs = 0
    for record in records:
        selected = _window_slice(record.gap_q, q_low, q_high)
        count = selected.stop - selected.start
        if count:
            graphs += 1
            total += count
            positive += int(np.sum(~record.gap_is_zero[selected]))
    return positive, total, graphs


def adaptive_quantile_window(
    records: Sequence[GraphGapRecords],
    *,
    nominal_q_low: float,
    nominal_q_high: float,
    expansion_step: float,
    minimum_positive_gaps: int = DEFAULT_MINIMUM_POSITIVE_GAPS,
) -> AdaptiveWindow:
    """Symmetrically widen a window until it has enough positive gaps."""
    if not 0.0 <= nominal_q_low < nominal_q_high <= 1.0:
        raise ValueError("invalid nominal quantile bounds")
    if expansion_step <= 0.0:
        raise ValueError("expansion_step must be positive")
    q_low = float(nominal_q_low)
    q_high = float(nominal_q_high)
    while True:
        positive, total, graphs = _counts_in_window(records, q_low, q_high)
        if positive >= minimum_positive_gaps:
            status = "OK" if (q_low, q_high) == (nominal_q_low, nominal_q_high) else "ADAPTED"
            return AdaptiveWindow(
                nominal_q_low=float(nominal_q_low),
                nominal_q_high=float(nominal_q_high),
                actual_q_low=q_low,
                actual_q_high=q_high,
                positive_gap_count=positive,
                total_gap_count=total,
                graph_count=graphs,
                status=status,
            )
        if q_low <= 0.0 and q_high >= 1.0:
            return AdaptiveWindow(
                nominal_q_low=float(nominal_q_low),
                nominal_q_high=float(nominal_q_high),
                actual_q_low=q_low,
                actual_q_high=q_high,
                positive_gap_count=positive,
                total_gap_count=total,
                graph_count=graphs,
                status="INSUFFICIENT_POSITIVE_GAPS",
            )
        q_low = max(0.0, q_low - expansion_step)
        q_high = min(1.0, q_high + expansion_step)


def build_cross_fitted_models(
    records: Sequence[GraphGapRecords],
    *,
    num_folds: int = 5,
    bulk_half_width: float = 0.20,
    tail_half_width: float = 0.08,
    tail_quantile_cutoff: float = 0.95,
    num_points: int = 2001,
) -> list[CrossFittedEmpiricalSpectralDensity]:
    return estimate_cross_fitted_variable_local_count_densities(
        [record.continuous_eigenvalues for record in records],
        [record.graph_index for record in records],
        num_folds=num_folds,
        bulk_local_count_half_width=bulk_half_width,
        tail_local_count_half_width=tail_half_width,
        tail_quantile_cutoff=tail_quantile_cutoff,
        num_points=num_points,
    )


def _model_by_graph(
    models: Sequence[CrossFittedEmpiricalSpectralDensity],
) -> dict[int, CrossFittedEmpiricalSpectralDensity]:
    mapping: dict[int, CrossFittedEmpiricalSpectralDensity] = {}
    for model in models:
        training = set(model.training_graph_ids)
        held = set(model.held_out_graph_ids)
        if training & held:
            raise ValueError(f"fold {model.fold} leaks graphs between train and test")
        for graph_id in held:
            if graph_id in mapping:
                raise ValueError(f"graph {graph_id} is held out in multiple folds")
            mapping[graph_id] = model
    return mapping


def cross_fitted_unfolded_gaps(
    record: GraphGapRecords,
    model: CrossFittedEmpiricalSpectralDensity,
) -> np.ndarray:
    if record.graph_index not in model.held_out_graph_ids:
        raise ValueError("graph must be unfolded only by its held-out-fold model")
    if record.graph_index in model.training_graph_ids:
        raise ValueError("cross-fitting leakage detected")
    density = model.density_model
    left_cdf = np.interp(record.gap_left, density.grid, density.cdf)
    right_cdf = np.interp(record.gap_right, density.grid, density.cdf)
    unfolded = np.maximum(
        record.continuous_dimension * (right_cdf - left_cdf), 0.0
    )
    supported = (
        (record.gap_left >= density.grid[0])
        & (record.gap_right <= density.grid[-1])
    )
    unfolded[~supported] = np.nan
    return unfolded


def build_unfolded_gap_cache(
    records: Sequence[GraphGapRecords],
    models: Sequence[CrossFittedEmpiricalSpectralDensity],
) -> dict[int, np.ndarray]:
    mapping = _model_by_graph(models)
    cache: dict[int, np.ndarray] = {}
    for record in records:
        if record.graph_index not in mapping:
            raise ValueError(f"no held-out density model for graph {record.graph_index}")
        cache[record.graph_index] = cross_fitted_unfolded_gaps(
            record, mapping[record.graph_index]
        )
    return cache


def ks_distance_unit_mean_spacing(values: np.ndarray, distribution: str) -> float:
    values = np.sort(np.asarray(values, dtype=float))
    if not len(values):
        return math.nan
    empirical_high = np.arange(1, len(values) + 1, dtype=float) / len(values)
    empirical_low = np.arange(0, len(values), dtype=float) / len(values)
    if distribution == "goe":
        reference = 1.0 - np.exp(-math.pi * np.square(values) / 4.0)
    elif distribution == "poisson":
        reference = 1.0 - np.exp(-values)
    else:
        raise ValueError(f"unknown spacing distribution {distribution!r}")
    return float(
        max(
            np.max(np.abs(empirical_high - reference)),
            np.max(np.abs(empirical_low - reference)),
        )
    )


def _mean_se(values: Iterable[float]) -> tuple[float, float, int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return math.nan, math.nan, 0
    mean = float(np.mean(array))
    se = float(np.std(array, ddof=1) / math.sqrt(len(array))) if len(array) > 1 else math.nan
    return mean, se, int(len(array))


def summarize_quantile_window(
    records: Sequence[GraphGapRecords],
    *,
    q_low: float,
    q_high: float,
    crossfit_models: Sequence[CrossFittedEmpiricalSpectralDensity] | None = None,
    minimum_positive_gaps: int = DEFAULT_MINIMUM_POSITIVE_GAPS,
    nominal_q_low: float | None = None,
    nominal_q_high: float | None = None,
    window_status: str | None = None,
    unfolded_gap_cache: dict[int, np.ndarray] | None = None,
) -> tuple[dict[str, object], np.ndarray]:
    """Summarize raw ratios and held-out unfolded gaps in one actual window."""
    models = _model_by_graph(crossfit_models or [])
    graph_raw_means: list[float] = []
    graph_positive_means: list[float] = []
    unfolded: list[np.ndarray] = []
    gap_energies: list[np.ndarray] = []
    total_gaps = 0
    positive_gaps = 0
    zero_gaps = 0
    total_ratio_positions = 0
    positive_positive_ratios = 0
    one_zero_ratios = 0
    zero_zero_omitted = 0
    atom_boundary_gaps_omitted = 0
    atom_boundary_ratios_omitted = 0
    valid_raw_ratios = 0
    contributing_graphs = 0

    for record in records:
        atom_boundary_gaps_omitted += record.omitted_atom_boundary_gap_count
        atom_boundary_ratios_omitted += record.omitted_atom_boundary_ratio_count
        gap_slice = _window_slice(record.gap_q, q_low, q_high)
        ratio_slice = _window_slice(record.ratio_q, q_low, q_high)
        has_gaps = gap_slice.stop > gap_slice.start
        has_ratios = ratio_slice.stop > ratio_slice.start
        if has_gaps or has_ratios:
            contributing_graphs += 1
        selected_gaps = record.gap_value[gap_slice]
        selected_zero = record.gap_is_zero[gap_slice]
        total_gaps += len(selected_gaps)
        zeros = int(np.sum(selected_zero))
        zero_gaps += zeros
        positive_gaps += len(selected_gaps) - zeros
        if has_gaps:
            gap_energies.append(record.gap_energy[gap_slice])
        if models and has_gaps:
            if unfolded_gap_cache is None:
                if record.graph_index not in models:
                    raise ValueError(f"no held-out density model for graph {record.graph_index}")
                graph_unfolded = cross_fitted_unfolded_gaps(
                    record, models[record.graph_index]
                )[gap_slice]
            else:
                graph_unfolded = unfolded_gap_cache[record.graph_index][gap_slice]
            unfolded.append(graph_unfolded[~selected_zero])

        categories = record.ratio_category[ratio_slice]
        ratios = record.ratio_value[ratio_slice]
        total_ratio_positions += len(categories)
        positive_positive_ratios += int(np.sum(categories == "positive_positive"))
        one_zero_ratios += int(np.sum(categories == "one_zero"))
        zero_zero_omitted += int(np.sum(categories == "zero_zero"))
        valid = np.isfinite(ratios)
        valid_raw_ratios += int(np.sum(valid))
        if np.any(valid):
            graph_raw_means.append(float(np.mean(ratios[valid])))
        positive_mask = categories == "positive_positive"
        if np.any(positive_mask):
            graph_positive_means.append(float(np.mean(ratios[positive_mask])))

    mean_r, se_r, graphs_with_r = _mean_se(graph_raw_means)
    mean_positive_r, se_positive_r, graphs_with_positive_r = _mean_se(graph_positive_means)
    positive_unfolded = np.concatenate(unfolded) if unfolded else np.array([], dtype=float)
    mean_unfolded = float(np.mean(positive_unfolded)) if len(positive_unfolded) else math.nan
    normalized = (
        positive_unfolded[np.isfinite(positive_unfolded)]
        / float(np.mean(positive_unfolded[np.isfinite(positive_unfolded)]))
        if np.any(np.isfinite(positive_unfolded))
        and float(np.mean(positive_unfolded[np.isfinite(positive_unfolded)])) > 0.0
        else np.array([], dtype=float)
    )
    finite_positive_unfolded = positive_unfolded[np.isfinite(positive_unfolded)]
    mean_unfolded = (
        float(np.mean(finite_positive_unfolded))
        if len(finite_positive_unfolded)
        else math.nan
    )
    energies = np.concatenate(gap_energies) if gap_energies else np.array([], dtype=float)
    sample_status = (
        "OK"
        if len(normalized) >= minimum_positive_gaps
        else "INSUFFICIENT_POSITIVE_GAPS"
    )
    if window_status == "INSUFFICIENT_POSITIVE_GAPS":
        sample_status = window_status
    row: dict[str, object] = {
        "analysis_version": ANALYSIS_VERSION,
        "gap_assignment_convention": "gap midpoint: q=(i+1/2)/M, E=(lambda_i+lambda_i+1)/2",
        "ratio_assignment_convention": "middle eigenvalue: q=(i+1)/M, E=lambda_i+1",
        "unfolding_convention": "five-fold graph-level cross-fitted empirical CDF",
        "spacing_histogram_normalization": "positive cross-fitted spacings divided by actual-window pooled mean",
        "nominal_q_low": q_low if nominal_q_low is None else nominal_q_low,
        "nominal_q_high": q_high if nominal_q_high is None else nominal_q_high,
        "actual_q_low": q_low,
        "actual_q_high": q_high,
        "q_center": 0.5 * (q_low + q_high),
        "E_mean": float(np.mean(energies)) if len(energies) else math.nan,
        "E_min": float(np.min(energies)) if len(energies) else math.nan,
        "E_max": float(np.max(energies)) if len(energies) else math.nan,
        "number_of_contributing_graphs": contributing_graphs,
        "number_of_total_raw_gaps": total_gaps,
        "number_of_positive_gaps": positive_gaps,
        "number_of_zero_gaps": zero_gaps,
        "zero_gap_fraction": zero_gaps / max(total_gaps, 1),
        "number_of_total_raw_ratio_positions": total_ratio_positions,
        "number_of_valid_raw_ratios": valid_raw_ratios,
        "number_of_positive_positive_ratios": positive_positive_ratios,
        "number_of_one_zero_ratios": one_zero_ratios,
        "number_of_zero_zero_ratios_omitted": zero_zero_omitted,
        "number_of_atom_boundary_gaps_omitted_ensemble_total": atom_boundary_gaps_omitted,
        "number_of_atom_boundary_ratios_omitted_ensemble_total": atom_boundary_ratios_omitted,
        "graph_equal_mean_raw_r_inclusive": mean_r,
        "raw_r_inclusive_standard_error": se_r,
        "graphs_with_raw_r_inclusive": graphs_with_r,
        "absolute_distance_raw_r_inclusive_to_GOE": abs(mean_r - GOE_MEAN_GAP_RATIO),
        "absolute_distance_raw_r_inclusive_to_Poisson": abs(mean_r - POISSON_MEAN_GAP_RATIO),
        "graph_equal_mean_raw_r_positive_conditioned": mean_positive_r,
        "raw_r_positive_conditioned_standard_error": se_positive_r,
        "graphs_with_raw_r_positive_conditioned": graphs_with_positive_r,
        "absolute_distance_raw_r_positive_conditioned_to_GOE": abs(
            mean_positive_r - GOE_MEAN_GAP_RATIO
        ),
        "absolute_distance_raw_r_positive_conditioned_to_Poisson": abs(
            mean_positive_r - POISSON_MEAN_GAP_RATIO
        ),
        "crossfit_unfolded_mean_positive_spacing": mean_unfolded,
        "crossfit_unfolded_histogram_sample_count": int(len(normalized)),
        "crossfit_out_of_training_support_gap_count": int(
            len(positive_unfolded) - len(finite_positive_unfolded)
        ),
        "crossfit_unfolded_KS_to_GOE": ks_distance_unit_mean_spacing(normalized, "goe"),
        "crossfit_unfolded_KS_to_Poisson": ks_distance_unit_mean_spacing(
            normalized, "poisson"
        ),
        "histogram_sample_status": sample_status,
        "window_status": window_status or "FIXED",
    }
    return row, normalized


def summarize_quantile_grid(
    records: Sequence[GraphGapRecords],
    *,
    width: float = 0.002,
    crossfit_models: Sequence[CrossFittedEmpiricalSpectralDensity] | None = None,
    minimum_positive_gaps: int = DEFAULT_MINIMUM_POSITIVE_GAPS,
) -> tuple[list[dict[str, object]], dict[int, np.ndarray]]:
    rows: list[dict[str, object]] = []
    histograms: dict[int, np.ndarray] = {}
    unfolded_gap_cache = (
        build_unfolded_gap_cache(records, crossfit_models)
        if crossfit_models
        else None
    )
    for index, (q_low, q_high) in enumerate(coarse_quantile_windows(width)):
        row, histogram = summarize_quantile_window(
            records,
            q_low=q_low,
            q_high=q_high,
            crossfit_models=crossfit_models,
            minimum_positive_gaps=minimum_positive_gaps,
            unfolded_gap_cache=unfolded_gap_cache,
        )
        rows.append(row)
        if row["histogram_sample_status"] == "OK":
            histograms[index] = histogram
    return rows, histograms


def graph_level_quantile_rows(
    records: Sequence[GraphGapRecords],
    *,
    width: float = 0.002,
    crossfit_models: Sequence[CrossFittedEmpiricalSpectralDensity] | None = None,
) -> list[dict[str, object]]:
    """Return graph-by-window primary statistics for paired inference.

    The ensemble tables intentionally use graph-equal aggregation.  These
    rows retain the underlying graph means so original/rewired controls can
    be compared by graph index instead of as two unpaired ensembles.
    """
    unfolded_cache = (
        build_unfolded_gap_cache(records, crossfit_models)
        if crossfit_models
        else {}
    )
    rows: list[dict[str, object]] = []
    windows = coarse_quantile_windows(width)
    for record in records:
        graph_unfolded = unfolded_cache.get(record.graph_index)
        for q_low, q_high in windows:
            gap_slice = _window_slice(record.gap_q, q_low, q_high)
            ratio_slice = _window_slice(record.ratio_q, q_low, q_high)
            gaps = record.gap_value[gap_slice]
            zero = record.gap_is_zero[gap_slice]
            categories = record.ratio_category[ratio_slice]
            ratios = record.ratio_value[ratio_slice]
            valid = np.isfinite(ratios)
            positive_ratio = categories == "positive_positive"
            unfolded = (
                np.asarray(graph_unfolded[gap_slice], dtype=float)
                if graph_unfolded is not None
                else np.array([], dtype=float)
            )
            unfolded = unfolded[(~zero) & np.isfinite(unfolded)] if len(unfolded) else unfolded
            energies = record.gap_energy[gap_slice]
            rows.append(
                {
                    "analysis_version": ANALYSIS_VERSION,
                    "graph_index": record.graph_index,
                    "graph_seed": record.graph_seed,
                    "q_low": q_low,
                    "q_high": q_high,
                    "q_center": 0.5 * (q_low + q_high),
                    "E_mean": float(np.mean(energies)) if len(energies) else math.nan,
                    "total_gap_count": len(gaps),
                    "positive_gap_count": int(np.sum(~zero)),
                    "zero_gap_count": int(np.sum(zero)),
                    "zero_gap_fraction": float(np.mean(zero)) if len(zero) else math.nan,
                    "valid_raw_ratio_count": int(np.sum(valid)),
                    "positive_positive_ratio_count": int(np.sum(positive_ratio)),
                    "mean_raw_r_inclusive": float(np.mean(ratios[valid])) if np.any(valid) else math.nan,
                    "mean_raw_r_positive_conditioned": (
                        float(np.mean(ratios[positive_ratio])) if np.any(positive_ratio) else math.nan
                    ),
                    "mean_crossfit_unfolded_positive_spacing": (
                        float(np.mean(unfolded)) if len(unfolded) else math.nan
                    ),
                    "crossfit_unfolded_positive_spacing_count": len(unfolded),
                }
            )
    return rows


def summarize_adaptive_window(
    records: Sequence[GraphGapRecords],
    *,
    nominal_q_low: float,
    nominal_q_high: float,
    expansion_step: float,
    crossfit_models: Sequence[CrossFittedEmpiricalSpectralDensity] | None = None,
    minimum_positive_gaps: int = DEFAULT_MINIMUM_POSITIVE_GAPS,
    unfolded_gap_cache: dict[int, np.ndarray] | None = None,
) -> tuple[dict[str, object], np.ndarray]:
    window = adaptive_quantile_window(
        records,
        nominal_q_low=nominal_q_low,
        nominal_q_high=nominal_q_high,
        expansion_step=expansion_step,
        minimum_positive_gaps=minimum_positive_gaps,
    )
    q_low = window.actual_q_low
    q_high = window.actual_q_high
    status = window.status
    while True:
        row, histogram = summarize_quantile_window(
            records,
            q_low=q_low,
            q_high=q_high,
            crossfit_models=crossfit_models,
            minimum_positive_gaps=minimum_positive_gaps,
            nominal_q_low=window.nominal_q_low,
            nominal_q_high=window.nominal_q_high,
            window_status=status,
            unfolded_gap_cache=unfolded_gap_cache,
        )
        if len(histogram) >= minimum_positive_gaps or not crossfit_models:
            return row, histogram
        if q_low <= 0.0 and q_high >= 1.0:
            row["histogram_sample_status"] = "INSUFFICIENT_POSITIVE_GAPS"
            return row, histogram
        q_low = max(0.0, q_low - expansion_step)
        q_high = min(1.0, q_high + expansion_step)
        status = "ADAPTED_CROSSFIT_SUPPORT"


def join_localization_and_level_statistics(
    localization_rows: Sequence[dict[str, object]],
    level_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Join aligned coarse tables without changing either source table."""
    localization = {
        (
            round(float(row.get("quantile_low", row.get("q_low", math.nan))), 12),
            round(float(row.get("quantile_high", row.get("q_high", math.nan))), 12),
        ): dict(row)
        for row in localization_rows
    }
    joined: list[dict[str, object]] = []
    for level in level_rows:
        key = (
            round(float(level["actual_q_low"]), 12),
            round(float(level["actual_q_high"]), 12),
        )
        row = dict(level)
        source = localization.get(key)
        row["localization_row_available"] = source is not None
        if source is not None:
            for name, value in source.items():
                if name not in {"quantile", "quantile_low", "quantile_high"}:
                    row[f"eigenvector_{name}"] = value
        joined.append(row)
    return joined


def detect_candidate_regions(
    joined_rows: Sequence[dict[str, object]],
    *,
    minimum_consecutive_bins: int = 3,
    bulk_q_low: float = 0.10,
    bulk_q_high: float = 0.90,
    ipr_ratio_threshold: float = 2.0,
    top50_ratio_threshold: float = 1.35,
    r_shift_threshold: float = 0.04,
    atom_zero_gap_fraction: float = 0.10,
) -> list[dict[str, object]]:
    """Screen sustained non-atom signals; do not label a mobility edge."""
    if minimum_consecutive_bins < 1:
        raise ValueError("minimum_consecutive_bins must be positive")

    def metric(row: dict[str, object], names: Sequence[str]) -> float:
        for name in names:
            if name in row:
                try:
                    return float(row[name])
                except (TypeError, ValueError):
                    pass
        return math.nan

    bulk = [
        row
        for row in joined_rows
        if bulk_q_low <= float(row["q_center"]) <= bulk_q_high
        and float(row.get("zero_gap_fraction", 0.0)) < atom_zero_gap_fraction
    ]
    bulk_ipr = np.nanmedian(
        [metric(row, ("eigenvector_mean_n_times_ipr", "mean_n_times_ipr")) for row in bulk]
    )
    bulk_top50 = np.nanmedian(
        [metric(row, ("eigenvector_mean_top_50_mass", "mean_top_50_mass")) for row in bulk]
    )
    bulk_r = np.nanmedian(
        [float(row.get("graph_equal_mean_raw_r_positive_conditioned", math.nan)) for row in bulk]
    )
    flags: list[dict[str, object]] = []
    for row in joined_rows:
        ipr = metric(row, ("eigenvector_mean_n_times_ipr", "mean_n_times_ipr"))
        top50 = metric(row, ("eigenvector_mean_top_50_mass", "mean_top_50_mass"))
        r_value = float(row.get("graph_equal_mean_raw_r_positive_conditioned", math.nan))
        contaminated = float(row.get("zero_gap_fraction", 0.0)) >= atom_zero_gap_fraction
        localization = (
            np.isfinite(ipr)
            and np.isfinite(top50)
            and ipr >= ipr_ratio_threshold * bulk_ipr
            and top50 >= top50_ratio_threshold * bulk_top50
        )
        r_shift = np.isfinite(r_value) and np.isfinite(bulk_r) and bulk_r - r_value >= r_shift_threshold
        flags.append(
            {
                "row": row,
                "contaminated": contaminated,
                "localization": bool(localization),
                "r_shift": bool(r_shift),
                "ipr_over_bulk": ipr / bulk_ipr if np.isfinite(ipr) and bulk_ipr > 0 else math.nan,
                "top50_over_bulk": top50 / bulk_top50 if np.isfinite(top50) and bulk_top50 > 0 else math.nan,
            }
        )
    regions: list[dict[str, object]] = []
    start = 0
    while start < len(flags):
        if not flags[start]["localization"]:
            start += 1
            continue
        stop = start + 1
        while stop < len(flags) and flags[stop]["localization"]:
            stop += 1
        group = flags[start:stop]
        if len(group) >= minimum_consecutive_bins:
            contaminated = any(bool(item["contaminated"]) for item in group)
            shifted = any(bool(item["r_shift"]) for item in group)
            category = (
                "ATOM_CONTAMINATED"
                if contaminated
                else "IPR_AND_R_SHIFT"
                if shifted
                else "IPR_ONLY"
            )
            regions.append(
                {
                    "analysis_version": ANALYSIS_VERSION,
                    "category": category,
                    "q_low": float(group[0]["row"]["actual_q_low"]),
                    "q_high": float(group[-1]["row"]["actual_q_high"]),
                    "number_of_consecutive_bins": len(group),
                    "maximum_ipr_over_bulk": float(
                        np.nanmax([item["ipr_over_bulk"] for item in group])
                    ),
                    "maximum_top50_over_bulk": float(
                        np.nanmax([item["top50_over_bulk"] for item in group])
                    ),
                    "bulk_median_n_times_ipr": float(bulk_ipr),
                    "bulk_median_top50_mass": float(bulk_top50),
                    "bulk_median_raw_r_positive_conditioned": float(bulk_r),
                    "criteria": {
                        "minimum_consecutive_bins": minimum_consecutive_bins,
                        "ipr_ratio_threshold": ipr_ratio_threshold,
                        "top50_ratio_threshold": top50_ratio_threshold,
                        "r_shift_threshold": r_shift_threshold,
                        "atom_zero_gap_fraction": atom_zero_gap_fraction,
                    },
                }
            )
        start = stop
    if not regions:
        regions.append(
            {
                "analysis_version": ANALYSIS_VERSION,
                "category": "NO_SUSTAINED_SIGNAL",
                "q_low": math.nan,
                "q_high": math.nan,
                "number_of_consecutive_bins": 0,
                "bulk_median_n_times_ipr": float(bulk_ipr),
                "bulk_median_top50_mass": float(bulk_top50),
                "bulk_median_raw_r_positive_conditioned": float(bulk_r),
            }
        )
    return regions


def write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty table")
    fieldnames = sorted({name for row in rows for name in row})
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serializable = {
                name: json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list, tuple))
                else value
                for name, value in row.items()
            }
            writer.writerow(serializable)


def save_crossfit_models(
    output_dir: Path,
    models: Sequence[CrossFittedEmpiricalSpectralDensity],
    *,
    metadata: dict[str, object] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    for model in models:
        density = model.density_model
        np.savez_compressed(
            output_dir / f"fold_{model.fold}.npz",
            grid=density.grid,
            density=density.density,
            cdf=density.cdf,
            training_graph_ids=np.asarray(model.training_graph_ids, dtype=int),
            held_out_graph_ids=np.asarray(model.held_out_graph_ids, dtype=int),
        )
        manifest_rows.append(
            {
                "fold": model.fold,
                "training_graph_ids": list(model.training_graph_ids),
                "held_out_graph_ids": list(model.held_out_graph_ids),
                "num_training_graphs": len(model.training_graph_ids),
                "num_held_out_graphs": len(model.held_out_graph_ids),
                "density_method": density.method,
                "grid_points": len(density.grid),
                "cdf_start": float(density.cdf[0]),
                "cdf_end": float(density.cdf[-1]),
                "cdf_nondecreasing": bool(np.all(np.diff(density.cdf) >= -1e-12)),
                "bulk_local_count_half_width": density.local_count_half_width,
            }
        )
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "fold_assignment": "graph_index % 5",
        "folds": manifest_rows,
        "metadata": metadata or {},
    }
    (output_dir / "crossfit_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
