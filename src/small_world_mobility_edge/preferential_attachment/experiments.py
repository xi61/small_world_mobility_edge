"""Joint spectral-tail, spacing, and localization experiments for PA graphs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "outputs" / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / "outputs" / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.stats import kstest

from ..chung_lu_degree_spectral_tail_match import (
    DEFAULT_EIGSH_MAXITER,
    DEFAULT_EIGSH_NCV_FACTOR,
    DEFAULT_EIGSH_TOL,
    DEFAULT_K_PER_SIDE,
    DEFAULT_TAIL_LOWER_QUANTILES,
    PRIMARY_TAIL_QUANTILE,
    SparseTailSpectrum,
    finite_mean_se,
    hill_ccdf_exponent_from_top,
    log_ccdf_regression,
    rank_match_fit,
    sparse_tail_spectrum,
    top_count_for_quantile,
    transformed_prefactor,
)
from ..chung_lu_eigenvector_diagnostics import (
    COORDINATE_DECAY_MAX_RANK,
    coordinate_decay_profile_rows,
    eigenvector_localization_summary,
)
from ..chung_lu_spacing_by_energy import (
    REMOVE_EXACT_ZERO_EIGENVALUES,
    SMALL_SPACING_THRESHOLD,
    ZERO_EIGENVALUE_ATOL,
    make_seeds,
)
from ..fixed_degree_configuration_baseline import rewire_graph_preserving_degrees
from ..regular_baseline import adjacency_eigendecomposition, adjacent_gap_ratios_from_spacings
from .baseline import (
    DEFAULT_M,
    DEFAULT_N,
    HOLME_KIM_INITIAL_GRAPH_CONVENTION,
    INITIAL_GRAPH_CONVENTION,
    MODEL_AFFINE_PA,
    MODEL_AFFINE_PA_REWIRED,
    MODEL_BA,
    MODEL_BA_REWIRED,
    MODEL_HOLME_KIM,
    MODEL_HOLME_KIM_REWIRED,
    TARGET_MEAN_DEGREE,
    THEORETICAL_DEGREE_CCDF_TAU,
    attachment_offset_for_ccdf_tau,
    canonical_model_name,
    generate_preferential_attachment_graph,
)


DEFAULT_NUM_GRAPHS = 50
DEFAULT_TARGET_TAIL_OBSERVATIONS = 120
DEFAULT_GRAPH_SEED_BASE = 424242
DEFAULT_REWIRE_SEED_BASE = 525252
DEFAULT_COMPONENT_MODE = "full"
DEFAULT_SWAPS_PER_EDGE = 10
DEFAULT_MAX_TRIES_FACTOR = 20
DEFAULT_RANK_WINDOW_WIDTH = 20
DEFAULT_NUM_RANK_WINDOWS = 6
SPACING_NORMALIZATION = "within_graph_within_fixed_rank_window_mean"
SPECTRAL_SELECTION_METHOD = "sparse_algebraic_edges_perron_excluded"
SPACING_WINDOW_MODE = "disjoint_positive_nonperron_edge_gap_ranks"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "preferential_attachment"

LOCALIZATION_METRICS = (
    "mean_ipr",
    "median_ipr",
    "mean_n_times_ipr",
    "median_n_times_ipr",
    "mean_participation_ratio",
    "mean_participation_fraction",
    "median_max_coordinate_squared",
    "q90_max_coordinate_squared",
    "mean_top_5_coordinate_mass",
    "mean_top_10_coordinate_mass",
)

PAIRED_METRICS = (
    "graph_mean_gap_ratio",
    "small_gap_probability",
    "mean_n_times_ipr",
    "mean_participation_fraction",
    "median_max_coordinate_squared",
    "mean_top_5_coordinate_mass",
    "mean_top_10_coordinate_mass",
)


@dataclass(frozen=True)
class PreferentialAttachmentConfig:
    """All controls needed to reproduce one PA experiment ensemble."""

    n: int = DEFAULT_N
    m: int = DEFAULT_M
    num_graphs: int = DEFAULT_NUM_GRAPHS
    model: str = MODEL_BA
    triad_probability: float = 0.0
    target_mean_degree: float = TARGET_MEAN_DEGREE
    theoretical_degree_ccdf_tau: float = THEORETICAL_DEGREE_CCDF_TAU
    component_mode: str = DEFAULT_COMPONENT_MODE
    graph_seed_base: int = DEFAULT_GRAPH_SEED_BASE
    rewire_seed_base: int = DEFAULT_REWIRE_SEED_BASE
    run_rewired_control: bool = True
    run_chung_lu_comparison: bool = False
    swaps_per_edge: int = DEFAULT_SWAPS_PER_EDGE
    max_tries_factor: int = DEFAULT_MAX_TRIES_FACTOR
    k_per_side: int = DEFAULT_K_PER_SIDE
    target_tail_observations: int = DEFAULT_TARGET_TAIL_OBSERVATIONS
    tail_lower_quantiles: tuple[float, ...] = DEFAULT_TAIL_LOWER_QUANTILES
    rank_window_width: int = DEFAULT_RANK_WINDOW_WIDTH
    num_rank_windows: int = DEFAULT_NUM_RANK_WINDOWS
    coordinate_decay_max_rank: int = COORDINATE_DECAY_MAX_RANK
    eigsh_tol: float = DEFAULT_EIGSH_TOL
    eigsh_maxiter: int = DEFAULT_EIGSH_MAXITER
    eigsh_ncv_factor: int = DEFAULT_EIGSH_NCV_FACTOR
    validate_dense: bool = False
    dense_full_spectrum: bool = False
    output_root: Path = OUTPUT_ROOT

    @property
    def rank_windows(self) -> tuple[tuple[int, int], ...]:
        return fixed_rank_windows(self.rank_window_width, self.num_rank_windows)

    @property
    def rewired_model(self) -> str:
        if self.model == MODEL_BA:
            return MODEL_BA_REWIRED
        if self.model == MODEL_AFFINE_PA:
            return MODEL_AFFINE_PA_REWIRED
        return MODEL_HOLME_KIM_REWIRED


def code_version() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0**2 if sys.platform == "darwin" else 1024.0
    return value / divisor


def slug_number(value: float | int) -> str:
    return str(value).replace(".", "p")


def model_slug(model: str) -> str:
    if model == MODEL_BA:
        return "barabasi_albert"
    if model == MODEL_AFFINE_PA:
        return "affine_linear"
    return "holme_kim"


def model_output_dir(config: PreferentialAttachmentConfig) -> Path:
    suffix = f"n_{config.n}_m_{config.m}"
    if config.model == MODEL_HOLME_KIM:
        suffix += f"_p_{slug_number(config.triad_probability)}"
    if config.model == MODEL_AFFINE_PA:
        suffix += f"_ccdf_tau_{slug_number(config.theoretical_degree_ccdf_tau)}"
    return Path(config.output_root) / model_slug(config.model) / suffix


def parameter_seed_offset(model: str, m: int, triad_probability: float) -> int:
    """Separate PA parameter settings while preserving graph-index ordering."""
    if model == MODEL_AFFINE_PA:
        return 0
    model_offset = 0 if model == MODEL_BA else 100_000
    return model_offset + 1000 * int(m) + int(round(100.0 * triad_probability))


def seed_lists(config: PreferentialAttachmentConfig) -> tuple[list[int], list[int]]:
    offset = parameter_seed_offset(config.model, config.m, config.triad_probability)
    return (
        make_seeds(config.num_graphs, config.graph_seed_base + offset),
        make_seeds(config.num_graphs, config.rewire_seed_base + offset),
    )


def fixed_rank_windows(width: int, count: int) -> tuple[tuple[int, int], ...]:
    """Return disjoint one-based inclusive edge-gap rank windows."""
    if width < 5:
        raise ValueError("primary rank windows require at least five consecutive gaps")
    if count < 1:
        raise ValueError("num_rank_windows must be positive")
    return tuple((1 + index * width, (index + 1) * width) for index in range(count))


def validate_rank_windows(windows: Sequence[tuple[int, int]]) -> None:
    used: set[int] = set()
    previous_stop = 0
    for start, stop in windows:
        if start < 1 or stop < start or stop - start + 1 < 5:
            raise ValueError(f"invalid fixed-rank gap window {(start, stop)!r}")
        ranks = set(range(start, stop + 1))
        if used & ranks:
            raise ValueError("adjacent spacing windows reuse a gap rank")
        if start <= previous_stop:
            raise ValueError("rank windows must be strictly ordered and disjoint")
        used.update(ranks)
        previous_stop = stop


def select_component(graph: nx.Graph, component_mode: str) -> nx.Graph:
    if component_mode == "full":
        return graph
    if component_mode == "largest_connected_component":
        nodes = max(nx.connected_components(graph), key=len)
        return graph.subgraph(nodes).copy()
    raise ValueError(f"unsupported component_mode={component_mode!r}")


def degree_array(graph: nx.Graph) -> np.ndarray:
    degree_map = dict(graph.degree())
    return np.asarray([degree_map[node] for node in sorted(graph.nodes())], dtype=float)


def encode_array(values: np.ndarray) -> str:
    return " ".join(f"{value:.17g}" for value in np.asarray(values, dtype=float))


def decode_array(value: object) -> np.ndarray:
    text = str(value).strip()
    return np.fromstring(text, sep=" ", dtype=float) if text else np.array([], dtype=float)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def primary_tail_quantile(config: PreferentialAttachmentConfig, denominator: int) -> float:
    configured = tuple(float(value) for value in config.tail_lower_quantiles)
    if PRIMARY_TAIL_QUANTILE in configured and top_count_for_quantile(denominator, PRIMARY_TAIL_QUANTILE) >= 20:
        return PRIMARY_TAIL_QUANTILE
    for q0 in configured:
        if top_count_for_quantile(denominator, q0) >= 20:
            return q0
    return configured[0]


def tail_threshold_rows(
    *,
    config: PreferentialAttachmentConfig,
    graph_index: int,
    model_variant: str,
    degree_desc: np.ndarray,
    spectrum: SparseTailSpectrum,
    analyzed_dimension: int,
) -> list[dict[str, object]]:
    """Fit nested realized-degree and non-Perron spectral tails."""
    rows: list[dict[str, object]] = []
    degree_denominator = len(degree_desc)
    spectral_denominator = analyzed_dimension - 1
    for q0 in config.tail_lower_quantiles:
        degree_target = top_count_for_quantile(degree_denominator, q0)
        spectral_target = top_count_for_quantile(spectral_denominator, q0)
        degree_tau, degree_xmin, degree_count = hill_ccdf_exponent_from_top(degree_desc, degree_target)
        spectral_tau, spectral_xmin, spectral_count = hill_ccdf_exponent_from_top(
            spectrum.absolute_nonperron_tail,
            spectral_target,
        )
        squared_tau, squared_xmin, squared_count = hill_ccdf_exponent_from_top(
            spectrum.lambda_squared_tail,
            spectral_target,
        )
        degree_reg = log_ccdf_regression(degree_desc, degree_denominator, degree_target)
        spectral_reg = log_ccdf_regression(
            spectrum.absolute_nonperron_tail,
            spectral_denominator,
            spectral_target,
        )
        squared_reg = log_ccdf_regression(
            spectrum.lambda_squared_tail,
            spectral_denominator,
            spectral_target,
        )
        rows.append(
            {
                "graph_index": graph_index,
                "model_variant": model_variant,
                "theoretical_degree_ccdf_tau": config.theoretical_degree_ccdf_tau,
                "tail_lower_quantile": q0,
                "is_primary_threshold": q0 == primary_tail_quantile(config, spectral_denominator),
                "degree_tail_observations": degree_count,
                "spectral_tail_observations": spectral_count,
                "squared_eigenvalue_tail_observations": squared_count,
                "degree_ccdf_denominator": degree_denominator,
                "spectral_ccdf_denominator": spectral_denominator,
                "degree_xmin": degree_xmin,
                "spectral_xmin": spectral_xmin,
                "squared_eigenvalue_xmin": squared_xmin,
                "realized_degree_tau_hat": degree_tau,
                "absolute_spectral_tau_hat": spectral_tau,
                "squared_eigenvalue_tau_hat": squared_tau,
                "absolute_spectral_tau_minus_2_degree_tau": (
                    spectral_tau - 2.0 * degree_tau
                    if np.isfinite([degree_tau, spectral_tau]).all()
                    else math.nan
                ),
                "squared_eigenvalue_tau_minus_degree_tau": (
                    squared_tau - degree_tau
                    if np.isfinite([degree_tau, squared_tau]).all()
                    else math.nan
                ),
                "realized_degree_tau_regression": degree_reg["ccdf_regression_exponent"],
                "realized_degree_log_ccdf_r2": degree_reg["ccdf_regression_r2"],
                "absolute_spectral_tau_regression": spectral_reg["ccdf_regression_exponent"],
                "absolute_spectral_log_ccdf_r2": spectral_reg["ccdf_regression_r2"],
                "squared_eigenvalue_tau_regression": squared_reg["ccdf_regression_exponent"],
                "squared_eigenvalue_log_ccdf_r2": squared_reg["ccdf_regression_r2"],
                # Compatibility aliases for the current Chung-Lu tail tables.
                "alpha_hill": degree_tau,
                "beta_hill": spectral_tau,
                "gamma_hill": squared_tau,
                "beta_minus_2alpha_hill": (
                    spectral_tau - 2.0 * degree_tau
                    if np.isfinite([degree_tau, spectral_tau]).all()
                    else math.nan
                ),
                "gamma_minus_alpha_hill": (
                    squared_tau - degree_tau
                    if np.isfinite([degree_tau, squared_tau]).all()
                    else math.nan
                ),
            }
        )
    return rows


def rank_match_rows(
    *,
    config: PreferentialAttachmentConfig,
    graph_index: int,
    model_variant: str,
    degree_desc: np.ndarray,
    spectrum: SparseTailSpectrum,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for side, values in (
        ("positive_nonperron", spectrum.positive_nonperron),
        ("negative_absolute", np.abs(spectrum.negative_eigenvalues)),
    ):
        rows.append(
            {
                "graph_index": graph_index,
                "model_variant": model_variant,
                "side": side,
                "reference_slope": 0.5,
                **rank_match_fit(
                    degree_desc,
                    values,
                    max_rank=config.target_tail_observations,
                ),
            }
        )
    return rows


def spacing_localization_rows(
    *,
    config: PreferentialAttachmentConfig,
    graph_index: int,
    model_variant: str,
    spectrum: SparseTailSpectrum,
    analyzed_dimension: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Compute rank-safe edge spacing and localization from matched eigenpairs."""
    order = np.argsort(spectrum.eigenvalues_la)[::-1]
    values_desc = np.asarray(spectrum.eigenvalues_la[order], dtype=float)
    vectors_desc = np.asarray(spectrum.eigenvectors_la[:, order], dtype=float)
    nonperron_values = values_desc[1:]
    nonperron_vectors = vectors_desc[:, 1:]
    rows: list[dict[str, object]] = []
    decay_rows: list[dict[str, object]] = []
    for rank_start, rank_stop in config.rank_windows:
        value_start = rank_start - 1
        value_stop = rank_stop + 1
        if value_stop > len(nonperron_values):
            raise ValueError(
                f"k_per_side={config.k_per_side} cannot support gap ranks "
                f"{rank_start}-{rank_stop}; need at least {value_stop + 1} LA eigenpairs including Perron"
            )
        selected_values = nonperron_values[value_start:value_stop]
        selected_vectors = nonperron_vectors[:, value_start:value_stop]
        raw_gaps = selected_values[:-1] - selected_values[1:]
        if len(raw_gaps) < 5 or not np.all(np.isfinite(raw_gaps)) or np.any(raw_gaps < -1e-12):
            raise ValueError("fixed-rank window does not contain at least five valid consecutive gaps")
        raw_gaps = np.maximum(raw_gaps, 0.0)
        mean_gap = float(np.mean(raw_gaps))
        normalized = raw_gaps / mean_gap if mean_gap > 0 else np.full_like(raw_gaps, np.nan)
        ratios = adjacent_gap_ratios_from_spacings(raw_gaps)
        localization = eigenvector_localization_summary(selected_vectors, analyzed_dimension)
        window_row: dict[str, object] = {
            "graph_index": graph_index,
            "model_variant": model_variant,
            "spacing_window_mode": SPACING_WINDOW_MODE,
            "spacing_normalization": SPACING_NORMALIZATION,
            "edge_gap_rank_start": rank_start,
            "edge_gap_rank_stop": rank_stop,
            "edge_gap_rank_midpoint": 0.5 * (rank_start + rank_stop),
            "number_of_eigenvalues": len(selected_values),
            "number_of_gaps": len(raw_gaps),
            "number_of_gap_ratios": len(ratios),
            "graph_mean_gap_ratio": float(np.mean(ratios)) if len(ratios) else math.nan,
            "small_gap_probability": float(np.mean(normalized < SMALL_SPACING_THRESHOLD)),
            "mean_raw_gap": mean_gap,
            "E_high": float(selected_values[0]),
            "E_low": float(selected_values[-1]),
            "analyzed_dimension": analyzed_dimension,
            "selected_zero_eigenvalue_count": int(
                np.sum(np.isclose(selected_values, 0.0, atol=ZERO_EIGENVALUE_ATOL, rtol=0.0))
            ),
            "raw_gaps": encode_array(raw_gaps),
            "normalized_spacings": encode_array(normalized),
            "gap_ratios": encode_array(ratios),
            **localization,
        }
        rows.append(window_row)
        for decay in coordinate_decay_profile_rows(
            selected_vectors,
            max_rank=config.coordinate_decay_max_rank,
        ):
            decay_rows.append(
                {
                    "graph_index": graph_index,
                    "model_variant": model_variant,
                    "edge_gap_rank_start": rank_start,
                    "edge_gap_rank_stop": rank_stop,
                    "edge_gap_rank_midpoint": 0.5 * (rank_start + rank_stop),
                    "number_of_eigenvectors": len(selected_values),
                    "analyzed_dimension": analyzed_dimension,
                    **decay,
                }
            )
    return rows, decay_rows


def dense_validation(
    graph: nx.Graph,
    spectrum: SparseTailSpectrum,
    *,
    retain_full_spectrum: bool,
) -> tuple[dict[str, object], np.ndarray]:
    if not retain_full_spectrum:
        return (
            {
                "dense_max_abs_error_recomputed": math.nan,
                "exact_zero_eigenvalue_multiplicity": None,
                "dense_spectrum_computed": False,
            },
            np.array([], dtype=float),
        )
    values, vectors = adjacency_eigendecomposition(graph)
    values = np.asarray(values, dtype=float)
    la = values[-len(spectrum.eigenvalues_la) :]
    sa = values[: len(spectrum.eigenvalues_sa)]
    edge_error = float(
        max(
            np.max(np.abs(np.sort(spectrum.eigenvalues_la) - np.sort(la))),
            np.max(np.abs(np.sort(spectrum.eigenvalues_sa) - np.sort(sa))),
        )
    )
    norms = np.linalg.norm(vectors, axis=0)
    return (
        {
            "dense_max_abs_error_recomputed": edge_error,
            "exact_zero_eigenvalue_multiplicity": int(
                np.sum(np.isclose(values, 0.0, atol=ZERO_EIGENVALUE_ATOL, rtol=0.0))
            ),
            "dense_eigenvector_max_norm_error": float(np.max(np.abs(norms - 1.0))),
            "dense_spectrum_computed": True,
        },
        values,
    )


def analyze_variant(
    *,
    graph: nx.Graph,
    config: PreferentialAttachmentConfig,
    graph_index: int,
    graph_seed: int,
    rewire_seed: int,
    model_variant: str,
    edge_overlap_fraction: float,
    requested_swaps: int,
    max_tries: int,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, np.ndarray],
]:
    start = time.perf_counter()
    analyzed_graph = select_component(graph, config.component_mode)
    degrees = degree_array(analyzed_graph)
    degree_desc = np.sort(degrees[degrees > 0])[::-1]
    spectrum = sparse_tail_spectrum(
        analyzed_graph,
        model_variant=model_variant,
        k_per_side=config.k_per_side,
        eigsh_tol=config.eigsh_tol,
        eigsh_maxiter=config.eigsh_maxiter,
        eigsh_ncv_factor=config.eigsh_ncv_factor,
        validate_dense=config.validate_dense and not config.dense_full_spectrum,
    )
    threshold_rows = tail_threshold_rows(
        config=config,
        graph_index=graph_index,
        model_variant=model_variant,
        degree_desc=degree_desc,
        spectrum=spectrum,
        analyzed_dimension=analyzed_graph.number_of_nodes(),
    )
    ranks = rank_match_rows(
        config=config,
        graph_index=graph_index,
        model_variant=model_variant,
        degree_desc=degree_desc,
        spectrum=spectrum,
    )
    spacing_rows, decay_rows = spacing_localization_rows(
        config=config,
        graph_index=graph_index,
        model_variant=model_variant,
        spectrum=spectrum,
        analyzed_dimension=analyzed_graph.number_of_nodes(),
    )
    dense_info, dense_values = dense_validation(
        analyzed_graph,
        spectrum,
        retain_full_spectrum=config.dense_full_spectrum,
    )
    components = list(nx.connected_components(graph))
    primary_q = primary_tail_quantile(config, analyzed_graph.number_of_nodes() - 1)
    primary = next(row for row in threshold_rows if float(row["tail_lower_quantile"]) == primary_q)
    prefactor = transformed_prefactor(
        degree_desc,
        spectrum.absolute_nonperron_tail,
        denominator_degree=analyzed_graph.number_of_nodes(),
        denominator_lambda=analyzed_graph.number_of_nodes() - 1,
        top_count=top_count_for_quantile(analyzed_graph.number_of_nodes() - 1, primary_q),
    )
    diagnostics: dict[str, object] = {
        "graph_index": graph_index,
        "model_variant": model_variant,
        "graph_seed": graph_seed,
        "rewire_seed": rewire_seed,
        "n_requested": config.n,
        "m": config.m,
        "target_mean_degree": config.target_mean_degree,
        "num_nodes": graph.number_of_nodes(),
        "analyzed_dimension": analyzed_graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "realized_mean_degree": float(np.mean(degree_array(graph))),
        "max_realized_degree": int(np.max(degree_array(graph))),
        "num_components": len(components),
        "largest_component_size": max(map(len, components)),
        "num_isolated_vertices": int(np.sum(degree_array(graph) == 0)),
        "component_mode": config.component_mode,
        "edge_overlap_fraction": edge_overlap_fraction,
        "requested_swaps": requested_swaps,
        "swaps_per_edge": config.swaps_per_edge if requested_swaps else 0,
        "max_tries": max_tries,
        "perron_eigenvalue": spectrum.perron_eigenvalue,
        "perron_excluded_from_primary_tail": True,
        "spectral_ccdf_denominator": analyzed_graph.number_of_nodes() - 1,
        "retained_absolute_tail_count": spectrum.retained_absolute_count,
        "primary_tail_lower_quantile": primary_q,
        "primary_degree_tail_observations": primary["degree_tail_observations"],
        "primary_spectral_tail_observations": primary["spectral_tail_observations"],
        "transformed_ccdf_prefactor": prefactor,
        "eigsh_seconds": spectrum.eigsh_seconds,
        "max_residual_la": float(np.max(spectrum.residuals_la)),
        "max_residual_sa": float(np.max(spectrum.residuals_sa)),
        "dense_max_abs_error": (
            dense_info["dense_max_abs_error_recomputed"]
            if config.dense_full_spectrum
            else spectrum.dense_max_abs_error
        ),
        "zero_count_in_retained_edge_eigenpairs": int(
            np.sum(np.isclose(spectrum.eigenvalues_la, 0.0, atol=ZERO_EIGENVALUE_ATOL, rtol=0.0))
            + np.sum(np.isclose(spectrum.eigenvalues_sa, 0.0, atol=ZERO_EIGENVALUE_ATOL, rtol=0.0))
        ),
        "analysis_seconds": time.perf_counter() - start,
        "process_peak_rss_mb": peak_rss_mb(),
        **dense_info,
    }
    arrays = {
        "degrees": degrees,
        "degree_desc": degree_desc,
        "absolute_nonperron_tail": spectrum.absolute_nonperron_tail,
        "lambda_squared_tail": spectrum.lambda_squared_tail,
        "positive_nonperron": spectrum.positive_nonperron,
        "negative_eigenvalues": spectrum.negative_eigenvalues,
        "retained_edge_eigenvalues": np.concatenate(
            (spectrum.eigenvalues_sa, spectrum.eigenvalues_la)
        ),
        "dense_eigenvalues": dense_values,
    }
    return diagnostics, threshold_rows, ranks, spacing_rows, decay_rows, arrays


def degree_preservation_row(
    *,
    graph_index: int,
    original: nx.Graph,
    rewired: nx.Graph,
    graph_seed: int,
    rewire_seed: int,
    requested_swaps: int,
    max_tries: int,
    overlap: float,
    config: PreferentialAttachmentConfig,
) -> dict[str, object]:
    original_degrees = degree_array(original)
    rewired_degrees = degree_array(rewired)
    return {
        "graph_index": graph_index,
        "graph_seed": graph_seed,
        "rewire_seed": rewire_seed,
        "same_node_labels": set(original.nodes()) == set(rewired.nodes()),
        "same_number_of_vertices": original.number_of_nodes() == rewired.number_of_nodes(),
        "same_number_of_edges": original.number_of_edges() == rewired.number_of_edges(),
        "degree_match": bool(np.array_equal(original_degrees, rewired_degrees)),
        "mean_degree_match": bool(np.mean(original_degrees) == np.mean(rewired_degrees)),
        "max_degree_match": bool(np.max(original_degrees) == np.max(rewired_degrees)),
        "number_of_edges": original.number_of_edges(),
        "realized_mean_degree": float(np.mean(original_degrees)),
        "max_realized_degree": int(np.max(original_degrees)),
        "requested_swaps": requested_swaps,
        "swaps_per_edge": config.swaps_per_edge,
        "max_tries": max_tries,
        "max_tries_factor": config.max_tries_factor,
        "edge_overlap_fraction": overlap,
    }


def save_graph_partial(
    path: Path,
    *,
    metadata: dict[str, object],
    diagnostics_rows: list[dict[str, object]],
    threshold_rows: list[dict[str, object]],
    rank_rows: list[dict[str, object]],
    spacing_rows: list[dict[str, object]],
    decay_rows: list[dict[str, object]],
    degree_check: dict[str, object] | None,
    arrays: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        metadata_json=np.asarray(json.dumps(metadata)),
        diagnostics_json=np.asarray(json.dumps(diagnostics_rows)),
        threshold_json=np.asarray(json.dumps(threshold_rows)),
        rank_json=np.asarray(json.dumps(rank_rows)),
        spacing_json=np.asarray(json.dumps(spacing_rows)),
        decay_json=np.asarray(json.dumps(decay_rows)),
        degree_check_json=np.asarray(json.dumps(degree_check)),
        **arrays,
    )


def load_graph_partial(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        json_keys = {
            "metadata_json",
            "diagnostics_json",
            "threshold_json",
            "rank_json",
            "spacing_json",
            "decay_json",
            "degree_check_json",
        }
        return {
            "metadata": json.loads(str(data["metadata_json"].item())),
            "diagnostics": json.loads(str(data["diagnostics_json"].item())),
            "threshold_rows": json.loads(str(data["threshold_json"].item())),
            "rank_rows": json.loads(str(data["rank_json"].item())),
            "spacing_rows": json.loads(str(data["spacing_json"].item())),
            "decay_rows": json.loads(str(data["decay_json"].item())),
            "degree_check": json.loads(str(data["degree_check_json"].item())),
            "arrays": {key: np.asarray(data[key]) for key in data.files if key not in json_keys},
        }


def validate_config(config: PreferentialAttachmentConfig) -> None:
    if config.n < 4 or not 1 <= config.m < config.n:
        raise ValueError("require n >= 4 and 1 <= m < n")
    if config.num_graphs < 1:
        raise ValueError("num_graphs must be positive")
    if config.model == MODEL_AFFINE_PA:
        attachment_offset_for_ccdf_tau(config.m, config.theoretical_degree_ccdf_tau)
    if config.target_tail_observations < 20:
        raise ValueError("target_tail_observations must be at least 20")
    if config.k_per_side <= config.rank_windows[-1][1] + 1:
        raise ValueError("k_per_side must exceed the last gap rank plus Perron/eigenvalue margin")
    validate_rank_windows(config.rank_windows)


def run_graph_partial(
    config: PreferentialAttachmentConfig,
    graph_index: int,
    partial_dir: Path | None = None,
) -> Path:
    """Generate and analyze one graph index, then save an aggregation-safe NPZ."""
    validate_config(config)
    if not 0 <= graph_index < config.num_graphs:
        raise ValueError(f"graph_index must lie in [0, {config.num_graphs})")
    graph_seeds, rewire_seeds = seed_lists(config)
    graph_seed = graph_seeds[graph_index]
    rewire_seed = rewire_seeds[graph_index]
    graph_start = time.perf_counter()
    original, generator_diagnostics = generate_preferential_attachment_graph(
        n=config.n,
        m=config.m,
        seed=graph_seed,
        model=config.model,
        triad_probability=config.triad_probability,
        attachment_ccdf_tau=config.theoretical_degree_ccdf_tau,
    )
    generation_seconds = time.perf_counter() - graph_start

    variants: list[tuple[str, nx.Graph, float, int, int, str]] = [
        (config.model, original, math.nan, 0, 0, "original")
    ]
    degree_check: dict[str, object] | None = None
    if config.run_rewired_control:
        rewire_start = time.perf_counter()
        rewired, requested_swaps, max_tries, overlap = rewire_graph_preserving_degrees(
            original,
            rewire_seed=rewire_seed,
            swaps_per_edge=config.swaps_per_edge,
            max_tries_factor=config.max_tries_factor,
        )
        rewiring_seconds = time.perf_counter() - rewire_start
        degree_check = degree_preservation_row(
            graph_index=graph_index,
            original=original,
            rewired=rewired,
            graph_seed=graph_seed,
            rewire_seed=rewire_seed,
            requested_swaps=requested_swaps,
            max_tries=max_tries,
            overlap=overlap,
            config=config,
        )
        degree_check["rewiring_seconds"] = rewiring_seconds
        variants.append(
            (config.rewired_model, rewired, overlap, requested_swaps, max_tries, "rewired")
        )

    diagnostics_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    rank_rows: list[dict[str, object]] = []
    spacing_rows: list[dict[str, object]] = []
    decay_rows: list[dict[str, object]] = []
    arrays: dict[str, np.ndarray] = {}
    for variant, graph, overlap, requested_swaps, max_tries, prefix in variants:
        diagnostics, thresholds, ranks, spacing, decay, variant_arrays = analyze_variant(
            graph=graph,
            config=config,
            graph_index=graph_index,
            graph_seed=graph_seed,
            rewire_seed=rewire_seed,
            model_variant=variant,
            edge_overlap_fraction=overlap,
            requested_swaps=requested_swaps,
            max_tries=max_tries,
        )
        diagnostics.update(
            {
                "model_generator": (
                    "networkx.barabasi_albert_graph"
                    if config.model == MODEL_BA
                    else (
                        "custom_affine_linear_weighted_without_replacement"
                        if config.model == MODEL_AFFINE_PA
                        else "networkx.powerlaw_cluster_graph"
                    )
                ),
                "initial_graph_convention": generator_diagnostics.initial_graph_convention,
                "expected_num_edges": generator_diagnostics.expected_num_edges,
                "generation_seconds": generation_seconds if prefix == "original" else 0.0,
            }
        )
        diagnostics_rows.append(diagnostics)
        threshold_rows.extend(thresholds)
        rank_rows.extend(ranks)
        spacing_rows.extend(spacing)
        decay_rows.extend(decay)
        arrays.update({f"{prefix}__{key}": value for key, value in variant_arrays.items()})

    metadata = {
        "graph_index": graph_index,
        "graph_seed": graph_seed,
        "rewire_seed": rewire_seed,
        "generator_diagnostics": generator_diagnostics.to_dict(),
        "elapsed_seconds": time.perf_counter() - graph_start,
        "peak_rss_mb": peak_rss_mb(),
    }
    destination = partial_dir or (model_output_dir(config) / "partials")
    path = Path(destination) / f"graph_{graph_index:06d}.npz"
    save_graph_partial(
        path,
        metadata=metadata,
        diagnostics_rows=diagnostics_rows,
        threshold_rows=threshold_rows,
        rank_rows=rank_rows,
        spacing_rows=spacing_rows,
        decay_rows=decay_rows,
        degree_check=degree_check,
        arrays=arrays,
    )
    print(
        f"finished graph {graph_index + 1}/{config.num_graphs}: "
        f"seed={graph_seed}, elapsed={metadata['elapsed_seconds']:.2f}s, partial={path}",
        flush=True,
    )
    return path


def aggregate_tail_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row["model_variant"]), float(row["tail_lower_quantile"])),
            [],
        ).append(row)
    metrics = (
        "realized_degree_tau_hat",
        "absolute_spectral_tau_hat",
        "squared_eigenvalue_tau_hat",
        "absolute_spectral_tau_minus_2_degree_tau",
        "squared_eigenvalue_tau_minus_degree_tau",
        "realized_degree_tau_regression",
        "realized_degree_log_ccdf_r2",
        "absolute_spectral_tau_regression",
        "absolute_spectral_log_ccdf_r2",
        "squared_eigenvalue_tau_regression",
        "squared_eigenvalue_log_ccdf_r2",
    )
    summaries: list[dict[str, object]] = []
    for (variant, q0), group in sorted(grouped.items()):
        summary: dict[str, object] = {
            "model_variant": variant,
            "tail_lower_quantile": q0,
            "is_primary_threshold": bool(group[0]["is_primary_threshold"]),
            "number_of_graphs": len(group),
            "valid_graphs": len(group),
            "degree_tail_observations": int(
                sum(int(row["degree_tail_observations"]) for row in group)
            ),
            "spectral_tail_observations": int(
                sum(int(row["spectral_tail_observations"]) for row in group)
            ),
            "squared_eigenvalue_tail_observations": int(
                sum(int(row["squared_eigenvalue_tail_observations"]) for row in group)
            ),
            "estimator_note": "graph-equal ensemble means and standard errors",
        }
        for metric in metrics:
            mean, se, count = finite_mean_se(row.get(metric, math.nan) for row in group)
            summary[f"mean_{metric}"] = mean
            summary[f"se_{metric}"] = se
            summary[f"valid_graphs_{metric}"] = count
        summaries.append(summary)
    return summaries


def aggregate_rank_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["model_variant"]), str(row["side"])), []).append(row)
    metrics = (
        "slope",
        "slope_se",
        "r2",
        "median_lambda_over_sqrt_degree",
        "iqr_lambda_over_sqrt_degree",
        "spearman_rank_correlation",
    )
    summaries: list[dict[str, object]] = []
    for (variant, side), group in sorted(grouped.items()):
        summary: dict[str, object] = {
            "model_variant": variant,
            "side": side,
            "number_of_graphs": len(group),
            "reference_slope": 0.5,
        }
        for metric in metrics:
            mean, se, count = finite_mean_se(row.get(metric, math.nan) for row in group)
            summary[f"mean_{metric}"] = mean
            summary[f"se_{metric}"] = se
            summary[f"valid_graphs_{metric}"] = count
        summaries.append(summary)
    return summaries


def _goe_cdf(values: np.ndarray) -> np.ndarray:
    x = np.maximum(np.asarray(values, dtype=float), 0.0)
    return 1.0 - np.exp(-np.pi * x**2 / 4.0)


def _poisson_cdf(values: np.ndarray) -> np.ndarray:
    x = np.maximum(np.asarray(values, dtype=float), 0.0)
    return 1.0 - np.exp(-x)


def aggregate_spacing_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, int], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(
            (
                str(row["model_variant"]),
                int(row["edge_gap_rank_start"]),
                int(row["edge_gap_rank_stop"]),
            ),
            [],
        ).append(row)
    summaries: list[dict[str, object]] = []
    for (variant, rank_start, rank_stop), group in sorted(grouped.items()):
        spacing_parts = [decode_array(row["normalized_spacings"]) for row in group]
        normalized = np.concatenate(spacing_parts) if spacing_parts else np.array([], dtype=float)
        normalized = normalized[np.isfinite(normalized)]
        ratio_parts = [decode_array(row["gap_ratios"]) for row in group]
        ratios = np.concatenate(ratio_parts) if ratio_parts else np.array([], dtype=float)
        ratio_mean, ratio_se, ratio_graphs = finite_mean_se(
            row["graph_mean_gap_ratio"] for row in group
        )
        summary: dict[str, object] = {
            "model_variant": variant,
            "edge_gap_rank_start": rank_start,
            "edge_gap_rank_stop": rank_stop,
            "edge_gap_rank_midpoint": 0.5 * (rank_start + rank_stop),
            "number_of_graphs": len(group),
            "valid_graphs": int(sum(int(row["number_of_gaps"]) >= 5 for row in group)),
            "graphs_with_gap_ratios": ratio_graphs,
            "number_of_eigenvalues": int(sum(int(row["number_of_eigenvalues"]) for row in group)),
            "number_of_gaps": int(len(normalized)),
            "number_of_gap_ratios": int(len(ratios)),
            "mean_normalized_spacing": float(np.mean(normalized)) if len(normalized) else math.nan,
            "graph_equal_mean_gap_ratio": ratio_mean,
            "gap_ratio_standard_error": ratio_se,
            "pooled_mean_gap_ratio": float(np.mean(ratios)) if len(ratios) else math.nan,
            "small_gap_probability": (
                float(np.mean(normalized < SMALL_SPACING_THRESHOLD))
                if len(normalized)
                else math.nan
            ),
            "KS_to_GOE": float(kstest(normalized, _goe_cdf).statistic) if len(normalized) else math.nan,
            "KS_to_Poisson": (
                float(kstest(normalized, _poisson_cdf).statistic)
                if len(normalized)
                else math.nan
            ),
            "spacing_window_mode": SPACING_WINDOW_MODE,
            "spacing_normalization": SPACING_NORMALIZATION,
            "normalized_spacings": encode_array(normalized),
        }
        for metric in LOCALIZATION_METRICS:
            mean, se, count = finite_mean_se(row.get(metric, math.nan) for row in group)
            summary[metric] = mean
            summary[f"{metric}_standard_error"] = se
            summary[f"{metric}_graph_count"] = count
        summaries.append(summary)
    return summaries


def aggregate_decay_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, int, int], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(
            (
                str(row["model_variant"]),
                int(row["edge_gap_rank_start"]),
                int(row["edge_gap_rank_stop"]),
                int(float(row["rank"])),
            ),
            [],
        ).append(row)
    summaries: list[dict[str, object]] = []
    for (variant, start, stop, rank), group in sorted(grouped.items()):
        mass_mean, mass_se, count = finite_mean_se(row["mean_coordinate_mass"] for row in group)
        cumulative_mean, cumulative_se, _ = finite_mean_se(
            row["mean_cumulative_mass"] for row in group
        )
        summaries.append(
            {
                "model_variant": variant,
                "edge_gap_rank_start": start,
                "edge_gap_rank_stop": stop,
                "edge_gap_rank_midpoint": 0.5 * (start + stop),
                "rank": rank,
                "number_of_graphs": count,
                "mean_coordinate_mass": mass_mean,
                "se_coordinate_mass": mass_se,
                "mean_cumulative_mass": cumulative_mean,
                "se_cumulative_mass": cumulative_se,
            }
        )
    return summaries


def paired_difference_rows(
    threshold_rows: list[dict[str, object]],
    spacing_rows: list[dict[str, object]],
    config: PreferentialAttachmentConfig,
) -> list[dict[str, object]]:
    differences: list[dict[str, object]] = []
    tail_groups: dict[tuple[int, float], dict[str, dict[str, object]]] = {}
    for row in threshold_rows:
        key = (int(row["graph_index"]), float(row["tail_lower_quantile"]))
        tail_groups.setdefault(key, {})[str(row["model_variant"])] = row
    tail_metrics = (
        "realized_degree_tau_hat",
        "absolute_spectral_tau_hat",
        "squared_eigenvalue_tau_hat",
        "absolute_spectral_tau_minus_2_degree_tau",
        "squared_eigenvalue_tau_minus_degree_tau",
    )
    for (graph_index, q0), variants in sorted(tail_groups.items()):
        if config.model not in variants or config.rewired_model not in variants:
            continue
        original = variants[config.model]
        rewired = variants[config.rewired_model]
        row: dict[str, object] = {
            "analysis_domain": "degree_spectral_tail",
            "graph_index": graph_index,
            "tail_lower_quantile": q0,
            "is_primary_threshold": bool(original["is_primary_threshold"]),
            "difference_definition": "rewired_minus_original",
        }
        for metric in tail_metrics:
            left = float(original[metric])
            right = float(rewired[metric])
            row[f"delta_{metric}"] = (
                right - left if np.isfinite(left) and np.isfinite(right) else math.nan
            )
        differences.append(row)

    spacing_groups: dict[tuple[int, int, int], dict[str, dict[str, object]]] = {}
    for row in spacing_rows:
        key = (
            int(row["graph_index"]),
            int(row["edge_gap_rank_start"]),
            int(row["edge_gap_rank_stop"]),
        )
        spacing_groups.setdefault(key, {})[str(row["model_variant"])] = row
    for (graph_index, rank_start, rank_stop), variants in sorted(spacing_groups.items()):
        if config.model not in variants or config.rewired_model not in variants:
            continue
        original = variants[config.model]
        rewired = variants[config.rewired_model]
        row = {
            "analysis_domain": "spacing_localization",
            "graph_index": graph_index,
            "edge_gap_rank_start": rank_start,
            "edge_gap_rank_stop": rank_stop,
            "edge_gap_rank_midpoint": 0.5 * (rank_start + rank_stop),
            "difference_definition": "rewired_minus_original",
        }
        for metric in PAIRED_METRICS:
            left = float(original[metric])
            right = float(rewired[metric])
            row[f"delta_{metric}"] = (
                right - left if np.isfinite(left) and np.isfinite(right) else math.nan
            )
        differences.append(row)
    return differences


def aggregate_paired_differences(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        if row["analysis_domain"] == "degree_spectral_tail":
            key = ("degree_spectral_tail", float(row["tail_lower_quantile"]))
        else:
            key = (
                "spacing_localization",
                int(row["edge_gap_rank_start"]),
                int(row["edge_gap_rank_stop"]),
            )
        grouped.setdefault(key, []).append(row)
    summaries: list[dict[str, object]] = []
    for key, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        summary: dict[str, object] = {
            "analysis_domain": key[0],
            "number_of_pairs": len(group),
            "difference_definition": "rewired_minus_original",
        }
        if key[0] == "degree_spectral_tail":
            summary["tail_lower_quantile"] = key[1]
        else:
            summary.update(
                {
                    "edge_gap_rank_start": key[1],
                    "edge_gap_rank_stop": key[2],
                    "edge_gap_rank_midpoint": 0.5 * (float(key[1]) + float(key[2])),
                }
            )
        delta_keys = sorted({name for row in group for name in row if name.startswith("delta_")})
        for metric in delta_keys:
            mean, se, count = finite_mean_se(row.get(metric, math.nan) for row in group)
            summary[f"mean_{metric}"] = mean
            summary[f"se_{metric}"] = se
            summary[f"valid_pairs_{metric}"] = count
        summaries.append(summary)
    return summaries


def pooled_curve_rows(
    config: PreferentialAttachmentConfig,
    partials: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    ccdf_rows: list[dict[str, object]] = []
    rank_rows: list[dict[str, object]] = []
    max_rank = max(
        config.target_tail_observations,
        top_count_for_quantile(config.n - 1, min(config.tail_lower_quantiles)),
    )
    variants = [("original", config.model)]
    if config.run_rewired_control:
        variants.append(("rewired", config.rewired_model))
    for partial in partials:
        graph_index = int(partial["metadata"]["graph_index"])
        arrays = partial["arrays"]
        degree_desc = np.asarray(arrays["original__degree_desc"], dtype=float)
        degree_denominator = len(np.asarray(arrays["original__degrees"], dtype=float))
        for rank, value in enumerate(degree_desc[:max_rank], start=1):
            ccdf_rows.append(
                {
                    "graph_index": graph_index,
                    "model_variant": "realized degree sequence",
                    "curve": "realized_degree_ccdf",
                    "rank": rank,
                    "value": float(value),
                    "tail_probability": rank / float(degree_denominator),
                    "denominator": degree_denominator,
                }
            )
        for prefix, variant in variants:
            absolute_values = np.asarray(arrays[f"{prefix}__absolute_nonperron_tail"], dtype=float)
            squared_values = np.asarray(arrays[f"{prefix}__lambda_squared_tail"], dtype=float)
            spectral_denominator = int(
                next(
                    row["spectral_ccdf_denominator"]
                    for row in partial["diagnostics"]
                    if row["model_variant"] == variant
                )
            )
            for curve, values in (
                ("absolute_spectral_ccdf", absolute_values),
                ("squared_eigenvalue_ccdf", squared_values),
            ):
                for rank, value in enumerate(values[:max_rank], start=1):
                    ccdf_rows.append(
                        {
                            "graph_index": graph_index,
                            "model_variant": variant,
                            "curve": curve,
                            "rank": rank,
                            "value": float(value),
                            "tail_probability": rank / float(spectral_denominator),
                            "denominator": spectral_denominator,
                        }
                    )
            for side, values in (
                ("positive_nonperron", np.asarray(arrays[f"{prefix}__positive_nonperron"], dtype=float)),
                ("negative_absolute", np.abs(np.asarray(arrays[f"{prefix}__negative_eigenvalues"], dtype=float))),
            ):
                count = min(config.target_tail_observations, len(degree_desc), len(values))
                for rank in range(count):
                    rank_rows.append(
                        {
                            "graph_index": graph_index,
                            "model_variant": variant,
                            "side": side,
                            "rank": rank + 1,
                            "sqrt_degree": float(np.sqrt(degree_desc[rank])),
                            "eigenvalue_magnitude": float(values[rank]),
                        }
                    )
    return ccdf_rows, rank_rows


def _median_curve(
    rows: list[dict[str, object]],
    *,
    curve: str,
    model_variant: str,
) -> tuple[np.ndarray, np.ndarray]:
    selected = [
        row
        for row in rows
        if row["curve"] == curve and row["model_variant"] == model_variant
    ]
    grouped: dict[int, list[tuple[float, float]]] = {}
    for row in selected:
        grouped.setdefault(int(row["rank"]), []).append(
            (float(row["value"]), float(row["tail_probability"]))
        )
    x = [float(np.median([value for value, _ in grouped[rank]])) for rank in sorted(grouped)]
    y = [float(np.median([prob for _, prob in grouped[rank]])) for rank in sorted(grouped)]
    return np.asarray(x), np.asarray(y)


def figure_metadata(
    config: PreferentialAttachmentConfig,
    diagnostics: list[dict[str, object]],
    number_of_graphs: int,
) -> str:
    originals = [row for row in diagnostics if row["model_variant"] == config.model]
    mean_degree = float(np.mean([float(row["realized_mean_degree"]) for row in originals]))
    zero_text = (
        "exact zero multiplicity recorded before removal"
        if config.dense_full_spectrum
        else "central zeros not computed; edge windows do not select them"
    )
    return (
        f"model={config.model}; N={config.n}; m={config.m}; mean degree={mean_degree:.4f}; "
        f"graphs={number_of_graphs}; component={config.component_mode}\n"
        f"Perron excluded; {zero_text}; selection={SPECTRAL_SELECTION_METHOD}"
    )


def save_figure(
    figure: plt.Figure,
    path: Path,
    *,
    title: str,
    metadata: str,
) -> None:
    figure.suptitle(f"{title}\n{metadata}", fontsize=9, y=1.02)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_ccdf_figures(
    *,
    config: PreferentialAttachmentConfig,
    output_dir: Path,
    ccdf_rows: list[dict[str, object]],
    diagnostics: list[dict[str, object]],
    number_of_graphs: int,
) -> None:
    metadata = figure_metadata(config, diagnostics, number_of_graphs)
    degree_x, degree_y = _median_curve(
        ccdf_rows,
        curve="realized_degree_ccdf",
        model_variant="realized degree sequence",
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(degree_x, degree_y, marker=".", label="realized degree")
    ax.set_xlabel("realized degree k")
    ax.set_ylabel("P(D >= k)")
    ax.legend()
    save_figure(fig, output_dir / "realized_degree_ccdf.png", title="Realized degree CCDF", metadata=metadata)

    variants = [config.model]
    if config.run_rewired_control:
        variants.append(config.rewired_model)
    fig, ax = plt.subplots(figsize=(7, 5))
    for variant in variants:
        x, y = _median_curve(ccdf_rows, curve="absolute_spectral_ccdf", model_variant=variant)
        ax.loglog(x, y, marker=".", label=variant)
    ax.set_xlabel("|lambda|, Perron excluded")
    ax.set_ylabel("spectral CCDF, denominator N-1")
    ax.legend(fontsize=8)
    save_figure(fig, output_dir / "absolute_spectral_ccdf.png", title="Absolute non-Perron spectral CCDF", metadata=metadata)

    fig, ax = plt.subplots(figsize=(7, 5))
    for variant in variants:
        x, y = _median_curve(ccdf_rows, curve="absolute_spectral_ccdf", model_variant=variant)
        ax.loglog(x, y, marker=".", label=f"{variant}: T_lambda(x)")
    prefactors = [
        float(row["transformed_ccdf_prefactor"])
        for row in diagnostics
        if row["model_variant"] == config.model
        and np.isfinite(float(row["transformed_ccdf_prefactor"]))
    ]
    prefactor = float(np.median(prefactors)) if prefactors else 1.0
    ax.loglog(np.sqrt(degree_x), prefactor * degree_y, color="black", linestyle="--", label=f"C T_D(x^2), C={prefactor:.3g}")
    ax.set_xlabel("x")
    ax.set_ylabel("tail probability")
    ax.legend(fontsize=8)
    save_figure(fig, output_dir / "degree_spectral_ccdf_match.png", title="Degree-to-spectral tail match", metadata=metadata)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(degree_x, degree_y, color="black", linestyle="--", label="realized degree")
    for variant in variants:
        x, y = _median_curve(ccdf_rows, curve="squared_eigenvalue_ccdf", model_variant=variant)
        ax.loglog(x, y, marker=".", label=f"{variant}: lambda^2")
    ax.set_xlabel("degree or lambda^2")
    ax.set_ylabel("CCDF")
    ax.legend(fontsize=8)
    save_figure(fig, output_dir / "squared_eigenvalue_degree_ccdf_match.png", title="Squared-eigenvalue and degree CCDFs", metadata=metadata)


def plot_rank_matches(
    *,
    config: PreferentialAttachmentConfig,
    output_dir: Path,
    rows: list[dict[str, object]],
    diagnostics: list[dict[str, object]],
    number_of_graphs: int,
) -> None:
    metadata = figure_metadata(config, diagnostics, number_of_graphs)
    for side, filename, label in (
        ("positive_nonperron", "positive_rank_match.png", "positive non-Perron"),
        ("negative_absolute", "negative_rank_match.png", "absolute negative"),
    ):
        fig, ax = plt.subplots(figsize=(7, 5))
        variants = [config.model] + ([config.rewired_model] if config.run_rewired_control else [])
        for variant in variants:
            selected = [row for row in rows if row["side"] == side and row["model_variant"] == variant]
            grouped: dict[int, list[tuple[float, float]]] = {}
            for row in selected:
                grouped.setdefault(int(row["rank"]), []).append(
                    (float(row["sqrt_degree"]), float(row["eigenvalue_magnitude"]))
                )
            x = [np.median([point[0] for point in grouped[rank]]) for rank in sorted(grouped)]
            y = [np.median([point[1] for point in grouped[rank]]) for rank in sorted(grouped)]
            ax.loglog(x, y, marker=".", linewidth=1, label=variant)
        limits = ax.get_xlim()
        grid = np.geomspace(max(limits[0], 1e-6), max(limits[1], limits[0] * 1.01), 100)
        ax.loglog(grid, grid, color="black", linestyle="--", linewidth=1, label="|lambda| = sqrt(D)")
        ax.set_xlabel("ranked sqrt(realized degree)")
        ax.set_ylabel(f"ranked {label} eigenvalue magnitude")
        ax.legend(fontsize=8)
        save_figure(fig, output_dir / filename, title=f"{label.capitalize()} rank match", metadata=metadata)


def plot_global_density(
    *,
    config: PreferentialAttachmentConfig,
    output_dir: Path,
    partials: list[dict[str, object]],
    diagnostics: list[dict[str, object]],
    number_of_graphs: int,
) -> None:
    metadata = figure_metadata(config, diagnostics, number_of_graphs)
    variants = [("original", config.model)]
    if config.run_rewired_control:
        variants.append(("rewired", config.rewired_model))
    dense_available = any(
        len(np.asarray(partial["arrays"].get("original__dense_eigenvalues", [])))
        for partial in partials
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    for prefix, variant in variants:
        key = f"{prefix}__dense_eigenvalues" if dense_available else f"{prefix}__retained_edge_eigenvalues"
        samples = []
        for partial in partials:
            sample = np.asarray(partial["arrays"].get(key, []), dtype=float)
            if not len(sample):
                continue
            perron = float(
                next(
                    row["perron_eigenvalue"]
                    for row in partial["diagnostics"]
                    if row["model_variant"] == variant
                )
            )
            sample = np.delete(sample, int(np.argmin(np.abs(sample - perron))))
            samples.append(sample)
        if not samples:
            continue
        values = np.concatenate(samples)
        if dense_available:
            if REMOVE_EXACT_ZERO_EIGENVALUES:
                values = values[~np.isclose(values, 0.0, atol=ZERO_EIGENVALUE_ATOL, rtol=0.0)]
        ax.hist(values, bins=100, density=True, histtype="step", linewidth=1.4, label=variant)
    ax.set_xlabel("adjacency eigenvalue")
    ax.set_ylabel("empirical density")
    ax.legend(fontsize=8)
    title = "Global continuous spectral density" if dense_available else "Retained sparse edge spectrum (not a global density estimate)"
    if not dense_available:
        ax.text(
            0.02,
            0.96,
            "Sparse pilot: central spectrum and exact zero multiplicity were not computed.",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )
    save_figure(fig, output_dir / "global_spectral_density.png", title=title, metadata=metadata)


def plot_spacing_diagnostics(
    *,
    config: PreferentialAttachmentConfig,
    output_dir: Path,
    summaries: list[dict[str, object]],
    diagnostics: list[dict[str, object]],
    number_of_graphs: int,
) -> None:
    metadata = figure_metadata(config, diagnostics, number_of_graphs)
    windows = config.rank_windows
    columns = 3
    rows_count = int(math.ceil(len(windows) / columns))
    fig, axes = plt.subplots(rows_count, columns, figsize=(12, 3.4 * rows_count), squeeze=False)
    grid = np.linspace(0.0, 3.5, 400)
    for ax, (rank_start, rank_stop) in zip(axes.flat, windows):
        for variant in [config.model] + ([config.rewired_model] if config.run_rewired_control else []):
            row = next(
                value
                for value in summaries
                if value["model_variant"] == variant
                and int(value["edge_gap_rank_start"]) == rank_start
                and int(value["edge_gap_rank_stop"]) == rank_stop
            )
            values = decode_array(row["normalized_spacings"])
            ax.hist(values, bins=np.linspace(0, 3.5, 24), density=True, histtype="step", label=variant)
        ax.plot(grid, 0.5 * np.pi * grid * np.exp(-np.pi * grid**2 / 4.0), color="black", linewidth=1, label="GOE")
        ax.plot(grid, np.exp(-grid), color="black", linestyle="--", linewidth=1, label="Poisson")
        ax.set_title(f"edge gap ranks {rank_start}-{rank_stop}")
        ax.set_xlim(0, 3.5)
        ax.set_xlabel("normalized spacing")
        ax.set_ylabel("density")
    for ax in axes.flat[len(windows) :]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=7)
    save_figure(fig, output_dir / "spacing_histograms.png", title="Fixed-rank edge spacing histograms", metadata=metadata)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for variant, marker in ((config.model, "o"), (config.rewired_model, "s")):
        group = [row for row in summaries if row["model_variant"] == variant]
        if not group:
            continue
        x = [float(row["edge_gap_rank_midpoint"]) for row in group]
        y = [float(row["graph_equal_mean_gap_ratio"]) for row in group]
        err = [float(row["gap_ratio_standard_error"]) if np.isfinite(float(row["gap_ratio_standard_error"])) else 0.0 for row in group]
        axes[0].errorbar(x, y, yerr=err, marker=marker, label=variant)
        axes[1].plot(x, [float(row["KS_to_GOE"]) for row in group], marker=marker, label=f"{variant}: GOE")
        axes[1].plot(x, [float(row["KS_to_Poisson"]) for row in group], marker=marker, linestyle="--", label=f"{variant}: Poisson")
    axes[0].axhline(0.5359, color="black", linewidth=1, label="GOE mean ratio")
    axes[0].axhline(0.3863, color="black", linewidth=1, linestyle="--", label="Poisson mean ratio")
    axes[0].set_xlabel("non-Perron edge gap rank")
    axes[0].set_ylabel("graph-equal mean adjacent-gap ratio")
    axes[1].set_xlabel("non-Perron edge gap rank")
    axes[1].set_ylabel("KS distance")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    save_figure(fig, output_dir / "gap_ratio_summary.png", title="Edge repulsion diagnostics", metadata=metadata)


def plot_localization_diagnostics(
    *,
    config: PreferentialAttachmentConfig,
    output_dir: Path,
    summaries: list[dict[str, object]],
    decay_summaries: list[dict[str, object]],
    diagnostics: list[dict[str, object]],
    number_of_graphs: int,
) -> None:
    metadata = figure_metadata(config, diagnostics, number_of_graphs)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for variant, marker in ((config.model, "o"), (config.rewired_model, "s")):
        group = [row for row in summaries if row["model_variant"] == variant]
        if not group:
            continue
        x = [float(row["edge_gap_rank_midpoint"]) for row in group]
        axes[0].plot(x, [float(row["mean_n_times_ipr"]) for row in group], marker=marker, label=variant)
        axes[1].plot(x, [float(row["mean_participation_fraction"]) for row in group], marker=marker, label=variant)
        axes[2].plot(x, [float(row["mean_top_10_coordinate_mass"]) for row in group], marker=marker, label=variant)
    axes[0].set_ylabel("mean N * IPR")
    axes[1].set_ylabel("mean PR / N")
    axes[2].set_ylabel("mean top-10 coordinate mass")
    for ax in axes:
        ax.set_xlabel("non-Perron edge gap rank")
        ax.legend(fontsize=7)
    save_figure(fig, output_dir / "localization_summary.png", title="Eigenvector localization toward the edge", metadata=metadata)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = plt.cm.viridis(np.linspace(0.12, 0.9, len(config.rank_windows)))
    for color, (rank_start, rank_stop) in zip(colors, config.rank_windows):
        for variant, linestyle in ((config.model, "-"), (config.rewired_model, "--")):
            group = [
                row
                for row in decay_summaries
                if row["model_variant"] == variant
                and int(row["edge_gap_rank_start"]) == rank_start
                and int(row["edge_gap_rank_stop"]) == rank_stop
            ]
            if not group:
                continue
            ranks = [int(row["rank"]) for row in group]
            label = f"ranks {rank_start}-{rank_stop}" if variant == config.model else None
            axes[0].loglog(ranks, [float(row["mean_coordinate_mass"]) for row in group], color=color, linestyle=linestyle, label=label)
            axes[1].semilogx(ranks, [float(row["mean_cumulative_mass"]) for row in group], color=color, linestyle=linestyle)
    axes[0].set_xlabel("sorted coordinate rank r")
    axes[0].set_ylabel("mean |v_(r)|^2")
    axes[1].set_xlabel("top-r coordinates")
    axes[1].set_ylabel("mean cumulative mass")
    axes[0].legend(fontsize=7)
    axes[1].text(0.03, 0.05, "solid: original; dashed: exact-degree rewired", transform=axes[1].transAxes, fontsize=8)
    save_figure(fig, output_dir / "coordinate_decay_summary.png", title="Sorted eigenvector coordinate decay", metadata=metadata)


def plot_paired_summary(
    *,
    config: PreferentialAttachmentConfig,
    output_dir: Path,
    summaries: list[dict[str, object]],
    diagnostics: list[dict[str, object]],
    number_of_graphs: int,
) -> None:
    if not config.run_rewired_control:
        return
    metadata = figure_metadata(config, diagnostics, number_of_graphs)
    spacing = [row for row in summaries if row["analysis_domain"] == "spacing_localization"]
    tail = [row for row in summaries if row["analysis_domain"] == "degree_spectral_tail"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].errorbar(
        [float(row["edge_gap_rank_midpoint"]) for row in spacing],
        [float(row["mean_delta_graph_mean_gap_ratio"]) for row in spacing],
        yerr=[float(row["se_delta_graph_mean_gap_ratio"]) if np.isfinite(float(row["se_delta_graph_mean_gap_ratio"])) else 0.0 for row in spacing],
        marker="o",
    )
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_xlabel("non-Perron edge gap rank")
    axes[0].set_ylabel("delta mean gap ratio")
    axes[1].errorbar(
        [float(row["edge_gap_rank_midpoint"]) for row in spacing],
        [float(row["mean_delta_mean_n_times_ipr"]) for row in spacing],
        yerr=[float(row["se_delta_mean_n_times_ipr"]) if np.isfinite(float(row["se_delta_mean_n_times_ipr"])) else 0.0 for row in spacing],
        marker="s",
    )
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xlabel("non-Perron edge gap rank")
    axes[1].set_ylabel("delta mean N * IPR")
    axes[2].errorbar(
        [float(row["tail_lower_quantile"]) for row in tail],
        [float(row["mean_delta_absolute_spectral_tau_hat"]) for row in tail],
        yerr=[float(row["se_delta_absolute_spectral_tau_hat"]) if np.isfinite(float(row["se_delta_absolute_spectral_tau_hat"])) else 0.0 for row in tail],
        marker="o",
        label="absolute spectral tau",
    )
    axes[2].errorbar(
        [float(row["tail_lower_quantile"]) for row in tail],
        [float(row["mean_delta_squared_eigenvalue_tau_hat"]) for row in tail],
        yerr=[float(row["se_delta_squared_eigenvalue_tau_hat"]) if np.isfinite(float(row["se_delta_squared_eigenvalue_tau_hat"])) else 0.0 for row in tail],
        marker="s",
        label="lambda^2 tau",
    )
    axes[2].axhline(0.0, color="black", linewidth=1)
    axes[2].set_xlabel("nested lower-tail quantile")
    axes[2].set_ylabel("delta fitted tail exponent")
    axes[2].legend(fontsize=8)
    save_figure(fig, output_dir / "pa_vs_rewired_summary.png", title="Paired exact-degree rewiring comparison", metadata=metadata)


def chung_lu_generation_command(config: PreferentialAttachmentConfig) -> str:
    return (
        "PYTHONPATH=src python3 -m "
        "small_world_mobility_edge.chung_lu_degree_spectral_tail_match "
        f"--tau {config.theoretical_degree_ccdf_tau:g} --n {config.n} "
        f"--degree {config.target_mean_degree:g} "
        f"--num-graphs {config.num_graphs} --k-per-side {config.k_per_side} "
        f"--target-tail-count {config.target_tail_observations} "
        "--no-run-fixed-degree-control"
    )


def matching_chung_lu_dir(config: PreferentialAttachmentConfig) -> Path:
    return (
        REPO_ROOT
        / "outputs"
        / "chung_lu"
        / "degree_spectral_tail_match"
        / f"n_{config.n}"
        / (
            f"tau_{slug_number(config.theoretical_degree_ccdf_tau)}_"
            f"avgdeg_{slug_number(config.target_mean_degree)}"
        )
    )


def validate_chung_lu_match(config: PreferentialAttachmentConfig) -> dict[str, object]:
    directory = matching_chung_lu_dir(config)
    manifest_path = directory / "run_manifest.json"
    status: dict[str, object] = {
        "requested": config.run_chung_lu_comparison,
        "comparison_type": "unpaired_ensemble",
        "comparison_scope": (
            "degree and sparse spectral tails only; no matched Chung-Lu spacing/localization output is loaded"
        ),
        "chung_lu_directory": str(directory),
        "generation_command": chung_lu_generation_command(config),
        "checks": {},
    }
    if not manifest_path.exists():
        status.update({"available": False, "compatible": False, "reason": "matching run_manifest.json is missing"})
        return status
    with manifest_path.open() as handle:
        manifest = json.load(handle)
    expected = {
        "tau_convention": "experimental_ccdf_exponent",
        "tau": config.theoretical_degree_ccdf_tau,
        "n": config.n,
        "target_mean_degree": config.target_mean_degree,
        "component_mode": config.component_mode,
        "num_graphs": config.num_graphs,
        "spectral_selection_method": SPECTRAL_SELECTION_METHOD,
        "zero_removal_rule": (
            "central zeros are not computed or removed; positive-threshold spectral CCDFs use denominator N-1"
        ),
    }
    checks: dict[str, object] = {}
    for key, expected_value in expected.items():
        actual = manifest.get(key)
        if isinstance(expected_value, float) and actual is not None:
            matches = bool(math.isclose(float(actual), expected_value, rel_tol=0.0, abs_tol=1e-12))
        else:
            matches = actual == expected_value
        checks[key] = {"expected": expected_value, "actual": actual, "matches": matches}
    summary_path = directory / "threshold_stability_summary.csv"
    checks["summary_file"] = {
        "expected": str(summary_path),
        "actual": str(summary_path) if summary_path.exists() else None,
        "matches": summary_path.exists(),
    }
    compatible = all(bool(value["matches"]) for value in checks.values())
    status.update(
        {
            "available": True,
            "compatible": compatible,
            "reason": "all controlled variables match" if compatible else "one or more manifest controls do not match",
            "checks": checks,
        }
    )
    return status


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_chung_lu_comparison(
    *,
    config: PreferentialAttachmentConfig,
    tail_summaries: list[dict[str, object]],
    status: dict[str, object],
    diagnostics: list[dict[str, object]],
    number_of_graphs: int,
) -> None:
    comparison_dir = (
        Path(config.output_root)
        / "comparisons"
        / (
            f"{model_slug(config.model)}_vs_chung_lu_tau_"
            f"{slug_number(config.theoretical_degree_ccdf_tau)}"
        )
    )
    comparison_dir.mkdir(parents=True, exist_ok=True)
    with (comparison_dir / "comparison_status.json").open("w") as handle:
        json.dump(status, handle, indent=2, sort_keys=True)
    metadata = figure_metadata(config, diagnostics, number_of_graphs)
    if not config.run_chung_lu_comparison or not bool(status.get("compatible")):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.axis("off")
        reason = str(status.get("reason", "comparison not requested"))
        command = str(status["generation_command"])
        ax.text(
            0.02,
            0.85,
            f"No numerical BA/Chung-Lu comparison was made.\nReason: {reason}\n\nGenerate the matched Chung-Lu run with:\n{command}",
            transform=ax.transAxes,
            va="top",
            family="monospace",
            fontsize=9,
            wrap=True,
        )
        save_figure(
            fig,
            comparison_dir / "pa_vs_chung_lu_summary.png",
            title=(
                f"{config.model} versus Chung-Lu tau="
                f"{config.theoretical_degree_ccdf_tau:g} comparison status"
            ),
            metadata=metadata,
        )
        return

    cl_rows = _read_csv(Path(status["chung_lu_directory"]) / "threshold_stability_summary.csv")
    comparison_rows: list[dict[str, object]] = []
    metrics = (
        ("realized_degree_tau_hat", "alpha_hill"),
        ("absolute_spectral_tau_hat", "beta_hill"),
        ("squared_eigenvalue_tau_hat", "gamma_hill"),
    )
    for ba_row in tail_summaries:
        if ba_row["model_variant"] != config.model:
            continue
        q0 = float(ba_row["tail_lower_quantile"])
        cl_row = next(
            (
                row
                for row in cl_rows
                if row["model_variant"] == "Chung-Lu"
                and math.isclose(float(row["tail_lower_quantile"]), q0, abs_tol=1e-12)
            ),
            None,
        )
        if cl_row is None:
            continue
        for ba_metric, cl_metric in metrics:
            ba_mean = float(ba_row[f"mean_{ba_metric}"])
            ba_se = float(ba_row[f"se_{ba_metric}"])
            cl_mean = float(cl_row[f"mean_{cl_metric}"])
            cl_se = float(cl_row[f"se_{cl_metric}"])
            difference = ba_mean - cl_mean
            difference_se = (
                math.sqrt(ba_se**2 + cl_se**2)
                if np.isfinite([ba_se, cl_se]).all()
                else math.nan
            )
            comparison_rows.append(
                {
                    "tail_lower_quantile": q0,
                    "metric": ba_metric,
                    "ba_mean": ba_mean,
                    "ba_standard_error": ba_se,
                    "chung_lu_mean": cl_mean,
                    "chung_lu_standard_error": cl_se,
                    "ba_minus_chung_lu": difference,
                    "unpaired_difference_standard_error": difference_se,
                    "comparison_type": "unpaired_ensemble",
                    "comparison_scope": "degree and sparse spectral tails",
                }
            )
    write_csv(comparison_dir / "unpaired_tail_comparison.csv", comparison_rows)
    primary = [row for row in comparison_rows if math.isclose(float(row["tail_lower_quantile"]), PRIMARY_TAIL_QUANTILE)]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(primary))
    labels = [str(row["metric"]).replace("_tau_hat", "").replace("_", " ") for row in primary]
    width = 0.35
    ax.bar(
        x - width / 2,
        [float(row["ba_mean"]) for row in primary],
        width,
        yerr=[float(row["ba_standard_error"]) for row in primary],
        capsize=3,
        label=config.model,
    )
    ax.bar(
        x + width / 2,
        [float(row["chung_lu_mean"]) for row in primary],
        width,
        yerr=[float(row["chung_lu_standard_error"]) for row in primary],
        capsize=3,
        label=f"Chung-Lu tau={config.theoretical_degree_ccdf_tau:g}",
    )
    ax.set_xticks(x, labels, rotation=15)
    ax.set_ylabel("graph-equal mean fitted CCDF exponent")
    ax.legend()
    save_figure(
        fig,
        comparison_dir / "pa_vs_chung_lu_summary.png",
        title="Unpaired matched-model tail comparison",
        metadata=metadata,
    )


def make_run_manifest(
    *,
    config: PreferentialAttachmentConfig,
    partials: list[dict[str, object]],
    diagnostics: list[dict[str, object]],
    tail_summaries: list[dict[str, object]],
    spacing_summaries: list[dict[str, object]],
    degree_checks: list[dict[str, object]],
    chung_lu_status: dict[str, object],
) -> dict[str, object]:
    original_diagnostics = [row for row in diagnostics if row["model_variant"] == config.model]
    primary_tail = next(
        row
        for row in tail_summaries
        if row["model_variant"] == config.model and bool(row["is_primary_threshold"])
    )
    overlaps = [float(row["edge_overlap_fraction"]) for row in degree_checks]
    graph_seeds, rewire_seeds = seed_lists(config)
    initial_convention = (
        INITIAL_GRAPH_CONVENTION
        if config.model in {MODEL_BA, MODEL_AFFINE_PA}
        else HOLME_KIM_INITIAL_GRAPH_CONVENTION
    )
    return {
        "model": config.model,
        "model_generator": (
            "networkx.barabasi_albert_graph"
            if config.model == MODEL_BA
            else (
                "custom_affine_linear_weighted_without_replacement"
                if config.model == MODEL_AFFINE_PA
                else "networkx.powerlaw_cluster_graph"
            )
        ),
        "initial_graph_convention": initial_convention,
        "n": config.n,
        "m": config.m,
        "triad_probability": config.triad_probability,
        "theoretical_degree_ccdf_tau": config.theoretical_degree_ccdf_tau,
        "attachment_offset_delta": (
            attachment_offset_for_ccdf_tau(config.m, config.theoretical_degree_ccdf_tau)
            if config.model == MODEL_AFFINE_PA
            else 0.0
        ),
        "attachment_kernel": (
            "P(new vertex t attaches to existing i | G_t) proportional to degree_i(t) + delta"
            if config.model == MODEL_AFFINE_PA
            else None
        ),
        "tau_convention": "P(D >= k) approximately k^(-tau)",
        "estimated_realized_degree_tau_mean": primary_tail["mean_realized_degree_tau_hat"],
        "estimated_realized_degree_tau_standard_error": primary_tail["se_realized_degree_tau_hat"],
        "target_mean_degree": config.target_mean_degree,
        "realized_mean_degree_mean": float(np.mean([float(row["realized_mean_degree"]) for row in original_diagnostics])),
        "realized_mean_degree_min": float(np.min([float(row["realized_mean_degree"]) for row in original_diagnostics])),
        "realized_mean_degree_max": float(np.max([float(row["realized_mean_degree"]) for row in original_diagnostics])),
        "number_of_graphs": config.num_graphs,
        "graphs_generated": len(partials),
        "valid_graphs": len({int(row["graph_index"]) for row in original_diagnostics}),
        "graph_seed_base": config.graph_seed_base,
        "rewire_seed_base": config.rewire_seed_base,
        "graph_seeds": graph_seeds,
        "rewire_seeds": rewire_seeds,
        "component_mode": config.component_mode,
        "perron_rule": "largest algebraic eigenvalue saved separately and excluded from primary tails and spacing windows",
        "zero_removal_rule": (
            "exact multiplicity recorded before removal in dense smoke; sparse production edge selection does not compute central zeros"
        ),
        "remove_exact_zero_eigenvalues": REMOVE_EXACT_ZERO_EIGENVALUES,
        "zero_eigenvalue_atol": ZERO_EIGENVALUE_ATOL,
        "tail_zero_removal_rule": (
            "central zeros are not computed or removed; positive-threshold spectral CCDFs use denominator N-1"
        ),
        "spectral_solver": "scipy.sparse.linalg.eigsh",
        "spectral_solver_sides": ["LA", "SA"],
        "spectral_selection_method": SPECTRAL_SELECTION_METHOD,
        "eigsh_tolerance": config.eigsh_tol,
        "eigsh_maxiter": config.eigsh_maxiter,
        "eigsh_ncv_factor": config.eigsh_ncv_factor,
        "k_per_side": config.k_per_side,
        "tail_lower_quantiles": config.tail_lower_quantiles,
        "primary_tail_lower_quantile": float(primary_tail["tail_lower_quantile"]),
        "target_tail_observations": config.target_tail_observations,
        "rank_windows": config.rank_windows,
        "rank_window_rule": SPACING_WINDOW_MODE,
        "spacing_normalization": SPACING_NORMALIZATION,
        "minimum_consecutive_gaps_per_graph": 5,
        "target_minimum_pooled_gaps_per_primary_window": 100,
        "pooled_gap_requirement_enforced": config.n > 400 and config.num_graphs >= 5,
        "observed_minimum_pooled_gaps_per_window": min(
            int(row["number_of_gaps"]) for row in spacing_summaries
        ),
        "spacing_sample_counts": [
            {
                "model_variant": row["model_variant"],
                "edge_gap_rank_start": row["edge_gap_rank_start"],
                "edge_gap_rank_stop": row["edge_gap_rank_stop"],
                "graphs": row["number_of_graphs"],
                "eigenvalues": row["number_of_eigenvalues"],
                "gaps": row["number_of_gaps"],
                "gap_ratios": row["number_of_gap_ratios"],
                "localization_eigenvectors": row["number_of_eigenvalues"],
            }
            for row in spacing_summaries
        ],
        "run_rewired_control": config.run_rewired_control,
        "rewiring_implementation": "networkx double_edge_swap via rewire_graph_preserving_degrees",
        "rewiring_sampling_caveat": (
            "finite double-edge-swap chain is an approximate randomized simple graph conditional on the exact labeled degree sequence; uniformity is not claimed"
        ),
        "swaps_per_edge": config.swaps_per_edge,
        "max_tries_factor": config.max_tries_factor,
        "all_labeled_degrees_preserved": bool(
            degree_checks and all(bool(row["degree_match"]) for row in degree_checks)
        ) if config.run_rewired_control else None,
        "edge_overlap_mean": float(np.mean(overlaps)) if overlaps else None,
        "edge_overlap_min": float(np.min(overlaps)) if overlaps else None,
        "edge_overlap_max": float(np.max(overlaps)) if overlaps else None,
        "validate_dense": config.validate_dense,
        "dense_full_spectrum": config.dense_full_spectrum,
        "global_density_warning": (
            None
            if config.dense_full_spectrum
            else "global_spectral_density.png contains retained sparse edge eigenvalues only and is explicitly labeled as non-global"
        ),
        "elapsed_seconds_sum_over_graph_partials": float(
            sum(float(partial["metadata"]["elapsed_seconds"]) for partial in partials)
        ),
        "peak_process_rss_mb_max": float(
            max(float(partial["metadata"]["peak_rss_mb"]) for partial in partials)
        ),
        "chung_lu_comparison": chung_lu_status,
        "git_commit": code_version(),
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def aggregate_partials(
    config: PreferentialAttachmentConfig,
    partial_dir: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Aggregate graph-level partials with graph-equal and paired uncertainty."""
    validate_config(config)
    source = Path(partial_dir or (model_output_dir(config) / "partials"))
    paths = [source / f"graph_{index:06d}.npz" for index in range(config.num_graphs)]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"missing {len(missing)} graph partials; first missing file is {missing[0]}"
        )
    partials = [load_graph_partial(path) for path in paths]
    diagnostics = [row for partial in partials for row in partial["diagnostics"]]
    threshold_rows = [row for partial in partials for row in partial["threshold_rows"]]
    rank_rows = [row for partial in partials for row in partial["rank_rows"]]
    spacing_rows = [row for partial in partials for row in partial["spacing_rows"]]
    decay_rows = [row for partial in partials for row in partial["decay_rows"]]
    degree_checks = [
        partial["degree_check"]
        for partial in partials
        if partial["degree_check"] is not None
    ]
    tail_summaries = aggregate_tail_rows(threshold_rows)
    rank_summaries = aggregate_rank_rows(rank_rows)
    spacing_summaries = aggregate_spacing_rows(spacing_rows)
    decay_summaries = aggregate_decay_rows(decay_rows)
    paired_rows = paired_difference_rows(threshold_rows, spacing_rows, config)
    paired_summaries = aggregate_paired_differences(paired_rows)
    ccdf_rows, rank_curve_rows = pooled_curve_rows(config, partials)

    if config.n > 400 and config.num_graphs >= 5:
        insufficient = [
            row
            for row in spacing_summaries
            if int(row["number_of_gaps"]) < 100
        ]
        if insufficient:
            first = insufficient[0]
            raise ValueError(
                "primary rank window has fewer than 100 pooled gaps: "
                f"{first['model_variant']} ranks {first['edge_gap_rank_start']}-{first['edge_gap_rank_stop']}"
            )

    destination = Path(output_dir or model_output_dir(config))
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / "graph_diagnostics_by_graph.csv", diagnostics)
    write_csv(destination / "degree_spectral_tail_by_graph.csv", threshold_rows)
    write_csv(destination / "degree_spectral_tail_summary.csv", tail_summaries)
    write_csv(destination / "rank_match_by_graph.csv", rank_rows)
    write_csv(destination / "rank_match_summary.csv", rank_summaries)
    write_csv(destination / "spacing_localization_by_graph.csv", spacing_rows)
    write_csv(destination / "spacing_localization_summary.csv", spacing_summaries)
    write_csv(destination / "coordinate_decay_summary.csv", decay_summaries)
    write_csv(destination / "pooled_ccdf_curves.csv", ccdf_rows)
    write_csv(destination / "rank_match_curves.csv", rank_curve_rows)
    if degree_checks:
        write_csv(destination / "degree_preservation_checks.csv", degree_checks)
        write_csv(destination / "paired_differences_by_graph.csv", paired_rows)
        write_csv(destination / "paired_differences_summary.csv", paired_summaries)

    plot_ccdf_figures(
        config=config,
        output_dir=destination,
        ccdf_rows=ccdf_rows,
        diagnostics=diagnostics,
        number_of_graphs=len(partials),
    )
    plot_rank_matches(
        config=config,
        output_dir=destination,
        rows=rank_curve_rows,
        diagnostics=diagnostics,
        number_of_graphs=len(partials),
    )
    plot_global_density(
        config=config,
        output_dir=destination,
        partials=partials,
        diagnostics=diagnostics,
        number_of_graphs=len(partials),
    )
    plot_spacing_diagnostics(
        config=config,
        output_dir=destination,
        summaries=spacing_summaries,
        diagnostics=diagnostics,
        number_of_graphs=len(partials),
    )
    plot_localization_diagnostics(
        config=config,
        output_dir=destination,
        summaries=spacing_summaries,
        decay_summaries=decay_summaries,
        diagnostics=diagnostics,
        number_of_graphs=len(partials),
    )
    if degree_checks:
        plot_paired_summary(
            config=config,
            output_dir=destination,
            summaries=paired_summaries,
            diagnostics=diagnostics,
            number_of_graphs=len(partials),
        )
        paired_dir = Path(config.output_root) / "comparisons" / "ba_vs_rewired"
        write_csv(paired_dir / "degree_preservation_checks.csv", degree_checks)
        write_csv(paired_dir / "paired_differences_by_graph.csv", paired_rows)
        write_csv(paired_dir / "paired_differences_summary.csv", paired_summaries)
        plot_paired_summary(
            config=config,
            output_dir=paired_dir,
            summaries=paired_summaries,
            diagnostics=diagnostics,
            number_of_graphs=len(partials),
        )

    chung_lu_status = validate_chung_lu_match(config)
    write_chung_lu_comparison(
        config=config,
        tail_summaries=tail_summaries,
        status=chung_lu_status,
        diagnostics=diagnostics,
        number_of_graphs=len(partials),
    )
    manifest = make_run_manifest(
        config=config,
        partials=partials,
        diagnostics=diagnostics,
        tail_summaries=tail_summaries,
        spacing_summaries=spacing_summaries,
        degree_checks=degree_checks,
        chung_lu_status=chung_lu_status,
    )
    with (destination / "run_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(f"saved preferential-attachment experiment to {destination}", flush=True)
    return destination


def run_experiment(config: PreferentialAttachmentConfig) -> Path:
    validate_config(config)
    partial_dir = model_output_dir(config) / "partials"
    for graph_index in range(config.num_graphs):
        run_graph_partial(config, graph_index, partial_dir=partial_dir)
    return aggregate_partials(config, partial_dir=partial_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--m", type=int, default=DEFAULT_M)
    parser.add_argument("--num-graphs", type=int, default=DEFAULT_NUM_GRAPHS)
    parser.add_argument("--graph-index", type=int)
    parser.add_argument(
        "--model",
        default="barabasi-albert",
        choices=["barabasi-albert", "ba", "affine", "affine-pa", "holme-kim"],
    )
    parser.add_argument("--triad-probability", type=float, default=0.0)
    parser.add_argument(
        "--attachment-ccdf-tau",
        type=float,
        default=THEORETICAL_DEGREE_CCDF_TAU,
        help="Affine-PA degree CCDF exponent; tau > 1 and delta=m*(tau-2).",
    )
    parser.add_argument("--graph-seed-base", type=int, default=DEFAULT_GRAPH_SEED_BASE)
    parser.add_argument("--rewire-seed-base", type=int, default=DEFAULT_REWIRE_SEED_BASE)
    parser.add_argument(
        "--component-mode",
        choices=["full", "largest_connected_component"],
        default=DEFAULT_COMPONENT_MODE,
    )
    parser.add_argument("--swaps-per-edge", type=int, default=DEFAULT_SWAPS_PER_EDGE)
    parser.add_argument("--max-tries-factor", type=int, default=DEFAULT_MAX_TRIES_FACTOR)
    parser.add_argument("--partial-dir", type=Path)
    parser.add_argument("--aggregate-partials", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--run-rewired-control",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--run-chung-lu-comparison",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--k-per-side", type=int, default=DEFAULT_K_PER_SIDE)
    parser.add_argument(
        "--target-tail-observations",
        type=int,
        default=DEFAULT_TARGET_TAIL_OBSERVATIONS,
    )
    parser.add_argument("--rank-window-width", type=int, default=DEFAULT_RANK_WINDOW_WIDTH)
    parser.add_argument("--num-rank-windows", type=int, default=DEFAULT_NUM_RANK_WINDOWS)
    parser.add_argument("--eigsh-tol", type=float, default=DEFAULT_EIGSH_TOL)
    parser.add_argument("--eigsh-maxiter", type=int, default=DEFAULT_EIGSH_MAXITER)
    parser.add_argument("--eigsh-ncv-factor", type=int, default=DEFAULT_EIGSH_NCV_FACTOR)
    parser.add_argument("--validate-dense", action="store_true")
    parser.add_argument("--dense-full-spectrum", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> PreferentialAttachmentConfig:
    model = canonical_model_name(args.model)
    config = PreferentialAttachmentConfig(
        n=args.n,
        m=args.m,
        num_graphs=args.num_graphs,
        model=model,
        triad_probability=args.triad_probability,
        theoretical_degree_ccdf_tau=args.attachment_ccdf_tau,
        target_mean_degree=2.0 * args.m,
        component_mode=args.component_mode,
        graph_seed_base=args.graph_seed_base,
        rewire_seed_base=args.rewire_seed_base,
        swaps_per_edge=args.swaps_per_edge,
        max_tries_factor=args.max_tries_factor,
        run_rewired_control=args.run_rewired_control,
        run_chung_lu_comparison=args.run_chung_lu_comparison,
        k_per_side=args.k_per_side,
        target_tail_observations=args.target_tail_observations,
        rank_window_width=args.rank_window_width,
        num_rank_windows=args.num_rank_windows,
        eigsh_tol=args.eigsh_tol,
        eigsh_maxiter=args.eigsh_maxiter,
        eigsh_ncv_factor=args.eigsh_ncv_factor,
        validate_dense=args.validate_dense,
        dense_full_spectrum=args.dense_full_spectrum,
        output_root=args.output_dir,
    )
    if args.smoke_test:
        config = replace(
            config,
            n=400,
            m=4,
            num_graphs=3,
            target_mean_degree=8.0,
            k_per_side=40,
            target_tail_observations=20,
            rank_window_width=5,
            num_rank_windows=4,
            validate_dense=True,
            dense_full_spectrum=True,
            swaps_per_edge=min(config.swaps_per_edge, 2),
        )
    return config


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    if args.aggregate_partials:
        aggregate_partials(config, partial_dir=args.partial_dir)
        return
    if args.graph_index is not None:
        path = run_graph_partial(config, args.graph_index, partial_dir=args.partial_dir)
        print(path)
        return
    run_experiment(config)


if __name__ == "__main__":
    main()
