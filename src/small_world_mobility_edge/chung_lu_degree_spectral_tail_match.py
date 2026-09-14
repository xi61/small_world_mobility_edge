from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Iterable
import warnings

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "outputs" / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / "outputs" / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import ArpackNoConvergence, eigsh
from scipy.stats import spearmanr

from .chung_lu_spacing_by_energy import (
    CUTOFF_COEFFICIENT,
    CUTOFF_MODE,
    N,
    SEQUENCE_MODE,
    SPACING_GRAPH_SEED,
    SPACING_SEQUENCE_SEED,
    W_MAX,
    W_MIN,
    make_seeds,
    parameter_seed_offset,
)
from .fixed_degree_configuration_baseline import (
    MODEL_CHUNG_LU,
    MODEL_CONFIGURATION,
    TAU_CONVENTION_CCDF,
    generate_expected_weights_for_tau_convention,
    rewire_graph_preserving_degrees,
)
from .regular_baseline import adjacency_eigendecomposition


TAU_CONVENTION = TAU_CONVENTION_CCDF
DEFAULT_TAU_VALUES = (0.8, 1.05, 1.5, 2.0, 2.5)
DEFAULT_TARGET_MEAN_DEGREE = 20.0
DEFAULT_NUM_GRAPHS = 50
DEFAULT_COMPONENT_MODE = "full"
DEFAULT_K_PER_SIDE = 180
DEFAULT_TARGET_TAIL_COUNT = 120
DEFAULT_TAIL_LOWER_QUANTILES = (0.95, 0.96, 0.97, 0.976, 0.98)
PRIMARY_TAIL_QUANTILE = 0.976
DEFAULT_EIGSH_TOL = 1e-8
DEFAULT_EIGSH_MAXITER = 20000
DEFAULT_EIGSH_NCV_FACTOR = 3
DEFAULT_REWIRE_SEED_BASE = 17320
DEFAULT_SWAPS_PER_EDGE = 10
DEFAULT_MAX_TRIES_FACTOR = 20
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "chung_lu" / "degree_spectral_tail_match"
MIN_TAIL_OBSERVATIONS_FOR_FIT = 20


@dataclass(frozen=True)
class TailMatchConfig:
    n: int = N
    tau_values: tuple[float, ...] = DEFAULT_TAU_VALUES
    target_mean_degree: float = DEFAULT_TARGET_MEAN_DEGREE
    num_graphs: int = DEFAULT_NUM_GRAPHS
    sequence_mode: str = SEQUENCE_MODE
    w_min: float = W_MIN
    w_max: float | None = W_MAX
    cutoff_mode: str = CUTOFF_MODE
    cutoff_coefficient: float = CUTOFF_COEFFICIENT
    component_mode: str = DEFAULT_COMPONENT_MODE
    tau_convention: str = TAU_CONVENTION
    graph_seed_base: int = SPACING_GRAPH_SEED
    sequence_seed_base: int = SPACING_SEQUENCE_SEED
    rewire_seed_base: int = DEFAULT_REWIRE_SEED_BASE
    k_per_side: int = DEFAULT_K_PER_SIDE
    eigsh_tol: float = DEFAULT_EIGSH_TOL
    eigsh_maxiter: int = DEFAULT_EIGSH_MAXITER
    eigsh_ncv_factor: int = DEFAULT_EIGSH_NCV_FACTOR
    target_tail_count: int = DEFAULT_TARGET_TAIL_COUNT
    tail_lower_quantiles: tuple[float, ...] = DEFAULT_TAIL_LOWER_QUANTILES
    run_fixed_degree_control: bool = True
    swaps_per_edge: int = DEFAULT_SWAPS_PER_EDGE
    max_tries_factor: int = DEFAULT_MAX_TRIES_FACTOR
    validate_dense: bool = False
    output_dir: Path = OUTPUT_DIR


def code_version() -> str | None:
    repo = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def slug_number(value: float) -> str:
    return str(value).replace(".", "p")


def tau_output_dir(config: TailMatchConfig, tau: float) -> Path:
    return (
        Path(config.output_dir)
        / f"n_{config.n}"
        / f"tau_{slug_number(tau)}_avgdeg_{slug_number(config.target_mean_degree)}"
    )


def graph_seeds_for(config: TailMatchConfig, tau: float) -> tuple[list[int], list[int], list[int]]:
    offset = parameter_seed_offset(tau, config.target_mean_degree)
    return (
        make_seeds(config.num_graphs, config.graph_seed_base + offset),
        make_seeds(config.num_graphs, config.sequence_seed_base + offset),
        make_seeds(config.num_graphs, config.rewire_seed_base + offset),
    )


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


def sparse_adjacency(graph: nx.Graph) -> sparse.csr_array:
    nodes = sorted(graph.nodes())
    matrix = nx.to_scipy_sparse_array(graph, nodelist=nodes, dtype=float, format="csr")
    return matrix


@dataclass(frozen=True)
class SparseTailSpectrum:
    model_variant: str
    eigenvalues_la: np.ndarray
    eigenvalues_sa: np.ndarray
    eigenvectors_la: np.ndarray
    eigenvectors_sa: np.ndarray
    residuals_la: np.ndarray
    residuals_sa: np.ndarray
    perron_eigenvalue: float
    positive_nonperron: np.ndarray
    negative_eigenvalues: np.ndarray
    absolute_nonperron_tail: np.ndarray
    lambda_squared_tail: np.ndarray
    eigsh_seconds: float
    dense_max_abs_error: float
    retained_absolute_count: int


def _eigsh_side(
    matrix: sparse.csr_array,
    *,
    k: int,
    which: str,
    tol: float,
    maxiter: int,
    ncv: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        values, vectors = eigsh(matrix, k=k, which=which, tol=tol, maxiter=maxiter, ncv=ncv)
    except ArpackNoConvergence as exc:
        converged_values = np.asarray(exc.eigenvalues if exc.eigenvalues is not None else [], dtype=float)
        if len(converged_values) < max(2, k // 2):
            raise
        warnings.warn(
            f"eigsh({which}) returned only {len(converged_values)} converged eigenpairs out of k={k}",
            RuntimeWarning,
            stacklevel=2,
        )
        values = converged_values
        vectors = np.asarray(exc.eigenvectors, dtype=float)
    order = np.argsort(values)
    return np.asarray(values[order], dtype=float), np.asarray(vectors[:, order], dtype=float)


def _residuals(matrix: sparse.csr_array, values: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return np.array([], dtype=float)
    residuals = []
    for j, value in enumerate(values):
        residuals.append(float(np.linalg.norm(matrix @ vectors[:, j] - value * vectors[:, j])))
    return np.asarray(residuals, dtype=float)


def sparse_tail_spectrum(
    graph: nx.Graph,
    *,
    model_variant: str,
    k_per_side: int,
    eigsh_tol: float,
    eigsh_maxiter: int,
    eigsh_ncv_factor: int,
    validate_dense: bool = False,
) -> SparseTailSpectrum:
    analyzed_n = graph.number_of_nodes()
    if analyzed_n < 4:
        raise ValueError("sparse tail spectrum needs at least four vertices")
    k = min(int(k_per_side), analyzed_n - 2)
    if k < 2:
        raise ValueError("k_per_side leaves fewer than two sparse eigenvalues")
    ncv = min(analyzed_n, max(2 * k + 1, eigsh_ncv_factor * k))
    if ncv <= k + 1:
        ncv = None

    matrix = sparse_adjacency(graph)
    start = time.perf_counter()
    la_values, la_vectors = _eigsh_side(
        matrix,
        k=k,
        which="LA",
        tol=eigsh_tol,
        maxiter=eigsh_maxiter,
        ncv=ncv,
    )
    sa_values, sa_vectors = _eigsh_side(
        matrix,
        k=k,
        which="SA",
        tol=eigsh_tol,
        maxiter=eigsh_maxiter,
        ncv=ncv,
    )
    eigsh_seconds = time.perf_counter() - start
    residuals_la = _residuals(matrix, la_values, la_vectors)
    residuals_sa = _residuals(matrix, sa_values, sa_vectors)

    la_desc = la_values[::-1]
    perron = float(la_desc[0])
    positive_nonperron = la_desc[1:]
    negative_eigenvalues = np.sort(sa_values)
    absolute_values = np.concatenate((np.abs(positive_nonperron), np.abs(negative_eigenvalues)))
    absolute_values = np.sort(absolute_values)[::-1]
    lambda_squared = absolute_values**2

    dense_error = math.nan
    if validate_dense:
        dense_values, _ = adjacency_eigendecomposition(graph)
        dense_values = np.asarray(dense_values, dtype=float)
        dense_la = dense_values[-len(la_values) :]
        dense_sa = dense_values[: len(sa_values)]
        dense_error = float(
            max(
                np.max(np.abs(np.sort(la_values) - np.sort(dense_la))),
                np.max(np.abs(np.sort(sa_values) - np.sort(dense_sa))),
            )
        )

    return SparseTailSpectrum(
        model_variant=model_variant,
        eigenvalues_la=la_values,
        eigenvalues_sa=sa_values,
        eigenvectors_la=la_vectors,
        eigenvectors_sa=sa_vectors,
        residuals_la=residuals_la,
        residuals_sa=residuals_sa,
        perron_eigenvalue=perron,
        positive_nonperron=positive_nonperron,
        negative_eigenvalues=negative_eigenvalues,
        absolute_nonperron_tail=absolute_values,
        lambda_squared_tail=lambda_squared,
        eigsh_seconds=float(eigsh_seconds),
        dense_max_abs_error=dense_error,
        retained_absolute_count=int(len(absolute_values)),
    )


def ccdf_points_from_values(values: np.ndarray, denominator: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = values[values > 0]
    if len(values) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    sorted_values = np.sort(values)[::-1]
    ranks = np.arange(1, len(sorted_values) + 1, dtype=float)
    return sorted_values, ranks / float(denominator)


def top_count_for_quantile(denominator: int, q0: float) -> int:
    return max(1, int(math.ceil((1.0 - q0) * denominator)))


def primary_tail_quantile_for_config(config: TailMatchConfig) -> float:
    configured = [float(q0) for q0 in config.tail_lower_quantiles]
    if (
        PRIMARY_TAIL_QUANTILE in configured
        and top_count_for_quantile(config.n - 1, PRIMARY_TAIL_QUANTILE) >= MIN_TAIL_OBSERVATIONS_FOR_FIT
    ):
        return PRIMARY_TAIL_QUANTILE
    for q0 in configured:
        if top_count_for_quantile(config.n - 1, q0) >= MIN_TAIL_OBSERVATIONS_FOR_FIT:
            return q0
    return configured[0]


def hill_ccdf_exponent_from_top(values_desc: np.ndarray, top_count: int) -> tuple[float, float, int]:
    values = np.asarray(values_desc, dtype=float)
    values = values[np.isfinite(values)]
    values = values[values > 0]
    if len(values) < top_count or top_count < MIN_TAIL_OBSERVATIONS_FOR_FIT:
        return math.nan, math.nan, int(min(len(values), max(top_count, 0)))
    tail = values[:top_count]
    xmin = float(tail[-1])
    logs = np.log(np.maximum(tail / xmin, 1.0))
    mean_log = float(np.mean(logs))
    if mean_log <= 0.0 or not np.isfinite(mean_log):
        return math.nan, xmin, int(len(tail))
    return float(1.0 / mean_log), xmin, int(len(tail))


def log_ccdf_regression(values_desc: np.ndarray, denominator: int, top_count: int) -> dict[str, float]:
    values = np.asarray(values_desc, dtype=float)
    values = values[np.isfinite(values)]
    values = values[values > 0]
    if len(values) < top_count or top_count < MIN_TAIL_OBSERVATIONS_FOR_FIT:
        return {"ccdf_regression_exponent": math.nan, "ccdf_regression_r2": math.nan}
    x = values[:top_count]
    y = np.arange(1, top_count + 1, dtype=float) / float(denominator)
    log_x = np.log(x)
    log_y = np.log(y)
    if np.std(log_x) <= 0.0:
        return {"ccdf_regression_exponent": math.nan, "ccdf_regression_r2": math.nan}
    slope, intercept = np.polyfit(log_x, log_y, 1)
    fitted = intercept + slope * log_x
    ss_res = float(np.sum((log_y - fitted) ** 2))
    ss_tot = float(np.sum((log_y - np.mean(log_y)) ** 2))
    return {
        "ccdf_regression_exponent": float(-slope),
        "ccdf_regression_r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else math.nan,
    }


def rank_match_fit(
    degree_desc: np.ndarray,
    lambda_desc: np.ndarray,
    *,
    max_rank: int,
    minimum_observations: int = MIN_TAIL_OBSERVATIONS_FOR_FIT,
) -> dict[str, float]:
    if minimum_observations < 3:
        raise ValueError("minimum_observations must be at least 3 for a fitted slope")
    count = min(int(max_rank), len(degree_desc), len(lambda_desc))
    if count < minimum_observations:
        return {
            "usable_ranks": count,
            "slope": math.nan,
            "slope_se": math.nan,
            "intercept": math.nan,
            "r2": math.nan,
            "median_lambda_over_sqrt_degree": math.nan,
            "iqr_lambda_over_sqrt_degree": math.nan,
            "spearman_rank_correlation": math.nan,
        }
    d = np.asarray(degree_desc[:count], dtype=float)
    lam = np.asarray(lambda_desc[:count], dtype=float)
    valid = np.isfinite(d) & np.isfinite(lam) & (d > 0) & (lam > 0)
    d = d[valid]
    lam = lam[valid]
    if len(d) < minimum_observations or np.std(np.log(d)) <= 0.0:
        return {
            "usable_ranks": int(len(d)),
            "slope": math.nan,
            "slope_se": math.nan,
            "intercept": math.nan,
            "r2": math.nan,
            "median_lambda_over_sqrt_degree": math.nan,
            "iqr_lambda_over_sqrt_degree": math.nan,
            "spearman_rank_correlation": math.nan,
        }
    x = np.log(d)
    y = np.log(lam)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    residuals = y - fitted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    sxx = float(np.sum((x - np.mean(x)) ** 2))
    sigma2 = ss_res / max(len(x) - 2, 1)
    slope_se = math.sqrt(sigma2 / sxx) if sxx > 0 else math.nan
    ratios = lam / np.sqrt(d)
    spearman_result = spearmanr(d, lam)
    # SciPy < 1.11 exposes ``correlation`` while newer releases also expose
    # the generic hypothesis-test name ``statistic``.
    rho = getattr(spearman_result, "statistic", spearman_result.correlation)
    return {
        "usable_ranks": int(len(d)),
        "slope": float(slope),
        "slope_se": float(slope_se),
        "intercept": float(intercept),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else math.nan,
        "median_lambda_over_sqrt_degree": float(np.median(ratios)),
        "iqr_lambda_over_sqrt_degree": float(np.quantile(ratios, 0.75) - np.quantile(ratios, 0.25)),
        "spearman_rank_correlation": float(rho) if np.isfinite(rho) else math.nan,
    }


def transformed_prefactor(
    degree_desc: np.ndarray,
    lambda_abs_desc: np.ndarray,
    *,
    denominator_degree: int,
    denominator_lambda: int,
    top_count: int,
) -> float:
    if len(lambda_abs_desc) < top_count or top_count < MIN_TAIL_OBSERVATIONS_FOR_FIT:
        return math.nan
    lam = np.asarray(lambda_abs_desc[:top_count], dtype=float)
    degree_sorted = np.asarray(degree_desc, dtype=float)
    degree_sorted = degree_sorted[np.isfinite(degree_sorted)]
    logs = []
    for rank, value in enumerate(lam, start=1):
        degree_threshold = value**2
        degree_tail_count = int(np.sum(degree_sorted >= degree_threshold))
        if degree_tail_count <= 0:
            continue
        t_lambda = rank / float(denominator_lambda)
        t_degree = degree_tail_count / float(denominator_degree)
        logs.append(math.log(t_lambda) - math.log(t_degree))
    if len(logs) < MIN_TAIL_OBSERVATIONS_FOR_FIT:
        return math.nan
    return float(math.exp(np.mean(logs)))


def threshold_rows_for_model(
    *,
    graph_index: int,
    model_variant: str,
    tau: float,
    config: TailMatchConfig,
    degree_desc: np.ndarray,
    absolute_desc: np.ndarray,
    lambda_squared_desc: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for q0 in config.tail_lower_quantiles:
        degree_count = top_count_for_quantile(config.n, q0)
        spectral_count = top_count_for_quantile(config.n - 1, q0)
        alpha, degree_xmin, n_degree = hill_ccdf_exponent_from_top(degree_desc, degree_count)
        beta, spectral_xmin, n_spectral = hill_ccdf_exponent_from_top(absolute_desc, spectral_count)
        gamma, y_xmin, n_y = hill_ccdf_exponent_from_top(lambda_squared_desc, spectral_count)
        degree_reg = log_ccdf_regression(degree_desc, config.n, degree_count)
        spectral_reg = log_ccdf_regression(absolute_desc, config.n - 1, spectral_count)
        y_reg = log_ccdf_regression(lambda_squared_desc, config.n - 1, spectral_count)
        rows.append(
            {
                "graph_index": graph_index,
                "model_variant": model_variant,
                "tau": tau,
                "tau_convention": config.tau_convention,
                "target_mean_degree": config.target_mean_degree,
                "tail_lower_quantile": q0,
                "degree_tail_count": n_degree,
                "spectral_tail_count": n_spectral,
                "lambda_squared_tail_count": n_y,
                "degree_xmin": degree_xmin,
                "spectral_xmin": spectral_xmin,
                "lambda_squared_xmin": y_xmin,
                "alpha_hill": alpha,
                "beta_hill": beta,
                "gamma_hill": gamma,
                "beta_minus_2alpha_hill": beta - 2.0 * alpha if np.isfinite([alpha, beta]).all() else math.nan,
                "beta_over_2alpha_hill": beta / (2.0 * alpha) if alpha > 0 and np.isfinite(beta) else math.nan,
                "gamma_minus_alpha_hill": gamma - alpha if np.isfinite([alpha, gamma]).all() else math.nan,
                "alpha_regression": degree_reg["ccdf_regression_exponent"],
                "alpha_regression_r2": degree_reg["ccdf_regression_r2"],
                "beta_regression": spectral_reg["ccdf_regression_exponent"],
                "beta_regression_r2": spectral_reg["ccdf_regression_r2"],
                "gamma_regression": y_reg["ccdf_regression_exponent"],
                "gamma_regression_r2": y_reg["ccdf_regression_r2"],
            }
        )
    return rows


def graph_tail_objects(
    *,
    graph_index: int,
    graph_seed: int,
    sequence_seed: int,
    rewire_seed: int,
    tau: float,
    config: TailMatchConfig,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, np.ndarray],
]:
    weights, diagnostics = generate_expected_weights_for_tau_convention(
        n=config.n,
        tau=tau,
        target_mean_degree=config.target_mean_degree,
        w_min=config.w_min,
        w_max=config.w_max,
        cutoff_mode=config.cutoff_mode,
        cutoff_coefficient=config.cutoff_coefficient,
        sequence_mode=config.sequence_mode,
        sequence_seed=sequence_seed,
        tau_convention=config.tau_convention,
    )
    if not diagnostics.admissibility_holds:
        raise ValueError("expected-weight sequence violates Chung-Lu admissibility")

    graph = nx.expected_degree_graph(weights, seed=int(graph_seed), selfloops=False)
    graph = nx.convert_node_labels_to_integers(graph, ordering="sorted")
    variants: list[tuple[str, nx.Graph, float, int, int]] = [(MODEL_CHUNG_LU, graph, math.nan, 0, 0)]
    if config.run_fixed_degree_control:
        rewired, requested_swaps, max_tries, overlap = rewire_graph_preserving_degrees(
            graph,
            rewire_seed=int(rewire_seed),
            swaps_per_edge=config.swaps_per_edge,
            max_tries_factor=config.max_tries_factor,
        )
        if not np.array_equal(degree_array(graph), degree_array(rewired)):
            raise AssertionError("configuration control did not preserve the labeled degree sequence")
        variants.append((MODEL_CONFIGURATION, rewired, overlap, requested_swaps, max_tries))

    realized_degrees = degree_array(graph)
    degree_desc = np.sort(realized_degrees[realized_degrees > 0])[::-1]

    tail_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    rank_rows: list[dict[str, object]] = []
    cutoff_rows: list[dict[str, object]] = []
    arrays: dict[str, np.ndarray] = {
        "expected_weights": np.asarray(weights, dtype=float),
        "realized_degrees": np.asarray(realized_degrees, dtype=float),
    }

    base_summary = {
        "graph_index": graph_index,
        "graph_seed": graph_seed,
        "sequence_seed": sequence_seed,
        "rewire_seed": rewire_seed,
        "n": config.n,
        "tau": tau,
        "tau_convention": config.tau_convention,
        "target_mean_degree": config.target_mean_degree,
        "sequence_mode": config.sequence_mode,
        "w_min": config.w_min,
        "w_max": config.w_max,
        "cutoff_mode": config.cutoff_mode,
        "cutoff_coefficient": config.cutoff_coefficient,
        "component_mode": config.component_mode,
        "structural_cap": diagnostics.structural_cap,
        "final_upper_cap": diagnostics.final_upper_cap,
        "fraction_at_upper_cap": diagnostics.fraction_at_upper_cap,
        "number_clipped_high": diagnostics.number_clipped_high,
        "max_expected_weight": diagnostics.final_max,
        "max_realized_degree": int(np.max(realized_degrees)),
        "realized_mean_degree": float(np.mean(realized_degrees)),
        "num_edges": graph.number_of_edges(),
        "num_components": nx.number_connected_components(graph),
        "weight_estimated_tau_after_cutoff": diagnostics.estimated_tau_after_cutoff,
        "tau_audit_note": "user-facing tau is experimental CCDF exponent; baseline effective tau is tau + 1",
    }

    for model_variant, variant_graph, edge_overlap, requested_swaps, max_tries in variants:
        analyzed_graph = select_component(variant_graph, config.component_mode)
        spectrum = sparse_tail_spectrum(
            analyzed_graph,
            model_variant=model_variant,
            k_per_side=config.k_per_side,
            eigsh_tol=config.eigsh_tol,
            eigsh_maxiter=config.eigsh_maxiter,
            eigsh_ncv_factor=config.eigsh_ncv_factor,
            validate_dense=config.validate_dense,
        )
        prefix = "cl" if model_variant == MODEL_CHUNG_LU else "configuration"
        arrays[f"{prefix}_eigenvalues_la"] = spectrum.eigenvalues_la
        arrays[f"{prefix}_eigenvalues_sa"] = spectrum.eigenvalues_sa
        arrays[f"{prefix}_residuals_la"] = spectrum.residuals_la
        arrays[f"{prefix}_residuals_sa"] = spectrum.residuals_sa
        arrays[f"{prefix}_positive_nonperron"] = spectrum.positive_nonperron
        arrays[f"{prefix}_negative_eigenvalues"] = spectrum.negative_eigenvalues
        arrays[f"{prefix}_absolute_nonperron_tail"] = spectrum.absolute_nonperron_tail
        arrays[f"{prefix}_lambda_squared_tail"] = spectrum.lambda_squared_tail

        largest_positive_nonperron = float(spectrum.positive_nonperron[0]) if len(spectrum.positive_nonperron) else math.nan
        absolute_most_negative = float(abs(spectrum.negative_eigenvalues[0])) if len(spectrum.negative_eigenvalues) else math.nan
        sqrt_max_degree = math.sqrt(max(float(np.max(realized_degrees)), 1.0))
        sqrt_cap = math.sqrt(float(diagnostics.final_upper_cap)) if diagnostics.final_upper_cap > 0 else math.nan
        spectral_count_primary = top_count_for_quantile(config.n - 1, PRIMARY_TAIL_QUANTILE)
        prefactor = transformed_prefactor(
            degree_desc,
            spectrum.absolute_nonperron_tail,
            denominator_degree=config.n,
            denominator_lambda=config.n - 1,
            top_count=spectral_count_primary,
        )
        primary_threshold = threshold_rows_for_model(
            graph_index=graph_index,
            model_variant=model_variant,
            tau=tau,
            config=config,
            degree_desc=degree_desc,
            absolute_desc=spectrum.absolute_nonperron_tail,
            lambda_squared_desc=spectrum.lambda_squared_tail,
        )
        threshold_rows.extend(primary_threshold)
        primary_quantile = primary_tail_quantile_for_config(config)
        primary = next(row for row in primary_threshold if float(row["tail_lower_quantile"]) == primary_quantile)

        row = {
            **base_summary,
            "model_variant": model_variant,
            "edge_overlap_fraction": edge_overlap,
            "requested_swaps": requested_swaps,
            "max_tries": max_tries,
            "perron_eigenvalue": spectrum.perron_eigenvalue,
            "largest_positive_nonperron_eigenvalue": largest_positive_nonperron,
            "absolute_most_negative_eigenvalue": absolute_most_negative,
            "retained_absolute_tail_count": spectrum.retained_absolute_count,
            "target_tail_count": config.target_tail_count,
            "eigsh_seconds": spectrum.eigsh_seconds,
            "max_residual_la": float(np.max(spectrum.residuals_la)) if len(spectrum.residuals_la) else math.nan,
            "max_residual_sa": float(np.max(spectrum.residuals_sa)) if len(spectrum.residuals_sa) else math.nan,
            "dense_max_abs_error": spectrum.dense_max_abs_error,
            "primary_tail_lower_quantile": primary_quantile,
            "alpha_hill": primary["alpha_hill"],
            "beta_hill": primary["beta_hill"],
            "gamma_hill": primary["gamma_hill"],
            "beta_minus_2alpha_hill": primary["beta_minus_2alpha_hill"],
            "beta_over_2alpha_hill": primary["beta_over_2alpha_hill"],
            "gamma_minus_alpha_hill": primary["gamma_minus_alpha_hill"],
            "transformed_ccdf_prefactor": prefactor,
        }
        tail_rows.append(row)

        cutoff_rows.append(
            {
                **base_summary,
                "model_variant": model_variant,
                "edge_overlap_fraction": edge_overlap,
                "expected_weight_cap": diagnostics.final_upper_cap,
                "predicted_hub_scale": sqrt_cap,
                "sqrt_max_realized_degree": sqrt_max_degree,
                "largest_positive_nonperron_eigenvalue": largest_positive_nonperron,
                "absolute_most_negative_eigenvalue": absolute_most_negative,
                "perron_eigenvalue": spectrum.perron_eigenvalue,
                "largest_positive_nonperron_over_sqrt_max_degree": largest_positive_nonperron / sqrt_max_degree,
                "absolute_most_negative_over_sqrt_max_degree": absolute_most_negative / sqrt_max_degree,
                "largest_positive_nonperron_over_sqrt_cap": largest_positive_nonperron / sqrt_cap if sqrt_cap > 0 else math.nan,
                "absolute_most_negative_over_sqrt_cap": absolute_most_negative / sqrt_cap if sqrt_cap > 0 else math.nan,
                "diagnostic_note": (
                    "final_upper_cap caps expected weights only; sqrt(final_upper_cap) is a heuristic hub scale, "
                    "not a hard spectral bound; realized degrees can exceed expected weights; Perron is separate"
                ),
            }
        )

        positive_fit = rank_match_fit(
            degree_desc,
            spectrum.positive_nonperron,
            max_rank=config.target_tail_count,
        )
        negative_fit = rank_match_fit(
            degree_desc,
            np.abs(spectrum.negative_eigenvalues),
            max_rank=config.target_tail_count,
        )
        rank_rows.append(
            {
                **base_summary,
                "model_variant": model_variant,
                "side": "positive_nonperron",
                "reference_slope": 0.5,
                "perron_eigenvalue_display_only": spectrum.perron_eigenvalue,
                **positive_fit,
            }
        )
        rank_rows.append(
            {
                **base_summary,
                "model_variant": model_variant,
                "side": "negative_absolute",
                "reference_slope": 0.5,
                "perron_eigenvalue_display_only": math.nan,
                **negative_fit,
            }
        )

    return base_summary, tail_rows, threshold_rows, rank_rows, cutoff_rows, arrays


def save_graph_partial(
    path: Path,
    *,
    base_summary: dict[str, object],
    tail_rows: list[dict[str, object]],
    threshold_rows: list[dict[str, object]],
    rank_rows: list[dict[str, object]],
    cutoff_rows: list[dict[str, object]],
    arrays: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        base_summary=json.dumps(base_summary),
        tail_rows=json.dumps(tail_rows),
        threshold_rows=json.dumps(threshold_rows),
        rank_rows=json.dumps(rank_rows),
        cutoff_rows=json.dumps(cutoff_rows),
        **arrays,
    )


def load_graph_partial(path: Path) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        base_summary = json.loads(str(data["base_summary"]))
        tail_rows = json.loads(str(data["tail_rows"]))
        threshold_rows = json.loads(str(data["threshold_rows"]))
        rank_rows = json.loads(str(data["rank_rows"]))
        cutoff_rows = json.loads(str(data["cutoff_rows"]))
        arrays = {key: np.asarray(data[key]) for key in data.files if key not in {"base_summary", "tail_rows", "threshold_rows", "rank_rows", "cutoff_rows"}}
    return base_summary, tail_rows, threshold_rows, rank_rows, cutoff_rows, arrays


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def finite_mean_se(values: Iterable[object]) -> tuple[float, float, int]:
    array = np.asarray([float(value) for value in values if str(value) != ""], dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return math.nan, math.nan, 0
    mean = float(np.mean(array))
    se = float(np.std(array, ddof=1) / math.sqrt(len(array))) if len(array) >= 2 else math.nan
    return mean, se, int(len(array))


def aggregate_threshold_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries = []
    keys = sorted({(row["model_variant"], float(row["tail_lower_quantile"])) for row in rows})
    metrics = [
        "alpha_hill",
        "beta_hill",
        "gamma_hill",
        "beta_minus_2alpha_hill",
        "beta_over_2alpha_hill",
        "gamma_minus_alpha_hill",
        "alpha_regression",
        "beta_regression",
        "gamma_regression",
    ]
    for model_variant, q0 in keys:
        group = [row for row in rows if row["model_variant"] == model_variant and float(row["tail_lower_quantile"]) == q0]
        summary: dict[str, object] = {
            "model_variant": model_variant,
            "tail_lower_quantile": q0,
            "number_of_graphs": len(group),
            "estimator_note": "graph-equal means and standard errors across graphs",
        }
        for metric in metrics:
            mean, se, count = finite_mean_se(row.get(metric, math.nan) for row in group)
            summary[f"mean_{metric}"] = mean
            summary[f"se_{metric}"] = se
            summary[f"valid_graphs_{metric}"] = count
        summaries.append(summary)
    return summaries


def aggregate_rank_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries = []
    keys = sorted({(row["model_variant"], row["side"]) for row in rows})
    metrics = [
        "slope",
        "slope_se",
        "r2",
        "median_lambda_over_sqrt_degree",
        "iqr_lambda_over_sqrt_degree",
        "spearman_rank_correlation",
    ]
    for model_variant, side in keys:
        group = [row for row in rows if row["model_variant"] == model_variant and row["side"] == side]
        summary: dict[str, object] = {
            "model_variant": model_variant,
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


def pooled_ccdf_rows(
    *,
    tau: float,
    config: TailMatchConfig,
    partials: list[tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, np.ndarray]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    max_rank = max(config.target_tail_count, top_count_for_quantile(config.n - 1, min(config.tail_lower_quantiles)))
    for base, _, _, _, _, arrays in partials:
        graph_index = int(base["graph_index"])
        degrees = np.sort(np.asarray(arrays["realized_degrees"], dtype=float))[::-1]
        weights = np.sort(np.asarray(arrays["expected_weights"], dtype=float))[::-1]
        for curve, values, denom in (
            ("realized_degree_ccdf", degrees, config.n),
            ("expected_weight_ccdf", weights, config.n),
        ):
            for rank, value in enumerate(values[:max_rank], start=1):
                rows.append(
                    {
                        "tau": tau,
                        "tau_convention": config.tau_convention,
                        "graph_index": graph_index,
                        "model_variant": "degree_sequence",
                        "curve": curve,
                        "rank": rank,
                        "value": float(value),
                        "tail_probability": rank / float(denom),
                    }
                )
        for model_variant, prefix in ((MODEL_CHUNG_LU, "cl"), (MODEL_CONFIGURATION, "configuration")):
            key = f"{prefix}_absolute_nonperron_tail"
            if key not in arrays:
                continue
            absolute_values = np.asarray(arrays[key], dtype=float)
            squared_values = np.asarray(arrays[f"{prefix}_lambda_squared_tail"], dtype=float)
            for curve, values in (
                ("absolute_spectral_ccdf", absolute_values),
                ("lambda_squared_spectral_ccdf", squared_values),
            ):
                for rank, value in enumerate(values[:max_rank], start=1):
                    rows.append(
                        {
                            "tau": tau,
                            "tau_convention": config.tau_convention,
                            "graph_index": graph_index,
                            "model_variant": model_variant,
                            "curve": curve,
                            "rank": rank,
                            "value": float(value),
                            "tail_probability": rank / float(config.n - 1),
                        }
                    )
    return rows


def _plot_curve_rows(
    ax: plt.Axes,
    rows: list[dict[str, object]],
    *,
    curve: str,
    model_variant: str | None = None,
    label: str,
    color: str | None = None,
    transform_x=lambda x: x,
    transform_y=lambda y: y,
) -> None:
    selected = [
        row
        for row in rows
        if row["curve"] == curve and (model_variant is None or row["model_variant"] == model_variant)
    ]
    if not selected:
        return
    by_rank: dict[int, list[tuple[float, float]]] = {}
    for row in selected:
        by_rank.setdefault(int(row["rank"]), []).append((float(row["value"]), float(row["tail_probability"])))
    xs = []
    ys = []
    for rank in sorted(by_rank):
        values = np.asarray(by_rank[rank], dtype=float)
        xs.append(float(np.median(values[:, 0])))
        ys.append(float(np.median(values[:, 1])))
    ax.loglog(transform_x(np.asarray(xs)), transform_y(np.asarray(ys)), marker=".", linewidth=1.2, label=label, color=color)


def make_tau_figures(output_dir: Path, tau: float, config: TailMatchConfig, ccdf_rows: list[dict[str, object]], threshold_summary: list[dict[str, object]], rank_rows: list[dict[str, object]], cutoff_rows: list[dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    _plot_curve_rows(ax, ccdf_rows, curve="realized_degree_ccdf", label="realized degree")
    _plot_curve_rows(ax, ccdf_rows, curve="expected_weight_ccdf", label="expected weight")
    ax.set_xlabel("degree or expected weight")
    ax.set_ylabel("CCDF")
    ax.set_title(f"Degree and expected-weight tails, tau={tau}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "degree_and_weight_ccdf.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    _plot_curve_rows(ax, ccdf_rows, curve="absolute_spectral_ccdf", model_variant=MODEL_CHUNG_LU, label="Chung-Lu")
    _plot_curve_rows(ax, ccdf_rows, curve="absolute_spectral_ccdf", model_variant=MODEL_CONFIGURATION, label="configuration")
    ax.set_xlabel("|lambda|, Perron excluded")
    ax.set_ylabel("CCDF denominator N-1")
    ax.set_title(f"Absolute spectral tail, tau={tau}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "spectral_absolute_ccdf.png", dpi=180)
    fig.savefig(output_dir / "chung_lu_vs_configuration_spectral_tail.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    _plot_curve_rows(ax, ccdf_rows, curve="lambda_squared_spectral_ccdf", model_variant=MODEL_CHUNG_LU, label="lambda^2 Chung-Lu")
    _plot_curve_rows(ax, ccdf_rows, curve="lambda_squared_spectral_ccdf", model_variant=MODEL_CONFIGURATION, label="lambda^2 configuration")
    _plot_curve_rows(ax, ccdf_rows, curve="realized_degree_ccdf", label="realized degree")
    ax.set_xlabel("k or lambda^2")
    ax.set_ylabel("CCDF")
    ax.set_title(f"lambda^2 and degree CCDF comparison, tau={tau}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "lambda_squared_vs_degree_ccdf.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    _plot_curve_rows(ax, ccdf_rows, curve="absolute_spectral_ccdf", model_variant=MODEL_CHUNG_LU, label="T_lambda(x)")
    _plot_curve_rows(ax, ccdf_rows, curve="realized_degree_ccdf", label="T_D(x^2)", transform_x=lambda x: np.sqrt(x))
    ax.set_xlabel("x")
    ax.set_ylabel("CCDF")
    ax.set_title(f"Transformed degree/spectral CCDF, tau={tau}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "transformed_degree_spectral_ccdf_match.png", dpi=180)
    plt.close(fig)

    for side, filename in (("positive_nonperron", "positive_rank_match.png"), ("negative_absolute", "negative_rank_match.png")):
        fig, ax = plt.subplots(figsize=(7, 5))
        side_rows = [row for row in rank_rows if row["side"] == side]
        for model_variant, marker in ((MODEL_CHUNG_LU, "o"), (MODEL_CONFIGURATION, "s")):
            values = [row for row in side_rows if row["model_variant"] == model_variant]
            if values:
                ax.errorbar(
                    [0.5],
                    [np.nanmean([float(row["slope"]) for row in values])],
                    yerr=[np.nanstd([float(row["slope"]) for row in values], ddof=1) / math.sqrt(max(len(values), 1)) if len(values) > 1 else 0],
                    marker=marker,
                    linestyle="",
                    label=model_variant,
                )
        ax.axhline(0.5, color="black", linewidth=1, linestyle="--", label="sqrt(D) slope")
        ax.set_xlim(0.25, 0.75)
        ax.set_xticks([])
        ax.set_ylabel("rank-size log-log slope")
        ax.set_title(f"{side} rank match, tau={tau}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for model_variant, marker in ((MODEL_CHUNG_LU, "o"), (MODEL_CONFIGURATION, "s")):
        group = [row for row in threshold_summary if row["model_variant"] == model_variant]
        if not group:
            continue
        xs = [float(row["tail_lower_quantile"]) for row in group]
        ys = [float(row["mean_beta_minus_2alpha_hill"]) for row in group]
        yerr = [float(row["se_beta_minus_2alpha_hill"]) if np.isfinite(float(row["se_beta_minus_2alpha_hill"])) else 0.0 for row in group]
        ax.errorbar(xs, ys, yerr=yerr, marker=marker, label=model_variant)
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("lower tail quantile q0")
    ax.set_ylabel("beta - 2 alpha")
    ax.set_title(f"Exponent threshold stability, tau={tau}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "exponent_match_across_thresholds.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for model_variant, marker in ((MODEL_CHUNG_LU, "o"), (MODEL_CONFIGURATION, "s")):
        group = [row for row in cutoff_rows if row["model_variant"] == model_variant]
        if not group:
            continue
        xs = [float(row["sqrt_max_realized_degree"]) for row in group]
        ys = [float(row["largest_positive_nonperron_eigenvalue"]) for row in group]
        ax.scatter(xs, ys, marker=marker, label=f"{model_variant}: +")
        ys_neg = [float(row["absolute_most_negative_eigenvalue"]) for row in group]
        ax.scatter(xs, ys_neg, marker=marker, facecolors="none", label=f"{model_variant}: -")
    lim = ax.get_xlim()
    grid = np.linspace(max(lim[0], 1e-6), lim[1], 100)
    ax.plot(grid, grid, color="black", linewidth=1, linestyle="--", label="lambda = sqrt(max degree)")
    ax.set_xlabel("sqrt(max realized degree)")
    ax.set_ylabel("edge eigenvalue magnitude")
    ax.set_title(f"Cutoff/hub scale diagnostic, tau={tau}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "cutoff_scale_diagnostic.png", dpi=180)
    plt.close(fig)

    if any(row["model_variant"] == MODEL_CONFIGURATION for row in threshold_summary):
        fig, ax = plt.subplots(figsize=(7, 5))
        q_values = sorted({float(row["tail_lower_quantile"]) for row in threshold_summary})
        diffs = []
        for q0 in q_values:
            cl = next((row for row in threshold_summary if row["model_variant"] == MODEL_CHUNG_LU and float(row["tail_lower_quantile"]) == q0), None)
            cfg = next((row for row in threshold_summary if row["model_variant"] == MODEL_CONFIGURATION and float(row["tail_lower_quantile"]) == q0), None)
            if cl and cfg:
                diffs.append(float(cfg["mean_beta_hill"]) - float(cl["mean_beta_hill"]))
            else:
                diffs.append(math.nan)
        ax.plot(q_values, diffs, marker="o")
        ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel("lower tail quantile q0")
        ax.set_ylabel("configuration beta - Chung-Lu beta")
        ax.set_title(f"Paired tail exponent differences, tau={tau}")
        fig.tight_layout()
        fig.savefig(output_dir / "paired_tail_exponent_differences.png", dpi=180)
        plt.close(fig)


def run_manifest(config: TailMatchConfig, tau: float) -> dict[str, object]:
    data = asdict(config)
    data["output_dir"] = str(config.output_dir)
    data["tau"] = tau
    data["tau_convention"] = config.tau_convention
    data["tau_convention_resolution"] = (
        "tau is the experimental CCDF exponent P(W >= t) ~ t^-tau; "
        "the checked-in Chung-Lu baseline still uses density exponent, so this experiment calls it with effective tau = tau + 1"
    )
    data["model"] = "power_law_chung_lu_degree_spectral_tail_match"
    data["spectral_solver"] = "scipy.sparse.linalg.eigsh"
    data["spectral_selection_method"] = "sparse_algebraic_edges_perron_excluded"
    data["zero_removal_rule"] = (
        "central zeros are not computed or removed; positive-threshold spectral CCDFs use denominator N-1"
    )
    data["remove_exact_zero_eigenvalues"] = False
    data["zero_eigenvalue_atol"] = None
    data["timestamp"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    data["code_version"] = code_version()
    data["notes"] = [
        "This is a tail-distribution experiment, not a spacing/unfolding experiment.",
        "Perron eigenvalue is saved but excluded from primary spectral-tail fits.",
        "Spectral CCDF denominators use N-1, not the number of sparse eigenvalues retained.",
        "final_upper_cap is a cap on expected weights, not a hard spectral bound.",
    ]
    return data


def aggregate_tau_output(config: TailMatchConfig, tau: float, partial_dir: Path | None = None) -> dict[str, object]:
    output_dir = tau_output_dir(config, tau)
    if partial_dir is None:
        partial_dir = output_dir / "partials"
    partial_paths = sorted(Path(partial_dir).glob("graph_*.npz"))
    if not partial_paths:
        raise SystemExit(f"no graph partials found in {partial_dir}")
    partials = [load_graph_partial(path) for path in partial_paths]
    tail_rows = [row for partial in partials for row in partial[1]]
    threshold_rows = [row for partial in partials for row in partial[2]]
    rank_rows = [row for partial in partials for row in partial[3]]
    cutoff_rows = [row for partial in partials for row in partial[4]]
    threshold_summary = aggregate_threshold_rows(threshold_rows)
    rank_summary = aggregate_rank_rows(rank_rows)
    ccdf_rows = pooled_ccdf_rows(tau=tau, config=config, partials=partials)

    write_csv(output_dir / "tail_match_by_graph.csv", tail_rows)
    write_csv(output_dir / "threshold_stability_by_graph.csv", threshold_rows)
    write_csv(output_dir / "threshold_stability_summary.csv", threshold_summary)
    write_csv(output_dir / "rank_match_by_graph.csv", rank_rows)
    write_csv(output_dir / "rank_match_summary.csv", rank_summary)
    write_csv(output_dir / "cutoff_scale_by_graph.csv", cutoff_rows)
    write_csv(output_dir / "pooled_ccdf_curves.csv", ccdf_rows)
    with (output_dir / "run_manifest.json").open("w") as handle:
        json.dump(run_manifest(config, tau), handle, indent=2, sort_keys=True)
    make_tau_figures(output_dir, tau, config, ccdf_rows, threshold_summary, rank_rows, cutoff_rows)

    primary_quantile = primary_tail_quantile_for_config(config)
    primary = [
        row
        for row in threshold_summary
        if float(row["tail_lower_quantile"]) == primary_quantile and row["model_variant"] == MODEL_CHUNG_LU
    ]
    result = {
        "tau": tau,
        "output_dir": str(output_dir),
        "number_of_partials": len(partial_paths),
    }
    if primary:
        row = primary[0]
        result.update(
            {
                "mean_alpha_hill": row["mean_alpha_hill"],
                "mean_beta_over_2_hill": float(row["mean_beta_hill"]) / 2.0,
                "mean_gamma_hill": row["mean_gamma_hill"],
                "mean_beta_minus_2alpha_hill": row["mean_beta_minus_2alpha_hill"],
                "mean_gamma_minus_alpha_hill": row["mean_gamma_minus_alpha_hill"],
            }
        )
    return result


def make_grid_summary(config: TailMatchConfig, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    output_dir = Path(config.output_dir) / f"n_{config.n}"
    write_csv(output_dir / "degree_spectral_tail_grid_summary.csv", rows)
    fig, ax = plt.subplots(figsize=(7, 5))
    tau_values = [float(row["tau"]) for row in rows]
    ax.plot(tau_values, [float(row["mean_alpha_hill"]) for row in rows], marker="o", label="realized degree alpha")
    ax.plot(tau_values, [float(row["mean_beta_over_2_hill"]) for row in rows], marker="s", label="spectral beta / 2")
    ax.plot(tau_values, [float(row["mean_gamma_hill"]) for row in rows], marker="^", label="lambda^2 gamma")
    ax.plot(tau_values, tau_values, color="black", linewidth=1, linestyle="--", label="input CCDF tau")
    ax.set_xlabel("input tau")
    ax.set_ylabel("tail exponent")
    ax.set_title("Degree versus spectral tail exponent comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "degree_spectral_tail_exponent_comparison.png", dpi=180)
    plt.close(fig)


def run_graph(config: TailMatchConfig, tau: float, graph_index: int, partial_dir: Path | None = None) -> Path:
    graph_seeds, sequence_seeds, rewire_seeds = graph_seeds_for(config, tau)
    graph_seed = graph_seeds[graph_index]
    sequence_seed = sequence_seeds[graph_index]
    rewire_seed = rewire_seeds[graph_index]
    result = graph_tail_objects(
        graph_index=graph_index,
        graph_seed=graph_seed,
        sequence_seed=sequence_seed,
        rewire_seed=rewire_seed,
        tau=tau,
        config=config,
    )
    if partial_dir is None:
        partial_dir = tau_output_dir(config, tau) / "partials"
    path = Path(partial_dir) / f"graph_{graph_index:04d}.npz"
    save_graph_partial(
        path,
        base_summary=result[0],
        tail_rows=result[1],
        threshold_rows=result[2],
        rank_rows=result[3],
        cutoff_rows=result[4],
        arrays=result[5],
    )
    return path


def run_tau(config: TailMatchConfig, tau: float) -> dict[str, object]:
    output_dir = tau_output_dir(config, tau)
    partial_dir = output_dir / "partials"
    partial_dir.mkdir(parents=True, exist_ok=True)
    for graph_index in range(config.num_graphs):
        run_graph(config, tau, graph_index, partial_dir=partial_dir)
    return aggregate_tau_output(config, tau, partial_dir=partial_dir)


def parse_tau_values(text: str | None) -> tuple[float, ...]:
    if text is None or text.strip() == "":
        return DEFAULT_TAU_VALUES
    return tuple(float(part.strip()) for part in text.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Degree-versus-spectral-tail matching for power-law Chung-Lu graphs.")
    parser.add_argument("--tau", default=None, help="Comma-separated tau values. Convention: experimental CCDF exponent.")
    parser.add_argument("--n", type=int, default=N)
    parser.add_argument("--degree", type=float, default=DEFAULT_TARGET_MEAN_DEGREE)
    parser.add_argument("--num-graphs", type=int, default=DEFAULT_NUM_GRAPHS)
    parser.add_argument("--graph-index", type=int, default=None)
    parser.add_argument("--partial-dir", type=Path, default=None)
    parser.add_argument("--aggregate-partials", action="store_true")
    parser.add_argument("--run-fixed-degree-control", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--k-per-side", type=int, default=DEFAULT_K_PER_SIDE)
    parser.add_argument("--target-tail-count", type=int, default=DEFAULT_TARGET_TAIL_COUNT)
    parser.add_argument("--eigsh-tol", type=float, default=DEFAULT_EIGSH_TOL)
    parser.add_argument("--eigsh-maxiter", type=int, default=DEFAULT_EIGSH_MAXITER)
    parser.add_argument("--eigsh-ncv-factor", type=int, default=DEFAULT_EIGSH_NCV_FACTOR)
    parser.add_argument("--sequence-mode", default=SEQUENCE_MODE)
    parser.add_argument("--w-min", type=float, default=W_MIN)
    parser.add_argument("--w-max", type=float, default=None)
    parser.add_argument("--cutoff-mode", default=CUTOFF_MODE)
    parser.add_argument("--cutoff-coefficient", type=float, default=CUTOFF_COEFFICIENT)
    parser.add_argument("--component-mode", default=DEFAULT_COMPONENT_MODE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--validate-dense", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> TailMatchConfig:
    return TailMatchConfig(
        n=args.n,
        tau_values=parse_tau_values(args.tau),
        target_mean_degree=args.degree,
        num_graphs=args.num_graphs,
        sequence_mode=args.sequence_mode,
        w_min=args.w_min,
        w_max=args.w_max,
        cutoff_mode=args.cutoff_mode,
        cutoff_coefficient=args.cutoff_coefficient,
        component_mode=args.component_mode,
        k_per_side=args.k_per_side,
        eigsh_tol=args.eigsh_tol,
        eigsh_maxiter=args.eigsh_maxiter,
        eigsh_ncv_factor=args.eigsh_ncv_factor,
        target_tail_count=args.target_tail_count,
        run_fixed_degree_control=args.run_fixed_degree_control,
        validate_dense=args.validate_dense,
        output_dir=args.output_dir,
    )


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    if args.smoke_test:
        config = TailMatchConfig(
            n=args.n,
            tau_values=parse_tau_values(args.tau),
            target_mean_degree=args.degree,
            num_graphs=args.num_graphs,
            sequence_mode=args.sequence_mode,
            w_min=args.w_min,
            w_max=args.w_max,
            cutoff_mode=args.cutoff_mode,
            cutoff_coefficient=args.cutoff_coefficient,
            component_mode=args.component_mode,
            k_per_side=args.k_per_side,
            eigsh_tol=args.eigsh_tol,
            eigsh_maxiter=args.eigsh_maxiter,
            eigsh_ncv_factor=args.eigsh_ncv_factor,
            target_tail_count=args.target_tail_count,
            run_fixed_degree_control=args.run_fixed_degree_control,
            validate_dense=True,
            output_dir=args.output_dir,
            swaps_per_edge=2,
        )

    if args.graph_index is not None:
        if len(config.tau_values) != 1:
            raise SystemExit("--graph-index requires exactly one --tau value")
        path = run_graph(config, config.tau_values[0], args.graph_index, partial_dir=args.partial_dir)
        print(path)
        return

    grid_rows = []
    for tau in config.tau_values:
        if args.aggregate_partials:
            grid_rows.append(aggregate_tau_output(config, tau, partial_dir=args.partial_dir))
        else:
            grid_rows.append(run_tau(config, tau))
    make_grid_summary(config, grid_rows)


if __name__ == "__main__":
    main()
