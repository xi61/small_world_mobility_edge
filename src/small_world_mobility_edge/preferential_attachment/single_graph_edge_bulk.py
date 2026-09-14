"""Single-realization PA spectral, eigenvector, and graph-structure audit.

One stored graph and one dense eigendecomposition feed every output.  The
module reuses the canonical PA generators, zero-atom convention, gap-first
bookkeeping, and held-out empirical CDFs from the ensemble production path.
"""

from __future__ import annotations

import csv
import gc
import gzip
import hashlib
import json
import math
import os
import tempfile
import platform
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "small-world-mobility-edge-pa-single-graph"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy
from scipy import integrate, linalg
from scipy.sparse import csgraph

from ..empirical_spectral_density import estimate_variable_local_count_density
from .baseline import (
    MODEL_AFFINE_PA,
    MODEL_NONLINEAR_PA,
    generate_preferential_attachment_graph,
)
from .full_spectrum_scan import HISTOGRAM_QUANTILES
from .mobility_edge_postprocess import (
    GOE_MEAN_GAP_RATIO,
    POISSON_MEAN_GAP_RATIO,
    build_gap_records,
    ks_distance_unit_mean_spacing,
)


ANALYSIS_VERSION = "pa_single_graph_edge_bulk_v1"
WINDOW_SIZES = (10, 25, 50)
SUPPORT_SIZE = 10
ZERO_ATOL = 1e-8
ZERO_GAP_ATOL = 1e-14
HISTOGRAM_WINDOW_WIDTH = 0.05
HISTOGRAM_BINS = np.linspace(0.0, 3.5, 30)
RATIO_BINS = np.linspace(0.0, 1.0, 26)
RANDOM_BASELINE_SEED = 945001
RANDOM_BASELINE_SAMPLES = 1000


@dataclass(frozen=True)
class SingleGraphConfig:
    n: int
    m: int
    model_family: str
    parameter_value: float
    graph_seed: int
    output_root: Path
    graph_index: int = 0
    reference_cdf_path: Optional[Path] = None
    histogram_window_width: float = HISTOGRAM_WINDOW_WIDTH
    random_baseline_seed: int = RANDOM_BASELINE_SEED
    random_baseline_samples: int = RANDOM_BASELINE_SAMPLES
    zero_atol: float = ZERO_ATOL
    smoke_only_self_fitted_cdf: bool = False

    def __post_init__(self) -> None:
        if not 1 <= self.m < self.n:
            raise ValueError("require 1 <= m < n")
        if self.model_family not in {"affine_linear", "nonlinear"}:
            raise ValueError("model_family must be affine_linear or nonlinear")
        if not math.isfinite(self.parameter_value):
            raise ValueError("parameter_value must be finite")
        if self.model_family == "affine_linear" and self.parameter_value <= 1.0:
            raise ValueError("affine degree_ccdf_alpha must exceed one")
        if self.model_family == "nonlinear" and self.parameter_value < 0.0:
            raise ValueError("attachment_zeta must be nonnegative")
        if not 0.0 < self.histogram_window_width <= 1.0:
            raise ValueError("histogram_window_width must lie in (0,1]")
        if self.random_baseline_samples < 1:
            raise ValueError("random_baseline_samples must be positive")
        if self.reference_cdf_path is None and not self.smoke_only_self_fitted_cdf:
            raise ValueError("production requires a held-out reference CDF")

    @property
    def parameter_name(self) -> str:
        return "degree_ccdf_alpha" if self.model_family == "affine_linear" else "attachment_zeta"

    @property
    def parameter_slug(self) -> str:
        return str(self.parameter_value).replace(".", "p")

    @property
    def model_key(self) -> str:
        prefix = "affine_alpha" if self.model_family == "affine_linear" else "nonlinear_zeta"
        return f"{prefix}_{self.parameter_slug}"

    @property
    def model_label(self) -> str:
        if self.model_family == "affine_linear":
            return rf"affine $\alpha={self.parameter_value:g}$"
        return rf"nonlinear $\zeta={self.parameter_value:g}$"

    @property
    def output_dir(self) -> Path:
        return Path(self.output_root) / self.model_key

    def metadata(self) -> dict[str, object]:
        result = asdict(self)
        result["output_root"] = str(self.output_root)
        result["reference_cdf_path"] = (
            str(self.reference_cdf_path) if self.reference_cdf_path is not None else None
        )
        result.update(
            {
                "analysis_version": ANALYSIS_VERSION,
                "parameter_name": self.parameter_name,
                "model_key": self.model_key,
                "model_label": self.model_label,
            }
        )
        return result


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    extras = sorted({key for row in rows for key in row} - set(fields))
    fields.extend(extras)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def generate_fixed_graph(config: SingleGraphConfig) -> tuple[nx.Graph, dict[str, object]]:
    if config.model_family == "affine_linear":
        graph, diagnostics = generate_preferential_attachment_graph(
            n=config.n,
            m=config.m,
            seed=config.graph_seed,
            model=MODEL_AFFINE_PA,
            degree_ccdf_alpha=config.parameter_value,
        )
    else:
        graph, diagnostics = generate_preferential_attachment_graph(
            n=config.n,
            m=config.m,
            seed=config.graph_seed,
            model=MODEL_NONLINEAR_PA,
            attachment_zeta=config.parameter_value,
        )
    return graph, diagnostics.to_dict()


def graph_adjacency_csr(graph: nx.Graph):
    nodes = list(range(graph.number_of_nodes()))
    if hasattr(nx, "to_scipy_sparse_array"):
        return nx.to_scipy_sparse_array(graph, nodelist=nodes, dtype=float, format="csr")
    return nx.to_scipy_sparse_matrix(graph, nodelist=nodes, dtype=float, format="csr")


def sorted_dense_eigendecomposition(graph: nx.Graph) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes = list(range(graph.number_of_nodes()))
    matrix = nx.to_numpy_array(graph, nodelist=nodes, dtype=float)
    unsorted_values, unsorted_vectors = linalg.eigh(
        matrix,
        overwrite_a=True,
        check_finite=False,
        driver="evd",
    )
    order = np.argsort(unsorted_values, kind="mergesort")
    values = np.asarray(unsorted_values[order], dtype=float)
    vectors = np.asarray(unsorted_vectors[:, order], dtype=float)
    if np.any(np.diff(values) < -1e-12):
        raise AssertionError("eigenvalues are not sorted ascending")
    return values, vectors, np.asarray(order, dtype=np.int32)


def sampled_eigenpair_residuals(
    adjacency,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    indices: Sequence[int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in sorted(set(int(i) for i in indices)):
        vector = eigenvectors[:, index]
        residual = np.asarray(adjacency.dot(vector)).reshape(-1) - eigenvalues[index] * vector
        absolute = float(np.linalg.norm(residual))
        relative = absolute / max(1.0, abs(float(eigenvalues[index])))
        rows.append(
            {
                "global_index_0based": index,
                "global_index_1based": index + 1,
                "eigenvalue": float(eigenvalues[index]),
                "absolute_residual_norm": absolute,
                "relative_residual_norm": relative,
            }
        )
    if rows and max(float(row["relative_residual_norm"]) for row in rows) > 1e-8:
        raise AssertionError("sorted eigenpair residual exceeds 1e-8")
    return rows


def localization_scalars(eigenvectors: np.ndarray, block_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    n, count = eigenvectors.shape
    ipr = np.empty(count, dtype=float)
    for start in range(0, count, block_size):
        stop = min(start + block_size, count)
        squared = np.square(eigenvectors[:, start:stop])
        ipr[start:stop] = np.einsum("ij,ij->j", squared, squared)
    return ipr, n * ipr


def valid_atom_free_segments(
    eigenvalues: np.ndarray, zero_atol: float = ZERO_ATOL
) -> tuple[np.ndarray, list[np.ndarray]]:
    valid_mask = ~np.isclose(eigenvalues, 0.0, atol=zero_atol, rtol=0.0)
    valid_indices = np.flatnonzero(valid_mask)
    if not len(valid_indices):
        return valid_mask, []
    boundaries = np.flatnonzero(np.diff(valid_indices) > 1) + 1
    segments = [part for part in np.split(valid_indices, boundaries) if len(part)]
    return valid_mask, segments


def adjacent_gap_ratio_arrays(
    eigenvalues: np.ndarray,
    segments: Sequence[np.ndarray],
    zero_gap_atol: float = ZERO_GAP_ATOL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(eigenvalues)
    left_gap = np.full(count, np.nan, dtype=float)
    right_gap = np.full(count, np.nan, dtype=float)
    ratio = np.full(count, np.nan, dtype=float)
    for segment in segments:
        values = eigenvalues[segment]
        gaps = np.maximum(np.diff(values), 0.0)
        if len(gaps):
            right_gap[segment[:-1]] = gaps
            left_gap[segment[1:]] = gaps
        if len(gaps) >= 2:
            maximum = np.maximum(gaps[:-1], gaps[1:])
            valid = maximum > zero_gap_atol
            local = np.full(len(maximum), np.nan, dtype=float)
            local[valid] = np.minimum(gaps[:-1][valid], gaps[1:][valid]) / maximum[valid]
            ratio[segment[1:-1]] = local
    return left_gap, right_gap, ratio


def nested_edge_windows(eigenvalues: np.ndarray) -> dict[int, np.ndarray]:
    if len(eigenvalues) < max(WINDOW_SIZES):
        raise ValueError("spectrum is too short for top-50 edge window")
    return {
        size: np.arange(len(eigenvalues) - size, len(eigenvalues), dtype=int)
        for size in WINDOW_SIZES
    }


def _window_from_center(center: int, size: int) -> np.ndarray:
    start = center - (size - 1) // 2
    return np.arange(start, start + size, dtype=int)


def nested_bulk_windows(
    eigenvalues: np.ndarray,
    segments: Sequence[np.ndarray],
) -> tuple[int, dict[int, np.ndarray]]:
    largest = max(WINDOW_SIZES)
    valid_indices = np.concatenate(segments) if segments else np.array([], dtype=int)
    if len(valid_indices) < largest + 2:
        raise ValueError("not enough atom-free eigenvalues for a bulk-50 window")
    valid_position = {int(index): pos for pos, index in enumerate(valid_indices)}
    target = 0.5 * (len(valid_indices) - 1)
    candidates: list[int] = []
    for segment in segments:
        for center in segment:
            window = _window_from_center(int(center), largest)
            if window[0] > segment[0] and window[-1] < segment[-1]:
                candidates.append(int(center))
    if not candidates:
        raise ValueError("no atom-free segment can support bulk-50 plus neighboring gaps")
    center = min(candidates, key=lambda index: (abs(valid_position[index] - target), abs(eigenvalues[index])))
    windows = {size: _window_from_center(center, size) for size in WINDOW_SIZES}
    for size, indices in windows.items():
        if not set(indices).issubset(set(windows[largest])):
            raise AssertionError(f"bulk-{size} is not nested in bulk-{largest}")
    return center, windows


def window_memberships(index: int, windows: dict[int, np.ndarray]) -> str:
    return "|".join(str(size) for size in WINDOW_SIZES if index in set(windows[size]))


def degree_rank_and_percentile(degrees: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(degrees, dtype=int)
    ranks = np.asarray([1 + int(np.sum(values > degree)) for degree in values], dtype=int)
    percentiles = np.asarray([100.0 * float(np.mean(values <= degree)) for degree in values])
    return ranks, percentiles


def induced_statistics(graph: nx.Graph, vertices: Sequence[int]) -> dict[str, object]:
    nodes = [int(value) for value in vertices]
    induced = graph.subgraph(nodes)
    components = list(nx.connected_components(induced))
    return {
        "induced_edge_count": int(induced.number_of_edges()),
        "induced_density": float(nx.density(induced)),
        "induced_num_components": len(components),
        "induced_largest_component": max((len(part) for part in components), default=0),
        "induced_degree_sequence": [int(induced.degree(node)) for node in nodes],
    }


def shortest_path_statistics(distance_rows: np.ndarray, vertices: Sequence[int]) -> dict[str, object]:
    nodes = [int(value) for value in vertices]
    distances = []
    for left in range(len(nodes)):
        for right in range(left + 1, len(nodes)):
            distances.append(float(distance_rows[left, nodes[right]]))
    array = np.asarray(distances, dtype=float)
    finite = array[np.isfinite(array)]
    return {
        "pair_count": len(array),
        "connected_pair_count": len(finite),
        "disconnected_pair_count": int(np.sum(~np.isfinite(array))),
        "mean_pairwise_distance": float(np.mean(finite)) if len(finite) else math.nan,
        "median_pairwise_distance": float(np.median(finite)) if len(finite) else math.nan,
        "min_pairwise_distance": float(np.min(finite)) if len(finite) else math.nan,
        "max_pairwise_distance": float(np.max(finite)) if len(finite) else math.nan,
        "fraction_distance_1": float(np.mean(array == 1.0)),
        "fraction_distance_le_2": float(np.mean(array <= 2.0)),
        "fraction_distance_le_3": float(np.mean(array <= 3.0)),
    }


def random_comparison(value: float, random_values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(random_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not math.isfinite(value) or not len(finite):
        return math.nan, math.nan
    scale = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
    zscore = (value - float(np.mean(finite))) / scale if scale > 0.0 else math.nan
    percentile = 100.0 * float(np.mean(finite <= value))
    return float(zscore), float(percentile)


def histogram_window(center: float, width: float) -> tuple[float, float]:
    half_width = min(width / 2.0, center, 1.0 - center)
    if half_width <= 0.0:
        raise ValueError("histogram center must lie strictly inside (0,1)")
    return float(center - half_width), float(center + half_width)


def normalized_positive(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array) & (array > ZERO_GAP_ATOL)]
    if not len(array):
        return np.array([], dtype=float)
    mean = float(np.mean(array))
    return array / mean if mean > 0.0 else np.array([], dtype=float)


def load_reference_cdf(
    config: SingleGraphConfig,
    continuous_nonperron: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    if config.reference_cdf_path is not None:
        path = Path(config.reference_cdf_path)
        with np.load(path, allow_pickle=False) as data:
            grid = np.asarray(data["grid"], dtype=float)
            density = np.asarray(data["density"], dtype=float)
            cdf = np.asarray(data["cdf"], dtype=float)
            training = (
                np.asarray(data["training_graph_ids"], dtype=int)
                if "training_graph_ids" in data.files
                else np.array([], dtype=int)
            )
            held_out = (
                np.asarray(data["held_out_graph_ids"], dtype=int)
                if "held_out_graph_ids" in data.files
                else np.array([], dtype=int)
            )
        if len(held_out) and config.graph_index not in set(held_out.tolist()):
            raise ValueError("selected graph is not held out by the supplied CDF fold")
        if config.graph_index in set(training.tolist()):
            raise ValueError("reference CDF leaks the selected graph into training")
        metadata = {
            "method": "existing five-fold graph-level cross-fitted variable-local-count CDF",
            "path": str(path),
            "sha256": sha256_file(path),
            "training_graph_count": len(training),
            "held_out_graph_count": len(held_out),
            "selected_graph_index_held_out": True,
        }
    else:
        model = estimate_variable_local_count_density(
            [np.asarray(continuous_nonperron, dtype=float)],
            num_graphs=1,
            per_graph_dimension=float(len(continuous_nonperron)),
            bulk_local_count_half_width=0.20,
            tail_local_count_half_width=0.08,
            tail_quantile_cutoff=0.95,
            num_points=2001,
        )
        grid, density, cdf = model.grid, model.density, model.cdf
        metadata = {
            "method": "SMOKE TEST ONLY: self-fitted variable-local-count CDF",
            "path": None,
            "sha256": None,
            "training_graph_count": 1,
            "held_out_graph_count": 0,
            "selected_graph_index_held_out": False,
        }
    if not (len(grid) == len(density) == len(cdf)):
        raise ValueError("reference CDF arrays have inconsistent lengths")
    if np.any(np.diff(grid) <= 0.0) or np.any(np.diff(cdf) < -1e-12):
        raise ValueError("reference grid/CDF must be monotone")
    if not np.isclose(cdf[0], 0.0, atol=1e-8) or not np.isclose(cdf[-1], 1.0, atol=1e-8):
        raise ValueError("reference CDF must span [0,1]")
    return grid, density, cdf, metadata


def summarize_histogram_windows(
    config: SingleGraphConfig,
    record,
    grid: np.ndarray,
    cdf: np.ndarray,
    histogram_dir: Path,
) -> tuple[list[dict[str, object]], list[np.ndarray], list[np.ndarray], np.ndarray]:
    left_cdf = np.interp(record.gap_left, grid, cdf)
    right_cdf = np.interp(record.gap_right, grid, cdf)
    unfolded_gaps = record.continuous_dimension * np.maximum(right_cdf - left_cdf, 0.0)
    supported = (record.gap_left >= grid[0]) & (record.gap_right <= grid[-1])
    unfolded_gaps[~supported] = np.nan
    gap_ensemble_q = np.interp(record.gap_energy, grid, cdf)
    ratio_ensemble_q = np.interp(record.ratio_energy, grid, cdf)
    rows: list[dict[str, object]] = []
    raw_histograms: list[np.ndarray] = []
    unfolded_histograms: list[np.ndarray] = []
    payload: dict[str, np.ndarray] = {
        "centers": np.asarray(HISTOGRAM_QUANTILES, dtype=float),
        "bin_edges": HISTOGRAM_BINS,
    }
    for position, center in enumerate(HISTOGRAM_QUANTILES):
        q_low, q_high = histogram_window(float(center), config.histogram_window_width)
        if q_high >= 1.0 - 1e-15:
            gap_mask = (gap_ensemble_q >= q_low) & (gap_ensemble_q <= q_high)
            ratio_mask = (ratio_ensemble_q >= q_low) & (ratio_ensemble_q <= q_high)
        else:
            gap_mask = (gap_ensemble_q >= q_low) & (gap_ensemble_q < q_high)
            ratio_mask = (ratio_ensemble_q >= q_low) & (ratio_ensemble_q < q_high)
        positive = gap_mask & (~record.gap_is_zero)
        raw = normalized_positive(record.gap_value[positive])
        unfolded = normalized_positive(unfolded_gaps[positive])
        ratios = record.ratio_value[ratio_mask]
        ratios = ratios[np.isfinite(ratios)]
        raw_histograms.append(raw)
        unfolded_histograms.append(unfolded)
        payload[f"raw_{position:02d}"] = raw
        payload[f"unfolded_{position:02d}"] = unfolded
        payload[f"ratio_{position:02d}"] = ratios
        energy_low, energy_high = np.interp([q_low, q_high], cdf, grid)
        rows.append(
            {
                "model": config.model_key,
                config.parameter_name: config.parameter_value,
                "graph_seed": config.graph_seed,
                "number_of_graphs": 1,
                "nominal_center_q": float(center),
                "actual_q_low": q_low,
                "actual_q_high": q_high,
                "E_low": float(energy_low),
                "E_high": float(energy_high),
                "raw_positive_gap_count": len(raw),
                "unfolded_positive_gap_count": len(unfolded),
                "valid_ratio_count": len(ratios),
                "raw_spacing_mean_before_unit_normalization": float(np.mean(record.gap_value[positive])) if np.any(positive) else math.nan,
                "unfolded_spacing_mean_before_unit_normalization": float(np.nanmean(unfolded_gaps[positive])) if np.any(positive) else math.nan,
                "raw_KS_to_GOE": ks_distance_unit_mean_spacing(raw, "goe"),
                "raw_KS_to_Poisson": ks_distance_unit_mean_spacing(raw, "poisson"),
                "unfolded_KS_to_GOE": ks_distance_unit_mean_spacing(unfolded, "goe"),
                "unfolded_KS_to_Poisson": ks_distance_unit_mean_spacing(unfolded, "poisson"),
                "mean_local_raw_gap_ratio": float(np.mean(ratios)) if len(ratios) else math.nan,
                "normalization_denominator": "mean positive gap in this window of this one graph",
                "gap_assignment": "held-out ensemble-CDF quantile of within-graph gap midpoint",
                "ratio_assignment": "held-out ensemble-CDF quantile of middle eigenvalue",
            }
        )
    histogram_dir.mkdir(parents=True, exist_ok=True)
    write_csv(histogram_dir / "single_graph_spacing_summary.csv", rows)
    np.savez_compressed(
        histogram_dir / "single_graph_spacing_histogram_data.npz",
        unfolded_gap_by_record=unfolded_gaps,
        gap_ensemble_q=gap_ensemble_q,
        ratio_ensemble_q=ratio_ensemble_q,
        **payload,
    )
    return rows, raw_histograms, unfolded_histograms, unfolded_gaps


def plot_local_spacing_histograms(
    config: SingleGraphConfig,
    rows: Sequence[dict[str, object]],
    histograms: Sequence[np.ndarray],
    path: Path,
    label: str,
) -> None:
    figure, axes = plt.subplots(3, 4, figsize=(14.5, 10.0), sharex=True, sharey=True)
    x = np.linspace(0.0, 3.5, 500)
    goe = 0.5 * math.pi * x * np.exp(-math.pi * x * x / 4.0)
    poisson = np.exp(-x)
    for axis, row, values in zip(axes.flat, rows, histograms):
        displayed = values[(values >= 0.0) & (values <= 3.5)]
        if len(displayed):
            axis.hist(displayed, bins=HISTOGRAM_BINS, density=True, color="#52b788", alpha=0.75)
        else:
            axis.text(0.5, 0.5, "no positive spacings", ha="center", va="center", transform=axis.transAxes)
        axis.plot(x, goe, color="black", lw=1.2, label="GOE")
        axis.plot(x, poisson, color="firebrick", ls="--", lw=1.2, label="Poisson")
        axis.set_title(
            "q=[%.3f,%.3f), E=[%.2f,%.2f]\nn=%d positive gaps"
            % (
                row["actual_q_low"],
                row["actual_q_high"],
                row["E_low"],
                row["E_high"],
                row[f"{label}_positive_gap_count"],
            ),
            fontsize=9,
        )
        axis.grid(alpha=0.15)
    axes[0, 0].legend(frameon=False)
    for axis in axes[-1]:
        axis.set_xlabel(f"unit-mean {label} spacing $s$")
    for axis in axes[:, 0]:
        axis.set_ylabel("density")
    figure.suptitle(
        f"{config.model_label}: single graph, N={config.n}, seed={config.graph_seed}; {label} spacings"
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_global_spacing_and_ratio(
    config: SingleGraphConfig,
    record,
    unfolded_gaps: np.ndarray,
    path: Path,
) -> None:
    raw = normalized_positive(record.gap_value[~record.gap_is_zero])
    unfolded = normalized_positive(unfolded_gaps[~record.gap_is_zero])
    ratios = record.ratio_value[np.isfinite(record.ratio_value)]
    x = np.linspace(0.0, 3.5, 500)
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.5))
    for axis, values, title in zip(
        axes[:2],
        (raw, unfolded),
        ("raw gaps", "held-out-CDF unfolded gaps"),
    ):
        displayed = values[(values >= 0.0) & (values <= 3.5)]
        axis.hist(displayed, bins=HISTOGRAM_BINS, density=True, color="#52b788", alpha=0.75)
        axis.plot(x, 0.5 * math.pi * x * np.exp(-math.pi * x * x / 4.0), color="black", label="GOE")
        axis.plot(x, np.exp(-x), color="firebrick", ls="--", label="Poisson")
        axis.set_xlabel("unit-mean spacing $s$")
        axis.set_ylabel("density")
        axis.set_title(f"{title}\nn={len(values)}")
        axis.grid(alpha=0.15)
    axes[0].legend(frameon=False)
    axes[2].hist(ratios, bins=RATIO_BINS, density=True, color="#74a9cf", alpha=0.8)
    axes[2].axvline(GOE_MEAN_GAP_RATIO, color="black", ls="--", label="GOE mean")
    axes[2].axvline(POISSON_MEAN_GAP_RATIO, color="firebrick", ls=":", label="Poisson mean")
    axes[2].axvline(float(np.mean(ratios)), color="tab:blue", label="single-graph mean")
    axes[2].set_xlabel("adjacent gap ratio $r$")
    axes[2].set_ylabel("density")
    axes[2].set_title(f"raw adjacent-gap ratios\nn={len(ratios)}, mean={np.mean(ratios):.4f}")
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].grid(alpha=0.15)
    figure.suptitle(
        f"{config.model_label}: full continuous non-Perron spectrum of one N={config.n} graph"
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_single_graph_density(
    config: SingleGraphConfig,
    eigenvalues: np.ndarray,
    continuous_nonperron: np.ndarray,
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.7))
    axes[0].hist(eigenvalues, bins=160, density=True, color="#457b9d", alpha=0.8)
    axes[0].axvline(eigenvalues[-1], color="firebrick", ls="--", label="Perron")
    axes[0].set_title(f"all {len(eigenvalues)} eigenvalues; zero atom retained")
    axes[0].legend(frameon=False)
    axes[1].hist(continuous_nonperron, bins=160, density=True, color="#52b788", alpha=0.8)
    axes[1].set_title(f"continuous non-Perron levels; n={len(continuous_nonperron)}")
    for axis in axes:
        axis.set_xlabel(r"energy $E=\lambda$")
        axis.set_ylabel("empirical density")
        axis.grid(alpha=0.15)
    figure.suptitle(
        f"{config.model_label}: single graph, N={config.n}, seed={config.graph_seed}"
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_energy_nipr(
    config: SingleGraphConfig,
    eigenvalues: np.ndarray,
    nipr: np.ndarray,
    valid_mask: np.ndarray,
    edge_windows: dict[int, np.ndarray],
    bulk_windows: dict[int, np.ndarray],
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(11.5, 6.5))
    axis.scatter(eigenvalues[valid_mask], nipr[valid_mask], s=8, color="0.6", alpha=0.45, label="valid spectrum")
    styles = [
        (50, "#f6bd60", 36, "edge top 50"),
        (25, "#f28482", 52, "edge top 25"),
        (10, "#d00000", 72, "edge top 10"),
    ]
    for size, color, marker_size, label in styles:
        indices = edge_windows[size]
        axis.scatter(eigenvalues[indices], nipr[indices], s=marker_size, facecolors="none", edgecolors=color, linewidths=1.1, label=label)
    styles = [
        (50, "#90caf9", 34, "bulk 50"),
        (25, "#3f88c5", 50, "bulk 25"),
        (10, "#003f88", 70, "bulk 10"),
    ]
    for size, color, marker_size, label in styles:
        indices = bulk_windows[size]
        axis.scatter(eigenvalues[indices], nipr[indices], s=marker_size, facecolors="none", edgecolors=color, linewidths=1.1, label=label)
    axis.set_xlabel(r"actual energy $E=\lambda_i$")
    axis.set_ylabel(r"$N\,\mathrm{IPR}(v_i)$")
    axis.set_yscale("log")
    axis.set_title(f"{config.model_label}: single graph, N={config.n}, seed={config.graph_seed}")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=3, fontsize=8)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def top_mass_vertices(vector: np.ndarray, count: int = SUPPORT_SIZE) -> tuple[np.ndarray, np.ndarray]:
    masses = np.square(np.asarray(vector, dtype=float))
    unordered = np.argpartition(masses, -count)[-count:]
    order = unordered[np.argsort(masses[unordered], kind="mergesort")[::-1]]
    return order.astype(int), masses[order]


def precompute_selected_support(
    config: SingleGraphConfig,
    graph: nx.Graph,
    eigenvalues: np.ndarray,
    selected_vectors: np.ndarray,
    selected_indices: np.ndarray,
    ipr: np.ndarray,
    nipr: np.ndarray,
    ratios: np.ndarray,
    left_gap: np.ndarray,
    right_gap: np.ndarray,
    valid_mask: np.ndarray,
    edge_windows: dict[int, np.ndarray],
    bulk_center: int,
    bulk_windows: dict[int, np.ndarray],
    degrees: np.ndarray,
    degree_ranks: np.ndarray,
    degree_percentiles: np.ndarray,
) -> list[dict[str, object]]:
    column = {int(index): pos for pos, index in enumerate(selected_indices)}
    valid_indices = np.flatnonzero(valid_mask)
    valid_position = {int(index): pos for pos, index in enumerate(valid_indices)}
    rows: list[dict[str, object]] = []
    common = {
        "analysis_version": ANALYSIS_VERSION,
        "model": config.model_key,
        "model_family": config.model_family,
        config.parameter_name: config.parameter_value,
        "N": config.n,
        "m": config.m,
        "graph_seed": config.graph_seed,
    }
    for region, indices, windows in (
        ("edge", edge_windows[50], edge_windows),
        ("bulk", bulk_windows[50], bulk_windows),
    ):
        for index in indices:
            index = int(index)
            vertices, masses = top_mass_vertices(selected_vectors[:, column[index]])
            induced = induced_statistics(graph, vertices)
            edge_rank = config.n - index if region == "edge" else math.nan
            bulk_offset = valid_position[index] - valid_position[bulk_center] if region == "bulk" else math.nan
            row: dict[str, object] = {
                **common,
                "region": region,
                "region_window_membership": window_memberships(index, windows),
                "in_window_10": index in set(windows[10]),
                "in_window_25": index in set(windows[25]),
                "in_window_50": True,
                "edge_rank": edge_rank,
                "bulk_offset": bulk_offset,
                "global_eigenvalue_index_0based": index,
                "global_eigenvalue_index_1based": index + 1,
                "algebraic_rank_ascending_1based": index + 1,
                "valid_spectral_rank_1based": valid_position[index] + 1,
                "valid_spectral_quantile": (valid_position[index] + 0.5) / len(valid_indices),
                "eigenvalue_energy": float(eigenvalues[index]),
                "adjacent_gap_ratio": float(ratios[index]),
                "left_gap": float(left_gap[index]),
                "right_gap": float(right_gap[index]),
                "ipr": float(ipr[index]),
                "NIPR": float(nipr[index]),
                "top10_mass": float(np.sum(masses)),
                "top10_max_degree": int(np.max(degrees[vertices])),
                "top10_mean_degree": float(np.mean(degrees[vertices])),
                "top10_median_degree": float(np.median(degrees[vertices])),
                "top10_max_degree_percentile": float(np.max(degree_percentiles[vertices])),
                "is_perron": index == len(eigenvalues) - 1,
                **induced,
                "_vertices": vertices,
                "_masses": masses,
                "_degree_ranks": degree_ranks[vertices],
                "_degree_percentiles": degree_percentiles[vertices],
            }
            if row["is_perron"] and math.isfinite(float(row["adjacent_gap_ratio"])):
                raise AssertionError("Perron eigenpair must have undefined adjacent gap ratio")
            rows.append(row)
    return rows


def distance_table_for_sets(adjacency, sets: Sequence[np.ndarray]) -> tuple[np.ndarray, dict[int, int]]:
    unique = np.unique(np.concatenate([np.asarray(nodes, dtype=int) for nodes in sets]))
    distances = csgraph.shortest_path(
        adjacency,
        directed=False,
        unweighted=True,
        indices=unique,
        method="D",
        return_predecessors=False,
    )
    distances = np.atleast_2d(np.asarray(distances, dtype=float))
    return distances, {int(vertex): position for position, vertex in enumerate(unique)}


def rows_for_vertex_set(
    distance_matrix: np.ndarray,
    source_rows: dict[int, int],
    vertices: Sequence[int],
) -> np.ndarray:
    return distance_matrix[[source_rows[int(vertex)] for vertex in vertices]]


def random_structural_baseline(
    config: SingleGraphConfig,
    graph: nx.Graph,
    adjacency,
    degrees: np.ndarray,
    selected_rows: Sequence[dict[str, object]],
) -> tuple[list[dict[str, object]], np.ndarray, dict[int, int]]:
    rng = np.random.default_rng(config.random_baseline_seed)
    random_sets = [
        np.asarray(rng.choice(config.n, size=SUPPORT_SIZE, replace=False), dtype=int)
        for _ in range(config.random_baseline_samples)
    ]
    selected_sets = [np.asarray(row["_vertices"], dtype=int) for row in selected_rows]
    distance_matrix, source_rows = distance_table_for_sets(adjacency, [*selected_sets, *random_sets])
    baseline: list[dict[str, object]] = []
    for sample_index, vertices in enumerate(random_sets):
        induced = induced_statistics(graph, vertices)
        distance = shortest_path_statistics(
            rows_for_vertex_set(distance_matrix, source_rows, vertices), vertices
        )
        baseline.append(
            {
                "sample_index": sample_index,
                "baseline_seed": config.random_baseline_seed,
                "vertex_ids": vertices.tolist(),
                "max_degree": int(np.max(degrees[vertices])),
                "mean_degree": float(np.mean(degrees[vertices])),
                "median_degree": float(np.median(degrees[vertices])),
                **induced,
                **distance,
            }
        )
    return baseline, distance_matrix, source_rows


def finalize_selected_rows(
    config: SingleGraphConfig,
    graph: nx.Graph,
    degrees: np.ndarray,
    selected_rows: list[dict[str, object]],
    baseline_rows: Sequence[dict[str, object]],
    distance_matrix: np.ndarray,
    source_rows: dict[int, int],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    baseline_arrays = {
        key: np.asarray([float(row[key]) for row in baseline_rows], dtype=float)
        for key in (
            "mean_pairwise_distance",
            "induced_edge_count",
            "induced_density",
            "max_degree",
            "mean_degree",
        )
    }
    eigenpair_rows: list[dict[str, object]] = []
    support_rows: list[dict[str, object]] = []
    for row in selected_rows:
        vertices = np.asarray(row["_vertices"], dtype=int)
        masses = np.asarray(row["_masses"], dtype=float)
        distance = shortest_path_statistics(
            rows_for_vertex_set(distance_matrix, source_rows, vertices), vertices
        )
        row.update(distance)
        comparisons = {
            "mean_pairwise_distance": "random_distance",
            "induced_edge_count": "random_induced_edge_count",
            "induced_density": "random_induced_density",
            "top10_max_degree": "random_max_degree",
            "top10_mean_degree": "random_mean_degree",
        }
        source_name = {
            "top10_max_degree": "max_degree",
            "top10_mean_degree": "mean_degree",
        }
        for metric, prefix in comparisons.items():
            random_key = source_name.get(metric, metric)
            zscore, percentile = random_comparison(float(row[metric]), baseline_arrays[random_key])
            row[f"{prefix}_zscore"] = zscore
            row[f"{prefix}_percentile"] = percentile
        induced_degrees = list(row["induced_degree_sequence"])
        for support_rank, (vertex, mass) in enumerate(zip(vertices, masses), start=1):
            support_rows.append(
                {
                    "analysis_version": ANALYSIS_VERSION,
                    "model": config.model_key,
                    "model_family": config.model_family,
                    config.parameter_name: config.parameter_value,
                    "N": config.n,
                    "m": config.m,
                    "graph_seed": config.graph_seed,
                    "region": row["region"],
                    "edge_rank": row["edge_rank"],
                    "bulk_offset": row["bulk_offset"],
                    "global_eigenvalue_index_0based": row["global_eigenvalue_index_0based"],
                    "eigenvalue_energy": row["eigenvalue_energy"],
                    "support_rank": support_rank,
                    "vertex_id": int(vertex),
                    "eigenvector_mass": float(mass),
                    "full_graph_degree": int(degrees[vertex]),
                    "full_graph_degree_rank": int(row["_degree_ranks"][support_rank - 1]),
                    "full_graph_degree_percentile": float(row["_degree_percentiles"][support_rank - 1]),
                    "induced_degree": int(induced_degrees[support_rank - 1]),
                }
            )
        clean = {key: value for key, value in row.items() if not key.startswith("_")}
        eigenpair_rows.append(clean)
    return eigenpair_rows, support_rows


def plot_induced_subgraph(
    config: SingleGraphConfig,
    graph: nx.Graph,
    eigenpair_row: dict[str, object],
    support_rows: Sequence[dict[str, object]],
    path: Path,
) -> None:
    vertices = [int(row["vertex_id"]) for row in support_rows]
    masses = np.asarray([float(row["eigenvector_mass"]) for row in support_rows])
    degrees = np.asarray([float(row["full_graph_degree"]) for row in support_rows])
    induced = graph.subgraph(vertices).copy()
    positions = nx.spring_layout(induced, seed=config.graph_seed % (2**32 - 1))
    sizes = 350.0 + 3400.0 * masses / max(float(np.max(masses)), 1e-12)
    figure, axis = plt.subplots(figsize=(7.2, 6.1))
    nodes = nx.draw_networkx_nodes(
        induced,
        positions,
        nodelist=vertices,
        node_size=sizes,
        node_color=degrees,
        cmap="viridis",
        edgecolors="black",
        linewidths=0.8,
        ax=axis,
    )
    nx.draw_networkx_edges(induced, positions, width=1.5, alpha=0.75, ax=axis)
    nx.draw_networkx_labels(induced, positions, labels={node: str(node) for node in vertices}, font_size=8, ax=axis)
    figure.colorbar(nodes, ax=axis, label="full-graph degree")
    rank_label = (
        f"edge rank {int(eigenpair_row['edge_rank'])}"
        if eigenpair_row["region"] == "edge"
        else f"bulk offset {int(eigenpair_row['bulk_offset']):+d}"
    )
    axis.set_title(
        f"{config.model_label}: {rank_label}, E={float(eigenpair_row['eigenvalue_energy']):.4g}\n"
        f"node size = eigenvector mass; labels = original vertex IDs"
    )
    axis.axis("off")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_region_diagnostics(
    config: SingleGraphConfig,
    rows: Sequence[dict[str, object]],
    region: str,
    path: Path,
) -> None:
    selected = [row for row in rows if row["region"] == region]
    if region == "edge":
        selected.sort(key=lambda row: float(row["edge_rank"]))
        x = np.asarray([float(row["edge_rank"]) for row in selected])
        xlabel = "rank from upper edge (1 = Perron)"
    else:
        selected.sort(key=lambda row: float(row["bulk_offset"]))
        x = np.asarray([float(row["bulk_offset"]) for row in selected])
        xlabel = "bulk offset in valid atom-free rank"
    ratio = np.asarray([float(row["adjacent_gap_ratio"]) for row in selected])
    nipr = np.asarray([float(row["NIPR"]) for row in selected])
    mass = np.asarray([float(row["top10_mass"]) for row in selected])
    in_window_10 = np.asarray([bool(row["in_window_10"]) for row in selected])
    in_window_25 = np.asarray([bool(row["in_window_25"]) for row in selected])
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    for axis_position, axis in enumerate(axes.flat[:3]):
        axis.axvspan(
            float(np.min(x[in_window_25])) - 0.5,
            float(np.max(x[in_window_25])) + 0.5,
            color="#f6bd60",
            alpha=0.10,
            label="nested 25-window" if axis_position == 0 else None,
        )
        axis.axvspan(
            float(np.min(x[in_window_10])) - 0.5,
            float(np.max(x[in_window_10])) + 0.5,
            color="#d00000",
            alpha=0.10,
            label="nested 10-window" if axis_position == 0 else None,
        )
    axes[0, 0].plot(x, ratio, marker="o", ms=3)
    axes[0, 0].axhline(GOE_MEAN_GAP_RATIO, color="black", ls="--", label="GOE mean")
    axes[0, 0].axhline(POISSON_MEAN_GAP_RATIO, color="firebrick", ls=":", label="Poisson mean")
    axes[0, 0].set_ylabel("individual local gap ratio $r_i$")
    axes[0, 0].set_ylim(0.0, 1.0)
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)
    axes[0, 1].plot(x, nipr, marker="o", ms=3, color="tab:orange")
    axes[0, 1].set_ylabel(r"$N\,\mathrm{IPR}$")
    axes[0, 1].set_yscale("log")
    axes[1, 0].plot(x, mass, marker="o", ms=3, color="tab:green")
    axes[1, 0].set_ylabel("top-10 coordinate mass")
    axes[1, 0].set_ylim(0.0, 1.0)
    scatter = axes[1, 1].scatter(nipr, ratio, c=x, cmap="viridis", s=35)
    axes[1, 1].set_xlabel(r"$N\,\mathrm{IPR}$")
    axes[1, 1].set_ylabel("individual local gap ratio $r_i$")
    axes[1, 1].set_ylim(0.0, 1.0)
    figure.colorbar(scatter, ax=axes[1, 1], label=xlabel)
    for axis in axes.flat[:3]:
        axis.set_xlabel(xlabel)
        axis.grid(alpha=0.2)
    axes[1, 1].grid(alpha=0.2)
    figure.suptitle(f"{config.model_label}: {region} eigenpair microscope, single graph")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_edge_bulk_comparison(
    config: SingleGraphConfig,
    rows: Sequence[dict[str, object]],
    path: Path,
) -> None:
    metrics = [
        ("NIPR", r"$N\,\mathrm{IPR}$"),
        ("top10_mass", "top-10 mass"),
        ("top10_mean_degree", "top-10 mean degree"),
        ("induced_density", "induced density"),
        ("mean_pairwise_distance", "mean pairwise distance"),
        ("random_distance_zscore", "distance z-score vs random sets"),
    ]
    regions = ("edge", "bulk")
    colors = ("#d95f02", "#1b9e77")
    figure, axes = plt.subplots(2, 3, figsize=(12.0, 7.6))
    rng = np.random.default_rng(2718)
    for axis, (metric, label) in zip(axes.flat, metrics):
        values = [np.asarray([float(row[metric]) for row in rows if row["region"] == region], dtype=float) for region in regions]
        axis.boxplot(values, labels=regions, showfliers=False)
        for pos, (array, color) in enumerate(zip(values, colors), start=1):
            jitter = rng.normal(pos, 0.035, size=len(array))
            axis.scatter(jitter, array, s=12, alpha=0.55, color=color)
        axis.set_ylabel(label)
        axis.grid(alpha=0.15, axis="y")
    axes[0, 0].set_yscale("log")
    figure.suptitle(f"{config.model_label}: upper-edge top 50 vs central bulk 50, same graph")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def representative_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    edge_targets = {1, 2, 5, 10, 25, 50}
    bulk_rows = sorted((row for row in rows if row["region"] == "bulk"), key=lambda row: abs(float(row["bulk_offset"])))
    bulk_targets = {int(row["global_eigenvalue_index_0based"]) for row in bulk_rows[:6]}
    result = []
    for row in rows:
        keep = (
            row["region"] == "edge" and int(row["edge_rank"]) in edge_targets
        ) or (
            row["region"] == "bulk" and int(row["global_eigenvalue_index_0based"]) in bulk_targets
        )
        if keep:
            result.append(
                {
                    "region": row["region"],
                    "rank": row["edge_rank"] if row["region"] == "edge" else row["bulk_offset"],
                    "energy": row["eigenvalue_energy"],
                    "r": row["adjacent_gap_ratio"],
                    "NIPR": row["NIPR"],
                    "top10_mass": row["top10_mass"],
                    "degree_summary": f"max={int(row['top10_max_degree'])}; mean={float(row['top10_mean_degree']):.2f}",
                    "structure_summary": (
                        f"induced_edges={int(row['induced_edge_count'])}; "
                        f"components={int(row['induced_num_components'])}; "
                        f"mean_distance={float(row['mean_pairwise_distance']):.2f}"
                    ),
                }
            )
    return result


def write_figure_index(output_dir: Path) -> list[dict[str, object]]:
    rows = [
        {
            "figure": str(path.relative_to(output_dir)),
            "format": path.suffix.lstrip("."),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(output_dir.rglob("*.png"))
    ]
    if rows:
        write_csv(output_dir / "figure_index.csv", rows)
    return rows


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n")


def implementation_report_text(
    config: SingleGraphConfig,
    manifest: dict[str, object],
    eigenpair_rows: Sequence[dict[str, object]],
    figure_rows: Sequence[dict[str, object]],
) -> str:
    edge = [row for row in eigenpair_rows if row["region"] == "edge"]
    bulk = [row for row in eigenpair_rows if row["region"] == "bulk"]
    finite_edge_r = [float(row["adjacent_gap_ratio"]) for row in edge if math.isfinite(float(row["adjacent_gap_ratio"]))]
    finite_bulk_r = [float(row["adjacent_gap_ratio"]) for row in bulk if math.isfinite(float(row["adjacent_gap_ratio"]))]
    return f"""# Single-graph edge/bulk implementation report

- Analysis version: `{ANALYSIS_VERSION}`
- Model: `{config.model_key}`
- Canonical parameter: `{config.parameter_name}={config.parameter_value:g}`
- Graph: `N={config.n}`, `m={config.m}`, seed `{config.graph_seed}`, graph index `{config.graph_index}`
- Hoffman job: `{manifest.get('job_id') or 'local smoke test'}`
- Status: `{manifest.get('status')}`
- Output directory: `{config.output_dir}`

## Existing implementations reused

- `preferential_attachment.baseline.generate_preferential_attachment_graph`: canonical affine/nonlinear PA graph.
- `preferential_attachment.full_spectrum_scan`: established dense symmetric adjacency eigensolver constants and histogram centers.
- `preferential_attachment.mobility_edge_postprocess.build_gap_records`: zero-atom removal, atom-boundary segmentation, within-graph gaps, and middle-eigenvalue adjacent-gap ratios.
- Existing held-out `unfolding/crossfit_k5/fold_0.npz`: CDF trained without graph index 0.

## Exact conventions

Eigenvalues are sorted algebraically in ascending order and the same permutation is applied to eigenvector columns.  The saved residual audit verifies `||A v_i-lambda_i v_i||/max(1,|lambda_i|) < 1e-8` for the selected eigenpairs.

The upper-edge windows contain the 10, 25, and 50 largest algebraic eigenvalues and include the Perron eigenpair.  `edge_rank=1` is the Perron eigenvalue, whose two-sided adjacent-gap ratio is `NA`.

Exact zero eigenvalues (absolute tolerance `{config.zero_atol:g}`) are excluded from continuous level statistics and split the spectrum into atom-free graph-local segments; no gap or ratio bridges the removed atom.  The bulk-50 window is the feasible 50-level atom-free window, with a neighboring level on both sides, whose center is closest to the median valid spectral rank.  Bulk-10 and bulk-25 are nested around the same reference.

For an interior valid eigenvalue, `r_i=min(lambda_i-lambda_(i-1),lambda_(i+1)-lambda_i)/max(...)`.  Each `r_i` is an individual diagnostic, not a GOE/Poisson distribution test.

The local spacing panels retain the established twelve histogram centers and GOE/Poisson references.  Because one graph supplies only about ten gaps in a width-0.002 cell, the maximum cell width is `{config.histogram_window_width:g}` to improve support.  Windows are clipped symmetrically at 0 or 1, so their requested centers remain exact and edge cells are narrower.  Within a window, positive gaps from this one graph are divided by their mean; therefore the normalization denominator is a one-graph, one-window mean.  Unfolding uses the held-out ensemble CDF and does not fit the focal graph.

Top-coordinate support uses the ten largest `|v_i(j)|^2` values.  Vertex IDs are the eigenvector coordinate indices.  Distances are shortest-path distances in the full original graph, while induced-subgraph connectivity is recorded separately.  The random baseline uses `{config.random_baseline_samples}` fixed-seed random ten-vertex sets.

## Compact numerical index

- Edge-50 mean NIPR: `{np.mean([float(row['NIPR']) for row in edge]):.6g}`
- Bulk-50 mean NIPR: `{np.mean([float(row['NIPR']) for row in bulk]):.6g}`
- Edge-50 mean top-10 mass: `{np.mean([float(row['top10_mass']) for row in edge]):.6g}`
- Bulk-50 mean top-10 mass: `{np.mean([float(row['top10_mass']) for row in bulk]):.6g}`
- Edge finite local-r mean (descriptive only): `{np.mean(finite_edge_r):.6g}`
- Bulk finite local-r mean (descriptive only): `{np.mean(finite_bulk_r):.6g}`
- Generated figures: `{len(figure_rows)}`; see `figure_index.csv`.

No random-Schrodinger/diagonal-disorder perturbation is implemented in this batch.
"""


def run_single_graph_experiment(config: SingleGraphConfig) -> Path:
    start_time = time.time()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    manifest: dict[str, object] = {
        **config.metadata(),
        "status": "RUNNING",
        "started_at_utc": utc_now(),
        "host": platform.node(),
        "job_id": os.environ.get("JOB_ID"),
        "sge_task_id": os.environ.get("SGE_TASK_ID"),
        "git_commit": os.environ.get("PA_GIT_COMMIT") or None,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "networkx_version": nx.__version__,
        "matplotlib_version": matplotlib.__version__,
    }
    write_json(manifest_path, manifest)
    try:
        graph, generator_metadata = generate_fixed_graph(config)
        adjacency = graph_adjacency_csr(graph)
        degrees = np.asarray([graph.degree(node) for node in range(config.n)], dtype=np.int32)
        degree_ranks, degree_percentiles = degree_rank_and_percentile(degrees)
        edges = np.asarray(sorted((int(u), int(v)) for u, v in graph.edges()), dtype=np.int32)
        tables_dir = output_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            tables_dir / "graph_realization.npz",
            edge_u=edges[:, 0],
            edge_v=edges[:, 1],
            degrees=degrees,
            graph_seed=np.asarray(config.graph_seed, dtype=np.uint64),
            n=np.asarray(config.n, dtype=np.int32),
            m=np.asarray(config.m, dtype=np.int32),
        )
        write_json(tables_dir / "graph_metadata.json", generator_metadata)

        eig_start = time.time()
        eigenvalues, eigenvectors, sorting_permutation = sorted_dense_eigendecomposition(graph)
        eig_seconds = time.time() - eig_start
        valid_mask, segments = valid_atom_free_segments(eigenvalues, config.zero_atol)
        left_gap, right_gap, ratios = adjacent_gap_ratio_arrays(eigenvalues, segments)
        edge_windows = nested_edge_windows(eigenvalues)
        bulk_center, bulk_windows = nested_bulk_windows(eigenvalues, segments)
        selected_indices = np.asarray(
            sorted(set(edge_windows[50].tolist()) | set(bulk_windows[50].tolist())),
            dtype=int,
        )
        residual_indices = sorted(
            set(selected_indices.tolist())
            | {0, len(eigenvalues) // 4, len(eigenvalues) // 2, 3 * len(eigenvalues) // 4, len(eigenvalues) - 1}
        )
        residual_rows = sampled_eigenpair_residuals(
            adjacency, eigenvalues, eigenvectors, residual_indices
        )
        write_csv(tables_dir / "eigenpair_residual_audit.csv", residual_rows)
        ipr, nipr = localization_scalars(eigenvectors)

        perron_index = len(eigenvalues) - 1
        nonperron = eigenvalues[:perron_index]
        record = build_gap_records(
            nonperron,
            graph_index=config.graph_index,
            graph_seed=config.graph_seed,
            excluded_atom_energies=(0.0,),
            audited_atom_energies=(-1.0, 0.0, 1.0),
            atom_atol=config.zero_atol,
            zero_gap_atol=ZERO_GAP_ATOL,
        )
        grid, density, cdf, cdf_metadata = load_reference_cdf(
            config, record.continuous_eigenvalues
        )
        unfolding_dir = output_dir / "unfolding"
        unfolding_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            unfolding_dir / "reference_cdf_used.npz",
            grid=grid,
            density=density,
            cdf=cdf,
        )
        write_json(unfolding_dir / "reference_cdf_provenance.json", cdf_metadata)

        spectrum_dir = output_dir / "spectrum"
        plot_single_graph_density(
            config,
            eigenvalues,
            record.continuous_eigenvalues,
            spectrum_dir / "single_graph_eigenvalue_density.png",
        )
        histogram_dir = output_dir / "histograms"
        histogram_rows, raw_histograms, unfolded_histograms, unfolded_gaps = summarize_histogram_windows(
            config, record, grid, cdf, histogram_dir
        )
        plot_local_spacing_histograms(
            config,
            histogram_rows,
            raw_histograms,
            histogram_dir / "single_graph_local_raw_spacing_histograms.png",
            "raw",
        )
        plot_local_spacing_histograms(
            config,
            histogram_rows,
            unfolded_histograms,
            histogram_dir / "single_graph_local_unfolded_spacing_histograms.png",
            "unfolded",
        )
        plot_global_spacing_and_ratio(
            config,
            record,
            unfolded_gaps,
            histogram_dir / "single_graph_global_spacing_and_ratio.png",
        )
        plot_energy_nipr(
            config,
            eigenvalues,
            nipr,
            valid_mask,
            edge_windows,
            bulk_windows,
            output_dir / "energy_ipr" / "single_graph_localization_vs_energy.png",
        )

        valid_indices = np.flatnonzero(valid_mask)
        valid_position = {int(index): pos for pos, index in enumerate(valid_indices)}
        spectrum_rows: list[dict[str, object]] = []
        for index in range(config.n):
            spectrum_rows.append(
                {
                    "model": config.model_key,
                    config.parameter_name: config.parameter_value,
                    "graph_seed": config.graph_seed,
                    "global_eigenvalue_index_0based": index,
                    "global_eigenvalue_index_1based": index + 1,
                    "eigenvalue_energy": float(eigenvalues[index]),
                    "valid_for_continuous_statistics": bool(valid_mask[index]),
                    "is_zero_atom": not bool(valid_mask[index]),
                    "is_perron": index == perron_index,
                    "valid_spectral_rank_1based": valid_position[index] + 1 if index in valid_position else "",
                    "valid_spectral_quantile": (valid_position[index] + 0.5) / len(valid_indices) if index in valid_position else math.nan,
                    "left_gap": float(left_gap[index]),
                    "right_gap": float(right_gap[index]),
                    "adjacent_gap_ratio": float(ratios[index]),
                    "ipr": float(ipr[index]),
                    "NIPR": float(nipr[index]),
                    "edge_window_membership": window_memberships(index, edge_windows) if index in set(edge_windows[50]) else "",
                    "bulk_window_membership": window_memberships(index, bulk_windows) if index in set(bulk_windows[50]) else "",
                }
            )
        write_csv(spectrum_dir / "single_graph_eigenpairs.csv.gz", spectrum_rows)
        np.savez_compressed(
            spectrum_dir / "single_graph_spectral_arrays.npz",
            eigenvalues=eigenvalues,
            sorting_permutation=sorting_permutation,
            valid_mask=valid_mask,
            ipr=ipr,
            n_times_ipr=nipr,
            left_gap=left_gap,
            right_gap=right_gap,
            adjacent_gap_ratio=ratios,
            edge_top10_indices=edge_windows[10],
            edge_top25_indices=edge_windows[25],
            edge_top50_indices=edge_windows[50],
            bulk10_indices=bulk_windows[10],
            bulk25_indices=bulk_windows[25],
            bulk50_indices=bulk_windows[50],
            bulk_center_index=np.asarray(bulk_center, dtype=np.int32),
        )

        selected_vectors = np.asarray(eigenvectors[:, selected_indices], dtype=float).copy()
        np.savez_compressed(
            tables_dir / "selected_eigenvectors.npz",
            global_indices=selected_indices,
            eigenvalues=eigenvalues[selected_indices],
            eigenvectors=selected_vectors,
        )
        selected_rows = precompute_selected_support(
            config,
            graph,
            eigenvalues,
            selected_vectors,
            selected_indices,
            ipr,
            nipr,
            ratios,
            left_gap,
            right_gap,
            valid_mask,
            edge_windows,
            bulk_center,
            bulk_windows,
            degrees,
            degree_ranks,
            degree_percentiles,
        )
        del eigenvectors
        gc.collect()

        baseline_rows, distance_matrix, source_rows = random_structural_baseline(
            config, graph, adjacency, degrees, selected_rows
        )
        write_csv(tables_dir / "random_ten_vertex_baseline.csv.gz", baseline_rows)
        eigenpair_rows, support_rows = finalize_selected_rows(
            config,
            graph,
            degrees,
            selected_rows,
            baseline_rows,
            distance_matrix,
            source_rows,
        )
        del distance_matrix
        gc.collect()
        write_csv(tables_dir / "edge_bulk_eigenpairs.csv", eigenpair_rows)
        write_csv(tables_dir / "top10_support_vertices.csv", support_rows)

        support_by_pair: dict[tuple[str, int], list[dict[str, object]]] = {}
        for support in support_rows:
            key = (str(support["region"]), int(support["global_eigenvalue_index_0based"]))
            support_by_pair.setdefault(key, []).append(support)
        for row in eigenpair_rows:
            should_plot = (
                row["region"] == "edge" and int(row["edge_rank"]) <= 10
            ) or (
                row["region"] == "bulk" and bool(row["in_window_10"])
            )
            if not should_plot:
                continue
            index = int(row["global_eigenvalue_index_0based"])
            if row["region"] == "edge":
                name = f"edge_rank_{int(row['edge_rank']):03d}.png"
            else:
                offset = int(row["bulk_offset"])
                name = f"bulk_offset_{'p' if offset >= 0 else 'm'}{abs(offset):03d}.png"
            plot_induced_subgraph(
                config,
                graph,
                row,
                sorted(support_by_pair[(str(row["region"]), index)], key=lambda item: int(item["support_rank"])),
                output_dir / "induced_subgraphs" / str(row["region"]) / name,
            )

        plot_region_diagnostics(
            config, eigenpair_rows, "edge", output_dir / "edge" / "edge_rank_diagnostics.png"
        )
        plot_region_diagnostics(
            config, eigenpair_rows, "bulk", output_dir / "bulk" / "bulk_offset_diagnostics.png"
        )
        plot_edge_bulk_comparison(
            config,
            eigenpair_rows,
            output_dir / "edge_vs_bulk" / "edge_vs_bulk_comparison.png",
        )
        write_csv(
            output_dir / "edge_vs_bulk" / "representative_eigenpairs.csv",
            representative_rows(eigenpair_rows),
        )

        figure_rows = write_figure_index(output_dir)
        runtime = time.time() - start_time
        try:
            import resource

            peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        except Exception:
            peak_rss = -1
        manifest.update(
            {
                "status": "COMPLETE",
                "finished_at_utc": utc_now(),
                "runtime_seconds": runtime,
                "eigendecomposition_seconds": eig_seconds,
                "peak_rss_platform_units": peak_rss,
                "graph_generator_metadata": generator_metadata,
                "reference_cdf": cdf_metadata,
                "zero_eigenvalue_count": int(np.sum(~valid_mask)),
                "audited_atom_counts_nonperron": record.audited_atom_counts,
                "excluded_atom_counts_nonperron": record.excluded_atom_counts,
                "atom_free_segment_lengths_full_spectrum": [len(segment) for segment in segments],
                "edge_selection": "K largest algebraic eigenvalues including Perron; nested K=10,25,50",
                "bulk_selection": "nested K=10,25,50 in one atom-free segment; bulk-50 center closest feasible valid rank to global valid median, leaving one neighbor on each side",
                "bulk_center_global_index_0based": bulk_center,
                "bulk_center_energy": float(eigenvalues[bulk_center]),
                "gap_ratio_indexing": "r_i uses the left and right gaps around eigenvalue i within one atom-free segment; segment endpoints including Perron have NA",
                "random_baseline_samples": config.random_baseline_samples,
                "edge_eigenpair_rows": 50,
                "bulk_eigenpair_rows": 50,
                "support_vertex_rows": len(support_rows),
                "figure_count": len(figure_rows),
                "failed_or_incomplete_runs": [],
                "deviations": [
                    "The historical width-0.002 histogram cells use a maximum width %.3g for a better-populated single-graph diagnostic and are clipped symmetrically at spectral boundaries; centers, bins, unfolding, and GOE/Poisson references are preserved."
                    % config.histogram_window_width
                ],
            }
        )
        write_json(manifest_path, manifest)
        (output_dir / "IMPLEMENTATION_REPORT.md").write_text(
            implementation_report_text(config, manifest, eigenpair_rows, figure_rows)
        )
        return output_dir
    except Exception as error:
        manifest.update(
            {
                "status": "FAILED",
                "finished_at_utc": utc_now(),
                "runtime_seconds": time.time() - start_time,
                "failure_type": type(error).__name__,
                "failure_message": str(error),
            }
        )
        write_json(manifest_path, manifest)
        raise
