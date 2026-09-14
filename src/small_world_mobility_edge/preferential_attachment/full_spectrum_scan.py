"""Full-spectrum spacing and eigenvector scans for affine PA graphs.

The production run deliberately shares one dense eigendecomposition between
the eigenvalue and eigenvector diagnostics.  Graph-level partials are safe to
aggregate after a Hoffman array job finishes.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import os
import tempfile
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "small-world-mobility-edge-pa"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy
from scipy import linalg

from ..fixed_degree_configuration_baseline import rewire_graph_preserving_degrees
from .baseline import (
    MODEL_AFFINE_PA,
    MODEL_NONLINEAR_PA,
    attachment_offset_for_degree_ccdf_alpha,
    attachment_offset_for_ccdf_tau,
    generate_preferential_attachment_graph,
)


N = 5000
M = 10
NUM_GRAPHS = 500
TARGET_MEAN_DEGREE = 20.0
Q_MIN = 0.0
Q_MAX = 1.0
Q_WIDTH = 0.002
ZERO_EIGENVALUE_ATOL = 1e-8
EXACT_RESONANCE_ATOL = 1e-8
NUMERICAL_COORDINATE_ATOL = 1e-12
COORDINATE_DECAY_MAX_RANK = 200
TOP_R_VALUES = (10, 50, 100, 200)
MASS_LEVELS = (0.50, 0.80, 0.90, 0.95)
DENSITY_GRID_POINTS = 2001
BULK_DENSITY_HALF_WIDTH = 0.20
TAIL_DENSITY_HALF_WIDTH = 0.08
TAIL_DENSITY_QUANTILE = 0.95
GOE_MEAN_GAP_RATIO = 0.53590
POISSON_MEAN_GAP_RATIO = 2.0 * math.log(2.0) - 1.0
HISTOGRAM_QUANTILES = (
    0.005,
    0.010,
    0.025,
    0.050,
    0.100,
    0.250,
    0.500,
    0.750,
    0.900,
    0.950,
    0.990,
    0.999,
)
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = (
    REPO_ROOT
    / "outputs"
    / "preferential_attachment"
    / "affine_full_spectrum_q0_1_width_0p002"
)


@dataclass(frozen=True)
class PreferenceSpecification:
    attachment_tail_exponent: float
    graph_seed_base: int
    matched_chung_lu_label: str

    @property
    def delta(self) -> float:
        return attachment_offset_for_ccdf_tau(M, self.attachment_tail_exponent)


# Seed bases follow the corresponding Chung-Lu graph ensembles.  The PA
# generator has no expected-weight sequence seed.
PREFERENCE_SPECIFICATIONS = (
    PreferenceSpecification(1.8, 27282, "legacy Chung-Lu tau_0p8 / density exponent 1.8"),
    PreferenceSpecification(1.05, 27307, "Chung-Lu density exponent 1.05"),
    PreferenceSpecification(1.5, 27352, "Chung-Lu density exponent 1.5"),
)


def slug_number(value: float) -> str:
    return str(value).replace(".", "p")


def specification(alpha: float) -> PreferenceSpecification:
    for item in PREFERENCE_SPECIFICATIONS:
        if math.isclose(item.attachment_tail_exponent, alpha, abs_tol=1e-12):
            return item
    raise ValueError("unsupported attachment-tail exponent: %r" % alpha)


def parameter_output_dir(alpha: float) -> Path:
    return OUTPUT_ROOT / (
        "n_%d_m_%d_preference_tau_%s_avgdeg_19p978"
        % (N, M, slug_number(alpha))
    )


def make_seeds(num_graphs: int, base_seed: int) -> list[int]:
    rng = np.random.default_rng(int(base_seed))
    values = rng.integers(0, 2**32 - 1, size=num_graphs, dtype=np.uint32)
    return [int(value) for value in values]


def graph_seeds(alpha: float) -> list[int]:
    return make_seeds(NUM_GRAPHS, specification(alpha).graph_seed_base)


def quantile_bins() -> list[tuple[float, float]]:
    count = int(round((Q_MAX - Q_MIN) / Q_WIDTH))
    edges = np.linspace(Q_MIN, Q_MAX, count + 1)
    return [(float(edges[index]), float(edges[index + 1])) for index in range(count)]


QUANTILE_BINS = quantile_bins()


def quantile_index_bin(length: int, q_low: float, q_high: float) -> tuple[int, int]:
    start = int(math.floor(q_low * length))
    stop = int(math.ceil(q_high * length))
    start = min(max(start, 0), length - 1)
    stop = min(max(stop, start + 1), length)
    return start, stop


def write_csv(
    rows: Iterable[dict[str, object]], path: Path, fieldnames: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _mean_and_se(values: Iterable[float]) -> tuple[float, float, int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return math.nan, math.nan, 0
    mean = float(np.mean(array))
    se = float(np.std(array, ddof=1) / math.sqrt(len(array))) if len(array) > 1 else math.nan
    return mean, se, int(len(array))


def dense_adjacency_eigendecomposition(graph: nx.Graph) -> tuple[np.ndarray, np.ndarray]:
    matrix = nx.to_numpy_array(graph, nodelist=range(graph.number_of_nodes()), dtype=float)
    values, vectors = linalg.eigh(
        matrix,
        overwrite_a=True,
        check_finite=False,
        driver="evd",
    )
    return np.asarray(values, dtype=float), np.asarray(vectors, dtype=float)


def filter_continuous_nonperron(
    eigenvalues: np.ndarray, eigenvectors: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int, float]:
    perron_index = int(np.argmax(eigenvalues))
    perron = float(eigenvalues[perron_index])
    values = np.delete(eigenvalues, perron_index)
    vectors = np.delete(eigenvectors, perron_index, axis=1)
    zero_mask = np.isclose(values, 0.0, atol=ZERO_EIGENVALUE_ATOL, rtol=0.0)
    return values[~zero_mask], vectors[:, ~zero_mask], int(np.sum(zero_mask)), perron


def localization_rows(
    *,
    continuous_values: np.ndarray,
    continuous_vectors: np.ndarray,
    graph_index: int,
    graph_seed: int,
    alpha: float,
    diagnostics: dict[str, object],
) -> tuple[list[dict[str, object]], np.ndarray]:
    dimension = continuous_vectors.shape[0]
    length = len(continuous_values)
    decay = np.zeros((len(QUANTILE_BINS), COORDINATE_DECAY_MAX_RANK), dtype=float)
    rows: list[dict[str, object]] = []
    for bin_index, (q_low, q_high) in enumerate(QUANTILE_BINS):
        start, stop = quantile_index_bin(length, q_low, q_high)
        values = continuous_values[start:stop]
        vectors = continuous_vectors[:, start:stop]
        masses = np.square(vectors)
        sorted_masses = np.sort(masses, axis=0)[::-1, :]
        cumulative = np.cumsum(sorted_masses, axis=0)
        ipr = np.sum(np.square(masses), axis=0)
        participation = 1.0 / np.maximum(ipr, 1e-300)
        numerical_support = np.sum(np.abs(vectors) > NUMERICAL_COORDINATE_ATOL, axis=0)
        recorded_ranks = min(COORDINATE_DECAY_MAX_RANK, dimension)
        decay[bin_index, :recorded_ranks] = np.mean(
            sorted_masses[:recorded_ranks], axis=1
        )
        row: dict[str, object] = {
            "graph_index": graph_index,
            "graph_seed": graph_seed,
            "attachment_tail_exponent": alpha,
            "attachment_offset_delta": specification(alpha).delta,
            "quantile": 0.5 * (q_low + q_high),
            "quantile_low": q_low,
            "quantile_high": q_high,
            "number_of_eigenvectors": stop - start,
            "E_low": float(values[0]),
            "E_high": float(values[-1]),
            "E_center": float(np.median(values)),
            "analyzed_dimension": dimension,
            "continuous_nonperron_dimension": length,
            "zero_eigenvalue_count": diagnostics["zero_eigenvalue_count"],
            "zero_eigenvalue_fraction": diagnostics["zero_eigenvalue_fraction"],
            "perron_eigenvalue": diagnostics["perron_eigenvalue"],
            "realized_mean_degree": diagnostics["realized_mean_degree"],
            "maximum_degree": diagnostics["maximum_degree"],
            "mean_n_times_ipr": float(dimension * np.mean(ipr)),
            "median_n_times_ipr": float(dimension * np.median(ipr)),
            "mean_participation_fraction": float(np.mean(participation) / dimension),
            "mean_numerical_support_fraction": float(np.mean(numerical_support) / dimension),
        }
        for top_r in TOP_R_VALUES:
            row["mean_top_%d_mass" % top_r] = float(
                np.mean(cumulative[min(top_r, dimension) - 1])
            )
        for level in MASS_LEVELS:
            reached = cumulative >= level - 1e-12
            radii = np.argmax(reached, axis=0) + 1
            key = "r_%d" % int(round(100 * level))
            row["mean_%s_fraction" % key] = float(np.mean(radii) / dimension)
        rows.append(row)
    return rows, decay


def graph_row_fieldnames() -> list[str]:
    fields = [
        "graph_index",
        "graph_seed",
        "attachment_tail_exponent",
        "attachment_offset_delta",
        "quantile",
        "quantile_low",
        "quantile_high",
        "number_of_eigenvectors",
        "E_low",
        "E_high",
        "E_center",
        "analyzed_dimension",
        "continuous_nonperron_dimension",
        "zero_eigenvalue_count",
        "zero_eigenvalue_fraction",
        "perron_eigenvalue",
        "realized_mean_degree",
        "maximum_degree",
        "mean_n_times_ipr",
        "median_n_times_ipr",
        "mean_participation_fraction",
        "mean_numerical_support_fraction",
    ]
    fields.extend("mean_top_%d_mass" % top_r for top_r in TOP_R_VALUES)
    fields.extend(
        "mean_r_%d_fraction" % int(round(100 * level)) for level in MASS_LEVELS
    )
    return fields


def fragment_fieldnames() -> list[str]:
    return [
        "graph_index",
        "graph_seed",
        "attachment_tail_exponent",
        "attachment_offset_delta",
        "num_nodes",
        "num_edges",
        "realized_mean_degree",
        "minimum_degree",
        "maximum_degree",
        "num_components",
        "largest_component_size",
        "num_isolated_vertices",
        "num_isolated_k2_components",
        "zero_eigenvalue_count",
        "zero_eigenvalue_fraction",
        "minus_one_multiplicity",
        "plus_one_multiplicity",
        "perron_eigenvalue",
    ]


def analyze_graph(
    *, graph_index: int, graph_seed: int, alpha: float
) -> tuple[list[dict[str, object]], np.ndarray, dict[str, object], np.ndarray]:
    graph, generator = generate_preferential_attachment_graph(
        n=N,
        m=M,
        seed=graph_seed,
        model=MODEL_AFFINE_PA,
        attachment_ccdf_tau=alpha,
    )
    eigenvalues, eigenvectors = dense_adjacency_eigendecomposition(graph)
    nonperron_values = eigenvalues[:-1].copy()
    continuous_values, continuous_vectors, zero_count, perron = filter_continuous_nonperron(
        eigenvalues, eigenvectors
    )
    isolated_k2_count = sum(
        1
        for component in nx.connected_components(graph)
        if len(component) == 2 and graph.subgraph(component).number_of_edges() == 1
    )
    fragment = {
        "graph_index": graph_index,
        "graph_seed": graph_seed,
        "attachment_tail_exponent": alpha,
        "attachment_offset_delta": generator.attachment_offset,
        "num_edges": generator.num_edges,
        "realized_mean_degree": generator.realized_mean_degree,
        "minimum_degree": generator.min_realized_degree,
        "maximum_degree": generator.max_realized_degree,
        "num_components": generator.num_components,
        "largest_component_size": generator.largest_component_size,
        "num_isolated_vertices": generator.num_isolated_vertices,
        "num_isolated_k2_components": isolated_k2_count,
        "zero_eigenvalue_count": zero_count,
        "zero_eigenvalue_fraction": zero_count / max(N - 1, 1),
        "minus_one_multiplicity": int(
            np.sum(np.isclose(nonperron_values, -1.0, atol=EXACT_RESONANCE_ATOL, rtol=0.0))
        ),
        "plus_one_multiplicity": int(
            np.sum(np.isclose(nonperron_values, 1.0, atol=EXACT_RESONANCE_ATOL, rtol=0.0))
        ),
        "perron_eigenvalue": perron,
    }
    rows, decay = localization_rows(
        continuous_values=continuous_values,
        continuous_vectors=continuous_vectors,
        graph_index=graph_index,
        graph_seed=graph_seed,
        alpha=alpha,
        diagnostics=fragment,
    )
    return rows, decay, fragment, nonperron_values


def run_chunk(alpha: float, start: int, stop: int) -> Path:
    if not 0 <= start < stop <= NUM_GRAPHS:
        raise ValueError("invalid graph range")
    seeds = graph_seeds(alpha)
    output_dir = parameter_output_dir(alpha)
    partial_dir = output_dir / "partials"
    partial_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    fragments: list[dict[str, object]] = []
    decay_sum = np.zeros((len(QUANTILE_BINS), COORDINATE_DECAY_MAX_RANK), dtype=float)
    decay_sum_squares = np.zeros_like(decay_sum)
    eigenvalue_samples = np.zeros((stop - start, N - 1), dtype=float)
    for local_index, graph_index in enumerate(range(start, stop)):
        graph_rows, graph_decay, fragment, eigenvalues = analyze_graph(
            graph_index=graph_index,
            graph_seed=seeds[graph_index],
            alpha=alpha,
        )
        rows.extend(graph_rows)
        fragments.append(fragment)
        decay_sum += graph_decay
        decay_sum_squares += np.square(graph_decay)
        eigenvalue_samples[local_index] = eigenvalues
        print(
            "alpha=%g finished graph_index=%d seed=%d" % (
                alpha,
                graph_index,
                seeds[graph_index],
            ),
            flush=True,
        )
    suffix = "%04d_%04d" % (start, stop)
    write_csv(
        rows,
        partial_dir / ("eigenvector_graphs_%s.csv.gz" % suffix),
        graph_row_fieldnames(),
    )
    write_csv(
        fragments,
        partial_dir / ("fragment_graphs_%s.csv.gz" % suffix),
        fragment_fieldnames(),
    )
    np.savez_compressed(
        partial_dir / ("spectral_arrays_%s.npz" % suffix),
        graph_indices=np.arange(start, stop, dtype=int),
        graph_seeds=np.asarray(seeds[start:stop], dtype=np.uint64),
        nonperron_eigenvalues=eigenvalue_samples,
        coordinate_decay_sum=decay_sum,
        coordinate_decay_sum_squares=decay_sum_squares,
        graph_count=stop - start,
    )
    return partial_dir


def _read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def summary_metric_fields() -> list[str]:
    fields = [
        "E_center",
        "continuous_nonperron_dimension",
        "zero_eigenvalue_fraction",
        "realized_mean_degree",
        "maximum_degree",
        "mean_n_times_ipr",
        "median_n_times_ipr",
        "mean_participation_fraction",
        "mean_numerical_support_fraction",
    ]
    fields.extend("mean_top_%d_mass" % top_r for top_r in TOP_R_VALUES)
    fields.extend(
        "mean_r_%d_fraction" % int(round(100 * level)) for level in MASS_LEVELS
    )
    return fields


def aggregate_eigenvectors(
    paths: list[Path], alpha: float, output_dir: Path
) -> list[dict[str, object]]:
    metrics = summary_metric_fields()
    count = np.zeros(len(QUANTILE_BINS), dtype=int)
    vector_count = np.zeros(len(QUANTILE_BINS), dtype=int)
    sums = {field: np.zeros(len(QUANTILE_BINS), dtype=float) for field in metrics}
    squares = {field: np.zeros(len(QUANTILE_BINS), dtype=float) for field in metrics}
    by_graph = output_dir / "eigenvector_percentile_scan_by_graph.csv.gz"
    with gzip.open(by_graph, "wt", newline="") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=graph_row_fieldnames())
        writer.writeheader()
        for path in paths:
            for row in _read_csv(path):
                writer.writerow(row)
                index = int(round((float(row["quantile_low"]) - Q_MIN) / Q_WIDTH))
                count[index] += 1
                vector_count[index] += int(row["number_of_eigenvectors"])
                for field in metrics:
                    value = float(row[field])
                    sums[field][index] += value
                    squares[field][index] += value * value
    rows: list[dict[str, object]] = []
    for index, (q_low, q_high) in enumerate(QUANTILE_BINS):
        if count[index] != NUM_GRAPHS:
            raise ValueError("quantile bin %d has %d graphs" % (index, count[index]))
        row: dict[str, object] = {
            "attachment_tail_exponent": alpha,
            "attachment_offset_delta": specification(alpha).delta,
            "quantile": 0.5 * (q_low + q_high),
            "quantile_low": q_low,
            "quantile_high": q_high,
            "number_of_graphs": int(count[index]),
            "number_of_eigenvectors": int(vector_count[index]),
        }
        for field in metrics:
            mean = sums[field][index] / count[index]
            variance = max(
                (squares[field][index] - sums[field][index] ** 2 / count[index])
                / (count[index] - 1),
                0.0,
            )
            row[field] = float(mean)
            row[field + "_standard_error"] = float(math.sqrt(variance / count[index]))
        rows.append(row)
    fields = list(rows[0].keys())
    write_csv(rows, output_dir / "eigenvector_percentile_scan_summary.csv", fields)
    return rows


def aggregate_coordinate_decay(
    paths: list[Path], output_dir: Path
) -> np.ndarray:
    total = np.zeros((len(QUANTILE_BINS), COORDINATE_DECAY_MAX_RANK), dtype=float)
    total_squares = np.zeros_like(total)
    graph_count = 0
    for path in paths:
        with np.load(path) as data:
            total += data["coordinate_decay_sum"]
            total_squares += data["coordinate_decay_sum_squares"]
            graph_count += int(data["graph_count"])
    if graph_count != NUM_GRAPHS:
        raise ValueError("coordinate decay has %d graphs" % graph_count)
    mean = total / graph_count
    variance = np.maximum(
        (total_squares - np.square(total) / graph_count) / (graph_count - 1), 0.0
    )
    se = np.sqrt(variance / graph_count)
    rows: list[dict[str, object]] = []
    for bin_index, (q_low, q_high) in enumerate(QUANTILE_BINS):
        cumulative = np.cumsum(mean[bin_index])
        for rank in range(COORDINATE_DECAY_MAX_RANK):
            rows.append(
                {
                    "quantile": 0.5 * (q_low + q_high),
                    "quantile_low": q_low,
                    "quantile_high": q_high,
                    "rank": rank + 1,
                    "number_of_graphs": graph_count,
                    "mean_coordinate_mass": float(mean[bin_index, rank]),
                    "coordinate_mass_standard_error": float(se[bin_index, rank]),
                    "mean_cumulative_mass": float(cumulative[rank]),
                }
            )
    write_csv(rows, output_dir / "eigenvector_coordinate_decay_summary.csv", list(rows[0]))
    return mean


def collect_eigenvalue_samples(paths: list[Path]) -> tuple[list[np.ndarray], list[int]]:
    samples: list[np.ndarray] = []
    indices: list[int] = []
    for path in paths:
        with np.load(path) as data:
            matrix = np.asarray(data["nonperron_eigenvalues"], dtype=float)
            graph_indices = np.asarray(data["graph_indices"], dtype=int)
        for graph_index, values in zip(graph_indices, matrix):
            samples.append(np.sort(values))
            indices.append(int(graph_index))
    order = np.argsort(indices)
    ordered_samples = [samples[int(index)] for index in order]
    ordered_indices = [indices[int(index)] for index in order]
    if ordered_indices != list(range(NUM_GRAPHS)):
        raise ValueError("eigenvalue partials do not cover graph indices 0..499")
    return ordered_samples, ordered_indices


def estimate_density(samples: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    continuous = [
        values[~np.isclose(values, 0.0, atol=ZERO_EIGENVALUE_ATOL, rtol=0.0)]
        for values in samples
    ]
    pooled = np.sort(np.concatenate(continuous))
    low = float(pooled[0])
    high = float(pooled[-1])
    padding = 0.05 * (high - low)
    grid = np.linspace(low - padding, high + padding, DENSITY_GRID_POINTS)
    empirical_q = np.searchsorted(pooled, grid, side="right") / len(pooled)
    tail = (empirical_q <= 1.0 - TAIL_DENSITY_QUANTILE) | (
        empirical_q >= TAIL_DENSITY_QUANTILE
    )
    widths = np.where(tail, TAIL_DENSITY_HALF_WIDTH, BULK_DENSITY_HALF_WIDTH)
    left = np.searchsorted(pooled, grid - widths, side="left")
    right = np.searchsorted(pooled, grid + widths, side="right")
    mean_dimension = float(np.mean([len(values) for values in continuous]))
    density = (right - left) / (len(samples) * mean_dimension * 2.0 * widths)
    density = np.maximum(density, 0.0)
    increments = 0.5 * (density[:-1] + density[1:]) * np.diff(grid)
    cdf = np.concatenate(([0.0], np.cumsum(increments)))
    if cdf[-1] <= 0:
        raise ValueError("density has no mass")
    density /= cdf[-1]
    cdf /= cdf[-1]
    return grid, density, cdf


def _gap_ratios(spacings: np.ndarray, positive_only: bool) -> np.ndarray:
    left = spacings[:-1]
    right = spacings[1:]
    maximum = np.maximum(left, right)
    mask = maximum > 1e-14
    if positive_only:
        mask &= (left > 1e-14) & (right > 1e-14)
    return np.minimum(left[mask], right[mask]) / maximum[mask]


def _ks_distance(values: np.ndarray, distribution: str) -> float:
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
        raise ValueError("unknown spacing distribution")
    return float(max(np.max(np.abs(empirical_high - reference)), np.max(np.abs(empirical_low - reference))))


def aggregate_spacing(
    samples: list[np.ndarray], grid: np.ndarray, cdf: np.ndarray, output_dir: Path
) -> tuple[list[dict[str, object]], dict[int, np.ndarray], np.ndarray]:
    continuous = [
        values[~np.isclose(values, 0.0, atol=ZERO_EIGENVALUE_ATOL, rtol=0.0)]
        for values in samples
    ]
    histogram_indices = {
        int(np.argmin(np.abs(np.asarray([0.5 * (a + b) for a, b in QUANTILE_BINS]) - q)))
        for q in HISTOGRAM_QUANTILES
    }
    selected_histograms: dict[int, np.ndarray] = {}
    heat_bins = np.linspace(0.0, 3.5, 48)
    heat = np.zeros((len(QUANTILE_BINS), len(heat_bins) - 1), dtype=float)
    rows: list[dict[str, object]] = []
    for bin_index, (q_low, q_high) in enumerate(QUANTILE_BINS):
        energy_low, energy_high = np.interp([q_low, q_high], cdf, grid)
        graph_ratios: list[float] = []
        graph_positive_ratios: list[float] = []
        all_spacings: list[np.ndarray] = []
        all_ratios: list[np.ndarray] = []
        all_positive_ratios: list[np.ndarray] = []
        raw_gap_count = 0
        zero_gap_count = 0
        contributing_graphs = 0
        for values in continuous:
            left = int(np.searchsorted(values, energy_low, side="left"))
            side = "right" if bin_index == len(QUANTILE_BINS) - 1 else "left"
            right = int(np.searchsorted(values, energy_high, side=side))
            selected = values[left:right]
            if len(selected) < 3:
                continue
            unfolded = len(values) * np.interp(selected, grid, cdf)
            spacings = np.maximum(np.diff(unfolded), 0.0)
            ratios = _gap_ratios(spacings, positive_only=False)
            positive_ratios = _gap_ratios(spacings, positive_only=True)
            if len(ratios):
                graph_ratios.append(float(np.mean(ratios)))
            if len(positive_ratios):
                graph_positive_ratios.append(float(np.mean(positive_ratios)))
            all_spacings.append(spacings)
            all_ratios.append(ratios)
            all_positive_ratios.append(positive_ratios)
            raw_gap_count += len(spacings)
            zero_gap_count += int(np.sum(spacings <= 1e-14))
            contributing_graphs += 1
        spacings = np.concatenate(all_spacings) if all_spacings else np.array([], dtype=float)
        positive_spacings = spacings[spacings > 1e-14]
        normalized_positive = (
            positive_spacings / np.mean(positive_spacings)
            if len(positive_spacings) and np.mean(positive_spacings) > 0
            else np.array([], dtype=float)
        )
        ratios = np.concatenate(all_ratios) if all_ratios else np.array([], dtype=float)
        positive_ratios = (
            np.concatenate(all_positive_ratios)
            if all_positive_ratios
            else np.array([], dtype=float)
        )
        mean_ratio, ratio_se, ratio_graphs = _mean_and_se(graph_ratios)
        mean_positive_ratio, positive_ratio_se, positive_ratio_graphs = _mean_and_se(
            graph_positive_ratios
        )
        row = {
            "quantile": 0.5 * (q_low + q_high),
            "quantile_low": q_low,
            "quantile_high": q_high,
            "E_low": float(energy_low),
            "E_high": float(energy_high),
            "E_center": float(np.interp(0.5 * (q_low + q_high), cdf, grid)),
            "number_of_graphs": contributing_graphs,
            "graphs_with_gap_ratios": ratio_graphs,
            "number_of_gaps": raw_gap_count,
            "number_of_positive_gaps": int(len(positive_spacings)),
            "number_of_gap_ratios": int(len(ratios)),
            "number_of_positive_positive_gap_ratios": int(len(positive_ratios)),
            "zero_gap_fraction": zero_gap_count / max(raw_gap_count, 1),
            "graph_equal_mean_gap_ratio": mean_ratio,
            "gap_ratio_standard_error": ratio_se,
            "absolute_distance_mean_r_to_goe": abs(mean_ratio - GOE_MEAN_GAP_RATIO),
            "absolute_distance_mean_r_to_poisson": abs(mean_ratio - POISSON_MEAN_GAP_RATIO),
            "graph_equal_mean_positive_positive_gap_ratio": mean_positive_ratio,
            "positive_positive_gap_ratio_standard_error": positive_ratio_se,
            "positive_positive_ratio_graphs": positive_ratio_graphs,
            "pooled_mean_gap_ratio": float(np.mean(ratios)) if len(ratios) else math.nan,
            "pooled_mean_positive_positive_gap_ratio": (
                float(np.mean(positive_ratios)) if len(positive_ratios) else math.nan
            ),
            "KS_positive_spacings_to_GOE": _ks_distance(normalized_positive, "goe"),
            "KS_positive_spacings_to_Poisson": _ks_distance(normalized_positive, "poisson"),
            "small_positive_spacing_probability": (
                float(np.mean(normalized_positive < 0.5))
                if len(normalized_positive)
                else math.nan
            ),
        }
        rows.append(row)
        if len(normalized_positive):
            counts, _ = np.histogram(normalized_positive, bins=heat_bins)
            if np.sum(counts) > 0:
                heat[bin_index] = counts / (np.sum(counts) * np.diff(heat_bins))
        if bin_index in histogram_indices:
            selected_histograms[bin_index] = normalized_positive
    write_csv(rows, output_dir / "eigenvalue_spacing_summary.csv", list(rows[0]))
    np.savez_compressed(
        output_dir / "eigenvalue_spacing_histogram_data.npz",
        spacing_bin_edges=heat_bins,
        spacing_density_by_quantile=heat,
        quantile_centers=np.asarray([row["quantile"] for row in rows]),
    )
    return rows, selected_histograms, heat


def aggregate_fragments(paths: list[Path], alpha: float, output_dir: Path) -> dict[str, object]:
    rows = [row for path in paths for row in _read_csv(path)]
    if len(rows) != NUM_GRAPHS:
        raise ValueError("fragment diagnostics have %d graphs" % len(rows))
    write_csv(rows, output_dir / "fragment_multiplicity_by_graph.csv", fragment_fieldnames())
    metrics = [
        "realized_mean_degree",
        "minimum_degree",
        "maximum_degree",
        "num_components",
        "largest_component_size",
        "num_isolated_vertices",
        "num_isolated_k2_components",
        "zero_eigenvalue_count",
        "minus_one_multiplicity",
        "plus_one_multiplicity",
        "perron_eigenvalue",
    ]
    summary: dict[str, object] = {
        "attachment_tail_exponent": alpha,
        "attachment_offset_delta": specification(alpha).delta,
        "number_of_graphs": len(rows),
        "graphs_with_more_than_one_component": sum(int(row["num_components"]) > 1 for row in rows),
        "graphs_with_isolated_k2": sum(int(row["num_isolated_k2_components"]) > 0 for row in rows),
        "graphs_with_minus_one_multiplicity": sum(int(row["minus_one_multiplicity"]) > 0 for row in rows),
        "graphs_with_plus_one_multiplicity": sum(int(row["plus_one_multiplicity"]) > 0 for row in rows),
    }
    for metric in metrics:
        mean, se, _ = _mean_and_se(float(row[metric]) for row in rows)
        summary["mean_" + metric] = mean
        summary["se_" + metric] = se
        summary["max_" + metric] = max(float(row[metric]) for row in rows)
    write_csv([summary], output_dir / "fragment_multiplicity_summary.csv", list(summary))
    return summary


def plot_eigenvector_summary(rows: list[dict[str, object]], alpha: float, output_dir: Path) -> None:
    q = np.asarray([float(row["quantile"]) for row in rows])
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.8), sharex=True)
    y = np.asarray([float(row["mean_n_times_ipr"]) for row in rows])
    se = np.asarray([float(row["mean_n_times_ipr_standard_error"]) for row in rows])
    axes[0, 0].plot(q, y, color="black")
    axes[0, 0].fill_between(q, y - se, y + se, color="0.7", alpha=0.35)
    axes[0, 0].axhline(3.0 * N / (N + 2.0), color="firebrick", ls="--", label="GOE/Haar")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel(r"mean $N\times\mathrm{IPR}$")
    axes[0, 0].set_title("Inverse participation ratio")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].plot(q, [row["mean_participation_fraction"] for row in rows], label=r"PR/$N$")
    axes[0, 1].plot(q, [row["mean_numerical_support_fraction"] for row in rows], label="numerical nonzero fraction")
    axes[0, 1].axhline(1.0 / 3.0, color="firebrick", ls="--", label=r"GOE PR/$N$")
    axes[0, 1].set_title("Participation ratio and numerical support")
    axes[0, 1].set_ylabel(r"fraction of $N$")
    axes[0, 1].legend(frameon=False)
    for top_r in TOP_R_VALUES:
        axes[1, 0].plot(q, [row["mean_top_%d_mass" % top_r] for row in rows], label="top %d" % top_r)
    axes[1, 0].set_title("Top-r mass")
    axes[1, 0].set_ylabel("cumulative coordinate mass")
    axes[1, 0].legend(frameon=False, ncol=2)
    for level in MASS_LEVELS:
        percentage = int(round(100 * level))
        axes[1, 1].plot(q, [row["mean_r_%d_fraction" % percentage] for row in rows], label=r"$r_{%d}/N$" % percentage)
    axes[1, 1].set_title("Coordinates needed for target mass")
    axes[1, 1].set_ylabel("minimum coordinate fraction")
    axes[1, 1].legend(frameon=False, ncol=2)
    for axis in axes.flat:
        axis.set_xlabel("continuous non-Perron spectral quantile")
        axis.grid(alpha=0.2)
    figure.suptitle(
        r"Affine preferential attachment: $N=5000$, $m=10$, $G=500$, "
        r"attachment-tail exponent $\alpha=%g$, $\delta=%g$, $\Delta q=0.002$"
        % (alpha, specification(alpha).delta)
    )
    figure.tight_layout()
    figure.savefig(output_dir / "eigenvector_percentile_scan_summary.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_coordinate_heatmap(mean_decay: np.ndarray, alpha: float, output_dir: Path) -> None:
    cumulative = np.cumsum(mean_decay, axis=1)
    figure, axis = plt.subplots(figsize=(10.5, 6.0))
    image = axis.imshow(
        cumulative,
        origin="lower",
        aspect="auto",
        extent=(1, COORDINATE_DECAY_MAX_RANK, Q_MIN, Q_MAX),
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("mean cumulative top-r mass")
    axis.set_xlabel("top-r coordinate count")
    axis.set_ylabel("continuous non-Perron spectral quantile")
    axis.set_title(
        r"Affine PA coordinate-mass concentration: $N=5000$, $m=10$, "
        r"$G=500$, $\alpha=%g$, $\delta=%g$" % (alpha, specification(alpha).delta)
    )
    figure.tight_layout()
    figure.savefig(output_dir / "eigenvector_coordinate_mass_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_density(grid: np.ndarray, density: np.ndarray, alpha: float, output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(9.0, 5.2))
    axis.plot(grid, density, color="black")
    axis.set_xlabel("energy E")
    axis.set_ylabel(r"estimated density $\widehat\rho(E)$")
    axis.set_title(r"Affine PA global non-Perron spectral density, $\alpha=%g$" % alpha)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "global_empirical_spectral_density.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_spacing_summary(rows: list[dict[str, object]], alpha: float, output_dir: Path) -> None:
    q = np.asarray([float(row["quantile"]) for row in rows])
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), sharex=True)
    mean_r = np.asarray([float(row["graph_equal_mean_gap_ratio"]) for row in rows])
    se = np.asarray([float(row["gap_ratio_standard_error"]) for row in rows])
    axes[0, 0].plot(q, mean_r, color="black")
    axes[0, 0].fill_between(q, mean_r - se, mean_r + se, color="0.7", alpha=0.35)
    axes[0, 0].axhline(GOE_MEAN_GAP_RATIO, color="tab:blue", ls="--", label="GOE")
    axes[0, 0].axhline(POISSON_MEAN_GAP_RATIO, color="tab:orange", ls="--", label="Poisson")
    axes[0, 0].set_ylabel(r"graph-equal mean $r$")
    axes[0, 0].set_title("Adjacent-gap ratio")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].plot(q, [row["absolute_distance_mean_r_to_goe"] for row in rows], label="distance to GOE")
    axes[0, 1].plot(q, [row["absolute_distance_mean_r_to_poisson"] for row in rows], label="distance to Poisson")
    axes[0, 1].set_ylabel("absolute distance")
    axes[0, 1].set_title("Distance between mean r and theory")
    axes[0, 1].legend(frameon=False)
    axes[1, 0].plot(q, [row["KS_positive_spacings_to_GOE"] for row in rows], label="KS to GOE")
    axes[1, 0].plot(q, [row["KS_positive_spacings_to_Poisson"] for row in rows], label="KS to Poisson")
    axes[1, 0].set_ylabel("KS distance")
    axes[1, 0].set_title("Positive-spacing distribution")
    axes[1, 0].legend(frameon=False)
    axes[1, 1].plot(q, [row["zero_gap_fraction"] for row in rows], color="tab:red", label="zero-gap fraction")
    axes[1, 1].plot(q, [row["small_positive_spacing_probability"] for row in rows], color="tab:green", label=r"$P(s<0.5\mid s>0)$")
    axes[1, 1].set_ylabel("fraction")
    axes[1, 1].set_title("Small and zero spacings")
    axes[1, 1].legend(frameon=False)
    for axis in axes.flat:
        axis.set_xlabel("ensemble spectral quantile")
        axis.grid(alpha=0.2)
    figure.suptitle(r"Affine PA local eigenvalue statistics: $N=5000$, $m=10$, $G=500$, $\alpha=%g$" % alpha)
    figure.tight_layout()
    figure.savefig(output_dir / "eigenvalue_spacing_summary.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_spacing_histograms(
    rows: list[dict[str, object]], histograms: dict[int, np.ndarray], alpha: float, output_dir: Path
) -> None:
    figure, axes = plt.subplots(3, 4, figsize=(14.5, 10.0), sharex=True, sharey=True)
    x = np.linspace(0.0, 3.5, 500)
    goe = 0.5 * math.pi * x * np.exp(-math.pi * x * x / 4.0)
    poisson = np.exp(-x)
    for axis, bin_index in zip(axes.flat, sorted(histograms)):
        values = histograms[bin_index]
        row = rows[bin_index]
        displayed_values = values[(values >= 0.0) & (values <= 3.5)]
        if len(displayed_values):
            axis.hist(displayed_values, bins=np.linspace(0.0, 3.5, 30), density=True, color="#52b788", alpha=0.75)
        else:
            axis.text(0.5, 0.5, "no positive spacings\nin displayed range", ha="center", va="center", transform=axis.transAxes)
        axis.plot(x, goe, color="black", lw=1.2, label="GOE")
        axis.plot(x, poisson, color="firebrick", ls="--", lw=1.2, label="Poisson")
        axis.set_title("q=%.3f, E=[%.2f, %.2f]\n%d positive gaps" % (row["quantile"], row["E_low"], row["E_high"], row["number_of_positive_gaps"]))
        axis.grid(alpha=0.15)
    axes[0, 0].legend(frameon=False)
    for axis in axes[-1]:
        axis.set_xlabel("unfolded positive spacing s")
    for axis in axes[:, 0]:
        axis.set_ylabel("density")
    figure.suptitle(r"Affine PA local-spacing histograms, $\alpha=%g$, $N=5000$, $m=10$, $G=500$" % alpha)
    figure.tight_layout()
    figure.savefig(output_dir / "eigenvalue_spacing_histograms.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_spacing_heatmap(rows: list[dict[str, object]], heat: np.ndarray, alpha: float, output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(10.5, 6.0))
    image = axis.imshow(
        heat,
        origin="lower",
        aspect="auto",
        extent=(0.0, 3.5, Q_MIN, Q_MAX),
        cmap="GnBu_r",
    )
    figure.colorbar(image, ax=axis, label="spacing density")
    axis.set_xlabel("unfolded positive spacing s")
    axis.set_ylabel("ensemble spectral quantile")
    axis.set_title(r"Affine PA spacing-density heatmap, $\alpha=%g$" % alpha)
    figure.tight_layout()
    figure.savefig(output_dir / "eigenvalue_spacing_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def aggregate_parameter(alpha: float) -> Path:
    output_dir = parameter_output_dir(alpha)
    partial_dir = output_dir / "partials"
    graph_paths = sorted(partial_dir.glob("eigenvector_graphs_*.csv.gz"))
    fragment_paths = sorted(partial_dir.glob("fragment_graphs_*.csv.gz"))
    array_paths = sorted(partial_dir.glob("spectral_arrays_*.npz"))
    expected_chunks = 100
    if not (len(graph_paths) == len(fragment_paths) == len(array_paths) == expected_chunks):
        raise FileNotFoundError(
            "expected %d partials of each kind, found %d/%d/%d"
            % (expected_chunks, len(graph_paths), len(fragment_paths), len(array_paths))
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    eigenvector_rows = aggregate_eigenvectors(graph_paths, alpha, output_dir)
    mean_decay = aggregate_coordinate_decay(array_paths, output_dir)
    samples, _ = collect_eigenvalue_samples(array_paths)
    grid, density, cdf = estimate_density(samples)
    np.savez_compressed(output_dir / "empirical_density_cdf.npz", grid=grid, density=density, cdf=cdf)
    spacing_rows, histograms, spacing_heat = aggregate_spacing(samples, grid, cdf, output_dir)
    fragment_summary = aggregate_fragments(fragment_paths, alpha, output_dir)
    plot_eigenvector_summary(eigenvector_rows, alpha, output_dir)
    plot_coordinate_heatmap(mean_decay, alpha, output_dir)
    plot_density(grid, density, alpha, output_dir)
    plot_spacing_summary(spacing_rows, alpha, output_dir)
    plot_spacing_histograms(spacing_rows, histograms, alpha, output_dir)
    plot_spacing_heatmap(spacing_rows, spacing_heat, alpha, output_dir)
    manifest = {
        "model": "affine linear preferential attachment",
        "attachment_kernel": "P(t->i | G_t) proportional to degree_i(t) + delta",
        "attachment_tail_exponent_alpha": alpha,
        "attachment_offset_delta": specification(alpha).delta,
        "exponent_convention": "1-F_D(k) approximately k^(-alpha); this is not the Chung-Lu density exponent",
        "matched_chung_lu_seed_label": specification(alpha).matched_chung_lu_label,
        "n": N,
        "m": M,
        "target_mean_degree_for_comparison": TARGET_MEAN_DEGREE,
        "exact_realized_mean_degree_by_edge_count": 2.0 * (M * N - M * (M + 1) // 2) / N,
        "num_graphs": NUM_GRAPHS,
        "graph_seed_base": specification(alpha).graph_seed_base,
        "same_seed_list_as_corresponding_chung_lu_graph_run": True,
        "quantile_min": Q_MIN,
        "quantile_max": Q_MAX,
        "quantile_width": Q_WIDTH,
        "num_quantile_bins": len(QUANTILE_BINS),
        "perron_removed": True,
        "exact_zero_eigenpairs_removed_from_continuous_scans": True,
        "density_method": "variable local-count density",
        "density_bulk_half_width": BULK_DENSITY_HALF_WIDTH,
        "density_tail_half_width": TAIL_DENSITY_HALF_WIDTH,
        "unfolding": "xi=M*F(lambda), gaps computed within each graph only",
        "gap_ratio_expectations": {"GOE": GOE_MEAN_GAP_RATIO, "Poisson": POISSON_MEAN_GAP_RATIO},
        "fragment_summary": fragment_summary,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return output_dir


# ---------------------------------------------------------------------------
# Parameterized mobility-edge production cache (legacy entry points above are
# intentionally preserved unchanged).


@dataclass(frozen=True)
class MobilityEdgeProductionConfig:
    n: int
    m: int
    num_graphs: int
    graph_seed_base: int
    output_root: Path
    model_family: str = "affine_linear"
    degree_ccdf_alpha: float | None = None
    attachment_zeta: float | None = None
    graph_variant: str = "original"
    rewire_seed_base: int | None = None
    swaps_per_edge: int = 10
    max_tries_factor: int = 20
    quantile_width: float = 0.002
    zero_eigenvalue_atol: float = ZERO_EIGENVALUE_ATOL
    exact_resonance_atol: float = EXACT_RESONANCE_ATOL
    localization_block_size: int = 64

    def __post_init__(self) -> None:
        if not 1 <= self.m < self.n:
            raise ValueError("production requires 1 <= m < n")
        if self.num_graphs < 1:
            raise ValueError("num_graphs must be positive")
        if self.model_family == "affine_linear":
            if self.degree_ccdf_alpha is None or self.attachment_zeta is not None:
                raise ValueError("affine production requires only degree_ccdf_alpha")
            attachment_offset_for_degree_ccdf_alpha(self.m, self.degree_ccdf_alpha)
        elif self.model_family == "nonlinear":
            if self.attachment_zeta is None or self.degree_ccdf_alpha is not None:
                raise ValueError("nonlinear production requires only attachment_zeta")
            if not math.isfinite(self.attachment_zeta) or self.attachment_zeta < 0.0:
                raise ValueError("nonlinear production requires finite attachment_zeta >= 0")
        else:
            raise ValueError(f"unsupported model_family={self.model_family!r}")
        if self.graph_variant not in {"original", "rewired"}:
            raise ValueError("graph_variant must be 'original' or 'rewired'")
        if self.graph_variant == "rewired" and self.rewire_seed_base is None:
            raise ValueError("rewired production requires rewire_seed_base")
        if self.graph_variant == "original" and self.rewire_seed_base is not None:
            raise ValueError("original production must not set rewire_seed_base")
        if self.swaps_per_edge < 1 or self.max_tries_factor < 1:
            raise ValueError("rewiring effort parameters must be positive")
        count = int(round(1.0 / self.quantile_width))
        if not np.isclose(count * self.quantile_width, 1.0, atol=1e-12):
            raise ValueError("quantile_width must divide [0,1]")

    @property
    def canonical_parameter_name(self) -> str:
        return "degree_ccdf_alpha" if self.model_family == "affine_linear" else "attachment_zeta"

    @property
    def canonical_parameter_value(self) -> float:
        value = self.degree_ccdf_alpha if self.model_family == "affine_linear" else self.attachment_zeta
        if value is None:
            raise AssertionError("missing canonical parameter")
        return float(value)

    @property
    def parameter_dir(self) -> Path:
        family = "affine_linear" if self.model_family == "affine_linear" else "nonlinear"
        family_root = Path(self.output_root) / family
        if self.model_family == "nonlinear" and self.canonical_parameter_value < 1.0:
            family_root = family_root / "sublinear"
        directory = family_root / f"n_{self.n}_m_{self.m}" / (
            f"{self.canonical_parameter_name}_{slug_number(self.canonical_parameter_value)}"
        )
        return directory if self.graph_variant == "original" else directory / "rewired"

    @property
    def partial_dir(self) -> Path:
        return self.parameter_dir / "partials"

    def graph_seeds(self) -> list[int]:
        return make_seeds(self.num_graphs, self.graph_seed_base)

    def rewire_seeds(self) -> list[int]:
        if self.graph_variant != "rewired" or self.rewire_seed_base is None:
            raise ValueError("rewire seeds are defined only for rewired production")
        return make_seeds(self.num_graphs, self.rewire_seed_base)

    def metadata(self) -> dict[str, object]:
        legacy = (
            {"name": "attachment_tail_exponent", "value": self.degree_ccdf_alpha}
            if self.model_family == "affine_linear"
            else None
        )
        metadata: dict[str, object] = {
            "analysis_version": "pa_mobility_edge_production_cache_v1",
            "model_family": self.model_family,
            "canonical_parameter": {
                "name": self.canonical_parameter_name,
                "value": self.canonical_parameter_value,
            },
            "legacy_parameter": legacy,
            "n": self.n,
            "m": self.m,
            "num_graphs": self.num_graphs,
            "graph_seed_base": self.graph_seed_base,
            "quantile_width": self.quantile_width,
            "zero_eigenvalue_atol": self.zero_eigenvalue_atol,
            "exact_resonance_atol": self.exact_resonance_atol,
            "perron_removed": True,
            "full_dense_eigenvectors_saved": False,
            "per_eigenpair_localization_scalars_saved": True,
        }
        if self.model_family == "nonlinear" and self.canonical_parameter_value < 1.0:
            metadata["attachment_regime"] = "sublinear"
        if self.graph_variant == "rewired":
            metadata.update(
                {
                    "graph_variant": "rewired",
                    "rewire_seed_base": self.rewire_seed_base,
                    "rewiring": {
                        "method": "networkx.double_edge_swap",
                        "swaps_per_edge": self.swaps_per_edge,
                        "max_tries_factor": self.max_tries_factor,
                        "degree_preservation": "exact labeled degree equality required",
                    },
                }
            )
        return metadata


def degree_gini(degrees: np.ndarray) -> float:
    values = np.sort(np.asarray(degrees, dtype=float))
    total = float(np.sum(values))
    if not len(values) or total <= 0.0:
        return 0.0
    ranks = np.arange(1, len(values) + 1, dtype=float)
    return float((2.0 * np.sum(ranks * values) / (len(values) * total)) - (len(values) + 1) / len(values))


def _top_degree_share(degrees: np.ndarray, count: int) -> float:
    values = np.sort(np.asarray(degrees, dtype=float))[::-1]
    return float(np.sum(values[: min(count, len(values))]) / np.sum(values))


def per_eigenpair_localization_scalars(
    vectors: np.ndarray,
    *,
    quantile_width: float,
    block_size: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Cache fine-rebinning scalars without retaining dense eigenvectors."""
    n, count = vectors.shape
    fields = {
        "ipr": np.empty(count, dtype=float),
        "n_times_ipr": np.empty(count, dtype=float),
        "participation_ratio": np.empty(count, dtype=float),
        "participation_fraction": np.empty(count, dtype=float),
    }
    for top_r in TOP_R_VALUES:
        fields[f"top_{top_r}_mass"] = np.empty(count, dtype=float)
    for level in MASS_LEVELS:
        fields[f"r_{int(round(100 * level))}_fraction"] = np.empty(count, dtype=float)
    num_bins = int(round(1.0 / quantile_width))
    decay_sum = np.zeros((num_bins, COORDINATE_DECAY_MAX_RANK), dtype=float)
    decay_count = np.zeros(num_bins, dtype=int)
    for start in range(0, count, block_size):
        stop = min(start + block_size, count)
        masses = np.square(vectors[:, start:stop])
        sorted_masses = np.sort(masses, axis=0)[::-1]
        cumulative = np.cumsum(sorted_masses, axis=0)
        ipr = np.sum(np.square(masses), axis=0)
        fields["ipr"][start:stop] = ipr
        fields["n_times_ipr"][start:stop] = n * ipr
        fields["participation_ratio"][start:stop] = 1.0 / np.maximum(ipr, 1e-300)
        fields["participation_fraction"][start:stop] = 1.0 / np.maximum(n * ipr, 1e-300)
        for top_r in TOP_R_VALUES:
            fields[f"top_{top_r}_mass"][start:stop] = cumulative[min(top_r, n) - 1]
        for level in MASS_LEVELS:
            radii = np.argmax(cumulative >= level - 1e-12, axis=0) + 1
            fields[f"r_{int(round(100 * level))}_fraction"][start:stop] = radii / n
        for local, eigenpair_index in enumerate(range(start, stop)):
            q = (eigenpair_index + 0.5) / count
            bin_index = min(int(q / quantile_width), num_bins - 1)
            recorded = min(COORDINATE_DECAY_MAX_RANK, n)
            decay_sum[bin_index, :recorded] += sorted_masses[:recorded, local]
            decay_count[bin_index] += 1
    graph_decay = np.divide(
        decay_sum,
        decay_count[:, None],
        out=np.zeros_like(decay_sum),
        where=decay_count[:, None] > 0,
    )
    return fields, graph_decay


def analyze_graph_for_mobility_cache(
    config: MobilityEdgeProductionConfig,
    graph_index: int,
) -> tuple[
    dict[str, object],
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    seeds = config.graph_seeds()
    graph_seed = seeds[graph_index]
    if config.model_family == "affine_linear":
        original_graph, generator = generate_preferential_attachment_graph(
            n=config.n,
            m=config.m,
            seed=graph_seed,
            model=MODEL_AFFINE_PA,
            degree_ccdf_alpha=config.degree_ccdf_alpha,
        )
    else:
        original_graph, generator = generate_preferential_attachment_graph(
            n=config.n,
            m=config.m,
            seed=graph_seed,
            model=MODEL_NONLINEAR_PA,
            attachment_zeta=float(config.attachment_zeta),
        )
    rewire_seed: int | None = None
    requested_swaps = 0
    max_tries = 0
    edge_overlap = math.nan
    original_connected = nx.is_connected(original_graph)
    if config.graph_variant == "rewired":
        rewire_seed = config.rewire_seeds()[graph_index]
        graph, requested_swaps, max_tries, edge_overlap = rewire_graph_preserving_degrees(
            original_graph,
            rewire_seed=rewire_seed,
            swaps_per_edge=config.swaps_per_edge,
            max_tries_factor=config.max_tries_factor,
        )
        original_degrees = np.asarray(
            [original_graph.degree(node) for node in range(config.n)], dtype=np.int32
        )
        rewired_degrees = np.asarray(
            [graph.degree(node) for node in range(config.n)], dtype=np.int32
        )
        if not np.array_equal(original_degrees, rewired_degrees):
            raise AssertionError("rewired graph failed exact labeled degree preservation")
    else:
        graph = original_graph
    eigenvalues, eigenvectors = dense_adjacency_eigendecomposition(graph)
    perron_index = int(np.argmax(eigenvalues))
    perron = float(eigenvalues[perron_index])
    nonperron = np.delete(eigenvalues, perron_index)
    nonperron_vectors = np.delete(eigenvectors, perron_index, axis=1)
    zero_mask = np.isclose(
        nonperron, 0.0, atol=config.zero_eigenvalue_atol, rtol=0.0
    )
    continuous_values = nonperron[~zero_mask]
    continuous_vectors = nonperron_vectors[:, ~zero_mask]
    scalars, graph_decay = per_eigenpair_localization_scalars(
        continuous_vectors,
        quantile_width=config.quantile_width,
        block_size=config.localization_block_size,
    )
    degrees = np.asarray([graph.degree(node) for node in range(config.n)], dtype=np.int32)
    degree_order = np.argsort(-degrees, kind="stable").astype(np.int32)
    maximum_degree = int(np.max(degrees))
    maximum_nodes = np.flatnonzero(degrees == maximum_degree)
    components = list(nx.connected_components(graph))
    total_degree = float(np.sum(degrees))
    structural = {
        "graph_index": graph_index,
        "graph_seed": graph_seed,
        "rewire_seed": "" if rewire_seed is None else rewire_seed,
        "rewiring_requested_swaps": requested_swaps,
        "rewiring_max_tries": max_tries,
        "rewiring_edge_overlap_fraction": edge_overlap,
        "original_connected": original_connected,
        "rewired_connected": nx.is_connected(graph) if config.graph_variant == "rewired" else "",
        "exact_labeled_degree_preserved": config.graph_variant == "rewired",
        "model_family": config.model_family,
        "graph_variant": config.graph_variant,
        "canonical_parameter_name": config.canonical_parameter_name,
        "canonical_parameter_value": config.canonical_parameter_value,
        "legacy_parameter_name": "attachment_tail_exponent" if config.model_family == "affine_linear" else "",
        "legacy_parameter_value": config.degree_ccdf_alpha if config.model_family == "affine_linear" else "",
        "n": config.n,
        "m": config.m,
        "num_edges": graph.number_of_edges(),
        "realized_mean_degree": float(np.mean(degrees)),
        "minimum_degree": int(np.min(degrees)),
        "maximum_degree": maximum_degree,
        "maximum_degree_vertex_birth_index": int(maximum_nodes[0]),
        "degree_gini": degree_gini(degrees),
        "top_1_degree_share": _top_degree_share(degrees, 1),
        "top_5_degree_share": _top_degree_share(degrees, 5),
        "top_10_degree_share": _top_degree_share(degrees, 10),
        "top_1_percent_degree_share": _top_degree_share(degrees, max(1, int(math.ceil(0.01 * config.n)))),
        "maximum_degree_over_n": maximum_degree / config.n,
        "maximum_degree_over_total_degree": maximum_degree / total_degree,
        "degree_m_count": int(np.sum(degrees == config.m)),
        "degree_m_fraction": float(np.mean(degrees == config.m)),
        "vertices_with_at_least_0p1_percent_total_degree": int(
            np.sum(degrees / total_degree >= 0.001)
        ),
        "vertices_with_at_least_0p5_percent_total_degree": int(
            np.sum(degrees / total_degree >= 0.005)
        ),
        "vertices_with_at_least_1_percent_total_degree": int(
            np.sum(degrees / total_degree >= 0.01)
        ),
        "num_components": len(components),
        "largest_component_size": max(map(len, components)),
        "connected": len(components) == 1,
        "zero_multiplicity": int(np.sum(zero_mask)),
        "minus_one_multiplicity": int(np.sum(np.isclose(nonperron, -1.0, atol=config.exact_resonance_atol, rtol=0.0))),
        "plus_one_multiplicity": int(np.sum(np.isclose(nonperron, 1.0, atol=config.exact_resonance_atol, rtol=0.0))),
        "perron_eigenvalue": perron,
        "lower_spectral_edge": float(nonperron[0]),
        "upper_nonperron_spectral_edge": float(nonperron[-1]),
        "nonperron_spectral_mean": float(np.mean(nonperron)),
        "nonperron_spectral_standard_deviation": float(np.std(nonperron)),
        "nonperron_mean_absolute_energy": float(np.mean(np.abs(nonperron))),
        "nonperron_positive_fraction": float(np.mean(nonperron > 0.0)),
        "continuous_nonperron_dimension": len(continuous_values),
        "generator_metadata": json.dumps(generator.to_dict(), sort_keys=True),
    }
    cache = {
        "eigenvalue": continuous_values,
        "continuous_rank": np.arange(len(continuous_values), dtype=np.int32),
        "continuous_quantile": (np.arange(len(continuous_values), dtype=float) + 0.5) / len(continuous_values),
        **scalars,
    }
    return structural, cache, graph_decay, nonperron, degrees, degree_order


def mobility_chunk_paths(
    config: MobilityEdgeProductionConfig, start: int, stop: int
) -> tuple[Path, Path]:
    suffix = f"{start:04d}_{stop:04d}"
    return (
        config.partial_dir / f"mobility_cache_{suffix}.npz",
        config.partial_dir / f"structural_graphs_{suffix}.csv.gz",
    )


def chunk_execution_metadata(start: int, stop: int) -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except OSError:
        commit = ""
    return {
        "sge_job_id": os.environ.get("JOB_ID", ""),
        "sge_task_id": os.environ.get("SGE_TASK_ID", ""),
        "graph_index_start": start,
        "graph_index_stop_exclusive": stop,
        "git_commit": commit or "UNAVAILABLE_DIRTY_OR_UNCOMMITTED_DEPLOYMENT",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "networkx_version": nx.__version__,
    }


def validate_mobility_chunk(
    config: MobilityEdgeProductionConfig, start: int, stop: int
) -> tuple[bool, str]:
    cache_path, structural_path = mobility_chunk_paths(config, start, stop)
    if not cache_path.is_file() or not structural_path.is_file():
        return False, "missing cache or structural partial"
    expected_indices = np.arange(start, stop, dtype=int)
    expected_seeds = np.asarray(config.graph_seeds()[start:stop], dtype=np.uint64)
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            indices = np.asarray(data["graph_indices"], dtype=int)
            seeds = np.asarray(data["graph_seeds"], dtype=np.uint64)
            nonperron = np.asarray(data["nonperron_eigenvalues"], dtype=float)
            degrees = np.asarray(data["degree_arrays"], dtype=np.int32)
            offsets = np.asarray(data["eigenpair_offsets"], dtype=int)
            values = np.asarray(data["eigenpair_eigenvalue"], dtype=float)
            metadata = json.loads(str(data["metadata_json"].item()))
            execution = (
                json.loads(str(data["execution_json"].item()))
                if "execution_json" in data.files
                else None
            )
            required = [
                "eigenpair_ipr",
                "eigenpair_n_times_ipr",
                "eigenpair_participation_ratio",
                "eigenpair_participation_fraction",
                "eigenpair_top_10_mass",
                "eigenpair_top_50_mass",
                "eigenpair_top_100_mass",
                "eigenpair_top_200_mass",
                "eigenpair_r_50_fraction",
                "eigenpair_r_80_fraction",
                "eigenpair_r_90_fraction",
                "eigenpair_r_95_fraction",
            ]
            for key in required:
                if len(np.asarray(data[key])) != len(values):
                    return False, f"ragged scalar length mismatch for {key}"
    except Exception as error:
        return False, f"unreadable cache: {error}"
    if not np.array_equal(indices, expected_indices):
        return False, "graph-index mismatch"
    if not np.array_equal(seeds, expected_seeds):
        return False, "graph-seed mismatch"
    if nonperron.shape != (stop - start, config.n - 1):
        return False, "non-Perron spectrum shape mismatch"
    if degrees.shape != (stop - start, config.n):
        return False, "degree-array shape mismatch"
    if len(offsets) != stop - start + 1 or offsets[0] != 0 or offsets[-1] != len(values):
        return False, "invalid eigenpair offsets"
    if metadata != config.metadata():
        return False, "parameter metadata mismatch"
    if execution is not None and (
        int(execution["graph_index_start"]) != start
        or int(execution["graph_index_stop_exclusive"]) != stop
    ):
        return False, "embedded execution graph range mismatch"
    try:
        rows = _read_csv(structural_path)
    except Exception as error:
        return False, f"unreadable structural CSV: {error}"
    if [int(row["graph_index"]) for row in rows] != list(range(start, stop)):
        return False, "structural graph-index mismatch"
    if [int(row["graph_seed"]) for row in rows] != expected_seeds.astype(int).tolist():
        return False, "structural graph-seed mismatch"
    if any(row.get("graph_variant", "original") != config.graph_variant for row in rows):
        return False, "structural graph-variant mismatch"
    if config.graph_variant == "rewired":
        expected_rewire_seeds = config.rewire_seeds()[start:stop]
        if [int(row["rewire_seed"]) for row in rows] != expected_rewire_seeds:
            return False, "structural rewire-seed mismatch"
        if not all(row.get("exact_labeled_degree_preserved") == "True" for row in rows):
            return False, "exact labeled-degree validation is absent"
    return True, "VALID"


def run_mobility_chunk(
    config: MobilityEdgeProductionConfig,
    start: int,
    stop: int,
    *,
    skip_valid: bool = True,
) -> Path:
    if not 0 <= start < stop <= config.num_graphs:
        raise ValueError("invalid graph range")
    valid, reason = validate_mobility_chunk(config, start, stop)
    if skip_valid and valid:
        print(f"skip valid chunk [{start},{stop}): {reason}", flush=True)
        return config.partial_dir
    config.partial_dir.mkdir(parents=True, exist_ok=True)
    structural_rows: list[dict[str, object]] = []
    graph_caches: list[dict[str, np.ndarray]] = []
    decays: list[np.ndarray] = []
    spectra: list[np.ndarray] = []
    degrees: list[np.ndarray] = []
    degree_orders: list[np.ndarray] = []
    execution = chunk_execution_metadata(start, stop)
    for graph_index in range(start, stop):
        structural, cache, decay, spectrum, degree, degree_order = analyze_graph_for_mobility_cache(
            config, graph_index
        )
        structural_rows.append(structural)
        structural_rows[-1].update(execution)
        graph_caches.append(cache)
        decays.append(decay)
        spectra.append(spectrum)
        degrees.append(degree)
        degree_orders.append(degree_order)
        print(
            f"finished {config.canonical_parameter_name}={config.canonical_parameter_value:g} graph={graph_index}",
            flush=True,
        )
    offsets = np.zeros(len(graph_caches) + 1, dtype=np.int64)
    for index, cache in enumerate(graph_caches):
        offsets[index + 1] = offsets[index] + len(cache["eigenvalue"])
    flattened = {
        f"eigenpair_{key}": np.concatenate([cache[key] for cache in graph_caches])
        for key in graph_caches[0]
    }
    decay_matrix = np.asarray(decays, dtype=float)
    cache_path, structural_path = mobility_chunk_paths(config, start, stop)
    np.savez_compressed(
        cache_path,
        graph_indices=np.arange(start, stop, dtype=int),
        graph_seeds=np.asarray(config.graph_seeds()[start:stop], dtype=np.uint64),
        nonperron_eigenvalues=np.asarray(spectra, dtype=float),
        degree_arrays=np.asarray(degrees, dtype=np.int32),
        degree_orderings=np.asarray(degree_orders, dtype=np.int32),
        eigenpair_offsets=offsets,
        coordinate_decay_sum=np.sum(decay_matrix, axis=0),
        coordinate_decay_sum_squares=np.sum(np.square(decay_matrix), axis=0),
        graph_count=len(graph_caches),
        metadata_json=np.asarray(json.dumps(config.metadata(), sort_keys=True)),
        execution_json=np.asarray(json.dumps(execution, sort_keys=True)),
        **flattened,
    )
    write_csv(structural_rows, structural_path, list(structural_rows[0]))
    valid, reason = validate_mobility_chunk(config, start, stop)
    if not valid:
        raise RuntimeError(f"newly written chunk failed integrity validation: {reason}")
    return config.partial_dir


def structural_replay_partial_dir(config: MobilityEdgeProductionConfig) -> Path:
    return config.parameter_dir / "structural" / "replay_partials"


def structural_replay_paths(
    config: MobilityEdgeProductionConfig, start: int, stop: int
) -> tuple[Path, Path]:
    suffix = f"{start:04d}_{stop:04d}"
    partial_dir = structural_replay_partial_dir(config)
    return (
        partial_dir / f"degree_arrays_{suffix}.npz",
        partial_dir / f"structural_graphs_{suffix}.csv.gz",
    )


def degree_only_structural_replay(
    config: MobilityEdgeProductionConfig,
    graph_index: int,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    graph_seed = config.graph_seeds()[graph_index]
    if config.model_family == "affine_linear":
        graph, generator = generate_preferential_attachment_graph(
            n=config.n,
            m=config.m,
            seed=graph_seed,
            model=MODEL_AFFINE_PA,
            degree_ccdf_alpha=config.degree_ccdf_alpha,
        )
    else:
        graph, generator = generate_preferential_attachment_graph(
            n=config.n,
            m=config.m,
            seed=graph_seed,
            model=MODEL_NONLINEAR_PA,
            attachment_zeta=float(config.attachment_zeta),
        )
    degrees = np.asarray([graph.degree(node) for node in range(config.n)], dtype=np.int32)
    degree_order = np.argsort(-degrees, kind="stable").astype(np.int32)
    maximum_degree = int(np.max(degrees))
    maximum_nodes = np.flatnonzero(degrees == maximum_degree)
    components = list(nx.connected_components(graph))
    total_degree = float(np.sum(degrees))
    row = {
        "graph_index": graph_index,
        "graph_seed": graph_seed,
        "model_family": config.model_family,
        "canonical_parameter_name": config.canonical_parameter_name,
        "canonical_parameter_value": config.canonical_parameter_value,
        "legacy_parameter_name": "attachment_tail_exponent" if config.model_family == "affine_linear" else "",
        "legacy_parameter_value": config.degree_ccdf_alpha if config.model_family == "affine_linear" else "",
        "n": config.n,
        "m": config.m,
        "num_edges": graph.number_of_edges(),
        "realized_mean_degree": float(np.mean(degrees)),
        "minimum_degree": int(np.min(degrees)),
        "maximum_degree": maximum_degree,
        "maximum_degree_vertex_birth_index": int(maximum_nodes[0]),
        "degree_gini": degree_gini(degrees),
        "top_1_degree_share": _top_degree_share(degrees, 1),
        "top_5_degree_share": _top_degree_share(degrees, 5),
        "top_10_degree_share": _top_degree_share(degrees, 10),
        "top_1_percent_degree_share": _top_degree_share(
            degrees, max(1, int(math.ceil(0.01 * config.n)))
        ),
        "maximum_degree_over_n": maximum_degree / config.n,
        "maximum_degree_over_total_degree": maximum_degree / total_degree,
        "degree_m_count": int(np.sum(degrees == config.m)),
        "degree_m_fraction": float(np.mean(degrees == config.m)),
        "vertices_with_at_least_0p1_percent_total_degree": int(
            np.sum(degrees / total_degree >= 0.001)
        ),
        "vertices_with_at_least_0p5_percent_total_degree": int(
            np.sum(degrees / total_degree >= 0.005)
        ),
        "vertices_with_at_least_1_percent_total_degree": int(
            np.sum(degrees / total_degree >= 0.01)
        ),
        "num_components": len(components),
        "largest_component_size": max(map(len, components)),
        "connected": len(components) == 1,
        "structural_replay_without_eigendecomposition": True,
        "generator_metadata": json.dumps(generator.to_dict(), sort_keys=True),
    }
    return row, degrees, degree_order


def validate_structural_replay_chunk(
    config: MobilityEdgeProductionConfig, start: int, stop: int
) -> tuple[bool, str]:
    degree_path, structural_path = structural_replay_paths(config, start, stop)
    if not degree_path.is_file() or not structural_path.is_file():
        return False, "missing degree or structural partial"
    try:
        with np.load(degree_path, allow_pickle=False) as data:
            indices = np.asarray(data["graph_indices"], dtype=int)
            seeds = np.asarray(data["graph_seeds"], dtype=np.uint64)
            degrees = np.asarray(data["degree_arrays"], dtype=np.int32)
            ordering = np.asarray(data["degree_orderings"], dtype=np.int32)
            metadata = json.loads(str(data["metadata_json"].item()))
        rows = _read_csv(structural_path)
    except Exception as error:
        return False, f"unreadable structural replay: {error}"
    expected_indices = np.arange(start, stop, dtype=int)
    expected_seeds = np.asarray(config.graph_seeds()[start:stop], dtype=np.uint64)
    if not np.array_equal(indices, expected_indices):
        return False, "graph-index mismatch"
    if not np.array_equal(seeds, expected_seeds):
        return False, "graph-seed mismatch"
    if degrees.shape != (stop - start, config.n):
        return False, "degree-array shape mismatch"
    if ordering.shape != degrees.shape:
        return False, "degree-ordering shape mismatch"
    if metadata != config.metadata():
        return False, "parameter metadata mismatch"
    if [int(row["graph_index"]) for row in rows] != list(range(start, stop)):
        return False, "structural CSV graph-index mismatch"
    if [int(row["graph_seed"]) for row in rows] != expected_seeds.astype(int).tolist():
        return False, "structural CSV graph-seed mismatch"
    return True, "VALID"


def run_structural_replay_chunk(
    config: MobilityEdgeProductionConfig,
    start: int,
    stop: int,
    *,
    skip_valid: bool = True,
) -> Path:
    if not 0 <= start < stop <= config.num_graphs:
        raise ValueError("invalid graph range")
    valid, reason = validate_structural_replay_chunk(config, start, stop)
    if skip_valid and valid:
        print(f"skip valid structural replay [{start},{stop}): {reason}", flush=True)
        return structural_replay_partial_dir(config)
    partial_dir = structural_replay_partial_dir(config)
    partial_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    degrees: list[np.ndarray] = []
    orderings: list[np.ndarray] = []
    for graph_index in range(start, stop):
        row, degree, ordering = degree_only_structural_replay(config, graph_index)
        rows.append(row)
        degrees.append(degree)
        orderings.append(ordering)
        print(f"replayed structure graph={graph_index}", flush=True)
    degree_path, structural_path = structural_replay_paths(config, start, stop)
    np.savez_compressed(
        degree_path,
        graph_indices=np.arange(start, stop, dtype=int),
        graph_seeds=np.asarray(config.graph_seeds()[start:stop], dtype=np.uint64),
        degree_arrays=np.asarray(degrees, dtype=np.int32),
        degree_orderings=np.asarray(orderings, dtype=np.int32),
        metadata_json=np.asarray(json.dumps(config.metadata(), sort_keys=True)),
    )
    write_csv(rows, structural_path, list(rows[0]))
    valid, reason = validate_structural_replay_chunk(config, start, stop)
    if not valid:
        raise RuntimeError(f"new structural replay failed validation: {reason}")
    return partial_dir
