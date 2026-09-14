from __future__ import annotations

import csv
import gzip
import json
import math
import os
import tempfile
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "small-world-mobility-edge-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import networkx as nx
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from .chung_lu_baseline import (
    WeightSequenceDiagnostics,
    generate_chung_lu_graph_from_weights,
    generate_power_law_expected_weights,
)
from .regular_baseline import adjacency_eigendecomposition


N = 5000
TARGET_MEAN_DEGREE = 20.0
NUM_GRAPHS = 500
COMPONENT_MODE = "full"
SEQUENCE_MODE = "quantile"
W_MIN = 1.0
W_MAX = None
CUTOFF_MODE = "structural"
REMOVE_PERRON_EIGENPAIR = True
REMOVE_EXACT_ZERO_EIGENPAIRS = True
ZERO_EIGENVALUE_ATOL = 1e-8
Q_MIN = 0.5
Q_MAX = 1.0
Q_WIDTH = 0.002
COORDINATE_DECAY_MAX_RANK = 200
NUMERICAL_COORDINATE_ATOL = 1e-12
TOP_R_VALUES = (1, 5, 10, 20, 50, 100, 200)
MASS_LEVELS = (0.50, 0.80, 0.90, 0.95, 0.99)
OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "chung_lu"
    / "power_law"
    / "eigenvectors"
    / "percentile_scan_q0p5_1_width_0p002"
)


@dataclass(frozen=True)
class LegacyTauSpecification:
    displayed_tau: float
    tau_convention: str
    effective_density_exponent: float
    cutoff_coefficient: float
    graph_seed_base: int
    sequence_seed_base: int
    source_eigenvalue_manifest: str


# These values reproduce the three legacy eigenvalue folders under
# power_law_tail_zoom_500.  The old runs mixed two exponent conventions, so
# both the displayed value and the exponent sent to the generator are explicit.
TAU_SPECIFICATIONS = (
    LegacyTauSpecification(
        displayed_tau=0.8,
        tau_convention="experimental_ccdf_exponent",
        effective_density_exponent=1.8,
        cutoff_coefficient=0.999,
        graph_seed_base=27282,
        sequence_seed_base=14242,
        source_eigenvalue_manifest=(
            "outputs/chung_lu/power_law_tail_zoom_500/n_5000/"
            "tau_0p8_avgdeg_20p0/run_manifest.json"
        ),
    ),
    LegacyTauSpecification(
        displayed_tau=1.05,
        tau_convention="density_exponent_ccdf_tau_minus_one",
        effective_density_exponent=1.05,
        cutoff_coefficient=1.0,
        graph_seed_base=27307,
        sequence_seed_base=14267,
        source_eigenvalue_manifest=(
            "outputs/chung_lu/power_law_tail_zoom_500/n_5000/"
            "tau_1p05_avgdeg_20p0/run_manifest.json"
        ),
    ),
    LegacyTauSpecification(
        displayed_tau=1.5,
        tau_convention="density_exponent_ccdf_tau_minus_one",
        effective_density_exponent=1.5,
        cutoff_coefficient=1.0,
        graph_seed_base=27352,
        sequence_seed_base=14312,
        source_eigenvalue_manifest=(
            "outputs/chung_lu/power_law_tail_zoom_500/n_5000/"
            "tau_1p5_avgdeg_20p0/run_manifest.json"
        ),
    ),
)


def tau_slug(tau: float) -> str:
    return str(tau).replace(".", "p")


def tau_specification(displayed_tau: float) -> LegacyTauSpecification:
    for specification in TAU_SPECIFICATIONS:
        if math.isclose(specification.displayed_tau, displayed_tau, rel_tol=0.0, abs_tol=1e-12):
            return specification
    raise ValueError(f"unsupported displayed tau={displayed_tau!r}")


def parameter_output_dir(displayed_tau: float, n: int = N) -> Path:
    return OUTPUT_DIR / f"n_{n}_tau_{tau_slug(displayed_tau)}_avgdeg_20p0"


def quantile_bins(
    q_min: float = Q_MIN,
    q_max: float = Q_MAX,
    q_width: float = Q_WIDTH,
) -> list[tuple[float, float]]:
    if not 0.0 <= q_min < q_max <= 1.0:
        raise ValueError("quantile limits must satisfy 0 <= q_min < q_max <= 1")
    if q_width <= 0.0:
        raise ValueError("q_width must be positive")
    count_float = (q_max - q_min) / q_width
    count = int(round(count_float))
    if not math.isclose(count_float, count, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError("q_width must exactly partition [q_min, q_max]")
    edges = np.linspace(q_min, q_max, count + 1)
    return [(float(edges[index]), float(edges[index + 1])) for index in range(count)]


LOCAL_QUANTILE_BINS = quantile_bins()


def make_seeds(num_graphs: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(value) for value in rng.integers(0, 2**32 - 1, size=num_graphs, dtype=np.uint32)]


def graph_and_sequence_seeds(displayed_tau: float) -> tuple[list[int], list[int]]:
    specification = tau_specification(displayed_tau)
    return (
        make_seeds(NUM_GRAPHS, specification.graph_seed_base),
        make_seeds(NUM_GRAPHS, specification.sequence_seed_base),
    )


def quantile_index_bin(num_values: int, q_low: float, q_high: float) -> tuple[int, int]:
    if num_values <= 0:
        raise ValueError("num_values must be positive")
    if not 0.0 <= q_low < q_high <= 1.0:
        raise ValueError(f"invalid quantile bin {(q_low, q_high)!r}")
    start = int(np.floor(q_low * num_values))
    stop = int(np.ceil(q_high * num_values))
    start = max(0, min(start, num_values - 1))
    stop = max(start + 1, min(stop, num_values))
    return start, stop


def select_component(graph: nx.Graph, component_mode: str = COMPONENT_MODE) -> nx.Graph:
    if component_mode == "full":
        return graph
    if component_mode == "largest_connected_component":
        nodes = max(nx.connected_components(graph), key=len)
        return graph.subgraph(nodes).copy()
    raise ValueError(f"unsupported component_mode={component_mode!r}")


def generate_legacy_expected_weights(
    displayed_tau: float,
    sequence_seed: int,
    n: int = N,
    target_mean_degree: float = TARGET_MEAN_DEGREE,
) -> tuple[np.ndarray, WeightSequenceDiagnostics]:
    specification = tau_specification(displayed_tau)
    return generate_power_law_expected_weights(
        n=n,
        tau=specification.effective_density_exponent,
        target_mean_degree=target_mean_degree,
        w_min=W_MIN,
        w_max=W_MAX,
        cutoff_mode=CUTOFF_MODE,
        cutoff_coefficient=specification.cutoff_coefficient,
        sequence_mode=SEQUENCE_MODE,
        sequence_seed=sequence_seed,
    )


def filter_continuous_nonperron_eigenpairs(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    values = np.asarray(eigenvalues, dtype=float)
    vectors = np.asarray(eigenvectors, dtype=float)
    if values.ndim != 1 or vectors.ndim != 2 or vectors.shape[1] != len(values):
        raise ValueError("eigenvalue/eigenvector shapes are incompatible")
    if REMOVE_PERRON_EIGENPAIR:
        if len(values) < 2:
            raise ValueError("need at least two eigenpairs to remove the Perron pair")
        perron = float(values[-1])
        values = values[:-1]
        vectors = vectors[:, :-1]
    else:
        perron = math.nan
    zero_mask = np.isclose(values, 0.0, atol=ZERO_EIGENVALUE_ATOL, rtol=0.0)
    zero_count = int(np.sum(zero_mask))
    if REMOVE_EXACT_ZERO_EIGENPAIRS:
        keep = ~zero_mask
        values = values[keep]
        vectors = vectors[:, keep]
    return values, vectors, zero_count, perron


def localization_arrays(eigenvectors: np.ndarray) -> dict[str, np.ndarray]:
    vectors = np.asarray(eigenvectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] == 0:
        raise ValueError("eigenvectors must contain at least one column")
    dimension = vectors.shape[0]
    masses = np.square(np.abs(vectors))
    sorted_masses = np.sort(masses, axis=0)[::-1, :]
    cumulative = np.cumsum(sorted_masses, axis=0)
    ipr = np.sum(np.square(masses), axis=0)
    entropy = -np.sum(np.where(masses > 0.0, masses * np.log(np.maximum(masses, 1e-300)), 0.0), axis=0)
    result: dict[str, np.ndarray] = {
        "sorted_masses": sorted_masses,
        "cumulative_masses": cumulative,
        "ipr": ipr,
        "participation_ratio": 1.0 / np.maximum(ipr, 1e-300),
        "entropy_support": np.exp(entropy),
        "numerical_support_count": np.sum(np.abs(vectors) > NUMERICAL_COORDINATE_ATOL, axis=0),
    }
    for top_r in TOP_R_VALUES:
        result[f"top_{top_r}_mass"] = cumulative[min(top_r, dimension) - 1, :]
    for level in MASS_LEVELS:
        reached = cumulative >= level - 1e-12
        result[f"r_{int(round(100 * level))}"] = np.argmax(reached, axis=0) + 1
    return result


def graph_scan_rows(
    graph_index: int,
    graph_seed: int,
    sequence_seed: int,
    displayed_tau: float,
    n: int = N,
    local_quantile_bins: list[tuple[float, float]] | None = None,
    component_mode: str = COMPONENT_MODE,
) -> tuple[list[dict[str, object]], np.ndarray, WeightSequenceDiagnostics]:
    if local_quantile_bins is None:
        local_quantile_bins = LOCAL_QUANTILE_BINS
    specification = tau_specification(displayed_tau)
    weights, diagnostics = generate_legacy_expected_weights(
        displayed_tau=displayed_tau,
        sequence_seed=sequence_seed,
        n=n,
    )
    if not diagnostics.admissibility_holds:
        raise ValueError("expected-weight sequence violates Chung-Lu admissibility")

    graph = generate_chung_lu_graph_from_weights(weights, seed=graph_seed)
    analyzed_graph = select_component(graph, component_mode=component_mode)
    eigenvalues, eigenvectors = adjacency_eigendecomposition(analyzed_graph)
    continuous_values, continuous_vectors, zero_count, perron = filter_continuous_nonperron_eigenpairs(
        eigenvalues,
        eigenvectors,
    )
    if len(continuous_values) == 0:
        raise ValueError("no continuous non-Perron eigenpairs remain")

    first_start, _ = quantile_index_bin(
        len(continuous_values),
        local_quantile_bins[0][0],
        local_quantile_bins[0][1],
    )
    upper_vectors = continuous_vectors[:, first_start:]
    arrays = localization_arrays(upper_vectors)
    dimension = analyzed_graph.number_of_nodes()
    degrees = np.asarray([degree for _, degree in graph.degree()], dtype=float)
    components = list(nx.connected_components(graph))
    largest_component_size = max(len(component) for component in components)
    decay_rank_count = min(COORDINATE_DECAY_MAX_RANK, dimension)
    coordinate_decay = np.zeros((len(local_quantile_bins), decay_rank_count), dtype=float)
    rows: list[dict[str, object]] = []

    for bin_index, (q_low, q_high) in enumerate(local_quantile_bins):
        start, stop = quantile_index_bin(len(continuous_values), q_low, q_high)
        relative_start = start - first_start
        relative_stop = stop - first_start
        selected_values = continuous_values[start:stop]
        selected_count = stop - start
        if relative_start < 0 or relative_stop > upper_vectors.shape[1]:
            raise AssertionError("quantile slice falls outside precomputed upper-half arrays")
        selected = slice(relative_start, relative_stop)
        sorted_mass = arrays["sorted_masses"][:, selected]
        coordinate_decay[bin_index, :] = np.mean(
            sorted_mass[:decay_rank_count, :],
            axis=1,
        )
        ipr = arrays["ipr"][selected]
        participation = arrays["participation_ratio"][selected]
        entropy_support = arrays["entropy_support"][selected]
        numerical_support = arrays["numerical_support_count"][selected]
        row: dict[str, object] = {
            "graph_index": graph_index,
            "graph_seed": graph_seed,
            "sequence_seed": sequence_seed,
            "displayed_tau": displayed_tau,
            "tau_convention": specification.tau_convention,
            "effective_density_exponent": specification.effective_density_exponent,
            "target_mean_degree": TARGET_MEAN_DEGREE,
            "component_mode": component_mode,
            "quantile": 0.5 * (q_low + q_high),
            "quantile_low": q_low,
            "quantile_high": q_high,
            "index_start": start,
            "index_stop": stop,
            "number_of_eigenvectors": selected_count,
            "E_low": float(selected_values[0]),
            "E_high": float(selected_values[-1]),
            "E_center": float(np.median(selected_values)),
            "analyzed_dimension": dimension,
            "continuous_bulk_dimension": len(continuous_values),
            "zero_eigenvalue_count": zero_count,
            "zero_eigenvalue_fraction": zero_count / max(len(eigenvalues) - 1, 1),
            "perron_eigenvalue": perron,
            "num_components": len(components),
            "largest_component_size": largest_component_size,
            "num_isolated_vertices": int(np.sum(degrees == 0)),
            "realized_mean_degree": float(np.mean(degrees)),
            "max_realized_degree": int(np.max(degrees)),
            "max_expected_weight": diagnostics.final_max,
            "mean_ipr": float(np.mean(ipr)),
            "median_ipr": float(np.median(ipr)),
            "mean_n_times_ipr": float(dimension * np.mean(ipr)),
            "median_n_times_ipr": float(dimension * np.median(ipr)),
            "mean_participation_ratio": float(np.mean(participation)),
            "mean_participation_fraction": float(np.mean(participation) / dimension),
            "mean_entropy_support": float(np.mean(entropy_support)),
            "mean_entropy_support_fraction": float(np.mean(entropy_support) / dimension),
            "mean_numerical_support_count": float(np.mean(numerical_support)),
            "mean_numerical_support_fraction": float(np.mean(numerical_support) / dimension),
        }
        for top_r in TOP_R_VALUES:
            row[f"mean_top_{top_r}_mass"] = float(np.mean(arrays[f"top_{top_r}_mass"][selected]))
        for level in MASS_LEVELS:
            key = f"r_{int(round(100 * level))}"
            values = arrays[key][selected]
            row[f"mean_{key}"] = float(np.mean(values))
            row[f"median_{key}"] = float(np.median(values))
            row[f"mean_{key}_fraction"] = float(np.mean(values) / dimension)
        rows.append(row)
    return rows, coordinate_decay, diagnostics


def graph_row_fieldnames() -> list[str]:
    fields = [
        "graph_index", "graph_seed", "sequence_seed", "displayed_tau", "tau_convention",
        "effective_density_exponent", "target_mean_degree", "component_mode", "quantile", "quantile_low",
        "quantile_high", "index_start", "index_stop", "number_of_eigenvectors", "E_low",
        "E_high", "E_center", "analyzed_dimension", "continuous_bulk_dimension",
        "zero_eigenvalue_count", "zero_eigenvalue_fraction", "perron_eigenvalue",
        "num_components", "largest_component_size", "num_isolated_vertices",
        "realized_mean_degree", "max_realized_degree", "max_expected_weight", "mean_ipr",
        "median_ipr", "mean_n_times_ipr", "median_n_times_ipr", "mean_participation_ratio",
        "mean_participation_fraction", "mean_entropy_support", "mean_entropy_support_fraction",
        "mean_numerical_support_count", "mean_numerical_support_fraction",
    ]
    fields.extend(f"mean_top_{top_r}_mass" for top_r in TOP_R_VALUES)
    for level in MASS_LEVELS:
        key = f"r_{int(round(100 * level))}"
        fields.extend((f"mean_{key}", f"median_{key}", f"mean_{key}_fraction"))
    return fields


def write_csv(rows: Iterable[dict[str, object]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def read_csv(paths: Iterable[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", newline="") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    return rows


def _mean_and_se(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, math.nan
    return mean, float(np.std(values, ddof=1) / np.sqrt(len(values)))


def summary_metric_fields() -> list[str]:
    fields = [
        "E_center", "analyzed_dimension", "continuous_bulk_dimension", "zero_eigenvalue_fraction",
        "largest_component_size", "num_isolated_vertices", "realized_mean_degree",
        "mean_n_times_ipr", "median_n_times_ipr", "mean_participation_fraction",
        "mean_entropy_support_fraction", "mean_numerical_support_fraction",
    ]
    fields.extend(f"mean_top_{top_r}_mass" for top_r in TOP_R_VALUES)
    for level in MASS_LEVELS:
        key = f"r_{int(round(100 * level))}"
        fields.extend((f"mean_{key}", f"mean_{key}_fraction"))
    return fields


def aggregate_graph_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[float, float], list[dict[str, str]]] = {}
    for row in rows:
        key = (float(row["quantile_low"]), float(row["quantile_high"]))
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, object]] = []
    for key in sorted(grouped):
        group = grouped[key]
        first = group[0]
        result: dict[str, object] = {
            "displayed_tau": float(first["displayed_tau"]),
            "tau_convention": first["tau_convention"],
            "effective_density_exponent": float(first["effective_density_exponent"]),
            "target_mean_degree": float(first["target_mean_degree"]),
            "component_mode": first.get("component_mode", COMPONENT_MODE),
            "quantile": 0.5 * (key[0] + key[1]),
            "quantile_low": key[0],
            "quantile_high": key[1],
            "number_of_graphs": len(group),
            "number_of_eigenvectors": sum(int(row["number_of_eigenvectors"]) for row in group),
        }
        for field in summary_metric_fields():
            mean, standard_error = _mean_and_se(np.asarray([float(row[field]) for row in group]))
            result[field] = mean
            result[f"{field}_standard_error"] = standard_error
        output.append(result)
    return output


def aggregate_graph_csv_paths(
    paths: list[Path],
    by_graph_path: Path,
) -> tuple[list[dict[str, object]], list[int]]:
    """Stream graph rows so the N=5000 aggregate does not retain them all in RAM."""
    metric_fields = summary_metric_fields()
    grouped: dict[tuple[float, float], dict[str, object]] = {}
    graph_ids: set[int] = set()
    by_graph_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(by_graph_path, "wt", newline="") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=graph_row_fieldnames())
        writer.writeheader()
        for path in paths:
            with gzip.open(path, "rt", newline="") as input_handle:
                for row in csv.DictReader(input_handle):
                    writer.writerow({field: row.get(field, "") for field in graph_row_fieldnames()})
                    graph_ids.add(int(row["graph_index"]))
                    key = (float(row["quantile_low"]), float(row["quantile_high"]))
                    if key not in grouped:
                        grouped[key] = {
                            "first": row,
                            "count": 0,
                            "number_of_eigenvectors": 0,
                            "sums": {field: 0.0 for field in metric_fields},
                            "sums_of_squares": {field: 0.0 for field in metric_fields},
                        }
                    accumulator = grouped[key]
                    accumulator["count"] = int(accumulator["count"]) + 1
                    accumulator["number_of_eigenvectors"] = (
                        int(accumulator["number_of_eigenvectors"]) + int(row["number_of_eigenvectors"])
                    )
                    sums = accumulator["sums"]
                    sums_of_squares = accumulator["sums_of_squares"]
                    if not isinstance(sums, dict) or not isinstance(sums_of_squares, dict):
                        raise AssertionError("invalid streaming accumulator")
                    for field in metric_fields:
                        value = float(row[field])
                        sums[field] += value
                        sums_of_squares[field] += value * value

    output: list[dict[str, object]] = []
    for key in sorted(grouped):
        accumulator = grouped[key]
        first = accumulator["first"]
        if not isinstance(first, dict):
            raise AssertionError("invalid first-row accumulator")
        count = int(accumulator["count"])
        sums = accumulator["sums"]
        sums_of_squares = accumulator["sums_of_squares"]
        if not isinstance(sums, dict) or not isinstance(sums_of_squares, dict):
            raise AssertionError("invalid metric accumulator")
        result: dict[str, object] = {
            "displayed_tau": float(first["displayed_tau"]),
            "tau_convention": first["tau_convention"],
            "effective_density_exponent": float(first["effective_density_exponent"]),
            "target_mean_degree": float(first["target_mean_degree"]),
            "component_mode": first.get("component_mode", COMPONENT_MODE),
            "quantile": 0.5 * (key[0] + key[1]),
            "quantile_low": key[0],
            "quantile_high": key[1],
            "number_of_graphs": count,
            "number_of_eigenvectors": int(accumulator["number_of_eigenvectors"]),
        }
        for field in metric_fields:
            mean = float(sums[field]) / count
            if count > 1:
                variance = max(
                    (float(sums_of_squares[field]) - float(sums[field]) ** 2 / count) / (count - 1),
                    0.0,
                )
                standard_error = math.sqrt(variance / count)
            else:
                standard_error = math.nan
            result[field] = mean
            result[f"{field}_standard_error"] = standard_error
        output.append(result)
    return output, sorted(graph_ids)


def summary_row_fieldnames() -> list[str]:
    fields = [
        "displayed_tau", "tau_convention", "effective_density_exponent", "target_mean_degree",
        "component_mode", "quantile", "quantile_low", "quantile_high", "number_of_graphs",
        "number_of_eigenvectors",
    ]
    for field in summary_metric_fields():
        fields.extend((field, f"{field}_standard_error"))
    return fields


def asymptotic_goe_top_r_mass(n: int, r: int) -> float:
    fraction = r / n
    threshold = norm.isf(fraction / 2.0)
    return float(2.0 * (threshold * norm.pdf(threshold) + norm.sf(threshold)))


def asymptotic_goe_mass_radius_fraction(mass_level: float) -> float:
    if not 0.0 < mass_level < 1.0:
        raise ValueError("mass_level must lie in (0, 1)")

    def difference(fraction: float) -> float:
        threshold = norm.isf(fraction / 2.0)
        mass = 2.0 * (threshold * norm.pdf(threshold) + norm.sf(threshold))
        return mass - mass_level

    return float(brentq(difference, 1e-12, 1.0 - 1e-12))


def goe_benchmarks(n: int = N) -> dict[str, float]:
    result = {
        "mean_n_times_ipr_exact": 3.0 * n / (n + 2.0),
        "mean_participation_fraction_asymptotic": 1.0 / 3.0,
    }
    for top_r in TOP_R_VALUES:
        result[f"top_{top_r}_mass_asymptotic"] = asymptotic_goe_top_r_mass(n, top_r)
    for level in MASS_LEVELS:
        key = f"r_{int(round(100 * level))}"
        fraction = asymptotic_goe_mass_radius_fraction(level)
        result[f"{key}_fraction_asymptotic"] = fraction
        result[f"{key}_count_asymptotic"] = fraction * n
    return result


def aggregate_coordinate_decay(npz_paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = np.zeros((len(LOCAL_QUANTILE_BINS), COORDINATE_DECAY_MAX_RANK), dtype=float)
    total_squares = np.zeros_like(total)
    graph_counts = np.zeros(len(LOCAL_QUANTILE_BINS), dtype=int)
    for path in npz_paths:
        with np.load(path) as data:
            total += data["decay_sum"]
            total_squares += data["decay_sum_squares"]
            graph_counts += data["graph_counts"]
    denominator = np.maximum(graph_counts[:, None], 1)
    mean = total / denominator
    variance_numerator = total_squares - np.square(total) / denominator
    variance = np.maximum(variance_numerator / np.maximum(denominator - 1, 1), 0.0)
    standard_error = np.sqrt(variance / denominator)
    return mean, standard_error, graph_counts


def coordinate_decay_rows(
    displayed_tau: float,
    mean_decay: np.ndarray,
    standard_error: np.ndarray,
    graph_counts: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for bin_index, (q_low, q_high) in enumerate(LOCAL_QUANTILE_BINS):
        cumulative = np.cumsum(mean_decay[bin_index])
        for rank in range(COORDINATE_DECAY_MAX_RANK):
            rows.append(
                {
                    "displayed_tau": displayed_tau,
                    "quantile": 0.5 * (q_low + q_high),
                    "quantile_low": q_low,
                    "quantile_high": q_high,
                    "rank": rank + 1,
                    "number_of_graphs": int(graph_counts[bin_index]),
                    "mean_coordinate_mass": float(mean_decay[bin_index, rank]),
                    "coordinate_mass_standard_error": float(standard_error[bin_index, rank]),
                    "mean_cumulative_mass": float(cumulative[rank]),
                }
            )
    return rows


def coordinate_decay_fieldnames() -> list[str]:
    return [
        "displayed_tau", "quantile", "quantile_low", "quantile_high", "rank",
        "number_of_graphs", "mean_coordinate_mass", "coordinate_mass_standard_error",
        "mean_cumulative_mass",
    ]


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_summary(
    rows: list[dict[str, object]],
    displayed_tau: float,
    component_mode: str = COMPONENT_MODE,
):
    plt = _pyplot()
    q = np.asarray([float(row["quantile"]) for row in rows])
    benchmark_n = int(round(float(np.mean([float(row["analyzed_dimension"]) for row in rows]))))
    benchmarks = goe_benchmarks(benchmark_n)
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.8), sharex=True)

    n_ipr = np.asarray([float(row["mean_n_times_ipr"]) for row in rows])
    n_ipr_se = np.asarray([float(row["mean_n_times_ipr_standard_error"]) for row in rows])
    axes[0, 0].plot(q, n_ipr, color="black", lw=1.5)
    axes[0, 0].fill_between(q, n_ipr - n_ipr_se, n_ipr + n_ipr_se, color="0.7", alpha=0.35)
    axes[0, 0].axhline(benchmarks["mean_n_times_ipr_exact"], color="firebrick", ls="--", label="GOE/Haar")
    axes[0, 0].set_ylabel("mean N × IPR")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Inverse participation ratio")
    axes[0, 0].legend(frameon=False)

    for field, label in (
        ("mean_participation_fraction", "PR / N"),
        ("mean_entropy_support_fraction", "entropy support / N"),
        ("mean_numerical_support_fraction", "numerical nonzero fraction"),
    ):
        axes[0, 1].plot(q, [float(row[field]) for row in rows], label=label)
    axes[0, 1].axhline(1.0 / 3.0, color="firebrick", ls="--", label="GOE PR/N")
    axes[0, 1].set_ylabel("fraction of N")
    axes[0, 1].set_title("Effective and numerical support")
    axes[0, 1].legend(frameon=False)

    for top_r in (10, 50, 100, 200):
        field = f"mean_top_{top_r}_mass"
        axes[1, 0].plot(q, [float(row[field]) for row in rows], label=f"top {top_r}")
        axes[1, 0].axhline(
            benchmarks[f"top_{top_r}_mass_asymptotic"],
            color="0.6",
            ls=":",
            lw=0.8,
        )
    axes[1, 0].set_ylabel("cumulative coordinate mass")
    axes[1, 0].set_title("Top-r mass (dotted: GOE)")
    axes[1, 0].legend(frameon=False, ncol=2)

    for level in (0.50, 0.80, 0.90, 0.95):
        key = f"mean_r_{int(round(100 * level))}_fraction"
        axes[1, 1].plot(q, [float(row[key]) for row in rows], label=f"r{int(100 * level)} / N")
        axes[1, 1].axhline(
            benchmarks[f"r_{int(round(100 * level))}_fraction_asymptotic"],
            color="0.6",
            ls=":",
            lw=0.8,
        )
    axes[1, 1].set_ylabel("minimum coordinate fraction")
    axes[1, 1].set_title("Coordinates needed for target mass")
    axes[1, 1].legend(frameon=False, ncol=2)

    for axis in axes.flat:
        axis.set_xlabel("continuous non-Perron spectral quantile")
        axis.grid(alpha=0.2)
    figure.suptitle(
        f"Chung–Lu eigenvector localization, N={N}, displayed tau={displayed_tau:g}, "
        f"Δq={Q_WIDTH:g}, G={NUM_GRAPHS}, component={component_mode}"
    )
    figure.tight_layout()
    return figure


def plot_coordinate_heatmap(
    mean_decay: np.ndarray,
    displayed_tau: float,
    component_mode: str = COMPONENT_MODE,
):
    plt = _pyplot()
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
        f"Coordinate-mass concentration, displayed tau={displayed_tau:g}, "
        f"component={component_mode}"
    )
    figure.tight_layout()
    return figure


def code_version() -> str | None:
    repo = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def run_manifest(
    specification: LegacyTauSpecification,
    num_partial_files: int,
    component_mode: str = COMPONENT_MODE,
    benchmark_n: int = N,
) -> dict[str, object]:
    return {
        "model": "power_law_chung_lu",
        "diagnostic": "continuous_nonperron_eigenvector_equal_percentile_scan",
        "n": N,
        "num_graphs": NUM_GRAPHS,
        "displayed_tau": specification.displayed_tau,
        "tau_convention": specification.tau_convention,
        "effective_density_exponent_passed_to_generator": specification.effective_density_exponent,
        "legacy_mixed_tau_conventions_preserved": True,
        "source_eigenvalue_manifest": specification.source_eigenvalue_manifest,
        "target_mean_degree": TARGET_MEAN_DEGREE,
        "component_mode": component_mode,
        "sequence_mode": SEQUENCE_MODE,
        "w_min": W_MIN,
        "w_max": W_MAX,
        "cutoff_mode": CUTOFF_MODE,
        "cutoff_coefficient": specification.cutoff_coefficient,
        "graph_seed_base": specification.graph_seed_base,
        "sequence_seed_base": specification.sequence_seed_base,
        "same_graph_seed_stream_as_eigenvalue_run": True,
        "remove_perron_eigenpair": REMOVE_PERRON_EIGENPAIR,
        "remove_exact_zero_eigenpairs": REMOVE_EXACT_ZERO_EIGENPAIRS,
        "zero_eigenvalue_atol": ZERO_EIGENVALUE_ATOL,
        "quantile_definition": "rank within continuous non-Perron eigenpairs of each graph",
        "quantile_min": Q_MIN,
        "quantile_max": Q_MAX,
        "quantile_width": Q_WIDTH,
        "num_quantile_bins": len(LOCAL_QUANTILE_BINS),
        "coordinate_decay_max_rank": COORDINATE_DECAY_MAX_RANK,
        "top_r_values": TOP_R_VALUES,
        "mass_levels": MASS_LEVELS,
        "numerical_coordinate_atol": NUMERICAL_COORDINATE_ATOL,
        "aggregation": "graph-equal mean with graph-level standard errors",
        "num_partial_files": num_partial_files,
        "goe_haar_benchmark_dimension": benchmark_n,
        "goe_haar_benchmarks": goe_benchmarks(benchmark_n),
        "code_version": code_version(),
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def aggregate_parameter(
    displayed_tau: float,
    output_dir: Path | None = None,
    component_mode: str = COMPONENT_MODE,
) -> Path:
    plt = _pyplot()
    output_dir = output_dir or parameter_output_dir(displayed_tau)
    partial_dir = output_dir / "partials"
    csv_paths = sorted(partial_dir.glob("eigenvector_scan_graphs_*.csv.gz"))
    npz_paths = sorted(partial_dir.glob("eigenvector_scan_decay_*.npz"))
    if not csv_paths or not npz_paths:
        raise FileNotFoundError(f"missing partial files in {partial_dir}")
    summary_rows, graph_ids = aggregate_graph_csv_paths(
        csv_paths,
        output_dir / "eigenvector_percentile_scan_by_graph.csv.gz",
    )
    expected = list(range(NUM_GRAPHS))
    if graph_ids != expected:
        missing = sorted(set(expected).difference(graph_ids))
        raise ValueError(f"missing graph indices: {missing[:20]} (total={len(missing)})")
    if len(summary_rows) != len(LOCAL_QUANTILE_BINS):
        raise ValueError("aggregated summary has the wrong number of quantile bins")
    mean_decay, decay_se, graph_counts = aggregate_coordinate_decay(npz_paths)
    if not np.all(graph_counts == NUM_GRAPHS):
        raise ValueError(f"coordinate-decay graph counts are incomplete: {graph_counts}")
    decay_rows = coordinate_decay_rows(displayed_tau, mean_decay, decay_se, graph_counts)
    write_csv(summary_rows, output_dir / "eigenvector_percentile_scan_summary.csv", summary_row_fieldnames())
    write_csv(decay_rows, output_dir / "eigenvector_coordinate_decay_summary.csv", coordinate_decay_fieldnames())
    benchmark_n = int(round(float(np.mean([float(row["analyzed_dimension"]) for row in summary_rows]))))
    figure = plot_summary(summary_rows, displayed_tau, component_mode=component_mode)
    figure.savefig(output_dir / "eigenvector_percentile_scan_summary.png", dpi=200, bbox_inches="tight")
    plt.close(figure)
    figure = plot_coordinate_heatmap(mean_decay, displayed_tau, component_mode=component_mode)
    figure.savefig(output_dir / "eigenvector_coordinate_decay_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(figure)
    specification = tau_specification(displayed_tau)
    with (output_dir / "run_manifest.json").open("w") as handle:
        json.dump(
            run_manifest(
                specification,
                len(csv_paths),
                component_mode=component_mode,
                benchmark_n=benchmark_n,
            ),
            handle,
            indent=2,
            sort_keys=True,
        )
    return output_dir
