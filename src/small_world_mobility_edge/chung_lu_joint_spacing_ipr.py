from __future__ import annotations

import csv
import json
import math
import os
import tempfile
import subprocess
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "small-world-mobility-edge-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.stats import ks_2samp

try:
    from .chung_lu_baseline import (
        WeightSequenceDiagnostics,
        generate_chung_lu_graph_from_weights,
        generate_power_law_expected_weights,
    )
    from .regular_baseline import (
        adjacency_eigendecomposition,
        adjacent_gap_ratios_from_spacings,
        inverse_participation_ratio,
        participation_ratio,
    )
except ImportError:  # pragma: no cover
    from chung_lu_baseline import (
        WeightSequenceDiagnostics,
        generate_chung_lu_graph_from_weights,
        generate_power_law_expected_weights,
    )
    from regular_baseline import (
        adjacency_eigendecomposition,
        adjacent_gap_ratios_from_spacings,
        inverse_participation_ratio,
        participation_ratio,
    )


N = 5000
TAU_VALUES = [1.5]
TARGET_MEAN_DEGREES = [20.0]
ACTIVE_PARAMETER_PAIRS = [(1.5, 20.0)]
RUN_FULL_GRID = False
NUM_GRAPHS = 500
GRAPH_SEED = 27182
SEQUENCE_SEED = 14142
COMPONENT_MODE = "full"
SEQUENCE_MODE = "quantile"
W_MIN = 1.0
W_MAX = None
CUTOFF_MODE = "structural"
CUTOFF_COEFFICIENT = 1.0
REMOVE_EXACT_ZERO_EIGENVALUES = True
ZERO_EIGENVALUE_ATOL = 1e-8
LOCAL_QUANTILE_BINS = [
    (0.5000, 0.7000),
    (0.7000, 0.8500),
    (0.8500, 0.9200),
    (0.9200, 0.9600),
    (0.9600, 0.9800),
    (0.9800, 0.9900),
    (0.9900, 0.9950),
    (0.9950, 0.9980),
]
MIN_POOLED_GAPS_FOR_SPACING = 200
MIN_POOLED_GAP_RATIOS = 100
MIN_GRAPHS_WITH_GAPS = 20
SMALL_SPACING_THRESHOLD = 0.20
KS_SAMPLE_SIZE = 4096
OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "chung_lu"
    / "power_law"
    / "joint_spacing_ipr"
)


def code_version() -> str | None:
    repo = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def make_seeds(num_graphs: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(value) for value in rng.integers(0, 2**32 - 1, size=num_graphs, dtype=np.uint32)]


def parameter_seed_offset(tau: float, target_mean_degree: float) -> int:
    return int(100 * tau) + int(target_mean_degree)


def run_slug(tau: float, target_mean_degree: float, n: int | None = None) -> str:
    if n is None:
        n = N
    tau_slug = str(tau).replace(".", "p")
    degree_slug = str(target_mean_degree).replace(".", "p")
    return f"n_{n}_tau_{tau_slug}_avgdeg_{degree_slug}"


def parameter_output_dir(tau: float, target_mean_degree: float, n: int | None = None) -> Path:
    return OUTPUT_DIR / run_slug(tau=tau, target_mean_degree=target_mean_degree, n=n)


def active_parameter_pairs() -> list[tuple[float, float]]:
    if RUN_FULL_GRID:
        return [(tau, degree) for tau in TAU_VALUES for degree in TARGET_MEAN_DEGREES]
    return list(ACTIVE_PARAMETER_PAIRS)


def select_component(graph: nx.Graph, component_mode: str) -> nx.Graph:
    if component_mode == "full":
        return graph
    if component_mode == "largest_connected_component":
        largest_nodes = max(nx.connected_components(graph), key=len)
        return graph.subgraph(largest_nodes).copy()
    raise ValueError(f"unsupported component_mode={component_mode!r}")


def zero_eigenvalue_mask(eigenvalues: np.ndarray) -> np.ndarray:
    return np.isclose(np.asarray(eigenvalues, dtype=float), 0.0, atol=ZERO_EIGENVALUE_ATOL, rtol=0.0)


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


def encode_array(values: np.ndarray) -> str:
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return ""
    return " ".join(f"{value:.12g}" for value in array)


def decode_array(text: object) -> np.ndarray:
    if text is None:
        return np.array([], dtype=float)
    value = str(text).strip()
    if not value:
        return np.array([], dtype=float)
    return np.fromstring(value, sep=" ", dtype=float)


def eigenvector_localization_summary(eigenvectors: np.ndarray, analyzed_dimension: int) -> dict[str, float]:
    vectors = np.asarray(eigenvectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] == 0:
        return {
            "mean_ipr": math.nan,
            "median_ipr": math.nan,
            "mean_n_times_ipr": math.nan,
            "median_n_times_ipr": math.nan,
            "mean_participation_ratio": math.nan,
            "mean_participation_fraction": math.nan,
            "median_max_coordinate_squared": math.nan,
            "q90_max_coordinate_squared": math.nan,
            "mean_top_5_coordinate_mass": math.nan,
            "mean_top_10_coordinate_mass": math.nan,
        }

    ipr = inverse_participation_ratio(vectors)
    participation = participation_ratio(vectors)
    coordinate_mass = np.sort(np.abs(vectors) ** 2, axis=0)[::-1, :]
    top_5 = np.sum(coordinate_mass[: min(5, coordinate_mass.shape[0]), :], axis=0)
    top_10 = np.sum(coordinate_mass[: min(10, coordinate_mass.shape[0]), :], axis=0)
    max_coordinate_squared = coordinate_mass[0, :]
    return {
        "mean_ipr": float(np.mean(ipr)),
        "median_ipr": float(np.median(ipr)),
        "mean_n_times_ipr": float(analyzed_dimension * np.mean(ipr)),
        "median_n_times_ipr": float(analyzed_dimension * np.median(ipr)),
        "mean_participation_ratio": float(np.mean(participation)),
        "mean_participation_fraction": float(np.mean(participation) / max(float(analyzed_dimension), 1.0)),
        "median_max_coordinate_squared": float(np.median(max_coordinate_squared)),
        "q90_max_coordinate_squared": float(np.quantile(max_coordinate_squared, 0.90)),
        "mean_top_5_coordinate_mass": float(np.mean(top_5)),
        "mean_top_10_coordinate_mass": float(np.mean(top_10)),
    }


def graph_joint_rows(
    graph_index: int,
    graph_seed: int,
    sequence_seed: int,
    tau: float,
    target_mean_degree: float,
) -> tuple[list[dict[str, object]], WeightSequenceDiagnostics]:
    weights, diagnostics = generate_power_law_expected_weights(
        n=N,
        tau=tau,
        target_mean_degree=target_mean_degree,
        w_min=W_MIN,
        w_max=W_MAX,
        cutoff_mode=CUTOFF_MODE,
        cutoff_coefficient=CUTOFF_COEFFICIENT,
        sequence_mode=SEQUENCE_MODE,
        sequence_seed=sequence_seed,
    )
    if not diagnostics.admissibility_holds:
        raise ValueError(
            "final expected-weight sequence violates max(w)^2 <= sum(w); "
            f"max(w)^2={diagnostics.max_weight_squared:.6f}, sum(w)={diagnostics.total_weight:.6f}"
        )

    graph = generate_chung_lu_graph_from_weights(weights, seed=graph_seed)
    analyzed_graph = select_component(graph, COMPONENT_MODE)
    eigenvalues, eigenvectors = adjacency_eigendecomposition(analyzed_graph)
    bulk_eigenvalues = np.asarray(eigenvalues[:-1], dtype=float)
    bulk_vectors = np.asarray(eigenvectors[:, :-1], dtype=float)
    zero_count = int(np.sum(zero_eigenvalue_mask(bulk_eigenvalues)))
    if REMOVE_EXACT_ZERO_EIGENVALUES:
        keep = ~zero_eigenvalue_mask(bulk_eigenvalues)
        bulk_eigenvalues = bulk_eigenvalues[keep]
        bulk_vectors = bulk_vectors[:, keep]

    analyzed_dimension = analyzed_graph.number_of_nodes()
    degrees = np.array([degree for _, degree in graph.degree()], dtype=float)
    num_components = nx.number_connected_components(graph)
    largest_component_size = max(len(component) for component in nx.connected_components(graph))
    num_isolated_vertices = int(np.sum(degrees == 0))

    rows: list[dict[str, object]] = []
    for q_low, q_high in LOCAL_QUANTILE_BINS:
        start, stop = quantile_index_bin(num_values=len(bulk_eigenvalues), q_low=q_low, q_high=q_high)
        selected_values = bulk_eigenvalues[start:stop]
        selected_vectors = bulk_vectors[:, start:stop]
        raw_gaps = np.diff(selected_values)
        positive_gaps = raw_gaps[raw_gaps > 0.0]
        if len(positive_gaps) > 0:
            normalized_spacings = positive_gaps / float(np.mean(positive_gaps))
        else:
            normalized_spacings = np.array([], dtype=float)
        gap_ratios = adjacent_gap_ratios_from_spacings(positive_gaps)
        summary = eigenvector_localization_summary(selected_vectors, analyzed_dimension=analyzed_dimension)
        rows.append(
            {
                "graph_index": graph_index,
                "graph_seed": graph_seed,
                "sequence_seed": sequence_seed,
                "tau": tau,
                "target_mean_degree": target_mean_degree,
                "window_mode": "paired_rank_quantile_bin",
                "unfolding_method": "within_graph_bin_mean_gap",
                "quantile": 0.5 * (q_low + q_high),
                "quantile_low": q_low,
                "quantile_high": q_high,
                "index_start": start,
                "index_stop": stop,
                "number_of_eigenvalues": int(len(selected_values)),
                "number_of_gaps": int(len(normalized_spacings)),
                "number_of_gap_ratios": int(len(gap_ratios)),
                "E_low": float(selected_values[0]),
                "E_high": float(selected_values[-1]),
                "E_center": float(np.median(selected_values)),
                "mean_raw_gap": float(np.mean(positive_gaps)) if len(positive_gaps) > 0 else math.nan,
                "analyzed_dimension": analyzed_dimension,
                "continuous_bulk_dimension": int(len(bulk_eigenvalues)),
                "zero_eigenvalue_count": zero_count,
                "zero_eigenvalue_fraction": float(zero_count / max(len(eigenvalues) - 1, 1)),
                "num_components": num_components,
                "largest_component_size": largest_component_size,
                "num_isolated_vertices": num_isolated_vertices,
                "realized_mean_degree": float(np.mean(degrees)),
                "max_realized_degree": int(np.max(degrees)),
                "max_expected_weight": diagnostics.final_max,
                **summary,
                "normalized_spacings": encode_array(normalized_spacings),
                "gap_ratios": encode_array(gap_ratios),
            }
        )
    return rows, diagnostics


def summarize_graph_level_ratios(ratio_samples: list[np.ndarray]) -> tuple[float, float, float]:
    nonempty = [values for values in ratio_samples if len(values) > 0]
    if not nonempty:
        return math.nan, math.nan, math.nan
    pooled = np.concatenate(nonempty)
    graph_means = np.array([float(np.mean(values)) for values in nonempty], dtype=float)
    se = float(np.std(graph_means, ddof=1) / np.sqrt(len(graph_means))) if len(graph_means) > 1 else math.nan
    return float(np.mean(pooled)), float(np.mean(graph_means)), se


def aggregate_quantile_rows(rows: list[dict[str, object]], rng_seed: int = 8675309) -> list[dict[str, object]]:
    rng = np.random.default_rng(rng_seed)
    quantiles = sorted({float(row["quantile"]) for row in rows})
    summary_rows: list[dict[str, object]] = []
    for q in quantiles:
        q_rows = [row for row in rows if float(row["quantile"]) == q]
        first = q_rows[0]
        spacings_by_graph = [decode_array(row.get("normalized_spacings", "")) for row in q_rows]
        ratios_by_graph = [decode_array(row.get("gap_ratios", "")) for row in q_rows]
        spacings = (
            np.concatenate([values for values in spacings_by_graph if len(values) > 0])
            if any(len(values) > 0 for values in spacings_by_graph)
            else np.array([], dtype=float)
        )
        ratios = (
            np.concatenate([values for values in ratios_by_graph if len(values) > 0])
            if any(len(values) > 0 for values in ratios_by_graph)
            else np.array([], dtype=float)
        )
        pooled_mean_r, graph_equal_mean_r, r_se = summarize_graph_level_ratios(ratios_by_graph)
        graphs_with_gaps = int(sum(len(values) > 0 for values in spacings_by_graph))
        status_parts = []
        if len(spacings) < MIN_POOLED_GAPS_FOR_SPACING:
            status_parts.append("insufficient_pooled_gaps")
        if len(ratios) < MIN_POOLED_GAP_RATIOS:
            status_parts.append("insufficient_gap_ratios")
        if graphs_with_gaps < MIN_GRAPHS_WITH_GAPS:
            status_parts.append("insufficient_graphs_with_gaps")
        status = "ok" if not status_parts else ";".join(status_parts)
        if len(spacings) > 0:
            sample_size = max(KS_SAMPLE_SIZE, len(spacings))
            goe_sample = rng.rayleigh(scale=np.sqrt(2.0 / np.pi), size=sample_size)
            poisson_sample = rng.exponential(scale=1.0, size=sample_size)
            ks_to_goe = float(ks_2samp(spacings, goe_sample).statistic)
            ks_to_poisson = float(ks_2samp(spacings, poisson_sample).statistic)
            small_gap_probability = float(np.mean(spacings < SMALL_SPACING_THRESHOLD))
            mean_spacing = float(np.mean(spacings))
        else:
            ks_to_goe = math.nan
            ks_to_poisson = math.nan
            small_gap_probability = math.nan
            mean_spacing = math.nan
        weights = np.array([float(row["number_of_eigenvalues"]) for row in q_rows], dtype=float)
        weights = np.where(weights > 0.0, weights, 1.0)
        summary_rows.append(
            {
                "tau": first["tau"],
                "target_mean_degree": first["target_mean_degree"],
                "window_mode": first["window_mode"],
                "unfolding_method": first["unfolding_method"],
                "quantile": q,
                "quantile_low": first["quantile_low"],
                "quantile_high": first["quantile_high"],
                "number_of_graphs": len(q_rows),
                "graphs_with_gaps": graphs_with_gaps,
                "number_of_eigenvalues": int(sum(int(row["number_of_eigenvalues"]) for row in q_rows)),
                "number_of_gaps": int(len(spacings)),
                "number_of_gap_ratios": int(len(ratios)),
                "mean_E_center": float(np.average([float(row["E_center"]) for row in q_rows], weights=weights)),
                "mean_spacing": mean_spacing,
                "pooled_mean_gap_ratio": pooled_mean_r,
                "graph_equal_mean_gap_ratio": graph_equal_mean_r,
                "gap_ratio_standard_error": r_se,
                "small_gap_probability": small_gap_probability,
                "KS_to_GOE": ks_to_goe,
                "KS_to_Poisson": ks_to_poisson,
                "mean_n_times_ipr": float(np.average([float(row["mean_n_times_ipr"]) for row in q_rows], weights=weights)),
                "median_n_times_ipr": float(np.average([float(row["median_n_times_ipr"]) for row in q_rows], weights=weights)),
                "mean_participation_fraction": float(
                    np.average([float(row["mean_participation_fraction"]) for row in q_rows], weights=weights)
                ),
                "mean_top_5_coordinate_mass": float(
                    np.average([float(row["mean_top_5_coordinate_mass"]) for row in q_rows], weights=weights)
                ),
                "mean_top_10_coordinate_mass": float(
                    np.average([float(row["mean_top_10_coordinate_mass"]) for row in q_rows], weights=weights)
                ),
                "q90_max_coordinate_squared": float(
                    np.average([float(row["q90_max_coordinate_squared"]) for row in q_rows], weights=weights)
                ),
                "mean_zero_eigenvalue_fraction": float(np.mean([float(row["zero_eigenvalue_fraction"]) for row in q_rows])),
                "status": status,
            }
        )
    return summary_rows


def write_csv(rows: list[dict[str, object]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def graph_row_fieldnames() -> list[str]:
    return [
        "graph_index", "graph_seed", "sequence_seed", "tau", "target_mean_degree", "window_mode",
        "unfolding_method", "quantile", "quantile_low", "quantile_high", "index_start", "index_stop",
        "number_of_eigenvalues", "number_of_gaps", "number_of_gap_ratios", "E_low", "E_high",
        "E_center", "mean_raw_gap", "analyzed_dimension", "continuous_bulk_dimension",
        "zero_eigenvalue_count", "zero_eigenvalue_fraction", "num_components", "largest_component_size",
        "num_isolated_vertices", "realized_mean_degree", "max_realized_degree", "max_expected_weight",
        "mean_ipr", "median_ipr", "mean_n_times_ipr", "median_n_times_ipr",
        "mean_participation_ratio", "mean_participation_fraction", "median_max_coordinate_squared",
        "q90_max_coordinate_squared", "mean_top_5_coordinate_mass", "mean_top_10_coordinate_mass",
        "normalized_spacings", "gap_ratios",
    ]


def summary_row_fieldnames() -> list[str]:
    return [
        "tau", "target_mean_degree", "window_mode", "unfolding_method", "quantile", "quantile_low",
        "quantile_high", "number_of_graphs", "graphs_with_gaps", "number_of_eigenvalues",
        "number_of_gaps", "number_of_gap_ratios", "mean_E_center", "mean_spacing",
        "pooled_mean_gap_ratio", "graph_equal_mean_gap_ratio", "gap_ratio_standard_error",
        "small_gap_probability", "KS_to_GOE", "KS_to_Poisson", "mean_n_times_ipr",
        "median_n_times_ipr", "mean_participation_fraction", "mean_top_5_coordinate_mass",
        "mean_top_10_coordinate_mass", "q90_max_coordinate_squared", "mean_zero_eigenvalue_fraction",
        "status",
    ]


def read_rows(paths: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        with path.open(newline="") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    return rows


def plot_joint_summary(rows: list[dict[str, object]], tau: float, target_mean_degree: float) -> plt.Figure:
    quantiles = np.array([float(row["quantile"]) for row in rows], dtype=float)
    mean_r = np.array([float(row["graph_equal_mean_gap_ratio"]) for row in rows], dtype=float)
    mean_r_se = np.array([float(row["gap_ratio_standard_error"]) for row in rows], dtype=float)
    ks_goe = np.array([float(row["KS_to_GOE"]) for row in rows], dtype=float)
    ks_poisson = np.array([float(row["KS_to_Poisson"]) for row in rows], dtype=float)
    n_ipr = np.array([float(row["mean_n_times_ipr"]) for row in rows], dtype=float)
    top_10 = np.array([float(row["mean_top_10_coordinate_mass"]) for row in rows], dtype=float)
    gap_counts = np.array([int(row["number_of_gaps"]) for row in rows], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2), sharex=True)
    axes[0, 0].plot(quantiles, mean_r, marker="o", color="black")
    axes[0, 0].fill_between(quantiles, mean_r - mean_r_se, mean_r + mean_r_se, color="0.75", alpha=0.35)
    axes[0, 0].axhline(0.53590, color="black", ls="--", lw=1.1, label="GOE")
    axes[0, 0].axhline(0.38629, color="firebrick", ls="--", lw=1.1, label="Poisson")
    axes[0, 0].set_ylabel("mean adjacent gap ratio")
    axes[0, 0].set_title("Spacing")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(quantiles, ks_goe, marker="o", color="black", label="KS to GOE")
    axes[0, 1].plot(quantiles, ks_poisson, marker="o", color="firebrick", label="KS to Poisson")
    axes[0, 1].set_ylabel("KS distance")
    axes[0, 1].set_title("Spacing Distribution Distance")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(quantiles, n_ipr, marker="o", color="steelblue", label="n * IPR")
    axes[1, 0].set_ylabel("mean n * IPR")
    twin = axes[1, 0].twinx()
    twin.plot(quantiles, top_10, marker="s", color="darkorange", label="top-10 mass")
    twin.set_ylabel("mean top-10 coordinate mass")
    axes[1, 0].set_title("Localization")

    axes[1, 1].plot(quantiles, gap_counts, marker="o", color="seagreen")
    axes[1, 1].axhline(MIN_POOLED_GAPS_FOR_SPACING, color="0.35", ls="--", lw=1.1)
    axes[1, 1].set_ylabel("pooled gaps")
    axes[1, 1].set_title("Spacing Sample Size")

    for ax in axes.flat:
        ax.set_xlabel("spectral quantile midpoint")
        ax.grid(alpha=0.2)
    fig.suptitle(
        f"Paired Chung-Lu spacing and IPR, n={N}, tau={tau:.2f}, mean degree={target_mean_degree:.1f}",
        y=1.01,
    )
    fig.tight_layout()
    return fig


def manifest(tau: float, target_mean_degree: float, num_partial_files: int | None = None) -> dict[str, object]:
    offset = parameter_seed_offset(tau=tau, target_mean_degree=target_mean_degree)
    data = {
        "model": "power_law_chung_lu",
        "diagnostic": "paired_rank_quantile_spacing_ipr",
        "n": N,
        "tau": tau,
        "target_mean_degree": target_mean_degree,
        "num_graphs": NUM_GRAPHS,
        "local_quantile_bins": LOCAL_QUANTILE_BINS,
        "min_pooled_gaps_for_spacing": MIN_POOLED_GAPS_FOR_SPACING,
        "min_pooled_gap_ratios": MIN_POOLED_GAP_RATIOS,
        "min_graphs_with_gaps": MIN_GRAPHS_WITH_GAPS,
        "unfolding_method": "within_graph_bin_mean_gap",
        "remove_exact_zero_eigenvalues": REMOVE_EXACT_ZERO_EIGENVALUES,
        "zero_eigenvalue_atol": ZERO_EIGENVALUE_ATOL,
        "sequence_mode": SEQUENCE_MODE,
        "cutoff_mode": CUTOFF_MODE,
        "cutoff_coefficient": CUTOFF_COEFFICIENT,
        "w_min": W_MIN,
        "w_max": W_MAX,
        "component_mode": COMPONENT_MODE,
        "graph_seed": GRAPH_SEED + offset,
        "sequence_seed": SEQUENCE_SEED + offset,
        "active_parameter_pairs": ACTIVE_PARAMETER_PAIRS,
        "run_full_grid": RUN_FULL_GRID,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "code_version": code_version(),
    }
    if num_partial_files is not None:
        data["num_partial_files"] = num_partial_files
    return data


def write_outputs(
    rows: list[dict[str, object]],
    tau: float,
    target_mean_degree: float,
    output_dir: Path,
    num_partial_files: int | None = None,
) -> list[dict[str, object]]:
    summary_rows = aggregate_quantile_rows(rows)
    write_csv(rows, output_dir / "joint_spacing_ipr_by_graph.csv", graph_row_fieldnames())
    write_csv(summary_rows, output_dir / "joint_spacing_ipr_summary.csv", summary_row_fieldnames())
    fig = plot_joint_summary(summary_rows, tau=tau, target_mean_degree=target_mean_degree)
    fig.savefig(output_dir / "joint_spacing_ipr_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    with (output_dir / "run_manifest.json").open("w") as handle:
        json.dump(manifest(tau=tau, target_mean_degree=target_mean_degree, num_partial_files=num_partial_files), handle, indent=2, sort_keys=True)
    return summary_rows


def run_parameter_pair(tau: float, target_mean_degree: float) -> tuple[list[dict[str, object]], list[dict[str, object]], Path]:
    output_dir = parameter_output_dir(tau=tau, target_mean_degree=target_mean_degree)
    output_dir.mkdir(parents=True, exist_ok=True)
    offset = parameter_seed_offset(tau=tau, target_mean_degree=target_mean_degree)
    graph_seeds = make_seeds(NUM_GRAPHS, GRAPH_SEED + offset)
    sequence_seeds = make_seeds(NUM_GRAPHS, SEQUENCE_SEED + offset)
    rows: list[dict[str, object]] = []
    for graph_index, (graph_seed, sequence_seed) in enumerate(zip(graph_seeds, sequence_seeds)):
        graph_rows, _ = graph_joint_rows(
            graph_index=graph_index,
            graph_seed=graph_seed,
            sequence_seed=sequence_seed,
            tau=tau,
            target_mean_degree=target_mean_degree,
        )
        rows.extend(graph_rows)
        print(f"tau={tau}, degree={target_mean_degree}: finished graph {graph_index + 1}/{NUM_GRAPHS}", flush=True)
    summary_rows = write_outputs(rows, tau=tau, target_mean_degree=target_mean_degree, output_dir=output_dir)
    print(f"saved paired joint spacing/IPR diagnostics to {output_dir}")
    return rows, summary_rows, output_dir


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for tau, target_mean_degree in active_parameter_pairs():
        run_parameter_pair(tau=tau, target_mean_degree=target_mean_degree)


if __name__ == "__main__":
    main()
