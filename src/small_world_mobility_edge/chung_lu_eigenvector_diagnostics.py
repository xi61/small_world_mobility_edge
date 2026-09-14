from __future__ import annotations

import csv
import json
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

try:
    from .chung_lu_baseline import (
        WeightSequenceDiagnostics,
        generate_chung_lu_graph_from_weights,
        generate_power_law_expected_weights,
    )
    from .regular_baseline import adjacency_eigendecomposition, inverse_participation_ratio, participation_ratio
    from .chung_lu_spacing_by_energy import (
        COMPONENT_MODE as SPACING_COMPONENT_MODE,
        CUTOFF_COEFFICIENT as SPACING_CUTOFF_COEFFICIENT,
        CUTOFF_MODE as SPACING_CUTOFF_MODE,
        LOCAL_QUANTILE_BINS as SPACING_LOCAL_QUANTILE_BINS,
        LOCAL_QUANTILES as SPACING_LOCAL_QUANTILES,
        LOCAL_TARGET_EIGENVALUES_PER_GRAPH as SPACING_LOCAL_TARGET_EIGENVALUES_PER_GRAPH,
        N as SPACING_N,
        NUM_SPACING_GRAPHS,
        SEQUENCE_MODE as SPACING_SEQUENCE_MODE,
        SPACING_GRAPH_SEED,
        SPACING_SEQUENCE_SEED,
        W_MAX as SPACING_W_MAX,
        W_MIN as SPACING_W_MIN,
        WINDOW_MODE as SPACING_WINDOW_MODE,
        validate_quantile_bins,
    )
except ImportError:  # pragma: no cover
    from chung_lu_baseline import (
        WeightSequenceDiagnostics,
        generate_chung_lu_graph_from_weights,
        generate_power_law_expected_weights,
    )
    from regular_baseline import adjacency_eigendecomposition, inverse_participation_ratio, participation_ratio
    from chung_lu_spacing_by_energy import (
        COMPONENT_MODE as SPACING_COMPONENT_MODE,
        CUTOFF_COEFFICIENT as SPACING_CUTOFF_COEFFICIENT,
        CUTOFF_MODE as SPACING_CUTOFF_MODE,
        LOCAL_QUANTILE_BINS as SPACING_LOCAL_QUANTILE_BINS,
        LOCAL_QUANTILES as SPACING_LOCAL_QUANTILES,
        LOCAL_TARGET_EIGENVALUES_PER_GRAPH as SPACING_LOCAL_TARGET_EIGENVALUES_PER_GRAPH,
        N as SPACING_N,
        NUM_SPACING_GRAPHS,
        SEQUENCE_MODE as SPACING_SEQUENCE_MODE,
        SPACING_GRAPH_SEED,
        SPACING_SEQUENCE_SEED,
        W_MAX as SPACING_W_MAX,
        W_MIN as SPACING_W_MIN,
        WINDOW_MODE as SPACING_WINDOW_MODE,
        validate_quantile_bins,
    )


N = SPACING_N
TAU_VALUES = [1.5, 2.1, 2.5, 3.0, 3.5, 4.0]
TARGET_MEAN_DEGREES = [20.0]
ACTIVE_PARAMETER_PAIRS = [(1.5, 20.0)]
RUN_FULL_GRID = False
NUM_GRAPHS = NUM_SPACING_GRAPHS
GRAPH_SEED = SPACING_GRAPH_SEED
SEQUENCE_SEED = SPACING_SEQUENCE_SEED
COMPONENT_MODE = SPACING_COMPONENT_MODE
SEQUENCE_MODE = SPACING_SEQUENCE_MODE
W_MIN = SPACING_W_MIN
W_MAX = SPACING_W_MAX
CUTOFF_MODE = SPACING_CUTOFF_MODE
CUTOFF_COEFFICIENT = SPACING_CUTOFF_COEFFICIENT
WINDOW_MODE = SPACING_WINDOW_MODE
LOCAL_QUANTILES = list(SPACING_LOCAL_QUANTILES)
LOCAL_QUANTILE_BINS = list(SPACING_LOCAL_QUANTILE_BINS)
TARGET_EIGENVECTORS_PER_GRAPH = SPACING_LOCAL_TARGET_EIGENVALUES_PER_GRAPH
COORDINATE_DECAY_MAX_RANK = 200
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "chung_lu" / "eigenvector_diagnostics"


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


def run_slug(tau: float, target_mean_degree: float, n: int = N) -> str:
    tau_slug = str(tau).replace(".", "p")
    degree_slug = str(target_mean_degree).replace(".", "p")
    return f"n_{n}_tau_{tau_slug}_avgdeg_{degree_slug}"


def parameter_output_dir(tau: float, target_mean_degree: float, n: int = N) -> Path:
    return OUTPUT_DIR / run_slug(tau=tau, target_mean_degree=target_mean_degree, n=n)


def select_component(graph: nx.Graph, component_mode: str) -> nx.Graph:
    if component_mode == "full":
        return graph
    if component_mode == "largest_connected_component":
        largest_nodes = max(nx.connected_components(graph), key=len)
        return graph.subgraph(largest_nodes).copy()
    raise ValueError(f"unsupported component_mode={component_mode!r}")


def quantile_index_window(num_values: int, quantile: float, target_count: int) -> tuple[int, int]:
    if num_values <= 0:
        raise ValueError("num_values must be positive")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie strictly between 0 and 1")
    count = max(1, min(int(target_count), num_values))
    center = int(round(quantile * (num_values - 1)))
    start = max(0, center - count // 2)
    stop = min(num_values, start + count)
    start = max(0, stop - count)
    return start, stop


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


def localization_windows(num_values: int) -> list[dict[str, object]]:
    if WINDOW_MODE == "nonoverlapping_quantile_bins":
        validate_quantile_bins(LOCAL_QUANTILE_BINS)
        windows = []
        for q_low, q_high in LOCAL_QUANTILE_BINS:
            start, stop = quantile_index_bin(num_values=num_values, q_low=q_low, q_high=q_high)
            windows.append(
                {
                    "window_mode": WINDOW_MODE,
                    "quantile": 0.5 * (q_low + q_high),
                    "quantile_low": q_low,
                    "quantile_high": q_high,
                    "index_start": start,
                    "index_stop": stop,
                }
            )
        return windows

    windows = []
    for q in LOCAL_QUANTILES:
        start, stop = quantile_index_window(
            num_values=num_values,
            quantile=q,
            target_count=TARGET_EIGENVECTORS_PER_GRAPH,
        )
        windows.append(
            {
                "window_mode": WINDOW_MODE,
                "quantile": q,
                "quantile_low": np.nan,
                "quantile_high": np.nan,
                "index_start": start,
                "index_stop": stop,
            }
        )
    return windows


def eigenvector_localization_summary(
    eigenvectors: np.ndarray,
    analyzed_dimension: int,
) -> dict[str, float]:
    vectors = np.asarray(eigenvectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] == 0:
        return {
            "mean_ipr": np.nan,
            "median_ipr": np.nan,
            "mean_n_times_ipr": np.nan,
            "median_n_times_ipr": np.nan,
            "mean_participation_ratio": np.nan,
            "mean_participation_fraction": np.nan,
            "median_max_coordinate_squared": np.nan,
            "q90_max_coordinate_squared": np.nan,
            "mean_top_5_coordinate_mass": np.nan,
            "mean_top_10_coordinate_mass": np.nan,
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


def coordinate_decay_profile_rows(
    eigenvectors: np.ndarray,
    max_rank: int = COORDINATE_DECAY_MAX_RANK,
) -> list[dict[str, float]]:
    """Average sorted coordinate masses for a group of eigenvector columns."""
    vectors = np.asarray(eigenvectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] == 0:
        return []

    coordinate_mass = np.sort(np.abs(vectors) ** 2, axis=0)[::-1, :]
    rank_count = min(max(1, int(max_rank)), coordinate_mass.shape[0])
    mean_mass = np.mean(coordinate_mass[:rank_count, :], axis=1)
    mean_cumulative_mass = np.cumsum(mean_mass)
    return [
        {
            "rank": float(rank + 1),
            "mean_coordinate_mass": float(mean_mass[rank]),
            "mean_cumulative_mass": float(mean_cumulative_mass[rank]),
        }
        for rank in range(rank_count)
    ]


def graph_localization_and_decay_rows(
    graph_index: int,
    graph_seed: int,
    sequence_seed: int,
    tau: float,
    target_mean_degree: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], WeightSequenceDiagnostics]:
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
    analyzed_dimension = analyzed_graph.number_of_nodes()
    degrees = np.array([degree for _, degree in graph.degree()], dtype=float)
    num_components = nx.number_connected_components(graph)
    largest_component_size = max(len(component) for component in nx.connected_components(graph))
    num_isolated_vertices = int(np.sum(degrees == 0))

    rows: list[dict[str, object]] = []
    decay_rows: list[dict[str, object]] = []
    for window in localization_windows(num_values=len(bulk_eigenvalues)):
        start = int(window["index_start"])
        stop = int(window["index_stop"])
        selected_values = bulk_eigenvalues[start:stop]
        selected_vectors = bulk_vectors[:, start:stop]
        summary = eigenvector_localization_summary(
            selected_vectors,
            analyzed_dimension=analyzed_dimension,
        )
        rows.append(
            {
                "graph_index": graph_index,
                "graph_seed": graph_seed,
                "sequence_seed": sequence_seed,
                "tau": tau,
                "target_mean_degree": target_mean_degree,
                "window_mode": window["window_mode"],
                "quantile": window["quantile"],
                "quantile_low": window["quantile_low"],
                "quantile_high": window["quantile_high"],
                "index_start": start,
                "index_stop": stop,
                "number_of_eigenvectors": int(stop - start),
                "E_low": float(selected_values[0]),
                "E_high": float(selected_values[-1]),
                "E_center": float(np.median(selected_values)),
                "analyzed_dimension": analyzed_dimension,
                "num_components": num_components,
                "largest_component_size": largest_component_size,
                "num_isolated_vertices": num_isolated_vertices,
                "realized_mean_degree": float(np.mean(degrees)),
                "max_realized_degree": int(np.max(degrees)),
                "max_expected_weight": diagnostics.final_max,
                **summary,
            }
        )
        for decay in coordinate_decay_profile_rows(selected_vectors):
            decay_rows.append(
                {
                    "graph_index": graph_index,
                    "graph_seed": graph_seed,
                    "sequence_seed": sequence_seed,
                    "tau": tau,
                    "target_mean_degree": target_mean_degree,
                    "window_mode": window["window_mode"],
                    "quantile": window["quantile"],
                    "quantile_low": window["quantile_low"],
                    "quantile_high": window["quantile_high"],
                    "index_start": start,
                    "index_stop": stop,
                    "number_of_eigenvectors": int(stop - start),
                    "E_center": float(np.median(selected_values)),
                    "analyzed_dimension": analyzed_dimension,
                    **decay,
                }
            )
    return rows, decay_rows, diagnostics


def graph_localization_rows(
    graph_index: int,
    graph_seed: int,
    sequence_seed: int,
    tau: float,
    target_mean_degree: float,
) -> tuple[list[dict[str, object]], WeightSequenceDiagnostics]:
    rows, _, diagnostics = graph_localization_and_decay_rows(
        graph_index=graph_index,
        graph_seed=graph_seed,
        sequence_seed=sequence_seed,
        tau=tau,
        target_mean_degree=target_mean_degree,
    )
    return rows, diagnostics


def aggregate_quantile_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    quantiles = sorted({float(row["quantile"]) for row in rows})
    summary_rows: list[dict[str, object]] = []
    for q in quantiles:
        q_rows = [row for row in rows if float(row["quantile"]) == q]
        first = q_rows[0]
        summary_rows.append(
            {
                "tau": first["tau"],
                "target_mean_degree": first["target_mean_degree"],
                "window_mode": first["window_mode"],
                "quantile": q,
                "quantile_low": first["quantile_low"],
                "quantile_high": first["quantile_high"],
                "number_of_graphs": len(q_rows),
                "number_of_eigenvectors": int(sum(int(row["number_of_eigenvectors"]) for row in q_rows)),
                "mean_E_center": float(np.mean([float(row["E_center"]) for row in q_rows])),
                "mean_n_times_ipr": float(np.mean([float(row["mean_n_times_ipr"]) for row in q_rows])),
                "median_n_times_ipr": float(np.mean([float(row["median_n_times_ipr"]) for row in q_rows])),
                "mean_participation_fraction": float(np.mean([float(row["mean_participation_fraction"]) for row in q_rows])),
                "mean_top_5_coordinate_mass": float(np.mean([float(row["mean_top_5_coordinate_mass"]) for row in q_rows])),
                "mean_top_10_coordinate_mass": float(np.mean([float(row["mean_top_10_coordinate_mass"]) for row in q_rows])),
                "q90_max_coordinate_squared": float(np.mean([float(row["q90_max_coordinate_squared"]) for row in q_rows])),
            }
        )
    return summary_rows


def aggregate_coordinate_decay_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped_rows: dict[tuple[float, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (float(row["quantile"]), int(float(row["rank"])))
        grouped_rows.setdefault(key, []).append(row)

    summary_rows: list[dict[str, object]] = []
    for q, rank in sorted(grouped_rows):
        q_rank_rows = grouped_rows[(q, rank)]
        first = q_rank_rows[0]
        weights = np.array([float(row["number_of_eigenvectors"]) for row in q_rank_rows], dtype=float)
        if float(np.sum(weights)) <= 0.0:
            weights = np.ones(len(q_rank_rows), dtype=float)
        coordinate_mass = np.array([float(row["mean_coordinate_mass"]) for row in q_rank_rows], dtype=float)
        cumulative_mass = np.array([float(row["mean_cumulative_mass"]) for row in q_rank_rows], dtype=float)
        summary_rows.append(
            {
                "tau": first["tau"],
                "target_mean_degree": first["target_mean_degree"],
                "window_mode": first["window_mode"],
                "quantile": q,
                "quantile_low": first["quantile_low"],
                "quantile_high": first["quantile_high"],
                "rank": rank,
                "number_of_graphs": len(q_rank_rows),
                "number_of_eigenvectors": int(np.sum(weights)),
                "mean_E_center": float(np.average([float(row["E_center"]) for row in q_rank_rows], weights=weights)),
                "mean_coordinate_mass": float(np.average(coordinate_mass, weights=weights)),
                "mean_cumulative_mass": float(np.average(cumulative_mass, weights=weights)),
            }
        )
    return summary_rows


def write_csv(rows: list[dict[str, object]], path: Path, fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def graph_row_fieldnames() -> list[str]:
    return [
        "graph_index", "graph_seed", "sequence_seed", "tau", "target_mean_degree", "window_mode",
        "quantile", "quantile_low", "quantile_high",
        "index_start", "index_stop", "number_of_eigenvectors", "E_low", "E_high", "E_center",
        "analyzed_dimension", "num_components", "largest_component_size", "num_isolated_vertices",
        "realized_mean_degree", "max_realized_degree", "max_expected_weight", "mean_ipr",
        "median_ipr", "mean_n_times_ipr", "median_n_times_ipr", "mean_participation_ratio",
        "mean_participation_fraction", "median_max_coordinate_squared", "q90_max_coordinate_squared",
        "mean_top_5_coordinate_mass", "mean_top_10_coordinate_mass",
    ]


def coordinate_decay_row_fieldnames() -> list[str]:
    return [
        "graph_index", "graph_seed", "sequence_seed", "tau", "target_mean_degree", "window_mode",
        "quantile", "quantile_low", "quantile_high", "index_start", "index_stop",
        "number_of_eigenvectors", "E_center", "analyzed_dimension", "rank",
        "mean_coordinate_mass", "mean_cumulative_mass",
    ]


def summary_row_fieldnames() -> list[str]:
    return [
        "tau", "target_mean_degree", "window_mode", "quantile", "quantile_low", "quantile_high",
        "number_of_graphs", "number_of_eigenvectors",
        "mean_E_center", "mean_n_times_ipr", "median_n_times_ipr", "mean_participation_fraction",
        "mean_top_5_coordinate_mass", "mean_top_10_coordinate_mass", "q90_max_coordinate_squared",
    ]


def coordinate_decay_summary_fieldnames() -> list[str]:
    return [
        "tau", "target_mean_degree", "window_mode", "quantile", "quantile_low", "quantile_high",
        "rank", "number_of_graphs", "number_of_eigenvectors", "mean_E_center",
        "mean_coordinate_mass", "mean_cumulative_mass",
    ]


def plot_localization_summary(rows: list[dict[str, object]], tau: float, target_mean_degree: float) -> plt.Figure:
    quantiles = np.array([float(row["quantile"]) for row in rows], dtype=float)
    n_ipr = np.array([float(row["mean_n_times_ipr"]) for row in rows], dtype=float)
    participation_fraction = np.array([float(row["mean_participation_fraction"]) for row in rows], dtype=float)
    top_10_mass = np.array([float(row["mean_top_10_coordinate_mass"]) for row in rows], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].plot(quantiles, n_ipr, marker="o", color="black")
    axes[0].set_title("Mean n * IPR")
    axes[0].set_xlabel("spectral quantile")
    axes[0].set_ylabel("n * IPR")

    axes[1].plot(quantiles, participation_fraction, marker="o", color="steelblue")
    axes[1].set_title("Participation Fraction")
    axes[1].set_xlabel("spectral quantile")
    axes[1].set_ylabel("PR / n")

    axes[2].plot(quantiles, top_10_mass, marker="o", color="firebrick")
    axes[2].set_title("Top-10 Coordinate Mass")
    axes[2].set_xlabel("spectral quantile")
    axes[2].set_ylabel("mass")

    fig.suptitle(
        f"Chung-Lu Eigenvector Localization, n={N}, tau={tau:.2f}, mean degree={target_mean_degree:.1f}",
        y=1.02,
    )
    fig.tight_layout()
    return fig


def plot_coordinate_decay_summary(rows: list[dict[str, object]], tau: float, target_mean_degree: float) -> plt.Figure:
    quantiles = sorted({float(row["quantile"]) for row in rows})
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(quantiles)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for color, q in zip(colors, quantiles):
        q_rows = sorted((row for row in rows if float(row["quantile"]) == q), key=lambda row: int(row["rank"]))
        ranks = np.array([int(row["rank"]) for row in q_rows], dtype=float)
        mass = np.array([float(row["mean_coordinate_mass"]) for row in q_rows], dtype=float)
        cumulative = np.array([float(row["mean_cumulative_mass"]) for row in q_rows], dtype=float)
        label = f"q={q:.4f}"
        axes[0].plot(ranks, mass, color=color, lw=1.8, label=label)
        axes[1].plot(ranks, cumulative, color=color, lw=1.8, label=label)

    axes[0].axhline(1.0 / max(float(N), 1.0), color="0.55", lw=1.0, ls="--", label="1 / n")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("coordinate rank after sorting |v_i|^2")
    axes[0].set_ylabel("mean sorted coordinate mass")
    axes[0].set_title("Sorted Coordinate Decay")

    axes[1].set_xlabel("top-r coordinate count")
    axes[1].set_ylabel("mean cumulative mass")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_title("Cumulative Top-r Mass")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle(
        f"Chung-Lu Eigenvector Coordinate Profiles, n={N}, tau={tau:.2f}, mean degree={target_mean_degree:.1f}",
        y=1.02,
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.86, 1.0))
    return fig


def run_parameter_pair(tau: float, target_mean_degree: float) -> tuple[list[dict[str, object]], list[dict[str, object]], Path]:
    output_dir = parameter_output_dir(tau=tau, target_mean_degree=target_mean_degree)
    output_dir.mkdir(parents=True, exist_ok=True)
    offset = parameter_seed_offset(tau=tau, target_mean_degree=target_mean_degree)
    graph_seeds = make_seeds(NUM_GRAPHS, GRAPH_SEED + offset)
    sequence_seeds = make_seeds(NUM_GRAPHS, SEQUENCE_SEED + offset)

    rows: list[dict[str, object]] = []
    decay_rows: list[dict[str, object]] = []
    diagnostics: list[WeightSequenceDiagnostics] = []
    for index, (graph_seed, sequence_seed) in enumerate(zip(graph_seeds, sequence_seeds)):
        graph_rows, graph_decay_rows, diagnostic = graph_localization_and_decay_rows(
            graph_index=index,
            graph_seed=graph_seed,
            sequence_seed=sequence_seed,
            tau=tau,
            target_mean_degree=target_mean_degree,
        )
        rows.extend(graph_rows)
        decay_rows.extend(graph_decay_rows)
        diagnostics.append(diagnostic)

    summary_rows = aggregate_quantile_rows(rows)
    decay_summary_rows = aggregate_coordinate_decay_rows(decay_rows)
    write_csv(rows, output_dir / "eigenvector_localization_by_graph.csv", graph_row_fieldnames())
    write_csv(summary_rows, output_dir / "eigenvector_localization_summary.csv", summary_row_fieldnames())
    write_csv(decay_rows, output_dir / "eigenvector_coordinate_decay_by_graph.csv", coordinate_decay_row_fieldnames())
    write_csv(
        decay_summary_rows,
        output_dir / "eigenvector_coordinate_decay_summary.csv",
        coordinate_decay_summary_fieldnames(),
    )

    fig = plot_localization_summary(summary_rows, tau=tau, target_mean_degree=target_mean_degree)
    fig.savefig(output_dir / "eigenvector_localization_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    decay_fig = plot_coordinate_decay_summary(decay_summary_rows, tau=tau, target_mean_degree=target_mean_degree)
    decay_fig.savefig(output_dir / "eigenvector_coordinate_decay_summary.png", dpi=200, bbox_inches="tight")
    plt.close(decay_fig)

    manifest = {
        "model": "power_law_chung_lu",
        "diagnostic": "eigenvector_localization",
        "n": N,
        "tau": tau,
        "target_mean_degree": target_mean_degree,
        "tau_values": TAU_VALUES,
        "target_mean_degrees": TARGET_MEAN_DEGREES,
        "active_parameter_pairs": ACTIVE_PARAMETER_PAIRS,
        "run_full_grid": RUN_FULL_GRID,
        "num_graphs": NUM_GRAPHS,
        "window_mode": WINDOW_MODE,
        "local_quantiles": LOCAL_QUANTILES,
        "local_quantile_bins": LOCAL_QUANTILE_BINS,
        "target_eigenvectors_per_graph": TARGET_EIGENVECTORS_PER_GRAPH,
        "coordinate_decay_max_rank": COORDINATE_DECAY_MAX_RANK,
        "sequence_mode": SEQUENCE_MODE,
        "cutoff_mode": CUTOFF_MODE,
        "cutoff_coefficient": CUTOFF_COEFFICIENT,
        "w_min": W_MIN,
        "w_max": W_MAX,
        "component_mode": COMPONENT_MODE,
        "graph_seed": GRAPH_SEED + offset,
        "sequence_seed": SEQUENCE_SEED + offset,
        "mean_final_weight_mean": float(np.mean([diagnostic.final_mean for diagnostic in diagnostics])),
        "mean_final_max_weight": float(np.mean([diagnostic.final_max for diagnostic in diagnostics])),
        "mean_fraction_at_upper_cap": float(np.mean([diagnostic.fraction_at_upper_cap for diagnostic in diagnostics])),
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "code_version": code_version(),
    }
    with (output_dir / "run_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    print(f"tau={tau:.2f}, mean degree={target_mean_degree:.1f}: saved eigenvector diagnostics to {output_dir}")
    return rows, summary_rows, output_dir


def active_parameter_pairs() -> list[tuple[float, float]]:
    if RUN_FULL_GRID:
        return [(tau, degree) for tau in TAU_VALUES for degree in TARGET_MEAN_DEGREES]
    return list(ACTIVE_PARAMETER_PAIRS)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for tau, target_mean_degree in active_parameter_pairs():
        run_parameter_pair(tau=tau, target_mean_degree=target_mean_degree)


if __name__ == "__main__":
    main()
