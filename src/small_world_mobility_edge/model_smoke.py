"""Small, seeded end-to-end diagnostics for the non-PA graph families.

This module is a discoverability wrapper around the project generators and
shared spectral helpers.  It deliberately uses modest dense spectra; the
specialized modules remain the production paths for sparse-edge, tail, and
projector experiments.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import integrate

from .chung_lu_baseline import (
    generate_chung_lu_graph_from_weights,
    generate_power_law_expected_weights,
)
from .empirical_spectral_density import estimate_empirical_spectral_density
from .erdos_renyi_baseline import generate_erdos_renyi_graph
from .fixed_degree_configuration_baseline import rewire_graph_preserving_degrees
from .regular_baseline import (
    adjacency_eigendecomposition,
    adjacent_gap_ratios_from_spacings,
    generate_random_regular_graph,
    inverse_participation_ratio,
    kesten_mckay_density,
    unfold_with_kesten_mckay,
    wigner_surmise_goe,
)


MODEL_REGULAR = "regular"
MODEL_ER = "erdos-renyi"
MODEL_CHUNG_LU = "chung-lu"
MODEL_FIXED_DEGREE = "fixed-degree-rewire"
MODEL_CHOICES = (MODEL_REGULAR, MODEL_ER, MODEL_CHUNG_LU, MODEL_FIXED_DEGREE)


@dataclass(frozen=True)
class PreparedGraph:
    graph: nx.Graph
    metadata: dict[str, object]


@dataclass(frozen=True)
class SpectrumSample:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    raw_spacings: np.ndarray
    normalized_spacings: np.ndarray
    unfolded_spacings: np.ndarray
    gap_ratios: np.ndarray
    ipr: np.ndarray


def _select_component(graph: nx.Graph, component_mode: str) -> nx.Graph:
    if component_mode == "full":
        return graph
    if component_mode == "largest_connected_component":
        nodes = max(nx.connected_components(graph), key=len)
        return graph.subgraph(nodes).copy()
    raise ValueError(f"unsupported component_mode={component_mode!r}")


def _validate_simple_symmetric_graph(graph: nx.Graph) -> None:
    if type(graph) is not nx.Graph or graph.is_directed() or graph.is_multigraph():
        raise TypeError("diagnostic inputs must be simple undirected networkx.Graph objects")
    if nx.number_of_selfloops(graph):
        raise ValueError("diagnostic inputs must not contain self-loops")
    adjacency = nx.to_numpy_array(graph, dtype=float)
    if not np.array_equal(adjacency, adjacency.T):
        raise AssertionError("adjacency matrix is not symmetric")


def _seed_list(seed: int, count: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(value) for value in rng.integers(0, 2**32 - 1, size=count, dtype=np.uint32)]


def _regular_factory(n: int, degree: int, seed: int) -> PreparedGraph:
    graph = generate_random_regular_graph(n=n, degree=degree, seed=seed)
    exact = all(value == degree for _, value in graph.degree())
    if not exact:
        raise AssertionError("random-regular generator did not preserve the requested degree")
    return PreparedGraph(graph, {"requested_degree": degree, "all_degrees_exact": exact})


def _er_factory(n: int, degree: int, seed: int) -> PreparedGraph:
    graph = generate_erdos_renyi_graph(n=n, average_degree=float(degree), seed=seed)
    return PreparedGraph(
        graph,
        {
            "requested_average_degree": float(degree),
            "edge_probability": float(degree / (n - 1)),
        },
    )


def _chung_lu_factory(n: int, degree: int, tau: float, seed: int, sequence_seed: int) -> PreparedGraph:
    weights, diagnostics = generate_power_law_expected_weights(
        n=n,
        tau=tau,
        target_mean_degree=float(degree),
        w_min=1.0,
        w_max=None,
        cutoff_mode="structural",
        cutoff_coefficient=1.0,
        sequence_mode="quantile",
        sequence_seed=sequence_seed,
    )
    graph = generate_chung_lu_graph_from_weights(weights, seed=seed)
    return PreparedGraph(
        graph,
        {
            "density_exponent_tau": tau,
            "weight_ccdf_exponent": tau - 1.0,
            "requested_average_degree": float(degree),
            "weight_sequence_mode": "quantile",
            "cutoff_mode": "structural",
            "final_mean_weight": diagnostics.final_mean,
            "final_max_weight": diagnostics.final_max,
            "admissibility_holds": diagnostics.admissibility_holds,
        },
    )


def _fixed_degree_factory(
    n: int,
    degree: int,
    tau: float,
    seed: int,
    sequence_seed: int,
    rewire_seed: int,
    swaps_per_edge: int,
) -> PreparedGraph:
    original = _chung_lu_factory(n, degree, tau, seed, sequence_seed)
    original_degrees = dict(original.graph.degree())
    rewired, requested_swaps, max_tries, overlap = rewire_graph_preserving_degrees(
        original.graph,
        rewire_seed=rewire_seed,
        swaps_per_edge=swaps_per_edge,
        max_tries_factor=20,
        overlap_warning_threshold=None,
    )
    exact = dict(rewired.degree()) == original_degrees
    if not exact:
        raise AssertionError("double-edge-swap control changed the labeled degree sequence")
    return PreparedGraph(
        rewired,
        {
            **original.metadata,
            "implementation": "networkx.double_edge_swap",
            "sampling_claim": "finite-chain degree-preserving simple-graph control; uniformity not claimed",
            "all_labeled_degrees_preserved": exact,
            "requested_swaps": requested_swaps,
            "max_tries": max_tries,
            "edge_overlap_fraction": overlap,
        },
    )


def _spectral_sample(
    graph: nx.Graph,
    *,
    model: str,
    regular_degree: int,
    empirical_density: object | None = None,
) -> SpectrumSample:
    eigenvalues, eigenvectors = adjacency_eigendecomposition(graph)
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    eigenvectors = np.asarray(eigenvectors, dtype=float)

    keep = np.ones(len(eigenvalues), dtype=bool)
    if model == MODEL_REGULAR:
        trivial_index = int(np.argmin(np.abs(eigenvalues - regular_degree)))
        if np.isclose(eigenvalues[trivial_index], regular_degree, atol=1e-10, rtol=0.0):
            keep[trivial_index] = False
    else:
        keep[-1] = False  # one Perron eigenpair
        keep &= ~np.isclose(eigenvalues, 0.0, atol=1e-8, rtol=0.0)

    values = eigenvalues[keep]
    vectors = eigenvectors[:, keep]
    raw = np.diff(values)
    raw = raw[np.isfinite(raw) & (raw > 0.0)]
    normalized = raw / np.mean(raw) if len(raw) else np.array([], dtype=float)
    if model == MODEL_REGULAR:
        unfolded_values = unfold_with_kesten_mckay(values, degree=regular_degree).unfolded
    elif empirical_density is not None:
        unfolded_values = len(values) * np.interp(
            values,
            empirical_density.grid,
            empirical_density.cdf,
        )
    else:
        unfolded_values = np.array([], dtype=float)
    unfolded = np.diff(unfolded_values)
    unfolded = unfolded[np.isfinite(unfolded) & (unfolded > 0.0)]
    return SpectrumSample(
        eigenvalues=values,
        eigenvectors=vectors,
        raw_spacings=raw,
        normalized_spacings=normalized,
        unfolded_spacings=unfolded,
        gap_ratios=adjacent_gap_ratios_from_spacings(raw),
        ipr=inverse_participation_ratio(vectors),
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_density(
    model: str,
    samples: list[SpectrumSample],
    histogram_density: object,
    local_density: object,
    output_dir: Path,
    regular_degree: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    pooled = np.concatenate([sample.eigenvalues for sample in samples])
    axes[0].hist(pooled, bins=40, density=True, alpha=0.35, label="ensemble histogram")
    axes[0].plot(histogram_density.grid, histogram_density.density, lw=1.5, label="interpolated histogram")
    axes[0].plot(local_density.grid, local_density.density, lw=1.8, label="local-count density")
    if model == MODEL_REGULAR:
        axes[0].plot(
            local_density.grid,
            kesten_mckay_density(local_density.grid, regular_degree),
            color="black",
            ls="--",
            lw=1.3,
            label="Kesten--McKay",
        )
    axes[0].set_title("Eigenvalue density estimators")
    axes[0].set_xlabel("eigenvalue")
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=8)

    low, high = local_density.quantiles_to_energies([0.40, 0.60])
    windowed = [sample.eigenvalues[(sample.eigenvalues >= low) & (sample.eigenvalues <= high)] for sample in samples]
    pooled_window = np.concatenate([values for values in windowed if len(values)])
    axes[1].hist(pooled_window, bins=24, density=True, alpha=0.7)
    axes[1].axvline(low, color="black", ls="--", lw=1)
    axes[1].axvline(high, color="black", ls="--", lw=1)
    axes[1].set_title("Eigenvalue histogram in q=[0.40, 0.60]")
    axes[1].set_xlabel("eigenvalue")
    axes[1].set_ylabel("conditional histogram density")
    fig.suptitle(model)
    fig.tight_layout()
    fig.savefig(output_dir / "density_diagnostics.png", dpi=160)
    plt.close(fig)


def _plot_spacings(model: str, samples: list[SpectrumSample], output_dir: Path) -> None:
    groups = [
        ("raw within-graph gaps", "raw_spacings"),
        ("per-graph mean-normalized gaps", "normalized_spacings"),
        ("unfolded gaps", "unfolded_spacings"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.1))
    for ax, (title, attribute) in zip(axes, groups):
        values = np.concatenate([getattr(sample, attribute) for sample in samples])
        ax.hist(values, bins=35, density=True, alpha=0.65, label="empirical")
        if attribute != "raw_spacings":
            grid = np.linspace(0.0, min(4.0, max(2.0, float(np.max(values)))), 300)
            ax.plot(grid, wigner_surmise_goe(grid), color="black", lw=1.3, label="GOE")
            ax.plot(grid, np.exp(-grid), color="tab:red", ls="--", lw=1.3, label="Poisson")
            ax.set_xlim(0.0, grid[-1])
        ax.set_title(title)
        ax.set_xlabel("spacing")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
    fig.suptitle(f"{model}: graph identity retained before pooling gaps")
    fig.tight_layout()
    fig.savefig(output_dir / "spacing_diagnostics.png", dpi=160)
    plt.close(fig)


def _plot_localization(model: str, samples: list[SpectrumSample], output_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for graph_index, sample in enumerate(samples):
        n = sample.eigenvectors.shape[0]
        n_ipr = n * sample.ipr
        ax.scatter(sample.eigenvalues, n_ipr, s=7, alpha=0.35)
        for eigenvalue, ipr, scaled in zip(sample.eigenvalues, sample.ipr, n_ipr):
            rows.append(
                {
                    "graph_index": graph_index,
                    "eigenvalue": float(eigenvalue),
                    "ipr": float(ipr),
                    "n_times_ipr": float(scaled),
                    "participation_ratio": float(1.0 / max(ipr, 1e-15)),
                    "participation_fraction": float(1.0 / max(n * ipr, 1e-15)),
                }
            )
    ax.set_xlabel("eigenvalue")
    ax.set_ylabel("N x IPR")
    ax.set_title(f"{model}: eigenvalue/localization alignment")
    fig.tight_layout()
    fig.savefig(output_dir / "localization_diagnostics.png", dpi=160)
    plt.close(fig)
    return rows


def run_model(
    *,
    model: str,
    n: int,
    degree: int,
    tau: float,
    num_graphs: int,
    seed: int,
    component_mode: str,
    swaps_per_edge: int,
    output_root: Path,
) -> dict[str, object]:
    if model not in MODEL_CHOICES:
        raise ValueError(f"unsupported model={model!r}")
    if model == MODEL_REGULAR and (n * degree) % 2:
        raise ValueError("n * degree must be even for a regular graph")
    graph_seeds = _seed_list(seed, num_graphs)
    sequence_seeds = _seed_list(seed + 1, num_graphs)
    rewire_seeds = _seed_list(seed + 2, num_graphs)
    prepared: list[PreparedGraph] = []
    for index, graph_seed in enumerate(graph_seeds):
        if model == MODEL_REGULAR:
            item = _regular_factory(n, degree, graph_seed)
        elif model == MODEL_ER:
            item = _er_factory(n, degree, graph_seed)
        elif model == MODEL_CHUNG_LU:
            item = _chung_lu_factory(n, degree, tau, graph_seed, sequence_seeds[index])
        else:
            item = _fixed_degree_factory(
                n,
                degree,
                tau,
                graph_seed,
                sequence_seeds[index],
                rewire_seeds[index],
                swaps_per_edge,
            )
        _validate_simple_symmetric_graph(item.graph)
        analyzed = _select_component(item.graph, component_mode)
        prepared.append(PreparedGraph(analyzed, item.metadata))

    preliminary = [
        _spectral_sample(item.graph, model=model, regular_degree=degree)
        for item in prepared
    ]
    eigenvalue_samples = [sample.eigenvalues for sample in preliminary]
    mean_dimension = float(np.mean([len(values) for values in eigenvalue_samples]))
    histogram_density = estimate_empirical_spectral_density(
        eigenvalue_samples,
        num_graphs=num_graphs,
        per_graph_dimension=mean_dimension,
        method="histogram",
        num_points=801,
        num_bins=40,
    )
    local_density = estimate_empirical_spectral_density(
        eigenvalue_samples,
        num_graphs=num_graphs,
        per_graph_dimension=mean_dimension,
        method="local_count",
        num_points=801,
        local_count_half_width=0.20,
    )
    samples = [
        _spectral_sample(
            item.graph,
            model=model,
            regular_degree=degree,
            empirical_density=local_density,
        )
        for item in prepared
    ]
    output_dir = Path(output_root) / model
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_density(model, samples, histogram_density, local_density, output_dir, degree)
    _plot_spacings(model, samples, output_dir)
    localization_rows = _plot_localization(model, samples, output_dir)
    _write_csv(output_dir / "localization.csv", localization_rows)

    raw = np.concatenate([sample.raw_spacings for sample in samples])
    normalized = np.concatenate([sample.normalized_spacings for sample in samples])
    unfolded = np.concatenate([sample.unfolded_spacings for sample in samples])
    ratios = np.concatenate([sample.gap_ratios for sample in samples])
    graph_rows = []
    for index, (item, sample) in enumerate(zip(prepared, samples)):
        graph_rows.append(
            {
                "graph_index": index,
                "nodes_analyzed": item.graph.number_of_nodes(),
                "edges_analyzed": item.graph.number_of_edges(),
                "mean_degree_analyzed": float(np.mean([value for _, value in item.graph.degree()])),
                "num_eigenvalues_retained": len(sample.eigenvalues),
                "num_raw_gaps": len(sample.raw_spacings),
                "num_gap_ratios": len(sample.gap_ratios),
                **item.metadata,
            }
        )
    _write_csv(output_dir / "graph_summary.csv", graph_rows)
    summary = {
        "model": model,
        "n_requested": n,
        "degree_parameter": degree,
        "density_exponent_tau": tau if model in {MODEL_CHUNG_LU, MODEL_FIXED_DEGREE} else None,
        "num_graphs": num_graphs,
        "seed": seed,
        "component_mode": component_mode,
        "degree_check": (
            all(bool(row.get("all_degrees_exact")) for row in graph_rows)
            if model == MODEL_REGULAR
            else all(bool(row.get("all_labeled_degrees_preserved")) for row in graph_rows)
            if model == MODEL_FIXED_DEGREE
            else None
        ),
        "spectral_filter": (
            "remove one trivial eigenvalue d"
            if model == MODEL_REGULAR
            else "remove one Perron eigenpair and the zero atom at atol=1e-8"
        ),
        "histogram_density_integral": float(integrate.trapezoid(histogram_density.density, histogram_density.grid)),
        "local_count_density_integral": float(integrate.trapezoid(local_density.density, local_density.grid)),
        "raw_gap_count": int(len(raw)),
        "mean_normalized_gap_count": int(len(normalized)),
        "unfolded_gap_count": int(len(unfolded)),
        "gap_ratio_count": int(len(ratios)),
        "mean_gap_ratio": float(np.mean(ratios)) if len(ratios) else None,
        "mean_n_times_ipr": float(np.mean([row["n_times_ipr"] for row in localization_rows])),
        "unfolding": (
            "Kesten-McKay CDF"
            if model == MODEL_REGULAR
            else "ensemble local-count empirical CDF (smoke diagnostic, not cross-fitted production inference)"
        ),
        "within_graph_spacing_construction": True,
        "files": [
            "graph_summary.csv",
            "localization.csv",
            "density_diagnostics.png",
            "spacing_diagnostics.png",
            "localization_diagnostics.png",
        ],
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("all", *MODEL_CHOICES), default="all")
    parser.add_argument("--n", type=int, default=160)
    parser.add_argument("--degree", type=int, default=6)
    parser.add_argument("--tau", type=float, default=2.5, help="Chung-Lu density exponent")
    parser.add_argument("--num-graphs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument(
        "--component-mode",
        choices=("full", "largest_connected_component"),
        default="full",
    )
    parser.add_argument("--swaps-per-edge", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/smoke"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = MODEL_CHOICES if args.model == "all" else (args.model,)
    summaries = [
        run_model(
            model=model,
            n=args.n,
            degree=args.degree,
            tau=args.tau,
            num_graphs=args.num_graphs,
            seed=args.seed,
            component_mode=args.component_mode,
            swaps_per_edge=args.swaps_per_edge,
            output_root=args.output_dir,
        )
        for model in models
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary_all.json").open("w") as handle:
        json.dump(summaries, handle, indent=2, sort_keys=True)
    for summary in summaries:
        print(
            f"{summary['model']}: gaps={summary['raw_gap_count']}, "
            f"ratios={summary['gap_ratio_count']}, "
            f"density_mass={summary['local_count_density_integral']:.6f}"
        )


if __name__ == "__main__":
    main()
