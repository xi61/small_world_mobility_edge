from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

try:
    from .regular_baseline import adjacency_eigendecomposition
except ImportError:  # pragma: no cover - script-style fallback
    from regular_baseline import adjacency_eigendecomposition


@dataclass(frozen=True)
class WeightSequenceDiagnostics:
    sequence_mode: str
    cutoff_mode: str
    tau: float
    target_mean_degree: float
    requested_w_min: float
    requested_w_max: float | None
    cutoff_coefficient: float
    raw_min: float
    raw_max: float
    scaling_constant: float
    final_lower_bound: float
    final_upper_cap: float
    structural_cap: float | None
    power_of_n_cap: float | None
    admissibility_holds: bool
    target_mean_weight: float
    final_mean: float
    final_max: float
    total_weight: float
    max_weight_squared: float
    admissibility_margin: float
    fraction_at_lower_bound: float
    fraction_at_upper_cap: float
    number_clipped_low: int
    number_clipped_high: int
    estimated_tau_before_cutoff: float | None
    estimated_tau_after_cutoff: float | None
    estimated_tau_tail: float | None


@dataclass(frozen=True)
class ChungLuSpectrum:
    graph_index: int
    graph_seed: int
    sequence_seed: int
    n_requested: int
    tau: float
    target_mean_degree_requested: float
    sequence_mode: str
    cutoff_mode: str
    requested_w_min: float
    requested_w_max: float | None
    cutoff_coefficient: float
    component_mode: str
    analyzed_dimension: int
    realized_mean_degree: float
    num_components: int
    largest_component_size: int
    num_isolated_vertices: int
    max_realized_degree: int
    lambda_max: float
    lambda_2: float
    lambda_min: float
    degrees: np.ndarray
    expected_weights: np.ndarray
    eigenvalues: np.ndarray
    bulk_eigenvalues: np.ndarray
    perron_eigenvalue: float
    weight_diagnostics: WeightSequenceDiagnostics


def constant_weight_value(n: int, average_degree: float) -> float:
    if n <= 1:
        raise ValueError("n must be at least 2")
    if average_degree <= 0:
        raise ValueError("average_degree must be positive")
    return average_degree * n / (n - 1)


def generate_constant_weight_chung_lu_graph(
    n: int,
    average_degree: float,
    seed: int | np.random.Generator | None = None,
) -> nx.Graph:
    weight = constant_weight_value(n=n, average_degree=average_degree)
    weights = np.full(n, weight, dtype=float)
    return nx.expected_degree_graph(weights, seed=seed, selfloops=False)


def adjacency_eigenvalues_only(graph: nx.Graph) -> np.ndarray:
    eigenvalues, _ = adjacency_eigendecomposition(graph)
    return np.sort(np.asarray(eigenvalues, dtype=float))


def bulk_eigenvalues_excluding_perron(eigenvalues: np.ndarray) -> tuple[np.ndarray, float]:
    eigenvalues = np.sort(np.asarray(eigenvalues, dtype=float))
    if len(eigenvalues) < 2:
        raise ValueError("need at least two eigenvalues to exclude the top Perron outlier")
    return eigenvalues[:-1], float(eigenvalues[-1])


def power_law_quantile_sequence(
    n: int,
    tau: float,
) -> np.ndarray:
    if tau <= 2.0:
        raise ValueError("tau must exceed 2 so the target mean degree is well-defined")
    quantiles = (np.arange(n, dtype=float) + 0.5) / n
    raw = quantiles ** (-1.0 / (tau - 1.0))
    return np.asarray(raw, dtype=float)


def sampled_pareto_sequence(
    n: int,
    tau: float,
    seed: int | np.random.Generator | None,
) -> np.ndarray:
    if tau <= 2.0:
        raise ValueError("tau must exceed 2 so the target mean degree is well-defined")
    rng = np.random.default_rng(seed)
    shape = tau - 1.0
    # NumPy Pareto with shape a has CCDF exponent a, so this matches
    # P(D >= x) proportional to x^{-(tau - 1)}.
    raw = rng.pareto(shape, size=n) + 1.0
    return np.asarray(raw, dtype=float)


def estimate_power_law_tail_exponent(
    values: np.ndarray,
    lower_cutoff: float,
) -> float | None:
    values = np.asarray(values, dtype=float)
    tail = values[values >= lower_cutoff]
    if len(tail) < 10:
        return None
    logs = np.log(np.maximum(tail / lower_cutoff, 1.0 + 1e-12))
    mean_log = float(np.mean(logs))
    if mean_log <= 0:
        return None
    alpha_hat = 1.0 / mean_log
    return float(alpha_hat + 1.0)


def _power_of_n_cutoff(
    n: int,
    tau: float,
    cutoff_coefficient: float,
) -> float:
    return float(cutoff_coefficient * n ** (1.0 / (tau - 1.0)))


def _candidate_upper_cutoff(
    n: int,
    tau: float,
    target_mean_degree: float,
    requested_w_max: float | None,
    cutoff_mode: str,
    cutoff_coefficient: float,
) -> tuple[float, float | None, float | None]:
    requested_cap = np.inf if requested_w_max is None else float(requested_w_max)
    structural_cap = None
    power_of_n_cap = None

    if cutoff_mode == "fixed":
        return requested_cap, structural_cap, power_of_n_cap
    if cutoff_mode == "power_of_n":
        power_of_n_cap = _power_of_n_cutoff(n=n, tau=tau, cutoff_coefficient=cutoff_coefficient)
        return float(min(requested_cap, power_of_n_cap)), structural_cap, power_of_n_cap
    if cutoff_mode == "structural":
        structural_cap = float(cutoff_coefficient * np.sqrt(n * target_mean_degree))
        return float(min(requested_cap, structural_cap)), structural_cap, power_of_n_cap
    raise ValueError(f"unsupported cutoff_mode={cutoff_mode!r}")


def clipped_mean_from_scale(
    raw_weights: np.ndarray,
    scaling_constant: float,
    lower_bound: float,
    upper_cap: float,
) -> float:
    clipped = np.clip(scaling_constant * raw_weights, lower_bound, upper_cap)
    return float(np.mean(clipped))


def solve_clipped_mean_scaling_constant(
    raw_weights: np.ndarray,
    target_mean_degree: float,
    lower_bound: float,
    upper_cap: float,
    tolerance: float = 1e-12,
    max_iterations: int = 200,
) -> float:
    if not np.isfinite(upper_cap):
        return float(target_mean_degree / np.mean(raw_weights))

    if not (lower_bound <= target_mean_degree <= upper_cap):
        raise ValueError(
            f"requested target mean {target_mean_degree:.6f} is infeasible for bounds "
            f"[{lower_bound:.6f}, {upper_cap:.6f}]"
        )

    low = 0.0
    high = 1.0
    while clipped_mean_from_scale(raw_weights, high, lower_bound, upper_cap) < target_mean_degree:
        high *= 2.0
        if high > 1e18:
            raise RuntimeError("failed to bracket the scaling constant for the clipped-mean equation")

    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        mid_mean = clipped_mean_from_scale(raw_weights, mid, lower_bound, upper_cap)
        if abs(mid_mean - target_mean_degree) <= tolerance:
            return float(mid)
        if mid_mean < target_mean_degree:
            low = mid
        else:
            high = mid
    return float(0.5 * (low + high))


def generate_power_law_expected_weights(
    n: int,
    tau: float,
    target_mean_degree: float,
    w_min: float,
    w_max: float | None,
    cutoff_mode: str,
    cutoff_coefficient: float,
    sequence_mode: str,
    sequence_seed: int | np.random.Generator | None = None,
    max_iterations: int = 200,
    tolerance: float = 1e-10,
) -> tuple[np.ndarray, WeightSequenceDiagnostics]:
    if n <= 1:
        raise ValueError("n must be at least 2")
    if tau <= 2.0:
        raise ValueError("tau must exceed 2")
    if target_mean_degree <= 0:
        raise ValueError("target_mean_degree must be positive")
    if w_min <= 0:
        raise ValueError("w_min must be positive")
    if cutoff_coefficient <= 0:
        raise ValueError("cutoff_coefficient must be positive")
    if w_max is not None and w_max < w_min:
        raise ValueError("w_max must be at least w_min")

    if sequence_mode == "quantile":
        raw_weights = power_law_quantile_sequence(n=n, tau=tau)
    elif sequence_mode == "sampled":
        raw_weights = sampled_pareto_sequence(n=n, tau=tau, seed=sequence_seed)
    else:
        raise ValueError(f"unsupported sequence_mode={sequence_mode!r}")

    raw_weights = np.asarray(raw_weights, dtype=float)
    raw_min = float(np.min(raw_weights))
    raw_max = float(np.max(raw_weights))
    estimated_tau_before_cutoff = estimate_power_law_tail_exponent(raw_weights, lower_cutoff=max(w_min, raw_min))

    final_upper_cap, structural_cap, power_of_n_cap = _candidate_upper_cutoff(
        n=n,
        tau=tau,
        target_mean_degree=target_mean_degree,
        requested_w_max=w_max,
        cutoff_mode=cutoff_mode,
        cutoff_coefficient=cutoff_coefficient,
    )
    if final_upper_cap < w_min:
        raise ValueError(
            f"requested target mean {target_mean_degree:.6f} is infeasible because final upper cap "
            f"{final_upper_cap:.6f} is below w_min={w_min:.6f}"
        )
    if np.isfinite(final_upper_cap) and not (w_min <= target_mean_degree <= final_upper_cap):
        raise ValueError(
            f"requested target mean {target_mean_degree:.6f} is infeasible for final bounds "
            f"[{w_min:.6f}, {final_upper_cap:.6f}]"
        )

    scaling_constant = solve_clipped_mean_scaling_constant(
        raw_weights=raw_weights,
        target_mean_degree=target_mean_degree,
        lower_bound=w_min,
        upper_cap=final_upper_cap,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    weights = np.clip(scaling_constant * raw_weights, w_min, final_upper_cap)
    final_mean = float(np.mean(weights))
    if abs(final_mean - target_mean_degree) > max(tolerance, 1e-10 * target_mean_degree):
        raise RuntimeError(
            f"clipped mean solve failed: final mean {final_mean:.12f} differs from target {target_mean_degree:.12f}"
        )

    total_weight = float(np.sum(weights))
    adjusted_max_weight = float(np.max(weights))
    max_weight_squared = adjusted_max_weight ** 2
    admissibility_margin = total_weight - max_weight_squared
    admissibility_tolerance = max(1e-8, 10.0 * tolerance * max(total_weight, 1.0), np.finfo(float).eps * max(total_weight, 1.0))
    admissibility_holds = bool(max_weight_squared <= total_weight + admissibility_tolerance)
    estimated_tau_after_cutoff = estimate_power_law_tail_exponent(weights, lower_cutoff=max(w_min, float(np.min(weights))))
    number_clipped_low = int(np.sum(np.isclose(weights, w_min, atol=1e-12, rtol=0.0)))
    number_clipped_high = (
        int(np.sum(np.isclose(weights, final_upper_cap, atol=1e-12, rtol=0.0)))
        if np.isfinite(final_upper_cap)
        else 0
    )

    diagnostics = WeightSequenceDiagnostics(
        sequence_mode=sequence_mode,
        cutoff_mode=cutoff_mode,
        tau=float(tau),
        target_mean_degree=float(target_mean_degree),
        requested_w_min=float(w_min),
        requested_w_max=None if w_max is None else float(w_max),
        cutoff_coefficient=float(cutoff_coefficient),
        raw_min=raw_min,
        raw_max=raw_max,
        scaling_constant=scaling_constant,
        final_lower_bound=float(w_min),
        final_upper_cap=float(final_upper_cap),
        structural_cap=structural_cap,
        power_of_n_cap=power_of_n_cap,
        target_mean_weight=float(target_mean_degree),
        admissibility_holds=admissibility_holds,
        final_mean=final_mean,
        final_max=adjusted_max_weight,
        total_weight=total_weight,
        max_weight_squared=max_weight_squared,
        admissibility_margin=admissibility_margin,
        fraction_at_lower_bound=float(number_clipped_low / n),
        fraction_at_upper_cap=float(number_clipped_high / n),
        number_clipped_low=number_clipped_low,
        number_clipped_high=number_clipped_high,
        estimated_tau_before_cutoff=estimated_tau_before_cutoff,
        estimated_tau_after_cutoff=estimated_tau_after_cutoff,
        estimated_tau_tail=estimated_tau_after_cutoff,
    )
    return weights, diagnostics


def generate_chung_lu_graph_from_weights(
    weights: np.ndarray,
    seed: int | np.random.Generator | None = None,
) -> nx.Graph:
    return nx.expected_degree_graph(weights, seed=seed, selfloops=False)


def analyze_power_law_chung_lu_graph(
    n: int,
    tau: float,
    target_mean_degree: float,
    w_min: float,
    w_max: float | None,
    cutoff_mode: str,
    cutoff_coefficient: float,
    sequence_mode: str,
    graph_seed: int,
    sequence_seed: int,
    component_mode: str = "full",
    graph_index: int = 0,
) -> ChungLuSpectrum:
    expected_weights, diagnostics = generate_power_law_expected_weights(
        n=n,
        tau=tau,
        target_mean_degree=target_mean_degree,
        w_min=w_min,
        w_max=w_max,
        cutoff_mode=cutoff_mode,
        cutoff_coefficient=cutoff_coefficient,
        sequence_mode=sequence_mode,
        sequence_seed=sequence_seed,
    )
    if not diagnostics.admissibility_holds:
        raise ValueError(
            "final expected-weight sequence violates max(w)^2 <= sum(w); "
            f"max(w)^2={diagnostics.max_weight_squared:.6f}, sum(w)={diagnostics.total_weight:.6f}"
        )

    graph = generate_chung_lu_graph_from_weights(expected_weights, seed=graph_seed)
    num_components = nx.number_connected_components(graph)
    component_sizes = sorted((len(component) for component in nx.connected_components(graph)), reverse=True)
    largest_component_size = component_sizes[0]
    degrees = np.array([degree for _, degree in graph.degree()], dtype=float)
    num_isolated_vertices = int(np.sum(degrees == 0))
    realized_mean_degree = float(np.mean(degrees))
    max_realized_degree = int(np.max(degrees))

    if component_mode == "full":
        analyzed_graph = graph
    elif component_mode == "largest_connected_component":
        largest_nodes = max(nx.connected_components(graph), key=len)
        analyzed_graph = graph.subgraph(largest_nodes).copy()
    else:
        raise ValueError(f"unsupported component_mode={component_mode!r}")

    eigenvalues = adjacency_eigenvalues_only(analyzed_graph)
    bulk_eigenvalues, perron_eigenvalue = bulk_eigenvalues_excluding_perron(eigenvalues)
    lambda_min = float(eigenvalues[0])
    lambda_max = float(eigenvalues[-1])
    lambda_2 = float(eigenvalues[-2])
    return ChungLuSpectrum(
        graph_index=graph_index,
        graph_seed=graph_seed,
        sequence_seed=sequence_seed,
        n_requested=n,
        tau=float(tau),
        target_mean_degree_requested=float(target_mean_degree),
        sequence_mode=sequence_mode,
        cutoff_mode=cutoff_mode,
        requested_w_min=float(w_min),
        requested_w_max=None if w_max is None else float(w_max),
        cutoff_coefficient=float(cutoff_coefficient),
        component_mode=component_mode,
        analyzed_dimension=analyzed_graph.number_of_nodes(),
        realized_mean_degree=realized_mean_degree,
        num_components=num_components,
        largest_component_size=largest_component_size,
        num_isolated_vertices=num_isolated_vertices,
        max_realized_degree=max_realized_degree,
        lambda_max=lambda_max,
        lambda_2=lambda_2,
        lambda_min=lambda_min,
        degrees=degrees,
        expected_weights=expected_weights,
        eigenvalues=eigenvalues,
        bulk_eigenvalues=bulk_eigenvalues,
        perron_eigenvalue=perron_eigenvalue,
        weight_diagnostics=diagnostics,
    )


def generate_power_law_chung_lu_ensemble(
    n: int,
    tau: float,
    target_mean_degree: float,
    w_min: float,
    w_max: float | None,
    cutoff_mode: str,
    cutoff_coefficient: float,
    sequence_mode: str,
    num_graphs: int,
    graph_seed: int,
    sequence_seed: int,
    component_mode: str = "full",
) -> list[ChungLuSpectrum]:
    graph_rng = np.random.default_rng(graph_seed)
    sequence_rng = np.random.default_rng(sequence_seed)
    graph_seeds = graph_rng.integers(0, 2**32 - 1, size=num_graphs, dtype=np.uint32)
    sequence_seeds = sequence_rng.integers(0, 2**32 - 1, size=num_graphs, dtype=np.uint32)
    return generate_power_law_chung_lu_ensemble_from_seeds(
        n=n,
        tau=tau,
        target_mean_degree=target_mean_degree,
        w_min=w_min,
        w_max=w_max,
        cutoff_mode=cutoff_mode,
        cutoff_coefficient=cutoff_coefficient,
        sequence_mode=sequence_mode,
        graph_seeds=[int(seed) for seed in graph_seeds],
        sequence_seeds=[int(seed) for seed in sequence_seeds],
        component_mode=component_mode,
    )


def generate_power_law_chung_lu_ensemble_from_seeds(
    n: int,
    tau: float,
    target_mean_degree: float,
    w_min: float,
    w_max: float | None,
    cutoff_mode: str,
    cutoff_coefficient: float,
    sequence_mode: str,
    graph_seeds: list[int],
    sequence_seeds: list[int],
    component_mode: str = "full",
) -> list[ChungLuSpectrum]:
    if len(graph_seeds) != len(sequence_seeds):
        raise ValueError("graph_seeds and sequence_seeds must have the same length")
    return [
        analyze_power_law_chung_lu_graph(
            n=n,
            tau=tau,
            target_mean_degree=target_mean_degree,
            w_min=w_min,
            w_max=w_max,
            cutoff_mode=cutoff_mode,
            cutoff_coefficient=cutoff_coefficient,
            sequence_mode=sequence_mode,
            graph_seed=int(graph_seed),
            sequence_seed=int(seq_seed),
            component_mode=component_mode,
            graph_index=index,
        )
        for index, (graph_seed, seq_seed) in enumerate(zip(graph_seeds, sequence_seeds))
    ]
