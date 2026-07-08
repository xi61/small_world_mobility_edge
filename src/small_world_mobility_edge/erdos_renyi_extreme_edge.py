from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import integrate
from scipy.optimize import brentq
from scipy.sparse.linalg import ArpackNoConvergence, eigsh
from scipy.stats import ks_2samp

try:
    from .empirical_spectral_density import EmpiricalSpectralDensity, estimate_empirical_spectral_density
    from .erdos_renyi_baseline import (
        ErdosRenyiEdgeLevels,
        flat_vector_overlap_index,
        generate_erdos_renyi_graph,
        normalized_adjacency_sparse_matrix,
        sparse_normalized_edge_levels,
    )
    from .regular_baseline import adjacent_gap_ratios_from_spacings, wigner_surmise_goe
except ImportError:  # pragma: no cover - script-style fallback
    from empirical_spectral_density import EmpiricalSpectralDensity, estimate_empirical_spectral_density
    from erdos_renyi_baseline import (
        ErdosRenyiEdgeLevels,
        flat_vector_overlap_index,
        generate_erdos_renyi_graph,
        normalized_adjacency_sparse_matrix,
        sparse_normalized_edge_levels,
    )
    from regular_baseline import adjacent_gap_ratios_from_spacings, wigner_surmise_goe


MODEL = "erdos_renyi_extreme_edge_fixed_n_supplement"
FIXED_N = 10_000
DEGREE_SWEEP_MODE = "targeted_hybrid"
EXPLICIT_D_VALUES = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 24.0, 30.0, 40.0, 50.0]
B_VALUES = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.40, 2.55, 2.70, 3.00]
LINEAR_D_MIN = 2.0
LINEAR_D_MAX = 50.0
LINEAR_D_COUNT = 25
LOG_D_MIN = 2.0
LOG_D_MAX = 50.0
LOG_D_COUNT = 25
THEOREM_WINDOW_D_COUNT = 12
PAPER_LOG_WINDOW_B_MIN = 0.50
PAPER_LOG_WINDOW_B_MAX_FRACTION = 0.98
PAPER_LOG_WINDOW_COUNT = 12

DEFAULT_COMPONENT_MODE = "largest_connected_component"
COMPONENT_MODES = ["full", "largest_connected_component"]
PERRON_MODES = ["flat_vector_overlap", "largest_eigenvalue"]
NUM_DENSITY_GRAPHS = 30
NUM_EDGE_GRAPHS = 30
NUM_BULK_GRAPHS = 30
DENSITY_SEED = 31415
EDGE_SEED = 27182
BULK_SEED = 16180
EDGE_LEVELS = 24
CENTER_LEVELS = 48
MOBILITY_EDGE_LEVELS = 48
MOBILITY_EDGE_ENERGIES = [-2.0, 2.0]
INCLUDE_MOBILITY_EDGE_WINDOWS = True
MOBILITY_PROFILE_ENERGIES = [-3.0, -2.5, -2.2, -2.0, -1.8, -1.0, 0.0, 1.0, 1.8, 2.0, 2.2, 2.5, 3.0]
INCLUDE_MOBILITY_PROFILE = True
DENSE_THRESHOLD = 3000
QUICK_MODE = False
QUICK_NUM_DENSITY_GRAPHS = 3
QUICK_NUM_EDGE_GRAPHS = 3
QUICK_NUM_BULK_GRAPHS = 3
QUICK_EDGE_LEVELS = 8
QUICK_CENTER_LEVELS = 18
QUICK_MOBILITY_EDGE_LEVELS = 18
VALIDATE_REPRODUCIBILITY = True

SMALL_SPACING_THRESHOLD = 0.20
NUM_BINS = 30
KS_SAMPLE_SIZE = 4096
BOOTSTRAP_REPLICATIONS = 1000
DENSITY_METHOD = "local_count"
DENSITY_NUM_POINTS = 2001
DENSITY_NUM_BINS = 120
DENSITY_BANDWIDTH_SCALE = 1.0
EDGE_LOCAL_COUNT_HALF_WIDTH = 0.05
CENTER_LOCAL_COUNT_HALF_WIDTH = 0.10
THEOREM_ZETA = 4.1
THEOREM_XI = 0.02
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "erdos_renyi_edge_fixed_n_sweep"

GOE_MEAN_GAP_RATIO = 0.53590
POISSON_MEAN_GAP_RATIO = 2.0 * math.log(2.0) - 1.0


@dataclass(frozen=True)
class CenterLevels:
    graph_index: int
    seed: int
    component_mode: str
    analyzed_dimension: int
    n_requested: int
    average_degree_requested: float
    p: float
    num_components: int
    largest_component_size: int
    num_isolated_vertices: int
    eigenvalues: np.ndarray
    zero_multiplicity: int
    target_energy: float = 0.0
    eigenvectors: np.ndarray | None = None


@dataclass(frozen=True)
class PaperScalingDiagnostic:
    available: bool
    status: str
    u: float = math.nan
    d_times_u: float = math.nan
    sigma_u: float = math.nan
    tau_u: float = math.nan
    predicted_spacing_scale: float = math.nan


def timestamp_slug() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def poisson_spacing_density(s: np.ndarray | float) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    density = np.zeros_like(s, dtype=float)
    mask = s >= 0.0
    density[mask] = np.exp(-s[mask])
    return density


def stable_seed(base_seed: int, n: int, d: float, role: str) -> int:
    payload = f"{base_seed}|n={int(n)}|d={float(d):.12g}|role={role}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def make_seeds(num_graphs: int, seed: int) -> list[int]:
    if num_graphs <= 0:
        raise ValueError("num_graphs must be positive")
    rng = np.random.default_rng(int(seed))
    return [int(value) for value in rng.integers(0, 2**32 - 1, size=num_graphs, dtype=np.uint32)]


def make_role_seeds(base_seed: int, n: int, d: float, role: str, num_graphs: int) -> list[int]:
    return make_seeds(num_graphs=num_graphs, seed=stable_seed(base_seed=base_seed, n=n, d=d, role=role))


def _deduplicate_sort_degrees(values: Iterable[float], n: int, atol: float = 1e-10) -> list[float]:
    retained: list[float] = []
    for raw in values:
        value = float(raw)
        if not np.isfinite(value):
            raise ValueError(f"degree values must be finite; got {raw!r}")
        if not (0.0 < value < n - 1):
            raise ValueError(f"degree value must satisfy 0 < d < N - 1; got d={value}, N={n}")
        if not any(np.isclose(value, existing, atol=atol, rtol=0.0) for existing in retained):
            retained.append(value)
    return sorted(retained)


def build_explicit_degree_sequence(values: Iterable[float], n: int) -> list[float]:
    return _deduplicate_sort_degrees(values, n=n)


def build_linear_degree_sequence(d_min: float, d_max: float, count: int, n: int) -> list[float]:
    if count <= 0:
        raise ValueError("linear degree count must be positive")
    return _deduplicate_sort_degrees(np.linspace(float(d_min), float(d_max), int(count)), n=n)


def build_log_degree_sequence(d_min: float, d_max: float, count: int, n: int) -> list[float]:
    if count <= 0:
        raise ValueError("log degree count must be positive")
    if d_min <= 0 or d_max <= 0:
        raise ValueError("log degree bounds must be positive")
    return _deduplicate_sort_degrees(np.geomspace(float(d_min), float(d_max), int(count)), n=n)


def build_b_log_n_degree_sequence(b_values: Iterable[float], n: int) -> list[float]:
    log_n = float(np.log(float(n)))
    return _deduplicate_sort_degrees((float(b) * log_n for b in b_values), n=n)


def formal_theorem_degree_window(n: int) -> tuple[float, float, bool]:
    probe = theorem_range_diagnostic(n=n, d=1.0)
    lower = float(probe["theorem_lower"])
    upper = float(probe["theorem_upper"])
    return lower, upper, bool(lower <= upper)


def build_theorem_window_degree_sequence(
    n: int,
    count: int = THEOREM_WINDOW_D_COUNT,
    include_endpoints: bool = True,
) -> list[float]:
    if count <= 0:
        raise ValueError("theorem-window degree count must be positive")
    lower, upper, available = formal_theorem_degree_window(n)
    if not available:
        return []
    if include_endpoints or count <= 2:
        values = np.linspace(lower, upper, count)
    else:
        values = np.linspace(lower, upper, count + 2)[1:-1]
    return _deduplicate_sort_degrees(values, n=n)


def build_paper_log_window_degree_sequence(
    n: int,
    count: int = PAPER_LOG_WINDOW_COUNT,
    b_min: float = PAPER_LOG_WINDOW_B_MIN,
    b_max_fraction: float = PAPER_LOG_WINDOW_B_MAX_FRACTION,
) -> list[float]:
    if count <= 0:
        raise ValueError("paper-log-window degree count must be positive")
    if b_min <= 0.0:
        raise ValueError("paper-log-window b_min must be positive")
    b_star = float(1.0 / (math.log(4.0) - 1.0))
    b_max = float(b_max_fraction) * b_star
    if b_max <= b_min:
        raise ValueError(f"paper-log-window b_max={b_max} must exceed b_min={b_min}")
    log_n = float(np.log(float(n)))
    return _deduplicate_sort_degrees(np.linspace(b_min, b_max, int(count)) * log_n, n=n)


def build_active_degree_sequence(
    n: int,
    mode: str = DEGREE_SWEEP_MODE,
    explicit_values: Iterable[float] = EXPLICIT_D_VALUES,
    b_values: Iterable[float] = B_VALUES,
    linear_min: float = LINEAR_D_MIN,
    linear_max: float = LINEAR_D_MAX,
    linear_count: int = LINEAR_D_COUNT,
    log_min: float = LOG_D_MIN,
    log_max: float = LOG_D_MAX,
    log_count: int = LOG_D_COUNT,
    theorem_window_count: int = THEOREM_WINDOW_D_COUNT,
    paper_log_window_count: int = PAPER_LOG_WINDOW_COUNT,
) -> list[float]:
    if mode == "explicit":
        return build_explicit_degree_sequence(explicit_values, n=n)
    if mode == "linear":
        return build_linear_degree_sequence(linear_min, linear_max, linear_count, n=n)
    if mode == "log":
        return build_log_degree_sequence(log_min, log_max, log_count, n=n)
    if mode == "b_log_n":
        return build_b_log_n_degree_sequence(b_values, n=n)
    if mode == "theorem_window":
        values = build_theorem_window_degree_sequence(n=n, count=theorem_window_count)
        if not values:
            lower, upper, _ = formal_theorem_degree_window(n)
            raise ValueError(
                f"formal theorem-window degree mode requested, but finite-N proxy window is empty: "
                f"lower={lower:.6g}, upper={upper:.6g}, N={n}"
            )
        return values
    if mode == "paper_log_window":
        return build_paper_log_window_degree_sequence(n=n, count=paper_log_window_count)
    if mode == "targeted_hybrid":
        values = (
            list(explicit_values)
            + build_b_log_n_degree_sequence(b_values, n=n)
            + build_theorem_window_degree_sequence(n=n, count=theorem_window_count)
            + build_paper_log_window_degree_sequence(n=n, count=paper_log_window_count)
        )
        return _deduplicate_sort_degrees(values, n=n)
    if mode == "hybrid":
        values = list(explicit_values) + build_b_log_n_degree_sequence(b_values, n=n)
        return _deduplicate_sort_degrees(values, n=n)
    raise ValueError(f"unsupported DEGREE_SWEEP_MODE={mode!r}")


def select_component_graph(graph: nx.Graph, component_mode: str) -> nx.Graph:
    if component_mode == "full":
        return graph
    if component_mode == "largest_connected_component":
        largest_nodes = max(nx.connected_components(graph), key=len)
        return graph.subgraph(largest_nodes).copy()
    raise ValueError(f"unsupported component_mode={component_mode!r}")


def paper_f_d(u: float, d: float) -> float:
    u = float(u)
    d = float(d)
    if u <= 0.0 or d <= 0.0:
        raise ValueError("u and d must be positive")
    return u * math.log(u) - (u - 1.0) + (1.0 / d) * math.log(math.sqrt(2.0 * math.pi * d * u))


def f_d(u: float, d: float) -> float:
    return paper_f_d(u=u, d=d)


def Lambda(alpha: float, beta: float) -> float:
    alpha = float(alpha)
    beta = float(beta)
    discriminant = (alpha + beta) ** 2 - 4.0 * alpha
    if discriminant < 0.0:
        raise ValueError(f"Lambda discriminant is negative: alpha={alpha}, beta={beta}, disc={discriminant}")
    inner = alpha - 0.5 * beta * (alpha + beta) + 0.5 * beta * math.sqrt(discriminant)
    if inner <= 0.0:
        raise ValueError(f"Lambda inner term is non-positive: alpha={alpha}, beta={beta}, inner={inner}")
    return alpha * inner ** (-0.5)


def sigma(u: float, d: float) -> float:
    d = float(d)
    d_fraktur = 1.0 + 1.0 / d
    return math.sqrt(d_fraktur) * Lambda(float(u) / d_fraktur, 1.0 / d_fraktur)


def tau(u: float) -> float:
    u = float(u)
    if u <= 2.0:
        raise ValueError("tau(u) requires u > 2")
    return 2.0 * (u - 1.0) ** 2.5 / (math.sqrt(u) * (u - 2.0))


def solve_u(n: int, d: float) -> tuple[float | None, str]:
    if n <= 1 or d <= 0.0:
        return None, "invalid n or d"
    target = math.log(float(n)) / float(d)
    lo = 2.0 + 1e-10
    try:
        g_lo = paper_f_d(lo, d) - target
    except Exception as exc:
        return None, f"failed to evaluate lower bracket: {exc}"
    if g_lo >= 0.0:
        return None, f"no admissible u>2 root: f_d(2+)={paper_f_d(lo, d):.6g} >= log(N)/d={target:.6g}"

    hi = 4.0
    try:
        g_hi = paper_f_d(hi, d) - target
        while g_hi <= 0.0 and hi < 1e8:
            hi *= 2.0
            g_hi = paper_f_d(hi, d) - target
    except Exception as exc:
        return None, f"failed while expanding upper bracket: {exc}"
    if g_hi <= 0.0:
        return None, f"failed to bracket u root up to hi={hi:.6g}"
    try:
        return float(brentq(lambda x: paper_f_d(x, d) - target, lo, hi, maxiter=200)), "ok"
    except Exception as exc:
        return None, f"brentq failed for N={n}, d={d}: {exc}"


def paper_scaling_diagnostic(n: int, d: float) -> PaperScalingDiagnostic:
    u_value, status = solve_u(n=n, d=d)
    if u_value is None:
        return PaperScalingDiagnostic(available=False, status=status)
    try:
        sigma_value = sigma(u_value, d)
        tau_value = tau(u_value)
        scale = 1.0 / (float(d) * tau_value)
    except Exception as exc:
        return PaperScalingDiagnostic(available=False, status=f"paper scaling formula failed: {exc}")
    if not all(np.isfinite([u_value, sigma_value, tau_value, scale])):
        return PaperScalingDiagnostic(available=False, status="paper scaling formula produced non-finite values")
    return PaperScalingDiagnostic(
        available=True,
        status="ok",
        u=u_value,
        d_times_u=float(d) * u_value,
        sigma_u=sigma_value,
        tau_u=tau_value,
        predicted_spacing_scale=scale,
    )


def paper_edge_coordinates(values: np.ndarray, n: int, d: float, side: str) -> dict[str, object]:
    diagnostic = paper_scaling_diagnostic(n=n, d=d)
    values = np.sort(np.asarray(values, dtype=float))
    if not diagnostic.available:
        return {"diagnostic": diagnostic, "z_values": np.full(len(values), np.nan, dtype=float)}
    if side == "positive_edge":
        z_values = float(d) * diagnostic.tau_u * (values - diagnostic.sigma_u)
    elif side == "negative_edge":
        z_values = float(d) * diagnostic.tau_u * (-values - diagnostic.sigma_u)
    else:
        raise ValueError(f"paper_edge_coordinates only supports edge sides; got {side!r}")
    return {"diagnostic": diagnostic, "z_values": np.asarray(z_values, dtype=float)}


def theorem_range_diagnostic(n: int, d: float) -> dict[str, float | bool]:
    log_n = float(np.log(float(n)))
    log_log_n = float(np.log(log_n))
    b_star = float(1.0 / (math.log(4.0) - 1.0))
    theorem_lower = float(log_log_n ** THEOREM_ZETA)
    theorem_upper = float((b_star - log_n ** (-THEOREM_XI)) * log_n)
    d_over_log_n = float(d / log_n)
    lower_ratio_zeta4 = float(d / (log_log_n**4))
    formal_proxy = bool(theorem_lower <= d <= theorem_upper)
    asymptotic_log_window = bool(0.0 < d_over_log_n < b_star)
    legacy_heuristic = bool(d_over_log_n <= 2.0)
    return {
        "logN": log_n,
        "log_N": log_n,
        "log_log_N": log_log_n,
        "loglogN_to_4": float(log_log_n**4),
        "d_over_logN": d_over_log_n,
        "d_over_log_N": d_over_log_n,
        "b_star": b_star,
        "d_over_bstar_log_N": float(d / (b_star * log_n)),
        "lower_ratio_zeta4": lower_ratio_zeta4,
        "theorem_lower": theorem_lower,
        "theorem_upper": theorem_upper,
        "formal_proxy_in_range": formal_proxy,
        "asymptotic_below_bstar_log_window": asymptotic_log_window,
        "legacy_d_over_logN_le_2_heuristic": legacy_heuristic,
        "heuristic_theorem_range": legacy_heuristic,
    }


def build_degree_window_report(n: int, degree_values: list[float]) -> list[dict[str, object]]:
    report: list[dict[str, object]] = []
    for d in degree_values:
        info = theorem_range_diagnostic(n=n, d=d)
        report.append(
            {
                "N": int(n),
                "d": float(d),
                "d_over_log_N": float(info["d_over_log_N"]),
                "b_star": float(info["b_star"]),
                "d_over_bstar_log_N": float(info["d_over_bstar_log_N"]),
                "lower_ratio_zeta4": float(info["lower_ratio_zeta4"]),
                "theorem_lower": float(info["theorem_lower"]),
                "theorem_upper": float(info["theorem_upper"]),
                "formal_proxy_in_range": bool(info["formal_proxy_in_range"]),
                "asymptotic_below_bstar_log_window": bool(info["asymptotic_below_bstar_log_window"]),
                "legacy_d_over_logN_le_2_heuristic": bool(info["legacy_d_over_logN_le_2_heuristic"]),
            }
        )
    return report


def print_degree_window_report(report: list[dict[str, object]]) -> None:
    if not report:
        print("Degree window report: no degrees configured.")
        return
    lower = float(report[0]["theorem_lower"])
    upper = float(report[0]["theorem_upper"])
    in_window = [row for row in report if bool(row["formal_proxy_in_range"])]
    in_asymptotic_log_window = [row for row in report if bool(row["asymptotic_below_bstar_log_window"])]
    if lower > upper:
        print(
            f"Formal finite-N proxy window is empty because lower={lower:.6g} exceeds upper={upper:.6g}. "
            "This is not theorem verification."
        )
    else:
        print(
            f"Formal finite-N proxy window for d: [{lower:.6g}, {upper:.6g}]. "
            "This is not theorem verification."
        )
    if in_window:
        values = ", ".join(f"{float(row['d']):.6g}" for row in in_window)
        print(f"Degrees inside formal proxy window ({len(in_window)}): {values}")
    else:
        print("Degrees inside formal proxy window: none")
    if in_asymptotic_log_window:
        values = ", ".join(f"{float(row['d']):.6g}" for row in in_asymptotic_log_window)
        print(f"Degrees below b* log(N) asymptotic upper log-window ({len(in_asymptotic_log_window)}): {values}")
    else:
        print("Degrees below b* log(N) asymptotic upper log-window: none")
    print("Degree sweep table:")
    print(f"{'d':>10}  {'d/logN':>10}  {'d/(b*logN)':>12}  {'in_proxy':>9}  {'below_b*':>9}")
    for row in report:
        print(
            f"{float(row['d']):>10.6g}  "
            f"{float(row['d_over_log_N']):>10.4f}  "
            f"{float(row['d_over_bstar_log_N']):>12.4f}  "
            f"{str(bool(row['formal_proxy_in_range'])):>9}  "
            f"{str(bool(row['asymptotic_below_bstar_log_window'])):>9}"
        )


def write_degree_window_report(report: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "N",
        "d",
        "d_over_log_N",
        "b_star",
        "d_over_bstar_log_N",
        "lower_ratio_zeta4",
        "theorem_lower",
        "theorem_upper",
        "formal_proxy_in_range",
        "asymptotic_below_bstar_log_window",
        "legacy_d_over_logN_le_2_heuristic",
    ]
    _write_csv(report, path=path, fieldnames=fieldnames)


def _component_metadata(graph: nx.Graph) -> tuple[int, int, int]:
    num_components = nx.number_connected_components(graph)
    component_sizes = sorted((len(component) for component in nx.connected_components(graph)), reverse=True)
    largest_component_size = int(component_sizes[0])
    num_isolated = int(sum(1 for _, degree in graph.degree() if degree == 0))
    return num_components, largest_component_size, num_isolated


def _partial_arpack_result(exc: ArpackNoConvergence, min_count: int = 2) -> tuple[np.ndarray, np.ndarray] | None:
    values = getattr(exc, "eigenvalues", None)
    vectors = getattr(exc, "eigenvectors", None)
    if values is None or vectors is None:
        return None
    values = np.asarray(values, dtype=float)
    vectors = np.asarray(vectors, dtype=float)
    if values.ndim != 1 or vectors.ndim != 2 or len(values) < min_count:
        return None
    order = np.argsort(values)
    return values[order], vectors[:, order]


def sparse_local_eigenpairs(
    matrix,
    k: int,
    target_energy: float,
    min_count: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    k = int(k)
    if k < min_count:
        raise ValueError("k must be at least min_count")
    attempts = [
        {"sigma": float(target_energy), "which": "LM"},
        {"sigma": float(target_energy) + 1e-8, "which": "LM"},
        {"sigma": float(target_energy) - 1e-8, "which": "LM"},
    ]
    if np.isclose(target_energy, 0.0, atol=1e-12, rtol=0.0):
        attempts.append({"sigma": None, "which": "SM"})
    errors: list[str] = []
    for params in attempts:
        try:
            if params["sigma"] is None:
                values, vectors = eigsh(matrix, k=k, which=str(params["which"]))
            else:
                values, vectors = eigsh(matrix, k=k, sigma=float(params["sigma"]), which=str(params["which"]))
            order = np.argsort(np.asarray(values, dtype=float))
            return np.asarray(values, dtype=float)[order], np.asarray(vectors, dtype=float)[:, order]
        except ArpackNoConvergence as exc:
            partial = _partial_arpack_result(exc, min_count=min_count)
            if partial is not None:
                return partial
            errors.append(str(exc))
        except RuntimeError as exc:
            errors.append(str(exc))
    if k > min_count:
        return sparse_local_eigenpairs(matrix, k=max(min_count, k // 2), target_energy=target_energy, min_count=min_count)
    raise RuntimeError(f"local eigensolver failed near E={target_energy}: {' | '.join(errors)}")


def build_edge_levels_ensemble(
    n: int,
    average_degree: float,
    graph_seeds: list[int],
    component_mode: str,
) -> list[ErdosRenyiEdgeLevels]:
    levels: list[ErdosRenyiEdgeLevels] = []
    for graph_index, seed in enumerate(graph_seeds):
        graph = generate_erdos_renyi_graph(n=n, average_degree=average_degree, seed=seed)
        analyzed_graph = select_component_graph(graph, component_mode=component_mode)
        levels.append(
            sparse_normalized_edge_levels(
                analyzed_graph,
                average_degree=average_degree,
                edge_levels=EDGE_LEVELS,
                graph_index=graph_index,
                seed=seed,
                component_mode=component_mode,
                dense_threshold=DENSE_THRESHOLD,
            )
        )
    return levels


def build_center_levels_ensemble(
    n: int,
    average_degree: float,
    graph_seeds: list[int],
    component_mode: str,
    target_energy: float = 0.0,
    level_count: int | None = None,
) -> list[CenterLevels]:
    levels: list[CenterLevels] = []
    for graph_index, seed in enumerate(graph_seeds):
        graph = generate_erdos_renyi_graph(n=n, average_degree=average_degree, seed=seed)
        num_components, largest_component_size, num_isolated = _component_metadata(graph)
        analyzed_graph = select_component_graph(graph, component_mode=component_mode)
        analyzed_dimension = analyzed_graph.number_of_nodes()
        matrix = normalized_adjacency_sparse_matrix(analyzed_graph, average_degree=average_degree)
        requested_count = CENTER_LEVELS if level_count is None else int(level_count)
        center_count = min(requested_count, max(2, analyzed_dimension - 2))
        if analyzed_dimension <= DENSE_THRESHOLD:
            eigenvalues, eigenvectors = np.linalg.eigh(matrix.toarray())
            center_order = np.argsort(np.abs(eigenvalues - float(target_energy)))[:center_count]
            center_values = np.sort(eigenvalues[center_order])
            center_vectors = eigenvectors[:, center_order[np.argsort(eigenvalues[center_order])]]
            zero_multiplicity = int(np.sum(np.isclose(eigenvalues, 0.0, atol=1e-10, rtol=0.0)))
        else:
            center_values, center_vectors = sparse_local_eigenpairs(
                matrix,
                k=center_count,
                target_energy=float(target_energy),
                min_count=2,
            )
            zero_multiplicity = int(np.sum(np.isclose(center_values, 0.0, atol=1e-10, rtol=0.0)))
        levels.append(
            CenterLevels(
                graph_index=graph_index,
                seed=seed,
                component_mode=component_mode,
                analyzed_dimension=analyzed_dimension,
                n_requested=n,
                average_degree_requested=float(average_degree),
                p=float(average_degree) / (n - 1),
                num_components=num_components,
                largest_component_size=largest_component_size,
                num_isolated_vertices=num_isolated,
                eigenvalues=np.sort(np.asarray(center_values, dtype=float)),
                zero_multiplicity=zero_multiplicity,
                target_energy=float(target_energy),
                eigenvectors=np.asarray(center_vectors, dtype=float),
            )
        )
    return levels


def validate_reproducible_edge_level(
    *,
    n: int,
    d: float,
    seed: int,
    component_mode: str,
) -> None:
    first = build_edge_levels_ensemble(n=n, average_degree=d, graph_seeds=[seed], component_mode=component_mode)[0]
    second = build_edge_levels_ensemble(n=n, average_degree=d, graph_seeds=[seed], component_mode=component_mode)[0]
    if not np.allclose(first.positive_eigenvalues, second.positive_eigenvalues):
        raise AssertionError(f"positive edge eigenvalues are not reproducible for N={n}, d={d}, seed={seed}")
    if not np.allclose(first.negative_eigenvalues, second.negative_eigenvalues):
        raise AssertionError(f"negative edge eigenvalues are not reproducible for N={n}, d={d}, seed={seed}")


def edge_values_from_levels(levels: ErdosRenyiEdgeLevels, side: str, perron_mode: str) -> np.ndarray:
    if side == "negative_edge":
        return np.sort(np.asarray(levels.negative_eigenvalues, dtype=float))
    if side != "positive_edge":
        raise ValueError(f"unsupported side={side!r}")
    positive_eigs = np.asarray(levels.positive_eigenvalues, dtype=float)
    if perron_mode == "largest_eigenvalue":
        retained = np.sort(positive_eigs[:-1])
        if len(positive_eigs) - len(retained) != 1:
            raise AssertionError("exactly one positive edge level must be removed")
        return retained
    if perron_mode == "flat_vector_overlap":
        if levels.positive_vectors is None:
            raise ValueError("flat_vector_overlap requires positive edge eigenvectors")
        flat = np.ones(levels.positive_vectors.shape[0], dtype=float) / math.sqrt(levels.positive_vectors.shape[0])
        if not np.isclose(np.linalg.norm(flat), 1.0, atol=1e-12, rtol=0.0):
            raise AssertionError("flat vector must have unit norm")
        overlap_index = flat_vector_overlap_index(levels.positive_vectors)
        mask = np.ones(len(positive_eigs), dtype=bool)
        mask[overlap_index] = False
        retained = np.sort(positive_eigs[mask])
        if len(positive_eigs) - len(retained) != 1:
            raise AssertionError("exactly one positive edge level must be removed")
        return retained
    raise ValueError(f"unsupported perron_mode={perron_mode!r}")


def edge_vectors_from_levels(levels: ErdosRenyiEdgeLevels, side: str, perron_mode: str) -> np.ndarray | None:
    if side == "negative_edge":
        return None if levels.negative_vectors is None else np.asarray(levels.negative_vectors, dtype=float)
    if side != "positive_edge":
        raise ValueError(f"unsupported side={side!r}")
    if levels.positive_vectors is None:
        return None
    vectors = np.asarray(levels.positive_vectors, dtype=float)
    if perron_mode == "largest_eigenvalue":
        return vectors[:, :-1]
    if perron_mode == "flat_vector_overlap":
        overlap_index = flat_vector_overlap_index(vectors)
        mask = np.ones(vectors.shape[1], dtype=bool)
        mask[overlap_index] = False
        return vectors[:, mask]
    raise ValueError(f"unsupported perron_mode={perron_mode!r}")


def center_values_from_levels(levels: CenterLevels) -> np.ndarray:
    return np.sort(np.asarray(levels.eigenvalues, dtype=float))


def center_vectors_from_levels(levels: CenterLevels) -> np.ndarray | None:
    return None if levels.eigenvectors is None else np.asarray(levels.eigenvectors, dtype=float)


def _density_from_samples(samples: list[np.ndarray], local_count_half_width: float) -> tuple[EmpiricalSpectralDensity, float]:
    sorted_samples = [np.sort(np.asarray(sample, dtype=float)) for sample in samples]
    if any(sample.ndim != 1 or len(sample) == 0 for sample in sorted_samples):
        raise ValueError("density samples must be non-empty one-dimensional arrays")
    per_graph_count = float(np.mean([len(sample) for sample in sorted_samples]))
    density = estimate_empirical_spectral_density(
        eigenvalue_samples=sorted_samples,
        num_graphs=len(sorted_samples),
        per_graph_dimension=per_graph_count,
        method=DENSITY_METHOD,
        num_points=DENSITY_NUM_POINTS,
        num_bins=min(80, DENSITY_NUM_BINS),
        bandwidth_scale=DENSITY_BANDWIDTH_SCALE,
        local_count_half_width=local_count_half_width,
    )
    if np.any(np.diff(density.cdf) < -1e-10) or not np.isclose(float(density.cdf[-1]), 1.0, atol=0.05, rtol=0.0):
        raise ValueError("empirical CDF must be monotone and end near one")
    return density, per_graph_count


def build_edge_unfolding_model(
    density_levels: list[ErdosRenyiEdgeLevels],
    side: str,
    perron_mode: str,
) -> tuple[EmpiricalSpectralDensity, float]:
    samples = [edge_values_from_levels(levels, side=side, perron_mode=perron_mode) for levels in density_levels]
    return _density_from_samples(samples, local_count_half_width=EDGE_LOCAL_COUNT_HALF_WIDTH)


def _raw_gaps(values: np.ndarray) -> np.ndarray:
    values = np.sort(np.asarray(values, dtype=float))
    if np.any(np.diff(values) < -1e-12):
        raise ValueError("eigenvalues must be sorted")
    return np.diff(values)


def _gap_ratio_stats(samples: list[np.ndarray], rng_seed: int) -> dict[str, object]:
    graph_ratios: list[np.ndarray] = []
    graph_means: list[float] = []
    num_eigenvalues = 0
    num_gaps = 0
    for sample in samples:
        values = np.sort(np.asarray(sample, dtype=float))
        if np.any(np.diff(values) < -1e-12):
            raise ValueError("eigenvalues must be sorted")
        gaps = _raw_gaps(values)
        num_eigenvalues += len(values)
        num_gaps += len(gaps)
        ratios = adjacent_gap_ratios_from_spacings(gaps)
        if len(ratios) > 0:
            if np.any((ratios < -1e-12) | (ratios > 1.0 + 1e-12)):
                raise ValueError("gap ratios must lie in [0, 1]")
            graph_ratios.append(ratios)
            graph_means.append(float(np.mean(ratios)))
    pooled = np.concatenate(graph_ratios) if graph_ratios else np.array([], dtype=float)
    graph_means_array = np.asarray(graph_means, dtype=float)
    if len(graph_means_array) == 0:
        ci_low = ci_high = graph_equal = se = pooled_mean = math.nan
    else:
        graph_equal = float(np.mean(graph_means_array))
        se = float(np.std(graph_means_array, ddof=1) / math.sqrt(len(graph_means_array))) if len(graph_means_array) >= 2 else math.nan
        rng = np.random.default_rng(rng_seed)
        bootstrap = [
            float(np.mean(rng.choice(graph_means_array, size=len(graph_means_array), replace=True)))
            for _ in range(BOOTSTRAP_REPLICATIONS)
        ]
        ci_low, ci_high = [float(x) for x in np.quantile(bootstrap, [0.025, 0.975])]
        pooled_mean = float(np.mean(pooled)) if len(pooled) > 0 else math.nan
    return {
        "number_of_eigenvalues": int(num_eigenvalues),
        "number_of_gaps": int(num_gaps),
        "number_of_gap_ratios": int(len(pooled)),
        "pooled_gap_ratios": pooled,
        "pooled_mean_gap_ratio": pooled_mean,
        "graph_equal_mean_gap_ratio": graph_equal,
        "gap_ratio_standard_error": se,
        "gap_ratio_ci_low": ci_low,
        "gap_ratio_ci_high": ci_high,
    }


def _localization_stats(
    vector_samples: list[np.ndarray | None] | None,
    dimensions: list[int] | None,
) -> dict[str, float]:
    if vector_samples is None or dimensions is None:
        return {
            "mean_ipr": math.nan,
            "mean_n_times_ipr": math.nan,
            "mean_participation_ratio": math.nan,
            "mean_participation_fraction": math.nan,
        }
    ipr_values: list[np.ndarray] = []
    n_times_ipr_values: list[np.ndarray] = []
    participation_values: list[np.ndarray] = []
    participation_fraction_values: list[np.ndarray] = []
    for vectors, n_dim in zip(vector_samples, dimensions):
        if vectors is None:
            continue
        array = np.asarray(vectors, dtype=float)
        if array.ndim != 2 or array.shape[1] == 0:
            continue
        ipr = np.sum(np.abs(array) ** 4, axis=0)
        participation = 1.0 / np.maximum(ipr, 1e-15)
        ipr_values.append(ipr)
        n_times_ipr_values.append(float(n_dim) * ipr)
        participation_values.append(participation)
        participation_fraction_values.append(participation / max(float(n_dim), 1.0))
    if not ipr_values:
        return {
            "mean_ipr": math.nan,
            "mean_n_times_ipr": math.nan,
            "mean_participation_ratio": math.nan,
            "mean_participation_fraction": math.nan,
        }
    return {
        "mean_ipr": float(np.mean(np.concatenate(ipr_values))),
        "mean_n_times_ipr": float(np.mean(np.concatenate(n_times_ipr_values))),
        "mean_participation_ratio": float(np.mean(np.concatenate(participation_values))),
        "mean_participation_fraction": float(np.mean(np.concatenate(participation_fraction_values))),
    }


def _empirical_cdf_unfolding_stats(
    spacing_samples: list[np.ndarray],
    density: EmpiricalSpectralDensity,
    per_graph_count: float,
    rng_seed: int,
) -> dict[str, object]:
    spacings: list[np.ndarray] = []
    for sample in spacing_samples:
        values = np.sort(np.asarray(sample, dtype=float))
        unfolded = per_graph_count * np.interp(values, density.grid, density.cdf)
        gaps = np.diff(np.sort(unfolded))
        if len(gaps) > 0:
            spacings.append(gaps)
    if not spacings:
        return {
            "unfolding_method": "empirical_cdf_unfolding",
            "mean_unfolded_spacing": math.nan,
            "small_gap_probability": math.nan,
            "KS_to_GOE": math.nan,
            "KS_to_Poisson": math.nan,
            "spacings": np.array([], dtype=float),
        }
    unfolded_spacings = np.concatenate(spacings)
    sample_size = max(KS_SAMPLE_SIZE, len(unfolded_spacings))
    rng = np.random.default_rng(rng_seed)
    goe_sample = rng.rayleigh(scale=math.sqrt(2.0 / math.pi), size=sample_size)
    poisson_sample = rng.exponential(scale=1.0, size=sample_size)
    return {
        "unfolding_method": "empirical_cdf_unfolding",
        "mean_unfolded_spacing": float(np.mean(unfolded_spacings)),
        "small_gap_probability": float(np.mean(unfolded_spacings < SMALL_SPACING_THRESHOLD)),
        "KS_to_GOE": float(ks_2samp(unfolded_spacings, goe_sample).statistic),
        "KS_to_Poisson": float(ks_2samp(unfolded_spacings, poisson_sample).statistic),
        "spacings": unfolded_spacings,
    }


def compute_region_summary(
    *,
    n: int,
    d: float,
    component_mode: str,
    spectral_region: str,
    perron_mode: str,
    spacing_samples: list[np.ndarray],
    density_samples: list[np.ndarray],
    theorem_info: dict[str, float | bool],
    paper_info: PaperScalingDiagnostic,
    mean_component_size: float,
    mean_isolated_vertices: float,
    mean_zero_multiplicity: float,
    rng_seed: int,
    vector_samples: list[np.ndarray | None] | None = None,
    dimensions: list[int] | None = None,
    target_energy: float | None = None,
) -> dict[str, object]:
    density_half_width = (
        CENTER_LOCAL_COUNT_HALF_WIDTH
        if spectral_region == "center" or spectral_region.startswith("mobility_edge_")
        else EDGE_LOCAL_COUNT_HALF_WIDTH
    )
    density, per_graph_count = _density_from_samples(density_samples, local_count_half_width=density_half_width)
    ratio_stats = _gap_ratio_stats(spacing_samples, rng_seed=rng_seed + 1)
    unfolding_stats = _empirical_cdf_unfolding_stats(
        spacing_samples=spacing_samples,
        density=density,
        per_graph_count=per_graph_count,
        rng_seed=rng_seed + 2,
    )
    localization_stats = _localization_stats(vector_samples=vector_samples, dimensions=dimensions)
    energy_value = math.nan if target_energy is None else float(target_energy)
    return {
        "N": int(n),
        "d": float(d),
        "p": float(d) / (n - 1),
        "d_over_log_N": float(theorem_info["d_over_log_N"]),
        "d_over_logN": float(theorem_info["d_over_logN"]),
        "b_star": float(theorem_info["b_star"]),
        "component_mode": component_mode,
        "perron_mode": perron_mode,
        "spectral_region": spectral_region,
        "target_energy": "" if target_energy is None else energy_value,
        "energy_over_two": "" if target_energy is None else energy_value / 2.0,
        "abs_energy_minus_two": "" if target_energy is None else abs(energy_value) - 2.0,
        "unfolding_method": "empirical_cdf_unfolding",
        "number_of_graphs": len(spacing_samples),
        "mean_component_size": mean_component_size,
        "mean_isolated_vertices": mean_isolated_vertices,
        "mean_zero_multiplicity": mean_zero_multiplicity,
        "number_of_eigenvalues": ratio_stats["number_of_eigenvalues"],
        "number_of_gaps": ratio_stats["number_of_gaps"],
        "number_of_gap_ratios": ratio_stats["number_of_gap_ratios"],
        "pooled_mean_gap_ratio": ratio_stats["pooled_mean_gap_ratio"],
        "graph_equal_mean_gap_ratio": ratio_stats["graph_equal_mean_gap_ratio"],
        "gap_ratio_standard_error": ratio_stats["gap_ratio_standard_error"],
        "gap_ratio_ci_low": ratio_stats["gap_ratio_ci_low"],
        "gap_ratio_ci_high": ratio_stats["gap_ratio_ci_high"],
        "mean_spacing": unfolding_stats["mean_unfolded_spacing"],
        "mean_unfolded_spacing": unfolding_stats["mean_unfolded_spacing"],
        "small_gap_probability": unfolding_stats["small_gap_probability"],
        "KS_to_GOE": unfolding_stats["KS_to_GOE"],
        "KS_to_Poisson": unfolding_stats["KS_to_Poisson"],
        **localization_stats,
        "u": paper_info.u,
        "d_times_u": paper_info.d_times_u,
        "sigma_u": paper_info.sigma_u,
        "tau_u": paper_info.tau_u,
        "predicted_spacing_scale": paper_info.predicted_spacing_scale,
        "paper_scaling_available": paper_info.available,
        "paper_scaling_status": paper_info.status,
        "theorem_lower": float(theorem_info["theorem_lower"]),
        "theorem_upper": float(theorem_info["theorem_upper"]),
        "formal_proxy_in_range": bool(theorem_info["formal_proxy_in_range"]),
        "asymptotic_below_bstar_log_window": bool(theorem_info["asymptotic_below_bstar_log_window"]),
        "legacy_d_over_logN_le_2_heuristic": bool(theorem_info["legacy_d_over_logN_le_2_heuristic"]),
        "spacings": unfolding_stats["spacings"],
        "gap_ratios": ratio_stats["pooled_gap_ratios"],
        "density": density,
    }


def compute_edge_summary(
    spacing_levels: list[ErdosRenyiEdgeLevels],
    density_levels: list[ErdosRenyiEdgeLevels],
    side: str,
    perron_mode: str,
    component_mode: str,
) -> dict[str, object]:
    density, per_graph_count = build_edge_unfolding_model(density_levels, side=side, perron_mode=perron_mode)
    spacing_samples = [edge_values_from_levels(levels, side=side, perron_mode=perron_mode) for levels in spacing_levels]
    ratio_stats = _gap_ratio_stats(spacing_samples, rng_seed=EDGE_SEED)
    unfolding_stats = _empirical_cdf_unfolding_stats(
        spacing_samples=spacing_samples,
        density=density,
        per_graph_count=per_graph_count,
        rng_seed=EDGE_SEED + (11 if side == "positive_edge" else 23),
    )
    return {
        "spectral_region": side,
        "component_mode": component_mode,
        "perron_mode": perron_mode,
        "number_of_graphs": len(spacing_levels),
        "number_of_gaps": ratio_stats["number_of_gaps"],
        "mean_spacing": unfolding_stats["mean_unfolded_spacing"],
        "mean_gap_ratio": ratio_stats["pooled_mean_gap_ratio"],
        "pooled_mean_gap_ratio": ratio_stats["pooled_mean_gap_ratio"],
        "graph_equal_mean_gap_ratio": ratio_stats["graph_equal_mean_gap_ratio"],
        "small_gap_probability": unfolding_stats["small_gap_probability"],
        "KS_to_Poisson": unfolding_stats["KS_to_Poisson"],
        "KS_to_GOE": unfolding_stats["KS_to_GOE"],
        "spacings": unfolding_stats["spacings"],
        "rank_values": spacing_samples,
        "density": density,
    }


def _clean_row(row: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in row.items() if key not in {"spacings", "gap_ratios", "density"}}


def _write_csv(rows: list[dict[str, object]], path: Path, fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_summary_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "N",
        "d",
        "p",
        "d_over_log_N",
        "b_star",
        "component_mode",
        "perron_mode",
        "spectral_region",
        "target_energy",
        "energy_over_two",
        "abs_energy_minus_two",
        "unfolding_method",
        "number_of_graphs",
        "mean_component_size",
        "mean_isolated_vertices",
        "mean_zero_multiplicity",
        "number_of_eigenvalues",
        "number_of_gaps",
        "number_of_gap_ratios",
        "pooled_mean_gap_ratio",
        "graph_equal_mean_gap_ratio",
        "gap_ratio_standard_error",
        "gap_ratio_ci_low",
        "gap_ratio_ci_high",
        "mean_unfolded_spacing",
        "mean_spacing",
        "small_gap_probability",
        "KS_to_GOE",
        "KS_to_Poisson",
        "mean_ipr",
        "mean_n_times_ipr",
        "mean_participation_ratio",
        "mean_participation_fraction",
        "u",
        "d_times_u",
        "sigma_u",
        "tau_u",
        "predicted_spacing_scale",
        "paper_scaling_available",
        "paper_scaling_status",
        "theorem_lower",
        "theorem_upper",
        "formal_proxy_in_range",
        "asymptotic_below_bstar_log_window",
        "legacy_d_over_logN_le_2_heuristic",
    ]
    _write_csv([_clean_row(row) for row in rows], path=path, fieldnames=fieldnames)


def write_graph_level_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "N",
        "d",
        "p",
        "component_mode",
        "spectral_region",
        "perron_mode",
        "graph_index",
        "seed",
        "analyzed_dimension",
        "number_of_eigenvalues",
        "number_of_gaps",
        "number_of_gap_ratios",
        "mean_gap_ratio",
        "mean_ipr",
        "mean_n_times_ipr",
        "mean_participation_fraction",
        "zero_multiplicity",
        "num_isolated_vertices",
    ]
    _write_csv(rows, path=path, fieldnames=fieldnames)


def write_paper_scaled_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "N",
        "d",
        "component_mode",
        "spectral_region",
        "perron_mode",
        "graph_index",
        "seed",
        "rank",
        "lambda",
        "paper_edge_coordinate",
        "u",
        "d_times_u",
        "sigma_u",
        "tau_u",
        "predicted_spacing_scale",
        "paper_scaling_available",
        "paper_scaling_status",
    ]
    _write_csv(rows, path=path, fieldnames=fieldnames)


def write_seed_manifest(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = ["N", "d", "component_mode", "ensemble_role", "graph_index", "seed"]
    _write_csv(rows, path=path, fieldnames=fieldnames)


def graph_level_rows_for_samples(
    *,
    n: int,
    d: float,
    component_mode: str,
    spectral_region: str,
    perron_mode: str,
    seeds: list[int],
    dimensions: list[int],
    zero_multiplicities: list[int],
    isolated_counts: list[int],
    samples: list[np.ndarray],
    vector_samples: list[np.ndarray | None] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, sample in enumerate(samples):
        gaps = _raw_gaps(sample)
        ratios = adjacent_gap_ratios_from_spacings(gaps)
        vectors = None if vector_samples is None else vector_samples[index]
        localization = _localization_stats(
            vector_samples=[vectors] if vectors is not None else None,
            dimensions=[dimensions[index]],
        )
        rows.append(
            {
                "N": n,
                "d": d,
                "p": d / (n - 1),
                "component_mode": component_mode,
                "spectral_region": spectral_region,
                "perron_mode": perron_mode,
                "graph_index": index,
                "seed": seeds[index],
                "analyzed_dimension": dimensions[index],
                "number_of_eigenvalues": len(sample),
                "number_of_gaps": len(gaps),
                "number_of_gap_ratios": len(ratios),
                "mean_gap_ratio": float(np.mean(ratios)) if len(ratios) > 0 else math.nan,
                "mean_ipr": localization["mean_ipr"],
                "mean_n_times_ipr": localization["mean_n_times_ipr"],
                "mean_participation_fraction": localization["mean_participation_fraction"],
                "zero_multiplicity": zero_multiplicities[index],
                "num_isolated_vertices": isolated_counts[index],
            }
        )
    return rows


def paper_scaled_rows_for_edge_samples(
    *,
    n: int,
    d: float,
    component_mode: str,
    spectral_region: str,
    perron_mode: str,
    levels_list: list[ErdosRenyiEdgeLevels],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for levels in levels_list:
        values = edge_values_from_levels(levels, side=spectral_region, perron_mode=perron_mode)
        payload = paper_edge_coordinates(values, n=n, d=d, side=spectral_region)
        diagnostic: PaperScalingDiagnostic = payload["diagnostic"]  # type: ignore[assignment]
        z_values = np.asarray(payload["z_values"], dtype=float)
        ordered_indices = np.arange(1, len(values) + 1)
        if spectral_region == "positive_edge":
            values_for_rank = values[::-1]
            z_for_rank = z_values[::-1]
        else:
            values_for_rank = values
            z_for_rank = z_values
        for rank, value, z_value in zip(ordered_indices, values_for_rank, z_for_rank):
            rows.append(
                {
                    "N": n,
                    "d": d,
                    "component_mode": component_mode,
                    "spectral_region": spectral_region,
                    "perron_mode": perron_mode,
                    "graph_index": levels.graph_index,
                    "seed": levels.seed,
                    "rank": int(rank),
                    "lambda": float(value),
                    "paper_edge_coordinate": float(z_value) if np.isfinite(z_value) else "",
                    "u": diagnostic.u,
                    "d_times_u": diagnostic.d_times_u,
                    "sigma_u": diagnostic.sigma_u,
                    "tau_u": diagnostic.tau_u,
                    "predicted_spacing_scale": diagnostic.predicted_spacing_scale,
                    "paper_scaling_available": diagnostic.available,
                    "paper_scaling_status": diagnostic.status,
                }
            )
    return rows


def _mean_metadata_from_center(center_levels: list[CenterLevels]) -> dict[str, float]:
    return {
        "mean_component_size": float(np.mean([levels.analyzed_dimension for levels in center_levels])),
        "mean_isolated_vertices": float(np.mean([levels.num_isolated_vertices for levels in center_levels])),
        "mean_zero_multiplicity": float(np.mean([levels.zero_multiplicity for levels in center_levels])),
    }


def run_degree(
    *,
    n: int,
    d: float,
    component_mode: str,
    num_density_graphs: int,
    num_edge_graphs: int,
    num_bulk_graphs: int,
    mobility_profile_energies: list[float],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    if not (0.0 < d < n - 1):
        raise ValueError(f"d must satisfy 0 < d < N - 1; got d={d}, N={n}")
    p = d / (n - 1)
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must lie in (0, 1); got p={p}")
    theorem_info = theorem_range_diagnostic(n=n, d=d)
    paper_info = paper_scaling_diagnostic(n=n, d=d)
    density_seeds = make_role_seeds(DENSITY_SEED, n=n, d=d, role="edge_density", num_graphs=num_density_graphs)
    edge_seeds = make_role_seeds(EDGE_SEED, n=n, d=d, role="edge_spacing", num_graphs=num_edge_graphs)
    center_seeds = make_role_seeds(BULK_SEED, n=n, d=d, role="center_control", num_graphs=num_bulk_graphs)

    seed_rows = [
        {"N": n, "d": d, "component_mode": component_mode, "ensemble_role": role, "graph_index": index, "seed": seed}
        for role, seeds in [("edge_density", density_seeds), ("edge_spacing", edge_seeds), ("center_control", center_seeds)]
        for index, seed in enumerate(seeds)
    ]

    edge_density_levels = build_edge_levels_ensemble(n=n, average_degree=d, graph_seeds=density_seeds, component_mode=component_mode)
    edge_spacing_levels = build_edge_levels_ensemble(n=n, average_degree=d, graph_seeds=edge_seeds, component_mode=component_mode)
    center_levels = build_center_levels_ensemble(n=n, average_degree=d, graph_seeds=center_seeds, component_mode=component_mode)
    mobility_levels_by_energy: dict[float, list[CenterLevels]] = {}
    if INCLUDE_MOBILITY_EDGE_WINDOWS:
        for energy in MOBILITY_EDGE_ENERGIES:
            mobility_levels_by_energy[float(energy)] = build_center_levels_ensemble(
                n=n,
                average_degree=d,
                graph_seeds=center_seeds,
                component_mode=component_mode,
                target_energy=float(energy),
                level_count=MOBILITY_EDGE_LEVELS,
            )
    profile_levels_by_energy: dict[float, list[CenterLevels]] = {}
    if INCLUDE_MOBILITY_PROFILE:
        for energy in mobility_profile_energies:
            energy = float(energy)
            matching_mobility_energy = next(
                (
                    existing
                    for existing in mobility_levels_by_energy
                    if np.isclose(energy, existing, atol=1e-12, rtol=0.0)
                ),
                None,
            )
            if np.isclose(energy, 0.0, atol=1e-12, rtol=0.0):
                profile_levels_by_energy[energy] = center_levels
            elif matching_mobility_energy is not None:
                profile_levels_by_energy[energy] = mobility_levels_by_energy[matching_mobility_energy]
            else:
                profile_levels_by_energy[energy] = build_center_levels_ensemble(
                    n=n,
                    average_degree=d,
                    graph_seeds=center_seeds,
                    component_mode=component_mode,
                    target_energy=energy,
                    level_count=MOBILITY_EDGE_LEVELS,
                )
    if VALIDATE_REPRODUCIBILITY and edge_seeds:
        regenerated = make_role_seeds(EDGE_SEED, n=n, d=d, role="edge_spacing", num_graphs=num_edge_graphs)
        if regenerated != edge_seeds:
            raise AssertionError("stable seed generation is not reproducible")
        validate_reproducible_edge_level(n=n, d=d, seed=edge_seeds[0], component_mode=component_mode)

    metadata = _mean_metadata_from_center(center_levels)
    summary_rows: list[dict[str, object]] = []
    graph_rows: list[dict[str, object]] = []
    paper_rows: list[dict[str, object]] = []

    center_samples = [center_values_from_levels(levels) for levels in center_levels]
    center_vectors = [center_vectors_from_levels(levels) for levels in center_levels]
    center_dimensions = [levels.analyzed_dimension for levels in center_levels]
    center_row = compute_region_summary(
        n=n,
        d=d,
        component_mode=component_mode,
        spectral_region="center",
        perron_mode="",
        spacing_samples=center_samples,
        density_samples=center_samples,
        theorem_info=theorem_info,
        paper_info=paper_info,
        rng_seed=stable_seed(BULK_SEED, n=n, d=d, role="center_stats"),
        vector_samples=center_vectors,
        dimensions=center_dimensions,
        target_energy=0.0,
        **metadata,
    )
    summary_rows.append(center_row)
    graph_rows.extend(
        graph_level_rows_for_samples(
            n=n,
            d=d,
            component_mode=component_mode,
            spectral_region="center",
            perron_mode="",
            seeds=center_seeds,
            dimensions=[levels.analyzed_dimension for levels in center_levels],
            zero_multiplicities=[levels.zero_multiplicity for levels in center_levels],
            isolated_counts=[levels.num_isolated_vertices for levels in center_levels],
            samples=center_samples,
            vector_samples=center_vectors,
        )
    )

    for energy, mobility_levels in mobility_levels_by_energy.items():
        mobility_samples = [center_values_from_levels(levels) for levels in mobility_levels]
        mobility_vectors = [center_vectors_from_levels(levels) for levels in mobility_levels]
        mobility_region = "mobility_edge_minus_2" if energy < 0.0 else "mobility_edge_plus_2"
        mobility_row = compute_region_summary(
            n=n,
            d=d,
            component_mode=component_mode,
            spectral_region=mobility_region,
            perron_mode="",
            spacing_samples=mobility_samples,
            density_samples=mobility_samples,
            theorem_info=theorem_info,
            paper_info=paper_info,
            rng_seed=stable_seed(BULK_SEED, n=n, d=d, role=f"mobility_edge_{energy:g}_stats"),
            vector_samples=mobility_vectors,
            dimensions=[levels.analyzed_dimension for levels in mobility_levels],
            target_energy=energy,
            **metadata,
        )
        summary_rows.append(mobility_row)
        graph_rows.extend(
            graph_level_rows_for_samples(
                n=n,
                d=d,
                component_mode=component_mode,
                spectral_region=mobility_region,
                perron_mode="",
                seeds=center_seeds,
                dimensions=[levels.analyzed_dimension for levels in mobility_levels],
                zero_multiplicities=[levels.zero_multiplicity for levels in mobility_levels],
                isolated_counts=[levels.num_isolated_vertices for levels in mobility_levels],
                samples=mobility_samples,
                vector_samples=mobility_vectors,
            )
        )

    for energy, profile_levels in profile_levels_by_energy.items():
        profile_samples = [center_values_from_levels(levels) for levels in profile_levels]
        profile_vectors = [center_vectors_from_levels(levels) for levels in profile_levels]
        profile_row = compute_region_summary(
            n=n,
            d=d,
            component_mode=component_mode,
            spectral_region="mobility_profile",
            perron_mode="",
            spacing_samples=profile_samples,
            density_samples=profile_samples,
            theorem_info=theorem_info,
            paper_info=paper_info,
            rng_seed=stable_seed(BULK_SEED, n=n, d=d, role=f"mobility_profile_{energy:g}_stats"),
            vector_samples=profile_vectors,
            dimensions=[levels.analyzed_dimension for levels in profile_levels],
            target_energy=energy,
            **metadata,
        )
        summary_rows.append(profile_row)
        graph_rows.extend(
            graph_level_rows_for_samples(
                n=n,
                d=d,
                component_mode=component_mode,
                spectral_region="mobility_profile",
                perron_mode="",
                seeds=center_seeds,
                dimensions=[levels.analyzed_dimension for levels in profile_levels],
                zero_multiplicities=[levels.zero_multiplicity for levels in profile_levels],
                isolated_counts=[levels.num_isolated_vertices for levels in profile_levels],
                samples=profile_samples,
                vector_samples=profile_vectors,
            )
        )

    negative_spacing_samples = [edge_values_from_levels(levels, side="negative_edge", perron_mode="largest_eigenvalue") for levels in edge_spacing_levels]
    negative_density_samples = [edge_values_from_levels(levels, side="negative_edge", perron_mode="largest_eigenvalue") for levels in edge_density_levels]
    negative_vectors = [edge_vectors_from_levels(levels, side="negative_edge", perron_mode="largest_eigenvalue") for levels in edge_spacing_levels]
    edge_dimensions = [levels.analyzed_dimension for levels in edge_spacing_levels]
    negative_row = compute_region_summary(
        n=n,
        d=d,
        component_mode=component_mode,
        spectral_region="negative_edge",
        perron_mode="",
        spacing_samples=negative_spacing_samples,
        density_samples=negative_density_samples,
        theorem_info=theorem_info,
        paper_info=paper_info,
        rng_seed=stable_seed(EDGE_SEED, n=n, d=d, role="negative_stats"),
        vector_samples=negative_vectors,
        dimensions=edge_dimensions,
        **metadata,
    )
    summary_rows.append(negative_row)
    graph_rows.extend(
        graph_level_rows_for_samples(
            n=n,
            d=d,
            component_mode=component_mode,
            spectral_region="negative_edge",
            perron_mode="",
            seeds=edge_seeds,
            dimensions=[levels.analyzed_dimension for levels in edge_spacing_levels],
            zero_multiplicities=[levels.zero_multiplicity for levels in edge_spacing_levels],
            isolated_counts=[0 for _ in edge_spacing_levels],
            samples=negative_spacing_samples,
            vector_samples=negative_vectors,
        )
    )
    paper_rows.extend(
        paper_scaled_rows_for_edge_samples(
            n=n,
            d=d,
            component_mode=component_mode,
            spectral_region="negative_edge",
            perron_mode="",
            levels_list=edge_spacing_levels,
        )
    )

    for perron_mode in PERRON_MODES:
        positive_spacing_samples = [edge_values_from_levels(levels, side="positive_edge", perron_mode=perron_mode) for levels in edge_spacing_levels]
        positive_density_samples = [edge_values_from_levels(levels, side="positive_edge", perron_mode=perron_mode) for levels in edge_density_levels]
        positive_vectors = [edge_vectors_from_levels(levels, side="positive_edge", perron_mode=perron_mode) for levels in edge_spacing_levels]
        positive_row = compute_region_summary(
            n=n,
            d=d,
            component_mode=component_mode,
            spectral_region="positive_edge",
            perron_mode=perron_mode,
            spacing_samples=positive_spacing_samples,
            density_samples=positive_density_samples,
            theorem_info=theorem_info,
            paper_info=paper_info,
            rng_seed=stable_seed(EDGE_SEED, n=n, d=d, role=f"positive_stats_{perron_mode}"),
            vector_samples=positive_vectors,
            dimensions=edge_dimensions,
            **metadata,
        )
        summary_rows.append(positive_row)
        graph_rows.extend(
            graph_level_rows_for_samples(
                n=n,
                d=d,
                component_mode=component_mode,
                spectral_region="positive_edge",
                perron_mode=perron_mode,
                seeds=edge_seeds,
                dimensions=[levels.analyzed_dimension for levels in edge_spacing_levels],
                zero_multiplicities=[levels.zero_multiplicity for levels in edge_spacing_levels],
                isolated_counts=[0 for _ in edge_spacing_levels],
                samples=positive_spacing_samples,
                vector_samples=positive_vectors,
            )
        )
        paper_rows.extend(
            paper_scaled_rows_for_edge_samples(
                n=n,
                d=d,
                component_mode=component_mode,
                spectral_region="positive_edge",
                perron_mode=perron_mode,
                levels_list=edge_spacing_levels,
            )
        )

    print(
        f"N={n}, d={d:.6g}, component_mode={component_mode}: "
        f"d/log(N)={theorem_info['d_over_log_N']:.4f}, "
        f"formal_proxy_in_range={theorem_info['formal_proxy_in_range']}. "
        "This is only a finite-N check of the written asymptotic inequalities, "
        "not a theorem guarantee and not theorem verification."
    )
    return summary_rows, graph_rows, paper_rows, seed_rows


def _finite_y(rows: list[dict[str, object]], key: str) -> np.ndarray:
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    return values[np.isfinite(values)]


def plot_metric(
    rows: list[dict[str, object]],
    spectral_region: str,
    y_keys: list[str],
    y_label: str,
    title: str,
    filename_prefix: str,
    output_dir: Path,
    perron_mode: str | None = None,
) -> None:
    subset = [row for row in rows if row["spectral_region"] == spectral_region]
    if perron_mode is not None:
        subset = [row for row in subset if row["perron_mode"] == perron_mode]
    if not subset:
        return
    for x_key, x_label, suffix in [("d", "d", "vs_d"), ("d_over_log_N", "d / log(N)", "vs_dlogN")]:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for y_key in y_keys:
            valid = [row for row in subset if row.get(y_key, "") != "" and np.isfinite(float(row[y_key]))]
            valid = sorted(valid, key=lambda row: float(row[x_key]))
            if not valid:
                continue
            x = np.asarray([float(row[x_key]) for row in valid], dtype=float)
            y = np.asarray([float(row[y_key]) for row in valid], dtype=float)
            ax.plot(x, y, marker="o", label=y_key)
        if "gap_ratio" in filename_prefix:
            ax.axhline(GOE_MEAN_GAP_RATIO, color="black", ls="--", label="GOE mean r")
            ax.axhline(POISSON_MEAN_GAP_RATIO, color="firebrick", ls=":", label="Poisson mean r")
        if x_key == "d":
            n = int(subset[0]["N"])
            log_n = math.log(float(n))
            b_star = float(subset[0]["b_star"])
            ax.axvline(log_n, color="gray", ls="--", alpha=0.7, label="d=log(N)")
            ax.axvline(b_star * log_n, color="gray", ls=":", alpha=0.7, label="d=b_star log(N)")
        if x_key == "d_over_log_N":
            b_star = float(subset[0]["b_star"])
            ax.axvline(1.0, color="gray", ls="--", alpha=0.7, label="d=log(N)")
            ax.axvline(b_star, color="gray", ls=":", alpha=0.7, label="d=b_star log(N)")
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{filename_prefix}_{suffix}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_mobility_profile_by_energy(rows: list[dict[str, object]], output_dir: Path) -> None:
    subset = [row for row in rows if row["spectral_region"] == "mobility_profile"]
    if not subset:
        return
    for y_key, y_label, filename in [
        ("mean_n_times_ipr", "mean n*IPR", "mobility_profile_n_times_ipr_vs_energy.png"),
        ("mean_participation_fraction", "mean PR / n", "mobility_profile_participation_fraction_vs_energy.png"),
        ("pooled_mean_gap_ratio", "pooled mean gap ratio", "mobility_profile_gap_ratio_vs_energy.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        for d in sorted({float(row["d"]) for row in subset}):
            valid = [
                row for row in subset
                if np.isclose(float(row["d"]), d, rtol=0.0, atol=1e-10)
                and row.get("target_energy", "") != ""
                and np.isfinite(float(row[y_key]))
            ]
            valid = sorted(valid, key=lambda row: float(row["target_energy"]))
            if not valid:
                continue
            x = np.asarray([float(row["target_energy"]) for row in valid], dtype=float)
            y = np.asarray([float(row[y_key]) for row in valid], dtype=float)
            dlog = float(valid[0]["d_over_log_N"])
            ax.plot(x, y, marker="o", label=f"d={d:.3g}, d/logN={dlog:.2f}")
        ax.axvline(-2.0, color="gray", ls="--", alpha=0.8, label="E=-2")
        ax.axvline(2.0, color="gray", ls=":", alpha=0.8, label="E=2")
        if y_key == "pooled_mean_gap_ratio":
            ax.axhline(GOE_MEAN_GAP_RATIO, color="black", ls="--", alpha=0.75, label="GOE mean r")
            ax.axhline(POISSON_MEAN_GAP_RATIO, color="firebrick", ls=":", alpha=0.75, label="Poisson mean r")
        ax.set_title("Mobility Profile Across H=A/sqrt(d) Energy")
        ax.set_xlabel("target energy E")
        ax.set_ylabel(y_label)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)


def make_summary_plots(rows: list[dict[str, object]], output_dir: Path) -> None:
    plot_metric(rows, "center", ["pooled_mean_gap_ratio", "graph_equal_mean_gap_ratio"], "mean gap ratio", "Center Mean Gap Ratio", "center_mean_gap_ratio", output_dir)
    plot_metric(rows, "positive_edge", ["pooled_mean_gap_ratio", "graph_equal_mean_gap_ratio"], "mean gap ratio", "Positive Edge Mean Gap Ratio", "positive_edge_mean_gap_ratio", output_dir, perron_mode="flat_vector_overlap")
    plot_metric(rows, "negative_edge", ["pooled_mean_gap_ratio", "graph_equal_mean_gap_ratio"], "mean gap ratio", "Negative Edge Mean Gap Ratio", "negative_edge_mean_gap_ratio", output_dir)
    plot_metric(rows, "center", ["KS_to_GOE", "KS_to_Poisson"], "KS distance", "Center Empirical-CDF KS Distances", "center_ks_distance", output_dir)
    plot_metric(rows, "positive_edge", ["KS_to_GOE", "KS_to_Poisson"], "KS distance", "Positive Edge Empirical-CDF KS Distances", "positive_edge_ks_distance", output_dir, perron_mode="flat_vector_overlap")
    plot_metric(rows, "negative_edge", ["KS_to_GOE", "KS_to_Poisson"], "KS distance", "Negative Edge Empirical-CDF KS Distances", "negative_edge_ks_distance", output_dir)
    plot_metric(rows, "center", ["small_gap_probability"], f"P(s < {SMALL_SPACING_THRESHOLD})", "Center Small-Gap Probability", "center_small_gap_probability", output_dir)
    plot_metric(rows, "positive_edge", ["small_gap_probability"], f"P(s < {SMALL_SPACING_THRESHOLD})", "Positive Edge Small-Gap Probability", "positive_edge_small_gap_probability", output_dir, perron_mode="flat_vector_overlap")
    plot_metric(rows, "negative_edge", ["small_gap_probability"], f"P(s < {SMALL_SPACING_THRESHOLD})", "Negative Edge Small-Gap Probability", "negative_edge_small_gap_probability", output_dir)
    plot_metric(rows, "center", ["mean_component_size", "mean_isolated_vertices"], "count", "Mean Component Size and Isolated Vertices", "component_size_isolates", output_dir)
    for region, title_prefix, filename_prefix in [
        ("center", "Center", "center"),
        ("mobility_edge_minus_2", "Mobility Edge E=-2", "mobility_edge_minus_2"),
        ("mobility_edge_plus_2", "Mobility Edge E=2", "mobility_edge_plus_2"),
        ("positive_edge", "Positive Edge", "positive_edge"),
        ("negative_edge", "Negative Edge", "negative_edge"),
    ]:
        perron_mode = "flat_vector_overlap" if region == "positive_edge" else None
        plot_metric(
            rows,
            region,
            ["mean_n_times_ipr"],
            "mean n*IPR",
            f"{title_prefix} Localization",
            f"{filename_prefix}_n_times_ipr",
            output_dir,
            perron_mode=perron_mode,
        )
        plot_metric(
            rows,
            region,
            ["mean_participation_fraction"],
            "mean PR / n",
            f"{title_prefix} Participation Fraction",
            f"{filename_prefix}_participation_fraction",
            output_dir,
            perron_mode=perron_mode,
        )
    plot_metric(rows, "positive_edge", ["sigma_u"], "sigma(u)", "Paper Coordinate sigma(u)", "sigma_u", output_dir, perron_mode="flat_vector_overlap")
    plot_metric(rows, "positive_edge", ["predicted_spacing_scale"], "1/(d tau(u))", "Paper Coordinate Predicted Spacing Scale", "predicted_spacing_scale", output_dir, perron_mode="flat_vector_overlap")
    plot_mobility_profile_by_energy(rows, output_dir)


def run_validation_checks(
    rows: list[dict[str, object]],
    degree_values: list[float] | None = None,
    seed_rows: list[dict[str, object]] | None = None,
) -> None:
    assert rows
    for row in rows:
        n = int(row["N"])
        d = float(row["d"])
        p = float(row["p"])
        assert 0.0 < d < n - 1
        assert 0.0 < p < 1.0
        assert row["component_mode"] in COMPONENT_MODES
        if int(row["number_of_gap_ratios"]) > 0:
            assert np.isfinite(float(row["pooled_mean_gap_ratio"]))
            assert 0.0 <= float(row["pooled_mean_gap_ratio"]) <= 1.0
        if int(row["number_of_gaps"]) > 0:
            assert np.isfinite(float(row["small_gap_probability"]))
        if bool(row["paper_scaling_available"]):
            for key in ["u", "d_times_u", "sigma_u", "tau_u", "predicted_spacing_scale"]:
                assert np.isfinite(float(row[key]))
    if degree_values is not None:
        assert all(0.0 < float(d) < int(rows[0]["N"]) - 1 for d in degree_values)
    if seed_rows is not None:
        assert seed_rows == list(seed_rows)


def config_payload(
    args: argparse.Namespace,
    degree_values: list[float],
    degree_window_report: list[dict[str, object]],
    output_dir: Path,
) -> dict[str, object]:
    return {
        "model": MODEL,
        "fixed_n": int(args.n),
        "degree_sweep_mode": args.degree_sweep_mode,
        "theorem_window_count": args.theorem_window_count,
        "paper_log_window_count": args.paper_log_window_count,
        "degree_values": degree_values,
        "d_over_log_n": [float(d / math.log(float(args.n))) for d in degree_values],
        "degree_window_report": degree_window_report,
        "degrees_inside_formal_proxy_window": [
            float(row["d"]) for row in degree_window_report if bool(row["formal_proxy_in_range"])
        ],
        "degrees_below_bstar_log_window": [
            float(row["d"]) for row in degree_window_report if bool(row["asymptotic_below_bstar_log_window"])
        ],
        "component_mode": args.component_mode,
        "perron_modes": PERRON_MODES,
        "num_density_graphs": args.num_density_graphs,
        "num_edge_graphs": args.num_edge_graphs,
        "num_bulk_graphs": args.num_bulk_graphs,
        "edge_levels": EDGE_LEVELS,
        "center_levels": CENTER_LEVELS,
        "mobility_edge_levels": MOBILITY_EDGE_LEVELS,
        "include_mobility_edge_windows": INCLUDE_MOBILITY_EDGE_WINDOWS,
        "include_mobility_profile": INCLUDE_MOBILITY_PROFILE,
        "mobility_profile_energies": args.mobility_profile_energies,
        "density_seed": DENSITY_SEED,
        "edge_seed": EDGE_SEED,
        "bulk_seed": BULK_SEED,
        "theorem_zeta": THEOREM_ZETA,
        "theorem_xi": THEOREM_XI,
        "output_dir": str(output_dir),
        "timestamp": timestamp_slug(),
        "warning": "This is only a finite-N check of the written asymptotic inequalities, not a theorem guarantee and not theorem verification.",
    }


def parse_float_list(text: str | None) -> list[float] | None:
    if text is None:
        return None
    values = [item.strip() for item in text.split(",") if item.strip()]
    return [float(item) for item in values]


def _deduplicate_sort_energies(values: Iterable[float], atol: float = 1e-10) -> list[float]:
    retained: list[float] = []
    for raw in values:
        value = float(raw)
        if not np.isfinite(value):
            raise ValueError(f"mobility profile energies must be finite; got {raw!r}")
        if not any(np.isclose(value, existing, atol=atol, rtol=0.0) for existing in retained):
            retained.append(value)
    return sorted(retained)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed-N exploratory ER extreme-edge degree sweep.")
    parser.add_argument("--n", type=int, default=FIXED_N)
    parser.add_argument(
        "--degree-sweep-mode",
        choices=["explicit", "linear", "log", "b_log_n", "hybrid", "theorem_window", "paper_log_window", "targeted_hybrid"],
        default=DEGREE_SWEEP_MODE,
    )
    parser.add_argument("--d-values", type=str, default=None, help="Comma-separated explicit d values.")
    parser.add_argument("--b-values", type=str, default=None, help="Comma-separated b values for d=b*log(N).")
    parser.add_argument("--theorem-window-count", type=int, default=THEOREM_WINDOW_D_COUNT)
    parser.add_argument("--paper-log-window-count", type=int, default=PAPER_LOG_WINDOW_COUNT)
    parser.add_argument("--theorem-zeta", type=float, default=THEOREM_ZETA)
    parser.add_argument("--theorem-xi", type=float, default=THEOREM_XI)
    parser.add_argument("--num-density-graphs", type=int, default=NUM_DENSITY_GRAPHS)
    parser.add_argument("--num-edge-graphs", type=int, default=NUM_EDGE_GRAPHS)
    parser.add_argument("--num-bulk-graphs", type=int, default=NUM_BULK_GRAPHS)
    parser.add_argument("--component-mode", choices=COMPONENT_MODES, default=DEFAULT_COMPONENT_MODE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--skip-mobility-edge-windows", action="store_true")
    parser.add_argument("--skip-mobility-profile", action="store_true")
    parser.add_argument("--edge-levels", type=int, default=None, help="Override the number of retained extreme-edge levels.")
    parser.add_argument("--center-levels", type=int, default=None, help="Override the number of retained center levels.")
    parser.add_argument("--mobility-edge-levels", type=int, default=None, help="Override the number of retained local levels near E=+/-2 and profile energies.")
    parser.add_argument(
        "--mobility-profile-energies",
        type=str,
        default=",".join(str(value) for value in MOBILITY_PROFILE_ENERGIES),
        help="Comma-separated target energies for mobility-profile local eigenpair scans.",
    )
    parser.add_argument("--quick", action="store_true", default=QUICK_MODE)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    global EDGE_LEVELS, CENTER_LEVELS, MOBILITY_EDGE_LEVELS, INCLUDE_MOBILITY_EDGE_WINDOWS, INCLUDE_MOBILITY_PROFILE, THEOREM_ZETA, THEOREM_XI
    args = parse_args(argv)
    THEOREM_ZETA = float(args.theorem_zeta)
    THEOREM_XI = float(args.theorem_xi)
    INCLUDE_MOBILITY_EDGE_WINDOWS = not bool(args.skip_mobility_edge_windows)
    INCLUDE_MOBILITY_PROFILE = not bool(args.skip_mobility_profile)
    args.mobility_profile_energies = _deduplicate_sort_energies(parse_float_list(args.mobility_profile_energies) or MOBILITY_PROFILE_ENERGIES)
    if args.quick:
        args.num_density_graphs = min(args.num_density_graphs, QUICK_NUM_DENSITY_GRAPHS)
        args.num_edge_graphs = min(args.num_edge_graphs, QUICK_NUM_EDGE_GRAPHS)
        args.num_bulk_graphs = min(args.num_bulk_graphs, QUICK_NUM_BULK_GRAPHS)
        EDGE_LEVELS = min(EDGE_LEVELS, QUICK_EDGE_LEVELS)
        CENTER_LEVELS = min(CENTER_LEVELS, QUICK_CENTER_LEVELS)
        MOBILITY_EDGE_LEVELS = min(MOBILITY_EDGE_LEVELS, QUICK_MOBILITY_EDGE_LEVELS)
    if args.edge_levels is not None:
        EDGE_LEVELS = int(args.edge_levels)
    if args.center_levels is not None:
        CENTER_LEVELS = int(args.center_levels)
    if args.mobility_edge_levels is not None:
        MOBILITY_EDGE_LEVELS = int(args.mobility_edge_levels)
    if EDGE_LEVELS < 3 or CENTER_LEVELS < 3 or MOBILITY_EDGE_LEVELS < 3:
        raise ValueError("edge, center, and mobility-edge level counts must each be at least 3")

    explicit_values = parse_float_list(args.d_values) or EXPLICIT_D_VALUES
    b_values = parse_float_list(args.b_values) or B_VALUES
    degree_values = build_active_degree_sequence(
        n=args.n,
        mode=args.degree_sweep_mode,
        explicit_values=explicit_values,
        b_values=b_values,
        theorem_window_count=args.theorem_window_count,
        paper_log_window_count=args.paper_log_window_count,
    )
    degree_window_report = build_degree_window_report(n=args.n, degree_values=degree_values)
    run_dir = args.output_dir / f"fixed_n_{args.n}_{timestamp_slug()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = config_payload(
        args=args,
        degree_values=degree_values,
        degree_window_report=degree_window_report,
        output_dir=run_dir,
    )
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_degree_window_report(degree_window_report, run_dir / "degree_window_report.csv")

    all_summary_rows: list[dict[str, object]] = []
    all_graph_rows: list[dict[str, object]] = []
    all_paper_rows: list[dict[str, object]] = []
    all_seed_rows: list[dict[str, object]] = []

    print(
        "This is only a finite-N check of the written asymptotic inequalities, "
        "not a theorem guarantee and not theorem verification."
    )
    print_degree_window_report(degree_window_report)
    for d in degree_values:
        summary_rows, graph_rows, paper_rows, seed_rows = run_degree(
            n=args.n,
            d=d,
            component_mode=args.component_mode,
            num_density_graphs=args.num_density_graphs,
            num_edge_graphs=args.num_edge_graphs,
            num_bulk_graphs=args.num_bulk_graphs,
            mobility_profile_energies=args.mobility_profile_energies,
        )
        all_summary_rows.extend(summary_rows)
        all_graph_rows.extend(graph_rows)
        all_paper_rows.extend(paper_rows)
        all_seed_rows.extend(seed_rows)

    run_validation_checks(all_summary_rows, degree_values=degree_values, seed_rows=all_seed_rows)
    write_summary_csv(all_summary_rows, run_dir / "degree_sweep_summary.csv")
    write_graph_level_csv(all_graph_rows, run_dir / "edge_graph_level_summary.csv")
    write_paper_scaled_csv(all_paper_rows, run_dir / "paper_scaled_edge_levels.csv")
    write_seed_manifest(all_seed_rows, run_dir / "seed_manifest.csv")
    make_summary_plots(all_summary_rows, run_dir)
    print(f"Saved fixed-N degree sweep outputs to {run_dir}")


if __name__ == "__main__":
    main()
