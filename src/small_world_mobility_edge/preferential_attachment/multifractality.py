"""Finite-size multifractality diagnostics for preferential-attachment spectra.

This module implements the direct-diagonalization protocol of Appendix C in
da Silva et al. (arXiv:2408.13322): generalized inverse participation ratios,
typical moments, two/multi-size estimates of tau(q), and histogram estimates
of the singularity spectrum f(alpha).  Spectral windows are expressed in
rank/quantile coordinates so that they remain comparable across PA sizes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.special import logsumexp
from scipy.stats import spearmanr


PA_MULTIFRACTALITY_VERSION = "pa_multifractality_v1"


def alpha_slug(alpha: float) -> str:
    return ("%.8g" % float(alpha)).replace("-", "m").replace(".", "p")


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError("cannot serialize %r" % (value,))


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.%d" % os.getpid())
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("")
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    temporary = path.with_name(path.name + ".tmp.%d" % os.getpid())
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def configuration_identity(config: Mapping[str, object]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def q_grid(moment_config: Mapping[str, object]) -> np.ndarray:
    lower = float(moment_config["q_min"])
    upper = float(moment_config["q_max"])
    step = float(moment_config["q_step"])
    if lower < 0.0:
        raise ValueError("negative q is disabled because exact/tiny coordinates are unstable")
    if upper < lower or step <= 0.0:
        raise ValueError("invalid q grid")
    count = int(round((upper - lower) / step))
    grid = lower + step * np.arange(count + 1, dtype=float)
    if not np.isclose(grid[-1], upper, atol=1e-10, rtol=0.0):
        raise ValueError("q range must be an integer number of q steps")
    grid[-1] = upper
    return np.round(grid, 12)


def histogram_edges(specification: Mapping[str, object]) -> np.ndarray:
    lower = float(specification["minimum"])
    upper = float(specification["maximum"])
    width = float(specification["width"])
    count = int(round((upper - lower) / width))
    if count <= 0 or not np.isclose(lower + count * width, upper, atol=1e-10):
        raise ValueError("histogram range must be an integer number of widths")
    return np.linspace(lower, upper, count + 1)


def spectral_window_indices(n: int, specification: Mapping[str, object]) -> np.ndarray:
    """Return ascending eigensystem column indices for a spectral window."""
    count = min(int(specification["count"]), int(n))
    if count <= 0:
        raise ValueError("window count must be positive")
    kind = str(specification["kind"])
    if kind == "quantile":
        center = float(specification["center"])
        if not 0.0 <= center <= 1.0:
            raise ValueError("quantile center must lie in [0,1]")
        center_index = int(round(center * (n - 1)))
        start = min(max(center_index - count // 2, 0), n - count)
        indices = np.arange(start, start + count, dtype=int)
    elif kind == "ascending_rank":
        first = int(specification.get("first_rank", 1))
        start = first - 1
        if start < 0 or start + count > n:
            raise ValueError("ascending-rank window lies outside the spectrum")
        indices = np.arange(start, start + count, dtype=int)
    elif kind == "descending_rank":
        first = int(specification.get("first_rank", 1))
        highest_index = n - first
        lowest_index = highest_index - count + 1
        if lowest_index < 0 or highest_index >= n:
            raise ValueError("descending-rank window lies outside the spectrum")
        indices = np.arange(lowest_index, highest_index + 1, dtype=int)
    else:
        raise ValueError("unknown spectral window kind %r" % kind)
    if len(indices) != count or len(np.unique(indices)) != count:
        raise AssertionError("spectral window construction failed")
    return indices


def generalized_log_ipr(masses: np.ndarray, q_values: np.ndarray) -> np.ndarray:
    """Compute log I_q stably for columns of normalized squared amplitudes."""
    masses = np.asarray(masses, dtype=float)
    if masses.ndim == 1:
        masses = masses[:, None]
    if masses.ndim != 2 or np.any(masses < 0.0):
        raise ValueError("masses must be a nonnegative one- or two-dimensional array")
    normalizers = np.sum(masses, axis=0)
    if np.any(normalizers <= 0.0):
        raise ValueError("every mass vector must have positive total mass")
    masses = masses / normalizers[None, :]
    log_masses = np.full(masses.shape, -np.inf, dtype=float)
    positive = masses > 0.0
    log_masses[positive] = np.log(masses[positive])
    result = np.empty((len(q_values), masses.shape[1]), dtype=float)
    for row, q in enumerate(np.asarray(q_values, dtype=float)):
        if np.isclose(q, 0.0, atol=1e-14):
            result[row, :] = math.log(masses.shape[0])
        elif np.isclose(q, 1.0, atol=1e-14):
            result[row, :] = 0.0
        else:
            result[row, :] = logsumexp(q * log_masses, axis=0)
    return result


def shannon_entropy(masses: np.ndarray) -> np.ndarray:
    masses = np.asarray(masses, dtype=float)
    log_masses = np.zeros_like(masses)
    positive = masses > 0.0
    log_masses[positive] = np.log(masses[positive])
    return -np.sum(masses * log_masses, axis=0)


def multiplicity_cluster_sizes(values: np.ndarray, tolerance: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n = len(values)
    result = np.ones(n, dtype=int)
    if n < 2:
        return result
    start = 0
    for index in range(n - 1):
        if values[index + 1] - values[index] > tolerance:
            result[start : index + 1] = index + 1 - start
            start = index + 1
    result[start:n] = n - start
    return result


def local_gap_diagnostic(
    values: np.ndarray, index: int, tolerance: float, radius: int
) -> dict[str, float]:
    left = float(values[index] - values[index - 1]) if index > 0 else math.nan
    right = float(values[index + 1] - values[index]) if index + 1 < len(values) else math.nan
    finite = [gap for gap in (left, right) if math.isfinite(gap)]
    minimum = min(finite) if finite else math.nan
    lo = max(0, index - int(radius))
    hi = min(len(values), index + int(radius) + 1)
    nearby = np.diff(values[lo:hi])
    positive = nearby[nearby > tolerance]
    local_scale = float(np.median(positive)) if len(positive) else math.nan
    normalized = minimum / local_scale if local_scale > 0.0 else math.nan
    return {
        "left_gap": left,
        "right_gap": right,
        "minimum_neighbor_gap": minimum,
        "local_positive_gap_median": local_scale,
        "minimum_gap_over_local_median": normalized,
    }


def _edge_data(graph_cache: Path, n: int) -> tuple[np.ndarray, set[tuple[int, int]]]:
    with np.load(graph_cache) as archive:
        degrees = np.asarray(archive["degrees"], dtype=int)
        edge_u = np.asarray(archive["edge_u"], dtype=int)
        edge_v = np.asarray(archive["edge_v"], dtype=int)
    if degrees.shape != (n,):
        raise ValueError("graph degree cache has incompatible size")
    edges = {
        (min(int(u), int(v)), max(int(u), int(v)))
        for u, v in zip(edge_u.tolist(), edge_v.tolist())
    }
    return degrees, edges


def _top_r_diagnostic(mass: np.ndarray, top_r: int) -> tuple[float, float, np.ndarray]:
    count = min(int(top_r), len(mass))
    selected = np.argpartition(mass, len(mass) - count)[-count:]
    selected = selected[np.argsort(mass[selected])[::-1]]
    total = float(np.sum(mass[selected]))
    if count <= 1 or total <= 0.0:
        evenness = 1.0
    else:
        shares = mass[selected] / total
        evenness = float(-np.sum(shares * np.log(shares)) / math.log(count))
    return total, evenness, selected


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or np.all(left == left[0]) or np.all(right == right[0]):
        return math.nan
    return float(spearmanr(left, right).correlation)


def _case_prefix(n: int, alpha: float, window_id: str) -> dict[str, object]:
    return {
        "n": int(n),
        "degree_ccdf_alpha": float(alpha),
        "window_id": str(window_id),
    }


def analyze_case(
    *,
    source_case_dir: Path,
    output_case_dir: Path,
    n: int,
    alpha: float,
    config: Mapping[str, object],
) -> dict[str, object]:
    source_case_dir = Path(source_case_dir)
    output_case_dir = Path(output_case_dir)
    values_path = source_case_dir / "arrays/full_eigenvalues_ascending.npy"
    vectors_path = source_case_dir / "arrays/full_eigenvectors_ascending.npy"
    graph_path = source_case_dir / "arrays/graph_edges.npz"
    source_manifest_path = source_case_dir / "spectral_cache_manifest.json"
    for required in (values_path, vectors_path, graph_path, source_manifest_path):
        if not required.exists():
            raise FileNotFoundError(required)

    values = np.load(values_path, mmap_mode="r")
    vectors = np.load(vectors_path, mmap_mode="r")
    if values.shape != (n,) or vectors.shape != (n, n):
        raise ValueError("cached eigensystem shape does not match configured N")
    if not np.all(np.isfinite(values)) or np.any(np.diff(values) < 0.0):
        raise ValueError("cached eigenvalues must be finite and ascending")
    degrees, edge_set = _edge_data(graph_path, n)
    source_manifest = json.loads(source_manifest_path.read_text())

    q_values = q_grid(config["moments"])
    alpha_edges = histogram_edges(config["alpha_histogram"])
    logx_edges = histogram_edges(config["log10_scaled_mass_histogram"])
    tolerance_factor = float(config["selection"]["multiplicity_tolerance_factor"])
    tolerance = tolerance_factor * np.finfo(float).eps * max(1, n) * max(
        1.0, float(np.max(np.abs(values)))
    )
    cluster_sizes = multiplicity_cluster_sizes(values, tolerance)
    top_r = int(config["selection"]["top_r"])
    gap_radius = int(config["selection"]["local_gap_radius"])
    high_degree_count = max(1, int(math.ceil(n * float(config["selection"]["high_degree_fraction"]))))
    high_degree_vertices = np.argsort(degrees)[-high_degree_count:]

    moment_rows: list[dict[str, object]] = []
    singularity_rows: list[dict[str, object]] = []
    amplitude_rows: list[dict[str, object]] = []
    eigenvector_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []

    for window in config["windows"]:
        window_id = str(window["id"])
        indices = spectral_window_indices(n, window)
        raw_vectors = np.asarray(vectors[:, indices], dtype=float)
        if not np.all(np.isfinite(raw_vectors)) or np.any(np.linalg.norm(raw_vectors, axis=0) == 0.0):
            raise ValueError("selected cached eigenvectors must be finite and nonzero")
        masses = np.square(raw_vectors)
        masses /= np.sum(masses, axis=0, keepdims=True)
        logs = generalized_log_ipr(masses, q_values)
        entropy = shannon_entropy(masses)
        ipr = np.sum(np.square(masses), axis=0)
        gap_normalized = []

        for q_index, q in enumerate(q_values):
            sample = logs[q_index]
            prefix = _case_prefix(n, alpha, window_id)
            moment_rows.append({
                **prefix,
                "q": float(q),
                "eigenvector_count": len(indices),
                "sum_log_iq": float(np.sum(sample)),
                "sum_squared_log_iq": float(np.sum(np.square(sample))),
                "mean_log_iq": float(np.mean(sample)),
                "sd_log_iq": float(np.std(sample, ddof=1)) if len(sample) > 1 else math.nan,
                "typical_iq": float(np.exp(np.mean(sample))),
                "arithmetic_mean_iq": float(np.mean(np.exp(sample))),
            })

        for column, eigen_index in enumerate(indices.tolist()):
            mass = masses[:, column]
            gap = local_gap_diagnostic(values, eigen_index, tolerance, gap_radius)
            gap_normalized.append(gap["minimum_gap_over_local_median"])
            top_mass, top_evenness, top_vertices = _top_r_diagnostic(mass, top_r)
            leading = top_vertices[:2].tolist()
            top_pair_adjacent = (
                len(leading) == 2
                and (min(leading), max(leading)) in edge_set
            )
            prefix = _case_prefix(n, alpha, window_id)
            eigenvector_rows.append({
                **prefix,
                "ascending_index_zero_based": int(eigen_index),
                "ascending_rank": int(eigen_index + 1),
                "decreasing_rank": int(n - eigen_index),
                "empirical_spectral_quantile": float(eigen_index / max(1, n - 1)),
                "eigenvalue": float(values[eigen_index]),
                "ipr": float(ipr[column]),
                "n_times_ipr": float(n * ipr[column]),
                "shannon_entropy": float(entropy[column]),
                "entropy_dimension_single_size": float(entropy[column] / math.log(n)),
                "effective_support_exp_entropy": float(np.exp(entropy[column])),
                "maximum_coordinate_mass": float(np.max(mass)),
                "top_r": top_r,
                "top_r_mass": top_mass,
                "top_r_evenness": top_evenness,
                "mass_on_highest_degree_fraction": float(np.sum(mass[high_degree_vertices])),
                "mass_weighted_log1p_degree": float(np.sum(mass * np.log1p(degrees))),
                "degree_mass_spearman": _safe_spearman(degrees, mass),
                "largest_mass_vertex": int(leading[0]),
                "largest_mass_vertex_degree": int(degrees[leading[0]]),
                "second_mass_vertex": int(leading[1]) if len(leading) > 1 else -1,
                "second_mass_vertex_degree": int(degrees[leading[1]]) if len(leading) > 1 else -1,
                "top_two_mass_vertices_adjacent": bool(top_pair_adjacent),
                "multiplicity_cluster_size": int(cluster_sizes[eigen_index]),
                "basis_invariant_individual_eigenvector": bool(cluster_sizes[eigen_index] == 1),
                **gap,
            })

        flat_mass = masses.ravel(order="F")
        alpha_values = np.full(flat_mass.shape, np.inf, dtype=float)
        positive_mass = flat_mass > 0.0
        alpha_values[positive_mass] = -np.log(flat_mass[positive_mass]) / math.log(n)
        alpha_counts, _ = np.histogram(alpha_values[np.isfinite(alpha_values)], bins=alpha_edges)
        alpha_widths = np.diff(alpha_edges)
        alpha_density = alpha_counts / (len(flat_mass) * alpha_widths)
        for bin_index, count in enumerate(alpha_counts.tolist()):
            density = float(alpha_density[bin_index])
            singularity_rows.append({
                **_case_prefix(n, alpha, window_id),
                "alpha_left": float(alpha_edges[bin_index]),
                "alpha_right": float(alpha_edges[bin_index + 1]),
                "alpha_center": float((alpha_edges[bin_index] + alpha_edges[bin_index + 1]) / 2.0),
                "count": int(count),
                "coordinate_sample_count": int(len(flat_mass)),
                "omega_probability_density": density,
                "f_alpha_single_size": float(1.0 + math.log(density) / math.log(n)) if density > 0.0 else math.nan,
            })

        scaled_mass = n * flat_mass
        positive_x = scaled_mass[scaled_mass > 0.0]
        logx = np.log10(positive_x)
        logx_counts, _ = np.histogram(logx, bins=logx_edges)
        for bin_index, count in enumerate(logx_counts.tolist()):
            width = float(logx_edges[bin_index + 1] - logx_edges[bin_index])
            amplitude_rows.append({
                **_case_prefix(n, alpha, window_id),
                "log10_x_left": float(logx_edges[bin_index]),
                "log10_x_right": float(logx_edges[bin_index + 1]),
                "log10_x_center": float((logx_edges[bin_index] + logx_edges[bin_index + 1]) / 2.0),
                "count": int(count),
                "coordinate_sample_count": int(len(flat_mass)),
                "probability_density_per_log10_x": float(count / (len(flat_mass) * width)),
            })

        selected_rows = eigenvector_rows[-len(indices) :]
        finite_gap = np.asarray([value for value in gap_normalized if math.isfinite(value)], dtype=float)
        nipr = np.asarray([float(row["n_times_ipr"]) for row in selected_rows])
        raw_min_gap = np.asarray([float(row["minimum_neighbor_gap"]) for row in selected_rows])
        valid_corr = np.isfinite(raw_min_gap) & (raw_min_gap > 0.0)
        window_rows.append({
            **_case_prefix(n, alpha, window_id),
            "window_kind": str(window["kind"]),
            "requested_count": int(window["count"]),
            "eigenvector_count": len(indices),
            "ascending_index_min": int(indices[0]),
            "ascending_index_max": int(indices[-1]),
            "energy_min": float(values[indices[0]]),
            "energy_max": float(values[indices[-1]]),
            "energy_mean": float(np.mean(values[indices])),
            "spectral_quantile_mean": float(np.mean(indices) / max(1, n - 1)),
            "mean_n_times_ipr": float(np.mean(nipr)),
            "median_n_times_ipr": float(np.median(nipr)),
            "geometric_mean_n_times_ipr": float(np.exp(np.mean(np.log(nipr)))),
            "mean_shannon_entropy": float(np.mean(entropy)),
            "mean_entropy_dimension_single_size": float(np.mean(entropy) / math.log(n)),
            "mean_effective_support": float(np.mean(np.exp(entropy))),
            "mean_top_r_mass": float(np.mean([float(row["top_r_mass"]) for row in selected_rows])),
            "mean_top_r_evenness": float(np.mean([float(row["top_r_evenness"]) for row in selected_rows])),
            "mean_degree_mass_spearman": float(np.nanmean([float(row["degree_mass_spearman"]) for row in selected_rows])),
            "top_two_mass_vertices_adjacent_fraction": float(np.mean([bool(row["top_two_mass_vertices_adjacent"]) for row in selected_rows])),
            "basis_ambiguous_eigenvector_count": int(np.sum(cluster_sizes[indices] > 1)),
            "median_minimum_gap_over_local_median": float(np.median(finite_gap)) if len(finite_gap) else math.nan,
            "spearman_log_gap_vs_log_nipr": _safe_spearman(
                np.log(raw_min_gap[valid_corr]), np.log(nipr[valid_corr])
            ),
            "alpha_histogram_underflow_fraction": float(np.mean(alpha_values < alpha_edges[0])),
            "alpha_histogram_overflow_fraction": float(np.mean(alpha_values >= alpha_edges[-1])),
            "scaled_mass_histogram_underflow_fraction": float(np.mean(np.log10(np.maximum(scaled_mass, np.finfo(float).tiny)) < logx_edges[0])),
            "scaled_mass_histogram_overflow_fraction": float(np.mean(np.log10(np.maximum(scaled_mass, np.finfo(float).tiny)) >= logx_edges[-1])),
        })

    tables = output_case_dir / "tables"
    write_csv(tables / "typical_moments.csv", moment_rows)
    write_csv(tables / "singularity_spectrum_single_size.csv", singularity_rows)
    write_csv(tables / "scaled_mass_histogram.csv", amplitude_rows)
    write_csv(tables / "eigenvector_diagnostics.csv", eigenvector_rows)
    write_csv(tables / "spectral_window_summary.csv", window_rows)
    manifest = {
        "status": "COMPLETE",
        "analysis_version": PA_MULTIFRACTALITY_VERSION,
        "analysis_id": config["analysis_id"],
        "configuration_sha256": configuration_identity(config),
        "finished_at_utc": utc_now(),
        "case": {"n": n, "degree_ccdf_alpha": alpha},
        "source_case_dir": str(source_case_dir),
        "source_spectral_cache_identity": source_manifest.get("spectral_cache_identity"),
        "source_values_shape": list(values.shape),
        "source_vectors_shape": list(vectors.shape),
        "method": {
            "iq": "I_q=sum_i |psi_i|^(2q)",
            "typical_iq": "exp(mean over selected eigenvectors of log I_q)",
            "alpha_i": "-log(|psi_i|^2)/log(N)",
            "single_size_f_alpha": "1+log(histogram probability density)/log(N)",
            "q_step": float(config["moments"]["q_step"]),
            "band_size": int(config["selection"]["band_size"]),
        },
        "caveats": [
            "Individual eigenvectors in an exactly degenerate eigenspace are basis-dependent; such rows are flagged.",
            "Overlapping spectral windows are separate views and must not be pooled as independent samples.",
            "Single-size entropy dimensions and f(alpha) have finite-size bias; use aggregate scaling estimates for inference.",
        ],
        "output_row_counts": {
            "typical_moments": len(moment_rows),
            "singularity_spectrum_single_size": len(singularity_rows),
            "scaled_mass_histogram": len(amplitude_rows),
            "eigenvector_diagnostics": len(eigenvector_rows),
            "spectral_window_summary": len(window_rows),
        },
    }
    write_json(output_case_dir / "case_manifest.json", manifest)
    return manifest


def _linear_slope(x_values: Sequence[float], y_values: Sequence[float]) -> tuple[float, float]:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    if len(x) < 2 or len(np.unique(x)) < 2:
        return math.nan, math.nan
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residual = float(np.sum(np.square(y - fitted)))
    total = float(np.sum(np.square(y - np.mean(y))))
    r_squared = 1.0 - residual / total if total > 0.0 else 1.0
    return float(slope), float(r_squared)


def estimate_tau(size_to_mean_log_iq: Mapping[int, float]) -> tuple[float, float]:
    sizes = sorted(int(size) for size in size_to_mean_log_iq)
    slope, r_squared = _linear_slope(
        [math.log(size) for size in sizes],
        [float(size_to_mean_log_iq[size]) for size in sizes],
    )
    return -slope, r_squared


def _numeric(row: Mapping[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan


def _discover_complete_cases(output_root: Path) -> list[Path]:
    cases = []
    for manifest_path in sorted(Path(output_root).glob("N_*/alpha_*/case_manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") == "COMPLETE":
            cases.append(manifest_path.parent)
    return cases


def aggregate_results(
    *, output_root: Path, report_dir: Path, config: Mapping[str, object]
) -> dict[str, object]:
    output_root = Path(output_root)
    report_dir = Path(report_dir)
    case_dirs = _discover_complete_cases(output_root)
    expected = len(config["sizes"]) * len(config["alpha_grid"])
    if len(case_dirs) != expected:
        raise ValueError("found %d complete cases, expected %d" % (len(case_dirs), expected))
    expected_keys = {(int(n), float(alpha)) for n in config["sizes"] for alpha in config["alpha_grid"]}
    seen = set()
    for case_dir in case_dirs:
        manifest = json.loads((case_dir / "case_manifest.json").read_text())
        if manifest.get("configuration_sha256") != configuration_identity(config):
            raise ValueError("case configuration identity differs from aggregation configuration")
        key = (int(manifest["case"]["n"]), float(manifest["case"]["degree_ccdf_alpha"]))
        if key in seen or key not in expected_keys:
            raise ValueError("duplicate or unexpected (N, alpha) case")
        seen.add(key)
    if seen != expected_keys:
        raise ValueError("aggregation is missing configured (N, alpha) cases")

    all_moments = []
    all_singularities = []
    all_windows = []
    for case_dir in case_dirs:
        all_moments.extend(read_csv(case_dir / "tables/typical_moments.csv"))
        all_singularities.extend(read_csv(case_dir / "tables/singularity_spectrum_single_size.csv"))
        all_windows.extend(read_csv(case_dir / "tables/spectral_window_summary.csv"))

    moment_groups: dict[tuple[float, str, float], list[dict[str, str]]] = defaultdict(list)
    for row in all_moments:
        moment_groups[(_numeric(row, "degree_ccdf_alpha"), row["window_id"], _numeric(row, "q"))].append(row)
    window_lookup = {
        (_numeric(row, "degree_ccdf_alpha"), row["window_id"], int(_numeric(row, "n"))): row
        for row in all_windows
    }

    tau_rows: list[dict[str, object]] = []
    for (alpha, window_id, q), rows in sorted(moment_groups.items()):
        size_to_samples: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            size_to_samples[int(_numeric(row, "n"))].append(row)
        size_to_mean = {}
        graph_counts = {}
        vector_counts = {}
        for n, sample_rows in size_to_samples.items():
            weights = np.asarray([_numeric(row, "eigenvector_count") for row in sample_rows])
            means = np.asarray([_numeric(row, "mean_log_iq") for row in sample_rows])
            size_to_mean[n] = float(np.average(means, weights=weights))
            graph_counts[n] = len(sample_rows)
            vector_counts[n] = int(np.sum(weights))
        tau, r_squared = estimate_tau(size_to_mean)
        entropy_by_size = {
            n: _numeric(window_lookup[(alpha, window_id, n)], "mean_shannon_entropy")
            for n in size_to_mean
        }
        d1, d1_r_squared = _linear_slope(
            [math.log(n) for n in sorted(entropy_by_size)],
            [entropy_by_size[n] for n in sorted(entropy_by_size)],
        )
        d_q = d1 if np.isclose(q, 1.0, atol=1e-10) else tau / (q - 1.0)
        tau_rows.append({
            "degree_ccdf_alpha": alpha,
            "window_id": window_id,
            "q": q,
            "tau_q": tau,
            "generalized_dimension_dq": d_q,
            "ergodic_tau_q_minus_1": q - 1.0,
            "tau_minus_ergodic": tau - (q - 1.0),
            "entropy_dimension_d1": d1,
            "size_count": len(size_to_mean),
            "sizes": ";".join(str(n) for n in sorted(size_to_mean)),
            "graphs_per_size": ";".join("%d:%d" % (n, graph_counts[n]) for n in sorted(graph_counts)),
            "eigenvectors_per_size": ";".join("%d:%d" % (n, vector_counts[n]) for n in sorted(vector_counts)),
            "fit_method": "two_size_secant" if len(size_to_mean) == 2 else "ordinary_least_squares",
            "tau_fit_r_squared": r_squared,
            "d1_fit_r_squared": d1_r_squared,
            "ensemble_uncertainty_available": bool(min(graph_counts.values()) >= 2),
        })

    singularity_groups: dict[tuple[float, str, float], list[dict[str, str]]] = defaultdict(list)
    for row in all_singularities:
        singularity_groups[
            (_numeric(row, "degree_ccdf_alpha"), row["window_id"], _numeric(row, "alpha_center"))
        ].append(row)
    f_rows: list[dict[str, object]] = []
    for (alpha, window_id, center), rows in sorted(singularity_groups.items()):
        positive = [row for row in rows if _numeric(row, "omega_probability_density") > 0.0]
        sizes = sorted({int(_numeric(row, "n")) for row in positive})
        if len(sizes) < 2:
            continue
        size_to_log_density = {}
        for n in sizes:
            densities = [
                _numeric(row, "omega_probability_density")
                for row in positive
                if int(_numeric(row, "n")) == n
            ]
            size_to_log_density[n] = float(np.mean(np.log(densities)))
        slope, r_squared = _linear_slope(
            [math.log(n) for n in sizes], [size_to_log_density[n] for n in sizes]
        )
        f_rows.append({
            "degree_ccdf_alpha": alpha,
            "window_id": window_id,
            "alpha_center": center,
            "f_alpha_scaling": 1.0 + slope,
            "size_count": len(sizes),
            "sizes": ";".join(str(n) for n in sizes),
            "fit_method": "two_size_secant" if len(sizes) == 2 else "ordinary_least_squares",
            "fit_r_squared": r_squared,
        })

    report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(report_dir / "tau_and_generalized_dimensions.csv", tau_rows)
    write_csv(report_dir / "singularity_spectrum_scaling.csv", f_rows)
    write_csv(report_dir / "all_case_window_summaries.csv", all_windows)
    write_csv(report_dir / "all_case_single_size_singularity_spectra.csv", all_singularities)
    _plot_aggregate(tau_rows, f_rows, report_dir, config)
    _write_method_report(tau_rows, report_dir, config, len(case_dirs))
    graph_counts = [int(item.split(":")[1]) for row in tau_rows for item in str(row["graphs_per_size"]).split(";")]
    manifest = {
        "status": "COMPLETE",
        "analysis_version": PA_MULTIFRACTALITY_VERSION,
        "analysis_id": config["analysis_id"],
        "configuration_sha256": configuration_identity(config),
        "finished_at_utc": utc_now(),
        "complete_case_count": len(case_dirs),
        "expected_case_count": expected,
        "tau_row_count": len(tau_rows),
        "f_alpha_scaling_row_count": len(f_rows),
        "ensemble_uncertainty_available": bool(graph_counts and min(graph_counts) >= 2),
        "inference_status": (
            "FINITE_SIZE_EXPLORATORY_SINGLE_GRAPH_PER_SIZE"
            if graph_counts and min(graph_counts) < 2
            else "FINITE_SIZE_ENSEMBLE"
        ),
        "paper_protocol": {
            "band_size": int(config["selection"]["band_size"]),
            "q_step": float(config["moments"]["q_step"]),
            "typical_moment": "exp(mean(log(I_q)))",
            "tau": "negative slope of mean(log(I_q)) against log(N)",
            "f_alpha": "single-size histogram estimator plus cross-size histogram-density slope",
        },
    }
    write_json(report_dir / "aggregate_manifest.json", manifest)
    return manifest


def _plot_aggregate(
    tau_rows: Sequence[Mapping[str, object]],
    f_rows: Sequence[Mapping[str, object]],
    report_dir: Path,
    config: Mapping[str, object],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = Path(report_dir) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    window_order = [str(window["id"]) for window in config["windows"]]
    for alpha in [float(value) for value in config["alpha_grid"]]:
        figure, axes = plt.subplots(1, 3, figsize=(17, 5))
        for window_id in window_order:
            selected = [
                row for row in tau_rows
                if math.isclose(float(row["degree_ccdf_alpha"]), alpha) and row["window_id"] == window_id
            ]
            if selected:
                selected.sort(key=lambda row: float(row["q"]))
                axes[0].plot([row["q"] for row in selected], [row["tau_q"] for row in selected], label=window_id)
                axes[1].plot([row["q"] for row in selected], [row["generalized_dimension_dq"] for row in selected], label=window_id)
            spectrum = [
                row for row in f_rows
                if math.isclose(float(row["degree_ccdf_alpha"]), alpha) and row["window_id"] == window_id
            ]
            if spectrum:
                spectrum.sort(key=lambda row: float(row["alpha_center"]))
                axes[2].plot([row["alpha_center"] for row in spectrum], [row["f_alpha_scaling"] for row in spectrum], label=window_id)
        q_values = q_grid(config["moments"])
        axes[0].plot(q_values, q_values - 1.0, "k--", linewidth=1, label="ergodic q-1")
        axes[0].set(xlabel="q", ylabel="tau(q)", title="Typical-moment scaling")
        axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
        axes[1].set(xlabel="q", ylabel="D_q", title="Generalized dimensions", ylim=(-0.15, 1.25))
        axes[2].axhline(1.0, color="black", linestyle="--", linewidth=1)
        axes[2].set(xlabel="alpha", ylabel="f(alpha)", title="Histogram scaling")
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncol=4, fontsize=7)
        figure.suptitle("Affine PA adjacency multifractality: degree CCDF alpha=%.3g" % alpha)
        figure.tight_layout(rect=(0, 0.13, 1, 0.94))
        figure.savefig(figure_dir / ("multifractality_alpha_%s.png" % alpha_slug(alpha)), dpi=180)
        plt.close(figure)


def _write_method_report(
    tau_rows: Sequence[Mapping[str, object]],
    report_dir: Path,
    config: Mapping[str, object],
    case_count: int,
) -> None:
    q2 = [row for row in tau_rows if math.isclose(float(row["q"]), 2.0, abs_tol=1e-10)]
    lines = [
        "# Preferential-attachment multifractality analysis",
        "",
        "Status: complete computational pass over %d cached eigensystems." % case_count,
        "",
        "## Method",
        "",
        "For each configured eigenvector band, the analysis computes "
        "`I_q=sum_i |psi_i|^(2q)` on the configured q grid and averages "
        "`log(I_q)` across eigenvectors. The finite-size exponent is the negative "
        "slope of this typical log moment against `log(N)`. It also computes "
        "`alpha_i=-log(|psi_i|^2)/log(N)` and both the paper's single-size "
        "histogram estimator and a cross-size histogram-density slope for `f(alpha)`. ",
        "",
        "Quantile windows have fixed centers but fixed state counts, so their "
        "relative widths change with N. Edge windows have fixed ranks and exclude "
        "the Perron vector on the positive side. These rules can compare different "
        "spectral depths; negative two-size Dq estimates are not physical negative dimensions.",
        "",
        "## Inference guardrail",
        "",
        "The reused production cache has one graph at each of N=%s. The curves "
        "are therefore finite-size exploratory estimates, not ensemble error-controlled "
        "evidence of a multifractal phase. Eigenvectors in numerical multiplicity "
        "clusters are flagged because individual-basis I_q is not invariant." % ", ".join(str(n) for n in config["sizes"]),
        "",
        "## D2 snapshot",
        "",
        "| degree CCDF alpha | window | tau(2)=D2 | D1 |",
        "|---:|---|---:|---:|",
    ]
    for row in sorted(q2, key=lambda item: (float(item["degree_ccdf_alpha"]), str(item["window_id"]))):
        lines.append(
            "| %.3g | %s | %.6g | %.6g |"
            % (
                float(row["degree_ccdf_alpha"]),
                row["window_id"],
                float(row["tau_q"]),
                float(row["entropy_dimension_d1"]),
            )
        )
    (Path(report_dir) / "METHOD_AND_RESULTS.md").write_text("\n".join(lines) + "\n")


def controlled_validation() -> tuple[list[dict[str, object]], bool]:
    q_values = np.asarray([0.0, 0.5, 1.0, 2.0, 3.0])
    sizes = [100, 1000]
    uniform = {}
    localized = {}
    for n in sizes:
        uniform[n] = generalized_log_ipr(np.full(n, 1.0 / n), q_values)[:, 0]
        point_mass = np.zeros(n)
        point_mass[0] = 1.0
        localized[n] = generalized_log_ipr(point_mass, q_values)[:, 0]
    rows = []
    passed = True
    for q_index, q in enumerate(q_values):
        uniform_tau, _ = estimate_tau({n: uniform[n][q_index] for n in sizes})
        localized_tau, _ = estimate_tau({n: localized[n][q_index] for n in sizes})
        uniform_pass = math.isclose(uniform_tau, q - 1.0, abs_tol=1e-12)
        localized_expected = -1.0 if np.isclose(q, 0.0) else 0.0
        localized_pass = math.isclose(localized_tau, localized_expected, abs_tol=1e-12)
        rows.append({
            "fixture": "uniform_and_point_mass",
            "q": q,
            "uniform_tau": uniform_tau,
            "uniform_expected": q - 1.0,
            "localized_tau": localized_tau,
            "localized_expected": localized_expected,
            "passed": bool(uniform_pass and localized_pass),
        })
        passed = bool(passed and uniform_pass and localized_pass)
    return rows, passed
