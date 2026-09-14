from __future__ import annotations

import csv
import gzip
import json
import math
import os
import tempfile
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "small-world-mobility-edge-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import networkx as nx
import numpy as np
from scipy.stats import kstest

from . import chung_lu_eigenvector_percentile_scan as eigenvector_scan
from .chung_lu_baseline import generate_chung_lu_graph_from_weights
from .regular_baseline import adjacency_matrix


N = eigenvector_scan.N
NUM_GRAPHS = eigenvector_scan.NUM_GRAPHS
TARGET_MEAN_DEGREE = eigenvector_scan.TARGET_MEAN_DEGREE
Q_MIN = 0.60
Q_MAX = 0.72
Q_WIDTH = eigenvector_scan.Q_WIDTH
LOCAL_QUANTILE_BINS = eigenvector_scan.quantile_bins(Q_MIN, Q_MAX, Q_WIDTH)
REMOVE_PERRON_EIGENPAIR = True
REMOVE_EXACT_ZERO_EIGENVALUES = True
ZERO_EIGENVALUE_ATOL = eigenvector_scan.ZERO_EIGENVALUE_ATOL
DEGENERATE_GAP_ATOL = 1e-10
ATOM_ENERGY = 1.0
ATOM_ENERGY_ATOL = 1e-8
SMALL_SPACING_THRESHOLD = 0.20
MIN_POOLED_SPACINGS = 200
MIN_POOLED_RATIOS = 100
OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "chung_lu"
    / "power_law"
    / "eigenvalues"
    / "resonance_zoom_q0p60_0p72_width_0p002"
)


def parameter_output_dir(displayed_tau: float, n: int = N) -> Path:
    return OUTPUT_DIR / f"n_{n}_tau_{eigenvector_scan.tau_slug(displayed_tau)}_avgdeg_20p0"


def encode_array(values: np.ndarray) -> str:
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return ""
    return " ".join(f"{value:.12g}" for value in array)


def decode_array(text: object) -> np.ndarray:
    value = "" if text is None else str(text).strip()
    if not value:
        return np.array([], dtype=float)
    return np.fromstring(value, sep=" ", dtype=float)


def gap_diagnostics(eigenvalues: np.ndarray) -> dict[str, np.ndarray | float | int]:
    values = np.sort(np.asarray(eigenvalues, dtype=float))
    raw_gaps = np.diff(values)
    near_degenerate = np.abs(raw_gaps) <= DEGENERATE_GAP_ATOL
    regular = raw_gaps > DEGENERATE_GAP_ATOL
    positive_gaps = raw_gaps[regular]
    if len(positive_gaps):
        normalized_positive = positive_gaps / float(np.mean(positive_gaps))
    else:
        normalized_positive = np.array([], dtype=float)

    inclusive_ratios: list[float] = []
    regular_ratios: list[float] = []
    for left, right in zip(raw_gaps[:-1], raw_gaps[1:]):
        denominator = max(float(left), float(right))
        if denominator > DEGENERATE_GAP_ATOL:
            inclusive_ratios.append(min(float(left), float(right)) / denominator)
        if left > DEGENERATE_GAP_ATOL and right > DEGENERATE_GAP_ATOL:
            regular_ratios.append(min(float(left), float(right)) / max(float(left), float(right)))

    return {
        "raw_gaps": raw_gaps,
        "normalized_positive_spacings": normalized_positive,
        "inclusive_gap_ratios": np.asarray(inclusive_ratios, dtype=float),
        "regular_gap_ratios": np.asarray(regular_ratios, dtype=float),
        "number_of_raw_gaps": int(len(raw_gaps)),
        "number_of_degenerate_gaps": int(np.sum(near_degenerate)),
        "degenerate_gap_fraction": float(np.mean(near_degenerate)) if len(raw_gaps) else math.nan,
        "mean_positive_raw_gap": float(np.mean(positive_gaps)) if len(positive_gaps) else math.nan,
    }


def filter_continuous_nonperron_eigenvalues(eigenvalues: np.ndarray) -> tuple[np.ndarray, int, float]:
    values = np.sort(np.asarray(eigenvalues, dtype=float))
    if REMOVE_PERRON_EIGENPAIR:
        if len(values) < 2:
            raise ValueError("need at least two eigenvalues to remove the Perron eigenvalue")
        perron = float(values[-1])
        values = values[:-1]
    else:
        perron = math.nan
    zero_mask = np.isclose(values, 0.0, atol=ZERO_EIGENVALUE_ATOL, rtol=0.0)
    zero_count = int(np.sum(zero_mask))
    if REMOVE_EXACT_ZERO_EIGENVALUES:
        values = values[~zero_mask]
    return values, zero_count, perron


def graph_scan_rows(
    graph_index: int,
    graph_seed: int,
    sequence_seed: int,
    displayed_tau: float,
    n: int = N,
    local_quantile_bins: list[tuple[float, float]] | None = None,
) -> list[dict[str, object]]:
    if local_quantile_bins is None:
        local_quantile_bins = LOCAL_QUANTILE_BINS
    specification = eigenvector_scan.tau_specification(displayed_tau)
    weights, diagnostics = eigenvector_scan.generate_legacy_expected_weights(
        displayed_tau=displayed_tau,
        sequence_seed=sequence_seed,
        n=n,
    )
    if not diagnostics.admissibility_holds:
        raise ValueError("expected-weight sequence violates Chung-Lu admissibility")
    graph = generate_chung_lu_graph_from_weights(weights, seed=graph_seed)
    adjacency = adjacency_matrix(graph)
    eigenvalues = np.linalg.eigvalsh(adjacency)
    continuous_values, zero_count, perron = filter_continuous_nonperron_eigenvalues(eigenvalues)
    if len(continuous_values) == 0:
        raise ValueError("no continuous non-Perron eigenvalues remain")

    full_atom_count = int(
        np.sum(np.isclose(continuous_values, ATOM_ENERGY, atol=ATOM_ENERGY_ATOL, rtol=0.0))
    )
    rows: list[dict[str, object]] = []
    for q_low, q_high in local_quantile_bins:
        start, stop = eigenvector_scan.quantile_index_bin(len(continuous_values), q_low, q_high)
        selected = continuous_values[start:stop]
        gaps = gap_diagnostics(selected)
        atom_count = int(np.sum(np.isclose(selected, ATOM_ENERGY, atol=ATOM_ENERGY_ATOL, rtol=0.0)))
        raw_gaps = np.asarray(gaps["raw_gaps"], dtype=float)
        rows.append(
            {
                "graph_index": graph_index,
                "graph_seed": graph_seed,
                "sequence_seed": sequence_seed,
                "displayed_tau": displayed_tau,
                "tau_convention": specification.tau_convention,
                "effective_density_exponent": specification.effective_density_exponent,
                "quantile": 0.5 * (q_low + q_high),
                "quantile_low": q_low,
                "quantile_high": q_high,
                "index_start": start,
                "index_stop": stop,
                "number_of_eigenvalues": int(len(selected)),
                "E_low": float(selected[0]),
                "E_high": float(selected[-1]),
                "E_center": float(np.median(selected)),
                "E_span": float(selected[-1] - selected[0]),
                "continuous_bulk_dimension": int(len(continuous_values)),
                "zero_eigenvalue_count": zero_count,
                "zero_eigenvalue_fraction": zero_count / max(len(eigenvalues) - 1, 1),
                "perron_eigenvalue": perron,
                "full_spectrum_atom_count_at_1": full_atom_count,
                "selected_atom_count_at_1": atom_count,
                "selected_atom_fraction_at_1": atom_count / len(selected),
                "number_of_raw_gaps": gaps["number_of_raw_gaps"],
                "number_of_degenerate_gaps": gaps["number_of_degenerate_gaps"],
                "degenerate_gap_fraction": gaps["degenerate_gap_fraction"],
                "mean_positive_raw_gap": gaps["mean_positive_raw_gap"],
                "minimum_absolute_raw_gap": float(np.min(np.abs(raw_gaps))) if len(raw_gaps) else math.nan,
                "normalized_positive_spacings": encode_array(
                    np.asarray(gaps["normalized_positive_spacings"], dtype=float)
                ),
                "inclusive_gap_ratios": encode_array(np.asarray(gaps["inclusive_gap_ratios"], dtype=float)),
                "regular_gap_ratios": encode_array(np.asarray(gaps["regular_gap_ratios"], dtype=float)),
                "selected_eigenvalues": encode_array(selected),
            }
        )
    return rows


def graph_row_fieldnames() -> list[str]:
    return [
        "graph_index", "graph_seed", "sequence_seed", "displayed_tau", "tau_convention",
        "effective_density_exponent", "quantile", "quantile_low", "quantile_high", "index_start",
        "index_stop", "number_of_eigenvalues", "E_low", "E_high", "E_center", "E_span",
        "continuous_bulk_dimension", "zero_eigenvalue_count", "zero_eigenvalue_fraction",
        "perron_eigenvalue", "full_spectrum_atom_count_at_1", "selected_atom_count_at_1",
        "selected_atom_fraction_at_1", "number_of_raw_gaps", "number_of_degenerate_gaps",
        "degenerate_gap_fraction", "mean_positive_raw_gap", "minimum_absolute_raw_gap",
        "normalized_positive_spacings", "inclusive_gap_ratios", "regular_gap_ratios",
        "selected_eigenvalues",
    ]


def _mean_and_se(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return math.nan, math.nan
    mean = float(np.mean(array))
    if len(array) == 1:
        return mean, math.nan
    return mean, float(np.std(array, ddof=1) / np.sqrt(len(array)))


def _pooled(rows: list[dict[str, str]], field: str) -> np.ndarray:
    arrays = [decode_array(row.get(field, "")) for row in rows]
    arrays = [array for array in arrays if len(array)]
    return np.concatenate(arrays) if arrays else np.array([], dtype=float)


def _graph_equal_ratio(rows: list[dict[str, str]], field: str) -> tuple[float, float, int]:
    means = [float(np.mean(values)) for row in rows if len(values := decode_array(row.get(field, "")))]
    mean, se = _mean_and_se(means)
    return mean, se, len(means)


def goe_spacing_cdf(values: np.ndarray) -> np.ndarray:
    x = np.maximum(np.asarray(values, dtype=float), 0.0)
    return 1.0 - np.exp(-np.pi * np.square(x) / 4.0)


def poisson_spacing_cdf(values: np.ndarray) -> np.ndarray:
    x = np.maximum(np.asarray(values, dtype=float), 0.0)
    return 1.0 - np.exp(-x)


def aggregate_graph_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[float, float], list[dict[str, str]]] = {}
    for row in rows:
        key = (float(row["quantile_low"]), float(row["quantile_high"]))
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, object]] = []
    for key in sorted(grouped):
        q_rows = grouped[key]
        first = q_rows[0]
        spacings = _pooled(q_rows, "normalized_positive_spacings")
        inclusive_ratios = _pooled(q_rows, "inclusive_gap_ratios")
        regular_ratios = _pooled(q_rows, "regular_gap_ratios")
        inclusive_mean, inclusive_se, graphs_with_inclusive = _graph_equal_ratio(
            q_rows, "inclusive_gap_ratios"
        )
        regular_mean, regular_se, graphs_with_regular = _graph_equal_ratio(q_rows, "regular_gap_ratios")
        total_eigenvalues = sum(int(row["number_of_eigenvalues"]) for row in q_rows)
        total_atom = sum(int(row["selected_atom_count_at_1"]) for row in q_rows)
        total_raw_gaps = sum(int(row["number_of_raw_gaps"]) for row in q_rows)
        total_degenerate = sum(int(row["number_of_degenerate_gaps"]) for row in q_rows)
        E_center_mean, E_center_se = _mean_and_se(float(row["E_center"]) for row in q_rows)
        atom_graph_mean, atom_graph_se = _mean_and_se(
            float(row["selected_atom_fraction_at_1"]) for row in q_rows
        )
        degenerate_graph_mean, degenerate_graph_se = _mean_and_se(
            float(row["degenerate_gap_fraction"]) for row in q_rows
        )
        full_atom_mean, full_atom_se = _mean_and_se(
            float(row["full_spectrum_atom_count_at_1"]) for row in q_rows
        )
        status_parts: list[str] = []
        if len(spacings) < MIN_POOLED_SPACINGS:
            status_parts.append("insufficient_positive_spacings")
        if len(regular_ratios) < MIN_POOLED_RATIOS:
            status_parts.append("insufficient_regular_gap_ratios")
        if len(spacings):
            ks_goe = float(kstest(spacings, goe_spacing_cdf).statistic)
            ks_poisson = float(kstest(spacings, poisson_spacing_cdf).statistic)
            small_spacing_probability = float(np.mean(spacings < SMALL_SPACING_THRESHOLD))
        else:
            ks_goe = ks_poisson = small_spacing_probability = math.nan
        output.append(
            {
                "displayed_tau": float(first["displayed_tau"]),
                "tau_convention": first["tau_convention"],
                "effective_density_exponent": float(first["effective_density_exponent"]),
                "quantile": 0.5 * (key[0] + key[1]),
                "quantile_low": key[0],
                "quantile_high": key[1],
                "number_of_graphs": len(q_rows),
                "number_of_eigenvalues": total_eigenvalues,
                "mean_E_center": E_center_mean,
                "mean_E_center_standard_error": E_center_se,
                "pooled_atom_fraction_at_1": total_atom / max(total_eigenvalues, 1),
                "graph_equal_atom_fraction_at_1": atom_graph_mean,
                "graph_equal_atom_fraction_at_1_standard_error": atom_graph_se,
                "mean_full_spectrum_atom_count_at_1": full_atom_mean,
                "mean_full_spectrum_atom_count_at_1_standard_error": full_atom_se,
                "pooled_degenerate_gap_fraction": total_degenerate / max(total_raw_gaps, 1),
                "graph_equal_degenerate_gap_fraction": degenerate_graph_mean,
                "graph_equal_degenerate_gap_fraction_standard_error": degenerate_graph_se,
                "number_of_positive_spacings": int(len(spacings)),
                "number_of_inclusive_gap_ratios": int(len(inclusive_ratios)),
                "number_of_regular_gap_ratios": int(len(regular_ratios)),
                "graphs_with_inclusive_gap_ratios": graphs_with_inclusive,
                "graphs_with_regular_gap_ratios": graphs_with_regular,
                "pooled_mean_inclusive_gap_ratio": (
                    float(np.mean(inclusive_ratios)) if len(inclusive_ratios) else math.nan
                ),
                "graph_equal_mean_inclusive_gap_ratio": inclusive_mean,
                "graph_equal_mean_inclusive_gap_ratio_standard_error": inclusive_se,
                "pooled_mean_regular_gap_ratio": (
                    float(np.mean(regular_ratios)) if len(regular_ratios) else math.nan
                ),
                "graph_equal_mean_regular_gap_ratio": regular_mean,
                "graph_equal_mean_regular_gap_ratio_standard_error": regular_se,
                "conditional_positive_spacing_KS_to_GOE": ks_goe,
                "conditional_positive_spacing_KS_to_Poisson": ks_poisson,
                "conditional_positive_small_spacing_probability": small_spacing_probability,
                "status": "ok" if not status_parts else ";".join(status_parts),
            }
        )
    return output


def summary_row_fieldnames() -> list[str]:
    return [
        "displayed_tau", "tau_convention", "effective_density_exponent", "quantile",
        "quantile_low", "quantile_high", "number_of_graphs", "number_of_eigenvalues",
        "mean_E_center", "mean_E_center_standard_error", "pooled_atom_fraction_at_1",
        "graph_equal_atom_fraction_at_1", "graph_equal_atom_fraction_at_1_standard_error",
        "mean_full_spectrum_atom_count_at_1", "mean_full_spectrum_atom_count_at_1_standard_error",
        "pooled_degenerate_gap_fraction", "graph_equal_degenerate_gap_fraction",
        "graph_equal_degenerate_gap_fraction_standard_error", "number_of_positive_spacings",
        "number_of_inclusive_gap_ratios", "number_of_regular_gap_ratios",
        "graphs_with_inclusive_gap_ratios", "graphs_with_regular_gap_ratios",
        "pooled_mean_inclusive_gap_ratio", "graph_equal_mean_inclusive_gap_ratio",
        "graph_equal_mean_inclusive_gap_ratio_standard_error", "pooled_mean_regular_gap_ratio",
        "graph_equal_mean_regular_gap_ratio", "graph_equal_mean_regular_gap_ratio_standard_error",
        "conditional_positive_spacing_KS_to_GOE", "conditional_positive_spacing_KS_to_Poisson",
        "conditional_positive_small_spacing_probability", "status",
    ]


def write_csv(
    rows: Iterable[dict[str, object]],
    path: Path,
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener: Callable[..., object] = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def read_csv_paths(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        opener: Callable[..., object] = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", newline="") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    return rows


def eigenvector_summary_path(displayed_tau: float) -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "outputs"
        / "chung_lu"
        / "power_law"
        / "eigenvectors"
        / "percentile_scan_q0p5_1_width_0p002"
        / f"n_{N}_tau_{eigenvector_scan.tau_slug(displayed_tau)}_avgdeg_20p0"
        / "eigenvector_percentile_scan_summary.csv"
    )


def alignment_rows(
    eigenvalue_rows: list[dict[str, object]],
    displayed_tau: float,
) -> list[dict[str, object]]:
    with eigenvector_summary_path(displayed_tau).open(newline="") as handle:
        vector_rows = list(csv.DictReader(handle))
    vector_by_q = {round(float(row["quantile"]), 12): row for row in vector_rows}
    output: list[dict[str, object]] = []
    for eigenvalue_row in eigenvalue_rows:
        q = round(float(eigenvalue_row["quantile"]), 12)
        vector = vector_by_q[q]
        output.append(
            {
                **eigenvalue_row,
                "eigenvector_mean_n_times_ipr": float(vector["mean_n_times_ipr"]),
                "eigenvector_mean_n_times_ipr_standard_error": float(
                    vector["mean_n_times_ipr_standard_error"]
                ),
                "eigenvector_mean_top_10_mass": float(vector["mean_top_10_mass"]),
                "eigenvector_mean_top_50_mass": float(vector["mean_top_50_mass"]),
                "eigenvector_mean_top_200_mass": float(vector["mean_top_200_mass"]),
                "eigenvector_mean_r_90": float(vector["mean_r_90"]),
            }
        )
    return output


def alignment_fieldnames() -> list[str]:
    return summary_row_fieldnames() + [
        "eigenvector_mean_n_times_ipr", "eigenvector_mean_n_times_ipr_standard_error",
        "eigenvector_mean_top_10_mass", "eigenvector_mean_top_50_mass",
        "eigenvector_mean_top_200_mass", "eigenvector_mean_r_90",
    ]


def peak_alignment(rows: list[dict[str, object]]) -> dict[str, object]:
    atom = max(rows, key=lambda row: float(row["pooled_atom_fraction_at_1"]))
    degenerate = max(rows, key=lambda row: float(row["pooled_degenerate_gap_fraction"]))
    localized = max(rows, key=lambda row: float(row["eigenvector_mean_n_times_ipr"]))
    return {
        "eigenvalue_atom_peak_quantile": float(atom["quantile"]),
        "eigenvalue_atom_peak_energy": float(atom["mean_E_center"]),
        "eigenvalue_atom_peak_fraction": float(atom["pooled_atom_fraction_at_1"]),
        "degenerate_gap_peak_quantile": float(degenerate["quantile"]),
        "degenerate_gap_peak_energy": float(degenerate["mean_E_center"]),
        "degenerate_gap_peak_fraction": float(degenerate["pooled_degenerate_gap_fraction"]),
        "eigenvector_ipr_peak_quantile": float(localized["quantile"]),
        "eigenvector_ipr_peak_energy": float(localized["mean_E_center"]),
        "eigenvector_ipr_peak_n_times_ipr": float(localized["eigenvector_mean_n_times_ipr"]),
        "atom_to_ipr_peak_quantile_difference": (
            float(atom["quantile"]) - float(localized["quantile"])
        ),
        "degenerate_gap_to_ipr_peak_quantile_difference": (
            float(degenerate["quantile"]) - float(localized["quantile"])
        ),
    }


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_alignment(rows: list[dict[str, object]], displayed_tau: float):
    plt = _pyplot()
    q = np.asarray([float(row["quantile"]) for row in rows])
    atom = np.asarray([float(row["pooled_atom_fraction_at_1"]) for row in rows])
    degenerate = np.asarray([float(row["pooled_degenerate_gap_fraction"]) for row in rows])
    inclusive_r = np.asarray([float(row["graph_equal_mean_inclusive_gap_ratio"]) for row in rows])
    regular_r = np.asarray([float(row["graph_equal_mean_regular_gap_ratio"]) for row in rows])
    ks_goe = np.asarray([float(row["conditional_positive_spacing_KS_to_GOE"]) for row in rows])
    ks_poisson = np.asarray([float(row["conditional_positive_spacing_KS_to_Poisson"]) for row in rows])
    n_ipr = np.asarray([float(row["eigenvector_mean_n_times_ipr"]) for row in rows])
    top_50 = np.asarray([float(row["eigenvector_mean_top_50_mass"]) for row in rows])

    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.7), sharex=True)
    axes[0, 0].plot(q, atom, label="eigenvalues at E=1", color="darkorange")
    axes[0, 0].plot(q, degenerate, label="near-zero gap fraction", color="firebrick")
    axes[0, 0].set_ylabel("fraction")
    axes[0, 0].set_title("Spectral atom and degeneracy")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(q, inclusive_r, label="gap ratio incl. zero neighbors", color="black")
    axes[0, 1].plot(q, regular_r, label="gap ratio, both gaps positive", color="steelblue")
    axes[0, 1].axhline(0.53590, color="0.25", ls="--", label="GOE")
    axes[0, 1].axhline(0.38629, color="firebrick", ls="--", label="Poisson")
    axes[0, 1].set_ylabel("graph-equal mean gap ratio")
    axes[0, 1].set_title("Gap-ratio response")
    axes[0, 1].legend(frameon=False, fontsize=8)

    axes[1, 0].plot(q, ks_goe, label="conditional KS to GOE", color="black")
    axes[1, 0].plot(q, ks_poisson, label="conditional KS to Poisson", color="firebrick")
    axes[1, 0].set_ylabel("KS distance")
    axes[1, 0].set_title("Positive spacings after removing atom gaps")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(q, n_ipr, color="purple", label="eigenvector N × IPR")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylabel("N × IPR", color="purple")
    twin = axes[1, 1].twinx()
    twin.plot(q, top_50, color="seagreen", label="eigenvector top-50 mass")
    twin.set_ylabel("top-50 mass", color="seagreen")
    axes[1, 1].set_title("Matched eigenvector localization")

    for axis in axes.flat:
        axis.set_xlabel("continuous non-Perron spectral quantile")
        axis.grid(alpha=0.2)
    figure.suptitle(
        f"Chung–Lu E=1 resonance: eigenvalues versus eigenvectors, "
        f"N={N}, displayed tau={displayed_tau:g}, G={NUM_GRAPHS}"
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
    specification: eigenvector_scan.LegacyTauSpecification,
    num_partial_files: int,
) -> dict[str, object]:
    return {
        "model": "power_law_chung_lu",
        "diagnostic": "nonperron_eigenvalue_resonance_zoom_matched_to_eigenvectors",
        "output_is_additive_and_does_not_replace_existing_eigenvalue_results": True,
        "n": N,
        "num_graphs": NUM_GRAPHS,
        "displayed_tau": specification.displayed_tau,
        "tau_convention": specification.tau_convention,
        "effective_density_exponent_passed_to_generator": specification.effective_density_exponent,
        "cutoff_coefficient": specification.cutoff_coefficient,
        "target_mean_degree": TARGET_MEAN_DEGREE,
        "graph_seed_base": specification.graph_seed_base,
        "sequence_seed_base": specification.sequence_seed_base,
        "same_graph_seed_stream_as_eigenvalue_and_eigenvector_runs": True,
        "component_mode": eigenvector_scan.COMPONENT_MODE,
        "sequence_mode": eigenvector_scan.SEQUENCE_MODE,
        "cutoff_mode": eigenvector_scan.CUTOFF_MODE,
        "remove_perron_eigenpair": REMOVE_PERRON_EIGENPAIR,
        "remove_exact_zero_eigenvalues": REMOVE_EXACT_ZERO_EIGENVALUES,
        "zero_eigenvalue_atol": ZERO_EIGENVALUE_ATOL,
        "quantile_definition": "rank within continuous non-Perron eigenvalues of each graph",
        "quantile_min": Q_MIN,
        "quantile_max": Q_MAX,
        "quantile_width": Q_WIDTH,
        "num_quantile_bins": len(LOCAL_QUANTILE_BINS),
        "degenerate_gap_atol": DEGENERATE_GAP_ATOL,
        "atom_energy": ATOM_ENERGY,
        "atom_energy_atol": ATOM_ENERGY_ATOL,
        "conditional_spacing_note": (
            "KS tests use positive gaps only; atom/degenerate gaps are retained in separate fractions "
            "and inclusive gap ratios"
        ),
        "source_eigenvector_summary": str(eigenvector_summary_path(specification.displayed_tau)),
        "source_eigenvalue_manifest": specification.source_eigenvalue_manifest,
        "num_partial_files": num_partial_files,
        "code_version": code_version(),
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def aggregate_parameter(displayed_tau: float) -> Path:
    output_dir = parameter_output_dir(displayed_tau)
    partial_paths = sorted((output_dir / "partials").glob("eigenvalue_resonance_graphs_*.csv.gz"))
    if not partial_paths:
        raise FileNotFoundError(f"missing partial files in {output_dir / 'partials'}")
    graph_rows = read_csv_paths(partial_paths)
    graph_ids = sorted({int(row["graph_index"]) for row in graph_rows})
    expected = list(range(NUM_GRAPHS))
    if graph_ids != expected:
        missing = sorted(set(expected).difference(graph_ids))
        raise ValueError(f"missing graph indices: {missing[:20]} (total={len(missing)})")
    summary = aggregate_graph_rows(graph_rows)
    if len(summary) != len(LOCAL_QUANTILE_BINS):
        raise ValueError("aggregated summary has the wrong number of quantile bins")
    aligned = alignment_rows(summary, displayed_tau)
    write_csv(
        graph_rows,
        output_dir / "eigenvalue_resonance_by_graph.csv.gz",
        graph_row_fieldnames(),
    )
    write_csv(
        summary,
        output_dir / "eigenvalue_resonance_summary.csv",
        summary_row_fieldnames(),
    )
    write_csv(
        aligned,
        output_dir / "eigenvalue_eigenvector_alignment.csv",
        alignment_fieldnames(),
    )
    with (output_dir / "peak_alignment.json").open("w") as handle:
        json.dump(peak_alignment(aligned), handle, indent=2, sort_keys=True)
    specification = eigenvector_scan.tau_specification(displayed_tau)
    with (output_dir / "run_manifest.json").open("w") as handle:
        json.dump(run_manifest(specification, len(partial_paths)), handle, indent=2, sort_keys=True)
    figure = plot_alignment(aligned, displayed_tau)
    figure.savefig(output_dir / "eigenvalue_eigenvector_resonance_alignment.png", dpi=200, bbox_inches="tight")
    _pyplot().close(figure)
    return output_dir
