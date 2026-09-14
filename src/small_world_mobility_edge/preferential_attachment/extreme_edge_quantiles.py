"""Cross-fitted extreme positive-edge scans from authoritative PA v2 caches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .full_spectrum_scan import (
    MobilityEdgeProductionConfig,
    make_seeds,
    mobility_chunk_paths,
    slug_number,
    validate_mobility_chunk,
)
from .mobility_edge_postprocess import (
    ANALYSIS_VERSION,
    GOE_MEAN_GAP_RATIO,
    POISSON_MEAN_GAP_RATIO,
    GraphGapRecords,
    build_cross_fitted_models,
    build_gap_records,
    cross_fitted_unfolded_gaps,
    ks_distance_unit_mean_spacing,
    write_rows,
)


EXTREME_EDGE_ANALYSIS_VERSION = "pa_extreme_positive_edge_crossfit_cdf_v1"
REQUESTED_BOUNDS = (0.995, 0.996, 0.997, 0.998, 0.999, 0.9995, 0.9999, 0.99995, 1.0)
REQUESTED_INTERVALS = tuple(zip(REQUESTED_BOUNDS[:-1], REQUESTED_BOUNDS[1:]))
MINIMUM_HISTOGRAM_GAPS = 100
PREFERRED_CONTRIBUTING_GRAPHS = 50
G_MAX = 2000
ZERO_ATOL = 1e-14
HISTOGRAM_EDGES = np.linspace(0.0, 3.5, 36)


@dataclass(frozen=True)
class ModelSpecification:
    model_family: str
    parameter_name: str
    parameter_value: float

    @property
    def label(self) -> str:
        prefix = "affine alpha" if self.model_family == "affine_linear" else "superlinear zeta"
        return f"{prefix}={self.parameter_value:g}"

    @property
    def slug(self) -> str:
        return f"{self.model_family}_{self.parameter_name}_{slug_number(self.parameter_value)}"


DEFAULT_MODELS = (
    ModelSpecification("affine_linear", "degree_ccdf_alpha", 2.00),
    ModelSpecification("affine_linear", "degree_ccdf_alpha", 1.50),
    ModelSpecification("affine_linear", "degree_ccdf_alpha", 1.25),
    ModelSpecification("affine_linear", "degree_ccdf_alpha", 1.05),
    ModelSpecification("nonlinear", "attachment_zeta", 1.05),
    ModelSpecification("nonlinear", "attachment_zeta", 1.20),
    ModelSpecification("nonlinear", "attachment_zeta", 1.50),
)


def _window_mask(values: np.ndarray, q_low: float, q_high: float) -> np.ndarray:
    if q_high >= 1.0 - 1e-15:
        return (values >= q_low) & (values <= q_high)
    return (values >= q_low) & (values < q_high)


def _mean_se(values: Iterable[float]) -> tuple[float, float, int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return math.nan, math.nan, 0
    return (
        float(np.mean(array)),
        float(np.std(array, ddof=1) / math.sqrt(len(array))) if len(array) > 1 else math.nan,
        int(len(array)),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cache_execution_fields(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        if "execution_json" not in data.files:
            return {
                "source_sge_job_id": "LEGACY_CACHE_NO_EMBEDDED_JOB_ID",
                "source_sge_task_id": "LEGACY_CACHE_NO_EMBEDDED_TASK_ID",
            }
        execution = json.loads(str(data["execution_json"].item()))
    return {
        "source_sge_job_id": execution.get("sge_job_id", ""),
        "source_sge_task_id": execution.get("sge_task_id", ""),
        "source_python_version": execution.get("python_version", ""),
        "source_numpy_version": execution.get("numpy_version", ""),
        "source_scipy_version": execution.get("scipy_version", ""),
        "source_networkx_version": execution.get("networkx_version", ""),
    }


def parameter_directory(root: Path, specification: ModelSpecification) -> Path:
    return (
        root
        / specification.model_family
        / "n_5000_m_10"
        / f"{specification.parameter_name}_{slug_number(specification.parameter_value)}"
    )


def config_from_manifest(root: Path, specification: ModelSpecification) -> MobilityEdgeProductionConfig:
    directory = parameter_directory(root, specification)
    manifest = json.loads((directory / "run_manifest.json").read_text())
    source_analysis_version = manifest.get(
        "postprocess_analysis_version", manifest.get("analysis_version")
    )
    if source_analysis_version != ANALYSIS_VERSION:
        raise ValueError(f"{directory} is not authoritative atom-segmented v2 output")
    canonical = manifest.get("canonical_parameter", {})
    if canonical.get("name") != specification.parameter_name or not math.isclose(
        float(canonical.get("value", math.nan)), specification.parameter_value, abs_tol=1e-12
    ):
        raise ValueError(f"canonical parameter mismatch in {directory}")
    common = {
        "n": int(manifest["n"]),
        "m": int(manifest["m"]),
        "num_graphs": int(manifest["num_graphs"]),
        "graph_seed_base": int(manifest["graph_seed_base"]),
        "output_root": root,
        "model_family": specification.model_family,
        "quantile_width": float(manifest.get("quantile_width", 0.002)),
        "zero_eigenvalue_atol": float(
            manifest.get(
                "zero_eigenvalue_atol",
                manifest.get("atom_filtering", {}).get("atol", 1e-8),
            )
        ),
    }
    if specification.model_family == "affine_linear":
        return MobilityEdgeProductionConfig(**common, degree_ccdf_alpha=specification.parameter_value)
    return MobilityEdgeProductionConfig(**common, attachment_zeta=specification.parameter_value)


def load_verified_records(
    config: MobilityEdgeProductionConfig,
    *,
    chunk_size: int = 5,
) -> tuple[list[GraphGapRecords], list[Path], list[dict[str, object]]]:
    manifest = json.loads((config.parameter_dir / "run_manifest.json").read_text())
    first_production_cache, _ = mobility_chunk_paths(
        config, 0, min(chunk_size, config.num_graphs)
    )
    if first_production_cache.exists():
        return _load_verified_production_records(config, chunk_size=chunk_size)
    return _load_verified_legacy_records(config, manifest, chunk_size=chunk_size)


def load_verified_augmentation_records(
    config: MobilityEdgeProductionConfig,
    specification: ModelSpecification,
    output_root: Path,
    *,
    base_graph_count: int,
) -> tuple[list[GraphGapRecords], list[Path], list[dict[str, object]]]:
    partial_dir = output_root / "augmentation_caches" / specification.slug / "partials"
    if not partial_dir.is_dir():
        return [], [], []
    pattern = re.compile(r"extreme_edge_spectral_arrays_(\d+)_(\d+)\.npz$")
    indexed_paths: list[tuple[int, int, Path]] = []
    for path in partial_dir.glob("extreme_edge_spectral_arrays_*.npz"):
        match = pattern.search(path.name)
        if match:
            indexed_paths.append((int(match.group(1)), int(match.group(2)), path))
    indexed_paths.sort()
    expected_start = base_graph_count
    records: list[GraphGapRecords] = []
    cache_paths: list[Path] = []
    source_rows: list[dict[str, object]] = []
    for start, stop, cache_path in indexed_paths:
        if start != expected_start or stop <= start:
            raise ValueError(
                f"non-contiguous extreme-edge augmentation at {cache_path}; expected start {expected_start}"
            )
        with np.load(cache_path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            connected = np.asarray(data["connected"], dtype=bool)
        expected_metadata = {
            "analysis_version": "pa_extreme_edge_eigenvalue_augmentation_v1",
            "model_family": specification.model_family,
            "parameter_name": specification.parameter_name,
            "parameter_value": specification.parameter_value,
            "n": config.n,
            "m": config.m,
            "graph_seed_base": config.graph_seed_base,
            "base_graph_count": base_graph_count,
            "perron_removed": True,
            "zero_eigenvalue_atol": config.zero_eigenvalue_atol,
        }
        for key, expected in expected_metadata.items():
            observed = metadata.get(key)
            if isinstance(expected, float):
                valid = math.isclose(float(observed), expected, abs_tol=1e-12)
            else:
                valid = observed == expected
            if not valid:
                raise ValueError(f"augmentation metadata mismatch for {key} in {cache_path}")
        chunk_records, indices = _records_from_spectral_arrays(
            cache_path, zero_eigenvalue_atol=config.zero_eigenvalue_atol
        )
        if not np.array_equal(indices, np.arange(start, stop)):
            raise ValueError(f"augmentation graph-index mismatch in {cache_path}")
        expected_seeds = np.asarray(
            make_seeds(stop, config.graph_seed_base)[start:stop], dtype=np.uint64
        )
        with np.load(cache_path, allow_pickle=False) as data:
            observed_seeds = np.asarray(data["graph_seeds"], dtype=np.uint64)
        if not np.array_equal(observed_seeds, expected_seeds):
            raise ValueError(f"augmentation graph-seed mismatch in {cache_path}")
        if len(connected) != stop - start or not np.all(connected):
            raise ValueError(f"disconnected graph in augmentation cache {cache_path}")
        records.extend(chunk_records)
        cache_paths.append(cache_path)
        source_rows.append(
            {
                "graph_index_start": start,
                "graph_index_stop_exclusive": stop,
                "cache_path": str(cache_path),
                "cache_sha256": _sha256(cache_path),
                "structural_path": "EIGENVALUES_ONLY_TARGETED_AUGMENTATION",
                "structural_sha256": "NOT_APPLICABLE",
                "validation": "augmentation cache metadata, connectivity, seeds, graph indices, finiteness, and spectral ordering valid",
                **_cache_execution_fields(cache_path),
            }
        )
        expected_start = stop
    return records, cache_paths, source_rows


def _records_from_spectral_arrays(
    cache_path: Path,
    *,
    zero_eigenvalue_atol: float,
) -> tuple[list[GraphGapRecords], np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as data:
        required = {"graph_indices", "graph_seeds", "nonperron_eigenvalues"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{cache_path} lacks required arrays: {sorted(missing)}")
        indices = np.asarray(data["graph_indices"], dtype=int)
        seeds = np.asarray(data["graph_seeds"], dtype=np.uint64)
        spectra = np.asarray(data["nonperron_eigenvalues"], dtype=float)
    if spectra.ndim != 2 or spectra.shape[0] != len(indices) or len(seeds) != len(indices):
        raise ValueError(f"inconsistent spectral array shapes in {cache_path}")
    if not np.all(np.isfinite(spectra)) or np.any(np.diff(spectra, axis=1) < 0.0):
        raise ValueError(f"non-finite or unsorted eigenspectrum in {cache_path}")
    return (
        [
            build_gap_records(
                spectra[local],
                graph_index=int(graph_index),
                graph_seed=int(seeds[local]),
                excluded_atom_energies=(0.0,),
                audited_atom_energies=(-1.0, 0.0, 1.0),
                atom_atol=zero_eigenvalue_atol,
            )
            for local, graph_index in enumerate(indices)
        ],
        indices,
    )


def _load_verified_production_records(
    config: MobilityEdgeProductionConfig,
    *,
    chunk_size: int,
) -> tuple[list[GraphGapRecords], list[Path], list[dict[str, object]]]:
    records: list[GraphGapRecords] = []
    cache_paths: list[Path] = []
    source_rows: list[dict[str, object]] = []
    for start in range(0, config.num_graphs, chunk_size):
        stop = min(start + chunk_size, config.num_graphs)
        valid, reason = validate_mobility_chunk(config, start, stop)
        if not valid:
            raise ValueError(f"invalid source cache [{start},{stop}): {reason}")
        cache_path, structural_path = mobility_chunk_paths(config, start, stop)
        cache_paths.append(cache_path)
        chunk_records, indices = _records_from_spectral_arrays(
            cache_path, zero_eigenvalue_atol=config.zero_eigenvalue_atol
        )
        records.extend(chunk_records)
        source_rows.append(
            {
                "graph_index_start": start,
                "graph_index_stop_exclusive": stop,
                "cache_path": str(cache_path),
                "cache_sha256": _sha256(cache_path),
                "structural_path": str(structural_path),
                "structural_sha256": _sha256(structural_path),
                "validation": reason,
                **_cache_execution_fields(cache_path),
            }
        )
    if [record.graph_index for record in records] != list(range(config.num_graphs)):
        raise AssertionError("verified records do not cover the exact graph-index range")
    return records, cache_paths, source_rows


def _load_verified_legacy_records(
    config: MobilityEdgeProductionConfig,
    manifest: dict[str, object],
    *,
    chunk_size: int,
) -> tuple[list[GraphGapRecords], list[Path], list[dict[str, object]]]:
    if manifest.get("analysis_version") != ANALYSIS_VERSION:
        raise ValueError("legacy cache fallback is allowed only for authoritative v2 output")
    if not manifest.get("perron_removed", True):
        raise ValueError("legacy source manifest does not confirm Perron removal")
    source_root = Path(str(manifest.get("source_cache", "")))
    partial_dir = source_root / "partials"
    records: list[GraphGapRecords] = []
    cache_paths: list[Path] = []
    source_rows: list[dict[str, object]] = []
    for start in range(0, config.num_graphs, chunk_size):
        stop = min(start + chunk_size, config.num_graphs)
        cache_path = partial_dir / f"spectral_arrays_{start:04d}_{stop:04d}.npz"
        if not cache_path.is_file():
            raise ValueError(f"missing legacy spectral source cache {cache_path}")
        chunk_records, indices = _records_from_spectral_arrays(
            cache_path, zero_eigenvalue_atol=config.zero_eigenvalue_atol
        )
        expected = np.arange(start, stop)
        if not np.array_equal(indices, expected):
            raise ValueError(f"legacy graph-index mismatch in {cache_path}")
        records.extend(chunk_records)
        cache_paths.append(cache_path)
        source_rows.append(
            {
                "graph_index_start": start,
                "graph_index_stop_exclusive": stop,
                "cache_path": str(cache_path),
                "cache_sha256": _sha256(cache_path),
                "structural_path": "NOT_REQUIRED_FOR_EDGE_SPACING",
                "structural_sha256": "NOT_APPLICABLE",
                "validation": "authoritative v2 legacy eigenspectrum cache; arrays readable, finite, sorted, and graph indices exact",
                **_cache_execution_fields(cache_path),
            }
        )
    if [record.graph_index for record in records] != list(range(config.num_graphs)):
        raise AssertionError("legacy records do not cover the exact graph-index range")
    return records, cache_paths, source_rows


def model_by_graph(models) -> dict[int, object]:
    output: dict[int, object] = {}
    for model in models:
        training = set(model.training_graph_ids)
        held = set(model.held_out_graph_ids)
        if training & held:
            raise AssertionError(f"cross-fit leakage in fold {model.fold}")
        for graph_index in held:
            if graph_index in output:
                raise AssertionError("graph is held out in multiple folds")
            output[graph_index] = model
    return output


def map_edge_records(records: Sequence[GraphGapRecords], models) -> dict[str, np.ndarray]:
    mapping = model_by_graph(models)
    fields: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "graph_index",
            "graph_seed",
            "fold",
            "gap_left",
            "gap_right",
            "gap_midpoint_energy",
            "raw_gap",
            "graph_local_edge_rank",
            "graph_local_rank_quantile",
            "ensemble_crossfit_cdf_quantile",
            "atom_segment_identifier",
            "gap_is_zero",
            "crossfit_unfolded_spacing",
            "ratio_energy",
            "ratio_value",
            "ratio_graph_local_rank_quantile",
            "ratio_ensemble_crossfit_cdf_quantile",
            "ratio_category_code",
        )
    }
    category_codes = {"zero_zero": 0, "one_zero": 1, "positive_positive": 2}
    for record in records:
        model = mapping[record.graph_index]
        density = model.density_model
        gap_q_ensemble = np.interp(
            record.gap_energy, density.grid, density.cdf, left=0.0, right=1.0
        )
        ratio_q_ensemble = np.interp(
            record.ratio_energy, density.grid, density.cdf, left=0.0, right=1.0
        )
        unfolded = cross_fitted_unfolded_gaps(record, model)
        gap_keep = gap_q_ensemble >= REQUESTED_BOUNDS[0]
        ratio_keep = ratio_q_ensemble >= REQUESTED_BOUNDS[0]
        gap_count = int(np.sum(gap_keep))
        ratio_count = int(np.sum(ratio_keep))
        gap_positions = np.arange(len(record.gap_value), dtype=np.int32)
        fields["graph_index"].append(np.full(gap_count, record.graph_index, dtype=np.int32))
        fields["graph_seed"].append(np.full(gap_count, record.graph_seed, dtype=np.uint64))
        fields["fold"].append(np.full(gap_count, model.fold, dtype=np.int8))
        fields["gap_left"].append(record.gap_left[gap_keep])
        fields["gap_right"].append(record.gap_right[gap_keep])
        fields["gap_midpoint_energy"].append(record.gap_energy[gap_keep])
        fields["raw_gap"].append(record.gap_value[gap_keep])
        fields["graph_local_edge_rank"].append((len(gap_positions) - gap_positions)[gap_keep])
        fields["graph_local_rank_quantile"].append(record.gap_q[gap_keep])
        fields["ensemble_crossfit_cdf_quantile"].append(gap_q_ensemble[gap_keep])
        segment = np.where(record.gap_right < 0.0, -1, np.where(record.gap_left > 0.0, 1, 0))
        if np.any(segment[gap_keep] == 0):
            raise AssertionError("an extreme-edge gap bridges the excluded zero atom")
        fields["atom_segment_identifier"].append(segment[gap_keep].astype(np.int8))
        fields["gap_is_zero"].append(record.gap_is_zero[gap_keep])
        fields["crossfit_unfolded_spacing"].append(unfolded[gap_keep])
        fields["ratio_energy"].append(record.ratio_energy[ratio_keep])
        fields["ratio_value"].append(record.ratio_value[ratio_keep])
        fields["ratio_graph_local_rank_quantile"].append(record.ratio_q[ratio_keep])
        fields["ratio_ensemble_crossfit_cdf_quantile"].append(ratio_q_ensemble[ratio_keep])
        fields["ratio_category_code"].append(
            np.asarray([category_codes[value] for value in record.ratio_category[ratio_keep]], dtype=np.int8)
        )
        if ratio_count and len(fields["ratio_value"][-1]) != ratio_count:
            raise AssertionError("ratio mapping length mismatch")
    output = {
        name: np.concatenate(values) if values else np.array([], dtype=float)
        for name, values in fields.items()
    }
    return output


def load_edge_localization(
    cache_paths: Sequence[Path], models, q_min: float = REQUESTED_BOUNDS[0]
) -> dict[str, np.ndarray]:
    mapping = model_by_graph(models)
    output = {name: [] for name in ("graph_index", "ensemble_q", "n_times_ipr", "top_50_mass", "participation_fraction")}
    for cache_path in cache_paths:
        with np.load(cache_path, allow_pickle=False) as data:
            required = {
                "eigenpair_offsets",
                "eigenpair_eigenvalue",
                "eigenpair_n_times_ipr",
                "eigenpair_top_50_mass",
                "eigenpair_participation_fraction",
            }
            if not required.issubset(data.files):
                continue
            indices = np.asarray(data["graph_indices"], dtype=int)
            offsets = np.asarray(data["eigenpair_offsets"], dtype=int)
            energies = np.asarray(data["eigenpair_eigenvalue"], dtype=float)
            nipr = np.asarray(data["eigenpair_n_times_ipr"], dtype=float)
            top50 = np.asarray(data["eigenpair_top_50_mass"], dtype=float)
            participation = np.asarray(data["eigenpair_participation_fraction"], dtype=float)
        for local, graph_index in enumerate(indices):
            left, right = int(offsets[local]), int(offsets[local + 1])
            density = mapping[int(graph_index)].density_model
            q = np.interp(energies[left:right], density.grid, density.cdf, left=0.0, right=1.0)
            keep = q >= q_min
            output["graph_index"].append(np.full(int(np.sum(keep)), graph_index, dtype=np.int32))
            output["ensemble_q"].append(q[keep])
            output["n_times_ipr"].append(nipr[left:right][keep])
            output["top_50_mass"].append(top50[left:right][keep])
            output["participation_fraction"].append(participation[left:right][keep])
    return {name: np.concatenate(values) if values else np.array([]) for name, values in output.items()}


def _graph_equal_metric(
    graph_ids: np.ndarray, values: np.ndarray, mask: np.ndarray
) -> tuple[float, float, int]:
    means = []
    for graph_index in np.unique(graph_ids[mask]):
        selected = mask & (graph_ids == graph_index) & np.isfinite(values)
        if np.any(selected):
            means.append(float(np.mean(values[selected])))
    return _mean_se(means)


def summarize_interval(
    mapped: dict[str, np.ndarray],
    localization: dict[str, np.ndarray],
    *,
    specification: ModelSpecification,
    q_low: float,
    q_high: float,
    analysis_kind: str,
    nominal_q_low: float,
    nominal_q_high: float,
    merged_cells: int,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    gap_mask = _window_mask(mapped["ensemble_crossfit_cdf_quantile"], q_low, q_high)
    raw_positive = gap_mask & (~mapped["gap_is_zero"].astype(bool)) & (mapped["raw_gap"] > ZERO_ATOL)
    unfolded_positive = raw_positive & np.isfinite(mapped["crossfit_unfolded_spacing"]) & (
        mapped["crossfit_unfolded_spacing"] > 0.0
    )
    unfolded = mapped["crossfit_unfolded_spacing"][unfolded_positive]
    normalized = unfolded / np.mean(unfolded) if len(unfolded) and float(np.mean(unfolded)) > 0.0 else np.array([])
    ratio_mask = _window_mask(mapped["ratio_ensemble_crossfit_cdf_quantile"], q_low, q_high)
    inclusive = ratio_mask & np.isfinite(mapped["ratio_value"])
    positive_ratio = ratio_mask & (mapped["ratio_category_code"] == 2) & np.isfinite(mapped["ratio_value"])
    ratio_graph_ids = mapped["ratio_graph_index"]
    mean_inclusive, se_inclusive, graphs_inclusive = _graph_equal_metric(
        ratio_graph_ids, mapped["ratio_value"], inclusive
    )
    mean_positive, se_positive, graphs_positive = _graph_equal_metric(
        ratio_graph_ids, mapped["ratio_value"], positive_ratio
    )
    energies = mapped["gap_midpoint_energy"][gap_mask]
    raw_values = mapped["raw_gap"][raw_positive]
    localization_mask = _window_mask(localization["ensemble_q"], q_low, q_high)
    localization_summary = {}
    for field in ("n_times_ipr", "top_50_mass", "participation_fraction"):
        mean, se, count = _graph_equal_metric(
            localization["graph_index"], localization[field], localization_mask
        )
        localization_summary[f"graph_equal_mean_{field}"] = mean
        localization_summary[f"graph_equal_se_{field}"] = se
        localization_summary[f"graphs_with_{field}"] = count
    positive_gap_graphs = int(len(np.unique(mapped["graph_index"][unfolded_positive])))
    total_gaps = int(np.sum(gap_mask))
    zero_gaps = int(np.sum(gap_mask & mapped["gap_is_zero"].astype(bool)))
    row: dict[str, object] = {
        "analysis_version": EXTREME_EDGE_ANALYSIS_VERSION,
        "authoritative_source_analysis_version": ANALYSIS_VERSION,
        "model_family": specification.model_family,
        "parameter_name": specification.parameter_name,
        "parameter_value": specification.parameter_value,
        "model_label": specification.label,
        "ensemble_graph_count": int(len(np.unique(mapped["graph_index"]))),
        "analysis_kind": analysis_kind,
        "nominal_q_low": nominal_q_low,
        "nominal_q_high": nominal_q_high,
        "actual_q_low": q_low,
        "actual_q_high": q_high,
        "number_of_merged_cells": merged_cells,
        "quantile_coordinate": "held-out five-fold training-only ensemble empirical-CDF at gap midpoint energy",
        "graph_local_quantile_preserved": True,
        "E_mean": float(np.mean(energies)) if len(energies) else math.nan,
        "E_min": float(np.min(energies)) if len(energies) else math.nan,
        "E_max": float(np.max(energies)) if len(energies) else math.nan,
        "number_of_total_raw_gaps": total_gaps,
        "number_of_raw_positive_gaps": int(np.sum(raw_positive)),
        "number_of_zero_gaps": zero_gaps,
        "zero_gap_fraction": zero_gaps / max(total_gaps, 1),
        "crossfit_unfolded_histogram_sample_count": int(len(normalized)),
        "number_of_contributing_graphs": positive_gap_graphs,
        "mean_raw_positive_gap": float(np.mean(raw_values)) if len(raw_values) else math.nan,
        "median_raw_positive_gap": float(np.median(raw_values)) if len(raw_values) else math.nan,
        "graph_equal_mean_raw_r_inclusive": mean_inclusive,
        "raw_r_inclusive_standard_error": se_inclusive,
        "graphs_with_raw_r_inclusive": graphs_inclusive,
        "graph_equal_mean_raw_r_positive_conditioned": mean_positive,
        "raw_r_positive_conditioned_standard_error": se_positive,
        "graphs_with_raw_r_positive_conditioned": graphs_positive,
        "number_of_valid_raw_ratios": int(np.sum(inclusive)),
        "number_of_positive_positive_ratios": int(np.sum(positive_ratio)),
        "absolute_distance_raw_r_positive_conditioned_to_GOE": abs(mean_positive - GOE_MEAN_GAP_RATIO),
        "absolute_distance_raw_r_positive_conditioned_to_Poisson": abs(mean_positive - POISSON_MEAN_GAP_RATIO),
        "crossfit_unfolded_mean_positive_spacing_before_window_normalization": (
            float(np.mean(unfolded)) if len(unfolded) else math.nan
        ),
        "crossfit_unfolded_KS_to_GOE": ks_distance_unit_mean_spacing(normalized, "goe"),
        "crossfit_unfolded_KS_to_Poisson": ks_distance_unit_mean_spacing(normalized, "poisson"),
        "atom_warning": "ZERO_GAPS_PRESENT" if zero_gaps else "NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW",
        "perron_removed": True,
        "histogram_status": (
            "OK"
            if len(normalized) >= MINIMUM_HISTOGRAM_GAPS and positive_gap_graphs >= PREFERRED_CONTRIBUTING_GRAPHS
            else "INSUFFICIENT_EDGE_SUPPORT_AT_N5000"
        ),
        **localization_summary,
    }
    return row, raw_values, normalized


def add_ratio_graph_ids(mapped: dict[str, np.ndarray], records: Sequence[GraphGapRecords], models) -> None:
    mapping = model_by_graph(models)
    values: list[np.ndarray] = []
    for record in records:
        density = mapping[record.graph_index].density_model
        q = np.interp(record.ratio_energy, density.grid, density.cdf, left=0.0, right=1.0)
        keep = q >= REQUESTED_BOUNDS[0]
        values.append(np.full(int(np.sum(keep)), record.graph_index, dtype=np.int32))
    mapped["ratio_graph_index"] = np.concatenate(values) if values else np.array([], dtype=np.int32)


def adaptive_cell_union(
    interval_index: int,
    mapped: dict[str, np.ndarray],
    localization: dict[str, np.ndarray],
    specification: ModelSpecification,
) -> tuple[dict[str, object], np.ndarray, np.ndarray] | None:
    selected = {interval_index}
    while len(selected) < len(REQUESTED_INTERVALS):
        low = min(selected)
        high = max(selected)
        candidates = []
        if low > 0:
            candidates.append(low - 1)
        if high + 1 < len(REQUESTED_INTERVALS):
            candidates.append(high + 1)
        if not candidates:
            return None
        chosen = min(candidates, key=lambda index: REQUESTED_INTERVALS[index][1] - REQUESTED_INTERVALS[index][0])
        selected.add(chosen)
        q_low = REQUESTED_INTERVALS[min(selected)][0]
        q_high = REQUESTED_INTERVALS[max(selected)][1]
        row, raw, unfolded = summarize_interval(
            mapped,
            localization,
            specification=specification,
            q_low=q_low,
            q_high=q_high,
            analysis_kind="ADAPTIVE_COMPANION",
            nominal_q_low=REQUESTED_INTERVALS[interval_index][0],
            nominal_q_high=REQUESTED_INTERVALS[interval_index][1],
            merged_cells=len(selected) - 1,
        )
        if row["histogram_status"] == "OK":
            return row, raw, unfolded
    return None


def plot_histogram_panels(
    rows: Sequence[dict[str, object]],
    histograms: Sequence[np.ndarray],
    *,
    specification: ModelSpecification,
    output_path: Path,
    raw: bool,
) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(14.5, 7.4))
    x = np.linspace(0.0, 3.5, 500)
    for axis, row, values in zip(axes.flat, rows, histograms):
        if row["histogram_status"] != "OK" or not len(values):
            axis.text(0.5, 0.5, str(row["histogram_status"]), ha="center", va="center", wrap=True)
        elif raw:
            axis.hist(values, bins=30, density=True, color="#457b9d", alpha=0.78)
            axis.set_xlabel("raw positive gap")
        else:
            axis.hist(values[(values >= 0.0) & (values <= 3.5)], bins=HISTOGRAM_EDGES, density=True, color="#52b788", alpha=0.75)
            axis.plot(x, 0.5 * math.pi * x * np.exp(-math.pi * np.square(x) / 4.0), color="black", lw=1.1, label="GOE")
            axis.plot(x, np.exp(-x), color="firebrick", ls="--", lw=1.1, label="Poisson")
            axis.set_xlim(0.0, 3.5)
            axis.set_xlabel("unit-mean cross-fitted spacing")
        axis.set_title(
            "q=[%.5g, %.5g)\nn=%d, graphs=%d"
            % (
                row["nominal_q_low"],
                row["nominal_q_high"],
                row["crossfit_unfolded_histogram_sample_count"],
                row["number_of_contributing_graphs"],
            ),
            fontsize=9,
        )
        axis.grid(alpha=0.15)
    if not raw:
        axes.flat[0].legend(frameon=False, fontsize=8)
    figure.suptitle(f"{specification.label}: {'raw positive gaps' if raw else 'cross-fitted unfolded positive-edge spacings'}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_adaptive_companion_histograms(
    row: dict[str, object],
    raw_values: np.ndarray,
    unfolded_values: np.ndarray,
    *,
    specification: ModelSpecification,
    output_path: Path,
) -> None:
    if row["histogram_status"] != "OK":
        raise ValueError("adaptive companion histograms require supported spacing data")
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    axes[0].hist(raw_values, bins=30, density=True, color="#457b9d", alpha=0.78)
    axes[0].set_xlabel("raw positive gap")
    axes[0].set_ylabel("density")

    x = np.linspace(0.0, 3.5, 500)
    axes[1].hist(
        unfolded_values[(unfolded_values >= 0.0) & (unfolded_values <= 3.5)],
        bins=HISTOGRAM_EDGES,
        density=True,
        color="#52b788",
        alpha=0.75,
    )
    axes[1].plot(
        x,
        0.5 * math.pi * x * np.exp(-math.pi * np.square(x) / 4.0),
        color="black",
        lw=1.1,
        label="GOE",
    )
    axes[1].plot(x, np.exp(-x), color="firebrick", ls="--", lw=1.1, label="Poisson")
    axes[1].set_xlim(0.0, 3.5)
    axes[1].set_xlabel("unit-mean cross-fitted spacing")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.15)
    figure.suptitle(
        "%s adaptive companion: nominal q=[%.5g, %.5g), actual q=[%.5g, %.5g); n=%d, graphs=%d"
        % (
            specification.label,
            row["nominal_q_low"],
            row["nominal_q_high"],
            row["actual_q_low"],
            row["actual_q_high"],
            row["crossfit_unfolded_histogram_sample_count"],
            row["number_of_contributing_graphs"],
        ),
        fontsize=10,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_edge_evolution(rows: Sequence[dict[str, object]], specification: ModelSpecification, output_path: Path) -> None:
    exact = [row for row in rows if row["analysis_kind"] == "EXACT_REQUESTED_INTERVAL"]
    distance = np.asarray([1.0 - 0.5 * (float(row["nominal_q_low"]) + float(row["nominal_q_high"])) for row in exact])
    figure, axes = plt.subplots(3, 1, figsize=(8.5, 9.5), sharex=True)
    axes[0].plot(distance, [row["graph_equal_mean_raw_r_positive_conditioned"] for row in exact], marker="o", label="raw r")
    axes[0].axhline(GOE_MEAN_GAP_RATIO, color="black", ls="--", label="GOE")
    axes[0].axhline(POISSON_MEAN_GAP_RATIO, color="firebrick", ls=":", label="Poisson")
    axes[0].set_ylabel("graph-equal raw r")
    axes[0].legend(frameon=False)
    axes[1].plot(distance, [row["crossfit_unfolded_KS_to_GOE"] for row in exact], marker="o", label="KS to GOE")
    axes[1].plot(distance, [row["crossfit_unfolded_KS_to_Poisson"] for row in exact], marker="s", label="KS to Poisson")
    axes[1].set_ylabel("KS distance")
    axes[1].legend(frameon=False)
    axes[2].plot(distance, [row["crossfit_unfolded_histogram_sample_count"] for row in exact], marker="o", label="positive gaps")
    axes[2].plot(distance, [row["number_of_contributing_graphs"] for row in exact], marker="s", label="graphs")
    axes[2].axhline(MINIMUM_HISTOGRAM_GAPS, color="black", ls="--", lw=1)
    axes[2].set_ylabel("support")
    axes[2].set_xlabel(r"distance from positive edge $1-q$")
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.set_xscale("log")
        axis.invert_xaxis()
        axis.grid(alpha=0.2)
    figure.suptitle(specification.label)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def process_model(
    root: Path,
    output_root: Path,
    specification: ModelSpecification,
    *,
    chunk_size: int = 5,
) -> list[dict[str, object]]:
    config = config_from_manifest(root, specification)
    records, cache_paths, source_rows = load_verified_records(config, chunk_size=chunk_size)
    augmentation_records, augmentation_paths, augmentation_rows = (
        load_verified_augmentation_records(
            config,
            specification,
            output_root,
            base_graph_count=config.num_graphs,
        )
    )
    records.extend(augmentation_records)
    cache_paths.extend(augmentation_paths)
    source_rows.extend(augmentation_rows)
    if [record.graph_index for record in records] != list(range(len(records))):
        raise AssertionError("base and augmentation caches do not form one exact graph-index range")
    ensemble_graph_count = len(records)
    models = build_cross_fitted_models(records, num_folds=5)
    mapped = map_edge_records(records, models)
    add_ratio_graph_ids(mapped, records, models)
    localization = load_edge_localization(cache_paths, models)
    destination = output_root / specification.model_family / specification.slug
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "edge_gap_and_ratio_records.npz", **mapped)
    write_rows(destination / "source_cache_checksums.csv", source_rows)

    rows: list[dict[str, object]] = []
    exact_rows: list[dict[str, object]] = []
    raw_histograms: list[np.ndarray] = []
    unfolded_histograms: list[np.ndarray] = []
    for index, (q_low, q_high) in enumerate(REQUESTED_INTERVALS):
        row, raw, unfolded = summarize_interval(
            mapped,
            localization,
            specification=specification,
            q_low=q_low,
            q_high=q_high,
            analysis_kind="EXACT_REQUESTED_INTERVAL",
            nominal_q_low=q_low,
            nominal_q_high=q_high,
            merged_cells=0,
        )
        histogram_data = destination / f"interval_{index + 1:02d}_histogram_data.npz"
        np.savez_compressed(
            histogram_data,
            raw_positive_gaps=raw,
            normalized_crossfit_unfolded_positive_spacings=unfolded,
            nominal_quantile_bounds=np.asarray([q_low, q_high]),
        )
        row["histogram_data_path"] = str(histogram_data)
        row["histogram_data_relative_path"] = str(histogram_data.relative_to(output_root))
        row["unfolded_histogram_panel_path"] = str(destination / "crossfit_unfolded_histogram_panels.png")
        row["unfolded_histogram_panel_relative_path"] = str(
            (destination / "crossfit_unfolded_histogram_panels.png").relative_to(output_root)
        )
        row["raw_gap_histogram_panel_path"] = str(destination / "raw_gap_histogram_panels.png")
        row["raw_gap_histogram_panel_relative_path"] = str(
            (destination / "raw_gap_histogram_panels.png").relative_to(output_root)
        )
        rows.append(row)
        exact_rows.append(row)
        raw_histograms.append(raw)
        unfolded_histograms.append(unfolded)
        if row["histogram_status"] != "OK":
            adaptive = adaptive_cell_union(index, mapped, localization, specification)
            if adaptive is not None:
                adaptive_row, adaptive_raw, adaptive_unfolded = adaptive
                adaptive_path = destination / f"interval_{index + 1:02d}_adaptive_histogram_data.npz"
                np.savez_compressed(
                    adaptive_path,
                    raw_positive_gaps=adaptive_raw,
                    normalized_crossfit_unfolded_positive_spacings=adaptive_unfolded,
                    nominal_quantile_bounds=np.asarray([q_low, q_high]),
                    actual_quantile_bounds=np.asarray([adaptive_row["actual_q_low"], adaptive_row["actual_q_high"]]),
                )
                adaptive_row["histogram_data_path"] = str(adaptive_path)
                adaptive_row["histogram_data_relative_path"] = str(
                    adaptive_path.relative_to(output_root)
                )
                adaptive_figure = destination / f"interval_{index + 1:02d}_adaptive_companion_histograms.png"
                plot_adaptive_companion_histograms(
                    adaptive_row,
                    adaptive_raw,
                    adaptive_unfolded,
                    specification=specification,
                    output_path=adaptive_figure,
                )
                adaptive_row["unfolded_histogram_panel_path"] = str(adaptive_figure)
                adaptive_row["unfolded_histogram_panel_relative_path"] = str(
                    adaptive_figure.relative_to(output_root)
                )
                adaptive_row["raw_gap_histogram_panel_path"] = str(adaptive_figure)
                adaptive_row["raw_gap_histogram_panel_relative_path"] = str(
                    adaptive_figure.relative_to(output_root)
                )
                rows.append(adaptive_row)
    write_rows(destination / "extreme_edge_summary.csv", rows)
    plot_histogram_panels(
        exact_rows, raw_histograms, specification=specification,
        output_path=destination / "raw_gap_histogram_panels.png", raw=True,
    )
    plot_histogram_panels(
        exact_rows, unfolded_histograms, specification=specification,
        output_path=destination / "crossfit_unfolded_histogram_panels.png", raw=False,
    )
    plot_edge_evolution(rows, specification, destination / "edge_evolution.png")
    minimum_yield = min(
        float(row["crossfit_unfolded_histogram_sample_count"]) / ensemble_graph_count
        for row in exact_rows
    )
    manifest = {
        "analysis_version": EXTREME_EDGE_ANALYSIS_VERSION,
        "authoritative_source_analysis_version": ANALYSIS_VERSION,
        "model": specification.__dict__,
        "n": config.n,
        "m": config.m,
        "graph_count": ensemble_graph_count,
        "base_graph_count": config.num_graphs,
        "augmentation_graph_count": len(augmentation_records),
        "graph_seed_base": config.graph_seed_base,
        "requested_intervals": REQUESTED_INTERVALS,
        "quantile_coordinate": "held-out five-fold training-only ensemble empirical-CDF at gap midpoint energy",
        "historical_chung_lu_coordinate_alignment": "ensemble empirical-CDF energy coordinate",
        "graph_local_coordinate": "gap-first midpoint rank (i+1/2)/M retained per graph",
        "gap_rule": "strictly within graph and atom segment; no graph or atom bridge",
        "ratio_rule": "raw adjacent-gap ratio before unfolding",
        "unfolding": "five-fold graph-level cross-fitted empirical CDF",
        "minimum_displayed_positive_gaps": MINIMUM_HISTOGRAM_GAPS,
        "preferred_contributing_graphs": PREFERRED_CONTRIBUTING_GRAPHS,
        "g_max": G_MAX,
        "minimum_observed_positive_gap_yield_per_graph": minimum_yield,
        "reuse_classification": (
            "AUGMENT_GRAPHS_ONLY" if augmentation_records else "REPOSTPROCESS_ONLY"
        ),
        "perron_removed": True,
        "ultrafine_resolution_audit": {
            "intervals": ((0.9999, 0.99995), (0.99995, 1.0)),
            "graph_local_rank_coordinate": "finer than one-graph rank lattice and not used for bin assignment",
            "ensemble_crossfit_cdf_coordinate": "mathematically resolvable; empirical support recorded per interval",
        },
        "source_cache_checksum_table": str(destination / "source_cache_checksums.csv"),
    }
    (destination / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return rows


def plot_comparison(rows: Sequence[dict[str, object]], output_dir: Path) -> None:
    exact = [row for row in rows if row["analysis_kind"] == "EXACT_REQUESTED_INTERVAL"]
    labels = []
    for row in exact:
        if row["model_label"] not in labels:
            labels.append(row["model_label"])
    interval_labels = [f"[{low:g},{high:g})" for low, high in REQUESTED_INTERVALS]
    metrics = (
        ("graph_equal_mean_raw_r_positive_conditioned", "graph_equal_raw_r_heatmap.png", "graph-equal raw r"),
        ("crossfit_unfolded_KS_to_Poisson_minus_GOE", "signed_ks_preference_heatmap.png", "KS(Poisson)-KS(GOE)"),
    )
    for metric, filename, title in metrics:
        matrix = np.full((len(labels), len(REQUESTED_INTERVALS)), np.nan)
        for row in exact:
            model_index = labels.index(row["model_label"])
            bounds = (float(row["nominal_q_low"]), float(row["nominal_q_high"]))
            interval_index = REQUESTED_INTERVALS.index(bounds)
            value = (
                float(row["crossfit_unfolded_KS_to_Poisson"]) - float(row["crossfit_unfolded_KS_to_GOE"])
                if metric == "crossfit_unfolded_KS_to_Poisson_minus_GOE"
                else float(row[metric])
            )
            matrix[model_index, interval_index] = value
        figure, axis = plt.subplots(figsize=(12.0, 5.8))
        image = axis.imshow(matrix, aspect="auto", cmap="coolwarm")
        axis.set_xticks(np.arange(len(interval_labels)))
        axis.set_xticklabels(interval_labels, rotation=35, ha="right")
        axis.set_yticks(np.arange(len(labels)))
        axis.set_yticklabels(labels)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                axis.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8)
        axis.set_title(title)
        figure.colorbar(image, ax=axis)
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=180)
        plt.close(figure)


def write_augmentation_requirement_audit(
    base_summary_path: Path,
    output_path: Path,
    *,
    base_graph_count: int = 500,
    chunk_size: int = 5,
) -> None:
    with base_summary_path.open(newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("analysis_kind") == "EXACT_REQUESTED_INTERVAL"
        ]
    audit_rows: list[dict[str, object]] = []
    for row in rows:
        positive_gaps = int(row["crossfit_unfolded_histogram_sample_count"])
        contributors = int(row["number_of_contributing_graphs"])
        yield_per_graph = positive_gaps / base_graph_count
        if positive_gaps >= MINIMUM_HISTOGRAM_GAPS and contributors >= PREFERRED_CONTRIBUTING_GRAPHS:
            estimated_target = base_graph_count
            target = base_graph_count
            action = "REPOSTPROCESS_ONLY"
        else:
            estimated_target = (
                math.ceil(MINIMUM_HISTOGRAM_GAPS / yield_per_graph * 1.25)
                if yield_per_graph > 0.0
                else G_MAX
            )
            target = min(
                G_MAX,
                max(
                    base_graph_count,
                    chunk_size * math.ceil(estimated_target / chunk_size),
                ),
            )
            action = "AUGMENT_GRAPHS_ONLY"
        audit_rows.append(
            {
                "model_family": row["model_family"],
                "parameter_name": row["parameter_name"],
                "parameter_value": row["parameter_value"],
                "nominal_q_low": row["nominal_q_low"],
                "nominal_q_high": row["nominal_q_high"],
                "base_graph_count": base_graph_count,
                "base_positive_gap_count": positive_gaps,
                "base_contributing_graph_count": contributors,
                "empirical_positive_gap_yield_per_graph": yield_per_graph,
                "formula_target_graph_count_before_chunk_rounding_and_cap": estimated_target,
                "submitted_target_graph_count": target,
                "g_max": G_MAX,
                "reuse_classification": action,
                "quantile_resolution": "RESOLVABLE_ENSEMBLE_CROSSFIT_CDF_COORDINATE",
            }
        )
    write_rows(output_path, audit_rows)


def run_comparison(root: Path, output_root: Path, *, chunk_size: int = 5) -> Path:
    all_rows: list[dict[str, object]] = []
    for specification in DEFAULT_MODELS:
        all_rows.extend(process_model(root, output_root, specification, chunk_size=chunk_size))
    comparison = output_root / "model_comparison"
    comparison.mkdir(parents=True, exist_ok=True)
    write_rows(comparison / "extreme_edge_model_by_interval_summary.csv", all_rows)
    base_summary = comparison / "base_g500_extreme_edge_model_by_interval_summary.csv"
    if base_summary.is_file():
        write_augmentation_requirement_audit(
            base_summary,
            comparison / "augmentation_requirement_audit.csv",
            chunk_size=chunk_size,
        )
    plot_comparison(all_rows, comparison)
    manifest = {
        "analysis_version": EXTREME_EDGE_ANALYSIS_VERSION,
        "models": [specification.__dict__ for specification in DEFAULT_MODELS],
        "requested_intervals": REQUESTED_INTERVALS,
        "absolute_values_saved_in": str(comparison / "extreme_edge_model_by_interval_summary.csv"),
        "all_existing_models_reuse_classification": "REPOSTPROCESS_ONLY",
    }
    (comparison / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pa-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=5)
    args = parser.parse_args()
    print(run_comparison(args.pa_root, args.output_root, chunk_size=args.chunk_size))


if __name__ == "__main__":
    main()
