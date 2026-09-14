"""Eigenvalues-only deterministic graph augmentation for extreme PA edge bins."""

from __future__ import annotations

import json
import math
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import linalg

from .baseline import (
    MODEL_AFFINE_PA,
    MODEL_NONLINEAR_PA,
    generate_preferential_attachment_graph,
)
from .extreme_edge_quantiles import ModelSpecification
from .full_spectrum_scan import (
    MobilityEdgeProductionConfig,
    chunk_execution_metadata,
    make_seeds,
)


AUGMENTATION_ANALYSIS_VERSION = "pa_extreme_edge_eigenvalue_augmentation_v1"


def augmentation_partial_dir(output_root: Path, specification: ModelSpecification) -> Path:
    return Path(output_root) / "augmentation_caches" / specification.slug / "partials"


def augmentation_chunk_path(
    output_root: Path,
    specification: ModelSpecification,
    start: int,
    stop: int,
) -> Path:
    return augmentation_partial_dir(output_root, specification) / (
        f"extreme_edge_spectral_arrays_{start:04d}_{stop:04d}.npz"
    )


def augmentation_metadata(
    config: MobilityEdgeProductionConfig,
    specification: ModelSpecification,
    *,
    base_graph_count: int,
    target_graph_count: int,
) -> dict[str, object]:
    return {
        "analysis_version": AUGMENTATION_ANALYSIS_VERSION,
        "model_family": specification.model_family,
        "parameter_name": specification.parameter_name,
        "parameter_value": specification.parameter_value,
        "n": config.n,
        "m": config.m,
        "graph_seed_base": config.graph_seed_base,
        "base_graph_count": base_graph_count,
        "target_graph_count_when_submitted": target_graph_count,
        "perron_removed": True,
        "zero_eigenvalue_atol": config.zero_eigenvalue_atol,
        "eigenvectors_computed": False,
        "purpose": "targeted support augmentation for extreme positive-edge spacing bins",
    }


def validate_augmentation_chunk(
    config: MobilityEdgeProductionConfig,
    specification: ModelSpecification,
    output_root: Path,
    *,
    base_graph_count: int,
    target_graph_count: int,
    start: int,
    stop: int,
) -> tuple[bool, str]:
    path = augmentation_chunk_path(output_root, specification, start, stop)
    if not path.is_file():
        return False, "missing augmentation cache"
    try:
        with np.load(path, allow_pickle=False) as data:
            indices = np.asarray(data["graph_indices"], dtype=int)
            seeds = np.asarray(data["graph_seeds"], dtype=np.uint64)
            spectra = np.asarray(data["nonperron_eigenvalues"], dtype=float)
            perron = np.asarray(data["perron_eigenvalues"], dtype=float)
            connected = np.asarray(data["connected"], dtype=bool)
            metadata = json.loads(str(data["metadata_json"].item()))
    except Exception as error:
        return False, f"unreadable augmentation cache: {error}"
    expected_metadata = augmentation_metadata(
        config,
        specification,
        base_graph_count=base_graph_count,
        target_graph_count=target_graph_count,
    )
    if metadata != expected_metadata:
        return False, "augmentation metadata mismatch"
    if not np.array_equal(indices, np.arange(start, stop)):
        return False, "graph-index mismatch"
    expected_seeds = np.asarray(
        make_seeds(target_graph_count, config.graph_seed_base)[start:stop], dtype=np.uint64
    )
    if not np.array_equal(seeds, expected_seeds):
        return False, "graph-seed mismatch"
    if spectra.shape != (stop - start, config.n - 1):
        return False, "non-Perron spectrum shape mismatch"
    if perron.shape != (stop - start,) or connected.shape != (stop - start,):
        return False, "diagnostic shape mismatch"
    if not np.all(connected):
        return False, "disconnected augmentation graph"
    if not np.all(np.isfinite(spectra)) or np.any(np.diff(spectra, axis=1) < 0.0):
        return False, "non-finite or unsorted spectrum"
    if not np.all(np.isfinite(perron)):
        return False, "non-finite Perron eigenvalue"
    return True, "VALID"


def _generate_graph(
    config: MobilityEdgeProductionConfig,
    graph_seed: int,
) -> nx.Graph:
    if config.model_family == "affine_linear":
        graph, _ = generate_preferential_attachment_graph(
            n=config.n,
            m=config.m,
            seed=graph_seed,
            model=MODEL_AFFINE_PA,
            degree_ccdf_alpha=config.degree_ccdf_alpha,
        )
    else:
        graph, _ = generate_preferential_attachment_graph(
            n=config.n,
            m=config.m,
            seed=graph_seed,
            model=MODEL_NONLINEAR_PA,
            attachment_zeta=float(config.attachment_zeta),
        )
    if not nx.is_connected(graph):
        raise AssertionError("preferential-attachment augmentation graph is disconnected")
    return graph


def _nonperron_eigenvalues(graph: nx.Graph) -> tuple[np.ndarray, float]:
    matrix = nx.to_numpy_array(
        graph, nodelist=range(graph.number_of_nodes()), dtype=float
    )
    eigenvalues = linalg.eigvalsh(
        matrix, overwrite_a=True, check_finite=False, driver="evd"
    )
    perron_index = int(np.argmax(eigenvalues))
    perron = float(eigenvalues[perron_index])
    nonperron = np.delete(np.asarray(eigenvalues, dtype=float), perron_index)
    if not np.all(np.isfinite(nonperron)) or np.any(np.diff(nonperron) < 0.0):
        raise AssertionError("invalid augmentation eigenspectrum")
    return nonperron, perron


def run_augmentation_chunk(
    config: MobilityEdgeProductionConfig,
    specification: ModelSpecification,
    output_root: Path,
    *,
    base_graph_count: int,
    target_graph_count: int,
    start: int,
    stop: int,
    skip_valid: bool = True,
) -> Path:
    if not base_graph_count <= start < stop <= target_graph_count:
        raise ValueError("invalid augmentation graph range")
    valid, reason = validate_augmentation_chunk(
        config,
        specification,
        output_root,
        base_graph_count=base_graph_count,
        target_graph_count=target_graph_count,
        start=start,
        stop=stop,
    )
    if skip_valid and valid:
        print(f"skip valid augmentation [{start},{stop}): {reason}", flush=True)
        return augmentation_partial_dir(output_root, specification)
    seeds = make_seeds(target_graph_count, config.graph_seed_base)
    spectra: list[np.ndarray] = []
    perron_values: list[float] = []
    connected: list[bool] = []
    for graph_index in range(start, stop):
        graph = _generate_graph(config, seeds[graph_index])
        spectrum, perron = _nonperron_eigenvalues(graph)
        spectra.append(spectrum)
        perron_values.append(perron)
        connected.append(nx.is_connected(graph))
        print(
            f"finished augmentation {specification.parameter_name}="
            f"{specification.parameter_value:g} graph={graph_index}",
            flush=True,
        )
    directory = augmentation_partial_dir(output_root, specification)
    directory.mkdir(parents=True, exist_ok=True)
    path = augmentation_chunk_path(output_root, specification, start, stop)
    np.savez_compressed(
        path,
        graph_indices=np.arange(start, stop, dtype=int),
        graph_seeds=np.asarray(seeds[start:stop], dtype=np.uint64),
        nonperron_eigenvalues=np.asarray(spectra, dtype=float),
        perron_eigenvalues=np.asarray(perron_values, dtype=float),
        connected=np.asarray(connected, dtype=bool),
        metadata_json=np.asarray(
            json.dumps(
                augmentation_metadata(
                    config,
                    specification,
                    base_graph_count=base_graph_count,
                    target_graph_count=target_graph_count,
                ),
                sort_keys=True,
            )
        ),
        execution_json=np.asarray(
            json.dumps(chunk_execution_metadata(start, stop), sort_keys=True)
        ),
    )
    valid, reason = validate_augmentation_chunk(
        config,
        specification,
        output_root,
        base_graph_count=base_graph_count,
        target_graph_count=target_graph_count,
        start=start,
        stop=stop,
    )
    if not valid:
        raise AssertionError(f"written augmentation cache failed validation: {reason}")
    return directory
