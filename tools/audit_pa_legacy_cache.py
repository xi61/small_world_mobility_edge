#!/usr/bin/env python3
"""Audit the legacy affine-PA full-spectrum cache without modifying it."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


NUM_GRAPHS = 500
CHUNK_SIZE = 5
N = 5000
M = 10
PARAMETERS = {
    "1.8": ("n_5000_m_10_preference_tau_1p8_avgdeg_19p978", 27282),
    "1.5": ("n_5000_m_10_preference_tau_1p5_avgdeg_19p978", 27352),
    "1.05": ("n_5000_m_10_preference_tau_1p05_avgdeg_19p978", 27307),
}


def make_seeds(base_seed: int) -> list[int]:
    rng = np.random.default_rng(base_seed)
    values = rng.integers(0, 2**32 - 1, size=NUM_GRAPHS, dtype=np.uint32)
    return [int(value) for value in values]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def audit_parameter(source_root: Path, alpha: str) -> dict[str, object]:
    dirname, seed_base = PARAMETERS[alpha]
    parameter_dir = source_root / dirname
    partial_dir = parameter_dir / "partials"
    expected_seeds = make_seeds(seed_base)
    file_records: list[dict[str, object]] = []
    covered: list[int] = []
    errors: list[str] = []

    for start in range(0, NUM_GRAPHS, CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, NUM_GRAPHS)
        suffix = f"{start:04d}_{stop:04d}"
        paths = {
            "eigenvector_rows": partial_dir / f"eigenvector_graphs_{suffix}.csv.gz",
            "fragment_rows": partial_dir / f"fragment_graphs_{suffix}.csv.gz",
            "spectral_arrays": partial_dir / f"spectral_arrays_{suffix}.npz",
        }
        for kind, path in paths.items():
            if not path.is_file():
                errors.append(f"missing {path}")
                continue
            file_records.append(
                {
                    "kind": kind,
                    "relative_path": str(path.relative_to(source_root)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
        if any(not path.is_file() for path in paths.values()):
            continue

        with np.load(paths["spectral_arrays"], allow_pickle=False) as data:
            indices = np.asarray(data["graph_indices"], dtype=int)
            seeds = np.asarray(data["graph_seeds"], dtype=np.uint64)
            values = np.asarray(data["nonperron_eigenvalues"], dtype=float)
            decay_sum = np.asarray(data["coordinate_decay_sum"], dtype=float)
            decay_sq = np.asarray(data["coordinate_decay_sum_squares"], dtype=float)
            graph_count = int(data["graph_count"])
        expected_indices = np.arange(start, stop, dtype=int)
        expected_chunk_seeds = np.asarray(expected_seeds[start:stop], dtype=np.uint64)
        if not np.array_equal(indices, expected_indices):
            errors.append(f"wrong graph indices in {paths['spectral_arrays']}")
        if not np.array_equal(seeds, expected_chunk_seeds):
            errors.append(f"wrong graph seeds in {paths['spectral_arrays']}")
        if values.shape != (stop - start, N - 1):
            errors.append(f"wrong eigenvalue shape {values.shape} in {paths['spectral_arrays']}")
        if decay_sum.shape != (500, 200) or decay_sq.shape != (500, 200):
            errors.append(f"wrong coordinate-decay shape in {paths['spectral_arrays']}")
        if graph_count != stop - start:
            errors.append(f"wrong graph_count in {paths['spectral_arrays']}")
        if not np.all(np.isfinite(values)) or np.any(np.diff(values, axis=1) < -1e-12):
            errors.append(f"invalid or unsorted eigenvalues in {paths['spectral_arrays']}")

        fragments = csv_rows(paths["fragment_rows"])
        localization = csv_rows(paths["eigenvector_rows"])
        if len(fragments) != stop - start:
            errors.append(f"wrong fragment row count in {paths['fragment_rows']}")
        if len(localization) != (stop - start) * 500:
            errors.append(f"wrong localization row count in {paths['eigenvector_rows']}")
        for rows, label in ((fragments, "fragment"), (localization, "localization")):
            row_indices = sorted({int(row["graph_index"]) for row in rows})
            if row_indices != list(range(start, stop)):
                errors.append(f"wrong {label} graph coverage for chunk {suffix}")
            for row in rows:
                graph_index = int(row["graph_index"])
                if int(row["graph_seed"]) != expected_seeds[graph_index]:
                    errors.append(f"wrong {label} seed for graph {graph_index}")
                    break
                recorded_alpha = float(row.get("attachment_tail_exponent", "nan"))
                if not np.isclose(recorded_alpha, float(alpha), atol=1e-12, rtol=0.0):
                    errors.append(f"wrong {label} alpha for graph {graph_index}")
                    break
        covered.extend(indices.tolist())

    complete_coverage = sorted(covered) == list(range(NUM_GRAPHS))
    if not complete_coverage:
        errors.append("spectral graph-index coverage is not exactly 0..499")
    return {
        "analysis_version": "pa_legacy_cache_audit_v1",
        "canonical_parameter": {"name": "degree_ccdf_alpha", "value": float(alpha)},
        "legacy_parameter": {"name": "attachment_tail_exponent", "value": float(alpha)},
        "legacy_directory_name": dirname,
        "n": N,
        "m": M,
        "num_graphs": NUM_GRAPHS,
        "chunk_size": CHUNK_SIZE,
        "graph_seed_base": seed_base,
        "graph_seed_rule": "numpy.default_rng(base).integers(0, 2**32-1, dtype=uint32)",
        "graph_indices": list(range(NUM_GRAPHS)),
        "graph_seeds": expected_seeds,
        "source_cache": str(parameter_dir),
        "source_cache_read_only": True,
        "spectral_array_semantics": "sorted full non-Perron adjacency eigenvalues; exact zeros retained",
        "legacy_gap_assignment": "select eigenvalues by ensemble-CDF energy window, then difference within graph",
        "legacy_unfolding": "same-ensemble empirical CDF",
        "new_postprocessing_required": "gap-first raw-r and five-fold graph-level cross-fitted unfolding",
        "complete_graph_index_coverage": complete_coverage,
        "expected_files": 3 * (NUM_GRAPHS // CHUNK_SIZE),
        "observed_files": len(file_records),
        "total_size_bytes": sum(int(item["size_bytes"]) for item in file_records),
        "files": file_records,
        "errors": errors,
        "status": "VALID" if not errors else "INVALID",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for alpha in PARAMETERS:
        record = audit_parameter(args.source_root, alpha)
        records.append(record)
        slug = alpha.replace(".", "p")
        path = args.output_dir / f"degree_ccdf_alpha_{slug}" / "source_cache.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(f"alpha={alpha} status={record['status']} files={record['observed_files']}")
    index = {
        "analysis_version": "pa_legacy_cache_audit_v1",
        "source_root": str(args.source_root),
        "all_valid": all(record["status"] == "VALID" for record in records),
        "parameters": [record["canonical_parameter"] for record in records],
        "total_size_bytes": sum(int(record["total_size_bytes"]) for record in records),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / "cache_index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )
    if not index["all_valid"]:
        raise SystemExit("legacy cache audit failed; inspect source_cache.json errors")


if __name__ == "__main__":
    main()
