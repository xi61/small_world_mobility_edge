#!/usr/bin/env python3
"""Run a small random-3-regular Anderson central-spectrum benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import linalg, sparse

from small_world_mobility_edge.random_schrodinger.anderson_rrg_benchmark import (
    _central_interval_solve,
    _central_tables,
    _generate_connected_rrg,
)
from small_world_mobility_edge.random_schrodinger.anderson_utils import (
    atomic_savez,
    atomic_write_csv,
    atomic_write_json,
    validate_selected_eigenpairs,
)
from small_world_mobility_edge.random_schrodinger.common import graph_to_csr, utc_now


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--graph-seed", type=int, default=5512)
    parser.add_argument("--potential-seed", type=int, default=887)
    parser.add_argument("--amplitude", type=float, default=9.0)
    parser.add_argument("--target-half-width", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.n * args.degree % 2 or args.degree >= args.n:
        raise ValueError("n*degree must be even and degree must be smaller than n")

    graph, attempts = _generate_connected_rrg(args.n, args.degree, args.graph_seed)
    adjacency = graph_to_csr(graph, nodelist=range(args.n))
    potential = np.random.Generator(np.random.PCG64(args.potential_seed)).uniform(-1.0, 1.0, args.n)
    dense = adjacency.toarray()
    dense[np.diag_indices_from(dense)] = args.amplitude * potential
    values, vectors, coverage = _central_interval_solve(
        dense,
        target_half_width=args.target_half_width,
        initial_half_width=max(0.4, 2.0 * args.target_half_width),
    )
    validation = validate_selected_eigenpairs(sparse.csr_matrix(dense), values, vectors)
    full_values = linalg.eigh(dense.copy(), eigvals_only=True, check_finite=False, driver="evd")
    reference = full_values[np.abs(full_values) <= args.target_half_width]
    selected = values[np.abs(values) <= args.target_half_width]
    if len(reference) != len(selected):
        raise AssertionError("central interval omitted a target eigenvalue")
    maximum_match_error = float(np.max(np.abs(reference - selected))) if len(reference) else 0.0
    if maximum_match_error > 1e-10:
        raise AssertionError("central interval eigenvalues disagree with the full spectrum")

    prefix = {
        "n": args.n,
        "degree": args.degree,
        "graph_seed": args.graph_seed,
        "potential_seed": args.potential_seed,
        "a": args.amplitude,
    }
    rows, gaps, summary = _central_tables(
        values,
        vectors,
        target_half_width=args.target_half_width,
        prefix=prefix,
        graph=graph,
        potential=potential,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_savez(
        args.output_dir / "central_eigenpairs.npz",
        eigenvalues=values,
        eigenvectors=vectors,
        potential=potential,
    )
    atomic_write_csv(args.output_dir / "central_eigenpair_metrics.csv", rows)
    atomic_write_csv(args.output_dir / "central_gap_triplets.csv", gaps)
    atomic_write_json(
        args.output_dir / "run_manifest.json",
        {
            "status": "COMPLETE",
            "scope": "small deterministic integration check, not production",
            **prefix,
            "operator": "H(a)=A+a*diag(xi), unit hopping",
            "potential_law": "iid Uniform[-1,1) from NumPy PCG64",
            "generation_attempts": attempts,
            "coverage": coverage,
            "validation": validation,
            "full_reference_target_count": int(len(reference)),
            "maximum_full_reference_match_error": maximum_match_error,
            "summary": summary,
            "finished_at_utc": utc_now(),
        },
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
