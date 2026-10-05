from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import linalg, sparse

from small_world_mobility_edge.preferential_attachment.linear_pa_two_part import _edge_hash
from small_world_mobility_edge.random_schrodinger.common import graph_to_csr, save_graph_cache, write_json
from small_world_mobility_edge.random_schrodinger.pa_diagonal_uniform_noise import (
    analyze_operator,
    construct_operator,
    draw_potential,
    first_edge_rows,
    localization_rows,
    multiplicity_labels,
    potential_seed,
    prepare_case,
    run_operator,
    spacing_rows,
    verify_eigensystem,
    window_summary_rows,
    zero_subspace_evidence,
)
from small_world_mobility_edge.random_schrodinger.pa_diagonal_uniform_noise_reporting import (
    aggregate_case,
    aggregate_global,
)


def test_config() -> dict[str, object]:
    return {
        "experiment_id": "test_raw_uniform",
        "degree_ccdf_alpha_grid": [2.0, 2.5],
        "m": 2,
        "amplitudes": [0.0, 0.5, 2.0],
        "potential_base_seed": 30720260924,
        "potential_seed_derivation": "test deterministic SeedSequence",
        "potential_rng": "PCG64 uniform(-1,1)",
        "solver": {
            "dense_driver": "evd",
            "residual_block_size": 4,
            "orthogonality_sample_size": 12,
            "relative_eigenpair_residual_tolerance": 1e-8,
            "identity_relative_tolerance": 5e-10,
            "multiplicity_tolerance_factor": 64.0,
        },
        "analysis": {
            "support_mass_fractions": [0.5, 0.8, 0.9, 0.95],
            "coordinate_counts": [10, 15, 20, 30, 50],
            "detailed_decreasing_algebraic_ranks": list(range(1, 12)),
            "quantile_edges": [0.0, 0.25, 0.5, 0.75, 1.0],
            "cumulative_upper_counts": {"12": [10, 12], "80": [10, 25, 50, 80]},
            "primary_window_membership_policy": "strict_internal",
            "secondary_window_membership_policy": "center_with_external_guard",
            "energy_bin_count": 10,
            "raw_gap_histogram_bin_count": 8,
            "energy_metric_bin_count": 8,
        },
    }


class PAUniformNoiseTests(unittest.TestCase):
    def test_potential_bounds_replay_distinct_cases_and_covariance(self) -> None:
        config = test_config()
        seed, xi = draw_potential(config, 4000, 2.0)
        seed_again, replay = draw_potential(config, 4000, 2.0)
        self.assertEqual(seed, seed_again)
        np.testing.assert_array_equal(xi, replay)
        self.assertTrue(np.all(xi >= -1.0) and np.all(xi < 1.0))
        self.assertNotEqual(seed, potential_seed(config, 4000, 2.5))
        rng = np.random.Generator(np.random.PCG64(seed))
        samples = rng.uniform(-1.0, 1.0, size=(20000, 4))
        covariance = np.cov(samples, rowvar=False, bias=True)
        self.assertLess(np.max(np.abs(covariance - np.diag(np.diag(covariance)))), 0.015)
        self.assertLess(np.max(np.abs(np.diag(covariance) - 1.0 / 3.0)), 0.015)

    def test_operator_fresh_copy_bounds_offdiagonal_and_trace_identities(self) -> None:
        graph = nx.path_graph(5)
        adjacency = graph_to_csr(graph, nodelist=range(5))
        xi = np.asarray([-0.9, -0.2, 0.0, 0.25, 0.8])
        first, first_checks = construct_operator(adjacency, xi, 2.0)
        second, second_checks = construct_operator(adjacency, xi, 0.5)
        np.testing.assert_allclose(np.diag(first), 2.0 * xi)
        np.testing.assert_allclose(np.diag(second), 0.5 * xi)
        self.assertTrue(first_checks["offdiagonal_unchanged_pass"])
        self.assertTrue(second_checks["offdiagonal_unchanged_pass"])
        values = np.linalg.eigvalsh(first)
        self.assertAlmostEqual(float(np.sum(values)), float(np.sum(2.0 * xi)), places=11)
        self.assertAlmostEqual(float(np.sum(values * values)), 2.0 * graph.number_of_edges() + float(np.sum((2.0 * xi) ** 2)), places=10)

    def test_analytical_two_by_two_and_h_based_residual(self) -> None:
        adjacency = sparse.csr_matrix(np.asarray([[0.0, 1.0], [1.0, 0.0]]))
        xi = np.asarray([-0.5, 0.5])
        dense, _ = construct_operator(adjacency, xi, 2.0)
        values, vectors = linalg.eigh(dense)
        expected = np.asarray([-math.sqrt(2.0), math.sqrt(2.0)])
        np.testing.assert_allclose(values, expected, atol=1e-13)
        summary, absolute, relative = verify_eigensystem(adjacency + sparse.diags(2.0 * xi), values, vectors, residual_block_size=2, orthogonality_sample_size=2)
        self.assertLess(summary["maximum_relative_residual"], 1e-12)
        self.assertLess(float(np.max(absolute)), 1e-12)
        self.assertLess(float(np.max(relative)), 1e-12)

    def test_localization_mass_support_and_top15(self) -> None:
        graph = nx.path_graph(20)
        adjacency = graph_to_csr(graph, nodelist=range(20))
        vector = np.arange(1.0, 21.0)
        vector /= np.linalg.norm(vector)
        seed_matrix = np.eye(20)
        seed_matrix[:, 0] = vector
        vectors, _ = np.linalg.qr(seed_matrix)
        self.assertGreater(abs(float(np.dot(vectors[:, 0], vector))), 0.999999)
        values = np.linspace(-2.0, 2.0, 20)
        rows = localization_rows(
            graph=graph,
            adjacency=adjacency,
            values=values,
            vectors=vectors,
            absolute_residuals=np.zeros(20),
            relative_residuals=np.zeros(20),
            potential=np.zeros(20),
            atom_mask=np.zeros(20, dtype=bool),
            multiplicity_group=np.arange(20),
            multiplicity=np.ones(20, dtype=int),
            prefix={},
            support_fractions=[0.5, 0.8, 0.9, 0.95],
            coordinate_counts=[10, 15, 20, 30, 50],
        )
        row = rows[0]
        self.assertIn("top_15_mass", row)
        self.assertLessEqual(row["top_10_mass"], row["top_15_mass"])
        self.assertAlmostEqual(row["top_20_mass"], 1.0, places=12)
        self.assertLessEqual(row["support_size_50"], row["support_size_80"])
        self.assertGreaterEqual(row["ipr"], 1.0 / 20.0)
        self.assertLessEqual(row["ipr"], 1.0)

    def test_gap_indices_two_policies_ten_gaps_and_no_bridging(self) -> None:
        values = np.asarray([-3.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 4.0, 7.0, 11.0, 16.0])
        atom = np.asarray([False, False, False, True, True, True, False, False, False, False, False, False])
        gaps, ratios = spacing_rows(values, atom_mask_ascending=atom, resolution=1e-12, prefix={})
        self.assertEqual(len(gaps), 11)
        self.assertEqual(len(ratios), 10)
        self.assertTrue(any(row["clean_zero_atom_triplet_excluded"] for row in ratios))
        self.assertTrue(all(row["triplet_is_consecutive_original_spectrum"] for row in ratios))
        eigen_rows, first_gaps = first_edge_rows(values, gaps, {})
        self.assertEqual(sum(row["rank_convention"] == "literal_including_rank_one" for row in first_gaps), 10)
        self.assertEqual(sum(row["rank_convention"] == "rank_one_excluded" for row in first_gaps), 10)
        self.assertEqual(len(eigen_rows), 20)
        fake_localization = [
            {
                "energy": float(value), "individual_state_primary_available": not atom[index],
                "clean_zero_atom_member": bool(atom[index]), "basis_dependent_solver_vector": bool(atom[index]),
                "n_times_ipr": 1.0, "support_size_80": 4,
            }
            for index, value in enumerate(values)
        ]
        windows = [("fixture", "literal", {"window_id": "ranks_2_5", "first_decreasing_rank": 2, "last_decreasing_rank": 5, "state_count": 4})]
        summaries, _ = window_summary_rows(n=12, localization=fake_localization, gaps=gaps, ratios=ratios, windows=windows, prefix={})
        strict = next(row for row in summaries if row["membership_policy"] == "strict_internal")
        guard = next(row for row in summaries if row["membership_policy"] == "center_with_external_guard")
        self.assertEqual(strict["candidate_ratio_count"], 2)
        self.assertEqual(guard["candidate_ratio_count"], 4)

    def test_clean_zero_atom_but_not_positive_near_zero_mask(self) -> None:
        values = np.asarray([-2.0, 0.0, 0.0, 0.0, 3.0])
        vectors = np.eye(5)
        operator = sparse.diags(values)
        with tempfile.TemporaryDirectory() as temp:
            evidence, mask = zero_subspace_evidence(values, vectors, operator, enabled=True, output_dir=Path(temp))
            self.assertTrue(evidence["active"])
            self.assertEqual(evidence["dimension"], 3)
            self.assertEqual(int(np.count_nonzero(mask)), 3)
            positive, positive_mask = zero_subspace_evidence(values, vectors, operator, enabled=False, output_dir=Path(temp) / "positive")
            self.assertFalse(positive["active"])
            self.assertEqual(int(np.count_nonzero(positive_mask)), 0)
        _, multiplicity, _ = multiplicity_labels(values, 64.0)
        self.assertEqual(int(np.max(multiplicity)), 3)

    def test_induced_distance_can_be_infinite_while_full_distance_is_finite(self) -> None:
        graph = nx.path_graph(3)
        induced = graph.subgraph([0, 2])
        self.assertFalse(nx.has_path(induced, 0, 2))
        self.assertEqual(nx.shortest_path_length(graph, 0, 2), 2)

    def _make_clean_fixture(self, root: Path) -> tuple[dict[str, object], Path]:
        config = test_config()
        config["degree_ccdf_alpha_grid"] = [2.0]
        config["amplitudes"] = [0.0]
        clean = root / "clean/N_12/alpha_2"
        (clean / "arrays").mkdir(parents=True)
        graph = nx.cycle_graph(12)
        adjacency = graph_to_csr(graph, nodelist=range(12))
        values, vectors = linalg.eigh(adjacency.toarray())
        residual = adjacency.dot(vectors) - vectors * values[None, :]
        absolute = np.linalg.norm(residual, axis=0)
        relative = absolute / np.maximum(1.0, np.abs(values))
        graph_hash = _edge_hash(graph)
        resolved = {
            "case_id": "fixture_N12_alpha2", "n": 12, "m": 2, "degree_ccdf_alpha": 2.0,
            "attachment_offset": 0.0, "operator": "unscaled simple undirected adjacency in vertex order 0..N-1", "graph_seed": 123,
        }
        save_graph_cache(clean / "arrays/graph_edges.npz", graph, np.arange(12), resolved)
        np.save(clean / "arrays/full_eigenvalues_ascending.npy", values)
        np.save(clean / "arrays/full_eigenvectors_ascending.npy", vectors)
        np.save(clean / "arrays/full_absolute_residuals.npy", absolute)
        np.save(clean / "arrays/full_relative_residuals.npy", relative)
        write_json(clean / "resolved_case.json", resolved)
        write_json(clean / "graph_cache_manifest.json", {**resolved, "graph_sha256_edges": graph_hash, "simple_undirected": True})
        write_json(clean / "part1_manifest.json", {"status": "COMPLETE"})
        write_json(clean / "spectral_cache_manifest.json", {"status": "COMPLETE", "maximum_relative_residual": float(np.max(relative))})
        return config, root / "clean"

    def test_prepare_restart_identity_baseline_dedup_and_output_completeness(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, clean_root = self._make_clean_fixture(root)
            run_root = root / "run"
            first = prepare_case(config=config, run_root=run_root, clean_root=clean_root, n=12, alpha=2.0, source_identity="fixture-source")
            manifest_before = (first / "case_manifest.json").read_bytes()
            second = prepare_case(config=config, run_root=run_root, clean_root=clean_root, n=12, alpha=2.0, source_identity="fixture-source")
            self.assertEqual(first, second)
            self.assertEqual(manifest_before, (second / "case_manifest.json").read_bytes())
            output = run_operator(config=config, run_root=run_root, n=12, alpha=2.0, amplitude=0.0, source_identity="fixture-source")
            manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "COMPLETE")
            self.assertEqual(manifest["spectral_cache"]["mode"], "reused_validated_clean_eigensystem_without_duplicate_storage")
            self.assertFalse((output / "arrays/full_eigenvectors_ascending.npy").exists())
            self.assertTrue((output / "tables/full_spectrum_localization.csv.gz").is_file())
            self.assertEqual(manifest["analysis_summary"]["localization_row_count"], 12)

    def test_reporting_smoke_and_completion_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, clean_root = self._make_clean_fixture(root)
            config["sizes"] = [12]
            run_root = root / "run"
            prepare_case(config=config, run_root=run_root, clean_root=clean_root, n=12, alpha=2.0, source_identity="fixture-source")
            run_operator(config=config, run_root=run_root, n=12, alpha=2.0, amplitude=0.0, source_identity="fixture-source")
            case_report = aggregate_case(config=config, run_root=run_root, n=12, alpha=2.0, source_identity="fixture-source")
            self.assertTrue((case_report / "CASE_REPORT.md").is_file())
            self.assertEqual(len(list((case_report / "figures").glob("*.png"))), 7)
            global_report = aggregate_global(config=config, run_root=run_root, source_identity="fixture-source")
            manifest = json.loads((global_report / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "COMPLETE")
            self.assertEqual(manifest["completed_operator_count"], 1)
            self.assertTrue((global_report / "tables/completion_matrix_14x8.csv").is_file())


if __name__ == "__main__":
    unittest.main()
