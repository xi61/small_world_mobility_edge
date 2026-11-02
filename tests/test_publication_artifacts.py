"""Checks for the October import and portable scientific utilities."""
import csv
import hashlib
import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PublicationArtifactTests(unittest.TestCase):
    def test_original_numerical_tables_and_images_are_unchanged(self):
        with (ROOT / "docs/october_import_provenance.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        checked = 0
        for row in rows:
            path = ROOT / row["path"]
            if not row["path"].startswith("docs/results/") or path.suffix not in {".csv", ".png"}:
                continue
            if path.name == "noise_completion.csv":  # Only a private path prefix was replaced.
                continue
            with self.subTest(path=row["path"]):
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row["original_sha256"])
            checked += 1
        self.assertGreater(checked, 100)

    def test_cluster_path_scan_covers_one_letter_shard(self):
        validator = load_tool("validate_release")
        pattern = validator.PRIVATE_PATTERNS["named cluster account"]
        self.assertIsNotNone(pattern.search("/u/" + "scratch/q/example_account/data"))
        self.assertIsNone(pattern.search("<HOFFMAN_SCRATCH>/data"))

    def test_audit_reconstructs_star_and_exact_residual(self):
        audit = load_tool("audit_pa_noncentral_targets")
        matrix, checks = audit.build_adjacency(5, np.zeros(4, dtype=int), np.arange(1, 5), np.array([4, 1, 1, 1, 1]))
        psi = np.array([1 / np.sqrt(2)] + [1 / np.sqrt(8)] * 4)
        self.assertTrue(checks["connected"])
        self.assertLess(audit.sparse_residual(matrix, psi, 2.0), 1e-14)
        self.assertEqual(audit.bfs_distances(matrix, 1).tolist(), [1, 0, 2, 2, 2])

    def test_audit_rejects_duplicate_edge(self):
        audit = load_tool("audit_pa_noncentral_targets")
        with self.assertRaisesRegex(AssertionError, "duplicate"):
            audit.build_adjacency(2, np.array([0, 1]), np.array([1, 0]), np.array([2, 2]))

    def test_tie_aware_quantile_edges_do_not_split_atom(self):
        audit = load_tool("audit_pa_noncentral_targets")
        audit.validate_tie_fixture()
        values = np.array([-1., 0., 0., 0., 0., 1., 2.])
        edges = audit.tie_aware_quantile_edges(values, 6)
        counts, _ = np.histogram(values, edges)
        self.assertEqual(int(counts.sum()), len(values))
        self.assertTrue(np.all(np.diff(edges) > 0))


if __name__ == "__main__":
    unittest.main()
