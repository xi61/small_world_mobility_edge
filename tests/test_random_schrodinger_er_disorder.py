import json
import tempfile
import unittest
from pathlib import Path

import networkx as nx
import numpy as np

from small_world_mobility_edge.random_schrodinger.common import (
    graph_to_csr,
    save_graph_cache,
)
from small_world_mobility_edge.random_schrodinger.disorder import (
    _gaps_and_ratios,
    run_pa_disorder_sweep,
)
from small_world_mobility_edge.random_schrodinger.er_benchmark import (
    run_er_figure10_benchmark,
)


class RandomSchrodingerERDisorderTests(unittest.TestCase):
    def test_gap_ratios_preserve_exact_zero_gaps(self):
        gaps, ratios = _gaps_and_ratios(np.array([0.0, 1.0, 1.0, 3.0]))
        self.assertTrue(np.array_equal(gaps, np.array([1.0, 0.0, 2.0])))
        self.assertAlmostEqual(ratios[1], 0.0)
        self.assertAlmostEqual(ratios[2], 0.0)

    def test_small_er_run_has_full_spectral_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "er"
            run_er_figure10_benchmark(
                n_requested=80,
                graph_seed=17,
                output_dir=output,
                project_root=Path.cwd(),
                run_label="unit_test_pilot",
                edge_vector_count=8,
                interior_vector_count=5,
            )
            manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "COMPLETE")
            self.assertTrue(manifest["full_spectrum_coverage"])
            self.assertEqual(manifest["eigenpair_count"], manifest["analyzed_n"])

    def test_small_disorder_sweep_checkpoints_both_ratios(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            (source / "tables").mkdir(parents=True)
            (source / "spectrum").mkdir(parents=True)
            graph = nx.cycle_graph(120)
            graph.add_edges_from((vertex, (vertex + 17) % 120) for vertex in range(60))
            adjacency = graph_to_csr(graph, nodelist=range(120))
            values, vectors = np.linalg.eigh(adjacency.toarray())
            save_graph_cache(
                source / "tables" / "graph_realization.npz",
                graph,
                np.arange(120),
                {"fixture": "deterministic_test_graph"},
            )
            np.savez_compressed(
                source / "spectrum" / "single_graph_spectral_arrays.npz",
                eigenvalues=values,
                ipr=np.sum(np.abs(vectors) ** 4, axis=0),
            )
            np.savez_compressed(
                source / "tables" / "selected_eigenvectors.npz",
                global_indices=np.arange(120),
                eigenvalues=values,
                eigenvectors=vectors,
            )
            (source / "run_manifest.json").write_text(
                json.dumps({"status": "COMPLETE", "graph_seed": 7}) + "\n"
            )
            output = root / "sweep"
            run_pa_disorder_sweep(
                source_dir=source,
                output_dir=output,
                project_root=Path.cwd(),
                potential_seed=19,
                amplitude_ratios=(0.0, 0.1),
            )
            manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "COMPLETE")
            self.assertEqual(manifest["completed_ratio_count"], 2)
            for slug in ("0", "0p1"):
                ratio_manifest = json.loads(
                    (output / ("ratio_" + slug) / "run_manifest.json").read_text()
                )
                self.assertEqual(ratio_manifest["status"], "COMPLETE")


if __name__ == "__main__":
    unittest.main()
