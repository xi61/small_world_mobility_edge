from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from small_world_mobility_edge.model_smoke import (
    MODEL_CHUNG_LU,
    MODEL_ER,
    MODEL_FIXED_DEGREE,
    MODEL_REGULAR,
    run_model,
)


class ModelSmokeTests(unittest.TestCase):
    def test_seeded_non_pa_smoke_workflows_write_normalized_diagnostics(self) -> None:
        with TemporaryDirectory() as directory:
            output_root = Path(directory)
            for model in (MODEL_REGULAR, MODEL_ER, MODEL_CHUNG_LU, MODEL_FIXED_DEGREE):
                summary = run_model(
                    model=model,
                    n=80,
                    degree=6,
                    tau=2.5,
                    num_graphs=2,
                    seed=101,
                    component_mode="full",
                    swaps_per_edge=1,
                    output_root=output_root,
                )
                self.assertAlmostEqual(summary["histogram_density_integral"], 1.0, places=8)
                self.assertAlmostEqual(summary["local_count_density_integral"], 1.0, places=8)
                self.assertGreater(summary["raw_gap_count"], 0)
                self.assertGreater(summary["gap_ratio_count"], 0)
                self.assertTrue(summary["within_graph_spacing_construction"])
                if model in {MODEL_REGULAR, MODEL_FIXED_DEGREE}:
                    self.assertTrue(summary["degree_check"])
                self.assertTrue((output_root / model / "summary.json").is_file())
                self.assertTrue((output_root / model / "density_diagnostics.png").is_file())


if __name__ == "__main__":
    unittest.main()
