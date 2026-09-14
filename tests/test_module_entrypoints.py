from __future__ import annotations

import importlib
import unittest


class ModuleEntrypointTests(unittest.TestCase):
    def test_regular_experiment_modules_import_from_installed_package(self) -> None:
        for module in (
            "small_world_mobility_edge.regular_baseline_experiments",
            "small_world_mobility_edge.unfolding_comparison",
            "small_world_mobility_edge.local_spacing_by_energy",
        ):
            imported = importlib.import_module(module)
            self.assertTrue(callable(imported.main))


if __name__ == "__main__":
    unittest.main()
