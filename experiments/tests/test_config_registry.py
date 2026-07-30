from __future__ import annotations

import unittest

from experiments.config import compose_config
from experiments.coupling import anchor_selection, grouping, parameter_blocks, residuals, solvers  # noqa: F401
from experiments.registry import REGISTRIES, Registry


class ConfigRegistryTest(unittest.TestCase):
    def test_fragment_and_lowercase_boolean(self):
        config = compose_config(["grouping=independent", "benchmark.fixed_topology=false", "grouping.group_size=4"])
        self.assertEqual(config["grouping"]["name"], "independent")
        self.assertEqual(config["grouping"]["group_size"], 4)
        self.assertIs(config["benchmark"]["fixed_topology"], False)

    def test_registry_duplicate_and_unknown(self):
        registry = Registry("test")
        registry.register("value")(lambda: 3)
        self.assertEqual(registry.create("value"), 3)
        with self.assertRaises(KeyError):
            registry.register("value")(lambda: 4)
        with self.assertRaises(KeyError):
            registry.create("missing")

    def test_expected_extension_points_registered(self):
        self.assertIn("random_contributor", REGISTRIES["anchor_selection"].names())
        self.assertIn("position", REGISTRIES["parameter_block"].names())
        self.assertIn("visible_overlap_knn", REGISTRIES["grouping"].names())
        self.assertIn("group_lm", REGISTRIES["solver"].names())
        with self.assertRaises(NotImplementedError):
            REGISTRIES["parameter_block"].create("opacity")

    def test_anchor_selection_fragment(self):
        config = compose_config(["anchor_selection=random_contributor", "anchor_selection.probe_count=7"])
        self.assertEqual(config["anchor_selection"]["name"], "random_contributor")
        self.assertEqual(config["anchor_selection"]["probe_count"], 7)


if __name__ == "__main__":
    unittest.main()
