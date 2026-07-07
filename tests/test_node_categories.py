from __future__ import annotations

import ast
import unittest
from pathlib import Path


ALLOWED_NODE_CATEGORIES = {
    "Inputs",
    "Audio",
    "Text",
    "ASR",
    "Synthesis",
    "Training",
    "Assets",
    "Dataset",
    "Testing",
}


class NodeCategoryTests(unittest.TestCase):
    def test_backend_node_categories_match_picker_groups(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "runner" / "nodes"
        categories: dict[str, str] = {}

        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                for statement in node.body:
                    if (
                        isinstance(statement, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == "CATEGORY" for target in statement.targets)
                        and isinstance(statement.value, ast.Constant)
                        and isinstance(statement.value.value, str)
                    ):
                        categories[node.name] = statement.value.value

        self.assertTrue(categories)
        self.assertEqual({category for category in categories.values() if "/" in category}, set())
        self.assertLessEqual(set(categories.values()), ALLOWED_NODE_CATEGORIES)


if __name__ == "__main__":
    unittest.main()
