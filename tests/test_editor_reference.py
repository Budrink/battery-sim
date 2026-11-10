"""The UI parameter reference and checked-in document must describe the same fields."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "native/ui/src/editor/help/parameters.def"


class EditorReferenceTests(unittest.TestCase):
    def test_reference_matches_ui_catalog(self):
        records = re.findall(r'PARAMETER\("([^"]+)", "([^"]+)", "([^"]+)"\)', CATALOG.read_text())
        self.assertGreater(len(records), 100)
        self.assertEqual(len(records), len({key for key, _, _ in records}))
        document = (ROOT / "docs/editor-parameters.md").read_text()
        for key, title, explanation in records:
            self.assertIn(f"### {title}", document)
            self.assertIn(f"`{key}`", document)
            self.assertIn(explanation, document)

    def test_exposed_json_fields_have_help(self):
        keys = set(re.findall(r'PARAMETER\("([^"]+)"', CATALOG.read_text()))
        for folder in ("panels", "forms"):
            for path in (ROOT / "native/ui/src/editor" / folder).glob("*.cpp"):
                exposed = re.findall(r'(?:form::)?(?:number|vector|text|integer)\([^,\n]+,\s*"([^"]+)"', path.read_text())
                for key in exposed:
                    self.assertIn(key, keys, f"Missing help for {key} in {path.name}")

    def test_no_manual_property_apply_or_json_tree(self):
        root = ROOT / "native/ui/src/editor"
        for path in root.rglob("*.cpp"):
            source = path.read_text()
            self.assertNotIn("TreeNode", source, str(path))
            self.assertNotIn("EnterReturnsTrue", source, str(path))
            self.assertNotIn('Button("Apply ', source, str(path))
            self.assertNotIn('toolbar_button("Refresh', source, str(path))


if __name__ == "__main__":
    unittest.main()
