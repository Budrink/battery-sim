"""Dependency boundaries: UI/run adapters must never leak into the solver."""
from pathlib import Path
import json
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native"


class NativeArchitectureTests(unittest.TestCase):
    def test_subsystems_do_not_return_to_flat_source_directories(self):
        roots = ["core/src", "core/include/battery_native", "tests", "tools",
                 "visualization/src", "visualization/include/battery_native"]
        for directory in roots:
            files = [p.name for p in (NATIVE / directory).iterdir()
                     if p.suffix in {".hpp", ".cpp", ".cu", ".cuh"}]
            self.assertEqual(files, [], f"Choose a subsystem folder, not {directory}: {files}")

    def test_build_source_paths_exist(self):
        for path in (ROOT / "cmake").glob("*.cmake"):
            for source in re.findall(r'native/[\w/.-]+\.(?:cpp|cu|hpp|cuh)(?:\.in)?\b', path.read_text()):
                self.assertTrue((ROOT / source).is_file(), f"Stale build path in {path}: {source}")

    def test_core_includes_only_core_or_external_headers(self):
        public = NATIVE / "core/include/battery_native"
        for path in (NATIVE / "core").rglob("*"):
            if path.suffix not in {".hpp", ".cpp", ".cu", ".cuh"}:
                continue
            source = path.read_text()
            for header in re.findall(r'#include\s+"battery_native/([^"]+)"', source):
                self.assertTrue((public / header).is_file(), f"Core imports an adapter: {path}: {header}")
            self.assertNotRegex(source, r'#include\s+[<"](?:GL/|GLFW/|imgui|implot|glm/)')
            self.assertNotIn("load_scenario(", source)
            self.assertNotIn("battery_ui/", source)

    def test_solver_public_api_has_no_json_or_graphics(self):
        for path in (NATIVE / "core/include").rglob("*.hpp"):
            self.assertNotIn("nlohmann", path.read_text(), str(path))
        cmake = (ROOT / "cmake/core.cmake").read_text()
        self.assertNotIn("battery_run", cmake)
        self.assertNotIn("battery_visual", cmake)
        self.assertNotIn("native/run/", cmake)
        self.assertNotIn("native/apps/", cmake)

    def test_run_adapter_has_no_viewer_dependency(self):
        for path in (NATIVE / "run").rglob("*"):
            if path.suffix in {".hpp", ".cpp"}:
                self.assertNotRegex(path.read_text(), r'#include\s+[<"](?:GL/|GLFW/|imgui|implot|glm/)')
                self.assertNotIn("visual_frame.hpp", path.read_text())
                self.assertNotIn("battery_ui/", path.read_text())

    def test_no_old_layout_references_in_active_build(self):
        paths = [ROOT / "CMakeLists.txt", *(ROOT / "cmake").glob("*.cmake")]
        for path in paths:
            self.assertNotIn("native/src/", path.read_text(), str(path))
            self.assertNotIn("native/include", path.read_text(), str(path))

    def test_job_contract_stays_outside_core_and_ui(self):
        for path in (NATIVE / "core").rglob("*"):
            if path.suffix in {".hpp", ".cpp", ".cu", ".cuh"}:
                self.assertNotIn("battery_job/", path.read_text(), str(path))
        for path in (NATIVE / "job").rglob("*"):
            if path.suffix in {".hpp", ".cpp"}:
                self.assertNotRegex(path.read_text(), r'#include\s+[<"](?:GL/|GLFW/|imgui|implot|glm/)')
                self.assertNotIn("battery_ui/", path.read_text())
        build = (ROOT / "cmake/job.cmake").read_text()
        self.assertNotIn("battery_renderer", build)
        self.assertNotIn("battery_visual", build)

    def test_viewer_is_only_a_module_entry_point(self):
        source = (NATIVE / "apps/viewer/main.cpp").read_text()
        self.assertIn("battery_ui/application.hpp", source)
        self.assertIn("battery::ui::run(argc, argv)", source)
        self.assertNotIn("Simulation", source)
        self.assertNotIn("ImGui", source)
        header = (NATIVE / "ui/include/battery_ui/application.hpp").read_text()
        self.assertNotIn("#include", header)

    def test_minimal_presets_disable_optional_layers(self):
        presets = {p["name"]: p for p in json.loads((ROOT / "CMakePresets.json").read_text())["configurePresets"]}
        for name in ("core", "server"):
            options = presets[name]["cacheVariables"]
            for key in ("BATTERY_SIM_BUILD_VIEWER", "BATTERY_SIM_BUILD_TOOLS", "BUILD_TESTING"):
                self.assertEqual(options[key], "OFF")
        self.assertEqual(presets["core"]["cacheVariables"]["BATTERY_SIM_BUILD_RUNTIME"], "OFF")
        self.assertEqual(presets["server"]["cacheVariables"]["BATTERY_SIM_BUILD_RUNTIME"], "ON")


if __name__ == "__main__":
    unittest.main()
