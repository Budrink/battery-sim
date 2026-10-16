"""Exercise native pressure-path validation, without running a calibration."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('mass_paths', ROOT/'scripts/upv_mass_paths.py')
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)
PROBE = Path(os.environ.get('BATTERY_CASING_PROBE', '/tmp/battery-sim-cuda-v1/battery_sim_casing_probe'))


@unittest.skipUnless(PROBE.exists(), 'Native casing probe is not built')
class NativeMassPaths(unittest.TestCase):
    def test_invalid_internal_connections(self):
        scene, _ = model.inputs('no-cooling')
        scene = model.attach(scene, model.initial())
        cases = [({'to': 'inside-C2-2'}, 'Nonadjacent internal pore path'),
                 ({'area_m2': 1.}, 'Internal pore path exceeds common face'),
                 ({'to': 'gap-TIM2-0'}, 'Internal pore path requires adjacent'),
                 ({'opening_pressure_pa': 1.}, 'Internal pore path requires adjacent')]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'invalid.json'
            for change, message in cases:
                with self.subTest(change=change):
                    candidate = copy.deepcopy(scene)
                    candidate['gas_network']['channels'][0].update(change)
                    path.write_text(json.dumps(candidate))
                    run = subprocess.run([str(PROBE), str(path)], input='{}\n',
                                         text=True, capture_output=True, timeout=20, check=True)
                    self.assertIn(message, json.loads(run.stdout).get('error', ''))


if __name__ == '__main__':
    unittest.main()
