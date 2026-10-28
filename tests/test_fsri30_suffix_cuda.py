"""Opt-in actual CUDA restart contract tests; run outside the filesystem sandbox."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import calibrate_upv_r123 as settings
from check_upv_restart import native
from fsri30_conditioned_start import PROBE
from fsri30_data import save,sha
from check_fsri30_closed_exhausts import close_exhausts
import fsri30_suffix_model as model
from search_fsri30_transport import REFERENCE


@unittest.skipUnless(os.environ.get('BATTERY_SIM_TEST_CUDA')=='1','Explicit CUDA integration opt-in required')
class CudaSuffixContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='fsri30-suffix-contract-')
        cls.root=Path(cls.temp.name)
        cls.probe=Path(os.environ.get('BATTERY_SIM_SUFFIX_PROBE',str(PROBE)))
        settings.PROBE=cls.probe
        cls.base=json.loads((REFERENCE/'scenario.json').read_text())
        cls.base['simulation']['duration_s']=2.
        cls.base['failure_sample_times_s']=[0.,1.,2.]
        cls.base['gas_film_trace']=True
        cls.source=cls.root/'source.json';save(cls.source,cls.base)
        cls.checkpoint=cls.root/'state.bin'
        native(cls.source,cls.root/'capture',dict(restart_contract=sha(cls.probe),
            save_committed_restart=str(cls.checkpoint),committed_restart_at_s=1.,
            stop_after_committed_restart=True,stop_after_first_rupture=False),60.,enforce_speed=False)
        cls.counter=0

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def launch(self,scene,**controls):
        type(self).counter+=1;folder=self.root/f'case-{self.counter}'
        folder.mkdir();path=folder/'scenario.json';save(path,scene)
        return native(path,folder/'replay',dict(restart_contract=sha(self.probe),
            restart_from=str(self.checkpoint),restart_transport_scenario=str(self.source),
            stop_after_first_rupture=False,**controls),60.,enforce_speed=False)

    def test_closed_exhaust_preserves_state_and_blocks_pressure_and_mixing(self):
        r=self.launch(close_exhausts(self.base),diagnostic_closed_exhausts=True)
        self.assertTrue(r['suffix_initial_state_bitwise_equal'])
        self.assertTrue(r['diagnostic_closed_exhausts'])
        self.assertLess(r['max_thermal_residual_j'],1e-4)
        self.assertLess(r['max_mass_residual_kg'],1e-10)

    def test_closing_exits_requires_explicit_diagnostic_request(self):
        with self.assertRaisesRegex(RuntimeError,'outside declared prior'):
            self.launch(close_exhausts(self.base))

    def test_closed_diagnostic_cannot_also_seal_an_internal_path(self):
        s=close_exhausts(self.base)
        next(e for e in s['gas_network']['channels'] if e['id']=='cap-up-0')['area_m2']=0.
        with self.assertRaisesRegex(RuntimeError,'outside declared prior'):
            self.launch(s,diagnostic_closed_exhausts=True)

    def test_partial_closure_is_not_a_closed_experiment(self):
        s=close_exhausts(self.base)
        e=next(e for e in s['gas_network']['channels'] if e['id']=='exhaust-path-0-0')
        e['area_m2']=1e-6
        with self.assertRaisesRegex(RuntimeError,'EVERY module exit'):
            self.launch(s,diagnostic_closed_exhausts=True)

    def test_valid_geometry_hypotheses_preserve_initial_state_and_ledgers(self):
        for name,x in [('identity',model.initial_point())]+model.experiments()+[('boundaries',np.ones(len(model.SPACE)))]:
            with self.subTest(name=name):
                r=self.launch(model.candidate(self.base,x))
                self.assertTrue(r['suffix_initial_state_bitwise_equal'])
                self.assertFalse(r['full_warmup_validation'])
                self.assertAlmostEqual(r['started_at_s'],1.,places=12)
                self.assertAlmostEqual(r['advanced_to_s'],2.,places=12)
                self.assertLess(r['max_thermal_residual_j'],1e-4)
                self.assertLess(r['max_chemical_residual_j'],1e-4)
                self.assertLess(r['max_mass_residual_kg'],1e-10)

    def test_fuel_change_rejected(self):
        s=deepcopy(self.base);s['cells'][0]['conversion']['net_heat_j_per_kg']*=1.1
        with self.assertRaisesRegex(RuntimeError,'frozen solids'):self.launch(s)

    def test_gas_volume_or_inventory_change_rejected(self):
        s=deepcopy(self.base);s['gas_network']['compartments'][0]['volume_m3']*=1.1
        with self.assertRaisesRegex(RuntimeError,'IDENTICAL gas volumes'):self.launch(s)

    def test_rupture_strength_change_rejected(self):
        s=deepcopy(self.base)
        next(e for e in s['gas_network']['channels'] if e['kind']=='rupture')['opening_pressure_pa']*=2
        with self.assertRaisesRegex(RuntimeError,'material barrier'):self.launch(s)

    def test_plume_to_core_shortcut_rejected(self):
        s=deepcopy(self.base)
        next(c for c in s['gas_network']['thermal_contacts'] if c['port_id'].startswith('B'))['compartment']='hood'
        with self.assertRaisesRegex(RuntimeError,'shortcut through a barrier'):self.launch(s)

    def test_heater_change_rejected(self):
        s=deepcopy(self.base)
        next(c for c in s['cells'] if c['id']=='heater')['electrical']['current_profile_a'][0][1]*=2
        with self.assertRaisesRegex(RuntimeError,'frozen solids'):self.launch(s)


if __name__=='__main__':unittest.main()
