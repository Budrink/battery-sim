import sys
import copy
import json
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from refine_upv_gas_access import temperature_pruner,same_prefix_physics,checked_seed_geometry,geometry
from upv_gas_access import CHANNELS


class TemperaturePruning(TestCase):
    def evaluate(self,t,temperature,missing=()):
        t=np.asarray(t,dtype=float)
        run=dict(time_s=t.tolist(),temperature_k={k:np.full_like(t,300.).tolist()for k in CHANNELS})
        run['temperature_k']['101']=np.full_like(t,temperature).tolist()
        def observed(data,times):
            result={k:np.full_like(times,300.)for k in CHANNELS}
            result['101'][np.isin(times,missing)]=np.nan
            return result
        with patch('refine_upv_gas_access.dataio.event',return_value={'time_s':100.}),\
             patch('refine_upv_gas_access.dataio.observed',side_effect=observed):
            return temperature_pruner({})(run)

    def test_sustained_witness_overheat_is_pruned(self):
        self.assertIn('101',self.evaluate(range(115,130),500.))

    def test_uncertain_peak_has_relaxed_but_finite_limit(self):
        self.assertIsNone(self.evaluate(range(99,109),500.))
        self.assertIn('101',self.evaluate(range(99,109),900.))

    def test_missing_scan_breaks_sustained_divergence(self):
        self.assertIsNone(self.evaluate(range(114,123),500.,missing=(117,)))

    def test_brief_or_small_error_does_not_abort(self):
        self.assertIsNone(self.evaluate(range(115,120),500.))
        self.assertIsNone(self.evaluate(range(115,130),350.))


class FreshGeometryContract(TestCase):
    def test_only_annotations_duration_and_output_times_are_ignored(self):
        a=dict(simulation=dict(duration_s=835.,minimum_dt_s=.02),gas_network=dict(area=1.),
            metadata=dict(label='prefix'),failure_sample_times_s=[0.,835.])
        b=dict(simulation=dict(duration_s=950.,minimum_dt_s=.02),gas_network=dict(area=1.),
            metadata=dict(label='suffix'),failure_sample_times_s=[0.,835.,950.])
        self.assertTrue(same_prefix_physics(a,b))
        b['gas_network']['area']=.5
        self.assertFalse(same_prefix_physics(a,b))
        b['gas_network']['area']=1.;b['simulation']['minimum_dt_s']=.005
        self.assertFalse(same_prefix_physics(a,b))

    def test_explicit_seed_geometry_is_reconstructed_not_trusted(self):
        path=Path(__file__).resolve().parents[1]/'data/calibration/upv-separated-chemistry-v6/cooling-fit-20261021-r5/trial-000/scenario.json'
        if not path.exists():self.skipTest('Optional local frozen scenario')
        source=json.loads(path.read_text())
        prefix=geometry(source,1.,end=835.,edge=.025,iss3=.003,protected=.0001,lip=.001)
        self.assertEqual(checked_seed_geometry(source,prefix),prefix)
        shadowed=geometry(source,1.,end=835.,edge=.025,iss3=.003,protected=.0001,lip=.001,pad=.5)
        self.assertEqual(checked_seed_geometry(source,shadowed),shadowed)
        self.assertFalse(same_prefix_physics(prefix,shadowed))
        for key in ['cells','simulation','gas_network']:
            changed=copy.deepcopy(prefix)
            if key=='simulation':changed[key]['minimum_dt_s']=.04
            elif key=='gas_network':changed[key]['averaged_pressure_iterations']=512
            else:changed[key][0]['pose']['position_m'][0]+=.001
            with self.assertRaises(ValueError):checked_seed_geometry(source,changed)
