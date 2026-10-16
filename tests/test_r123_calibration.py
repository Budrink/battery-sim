"""Fresh calibration contract: no old profile reads and no source-temperature drive."""
import copy
import math
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import upv_r123_data as dataio
import upv_r123_model as model
import calibrate_upv_r123 as calibration


class CalibrationContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene,cls.meta=model.build('no-cooling',warmup=True)

    def test_primary_cell_is_solved_and_not_forced(self):
        scene=self.scene;cells={c['id']:c for c in scene['cells']}
        self.assertIn('C2',cells)
        self.assertEqual(cells['C2']['thermal']['divisions'],[3,3,3])
        self.assertIn('electrical',cells['C2'])
        self.assertFalse(scene['primary_failures'])
        self.assertFalse(any(b['id'].startswith('source-') for b in scene['boundaries']))
        self.assertTrue(all('conversion' in cells[name] for name in ('C1','C2','C3')))
        self.assertFalse(any('temperature_profile' in b for b in scene['boundaries']))

    def test_material_properties_not_fitted(self):
        a=self.scene;b,_=model.build('no-cooling',dict(model.DEFAULT,resistance_ohm=.07,iss_h=9000.,tim_h=10.,surface_iss=3.),warmup=True)
        for x,y in zip(a['cells'],b['cells']):
            self.assertEqual(x['geometry'],y['geometry'])
            self.assertEqual(x['thermal']['heat_capacity_j_per_k'],y['thermal']['heat_capacity_j_per_k'])
            self.assertEqual(x['thermal']['conductivity_w_per_m_k'],y['thermal']['conductivity_w_per_m_k'])
        for name in ('C1','C2','C3'):
            c=next(v for v in a['cells'] if v['id']==name);outer=[.011,.087,.187];inner=c['geometry']['dimensions_m']
            case=(math.prod(outer)-math.prod(inner))*1600.*1250.
            self.assertAlmostEqual(c['thermal']['heat_capacity_j_per_k']+case,.410*1007.)

    def test_electrical_record_is_not_ui_heating(self):
        values,meta=dataio.electrical_record()
        self.assertGreater(len(values),1000)
        self.assertAlmostEqual(values[0,2],34.9)
        self.assertEqual(values[-1,2],0.)
        self.assertIn('sha256',meta)
        self.assertFalse(self.scene['metadata']['electrical']['synchronization_measured'])
        self.assertTrue(self.scene['metadata']['electrical']['electrical_power_not_used_as_heat'])
        e=next(v['electrical'] for v in self.scene['cells'] if v['id']=='C2')
        self.assertNotIn('power_w',e)

    def test_gas_reservoir_volume_and_casing(self):
        gas=self.scene['gas_network']
        self.assertAlmostEqual(sum(n['volume_m3'] for n in gas['compartments'] if not n['id'].startswith('inside-')),.120)
        self.assertEqual(gas['mass_backend'],'cuda_network')
        self.assertEqual(sum(n['id'].startswith('inside-') for n in gas['compartments']),81)
        self.assertTrue(all(q['damage_opening']['mode']=='deterministic_aperture' for q in gas['channels'] if q['kind']=='rupture'))

    def test_overcharge_is_explicit_and_local_to_source(self):
        cells={c['id']:c for c in self.scene['cells']}
        self.assertEqual(cells['C2']['electrical']['overcharge_fraction'],1.)
        self.assertEqual(cells['C2']['electrical']['nominal_capacity_coulomb'],35.*3600.)
        for name in ('C1','C3'):
            self.assertNotIn('electrical',cells[name])
        self.assertGreater(cells['C2']['conversion']['overcharge_conversion_per_charge'],0.)
        self.assertNotIn('r1_rate',calibration.HEAT)
        self.assertIn('overcharge_gain',calibration.HEAT)

    def test_initial_voltage_rejects_resistance_compensation(self):
        calibration.check_initial_voltage(self.scene)
        impossible=copy.deepcopy(self.scene)
        next(c for c in impossible['cells'] if c['id']=='C2')['electrical']['resistance_ohm']=.07
        with self.assertRaisesRegex(ValueError,'Initial I\\*R'):
            calibration.check_initial_voltage(impossible)


if __name__=='__main__':unittest.main()
