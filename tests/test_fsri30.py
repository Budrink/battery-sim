import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from generate_fsri30 import build, positions, WIDTH, HEIGHT, DIAMETER, MASS, CASE_T
from fsri30_data import load
import math


class Fsri30Tests(unittest.TestCase):
    def test_cylinder_volume_and_total_capacity(self):
        s=build(10.)
        batteries=[c for c in s['cells']if c['role']=='battery']
        self.assertEqual(len(batteries),30)
        self.assertAlmostEqual(WIDTH*WIDTH*HEIGHT,math.pi*DIAMETER**2/4*HEIGHT,places=15)
        m=s['metadata']['geometry_budget']
        self.assertAlmostEqual(m['core_capacity_j_k']+m['steel_mass_kg']*500.,MASS*1000.,places=12)
        for c in batteries:
            law=c['conversion'];self.assertEqual(law['initial_precursor_fraction'],0.)
            self.assertEqual(law['initial_charged_fraction'],1.)
            self.assertLess(.5e-6+law['mass_kg']/2300.+.003/1200.,math.prod(c['geometry']['dimensions_m']))

    def test_no_prescribed_runaway_or_battery_current(self):
        s=build(3600.)
        self.assertEqual(s['primary_failures'],[])
        self.assertFalse(any('electrical'in c for c in s['cells']if c['role']=='battery'))
        self.assertEqual(len({tuple(p)for p in positions()}),30)
        self.assertTrue(all(e['law']['kind']=='directed_geometry'for e in s['interfaces']))
        self.assertTrue(all('connectivity'in e for e in s['gas_network']['thermal_contacts']))
        source={g['id']:g for g in s['gas_network']['compartments']}
        for e in s['gas_network']['channels']:
            if e['kind']=='passage':
                self.assertNotIn('local_equilibrium',source[e['from']])
                self.assertNotIn('local_equilibrium',source[e['to']])
            if e['kind']=='rupture':self.assertIn(':case:+z:outer',e['port_id'])

    def test_raw_timestamps_and_missing_samples_not_fabricated(self):
        d,m=load()
        self.assertEqual(d['time_s'][-1],5945.)
        self.assertEqual(m['untimed_rows_excluded'],136)
        self.assertEqual(m['missing_temperature_samples']['T1'],0)

    def test_heater_cutoff_knots_remain_ordered(self):
        for end in (1799.,1800.,1800.0001,1800.001,5945.):
            h=next(c for c in build(end)['cells']if c['id']=='heater')
            times=[q[0]for q in h['electrical']['current_profile_a']]
            self.assertTrue(all(b>a for a,b in zip(times,times[1:])))


if __name__=='__main__':unittest.main()
