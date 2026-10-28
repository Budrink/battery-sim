from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fsri30_internal_radiation import add_internal_radiation,rectangle_view_factor
from calibrate_fsri30_sequence import BASE
from check_fsri30_internal_radiation import propose,secondary_chemistry


class InternalRadiationTests(unittest.TestCase):
    def test_primary_recenter_does_not_change_geometry_or_invent_time_shift(self):
        p=dict(r2_rate_scale=.023,gas_can_contact_scale=1.)
        row=dict(parameters=p,internal_emissivity=.4,assessment=dict(primary_offset_s=50.))
        e,q,why=propose([row],p)
        self.assertEqual(e,.4);self.assertGreater(q['r2_rate_scale'],p['r2_rate_scale'])
        self.assertEqual(q['gas_can_contact_scale'],1.)

    def test_second_wave_bracket_and_incomplete_observations(self):
        p=dict(r2_rate_scale=.023)
        row=dict(parameters=p,internal_emissivity=.4,assessment=dict(primary_offset_s=0.,last_observed_s=2290.,model={'arrivals':[]}))
        e,q,why=propose([row],p);self.assertEqual(e,.8);self.assertEqual(q,p)
        row['assessment']['last_observed_s']=2100.
        self.assertIsNone(propose([row],p)[0])
        row['assessment']['model']['arrivals']=[dict(channel='B09',time_s=2136.)]
        self.assertIn('not-yet',propose([row],p)[2])
        row['secondary_chemistry']={'reacted_in_second_wave':2}
        self.assertIn('target-met',propose([row],p)[2])

    def test_chemistry_requires_actual_non_source_conversion(self):
        scene=dict(metadata={'heated_component':'B10'},cells=[dict(id=k,role='battery',conversion=dict(mass_kg=1.,gas_yield=.3)) for k in ('B10','B09','B11')])
        run=dict(time_s=[0.,2135.,2140.],chemistry={k:dict(r2_generated_gas_kg=v) for k,v in
            [('B10',[0.,.3,.3]),('B09',[0.,.15,.3]),('B11',[0.,0.,.01])]})
        result=secondary_chemistry(scene,run,2136.)
        self.assertEqual(result['reacted_in_second_wave'],1)
        self.assertEqual(result['events'],[dict(id='B09',half_R2_conversion_s=2135.)])

    def test_rectangle_against_independent_quadrature(self):
        nodes,weights=np.polynomial.legendre.leggauss(120)
        # Reduced four-dimensional view integral: count pairs with displacement x,y.
        for w,h,g in ((1.,1.,1.),(1.,4.,.3),(.016,.065,.003)):
            x=(nodes+1)*w/2;y=(nodes+1)*h/2
            integrand=(w-x[:,None])*(h-y[None,:])*g*g/(np.pi*(g*g+x[:,None]**2+y[None,:]**2)**2)
            oracle=np.sum(integrand*weights[:,None]*weights[None,:])
            self.assertAlmostEqual(rectangle_view_factor(w,h,g),oracle,places=10)
            self.assertAlmostEqual(rectangle_view_factor(w,h,g),rectangle_view_factor(h,w,g),places=12)

    def test_native_fixture_inventory_and_existing_paths_unchanged(self):
        base=json.loads(BASE.read_text());copy=deepcopy(base);scene=add_internal_radiation(base,.5)
        self.assertEqual(base,copy)
        for key in ('cells','shells','gas_network','boundaries','heating_protocols'):
            self.assertEqual(base[key],scene[key])
        pairs=scene['metadata']['internal_radiation']['pairs'];self.assertGreater(len(pairs),25)
        self.assertTrue(any(p['a']=='B09' and p['b']=='B10' for p in pairs))
        self.assertFalse(any(p['a']=='B08' and p['b']=='B10' for p in pairs))
        self.assertFalse(any(p['a'] in ('B02','B03') and p['b']=='B10' for p in pairs))
        # Removing the actual opaque heater restores precisely its two views.
        without=deepcopy(base);without['cells']=[c for c in without['cells'] if c['id']!='heater']
        unblocked=add_internal_radiation(without,.5)['metadata']['internal_radiation']['pairs']
        self.assertEqual(len(unblocked),len(pairs)+2)
        for p in pairs:self.assertTrue(0<p['view_factor']<1)
        with self.assertRaisesRegex(ValueError,'already'):add_internal_radiation(scene,.5)
        for bad in (-.1,1.1,float('nan')):
            with self.assertRaises(ValueError):add_internal_radiation(base,bad)


if __name__=='__main__':unittest.main()
