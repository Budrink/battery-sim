import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fsri30_abs_contact import hot_abs_contact,conductance_at
from fsri30_data import ROOT


def cold(law):
    return law['background_conductance_w_per_k']+law['active_conductance_w_per_k']/(1.+math.exp(-law.get('geometry_reference_logit',0.)))


class SharedAbsContactTests(unittest.TestCase):
    def test_zero_strength_is_exact_old_profile_not_an_approximation(self):
        base=json.loads((ROOT/'data/development/fsri30-abs-seams-20261025-r2/scenario.json').read_text())
        self.assertEqual(hot_abs_contact(base,strength=0.),base)
        for strength in (-.1,1.1,float('nan')):
            with self.assertRaises(ValueError):hot_abs_contact(base,strength=strength)

    def test_materials_chemistry_geometry_and_numerics_are_unchanged(self):
        base=json.loads((ROOT/'data/development/fsri30-abs-seams-20261025-r2/scenario.json').read_text())
        trial=hot_abs_contact(base)
        for key in base:
            if key not in ('gas_network','metadata'):self.assertEqual(base[key],trial[key],key)
        for key in base['gas_network']:
            if key!='thermal_contacts':self.assertEqual(base['gas_network'][key],trial['gas_network'][key],key)
        changed=0
        for a,b in zip(base['gas_network']['thermal_contacts'],trial['gas_network']['thermal_contacts']):
            if not a['port_id'].startswith('gasmesh:ABS:'):self.assertEqual(a,b);continue
            changed+=1
            self.assertAlmostEqual(cold(a['connectivity']),cold(b['connectivity']),places=13)
            self.assertEqual(b['gas_surface_connectivity'],b['solid_surface_connectivity'])
        self.assertGreater(changed,100)

    def test_hot_response_is_not_the_former_twofold_saturation(self):
        scene={'gas_network':{'thermal_contacts':[dict(port_id='gasmesh:ABS:+x:inside:0',
            connectivity=dict(geometry_reference_logit=0.,background_conductance_w_per_k=.01,active_conductance_w_per_k=1.98))],
            'surface_connectivity':{'reference_temperature_k':298.15,'temperature_log_response':[[298.15,0.],[1200.,math.log(10.)]]}},'metadata':{}}
        law=hot_abs_contact(scene)['gas_network']['thermal_contacts'][0]['connectivity']
        hot=law['background_conductance_w_per_k']+law['active_conductance_w_per_k']/(1.+math.exp(-law['geometry_reference_logit']-math.log(10.)))
        self.assertGreater(hot,8.);self.assertLess(hot,10.);self.assertAlmostEqual(cold(law),1.)

    def test_one_strength_interpolates_ordered_conductances_not_the_directions(self):
        base=json.loads((ROOT/'data/development/fsri30-abs-seams-20261025-r2/scenario.json').read_text())
        a=next(c for c in base['gas_network']['thermal_contacts']if c['port_id'].startswith('gasmesh:ABS:'))
        small={'gas_network':{'thermal_contacts':[a],'surface_connectivity':base['gas_network']['surface_connectivity']},'metadata':{}}
        hot=hot_abs_contact(small)['gas_network']['thermal_contacts'][0]
        mid=hot_abs_contact(small,strength=.5)['gas_network']['thermal_contacts'][0]
        for endpoint in ('gas_surface_connectivity','solid_surface_connectivity'):
            for t in [298.15,400.,500.,600.,750.,900.,1050.,1200.]:
                expected=.5*(conductance_at(a['connectivity'],a.get(endpoint,{}),t)+conductance_at(hot['connectivity'],hot[endpoint],t))
                actual=conductance_at(mid['connectivity'],mid[endpoint],t)
                self.assertAlmostEqual(expected,actual,places=13)
        forward=conductance_at(mid['connectivity'],mid['gas_surface_connectivity'],1200.)
        reverse=conductance_at(mid['connectivity'],mid['solid_surface_connectivity'],1200.)
        self.assertNotAlmostEqual(forward,reverse,places=8)


if __name__=='__main__':unittest.main()
