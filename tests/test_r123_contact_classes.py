"""Material-pair contact references must not alter solid material properties."""
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import upv_r123_model as model
from migrate_upv_chemistry_profile import migrate


class MaterialPairContacts(unittest.TestCase):
    def test_reference_factors_only_change_matching_material_pairs(self):
        plain,_=model.build('no-cooling',warmup=True,duration=626.956)
        changed,_=model.build('no-cooling',dict(battery_iss_factor=2.,battery_tim_factor=3.),warmup=True,duration=626.956)
        for field in ('cells','ports','shells','boundaries','gas_network','measurement_probes'):
            self.assertEqual(plain[field],changed[field],field+' changed with a contact-only factor')
        cells={c['id']:c for c in plain['cells']};ports={p['id']:p for p in plain['ports']}
        counts={2:0,3:0}
        for a,b in zip(plain['interfaces'],changed['interfaces']):
            material={cells[ports[a[p]]['cell_id'].split(':')[0]]['material_id']for p in ('port_a','port_b')}
            factor=2 if material=={'battery','ISS'} else 3 if material=={'battery','TIM'} else 1
            if factor>1:counts[factor]+=1
            for key in a['law']:
                expected=a['law'][key]*factor if key in (
                    'background_conductance_w_per_k','active_conductance_w_per_k','destroyed_background_conductance_w_per_k') else a['law'][key]
                if key in ('background_conductance_w_per_k','active_conductance_w_per_k',
                           'destroyed_background_conductance_w_per_k'):
                    self.assertAlmostEqual(expected,b['law'][key],places=11,msg=(a['id'],key))
                else:
                    self.assertEqual(expected,b['law'][key],(a['id'],key))
        self.assertEqual(counts,{2:3,3:3})

    def test_neutral_factors_reproduce_last_saved_physical_inputs(self):
        base=ROOT/'data/calibration/upv-charge-degradation-v5/shape-fit-20261017'
        source=base/'frozen-shape-check/frozen-parameters.json'
        saved=base/'local-shape-r0-step-1/scenario.json'
        if not source.exists() or not saved.exists():self.skipTest('Local completed CUDA fixture unavailable')
        p=json.loads(source.read_text())['parameters'];prior=json.loads(saved.read_text())
        p=migrate(p)
        # This fixture predates the independently declared gas/wall contact.
        # Preserve its explicit historical coefficient, not today's default.
        p['chamber_gas_h']=next(c['h_w_per_m2_k']for c in prior['gas_network']['thermal_contacts']
            if c['compartment']=='chamber'and c['port_id']=='chamber-wall:+x')
        scene,_=model.build('no-cooling',p,warmup=True,duration=626.956)
        # Spatial gas reconstruction deliberately replaces the old gas graph
        # and adds cropped gas-film ports; the original solid ports stay exact.
        actual_ports={q['id']:q for q in scene['ports']}
        for port in prior['ports']:
            self.assertEqual(port,actual_ports[port['id']])
        for key in ('interfaces','shells','boundaries','measurement_probes'):
            self.assertEqual(prior[key],scene[key],key+' silently changed while adding contact classes')
        for before,after in zip(prior['cells'],scene['cells']):
            for key in before:
                if key not in ('conversion','electrical'):self.assertEqual(before[key],after[key])

    def test_invalid_or_out_of_range_contact_factors_are_rejected(self):
        for factor in (0.,-.1,.01,11.,float('nan')):
            with self.subTest(factor=factor),self.assertRaises(ValueError):
                model.build('no-cooling',dict(battery_iss_factor=factor),warmup=True,duration=30.)
        with self.assertRaises(ValueError):
            model.build('no-cooling',dict(iss_h=10000.,battery_iss_factor=2.),warmup=True,duration=30.)


if __name__=='__main__':unittest.main()
