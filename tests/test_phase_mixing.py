import copy
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import upv_r123_model as model
from check_upv_phase_mixing import CASES

class PhaseMixingContract(unittest.TestCase):
    def test_extensions_do_not_modify_fixed_solid_or_electrical_properties(self):
        base,_=model.build('no-cooling',duration=10.)
        for name,extension in CASES.items():
            scene,_=model.build('no-cooling',duration=10.,gas_extensions=extension)
            reference=copy.deepcopy(base['cells'])
            for c in scene['cells']:c.get('conversion',{}).pop('volatile_phase',None)
            self.assertEqual(scene['cells'],reference,name)
            self.assertEqual(scene['interfaces'],base['interfaces'])
            self.assertEqual(scene['shells'],base['shells'])
            self.assertEqual(scene['boundaries'],base['boundaries'])
            self.assertTrue(scene['metadata']['zone_diagnostics'])
            self.assertTrue(all(c['mixing_length_m']>0 for c in scene['gas_network']['channels']))
    def test_new_parameters_have_schema_definitions(self):
        try:import jsonschema
        except ImportError:self.skipTest('Optional jsonschema not installed')
        schema=json.loads((ROOT/'schemas/scenario-v0.schema.json').read_text())
        scene,_=model.build('no-cooling',duration=10.,gas_extensions=CASES['combined'])
        validator=jsonschema.Draft202012Validator({'$defs':schema['$defs'],'$ref':'#/$defs/gas_network'})
        validator.validate(scene['gas_network'])
        validator=jsonschema.Draft202012Validator({'$defs':schema['$defs'],'$ref':'#/$defs/conversion'})
        for c in scene['cells']:
            if 'conversion'in c:validator.validate(c['conversion'])
    def test_unknown_or_nonfinite_mixing_is_rejected(self):
        for extension in ({'typo':1.},{'mixing_flow_fraction':float('nan')},{'mixing_diffusivity_m2_s':-1.}):
            with self.assertRaises(ValueError):model.build('no-cooling',duration=10.,gas_extensions=extension)

if __name__=='__main__':unittest.main()
