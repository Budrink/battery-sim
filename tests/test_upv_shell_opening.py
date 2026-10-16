import copy
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import calibrate_upv_shell_opening as fit
import calibrate_upv_shell_energy as energy
import json

class ShellCalibrationContract(unittest.TestCase):
    def test_only_channels_and_metadata_change(self):
        original,_=fit.inputs('no-cooling')
        result=fit.bake(original,fit.INITIAL)
        frozen=copy.deepcopy(result)
        frozen['gas_network']['channels']=original['gas_network']['channels']
        frozen['metadata']=original['metadata']
        self.assertEqual(frozen,original)
        channels=result['gas_network']['channels']
        self.assertEqual(len(channels),6)
        self.assertEqual(len({c['port_id'] for c in channels}),6)
        self.assertAlmostEqual(sum(c['area_m2'] for c in channels),fit.INITIAL['total_area_m2'])
        self.assertTrue(all(c['damage_opening']['seam_strength_factor']==.3 for c in channels))

    def test_battery_fit_is_forbidden(self):
        scene,_=fit.inputs('no-cooling')
        with self.assertRaises(ValueError):fit.bake(scene,dict(fit.INITIAL,heat_capacity=2))
        with self.assertRaises(ValueError):fit.bake(scene,dict(fit.INITIAL,strength_pa=-1))
        self.assertEqual(fit.PROTOCOL['calibration'],'no-cooling')
        self.assertEqual(fit.PROTOCOL['validation'],'active-cooling')

    @unittest.skipUnless((fit.OUT/'profile.json').is_file(),'Shell profile not frozen yet')
    def test_energy_stage_identity_and_frozen_materials(self):
        scene,_=fit.inputs('no-cooling')
        p=json.loads((fit.PRIOR/'profile.json').read_text())['parameters']
        sh=json.loads((fit.OUT/'profile.json').read_text())['parameters']
        expected=fit.bake(scene,sh);actual=energy.bake(scene,p)
        actual['metadata'].pop('progressive_energy_protocol')
        self.assertEqual(actual,expected)
        changed=energy.bake(scene,dict(p,gas_energy_j=p['gas_energy_j']*1.1,exposure_scale=p['exposure_scale']*.9))
        self.assertEqual(changed['gas_network']['channels'],expected['gas_network']['channels'])
        for before,after in zip(expected['cells'],changed['cells']):
            self.assertEqual(before['thermal'],after['thermal'])
            self.assertEqual(before['geometry'],after['geometry'])
        self.assertEqual(changed['shells'],expected['shells'])

if __name__=='__main__':unittest.main()
