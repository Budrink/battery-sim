import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import upv_r123_model as model
from upv_pad_ablation import PADS,CONDUCTANCES
from upv_gas_remap import unvented_remap
from gas_void_geometry import cell_box

class PadAblation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.full,_=model.build('no-cooling',duration=950.)
        cls.empty,_=model.build('no-cooling',duration=950.,diagnostic_removed_pads=PADS)

    def test_no_change_to_materials_chemistry_and_no_heat_to_removed_pads(self):
        a,b=self.full,self.empty
        for key in ('cells','shells','boundaries','measurement_probes'):self.assertEqual(a[key],b[key])
        ports={p['id']:p for p in b['ports']};old={e['id']:e for e in a['interfaces']}
        for e in b['interfaces']:
            if any(ports[e[k]]['cell_id']in PADS for k in ('port_a','port_b')):
                self.assertTrue(all(e['law'][k]==0. for k in CONDUCTANCES))
            else:self.assertEqual(e,old[e['id']])
        self.assertTrue(all(ports[c['port_id']]['cell_id']not in PADS for c in b['gas_network']['thermal_contacts']))

    def test_removed_slots_are_free_volume_with_full_face_access(self):
        boxes=self.empty['metadata']['gas_geometry']['void_boxes'];objects={c['id']:c for c in self.empty['cells']}
        for name in PADS:
            p=objects[name]['pose']['position_m']
            self.assertTrue(any(all(b['lo'][k]<=p[k]<=b['hi'][k]for k in range(3))for b in boxes.values()),name)
        channels=self.empty['gas_network']['channels']
        normal=[e for e in channels if e['id'].startswith('ablated-face-ISS1-')]
        self.assertEqual(len(normal),8)
        self.assertAlmostEqual(sum(e['area_m2']for e in normal),.087*.187,places=12)

    def test_free_volume_and_remap_all_return_cases(self):
        path=Path(__file__).resolve().parents[1]/'data/calibration/upv-separated-chemistry-v6/restart-gas-baseline-20261019-r1/scenario.json'
        source=json.loads(path.read_text())if path.exists()else None
        for k in range(8):
            scene,_=model.build('no-cooling',duration=950.,diagnostic_removed_pads=PADS[k:])
            gas=scene['gas_network'];self.assertAlmostEqual(sum(n['volume_m3']for n in gas['compartments']if not n['id'].startswith('inside-')),.12,places=12)
            if source is not None:self.assertTrue(unvented_remap(source,scene))

    def test_invalid_removal(self):
        for pads in (['C2'],['ISS1','ISS1'],['unknown']):
            with self.assertRaises(ValueError):model.build('no-cooling',duration=1.,diagnostic_removed_pads=pads)

if __name__=='__main__':unittest.main()
