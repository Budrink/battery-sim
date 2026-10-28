import json
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from fsri30_open_boundary import open_boundary,exterior_faces
from check_fsri30_open_boundary import cooling_metrics


class OpenBoundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=json.loads((ROOT/'data/development/fsri30-full20-10-20261025-r1/checks/h-20mm/scenario.json').read_text())

    def test_only_exterior_changes_and_mass_flow_sums(self):
        base=self.base;s=open_boundary(base,.01);areas=exterior_faces(base)
        self.assertEqual(len(s['gas_network']['compartments']),2490)
        self.assertEqual(s['gas_network']['channels'],base['gas_network']['channels'])
        for section in ('cells','interfaces','shells','boundaries','heating_protocols'):
            self.assertEqual(s[section],base[section])
        total=0.
        for a,b in zip(s['gas_network']['compartments'],base['gas_network']['compartments']):
            if a['id'] in areas:
                boundary=a.pop('ambient_boundary');total+=boundary['renewal_flow_m3_s']
                self.assertEqual(boundary['area_m2'],areas[a['id']])
            self.assertEqual(a,b)
        self.assertAlmostEqual(total,.01,places=14)
        self.assertTrue(all('ambient_boundary' not in c for c in base['gas_network']['compartments']))

    def test_closed_identity(self):
        self.assertEqual(open_boundary(self.base,0.,pressure_open=False),self.base)

    def test_invalid_flow(self):
        for q in (-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):open_boundary(self.base,q)

    def test_tail_metrics_are_serializable_and_do_not_invent_future(self):
        t=np.arange(5946.)
        data=dict(time_s=t,runaway=t>=1000.)
        for i in range(1,10):data[f'T{i}_k']=300.+i+100.*np.exp(-t/1000.)
        run=dict(time_s=t.tolist(),cell_mean_temperature_k={
            f'B{i:02}':(300.+i+100.*np.exp(-t/1000.)).tolist() for i in range(30)})
        metrics=cooling_metrics(run,data)
        self.assertTrue(metrics['complete']);json.dumps(metrics,allow_nan=False)
        self.assertEqual(metrics['positive_rise_sum_k'],[0.]*4)
        partial=dict(time_s=t[:3000].tolist(),cell_mean_temperature_k={
            k:v[:3000] for k,v in run['cell_mean_temperature_k'].items()})
        self.assertEqual(cooling_metrics(partial,data),dict(complete=False))


if __name__=='__main__':unittest.main()
