from copy import deepcopy
from pathlib import Path
import sys
import struct
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fsri30_spatial_patterns import overlap_restriction,event_patterns,read_fields,compare_spatial_fields


def box(a,b):return dict(low_m=[a,0.,0.],high_m=[b,1.,1.])


class SpatialPatternTests(unittest.TestCase):
    def test_constant_temperature_and_extensive_totals_preserved(self):
        t,e,cv,fv=overlap_restriction([box(0.,1.)],[box(0.,.25),box(.25,1.)])
        np.testing.assert_allclose(t@np.array([500.,500.]),[500.])
        np.testing.assert_allclose(e@np.array([2.,6.]),[8.])
        np.testing.assert_allclose(cv,[1.]);np.testing.assert_allclose(fv,[.25,.75])

    def test_nonnested_partitions_preserve_matter(self):
        t,e,_,_=overlap_restriction([box(0.,.5),box(.5,1.)],[box(0.,.25),box(.25,1.)])
        np.testing.assert_allclose(e@np.array([2.,6.]),[4.,4.])
        np.testing.assert_allclose(t@np.array([300.,500.]),[400.,500.])

    def test_hole_is_not_renormalized(self):
        with self.assertRaisesRegex(ValueError,'cover coarse'):
            overlap_restriction([box(0.,1.)],[box(0.,.9)])

    def test_overlap_is_not_renormalized(self):
        with self.assertRaisesRegex(ValueError,'cover coarse'):
            overlap_restriction([box(0.,1.)],[box(0.,.75),box(.25,1.)])

    def test_fine_volume_outside_domain_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'cover fine'):
            overlap_restriction([box(0.,.5)],[box(0.,1.)])

    def test_event_shift_diagnostic_does_not_modify_trajectories(self):
        t=np.arange(0.,401.,5.).tolist()
        runs={20:dict(time_s=t,cell_mean_temperature_k=dict(B10=(300.+np.maximum(0.,np.asarray(t)-100.)).tolist())),
              10:dict(time_s=t,cell_mean_temperature_k=dict(B10=(300.+np.maximum(0.,np.asarray(t)-110.)).tolist()))}
        events={20:[dict(component='B10',time_s=100.)],10:[dict(component='B10',time_s=110.)]}
        before=deepcopy(runs);p=event_patterns(runs,events)
        self.assertEqual(p['events'][0]['raw_delay_s'],10.)
        self.assertEqual(p['events'][0]['event_aligned_rms_relative_kelvin'],0.)
        self.assertEqual(before,runs)
        self.assertTrue(p['shape_metric_is_diagnostic_only']);self.assertFalse(p['patterns_promoted'])

    def test_late_event_cannot_extrapolate_shape(self):
        runs={mm:dict(time_s=[0.,100.,200.],cell_mean_temperature_k=dict(B10=[300.,500.,400.])) for mm in (20,10)}
        events={mm:[dict(component='B10',time_s=100.)] for mm in (20,10)}
        row=event_patterns(runs,events)['events'][0]
        self.assertNotIn('event_aligned_rms_relative_kelvin',row)

    def test_missing_event_is_not_invented(self):
        runs={mm:dict(time_s=[0.,100.,400.],cell_mean_temperature_k=dict(B10=[300.,500.,400.])) for mm in (20,10)}
        events={20:[dict(component='B10',time_s=100.),dict(component='B09',time_s=200.)],
                10:[dict(component='B10',time_s=100.)]}
        self.assertEqual(event_patterns(runs,events)['missing_on_fine'],['B09'])

    def test_full_field_pipeline_matches_non_air_by_id(self):
        a=dict(gas_network=dict(compartments=[dict(id='body-00'),dict(id='air-c')]),
            metadata=dict(gas_volume_geometry={'air-c':box(0.,1.)}))
        b=dict(gas_network=dict(compartments=[dict(id='air-f0'),dict(id='body-00'),dict(id='air-f1')]),
            metadata=dict(gas_volume_geometry={'air-f0':box(0.,.25),'air-f1':box(.25,1.)}))
        def write(p,values):
            with p.open('wb') as out:
                out.write(b'BATTGAS1'+struct.pack('<Q',len(values)))
                for t in (0.,5.):out.write(struct.pack('<d',t)+np.array(values,dtype='<f4').tobytes())
        with tempfile.TemporaryDirectory() as folder:
            pa,pb=Path(folder)/'coarse.bin',Path(folder)/'fine.bin'
            write(pa,[[600.,0.,0.],[400.,400.,.5]])
            write(pb,[[400.,100.,.125],[650.,0.,0.],[400.,300.,.375]])
            result=compare_spatial_fields(a,b,pa,pb,5.)
        self.assertEqual(result['air_volume_rms_relative_kelvin'],0.)
        self.assertEqual(result['maximum_air_fuel_l1_relative'],0.)
        self.assertEqual(result['maximum_extensive_projection_relative_residual'],0.)
        self.assertAlmostEqual(result['same_node_maximum_local_relative_kelvin'],50./650.)
        self.assertEqual(result['worst_same_node']['node'],'body-00')

    def test_truncated_field_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'truncated.bin'
            p.write_bytes(b'BATTGAS1'+struct.pack('<Q',1)+b'bad')
            with self.assertRaisesRegex(ValueError,'Truncated'):read_fields(p)

    def test_arrival_hypothesis_is_fit_without_held_out_events(self):
        names=['B10']+[f'B{i:02}' for i in range(8)]
        coarse=[100.]+[200.+i*10. for i in range(8)]
        fine=[100.]+[100.+(t-100.)*1.01 for t in coarse[1:]]
        runs={mm:dict(time_s=[0.,1000.],cell_mean_temperature_k={k:[300.,400.] for k in names}) for mm in (20,10)}
        events={mm:[dict(component=k,time_s=t) for k,t in zip(names,ts)] for mm,ts in ((20,coarse),(10,fine))}
        p=event_patterns(runs,events)['time_predictor']
        self.assertAlmostEqual(p['scale'],1.01)
        self.assertLess(p['held_out_rmse_s'],1e-10)
        train=set(p['training_components']);held={r['component'] for r in p['held_out']}
        self.assertFalse(train&held);self.assertEqual(train|held,set(names)-{'B10'})
        self.assertFalse(p['promoted'])


if __name__=='__main__':unittest.main()
