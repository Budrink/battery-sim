import sys
import unittest
import tempfile
from copy import deepcopy
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from check_fsri30_grid_compromise import pattern_agreement,gas_field_agreement


class PatternVerificationTests(unittest.TestCase):
    def test_predictor_cannot_consume_spatial_error_budget(self):
        run=dict(time_s=[0.,10.],cell_mean_temperature_k={'B00':[298.15,500.]},
            process_totals=dict(gas_released_energy_j=[0.,100.]),
            gas_groups=dict(outside=dict(maximum_temperature_k=[298.15,1500.])))
        info=dict(all_gas_r3_j=100.,r2_converted_components=1,
            model_conversion_events=[dict(component='B00',time_s=5.)])
        self.assertTrue(pattern_agreement(run,run,info,info)['accepted'])
        changed=deepcopy(run);changed['gas_groups']['outside']['maximum_temperature_k'][-1]*=1.01
        self.assertFalse(pattern_agreement(run,changed,info,info)['accepted'])

    def test_unchanged_maximum_cannot_hide_a_different_local_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=[Path(tmp)/name for name in ('a.bin','b.bin')]
            dtype=np.dtype([('time','<f8'),('fields','<f4',(2,3))]);v=np.zeros(2,dtype=dtype)
            v['time']=[0.,10.];v['fields'][:,:,0]=[[300.,300.],[1500.,500.]]
            v['fields'][:,:,1]=1.;v['fields'][:,:,2]=.001
            header=b'BATTGAS1'+(2).to_bytes(8,'little')
            a.write_bytes(header+v.tobytes());b.write_bytes(a.read_bytes())
            self.assertTrue(gas_field_agreement(a,b)['accepted'])
            v['fields'][1,1,0]=800.;b.write_bytes(header+v.tobytes())
            result=gas_field_agreement(a,b)
            self.assertFalse(result['accepted']);self.assertEqual(result['maximum_absolute_temperature_difference_k'],300.)


if __name__=='__main__':unittest.main()
