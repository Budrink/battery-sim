from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from report_fsri30_abs_cooling import audit,outside_state


class AbsCoolingAuditTests(unittest.TestCase):
    def test_observed_outside_temperatures_and_no_future_substitution(self):
        scene=dict(gas_network=dict(compartments=[dict(id='air-a'),dict(id='air-b')]))
        run=dict(time_s=[10.],gas_film_trace={
            'a':dict(solid='ABS:a',gas='air-a',instantaneous_conductance_w_per_k=[1.],instantaneous_power_w=[-100.]),
            'b':dict(solid='ABS:b',gas='air-b',instantaneous_conductance_w_per_k=[3.],instantaneous_power_w=[-300.])})
        dtype=np.dtype([('time','<f8'),('values','<f4',(2,3))]);frame=np.zeros(1,dtype=dtype)
        frame['time']=10.;frame['values'][0,:,0]=[300.,400.]
        with tempfile.TemporaryDirectory()as folder:
            path=Path(folder)/'frames.bin'
            with path.open('wb')as f:
                f.write(b'BATTGAS1');f.write(np.array([2],dtype='<u8').tobytes());f.write(frame.tobytes())
            result=outside_state(scene,run,path,10.)
            self.assertEqual(result['conductance_weighted_gas_temperature_k'],375.)
            self.assertEqual(result['conductance_weighted_wall_temperature_k'],475.)
            self.assertEqual(result['instantaneous_wall_to_gas_w'],400.)
            with self.assertRaisesRegex(ValueError,'not been observed'):outside_state(scene,run,path,11.)
            scene['gas_network']['compartments'].pop()
            with self.assertRaisesRegex(ValueError,'topology'):outside_state(scene,run,path,10.)

    def test_accepted_flux_closes_union_without_double_counting_seams(self):
        run=dict(time_s=[0.,10.],applied_objects=[dict(id='ABS:a',capacity_j_k=2.)],
            thermal_paths=dict(objects={'ABS:a':dict(reference_capacity_weighted_temperature_k=[300.,303.])}),
            gas_film_trace={'in':dict(solid='ABS:a',gas='body-00',heat_to_solid_j=[5.,25.]),
                'out':dict(solid='ABS:a',gas='air-0-0-0',heat_to_solid_j=[-1.,-15.])},
            process_totals=dict(escaped_energy_j=[0.,14.],gas_released_energy_j=[0.,0.]))
        result=audit(run,0.,10.)
        self.assertEqual(result['total_power_w']['gas_inside_to_wall'],2.)
        self.assertEqual(result['total_power_w']['wall_to_external_gas'],1.4)
        self.assertEqual(result['abs_union_energy_closure_j'],0.)
        with self.assertRaisesRegex(ValueError,'not been observed'):audit(run,0.,11.)
        run['gas_film_trace']['out']['heat_to_solid_j'][-1]-=1.
        with self.assertRaisesRegex(ValueError,'closure failed'):audit(run,0.,10.)


if __name__=='__main__':unittest.main()
