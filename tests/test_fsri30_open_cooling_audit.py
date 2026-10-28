import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_fsri30_open_cooling import radiative_potential,tail_flows,SIGMA


class CoolingAudit(unittest.TestCase):
    def fixture(self,temperature):
        h=1/(1/8.+.004/(2*.2));g=.01*h
        scene=dict(cells=[dict(id='ABS:-x',geometry=dict(dimensions_m=[.004,.1,.1]),
            thermal=dict(conductivity_w_per_m_k=[.2]*3))],
            ports=[dict(id='gasmesh:ABS:-x:outside:0',cell_id='ABS:-x',surface='-x')],
            gas_network=dict(thermal_contacts=[dict(port_id='gasmesh:ABS:-x:outside:0',thermal_volume=0,
                connectivity=dict(background_conductance_w_per_k=.01*g,active_conductance_w_per_k=1.98*g,geometry_reference_logit=0.))]))
        run=dict(time_s=[0.,5945.],partial=False,final_fields_k={'ABS:-x':[temperature]})
        return scene,run

    def test_radiation_is_a_potential_not_added_heat(self):
        scene,run=self.fixture(800.);r=radiative_potential(scene,run)
        self.assertAlmostEqual(r['external_abs_area_m2'],.01)
        self.assertAlmostEqual(r['blackbody_power_using_cell_center_w'],.01*SIGMA*(800.**4-298.15**4))
        self.assertFalse(r['added_to_solver']);self.assertIsNone(r['assumed_emissivity'])
        self.assertLess(r['blackbody_power_with_half_slab_surface_estimate_w'],r['blackbody_power_using_cell_center_w'])

    def test_ambient_zero_and_incomplete_rejected(self):
        scene,run=self.fixture(298.15);self.assertAlmostEqual(radiative_potential(scene,run)['blackbody_power_using_cell_center_w'],0.)
        run['partial']=True
        with self.assertRaises(ValueError):radiative_potential(scene,run)

    def test_channel_energy_is_not_counted_twice(self):
        run=dict(time_s=[3300.,3500.],channels={'vent-a':dict(effective_area_m2=[1.,1.],mixed_mass_kg=[0.,.02],
            transferred_mass_kg=[0.,.001],delivered_gas_energy_j=[0.,4.])},
            process_totals=dict(escaped_energy_j=[0.,10000.],gas_released_energy_j=[0.,10.]))
        result=tail_flows(run);self.assertAlmostEqual(result['module_vent_net_thermal_w'],.02)
        self.assertAlmostEqual(result['ambient_thermal_w'],50.)
        with self.assertRaises(ValueError):tail_flows(run,end=4000.)


if __name__=='__main__':unittest.main()
