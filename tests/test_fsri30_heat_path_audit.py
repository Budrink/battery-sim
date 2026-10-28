from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fsri30_heat_path_audit import audit


class HeatPathAuditTests(unittest.TestCase):
    def test_global_coverage_and_actual_surface_integrals(self):
        scene={'gas_network':{'thermal_contacts':[{'compartment':'body-00'}]}}
        node=lambda heat:dict(released_energy_j=[10.,10.+heat],chemical_j=[20.,30.],temperature_k=[400.,1000.])
        run=dict(time_s=[1755.,1800.],partial=True,gas={'body-00':node(3.),'plume-0-00':node(7.)},
            process_totals={'gas_released_energy_j':[15.,30.]},
            gas_film_trace={'a':dict(solid='B00:case:+z',heat_to_solid_j=[0.,4.],
                positive_step_heat_j=[0.,6.],negative_step_heat_j=[0.,-2.])})
        result=audit(scene,run)
        self.assertEqual(result['global_reaction_heat_j'],15.)
        self.assertEqual(result['unobserved_reaction_heat_j'],5.)
        self.assertEqual(result['observed_gas_groups']['plume']['reaction_heat_j'],7.)
        self.assertEqual(result['accepted_film_heat']['B00']['net_j'],4.)
        self.assertEqual(result['hot_reacting_nodes_without_direct_solid_contact'][0]['id'],'plume-0-00')
        self.assertTrue(result['partial'])
        self.assertFalse(result['calibrated'])

    def test_no_global_observer_is_not_replaced_with_selected_sum(self):
        r=audit({'gas_network':{'thermal_contacts':[]}},dict(time_s=[1.,2.],gas={}))
        self.assertIsNone(r['global_reaction_heat_j'])
        self.assertIsNone(r['unobserved_reaction_heat_j'])


if __name__=='__main__':unittest.main()
