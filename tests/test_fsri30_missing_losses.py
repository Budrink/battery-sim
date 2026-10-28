from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_fsri30_missing_losses import radiative_potential,material_inventory,SIGMA


class MissingLossesTests(unittest.TestCase):
    def test_radiation_is_potential_not_assumed_boundary_or_future_sample(self):
        scene=dict(metadata=dict(air_mesh=dict(actual_abs_exposure={'+x':dict(outside_m2=2.)})))
        run=dict(time_s=[0.,1.],thermal_paths=dict(objects={'ABS:+x':dict(
            reference_capacity_weighted_temperature_k=[300.,600.],minimum_temperature_k=[300.,550.],maximum_temperature_k=[300.,650.])}))
        result=radiative_potential(scene,run,1.)
        self.assertAlmostEqual(result['blackbody_center_based_potential_per_unit_emissivity_viewfactor_w'],2.*SIGMA*(600.**4-298.15**4))
        self.assertFalse(result['added_to_solver']);self.assertFalse(result['emissivity_measured'])
        with self.assertRaisesRegex(ValueError,'not observed'):radiative_potential(scene,run,2.)

    def test_residue_inventory_does_not_claim_that_mass_should_be_ejected(self):
        scene=dict(cells=[dict(id='B00',role='battery',conversion=dict(residue_cp=800.))])
        run=dict(time_s=[0.,10.],chemistry={'B00':dict(residue_kg=[0.,.01])},cell_mean_temperature_k={'B00':[300.,600.]})
        result=material_inventory(scene,run)
        self.assertEqual(result['retained_r1_r2_residue_kg'],.01)
        self.assertEqual(result['retained_residue_capacity_j_k'],8.)
        self.assertAlmostEqual(result['retained_residue_sensible_energy_uniform_mean_estimate_j'],8.*(600.-298.15))


if __name__=='__main__':unittest.main()
