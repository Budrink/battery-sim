import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from upv_gas_heat_budget import inspect, cumulative,gas_region


class GasHeatBudget(unittest.TestCase):
    def fixture(self):
        scene=dict(gas_network=dict(cv_j_per_kg_k=1000.,reference_temperature_k=298.,channels=[],
            compartments=[dict(id='gap-ISS1-0',volume_m3=.001)],
            thermal_contacts=[dict(compartment='gap-ISS1-0',port_id='chamber-wall:gas-partition:0')]),metadata={})
        gas=dict(mass_kg=[.001,.001],temperature_k=[300.,302.],latent_j=[0.,0.],
                 released_energy_j=[0.,10.],oxidizer_kg=[.0002,.00019],consumed_oxidizer_kg=[0.,.00001])
        def contact(solid,net,positive,negative):
            return dict(gas='gap-ISS1-0',solid=solid,heat_to_solid_j=[0.,net],
                        positive_step_heat_j=[0.,positive],negative_step_heat_j=[0.,negative])
        run=dict(time_s=[0.,10.],gas={'gap-ISS1-0':gas},channels={},gas_film_trace={
            'a':contact('chamber-wall',5.,7.,-2.),'b':contact('ISS1',3.,3.,0.)},
            wall_s=1.,advanced_to_s=10.,started_at_s=0.,solver_profile={'minimum_regular_step_s':.005})
        return scene,run

    def test_accepted_heat_closes_independent_gas_budget(self):
        scene,run=self.fixture();report=inspect(scene,run);last=report['snapshots'][-1]
        self.assertEqual(last['groups']['chamber-wall']['net_j'],5.)
        self.assertEqual(last['groups']['ISS pads']['net_j'],3.)
        self.assertAlmostEqual(last['external_gas']['energy_closure_j'],0.)
        self.assertAlmostEqual(last['external_gas']['oxygen_closure_kg'],0.)
        self.assertEqual(last['reaction_location']['wall_touching_r3_fraction'],1.)
        self.assertEqual(last['reaction_location']['wall_touching_reference_volume_fraction'],1.)
        self.assertEqual(cumulative(run,5.)['groups']['chamber-wall']['net_j'],2.5)

    def test_missing_or_miscredited_heat_is_rejected(self):
        scene,run=self.fixture();bad=copy.deepcopy(run)
        bad['gas_film_trace']['b']['heat_to_solid_j'][-1]+=.1
        with self.assertRaises(ValueError):inspect(scene,bad)

    def test_sparse_storage_interpolates_extensive_energy(self):
        scene,run=self.fixture();g=run['gas']['gap-ISS1-0']
        g['mass_kg']=[.001,.002];g['temperature_k']=[300.,304.]
        # Stored sensible energy is 2 -> 12 J, so the midpoint change is 5 J,
        # not 4 J from multiplying independently interpolated m and T.
        g['released_energy_j']=[0.,10.]
        for row in run['gas_film_trace'].values():row['heat_to_solid_j']=[0.,0.]
        row=gas_region(scene,run,{'gap-ISS1-0'},5.)
        self.assertAlmostEqual(row['gas_energy_change_j'],5.)
        self.assertAlmostEqual(row['energy_closure_j'],0.)


if __name__=='__main__':unittest.main()
