import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_fsri30_energy_routes import audit


class EnergyRouteTests(unittest.TestCase):
    def test_direction_and_measured_window(self):
        run=dict(time_s=[0.,1.,2.],gas_film_trace={
            'x':dict(solid='B09:case:+x',heat_to_solid_j=[0.,2.,4.])},
            thermal_paths=dict(interfaces={'can-contact:B09:B10':dict(
                **{'from':'B09:case:+y','to':'B10:case:-y','a_to_b_power_w':[-3.,-3.,-3.]})}),
            channels={'vent-x':dict(transferred_mass_kg=[0.,.1,.2],transferred_chemical_j=[0.,1.,2.])},
            gas_groups={k:dict(released_energy_j=[0.,1.,2.]) for k in ('module','outside')})
        result=audit(run,0.,2.)
        self.assertEqual(result['can_contact_j']['B09'],6.)
        self.assertEqual(result['gas_to_solid_j']['B09'],4.)
        self.assertEqual(result['vent_mass_kg'],.2)
        with self.assertRaises(ValueError):audit(run,0.,3.)


if __name__=='__main__':unittest.main()
