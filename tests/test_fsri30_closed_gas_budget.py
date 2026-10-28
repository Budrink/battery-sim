from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fsri30_closed_gas_budget import budget


class ClosedGasBudgetTests(unittest.TestCase):
    def test_old_outside_burning_is_not_misreported_as_module_burning(self):
        scene={'gas_network':{'compartments':[dict(id='body-00',oxidizer_heat_j_per_kg=10.),
            dict(id='hood',oxidizer_heat_j_per_kg=10.)]}}
        run=dict(process_totals={'gas_mass_kg':[5.,5.],'gas_oxidizer_kg':[.9,.6],
            'gas_chemical_j':[20.,17.],'gas_released_energy_j':[2.,5.]},
            gas={'hood':dict(mass_kg=[4.,4.],oxidizer_kg=[.6,.5],chemical_j=[10.,9.],released_energy_j=[2.,3.])})
        r=budget(scene,run)
        self.assertAlmostEqual(r['module']['released_energy_j']['change'],2.)
        self.assertAlmostEqual(r['outside']['released_energy_j']['change'],1.)
        self.assertAlmostEqual(r['initial_module_oxygen_heat_ceiling_j'],3.)
        self.assertAlmostEqual(r['maximum_oxygen_heat_closure_error_j'],0.)

    def test_unobserved_external_volume_is_not_implicitly_ignored(self):
        with self.assertRaises(ValueError):budget({'gas_network':{'compartments':[dict(id='hood')]}},{'gas':{}})


if __name__=='__main__':unittest.main()
