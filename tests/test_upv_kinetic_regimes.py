import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from probe_upv_kinetic_regimes import electrical_work_limit,electrical_violation


def scene(current,voltage):
    return {'cells':[{'id':'C2','electrical':{'current_profile_a':current,'voltage_profile_v':voltage}}]}


class ElectricalBudget(unittest.TestCase):
    def test_constant_drive_budget_is_not_ui_work(self):
        s=scene([[0.,2.],[10.,2.]],[[0.,4.],[10.,4.]])
        np.testing.assert_allclose(electrical_work_limit(s,[0.,5.,10.],3.),[0.,10.,20.])

    def test_two_linear_drives_integrated_exactly(self):
        s=scene([[0.,0.],[10.,2.]],[[0.,4.],[10.,6.]])
        coarse=electrical_work_limit(s,[0.,10.],3.)
        fine=electrical_work_limit(s,np.arange(0.,10.1,.1),3.)
        self.assertAlmostEqual(coarse[-1],70./3.,places=12)
        self.assertAlmostEqual(coarse[-1],fine[-1],places=12)

    def test_early_budget_violation_cannot_hide_in_later_work(self):
        s=scene([[0.,2.],[10.,2.]],[[0.,4.],[10.,4.]])
        s['cells'][0]['electrical']['reversible_voltage_floor_v']=3.65
        # Explicit sensitivity floor allows 3.5 J per 5 s, not 40 J of U*I.
        good={'time_s':[0.,5.,10.],'joule_energy_j':{'C2':[0.,3.,6.]}}
        bad={'time_s':[0.,5.,10.],'joule_energy_j':{'C2':[0.,5.,5.]}}
        self.assertIsNone(electrical_violation(good,s))
        self.assertIn('5.000 s',electrical_violation(bad,s))

    def test_voltage_floor_remains_explicit_assumption(self):
        s=scene([[0.,2.],[10.,2.]],[[0.,3.],[10.,3.]])
        np.testing.assert_allclose(electrical_work_limit(s,[0.,10.]),[0.,60.])
        with self.assertRaises(ValueError):electrical_work_limit(s,[0.,10.],3.65)

    def test_charge_storage_and_joule_share_full_ui(self):
        s=scene([[0.,2.],[10.,2.]],[[0.,4.],[10.,4.]])
        run={'time_s':[0.,5.,10.],'joule_energy_j':{'C2':[0.,30.,60.]},
             'chemistry':{'C2':{'electrical_chemical_input_j':[0.,15.,15.]}}}
        self.assertIn('full UI budget at 5.000 s',electrical_violation(run,s))


if __name__=='__main__':unittest.main()
