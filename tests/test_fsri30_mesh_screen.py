from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fsri30_mesh_screen import assess, window, match, primary_bias_from_pair, uncertain_temperature_window
from fsri30_sequence_model import initial, suggest
from fsri30_wave_fit import assess_fields
from test_fsri30_wave_fit import observations,target

EVIDENCE=dict(primary_bias_s=dict(pulse_groups=143.,arrival_groups=145.),
              primary_uncertainty_s=100.,unmeasured_allowance_s=150.)


class MeshScreenTests(unittest.TestCase):
    def test_late_coarse_primary_can_trigger_refinement_not_false_success(self):
        ref=target([(1800,[0]),(2300,[1,2]),(2800,[3,4,5])])
        actual=observations([(1946,[0]),(2300,[1,2]),(2800,[3,4,5])],30)
        before=deepcopy(ref);before_y=actual[1].copy()
        coarse=assess(*actual,ref,80,EVIDENCE)
        self.assertTrue(coarse['accepted'])
        self.assertFalse(coarse['raw_assessment']['accepted'])
        self.assertTrue(coarse['fine_grid_verification_required'])
        self.assertFalse(coarse['timing_target_met'])
        self.assertAlmostEqual(coarse['primary_offset_s'],146.)
        self.assertAlmostEqual(coarse['search_primary_offset_s'],3.)
        self.assertEqual(ref,before)
        self.assertTrue((actual[1]==before_y).all())

    def test_fine_grid_policy_is_exactly_unchanged(self):
        ref=target([(1800,[0]),(2300,[1,2])])
        actual=observations([(1946,[0]),(2300,[1,2])],30)
        self.assertEqual(assess(*actual,ref,20,EVIDENCE),assess_fields(*actual,ref))
        self.assertFalse(assess(*actual,ref,20,EVIDENCE)['accepted'])

    def test_missing_window_waits_for_mesh_delay_without_extrapolation(self):
        ref=target([(1800,[0])])
        result=assess(*observations([],30,end=1950.),ref,80,EVIDENCE)
        self.assertIsNone(result['accepted'])
        self.assertGreater(result['primary_deadline_s'],2100.)
        result=assess(*observations([],30,end=2250.),ref,80,EVIDENCE)
        self.assertFalse(result['accepted'])

    def test_gross_early_primary_still_rejects(self):
        ref=target([(1800,[0])])
        result=assess(*observations([(1260,[0])],30,end=1350.),ref,80,EVIDENCE)
        self.assertFalse(result['accepted'])

    def test_no_invented_bias_for_other_waves_or_40mm(self):
        self.assertEqual(window(80,'pulse_groups',1,EVIDENCE),(0.,250.))
        self.assertEqual(window(40,'pulse_groups',0,EVIDENCE),(0.,250.))

    def test_one_wave_cannot_fill_two_even_with_wider_windows(self):
        group=lambda t:dict(time_s=t,end_s=t,channels=['B10'])
        result=match([group(1800),group(1880)],[group(1946)],2500,80,'pulse_groups',EVIDENCE)
        self.assertEqual(result['matched'],1)
        self.assertEqual(len(result['missing']),1)

    def test_pair_requires_same_model_primary(self):
        a=dict(model={k:[dict(time_s=1261,channels=['B10'])] for k in ('pulse_groups','arrival_groups')})
        b=dict(model={k:[dict(time_s=1118,channels=['B10'])] for k in ('pulse_groups','arrival_groups')})
        self.assertEqual(primary_bias_from_pair(a,b)['primary_bias_s']['pulse_groups'],143.)
        b['model']['pulse_groups'][0]['channels']=['B11']
        with self.assertRaises(ValueError):primary_bias_from_pair(a,b)

    def test_temperature_guard_also_respects_coarse_event_uncertainty(self):
        ref=target([(1800,[0])])
        self.assertTrue(uncertain_temperature_window(2050.,ref,80,EVIDENCE))
        self.assertFalse(uncertain_temperature_window(2050.,ref,20,EVIDENCE))
        self.assertFalse(uncertain_temperature_window(2400.,ref,80,EVIDENCE))

    def test_search_moves_toward_delayed_coarse_target_not_zero_raw_error(self):
        from calibrate_fsri30_sequence import BASE
        if not BASE.exists():self.skipTest('Retained FSRI fixture not installed')
        base=json.loads(BASE.read_text());p=initial(base);p['r2_rate_scale']=.049
        row=dict(parameters=p,trial='observed-coarse',assessment=dict(primary_offset_s=40.,
                 search_primary_offset_s=-103.,primary_search_tolerance_s=50.,last_observed_s=2360.))
        next_p,_=suggest([row],base)
        self.assertLess(next_p['r2_rate_scale'],p['r2_rate_scale'])
        # Under the old zero-offset target the same measured +40 s requested
        # a FASTER reaction. The mesh-aware search must do the opposite.
        self.assertEqual(next_p['gas_can_contact_scale'],p['gas_can_contact_scale'])


if __name__=='__main__':unittest.main()
