from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fsri30_balance import suggest_balance,timing_compensation
from fsri30_cooling_fit import observation,burst_suggest,encode


def row(e,m,bias=None,full=True,valid=True):
    out=dict(trial=f'e{e}-m{m}',parameters=dict(effective_emissivity=e,thermal_stirring_coefficient=m),
        full_trajectory=full,burst=dict(accepted=valid,phase_window_complete=True,timing_target_met=valid,
            timing_error_s=0.,timing_offsets_s=dict(peak_time_s=0.,rise50_time_s=0.,rise90_time_s=0.)))
    if bias is not None:out['full_median_bias_k']=bias
    return out


class ThermalBalanceTests(unittest.TestCase):
    def test_only_cold_full_tail_tests_a_lower_intermediate_emissivity(self):
        p,how=suggest_balance([row(.8,.07,-100.)])
        self.assertAlmostEqual(p['effective_emissivity'],.41)
        self.assertIn('expand',how['method']);self.assertIn('NOT yet',how['note'])

    def test_actual_hot_cold_signs_give_interior_secant(self):
        p,how=suggest_balance([row(.2,.07,100.),row(.8,.07,-100.)])
        self.assertAlmostEqual(p['effective_emissivity'],.5)
        self.assertIn('measured-full-tail',how['method'])

    def test_partial_tail_and_bad_burst_cannot_supply_a_full_tail_bracket(self):
        cold=row(.8,.07,-100.)
        for hot in (row(.2,.07,100.,full=False),row(.2,.07,100.,valid=False)):
            p,how=suggest_balance([cold,hot]);self.assertAlmostEqual(p['effective_emissivity'],.41)
            self.assertIn('expand',how['method'])

    def test_no_full_tail_does_not_invent_a_sign_or_temperature(self):
        self.assertIsNone(suggest_balance([row(.8,.07,full=False)]))

    def test_timing_compensation_uses_measured_two_dimensional_response(self):
        center=row(.8,.07,-100.);rows=[center]
        for e,m in ((.7,.07),(.8,.074),(.8,.067)):
            r=row(e,m);value=100.*(e-.8)+600.*np.log(m/.07)
            r['burst']['timing_offsets_s']={k:float(value)for k in ('peak_time_s','rise50_time_s','rise90_time_s')}
            rows.append(r)
        p,how=timing_compensation(rows,.4,center)
        self.assertGreater(p['thermal_stirring_coefficient'],.07)
        self.assertLess(p['thermal_stirring_coefficient'],.14)
        self.assertTrue(how['proposal_only']);self.assertTrue(how['actual_burst_gate_required'])
        self.assertLess(max(abs(x)for x in how['predicted_timing_offsets_s'].values()),.1)

    def test_measured_balance_not_repeated_indefinitely(self):
        self.assertIsNone(suggest_balance([row(.5,.07,1.)]))

    def test_measured_same_boundary_bracket_overrides_broad_regression(self):
        a=row(.3932387554878369,.07452091483809048,full=False)
        b=row(.3932387554878369,.0786871284805109,full=False)
        for r,values in ((a,(-10.,-15.99516294665,-14.83876452453)),(b,(24.,17.43439752302,19.49975999866))):
            r['burst']['timing_offsets_s']=dict(zip(('peak_time_s','rise50_time_s','rise90_time_s'),values))
        params,how=timing_compensation([a,b],a['parameters']['effective_emissivity'],b)
        self.assertAlmostEqual(params['thermal_stirring_coefficient'],.07616407585243641)
        self.assertIn('phase-secant',how['method'])
        self.assertLess(max(abs(x)for x in how['predicted_timing_offsets_s'].values()),4.)
        self.assertTrue(how['actual_burst_gate_required'])

    def test_bracket_on_another_boundary_cannot_supply_local_phase_secant(self):
        a=row(.4,.07,full=False);b=row(.5,.08,full=False)
        a['burst']['timing_offsets_s']={k:-20. for k in ('peak_time_s','rise50_time_s','rise90_time_s')}
        b['burst']['timing_offsets_s']={k:20. for k in ('peak_time_s','rise50_time_s','rise90_time_s')}
        params,how=timing_compensation([a,b],.4,a)
        self.assertAlmostEqual(params['thermal_stirring_coefficient'],.07)
        self.assertFalse(how['timing_prediction_available'])

    def test_signed_bias_requires_actual_full_tail(self):
        data=dict(time_s=np.arange(5946,dtype=float),runaway=np.arange(5946)==1700)
        for i in range(1,10):data[f'T{i}_k']=np.full(5946,500.)
        def run(times):
            return dict(time_s=times,partial=False,cell_mean_temperature_k={
                f'B{i:02d}':[480.]*len(times)for i in range(30)})
        full=observation(run([0.,3600.,5945.]),data)
        self.assertTrue(full['full_trajectory']);self.assertAlmostEqual(full['full_median_bias_k'],-20.)
        for times in ([0.,3600.,4000.],[3600.,5945.]):
            partial=observation(run(times),data)
            self.assertFalse(partial['full_trajectory']);self.assertNotIn('full_median_bias_k',partial)

    def test_gp_never_trains_invented_tail_or_wrong_burst_as_full_success(self):
        good=row(.8,.07,-100.);good['full_cooling_rmse_k']=100.
        partial=row(.4,.075,full=False);partial['anchor_rmse_k']=0.
        delayed=row(.3,.08,-10.);delayed['burst'].update(timing_target_met=False,timing_error_s=60.)
        delayed['full_cooling_rmse_k']=10.
        rows=[good,partial,delayed]
        for r in rows:r['burst'].update(evaluated=True,constraint=0.)
        with patch('fsri30_cooling_fit.propose',return_value=(encode(good['parameters']),{}))as gp:
            burst_suggest(rows,full_tail=True)
        observations=gp.call_args.args[0]
        self.assertEqual(observations[0]['objective'],100.);self.assertTrue(observations[0]['complete'])
        for o in observations[1:]:self.assertFalse(o['complete']);self.assertIsNone(o['objective'])
        self.assertGreater(observations[2]['constraint'],0.)


if __name__=='__main__':unittest.main()
