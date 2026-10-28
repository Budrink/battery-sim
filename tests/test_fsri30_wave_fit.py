import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fsri30_wave_fit import POLICY, signatures, reference, assess_fields, match_groups


def observations(peaks, count, step=1., end=3600.):
    t=np.arange(0.,end+.01,step)
    y=np.full((len(t),count),298.15)
    for when,channels in peaks:
        for i in channels:
            y[:,i]+=800.*np.exp(-.5*((t-when)/12.)**2)
    return t,y,[str(i)for i in range(count)]


def target(peaks):
    t,y,_=observations(peaks,9)
    data=dict(time_s=t,runaway=t==1700.)
    data.update({f'T{i+1}_k':y[:,i]for i in range(9)})
    return reference(data)


class WaveTests(unittest.TestCase):
    def test_identical_sequence_accepts_without_equal_channel_counts(self):
        ref=target([(1800,[0]),(2140,[1,2,3]),(2580,[4,5]),(2960,[6,7,8])])
        actual=observations([(1800,[0]),(2140,list(range(1,10))),
                             (2580,list(range(10,20))),(2960,list(range(20,30)))],30,step=5)
        result=assess_fields(*actual,ref)
        self.assertTrue(result['accepted']);self.assertTrue(result['timing_target_met'])

    def test_late_peak_cannot_replace_earlier_wave(self):
        ref=target([(1800,[0]),(2140,[1,2,3]),(2580,[4,5,6,7,8])])
        result=assess_fields(*observations([(1800,[0]),(2580,list(range(1,30)))],30),ref)
        self.assertFalse(result['accepted']);self.assertTrue(result['pulse_match']['missing'])

    def test_early_primary_rejected_without_waiting_for_late_peak(self):
        ref=target([(1765,[0]),(2140,[1,2,3]),(2580,[4,5,6,7,8])])
        result=assess_fields(*observations([(1120,[0])],30,end=1200),ref)
        self.assertFalse(result['accepted']);self.assertIn('Primary',result['reason'])
        self.assertFalse(result['phase_window_complete']);self.assertIsNone(result['objective'])

    def test_unobserved_future_remains_pending(self):
        ref=target([(1800,[0]),(2140,[1,2,3]),(2580,[4,5,6,7,8])])
        result=assess_fields(*observations([],30,end=1700),ref)
        self.assertFalse(result['evaluated']);self.assertIsNone(result['accepted'])

    def test_one_wave_cannot_match_two(self):
        group=lambda t:dict(time_s=t,end_s=t,channels=['a'])
        result=match_groups([group(1800),group(1880)],[group(1840)],2200)
        self.assertEqual(result['matched'],1);self.assertEqual(len(result['missing']),1)

    def test_coarse_window_stops_on_missing_second_wave_but_not_early(self):
        ref=target([(1800,[0]),(2140,[1,2,3]),(2580,[4,5,6,7,8])])
        policy=replace(POLICY,timing_limit_s=250.)
        early=assess_fields(*observations([(1900,[0])],30,end=2400),ref,policy)
        self.assertIsNone(early['accepted'])
        expired=assess_fields(*observations([(1900,[0])],30,end=2460),ref,policy)
        self.assertFalse(expired['accepted'])
        self.assertTrue(expired['pulse_match']['missing'])

    def test_reheating_and_first_arrival_are_distinct(self):
        events=signatures(*observations([(1800,[0]),(2140,[0])],1))
        self.assertEqual(len(events['pulse_groups']),2)
        self.assertEqual(len(events['arrival_groups']),1)

    def test_real_data_contains_early_and_late_waves(self):
        from fsri30_data import load
        ref=reference(load()[0])
        self.assertEqual(len(ref['pulse_groups']),8)
        self.assertEqual(len(ref['arrival_groups']),6)
        self.assertEqual(ref['pulse_groups'][0]['time_s'],1763.)
        self.assertEqual(ref['pulse_groups'][1]['time_s'],2134.)
        self.assertGreater(ref['pulse_groups'][-1]['time_s'],2900.)


if __name__=='__main__':unittest.main()
