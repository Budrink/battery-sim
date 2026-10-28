import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fsri30_conditioned_start import join_observations, same_series
import fsri30_suffix_model as model
from fsri30_search_model import candidate as initial_scene, initial_point
from generate_fsri30 import build


class SuffixTests(unittest.TestCase):
    def test_sensitivity_controls_preserve_other_learned_parameters(self):
        anchor=model.point(shared_gap_fraction=.31,exhaust_area_scale=.08,
            passage_area_scale=.4,hot_contact_span=23.,mixing_flow_fraction=.7)
        values=model.decode(anchor)
        experiments=dict(model.experiments(anchor))
        for key in ('exhaust_area_scale','passage_area_scale','hot_contact_span','mixing_flow_fraction'):
            self.assertAlmostEqual(model.decode(experiments['shared-facing-cavities'])[key],values[key])
        self.assertAlmostEqual(model.decode(experiments['hot-directed-contact'])['shared_gap_fraction'],.31)

    def test_fixed_volume_extremes_cannot_reset_vented_gas(self):
        base=initial_scene(build(5945.),initial_point(),accuracy_level=1)
        for x in [model.initial_point(),np.zeros(len(model.SPACE)),np.ones(len(model.SPACE))]+[x for _,x in model.experiments()]:
            s=model.candidate(base,x)
            self.assertEqual(base['gas_network']['compartments'],s['gas_network']['compartments'])
            self.assertEqual(base['ports'],s['ports'])
            self.assertEqual(base['cells'],s['cells'])
            self.assertEqual(base['shells'],s['shells'])
            a=[e for e in base['gas_network']['channels'] if e['kind']!='passage']
            b=[e for e in s['gas_network']['channels'] if e['kind']!='passage']
            self.assertEqual(a,b)
            self.assertTrue(s['metadata']['conditioned_transport_diagnostic'])
            self.assertFalse(s['metadata']['full_warmup_validation'])

    def pair(self):
        def trajectory(t):
            r=dict(time_s=t,started_at_s=t[0],partial=False)
            for section in ('cell_mean_temperature_k','gas','chemistry','channels','joule_energy_j','process_totals','electrical'):
                r[section]={'value':list(t)}
            for section in ('released_energy_j_history','remaining_energy_j','source_damage','source_max_temperature_k'):
                r[section]=[time for time in t for _ in range(30)]
            r['gas_film_trace']={'suffix_only':[0.,1.]}
            return r
        return trajectory([0.,1750.,1755.]),trajectory([1755.,1760.,1800.])

    def test_join_keeps_conditioning_explicit_and_does_not_forge_film_history(self):
        prefix,suffix=self.pair();r=join_observations(prefix,suffix)
        self.assertEqual(r['time_s'],[0.,1750.,1755.,1760.,1800.])
        self.assertEqual(r['cell_mean_temperature_k']['value'],r['time_s'])
        self.assertEqual(len(r['released_energy_j_history']),5*30)
        self.assertEqual(r['gas_film_trace'],suffix['gas_film_trace'])
        self.assertTrue(r['conditioned_transport_diagnostic'])
        self.assertFalse(r['full_warmup_validation'])

    def test_join_rejects_temperature_mass_chemistry_and_rupture_jumps(self):
        for field in ('cell_mean_temperature_k','gas','chemistry','channels'):
            a,b=self.pair();b[field]['value'][0]+=1.
            with self.assertRaisesRegex(ValueError,'physical state jump'):
                join_observations(a,b)
        a,b=self.pair();b['time_s'][0]+=1.
        with self.assertRaises(ValueError):join_observations(a,b)

    def test_actual_film_heat_and_complete_gas_sum_are_used(self):
        run=dict(time_s=[1755.,1800.],process_totals={'gas_released_energy_j':[12.,21.]},
            gas_film_trace={'contact':dict(solid='B02:case:+x',heat_to_solid_j=[0.,5.],
                positive_step_heat_j=[0.,7.],negative_step_heat_j=[0.,-2.])})
        r=model.energy_delivery(run)
        self.assertEqual(r['shells']['B02'],dict(net_j=5.,positive_j=7.,negative_j=-2.))
        self.assertEqual(r['global_gas_reaction_heat_j'],9.)
        self.assertTrue(r['all_gas_volumes_included'])


if __name__=='__main__':unittest.main()
