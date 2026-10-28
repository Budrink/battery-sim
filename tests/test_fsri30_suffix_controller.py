"""Orchestration-only tests. No CPU evolution and no CUDA invocation here."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_fsri30_bayes import synthetic_run
from fsri30_data import load
from fsri30_search_model import candidate,initial_point
from generate_fsri30 import build
import search_fsri30_suffix as worker


class SuffixControllerTests(unittest.TestCase):
    def test_retained_signal_does_not_require_another_ten_kelvin_per_search(self):
        rows=[dict(complete=True,constraint=-1.),dict(complete=False,constraint=1.)]
        with patch.object(worker,'useful_response',return_value=False):
            self.assertTrue(worker.supported_search(rows,True))
            self.assertFalse(worker.supported_search(rows,False))
            self.assertFalse(worker.supported_search([dict(complete=False)],True))

    def test_raw_trajectory_reuse_rejects_different_checkpoint_before_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);reference=root/'prior'/'trial';reference.mkdir(parents=True)
            (reference.parent/'contract.json').write_text(json.dumps(dict(native_sha256='core',checkpoint_sha256='old')))
            with patch.object(worker,'sha',side_effect=['core','new']):
                with self.assertRaisesRegex(ValueError,'different core, state or scene'):
                    worker.reuse_native_trial(reference,root/'new',{},root/'state.bin',worker.PROBE)
            self.assertFalse((root/'new').exists())

    def test_resume_pins_original_evidence_not_latest_pointer(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);ident='a'*64
            (out/'contract.json').write_text(json.dumps({'carried_seed_record':ident}))
            with patch.object(worker,'seed',return_value={'record':ident}) as fetch:
                result=worker.carried_profile(out,worker.model.candidate(
                    candidate(build(5945.),initial_point(),accuracy_level=1),worker.model.initial_point()),worker.PROBE)
            self.assertEqual(result['record'],ident)
            self.assertEqual(fetch.call_args.kwargs['record_id'],ident)
            with self.assertRaises(ValueError):worker.carried_profile(out,{},worker.PROBE,True)

    def inputs(self):
        scene=candidate(build(5945.),initial_point(),accuracy_level=1)
        full=synthetic_run(scene);count=len(full['time_s'])
        full.update(started_at_s=0.,electrical={},joule_energy_j={},
            process_totals={'gas_released_energy_j':[0.]*count},
            remaining_energy_j=[0.]*(30*count),source_damage=[0.]*(30*count),
            source_max_temperature_k=[298.15]*(30*count),channels={})
        cut=full['time_s'].index(1755.)
        prefix,suffix=deepcopy(full),deepcopy(full)
        def take(a,start,end):
            if isinstance(a,dict):return {k:take(v,start,end) for k,v in a.items()}
            return list(a[start:end])  # Native JSON contains lists, not NumPy arrays.
        for key in ('time_s','cell_mean_temperature_k','chemistry','gas','channels','electrical','joule_energy_j','process_totals'):
            prefix[key]=take(full[key],0,cut+1);suffix[key]=take(full[key],cut,None)
        for key in ('released_energy_j_history','remaining_energy_j','source_damage','source_max_temperature_k'):
            prefix[key]=full[key][:(cut+1)*30];suffix[key]=full[key][cut*30:]
        suffix.update(suffix_initial_state_bitwise_equal=True,partial=False,started_at_s=1755.,
            advanced_to_s=5945.,simulated_duration_s=4190.,wall_s=10.,
            solver_profile={'minimum_regular_step_s':.1},gas_film_trace={})
        return scene,prefix,suffix

    def test_conditioned_complete_suffix_is_not_full_warmup_validation(self):
        scene,prefix,suffix=self.inputs();data,_=load()
        with tempfile.TemporaryDirectory(prefix='fsri30-suffix-worker-') as temp:
            root=Path(temp)
            with patch.object(worker,'native',return_value=suffix):
                result=worker.evaluate(scene,root/'trial',prefix,root/'state.bin',root/'source.json',data,worker.PROBE,60.)
            self.assertTrue(result['complete'],result.get('error'))
            self.assertTrue(result['conditioned_transport_diagnostic'])
            self.assertFalse(result['full_warmup_validation'])
            self.assertFalse(result['calibrated'])
            self.assertFalse(result['validation_passed'])
            self.assertEqual(result['realtime_ratio'],419.)

    def test_unverified_initial_state_is_rejected_not_scored_as_good(self):
        scene,prefix,suffix=self.inputs();data,_=load()
        suffix['suffix_initial_state_bitwise_equal']=False
        with tempfile.TemporaryDirectory(prefix='fsri30-suffix-worker-') as temp:
            root=Path(temp)
            with patch.object(worker,'native',return_value=suffix):
                result=worker.evaluate(scene,root/'trial',prefix,root/'state.bin',root/'source.json',data,worker.PROBE,60.)
            self.assertFalse(result['complete'])
            self.assertIsNone(result['objective'])
            self.assertGreater(result['constraint'],0.)


if __name__=='__main__':unittest.main()
