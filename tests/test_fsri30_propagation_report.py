from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import fsri30_propagation_patterns as report
from fsri30_prefix_history import trim_observed


def fixture():
    t=np.arange(0.,5945.+1e-7,5.)
    heat=np.where(t<1100.,300.,1000.)
    r3=np.where(t<1100.,0.,1000.)
    run=dict(time_s=t.tolist(),cell_mean_temperature_k={'B10':heat.tolist()},
        process_totals=dict(gas_released_energy_j=r3.tolist()),
        gas_groups=dict(outside=dict(maximum_temperature_k=heat.tolist())))
    events=[dict(component='B10',time_s=1100.)]
    scene=dict(metadata=dict(approximate_positions_m=dict(B10=[0.,0.,0.])))
    meta=dict(partial=False,last_observation_s=5945.,advanced_to_s=5945.)
    spatial=dict(air_volume_rms_relative_kelvin=0.,maximum_air_fuel_l1_relative=0.,
        maximum_air_oxygen_l1_relative=0.,same_node_maximum_local_relative_kelvin=0.,
        history=[dict(time_s=t,air_fuel_l1_relative=0.,normalized_fuel_distribution_total_variation=0.,
            air_volume_rms_relative_kelvin=0.) for t in (0.,5945.)])
    return run,events,meta,scene,spatial


class PropagationReportTests(unittest.TestCase):
    def produce(self,folder,bad_field=False):
        run,events,meta,scene,spatial=fixture()
        if bad_field:spatial['maximum_air_fuel_l1_relative']=.2
        items=[(run,events,meta,scene),(deepcopy(run),events,meta,scene),
            (trim_observed(run,2240.),events,dict(partial=True),scene)]
        with patch.object(report,'verify_fine_reference'),patch.object(report,'load_available',side_effect=items),\
             patch.object(report,'compare_spatial_fields',return_value=spatial),\
             patch.object(report,'load',return_value=({},None)),patch.object(report,'plot'),\
             patch.object(report,'plot_patterns',return_value='diagnostic.png'):
            return report.write_report(Path('source'),Path('fine'),Path(folder)/'comparison')

    def test_full_comparison_automatically_reports(self):
        with tempfile.TemporaryDirectory() as folder:
            summary=self.produce(folder)
            value=json.loads((Path(folder)/'comparison/report.json').read_text())
        self.assertTrue(summary['spatially_converged'])
        self.assertTrue(value['prefix_repeat']['accepted'])
        self.assertIn('cascade',value['phase_comparisons'])
        self.assertIn('fixed-5700s',value['phase_comparisons'])
        self.assertFalse(value['validation_passed']);self.assertFalse(value['patterns_promoted'])

    def test_aggregate_pass_cannot_hide_redistributed_gas(self):
        with tempfile.TemporaryDirectory() as folder:
            summary=self.produce(folder,bad_field=True)
            value=json.loads((Path(folder)/'comparison/report.json').read_text())
        self.assertTrue(value['macroscopic_comparison_accepted'])
        self.assertFalse(summary['spatially_converged'])
        self.assertFalse(value['field_pattern_qualification']['accepted'])

    def test_partial_run_is_not_full_reference(self):
        run,events,meta,scene,_=fixture();meta['partial']=True
        with tempfile.TemporaryDirectory() as folder,patch.object(report,'verify_fine_reference'),\
                patch.object(report,'load_available',return_value=(run,events,meta,scene)):
            with self.assertRaisesRegex(ValueError,'incomplete reference'):
                report.write_report(Path('source'),Path('fine'),Path(folder)/'comparison')


if __name__=='__main__':unittest.main()
