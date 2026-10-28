import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from upv_boundary_transport import candidate
from upv_gas_access import AREA_COEFFICIENTS


class BoundaryTransport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).resolve().parents[1]/'data/calibration/upv-separated-chemistry-v6/damaged-contact-20261022-r1/trial-002/scenario.json'
        if not path.exists():raise unittest.SkipTest('Optional current geometry')
        cls.source=json.loads(path.read_text())

    def test_neutral_and_no_input_mutation(self):
        source=copy.deepcopy(self.source);trial=candidate(source)
        self.assertEqual(source,self.source)
        for key in source:
            if key!='metadata':self.assertEqual(source[key],trial[key])

    def test_only_counterexchange_path_and_shared_wet_area_change(self):
        trial=candidate(self.source,2.,.6)
        for key in self.source:
            if key not in ('metadata','gas_network'):self.assertEqual(self.source[key],trial[key])
        a,b=self.source['gas_network'],trial['gas_network']
        for key in a:
            if key not in ('channels','thermal_contacts'):self.assertEqual(a[key],b[key])
        for before,after in zip(a['channels'],b['channels']):
            expected=copy.deepcopy(before)
            if expected['kind']=='rupture':expected['mixing_length_m']*=2.
            self.assertEqual(expected,after)
        for before,after in zip(a['thermal_contacts'],b['thermal_contacts']):
            expected=copy.deepcopy(before)
            if before['compartment'].startswith(('gap-ISS1-','gap-ISS2-')):
                ratio=.6/self.source['metadata']['gas_access_geometry']['jet_wetted_fraction']
                for key in AREA_COEFFICIENTS:expected['connectivity'][key]*=ratio
            self.assertEqual(expected,after)

    def test_unphysical_bounds_and_compounding_are_rejected(self):
        for scale,jet in ((0.,.5),(11.,.5),(1.,0.),(1.,1.1),(float('nan'),.5)):
            with self.assertRaises(ValueError):candidate(self.source,scale,jet)
        with self.assertRaises(ValueError):candidate(candidate(self.source),2.,.5)

    def test_explicit_relative_continuation_keeps_current_geometry(self):
        current=candidate(self.source,2.,.6)
        trial=candidate(current,3.,.6,relative_to_current=True)
        for before,after in zip(current['gas_network']['channels'],trial['gas_network']['channels']):
            expected=copy.deepcopy(before)
            if expected['kind']=='rupture':expected['mixing_length_m']*=3.
            self.assertEqual(expected,after)
        self.assertEqual(current['gas_network']['thermal_contacts'],trial['gas_network']['thermal_contacts'])
        self.assertEqual(trial['metadata']['boundary_transport_fit']['previous_fit'],current['metadata']['boundary_transport_fit'])


if __name__=='__main__':unittest.main()
