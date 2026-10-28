from copy import deepcopy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from check_fsri30_closed_exhausts import close_exhausts,exits
from fsri30_search_model import candidate,initial_point
from generate_fsri30 import build


class ClosedExhaustTests(unittest.TestCase):
    def test_only_exact_external_cut_changes_not_internal_or_casing(self):
        scene=candidate(build(5945.),initial_point(),accuracy_level=1);snapshot=deepcopy(scene)
        closed=close_exhausts(scene);cut=set(exits(scene))
        self.assertEqual(scene,snapshot);self.assertEqual(len(cut),30)
        for a,b in zip(scene['gas_network']['channels'],closed['gas_network']['channels']):
            if a['id'] in cut:
                self.assertEqual(b['area_m2'],0.);b=deepcopy(b);b['area_m2']=a['area_m2']
            self.assertEqual(a,b)
        closed['metadata']=scene['metadata'];closed['gas_network']['channels']=scene['gas_network']['channels']
        self.assertEqual(scene,closed)

    def test_unaccounted_exterior_path_is_rejected(self):
        scene=candidate(build(5945.),initial_point(),accuracy_level=1)
        extra=deepcopy(scene['gas_network']['channels'][-1]);extra.update(id='unresolved',**{'from':'body-00','to':'hood'})
        scene['gas_network']['channels'].append(extra)
        with self.assertRaisesRegex(ValueError,'Unresolved exterior crossing'):close_exhausts(scene)


if __name__=='__main__':unittest.main()
