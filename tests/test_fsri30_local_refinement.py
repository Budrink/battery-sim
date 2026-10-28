from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from check_fsri30_local_refinement import local_scene


class LocalRefinementContract(unittest.TestCase):
    def fixture(self):
        s={k:[] for k in ('cells','interfaces','boundaries','shells','primary_failures','heating_protocols')}
        s['cells']=[{'id':'battery','cp':1000.}]
        s['simulation']={'duration_s':5945.,'minimum_dt_s':.01,'max_dt_s':.1}
        s['failure_sample_times_s']=[0.,1100.,1230.,2525.,5945.]
        s['gas_network']={'thermal_stirring_coefficient':.05,'thermal_stirring_length_m':.02,
            'compartments':[{'id':'body-00','volume_m3':.1},
                {'id':'air-0','volume_m3':1.,'initial_mass_kg':1.,'initial_oxidizer_mass_fraction':.232}],
            'channels':[],'thermal_contacts':[]}
        return s

    def invoke(self,s,refined):
        with patch('check_fsri30_local_refinement.build_resolved',return_value=refined):
            return local_scene({},s,.01,2525.)

    def test_same_volume_inventory_and_frozen_physics(self):
        s=self.fixture();r=deepcopy(s);air=r['gas_network']['compartments'].pop()
        for i in range(8):
            n=deepcopy(air);n.update(id='air-'+str(i),volume_m3=1./8,initial_mass_kg=1./8)
            r['gas_network']['compartments'].append(n)
        result=self.invoke(s,r)
        self.assertEqual(result['cells'],s['cells'])
        self.assertEqual(result['simulation']['minimum_dt_s'],.01)
        self.assertEqual(result['failure_sample_times_s'][-1],2525.)
        self.assertNotIn('reactive_reconstruction',result['gas_network'])

    def test_reject_material_change(self):
        s=self.fixture();r=deepcopy(s);r['cells'][0]['cp']=900.
        with self.assertRaises(ValueError):self.invoke(s,r)

    def test_reject_chemistry_change(self):
        s=self.fixture();r=deepcopy(s);r['gas_network']['compartments'][0]['volume_m3']=.2
        with self.assertRaises(ValueError):self.invoke(s,r)

    def test_reject_extra_air_or_oxygen(self):
        for key,value in [('initial_mass_kg',2.),('initial_oxidizer_mass_fraction',.5)]:
            s=self.fixture();r=deepcopy(s);r['gas_network']['compartments'][-1][key]=value
            with self.assertRaises(ValueError):self.invoke(s,r)


if __name__=='__main__':unittest.main()
