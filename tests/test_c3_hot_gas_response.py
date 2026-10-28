import copy
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from resolve_c3_heat_transfer import candidate,hot_law,reference,warm_surface_law,warm_mixing_law,ambient_mixing_law,wall_mass_scene


class HotGasResponse(unittest.TestCase):
    def test_wall_mass_probe_changes_only_the_vessel_inventory(self):
        s=dict(cells=[dict(id='C2',thermal={'heat_capacity_j_per_k':400.}),
                      dict(id='chamber-wall',thermal={'heat_capacity_j_per_k':5000.,'conductivity_w_per_m_k':[45.]*3})],
               gas_network={'frozen':True},interfaces=['fixed'],shells=['fixed'])
        original=copy.deepcopy(s);out=wall_mass_scene(s,4.)
        self.assertEqual(s,original)
        out.pop('metadata')
        out['cells'][1]['thermal']['heat_capacity_j_per_k']=5000.
        self.assertEqual(out,s)
        for bad in (0.,.5,9.,float('nan')):
            with self.assertRaises(ValueError):wall_mass_scene(s,bad)

    def test_ambient_convection_retains_reference_but_does_not_stall_at_350K(self):
        law,profile=ambient_mixing_law()
        self.assertAlmostEqual(reference(law),1.)
        gains=[]
        for _,response in profile['temperature_log_response']:
            z=law['geometry_reference_logit']+response
            gains.append(law['background_conductance_w_per_k']+law['active_conductance_w_per_k']/(1.+math.exp(-z)))
        for got,want in zip(gains,(1.,10.,100.,1000.,1000.,3000.,9000.)):
            self.assertAlmostEqual(got,want)

    def test_warm_convection_has_no_artificial_450K_cutoff(self):
        law,profile=warm_mixing_law()
        self.assertAlmostEqual(reference(law),1.)
        for (t,response),gain in zip(profile['temperature_log_response'],(1.,1.,1000.,10000.,30000.,90000.)):
            z=law['geometry_reference_logit']+response
            g=law['background_conductance_w_per_k']+law['active_conductance_w_per_k']/(1.+math.exp(-z))
            self.assertAlmostEqual(g,gain)
        self.assertEqual(profile['temperature_log_response'][:2],[[298.15,0.],[350.,0.]])

    def test_reverse_boundary_is_explicit_not_a_bulk_material_refit(self):
        law,profile=hot_law(.2,30.,2.,6.)
        original=dict(cells=[dict(thermal={'surface_connectivity':profile})],gas_network=dict(
            compartments=[],channels=[],thermal_contacts=[dict(port_id='ordinary',connectivity=law),
                dict(port_id='chamber-wall:1',connectivity=law)]))
        out=candidate(original,True,True,warm_surface=True,reverse_surface=True)
        self.assertEqual(out['cells'],original['cells'])
        self.assertNotIn('solid_surface_connectivity',original['gas_network']['thermal_contacts'][0])
        self.assertIn('solid_surface_connectivity',out['gas_network']['thermal_contacts'][0])
        self.assertEqual(out['gas_network']['thermal_contacts'][1],original['gas_network']['thermal_contacts'][1])

    def test_warm_flow_keeps_cold_reference(self):
        law,profile=warm_surface_law(.7)
        self.assertAlmostEqual(reference(law),.7)
        for row,gain in zip(profile['temperature_log_response'],(1.,1.,100.,600.,1000.,1500.)):
            z=law['geometry_reference_logit']+row[1]
            g=law['background_conductance_w_per_k']+law['active_conductance_w_per_k']/(1+math.exp(-z))
            self.assertAlmostEqual(g,.7*gain)
    def test_reference_and_targets(self):
        law,p=hot_law(.7,10000.,300.,3000.)
        self.assertAlmostEqual(reference(law),.7)
        z=law['geometry_reference_logit']
        for row,gain in zip(p['temperature_log_response'],(1.,1.,1.,300.,3000.)):
            g=law['background_conductance_w_per_k']+law['active_conductance_w_per_k']/(1+math.exp(-z-row[1]))
            self.assertAlmostEqual(g,.7*gain)

    def test_only_hot_gas_laws_change(self):
        law,p=hot_law(.2,30.,2.,6.)
        s=dict(cells=['fixed'],shells=['fixed'],interfaces=['fixed'],gas_network=dict(
            compartments=['fixed'],channels=['fixed'],surface_connectivity=p,mixing_connectivity=law,
            thermal_contacts=[dict(port_id='ordinary',connectivity=law,gas_surface_connectivity=p),
                              dict(port_id='chamber-wall:1',connectivity=law,gas_surface_connectivity=p)]))
        original=copy.deepcopy(s);out=candidate(s,True,True)
        self.assertEqual(s,original)
        self.assertEqual(out['cells'],s['cells']);self.assertEqual(out['shells'],s['shells'])
        self.assertEqual(out['interfaces'],s['interfaces'])
        for k in ('compartments','channels'):self.assertEqual(out['gas_network'][k],s['gas_network'][k])
        self.assertEqual(out['gas_network']['thermal_contacts'][1],s['gas_network']['thermal_contacts'][1])
        self.assertAlmostEqual(reference(out['gas_network']['thermal_contacts'][0]['connectivity']),reference(law))
        self.assertEqual(out['gas_network']['surface_connectivity']['temperature_log_response'][:3],
                         [[298.15,0.],[350.,0.],[450.,0.]])


if __name__=='__main__':unittest.main()
