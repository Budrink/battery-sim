import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import upv_frozen_validation as validation

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/calibration/upv-separated-chemistry-v6/five-channel-calibration-20261022-r1/hot-contact400-response-search/trial-000/scenario.json'


class FrozenValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SOURCE.exists():raise unittest.SkipTest('Optional local frozen calibration scene')
        cls.source=json.loads(SOURCE.read_text())

    def test_training_roundtrip_keeps_all_physical_laws(self):
        old=validation.digest(self.source)
        target,_=validation.build(self.source,'no-cooling')
        for key in ('cells','shells','interfaces','boundaries','gas_network','ports','simulation'):
            self.assertTrue(validation._same(self.source[key],target[key]),key)
        self.assertEqual(validation.digest(self.source),old)

    def test_cooled_transfer_keeps_common_laws_and_rebuilds_voids(self):
        target,_=validation.build(self.source)
        for old in self.source['cells']:
            now=next(c for c in target['cells']if c['id']==old['id'])
            old=copy.deepcopy(old);now=copy.deepcopy(now)
            old['thermal'].pop('initial_temperature_k');now['thermal'].pop('initial_temperature_k')
            if 'electrical'in old:
                for key in ('current_profile_a','voltage_profile_v'):
                    old['electrical'].pop(key);now['electrical'].pop(key)
            self.assertEqual(old,now)
        frozen={e['id']:e for e in target['interfaces']}
        self.assertTrue(all(frozen[e['id']]==e for e in self.source['interfaces']))
        self.assertEqual(target['gas_network']['mixing_connectivity'],self.source['gas_network']['mixing_connectivity'])
        groups=validation._contact_groups(self.source)
        for key,value in validation._contact_groups(target).items():
            self.assertTrue(validation._same(value,groups[key]),key)
        self.assertEqual(target['metadata']['frozen_validation']['removed_blocked_nonvoid_routes'],27)
        self.assertFalse(target['metadata']['frozen_validation']['parameters_refitted'])
        self.assertIn('cooling-plate',{c['id']for c in target['cells']})
        from gas_void_geometry import cell_box
        plate=next(c for c in target['cells']if c['id']=='cooling-plate');lo,hi=cell_box(plate)
        for box in target['metadata']['gas_geometry']['void_boxes'].values():
            self.assertFalse(all(min(hi[k],box['hi'][k])-max(lo[k],box['lo'][k])>1e-11 for k in range(3)))
        volume=sum(n['volume_m3']for n in target['gas_network']['compartments']if not n['id'].startswith('inside-'))
        self.assertAlmostEqual(volume,.120,places=11)

    def test_unknown_modified_void_route_is_not_silently_reset(self):
        source=copy.deepcopy(self.source)
        edge=next(e for e in source['gas_network']['channels']if e['id'].startswith('void-face-'))
        edge['opening_pressure_pa']+=1.
        with self.assertRaisesRegex(ValueError,'Unsupported modified free-space face'):validation.build(source)

    def test_current_setpoint_is_explicit_and_never_applied_to_training(self):
        target,_=validation.build(self.source,voltage_limited_setpoint=True)
        strict,_=validation.build(self.source)
        for now,old in zip(target['cells'],strict['cells']):
            now=copy.deepcopy(now)
            if 'electrical'in now:self.assertEqual(now['electrical'].pop('current_input'),'voltage_limited_setpoint')
            self.assertEqual(now,old)
        with self.assertRaisesRegex(ValueError,'measured training current'):
            validation.build(self.source,'no-cooling',voltage_limited_setpoint=True)

    def test_transfer_uses_actual_new_directed_laws_not_metadata_defaults(self):
        from upv_final_fit import candidate
        source,_=candidate(self.source,dict(core_surface_power=4.,
            core_contact_damage=6.,casing_damage_contrast=-6.,
            casing_gas_damage=-4.5,damaged_iss_scale=.1,
            intact_iss_scale=3.,iss_temperature_contrast=-4.,damaged_tim_scale=.2,
            tim_al_scale=6., resistance_scale=1.01,
            resistance_temperature_delta=.003, resistance_charge_delta=-1.,
            r2_activation_delta=-20000., r2_rate=1.35,
            product_conductivity_scale=3., wall_hot_knee_k=700.))
        target,_=validation.build(source)
        cells={c['id']:c for c in target['cells']}
        for old in source['cells']:
            a=copy.deepcopy(old['thermal']);b=copy.deepcopy(cells[old['id']]['thermal'])
            a.pop('initial_temperature_k');b.pop('initial_temperature_k')
            self.assertEqual(a,b)
            if 'conversion'in old:
                self.assertEqual(old['conversion'],cells[old['id']]['conversion'])
            if 'electrical'in old:
                a=copy.deepcopy(old['electrical'])
                b=copy.deepcopy(cells[old['id']]['electrical'])
                for name in ('current_profile_a','voltage_profile_v'):
                    a.pop(name);b.pop(name)
                self.assertEqual(a,b)
        shells={s['contents_id']:s for s in target['shells']}
        for old in source['shells']:
            self.assertEqual(old['surface_connectivity'],shells[old['contents_id']]['surface_connectivity'])
        interfaces={e['id']:e for e in target['interfaces']}
        for old in source['interfaces']:
            self.assertEqual(old,interfaces[old['id']])
        expected=validation._contact_groups(source)
        for key,value in validation._contact_groups(target).items():
            self.assertTrue(validation._same(value,expected[key]),key)

    def test_new_cooling_plate_inherits_same_al_response_but_not_capacity(self):
        from upv_final_fit import candidate
        source,_=candidate(self.source,dict(al_surface_softening=2.))
        target,_=validation.build(source)
        old=next(c for c in source['cells']if c['id']=='Al1')
        plate=next(c for c in target['cells']if c['id']=='cooling-plate')
        self.assertEqual(plate['material_id'],old['material_id'])
        for key in ('surface_connectivity','bulk_connectivity','bulk_response'):
            self.assertEqual(plate['thermal'][key],old['thermal'][key])
        self.assertNotEqual(plate['thermal']['heat_capacity_j_per_k'],old['thermal']['heat_capacity_j_per_k'])
        self.assertIn('cooling-plate',target['metadata']['frozen_validation']['new_objects_inheriting_material_profiles'])


if __name__=='__main__':unittest.main()
