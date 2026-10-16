"""Diagnostic isolation and conservation; not acceptance of the coarse solver."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_fixed_gas_network as audit


class FixedGasAudit(unittest.TestCase):
    def test_pressure_controls_schema(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest('Optional jsonschema not installed')
        schema=json.loads((audit.ROOT/'schemas/scenario-v0.schema.json').read_text())
        validator=jsonschema.Draft202012Validator({'$defs':schema['$defs'],'$ref':'#/$defs/gas_network'})
        gas=audit.chain()['gas_network']
        gas.update(pressure_relaxation_fraction=.2,pressure_absolute_tolerance_pa=1.,pressure_relative_tolerance=1e-5)
        validator.validate(gas)
        for key,value in [('pressure_relaxation_fraction',0),('pressure_relaxation_fraction',.6),
                          ('pressure_absolute_tolerance_pa',0),('pressure_relative_tolerance',-1)]:
            with self.subTest(key=key,value=value),self.assertRaises(jsonschema.ValidationError):
                validator.validate(dict(gas,**{key:value}))

    def test_chain_has_no_sources_or_changing_geometry(self):
        s=audit.chain();g=s['gas_network']
        self.assertEqual(len(g['compartments']),3)
        self.assertEqual(len(g['channels']),2)
        self.assertEqual(g['thermal_contacts'],[])
        self.assertTrue(all(c['kind']=='passage' and 'damage_opening' not in c for c in g['channels']))
        pressures=[n['initial_mass_kg']*g['specific_r_j_per_kg_k']*n['initial_temperature_k']/n['volume_m3'] for n in g['compartments']]
        for a,b in zip(pressures,[100100.,100000.,100000.]):self.assertAlmostEqual(a,b,places=8)

    def test_snapshot_preserves_shared_gas_state_and_routes(self):
        if not (audit.DEFAULT_SOURCE/'run.json').exists():self.skipTest('UPV diagnostic snapshot is not installed')
        s=audit.snapshot(audit.DEFAULT_SOURCE,1.1);g=s['gas_network']
        source=json.loads((audit.DEFAULT_SOURCE/'scenario.json').read_text())
        self.assertEqual(len(g['compartments']),len(source['gas_network']['compartments']))
        paths={c['id']:c for c in source['gas_network']['channels']}
        self.assertEqual(len(paths),len(g['channels'])+len(s['metadata']['closed_paths']))
        for c in g['channels']:
            old=paths[c['id']]
            for key in ('from','to'):self.assertEqual(c[key],old[key])
            self.assertEqual(c['discharge_coefficient'],old.get('discharge_coefficient',.7))
            self.assertGreater(c['area_m2'],0)
        for n in g['compartments']:
            self.assertNotIn('local_equilibrium',n)
            self.assertNotIn('initial_chemical_j',n)
            self.assertNotIn('reaction_rate_per_s',n)

    def test_native_fixed_chain_refines_and_conserves(self):
        if not audit.PROBE.exists():self.skipTest('Build gas transport audit probe first')
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'scenario.json';path.write_text(json.dumps(audit.chain()))
            def run(dt):
                r=subprocess.run([str(audit.PROBE),str(path),str(dt),'.1','0.00002','0.00002'],
                                 capture_output=True,text=True,check=True,timeout=30)
                return json.loads(r.stdout)
            coarse,fine,finer=run(2e-5),run(2e-8),run(5e-9)
        expected=100000+100/3
        self.assertEqual(coarse['pressure_relaxation_fraction'],.2)
        self.assertEqual(coarse['pressure_absolute_tolerance_pa'],1.)
        self.assertEqual(coarse['pressure_relative_tolerance'],1e-5)
        for r in (coarse,fine,finer):
            self.assertLess(r['max_mass_residual_kg'],1e-18)
            self.assertLess(r['max_thermal_residual_j'],1e-12)
        for r in (coarse,fine,finer):
            for node in r['gas'].values():self.assertLess(abs(node['pressure_pa'][-1]-expected),1e-3)
        delta=audit.summarize(fine,finer)
        self.assertLess(delta['max_final_pressure_difference_pa'],1e-7)
        # The old operator accepted one capped step, left 75 Pa imbalance and
        # transferred no gas on the second edge. Resolve the physical interval.
        self.assertGreater(coarse['steps'],1)
        self.assertLess(coarse['steps'],finer['steps']/5)
        self.assertGreater(coarse['channels']['path1']['transferred_mass_kg'][-1],0)
        delta=audit.summarize(coarse,finer)
        self.assertLess(delta['max_final_pressure_difference_pa'],1e-3)
        self.assertLess(delta['max_final_path_transfer_difference_kg'],1e-14)


if __name__=='__main__':unittest.main()
