import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from model_evidence import record,seed


class EvidenceTests(unittest.TestCase):
    def test_learning_is_scoped_and_core_updates_keep_seed_not_score(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);compatibility={'fixture':'one','stage':'conditioned'}
            ident=record('geometry_prior','test',{'area':.5},{'objective':3.},
                compatibility=compatibility,producer_core='old',registry=root,activate_development=True)
            profile=seed('geometry_prior','test',compatibility=compatibility,consumer_core='new',registry=root)
            self.assertEqual(profile['record'],ident)
            self.assertEqual(profile['parameters'],{'area':.5})
            self.assertTrue(profile['requires_recheck'])
            self.assertFalse(profile['reuse_scores'])
            self.assertFalse(profile['reuse_checkpoint'])
            with self.assertRaises(ValueError):
                seed('geometry_prior','test',compatibility={'fixture':'two'},consumer_core='new',registry=root)
            self.assertIsNone(seed('constitutive_law','test',compatibility=compatibility,consumer_core='new',registry=root))

    def test_no_promotion_from_conditioned_fit_or_absent_checks(self):
        with tempfile.TemporaryDirectory() as temp:
            for state in ('numerically_verified','validated'):
                with self.assertRaises(ValueError):
                    record('constitutive_law','material',{}, {},compatibility={'cell':'a'},
                        qualification=state,registry=Path(temp))
            with self.assertRaises(ValueError):
                record('geometry_prior','test',{},dict(numerical_checks={'accepted':True},
                    conditioned_transport_diagnostic=True),compatibility={'fixture':'a'},
                    qualification='numerically_verified',registry=Path(temp))

    def test_tampered_evidence_is_not_silently_used(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);compatibility={'fixture':'a'}
            ident=record('geometry_prior','test',{}, {},compatibility=compatibility,registry=root,activate_development=True)
            path=root/'records'/f'{ident}.json';value=json.loads(path.read_text());value['parameters']={'fake':1.}
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                seed('geometry_prior','test',compatibility=compatibility,consumer_core='core',registry=root)

    def test_resume_seed_does_not_follow_a_newer_mutable_pointer(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);compatibility={'fixture':'one'}
            first=record('geometry_prior','test',{'area':.5},{},compatibility=compatibility,
                registry=root,activate_development=True)
            second=record('geometry_prior','test',{'area':.7},{},compatibility=compatibility,
                registry=root,activate_development=True)
            latest=seed('geometry_prior','test',compatibility=compatibility,consumer_core='core',registry=root)
            original=seed('geometry_prior','test',compatibility=compatibility,consumer_core='core',registry=root,record_id=first)
            self.assertEqual(latest['record'],second)
            self.assertEqual(original['record'],first)
            self.assertEqual(original['parameters']['area'],.5)


if __name__=='__main__':unittest.main()
