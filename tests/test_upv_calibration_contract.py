"""Measured battery properties must not become contact-fit nuisance parameters."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
try:
    import calibrate_upv as calibration
finally:
    sys.path.pop(0)


class UPVCalibrationContractTests(unittest.TestCase):
    def test_only_contact_environment_and_unknown_pad_cp_can_be_fitted(self):
        self.assertEqual(calibration.PARAMETER_SETS['shared'],('contact_scale','ambient_scale'))
        self.assertEqual(calibration.PARAMETER_SETS['contacts'],calibration.CONTACTS+('ambient_scale',))
        self.assertEqual(calibration.PARAMETER_SETS['pads'],calibration.CONTACTS+('ambient_scale','iss_cp_scale','tim_cp_scale'))
        self.assertEqual(set(calibration.BOUNDS),set(calibration.PARAMETERS))
        calibration.validate_parameters({'contact_scale':2.,'ambient_scale':1.})
        calibration.validate_parameters({'battery_iss_scale':2.,'iss_cp_scale':1.2})

    def test_measured_pad_conductivity_cannot_be_fitted(self):
        for key in ('iss_k_scale','tim_k_scale'):
            with self.assertRaises(ValueError):calibration.validate_parameters({key:1.})

    def test_contact_groups_cannot_be_scaled_twice(self):
        with self.assertRaisesRegex(ValueError,'Cannot combine'):
            calibration.validate_parameters({'contact_scale':2.,'battery_iss_scale':2.})

    def test_native_audit_detects_changed_battery_or_bulk_conductivity(self):
        thermal={'heat_capacity_j_per_k':412.87,'conductivity_w_per_m_k':[.9,28.3,28.3]}
        scene={'cells':[{'id':'C1','role':'battery','material_id':'battery','thermal':thermal}]}
        run={'applied_model':{'objects':[dict(thermal,id='C1')]}}
        calibration.verify_fixed_properties(run,scene)
        run['applied_model']['objects'][0]['heat_capacity_j_per_k']=330.
        with self.assertRaisesRegex(ValueError,'Fixed heat capacity'):calibration.verify_fixed_properties(run,scene)
        run['applied_model']['objects'][0]=dict(thermal,id='C1',conductivity_w_per_m_k=[.36,28.3,28.3])
        with self.assertRaisesRegex(ValueError,'Bulk conductivity'):calibration.verify_fixed_properties(run,scene)

    def test_battery_scaling_is_rejected_even_when_one(self):
        for key in ('cell_k_scale','cell_capacity_scale'):
            for scale in (.5,1.):
                with self.assertRaisesRegex(ValueError,'battery properties are fixed'):
                    calibration.validate_parameters({key:scale})

    def test_rejected_profile_cannot_be_validated(self):
        with self.assertRaisesRegex(ValueError,'Rejected calibration profile'):
            calibration.validate_profile({'status':'rejected','rejection_reason':'changed measured battery properties','parameters':{}})

    def test_legacy_battery_fit_cannot_be_validated_without_status_flag(self):
        with self.assertRaisesRegex(ValueError,'battery properties are fixed'):
            calibration.validate_profile({'parameters':{'contact_scale':2.,'cell_k_scale':.4}})

    def test_nonphysical_scales_are_rejected(self):
        for value in (0.,-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):calibration.validate_parameters({'contact_scale':value})


if __name__=='__main__':unittest.main()
