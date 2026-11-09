"""Optional editor/API schema checks; native contract checks require no Python packages."""
import copy
import json
from pathlib import Path
import unittest

try:
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource
except ImportError:
    Draft202012Validator = None

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(Draft202012Validator is None, "Optional test dependency: jsonschema>=4.18")
class ExperimentSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schemas = {}
        registry = Registry()
        for name in ("scenario-v0", "battery-v1", "experiment-v1", "experiment-v2", "editor-v1"):
            schema = json.loads((ROOT / f"schemas/{name}.schema.json").read_text())
            Draft202012Validator.check_schema(schema)
            registry = registry.with_resource(schema["$id"], Resource.from_contents(schema))
            cls.schemas[name] = schema
        cls.experiment = json.loads((ROOT / "examples/contracts/reaction-chain-experiment.json").read_text())
        cls.validator = Draft202012Validator(cls.schemas["experiment-v1"], registry=registry)
        cls.battery_validator = Draft202012Validator(cls.schemas["battery-v1"], registry=registry)
        cls.v2_validator = Draft202012Validator(cls.schemas["experiment-v2"], registry=registry)
        cls.editor_validator = Draft202012Validator(cls.schemas["editor-v1"], registry=registry)
        cls.v2 = json.loads((ROOT / "examples/contracts/editor-experiment.json").read_text())

    def test_example_and_embedded_model(self):
        self.validator.validate(self.experiment)
        self.battery_validator.validate(self.experiment["battery_models"][0])
        standalone = json.loads((ROOT / "examples/contracts/synthetic-battery-model.json").read_text())
        self.assertEqual(standalone, self.experiment["battery_models"][0])

    def test_constant_power_temperature_cutoff(self):
        candidate = copy.deepcopy(self.experiment)
        protocol = dict(id="test-cutoff", heater_id="heater", sensor_id="battery",
                        ramp_k_per_s=0, feedforward_w=5, proportional_w_per_k=0,
                        maximum_power_w=5, cutoff_temperature_k=423.15, cutoff_rate_k_per_s=0)
        candidate["conditions"]["heating_protocols"] = [protocol]
        self.validator.validate(candidate)
        for field in ("ramp_k_per_s", "proportional_w_per_k", "cutoff_rate_k_per_s"):
            protocol[field] = -1
            self.assertFalse(self.validator.is_valid(candidate), field)
            protocol[field] = 0

    def test_global_r3_model(self):
        candidate = copy.deepcopy(self.experiment)
        law = dict(reaction_rate_per_s=.75, reaction_reference_k=600,
                   reaction_activation_j_per_mol=40000, oxidizer_heat_j_per_kg=13.1e6)
        candidate["assembly"]["gas_network"]["combustion"] = law
        self.validator.validate(candidate)
        for key, value in (("reaction_rate_per_s", -1), ("reaction_reference_k", 0),
                           ("reaction_activation_j_per_mol", -1), ("oxidizer_heat_j_per_kg", 0),
                           ("ignition_temperature_k", -1), ("misspelled_rate", 1)):
            invalid = copy.deepcopy(candidate)
            invalid["assembly"]["gas_network"]["combustion"][key] = value
            self.assertFalse(self.validator.is_valid(invalid), key)
        for key in law:
            invalid = copy.deepcopy(candidate)
            del invalid["assembly"]["gas_network"]["combustion"][key]
            self.assertFalse(self.validator.is_valid(invalid), key)

    def test_shell_faces_keep_the_existing_material_law(self):
        candidate = copy.deepcopy(self.experiment)
        candidate["battery_models"][0]["shell"]["face_materials"] = {
            "+z": {"material_id": "synthetic-cap", "conductivity_w_per_m_k": [.1, .2, .3],
                   "thickness_m": .0002, "density_kg_m3": 1200, "specific_heat_j_kg_k": 1300}}
        self.validator.validate(candidate)
        for key in ("thickness_m", "density_kg_m3", "specific_heat_j_kg_k"):
            invalid = copy.deepcopy(candidate)
            invalid["battery_models"][0]["shell"]["face_materials"]["+z"][key] = 0
            self.assertFalse(self.validator.is_valid(invalid), key)

    def test_unknown_top_level_field_and_version(self):
        for key, value in (("verison", 1), ("version", 2)):
            candidate = copy.deepcopy(self.experiment)
            candidate[key] = value
            self.assertFalse(self.validator.is_valid(candidate))

    def test_runtime_values_cannot_be_catalog_parameters(self):
        for section, key, value in (("thermal", "divisions", [3, 3, 3]),
                                    ("thermal", "initial_temperature_k", 300),
                                    ("conversion", "initial_charged_fraction", .5),
                                    ("conversion", "gas_compartment", "inside-B"),
                                    ("conversion", "max_temperature_step_k", 1)):
            candidate = copy.deepcopy(self.experiment)
            candidate["battery_models"][0]["contents"][section][key] = value
            self.assertFalse(self.validator.is_valid(candidate), key)

    def test_unknown_sensor_does_not_silently_become_cell_measurement(self):
        candidate = copy.deepcopy(self.experiment)
        candidate["measurements"][0]["mapping"]["cell_id"] = "B"
        self.assertFalse(self.validator.is_valid(candidate))
        candidate["measurements"][0]["mapping"]["status"] = "hypothesis"
        self.validator.validate(candidate)
        candidate["measurements"][0]["unit"] = "Pa"
        self.assertFalse(self.validator.is_valid(candidate))

    def test_electrical_model_is_separate_from_drive(self):
        candidate = copy.deepcopy(self.experiment)
        electrical = {"resistance_ohm": .01, "charge_acceptance_fraction": .3,
                      "nominal_capacity_coulomb": 10000}
        candidate["battery_models"][0]["contents"]["electrical"] = electrical
        candidate["conditions"]["electrical_drives"] = [{"cell_id": "B", "current_a": 2}]
        self.validator.validate(candidate)
        electrical["current_a"] = 2
        self.assertFalse(self.validator.is_valid(candidate))

    def test_r1_r2_does_not_accept_legacy_bulk_damage(self):
        candidate = copy.deepcopy(self.experiment)
        candidate["assembly"]["batteries"][0]["initial_state"]["damage_fraction"] = .1
        self.assertFalse(self.validator.is_valid(candidate))

    def test_v2_and_editor_examples(self):
        self.v2_validator.validate(self.v2)
        editor = json.loads((ROOT / "examples/contracts/editor-state.json").read_text())
        self.editor_validator.validate(editor)
        editor["physics"] = {"current_a": 100}
        self.assertFalse(self.editor_validator.is_valid(editor))

    def test_solved_dc_laboratory(self):
        lab = json.loads((ROOT / "examples/contracts/laboratory-demo.json").read_text())
        self.v2_validator.validate(lab)
        del lab["assembly"]["electrical_circuit"]["branches"][0]["emf_v"]
        self.assertFalse(self.v2_validator.is_valid(lab))

    def test_module_case_and_weak_top(self):
        module = json.loads((ROOT / "examples/contracts/module-4x4.json").read_text())
        self.v2_validator.validate(module)

    def test_v2_ports_require_explicit_frame(self):
        candidate = copy.deepcopy(self.v2)
        del candidate["assembly"]["ports"][0]["frame"]
        self.assertFalse(self.v2_validator.is_valid(candidate))
        candidate["assembly"]["ports"][0]["frame"] = "world"
        self.v2_validator.validate(candidate)
        candidate["assembly"]["ports"][0]["frame"] = "automatic"
        self.assertFalse(self.v2_validator.is_valid(candidate))

    def test_v2_unmapped_is_explicitly_empty(self):
        candidate = copy.deepcopy(self.v2)
        candidate["assembly"]["electrical_circuit"]["mode"] = "unmapped_drives"
        self.assertFalse(self.v2_validator.is_valid(candidate))
        candidate["assembly"]["electrical_circuit"]["nodes"] = []
        candidate["assembly"]["electrical_circuit"]["branches"] = []
        self.assertFalse(self.v2_validator.is_valid(candidate))
        candidate["conditions"]["source_drives"] = []
        self.v2_validator.validate(candidate)

    def test_v2_rejects_unknown_fields_and_ambiguous_source(self):
        candidate = copy.deepcopy(self.v2)
        candidate["conditions"]["source_drives"][0]["current_profile_a"] = [[0, 2]]
        self.assertFalse(self.v2_validator.is_valid(candidate))
        candidate = copy.deepcopy(self.v2)
        candidate["assembly"]["electrical_circuit"]["branches"][0]["resistance_ohm"] = 1
        self.assertFalse(self.v2_validator.is_valid(candidate))


if __name__ == "__main__":
    unittest.main()
