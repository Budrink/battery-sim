#include "battery_native/contracts/experiment_contract.hpp"
#include "contract_internal.hpp"

namespace battery {
void validate_editor_document(const nlohmann::json& editor, const nlohmann::json& experiment) {
    using namespace contract_detail;
    finite_tree(editor);
    object(editor, "editor", {"format", "version", "experiment_id", "camera", "hidden_objects", "layout"},
           {"format", "version", "experiment_id"});
    if (editor.at("format") != "battery-sim.editor" || !editor.at("version").is_number_integer() ||
        editor.at("version") != 1)
        fail("editor", "unsupported format/version");
    if (id(editor.at("experiment_id"), "editor/experiment_id") != experiment.at("id"))
        fail("editor/experiment_id", "sidecar belongs to a different experiment");
    if (editor.contains("layout") && !editor.at("layout").is_object())
        fail("editor/layout", "expected UI-owned object");
    if (editor.contains("camera")) {
        const auto& c = editor.at("camera");
        object(c, "editor/camera", {"target_m", "distance_m", "yaw_rad", "pitch_rad"},
               {"target_m", "distance_m", "yaw_rad", "pitch_rad"});
        const auto& target = c.at("target_m");
        if (!target.is_array() || target.size() != 3)
            fail("editor/camera", "expected 3D target");
        for (const auto& x : target)
            number(x, "target_m", -1e9, 1e9);
        number(c.at("distance_m"), "distance_m", 1e-12, 1e9);
        number(c.at("yaw_rad"), "yaw_rad", -1e9, 1e9);
        number(c.at("pitch_rad"), "pitch_rad", -1.5707963267948966, 1.5707963267948966);
    }
    if (editor.contains("hidden_objects")) {
        std::set<std::string> objects, seen;
        const auto& a = experiment.at("assembly");
        for (const auto* key : {"batteries", "passive_cells"})
            for (const auto& obj : a.value(key, J::array()))
                objects.insert(obj.at("id").get<std::string>());
        array(editor.at("hidden_objects"), "hidden_objects");
        for (const auto& entry : editor.at("hidden_objects")) {
            const auto oid = text(entry, "hidden_objects/id");
            if (!objects.contains(oid) || !seen.insert(oid).second)
                fail("hidden_objects", "unknown or duplicate object");
        }
    }
}
} // namespace battery
