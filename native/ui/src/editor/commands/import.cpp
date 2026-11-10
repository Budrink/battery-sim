#include "editor/editor_internal.hpp"
#include "battery_native/contracts/experiment_contract.hpp"
#include "battery_native/contracts/experiment_editing.hpp"
#include "battery_native/contracts/enclosure_openings.hpp"
#include "battery_native/scenario_document.hpp"
#include <algorithm>
namespace battery::ui {
using J = nlohmann::json;
void ExperimentEditor::try_load(const std::string& path) {
    auto& p = *impl_;
    p.file_path = path;
    try {
        // Validate the entire candidate before replacing a usable draft. A
        // syntactically valid JSON file may still contain an invalid graph.
        EditorDocument next;
        next.load(path);
        const auto compiled = next.compile();
        const auto scene = scenario_from_json(compiled);
        (void)build_thermal_mesh(scene);
        p.document = std::move(next);
        p.import_error.clear();
        p.pending.reset();
        const auto cases = editable_enclosures(p.document.draft);
        p.enclosure_id = cases.empty() ? "case" : cases.begin().key();
        p.enclosure_loaded.clear();
        p.enclosure_properties = nullptr;
        p.enclosure_source = nullptr;
        p.shell_owner.clear();
        p.shell_source = nullptr;
        p.shape_owner.clear();
        p.shape_source = nullptr;
        p.model_selected_battery.clear();
        p.model_index = 0;
        p.heater_target.clear();
        p.model_edits.clear();
        p.shell_dirty = p.enclosure_dirty = p.shape_dirty = p.opening_dirty = false;
        p.panel_errors.clear();
        p.panel_attempt = nullptr;
        p.file_path = p.document.path.string();
        p.selected = p.document.draft["assembly"]["batteries"][0].at("id");
        p.case_opening_selection = {};
        p.case_opening_error.clear();
        p.place_case_opening = false;
        p.preview_updates.reset();
        p.preview_error.clear();
        try {
            const auto layout = p.document.sidecar.value("layout", nlohmann::json::object());
            if (layout.contains("preview_orbit")) {
                const auto& orbit = layout.at("preview_orbit");
                p.camera.yaw = orbit.value("yaw_rad", p.camera.yaw);
                p.camera.pitch = std::clamp(orbit.value("pitch_rad", p.camera.pitch), -1.4f, 1.4f);
                p.camera.distance = std::clamp(orbit.value("distance_scale", p.camera.distance), .55f, 6.f);
            }
        } catch (const std::exception&) {
        } // optional, UI-owned view metadata
        p.message =
            "Imported: " + p.document.draft.at("name").get<std::string>() +
            ". Preview shows this file; Run simulation starts it. The running simulation is unchanged.";
    } catch (const std::exception& e) {
        p.import_error = "Import failed: " + std::string(e.what());
        p.message = p.import_error;
    }
}
} // namespace battery::ui
