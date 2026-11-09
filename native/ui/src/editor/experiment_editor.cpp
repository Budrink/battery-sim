#include "editor_internal.hpp"
#include "toolbar.hpp"
#include "battery_native/contracts/experiment_contract.hpp"
#include "battery_native/contracts/enclosure_openings.hpp"
#include "battery_native/scenario_document.hpp"
#include "imgui.h"
#include "misc/cpp/imgui_stdlib.h"
#include <algorithm>

namespace battery::ui {
ExperimentEditor::ExperimentEditor() : impl_(std::make_unique<Impl>()) {
    impl_->camera.distance = 1.5f;
    impl_->visual.show_ports = false;
    impl_->visual.show_interfaces = false;
    impl_->visual.show_gas = false;
    impl_->visual.show_combustion = false; // initial-state editor, not a running reactor
    impl_->visual.battery_surface_opacity = .65f;
}
ExperimentEditor::~ExperimentEditor() = default;
void ExperimentEditor::select_object(const std::string& id) {
    if (impl_->document.draft.is_null())
        return;
    const auto& a = impl_->document.draft.at("assembly");
    for (const auto* list : {"batteries", "passive_cells"})
        if (a.contains(list))
            for (const auto& object : a.at(list))
                if (object.at("id") == id) {
                    impl_->selected = id;
                    impl_->case_opening_selection.id.clear();
                }
}
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
        p.file_path = p.document.path.string();
        p.selected = p.document.draft["assembly"]["batteries"][0].at("id");
        p.case_opening_selection = {};
        p.case_opening_error.clear();
        p.place_case_opening = false;
        p.json_text = p.document.draft.dump(2);
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
        p.message = "Imported: " + p.document.draft.at("name").get<std::string>() +
                    ". Preview shows this file; RUN SNAPSHOT starts it. The running simulation is unchanged.";
    } catch (const std::exception& e) {
        p.import_error = "Import failed: " + std::string(e.what());
        p.message = p.import_error;
    }
}
std::optional<nlohmann::json> ExperimentEditor::take_run() {
    auto result = std::move(impl_->pending);
    impl_->pending.reset();
    return result;
}
void ExperimentEditor::Impl::draw_browser() {
    if (!browser)
        return;
    ImGui::SetNextWindowSize({650, 500}, ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Import experiment JSON", &browser)) {
        ImGui::TextWrapped("%s", std::filesystem::absolute(browse_path).string().c_str());
        if (ImGui::Button("Parent folder"))
            browse_path = std::filesystem::absolute(browse_path).parent_path();
        ImGui::BeginChild("files", {0, 0}, true);
        try {
            std::vector<std::filesystem::directory_entry> entries;
            for (const auto& e : std::filesystem::directory_iterator(browse_path))
                if (e.is_directory() || e.path().extension() == ".json")
                    entries.push_back(e);
            std::sort(entries.begin(), entries.end(),
                      [](const auto& a, const auto& b) { return a.path() < b.path(); });
            for (const auto& entry : entries) {
                const bool dir = entry.is_directory();
                const auto label = (dir ? "[dir] " : "") + entry.path().filename().string();
                if (ImGui::Selectable(label.c_str())) {
                    if (dir)
                        browse_path = entry.path();
                    else {
                        file_path = entry.path().string();
                        import_requested = true;
                        browser = false;
                    }
                    break;
                }
            }
        } catch (const std::exception& e) {
            ImGui::TextWrapped("%s", e.what());
        }
        ImGui::EndChild();
    }
    ImGui::End();
}
void ExperimentEditor::draw() {
    if (!open)
        return;
    auto& p = *impl_;
    if (ImGui::IsKeyPressed(ImGuiKey_Escape) && p.place_case_opening) {
        p.place_case_opening = false;
        p.case_opening_error.clear();
    }
    ImGui::SetNextWindowSize({1320, 820}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos({60, 80}, ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Virtual laboratory / Experiment editor", &open,
                      ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar |
                          ImGuiWindowFlags_NoScrollWithMouse)) {
        ImGui::End();
        return;
    }
    auto action = [&](auto fn) {
        try {
            fn();
        } catch (const std::exception& e) {
            p.message = e.what();
        }
    };
    ImGui::BeginChild("editor toolbar", {0, 0}, ImGuiChildFlags_AutoResizeY,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::SetNextItemWidth(std::max(100.f, ImGui::GetContentRegionAvail().x - 160));
    if (ImGui::InputText("##path", &p.file_path, ImGuiInputTextFlags_EnterReturnsTrue))
        p.import_requested = true;
    ImGui::SameLine();
    if (toolbar_button("Browse"))
        p.browser = true;
    if (toolbar_button("Import")) p.import_requested = true;
    if (p.import_requested) {
        p.import_requested = false;
        if (p.document.dirty || p.model_edits.pending())
            ImGui::OpenPopup("Discard draft?");
        else
            try_load(p.file_path);
    }
    ImGui::NewLine();
    if (ImGui::BeginPopupModal("Discard draft?", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("Unsaved edits will be replaced only if import succeeds.");
        if (ImGui::Button("Import file")) {
            try_load(p.file_path);
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel"))
            ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }
    if (p.document.draft.is_null()) {
        if (ImGui::Button("Load synthetic starter"))
            try_load("examples/contracts/module-4x4.json");
        ImGui::TextWrapped("%s", p.message.c_str());
        ImGui::EndChild();
        ImGui::End();
        p.draw_browser();
        return;
    }
    p.document.sidecar["layout"]["preview_orbit"] = {
        {"yaw_rad", p.camera.yaw}, {"pitch_rad", p.camera.pitch}, {"distance_scale", p.camera.distance}};
    p.document.sidecar["layout"]["selected_object"] = p.selected;
    p.model_edits.synchronize(p.document.draft);
    ImGui::BeginDisabled(p.model_edits.pending());
    if (toolbar_button("Save to path")) {
        if (std::filesystem::exists(p.file_path) && std::filesystem::path(p.file_path) != p.document.path)
            ImGui::OpenPopup("Overwrite file?");
        else
            action([&] {
                p.document.save(p.file_path);
                p.message = "Experiment and editor sidecar saved.";
            });
    }
    ImGui::EndDisabled();
    if (ImGui::BeginPopupModal("Overwrite file?", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextWrapped("%s already exists", p.file_path.c_str());
        if (ImGui::Button("Overwrite")) {
            action([&] { p.document.save(p.file_path); });
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel"))
            ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }
    if (toolbar_button("Undo"))
        p.document.undo();
    if (toolbar_button("Redo"))
        p.document.redo();
    ImGui::BeginDisabled(p.selected.empty() && p.case_opening_selection.id.empty());
    if (toolbar_button("Delete")) {
        const auto before_delete = p.document.draft;
        p.delete_selected();
        p.document.record(before_delete);
    }
    ImGui::EndDisabled();
    if (toolbar_button("Validate"))
        action([&] {
            (void)p.document.compile();
            p.message = "Contract valid; Preview/Run also checks the computational mesh.";
        });
    if (toolbar_button("Refresh 3D preview"))
        action([&] {
            p.preview_updates.attempted();
            p.refresh_preview();
        });
    ImGui::BeginDisabled(p.model_edits.pending());
    if (toolbar_button("RUN SNAPSHOT"))
        action([&] {
            p.refresh_preview();
            p.pending = p.document.compile();
            p.message = "Validated snapshot queued for a new simulation.";
        });
    ImGui::EndDisabled();
    ImGui::NewLine();
    ImGui::TextColored({.3f, .9f, .8f, 1}, "%s", p.document.dirty ? "UNSAVED DRAFT" : "SAVED");
    ImGui::SameLine();
    ImGui::TextDisabled("Automatic preview updates");
    ImGui::TextWrapped("%s", p.message.c_str());
    if (!p.import_error.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{1,.4f,.25f,1});
        ImGui::TextWrapped("%s", p.import_error.c_str());
        ImGui::PopStyleColor();
    }
    if (p.model_edits.pending()) {
        ImGui::TextWrapped("Models has unapplied input. Other elements remain editable. "
                           "Correct the marked model fields or discard that input before Save / Run.");
        if (ImGui::Button("Discard pending model input"))
            p.model_edits.clear();
    }
    ImGui::EndChild();
    ImGui::Separator();
    const auto before = p.document.draft;
    const float width = ImGui::GetContentRegionAvail().x;
    const bool split_preview = width >= 900.f;
    ImGui::BeginChild("editors", {split_preview ? width * .48f : 0.f, 0}, true);
    if (ImGui::BeginTabBar("laboratory tabs", ImGuiTabBarFlags_FittingPolicyScroll)) {
        if (ImGui::BeginTabItem("Assembly")) {
            p.draw_assembly();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Circuit")) {
            p.draw_circuit();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Contacts")) {
            p.draw_connections();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Heating")) {
            p.draw_heating();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Models")) {
            p.draw_models();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Shell / vents")) {
            p.draw_shell();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Case")) {
            p.draw_enclosure();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Run")) {
            p.draw_run_settings();
            ImGui::EndTabItem();
        }
        if (!split_preview && ImGui::BeginTabItem("3D preview")) {
            p.update_preview();
            p.draw_preview();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("JSON")) {
            ImGui::TextWrapped(
                "Advanced, lossless document editing. Apply validates before replacing the draft.");
            if (ImGui::Button("Reload text from draft"))
                p.json_text = p.document.draft.dump(2);
            ImGui::SameLine();
            if (ImGui::Button("Apply JSON"))
                action([&] {
                    auto j = nlohmann::json::parse(p.json_text);
                    p.document.draft = upgrade_experiment(j);
                    p.message = "JSON validated and applied.";
                });
            ImGui::InputTextMultiline("##document", &p.json_text, {-1, -1},
                                      ImGuiInputTextFlags_AllowTabInput);
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::EndChild();
    // Also runs when the user leaves Models for another tab. Do not validate
    // half-typed numbers or rebuild CUDA on every keystroke.
    if (!ImGui::IsAnyItemActive())
        p.model_edits.commit_valid(p.document.draft);
    if (split_preview) {
        ImGui::SameLine();
        ImGui::BeginChild("preview", {0, 0}, true);
        p.update_preview();
        p.draw_preview();
        ImGui::EndChild();
    }
    // Delete in a text/numeric input must edit text, never remove its object.
    if ((!p.selected.empty() || !p.case_opening_selection.id.empty()) &&
        ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) && !ImGui::GetIO().WantTextInput &&
        !ImGui::IsAnyItemActive() &&
        !ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel) &&
        ImGui::IsKeyPressed(ImGuiKey_Delete, false))
        p.delete_selected();
    p.document.record(before);
    ImGui::End();
    p.draw_browser();
}
} // namespace battery::ui
