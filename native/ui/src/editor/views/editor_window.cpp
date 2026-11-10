#include "editor/editor_internal.hpp"
#include "shared/toolbar.hpp"
#include "shared/form_fields.hpp"
#include <algorithm>

namespace battery::ui {
void ExperimentEditor::draw() {
    if (!open)
        return;
    auto& p = *impl_;
    if (ImGui::IsKeyPressed(ImGuiKey_Escape) && p.place_case_opening) {
        p.place_case_opening = false;
        p.case_opening_error.clear();
    }
    // This fills its OWN native editor window, never floats over the simulation.
    const auto* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGui::Begin("Experiment editor", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
                     ImGuiWindowFlags_NoBringToFrontOnFocus);

    auto action = [&](auto fn) {
        try {
            fn();
        } catch (const std::exception& e) {
            p.message = e.what();
        }
    };
    auto pending = [&] {
        return p.model_edits.pending() || p.shell_dirty || p.enclosure_dirty || p.shape_dirty ||
               p.opening_dirty;
    };
    auto finish_input = [&] {
        const auto before = p.document.draft;
        p.commit_panel_edits();
        p.model_edits.commit_valid(p.document.draft);
        p.document.record(before);
        if (pending())
            throw std::runtime_error("Correct the marked input or discard it before saving/running.");
    };
    ImGui::BeginChild("editor toolbar", {0, 0}, ImGuiChildFlags_AutoResizeY,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::TextUnformatted("EXPERIMENT EDITOR");
    ImGui::SameLine();
    form::hint("Independent window / Alt+Tab to simulation");
    form::caption("Experiment file (Import reads it; Save writes it)");
    ImGui::SetNextItemWidth(std::max(140.f, ImGui::GetContentRegionAvail().x - 175));
    ImGui::InputText("##path", &p.file_path); // typing a path never imports it implicitly
    ImGui::SameLine();
    if (toolbar_button("Browse"))
        p.browser = true;
    if (toolbar_button("Import"))
        p.import_requested = true;
    ImGui::NewLine();
    if (p.import_requested) {
        p.import_requested = false;
        if (p.document.dirty || pending())
            ImGui::OpenPopup("Replace editor draft?");
        else
            try_load(p.file_path);
    }
    if (ImGui::BeginPopupModal("Replace editor draft?", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextWrapped("Unsaved edits are replaced only if the new file imports successfully.");
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
        if (ImGui::Button("Open 4x4 example"))
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
    if (toolbar_button("Save"))
        action([&] {
            finish_input();
            (void)p.document.compile();
            if (std::filesystem::exists(p.file_path) && std::filesystem::path(p.file_path) != p.document.path)
                ImGui::OpenPopup("Overwrite file?");
            else {
                p.document.save(p.file_path);
                p.message = "Experiment and editor view saved.";
            }
        });
    if (ImGui::BeginPopupModal("Overwrite file?", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextWrapped("%s already exists.", p.file_path.c_str());
        if (ImGui::Button("Overwrite")) {
            action([&] {
                finish_input();
                p.document.save(p.file_path);
            });
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
    if (toolbar_button("Delete selected")) {
        const auto before = p.document.draft;
        p.delete_selected();
        p.document.record(before);
    }
    ImGui::EndDisabled();
    if (toolbar_button("Check"))
        action([&] {
            finish_input();
            (void)p.document.compile();
            p.message = "Contract valid. Preview also validates the computational mesh.";
        });
    if (toolbar_button("Run simulation"))
        action([&] {
            finish_input();
            p.refresh_preview();
            p.pending = p.document.compile();
            p.message = "Validated snapshot queued. Existing simulation will be replaced.";
        });
    if (toolbar_button("Simulation window"))
        viewer_focus_requested = true;
    ImGui::NewLine();
    ImGui::TextColored({.3f, .9f, .8f, 1}, "%s", p.document.dirty ? "Unsaved changes" : "Saved");
    ImGui::SameLine();
    ImGui::TextWrapped(
        "Edit, then Tab or click elsewhere. Preview updates automatically; no Apply / Enter required.");
    if (!p.message.empty())
        ImGui::TextWrapped("%s", p.message.c_str());
    if (!p.import_error.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{1, .45f, .3f, 1});
        ImGui::TextWrapped("%s", p.import_error.c_str());
        ImGui::PopStyleColor();
    }
    if (pending()) {
        ImGui::TextWrapped(
            "Unapplied input: last valid values remain in the preview. Correct the marked fields.");
        for (const auto& [panel, error] : p.panel_errors) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{1, .45f, .3f, 1});
            ImGui::TextWrapped("%s: %s", panel.c_str(), error.c_str());
            ImGui::PopStyleColor();
        }
        if (ImGui::Button("Discard unapplied input")) {
            p.model_edits.clear();
            p.shell_dirty = p.enclosure_dirty = p.shape_dirty = p.opening_dirty = false;
            p.case_opening_selection.source = nullptr;
            p.shell_source = p.shape_source = p.enclosure_source = nullptr;
            p.enclosure_properties = nullptr;
            p.panel_attempt = nullptr;
            p.panel_errors.clear();
        }
    }
    ImGui::EndChild();
    ImGui::Separator();
    const auto before = p.document.draft;
    ImGui::BeginChild("section navigation", {190, 0}, true);
    form::hint("EXPERIMENT");
    for (int i = 0; i < int(std::size(editor_sections)); ++i) {
        ImGui::PushID(i);
        if (ImGui::Selectable(editor_sections[i].title, int(p.section) == i, 0, {0, 30})) {
            p.commit_panel_edits();
            p.model_edits.commit_valid(p.document.draft);
            p.section = EditorSection(i);
        }
        ImGui::PopID();
    }
    ImGui::Separator();
    ImGui::Checkbox("Side preview", &p.show_preview);
    ImGui::TextWrapped("Hover field labels for help. Full descriptions are in Parameter reference.");
    ImGui::EndChild();
    ImGui::SameLine();
    const float width = ImGui::GetContentRegionAvail().x;
    const bool preview_page = p.section == EditorSection::Preview;
    const bool split =
        p.show_preview && width >= 950 && !preview_page && p.section != EditorSection::Reference;
    ImGui::BeginChild("section content", {split ? width * .54f : 0.f, 0}, true);
    const auto& info = editor_sections[int(p.section)];
    ImGui::TextWrapped("%s", info.title);
    ImGui::TextWrapped("%s", info.description);
    ImGui::Separator();
    switch (p.section) {
    case EditorSection::Assembly:
        p.draw_assembly();
        break;
    case EditorSection::Circuit:
        p.draw_circuit();
        break;
    case EditorSection::Contacts:
        p.draw_connections();
        break;
    case EditorSection::Heating:
        p.draw_heating();
        break;
    case EditorSection::Models:
        p.draw_models();
        break;
    case EditorSection::GasModel:
        p.draw_gas_model();
        break;
    case EditorSection::Shell:
        p.draw_shell();
        break;
    case EditorSection::Case:
        p.draw_enclosure();
        break;
    case EditorSection::Run:
        p.draw_run_settings();
        break;
    case EditorSection::Preview:
        p.update_preview();
        p.draw_preview();
        break;
    case EditorSection::Reference:
        p.draw_reference();
        break;
    }
    ImGui::EndChild();
    if (!ImGui::IsAnyItemActive()) {
        p.commit_panel_edits();
        p.model_edits.commit_valid(p.document.draft);
    }
    if (split) {
        ImGui::SameLine();
        ImGui::BeginChild("preview", {0, 0}, true);
        p.update_preview();
        p.draw_preview();
        ImGui::EndChild();
    }
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
