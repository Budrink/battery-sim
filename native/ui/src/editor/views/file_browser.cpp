#include "editor/editor_internal.hpp"
#include "shared/toolbar.hpp"
#include "shared/form_fields.hpp"
#include "battery_native/contracts/experiment_contract.hpp"
#include "battery_native/contracts/experiment_editing.hpp"
#include "battery_native/contracts/enclosure_openings.hpp"
#include "imgui.h"
#include "misc/cpp/imgui_stdlib.h"
#include <algorithm>
#include <cmath>
namespace battery::ui {
using J = nlohmann::json;
void ExperimentEditor::Impl::draw_browser() {
    if (!browser)
        return;
    ImGui::SetNextWindowSize({650, 500}, ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Import experiment JSON", &browser)) {
        ImGui::TextWrapped(
            "Select a complete experiment. Companion .editor.json files only store the view and are hidden.");
        ImGui::TextWrapped("%s", std::filesystem::absolute(browse_path).string().c_str());
        if (ImGui::Button("Parent folder"))
            browse_path = std::filesystem::absolute(browse_path).parent_path();
        ImGui::BeginChild("files", {0, 0}, true);
        try {
            std::vector<std::filesystem::directory_entry> entries;
            for (const auto& e : std::filesystem::directory_iterator(browse_path))
                if (e.is_directory() || (e.path().extension() == ".json" &&
                                         !e.path().filename().string().ends_with(".editor.json")))
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
} // namespace battery::ui
