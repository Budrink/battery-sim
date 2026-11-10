#include "editor/editor_internal.hpp"
#include "editor/help/parameter_catalog.hpp"
#include "shared/form_fields.hpp"
#include <algorithm>
#include <cctype>

namespace battery::ui {
void ExperimentEditor::Impl::draw_reference() {
    ImGui::TextWrapped("Workflow: edit a value, then leave the field (Tab or click). Valid changes apply "
                       "automatically to the draft and preview. Save writes the file. Run simulation starts "
                       "a NEW snapshot; editing never mutates an existing run.");
    ImGui::TextWrapped("Only structural actions need buttons: create/delete, import, save, run and confirmed "
                       "network replacement. Hover a field label for its explanation.");
    form::text_input("Search name, meaning or contract key", &help_search);
    auto lower = [](std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
        return s;
    };
    const auto query = lower(help_search);
    for (const auto& p : parameter_catalog) {
        if (!query.empty() &&
            lower(std::string(p.title) + " " + p.key + " " + p.explanation).find(query) == std::string::npos)
            continue;
        ImGui::Separator();
        ImGui::TextWrapped("%s", p.title);
        ImGui::TextWrapped("%s", p.explanation);
        form::hint("Contract key: %s", p.key);
    }
}
} // namespace battery::ui
