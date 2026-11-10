#include "editor/editor_internal.hpp"
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
std::optional<nlohmann::json> ExperimentEditor::take_run() {
    auto result = std::move(impl_->pending);
    impl_->pending.reset();
    return result;
}
} // namespace battery::ui
