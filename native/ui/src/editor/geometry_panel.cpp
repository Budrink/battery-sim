#include "editor_internal.hpp"
#include "battery_native/contracts/experiment_editing.hpp"
#include "imgui.h"
#include <cmath>

namespace battery::ui {
bool ExperimentEditor::Impl::draw_battery_shape() {
    const auto& batteries = document.draft.at("assembly").at("batteries");
    const nlohmann::json* geometry = nullptr;
    for (const auto& b : batteries)
        if (b.at("id") == selected)
            for (const auto& model : document.draft.at("battery_models"))
                if (model.at("id") == b.at("model_id") && model.at("revision") == b.at("model_revision"))
                    geometry = &model.at("geometry");
    if (!geometry)
        return false;
    if (shape_owner != selected || shape_source != *geometry) {
        shape_owner = selected;
        shape_source = *geometry;
        shape_kind = geometry->at("kind") == "cylinder" ? 1 : 0;
        const auto d = geometry->at("dimensions_m").get<std::vector<double>>();
        shape_dimensions_mm[0] = 1000 * d[0] * (shape_kind ? 2 : 1);
        shape_dimensions_mm[1] = 1000 * d[1];
        shape_dimensions_mm[2] = shape_kind ? 0 : 1000 * d[2];
    }
    ImGui::SeparatorText("Battery shape / contents dimensions");
    const int old_kind = shape_kind;
    bool changed = ImGui::Combo("Shape", &shape_kind, "Prism\0Cylinder\0");
    if (changed) {
        const double root_pi = std::sqrt(std::acos(-1.));
        if (old_kind == 0) {
            const double diameter =
                2 * std::sqrt(std::abs(shape_dimensions_mm[0] * shape_dimensions_mm[1])) / root_pi;
            shape_dimensions_mm[0] = diameter;
            shape_dimensions_mm[1] = shape_dimensions_mm[2];
        } else {
            shape_dimensions_mm[2] = shape_dimensions_mm[1];
            shape_dimensions_mm[0] *= root_pi / 2;
            shape_dimensions_mm[1] = shape_dimensions_mm[0];
        }
    }
    ImGui::SetNextItemWidth(160);
    changed |=
        ImGui::InputDouble(shape_kind ? "Diameter mm" : "Width X mm", &shape_dimensions_mm[0], 0, 0, "%.3f");
    ImGui::SetNextItemWidth(160);
    changed |=
        ImGui::InputDouble(shape_kind ? "Height Z mm" : "Length Y mm", &shape_dimensions_mm[1], 0, 0, "%.3f");
    if (!shape_kind) {
        ImGui::SetNextItemWidth(160);
        changed |= ImGui::InputDouble("Height Z mm", &shape_dimensions_mm[2], 0, 0, "%.3f");
    }
    ImGui::TextWrapped("Selected battery only. Dimensions exclude the shell. Mass and total heat capacity "
                       "are retained; review them and external contacts after resizing.");
    if (shape_kind)
        ImGui::TextWrapped("Rendered as a cylinder; current solver uses an explicit equal-volume prism. "
                           "This does not preserve cylindrical contact areas automatically.");
    if (changed) {
        try {
            std::vector<double> d;
            if (shape_kind)
                d = {shape_dimensions_mm[0] * .0005, shape_dimensions_mm[1] * .001};
            else
                d = {shape_dimensions_mm[0] * .001, shape_dimensions_mm[1] * .001,
                     shape_dimensions_mm[2] * .001};
            document.draft =
                set_battery_geometry(document.draft, selected, shape_kind ? "cylinder" : "prismatic", d);
            message = "Shape applied to selected battery. Review external contacts, heater placement and gas "
                      "volumes.";
            return true;
        } catch (const std::exception& e) {
            message = e.what();
        }
    }
    return false;
}
} // namespace battery::ui
