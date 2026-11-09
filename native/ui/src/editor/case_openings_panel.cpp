#include "editor_internal.hpp"
#include "form_fields.hpp"
#include "battery_native/contracts/enclosure_openings.hpp"
#include "imgui.h"
#include <algorithm>
#include <cmath>

namespace battery::ui {
using J = nlohmann::json;
void ExperimentEditor::Impl::select_case_opening(const J& hole) {
    selected.clear();
    case_opening_selection.select(enclosure_id, hole);
    case_opening_error.clear();
    case_opening_face = hole.at("face");
    case_opening_shape = hole.at("shape") == "circle" ? 0 : 1;
    for (int i = 0; i < 2; ++i) {
        case_opening_size_mm[i] = hole.at("size_m")[i].get<double>() * 1000;
        case_opening_center_mm[i] = hole.at("center_m")[i].get<double>() * 1000;
    }
}
void ExperimentEditor::Impl::draw_case_openings() {
    const auto meshes = editable_enclosures(document.draft);
    if (!meshes.contains(enclosure_id)) {
        ImGui::TextDisabled("Create the case first to place openings.");
        return;
    }
    auto openings = meshes.at(enclosure_id).value("openings", J::array());
    if (case_opening_selection.sync(enclosure_id, openings)) {
        const auto source = case_opening_selection.source;
        select_case_opening(source);
    }
    ImGui::SeparatorText("Case openings / outside air");
    if (ImGui::Button("Add opening")) {
        add_case_opening();
        openings = editable_enclosures(document.draft).at(enclosure_id).value("openings", J::array());
    }
    ImGui::SameLine();
    if (ImGui::Button(place_case_opening ? "Cancel placement" : "Place with mouse")) {
        place_case_opening = !place_case_opening;
        if (place_case_opening) {
        case_opening_selection.begin_new();
        case_opening_selection.owner = enclosure_id;
        case_opening_error.clear();
        selected.clear();
        }
    }
    // Fixed-height instruction/error region: entering placement mode and a
    // validation error must not move controls or the viewport under the cursor.
    ImGui::BeginChild("opening feedback", {0, ImGui::GetTextLineHeightWithSpacing()*3}, false);
    if (!case_opening_error.empty())
        ImGui::TextWrapped("Not applied: %s", case_opening_error.c_str());
    else if (place_case_opening)
        ImGui::TextWrapped("Click a case wall in the 3D preview to place a NEW opening. Drag rotates. Esc cancels.");
    else
        ImGui::TextWrapped("Add opening creates it immediately in a free spot on the selected face. Select it below to edit.");
    ImGui::EndChild();
    ImGui::BeginChild("opening list", {0, ImGui::GetTextLineHeightWithSpacing()*4}, true);
    for (const auto& h : openings)
        if (ImGui::Selectable(
                (h.at("id").get<std::string>() + "  " + h.at("face").get<std::string>()).c_str(),
                case_opening_selection.id == h.at("id")))
            { select_case_opening(h); place_case_opening = false; }
    ImGui::EndChild();
    ImGui::TextWrapped("%s", case_opening_selection.id.empty() ? "New opening settings" : case_opening_selection.id.c_str());
    const char* faces[] = {"+z", "-z", "+x", "-x", "+y", "-y"};
    const char* names[] = {"Top (+Z)", "Bottom (-Z)", "Right (+X)", "Left (-X)", "Back (+Y)", "Front (-Y)"};
    int face = 0;
    for (int i = 0; i < 6; ++i) if (case_opening_face == faces[i]) face = i;
    ImGui::TextUnformatted("Case face"); ImGui::SetNextItemWidth(-1);
    bool changed = ImGui::Combo("##opening face", &face, names, 6);
    case_opening_face = faces[face];
    ImGui::TextUnformatted("Hole shape"); ImGui::SetNextItemWidth(-1);
    changed |= ImGui::Combo("##opening shape", &case_opening_shape, "Circle\0Rectangle\0");
    changed |=
        form::input("Diameter / width U [mm]", &case_opening_size_mm[0]);
    ImGui::BeginDisabled(case_opening_shape == 0);
    changed |= form::input("Height V [mm]", &case_opening_size_mm[1]);
    ImGui::EndDisabled();
    if (case_opening_shape == 0)
        case_opening_size_mm[1] = case_opening_size_mm[0];
    changed |= form::input("Center U [mm]", &case_opening_center_mm[0]);
    changed |= form::input("Center V [mm]", &case_opening_center_mm[1]);
    ImGui::BeginDisabled(case_opening_selection.id.empty());
    const bool remove = ImGui::Button("Delete opening");
    ImGui::EndDisabled();
    if (case_opening_selection.id.empty() || (!changed && !remove))
        return;
    try {
        auto it = std::find_if(openings.begin(), openings.end(),
                               [&](const auto& h) { return h.at("id") == case_opening_selection.id; });
        if (it == openings.end()) {
            case_opening_selection.id.clear();
            return;
        }
        if (remove)
            openings.erase(it);
        else {
            (*it)["face"] = case_opening_face;
            (*it)["shape"] = case_opening_shape == 0 ? "circle" : "rectangle";
            (*it)["size_m"] = {case_opening_size_mm[0] / 1000, case_opening_size_mm[1] / 1000};
            (*it)["center_m"] = {case_opening_center_mm[0] / 1000, case_opening_center_mm[1] / 1000};
        }
        document.draft = set_enclosure_openings(document.draft, enclosure_id, openings);
        if (remove)
            case_opening_selection.id.clear();
        case_opening_error.clear();
        message = "Case openings updated. Pressure-driven exhaust/inlet; no forced ventilation.";
    } catch (const std::exception& e) {
        message = e.what();
        case_opening_error = e.what();
    }
}
bool ExperimentEditor::Impl::commit_case_opening(const std::string& case_id, const std::string& face,
                                                 std::array<double, 2> center_m) {
    try {
        auto list = editable_enclosures(document.draft).at(case_id).value("openings", J::array());
        std::string id;
        for (int n = 1;; ++n) {
            id = "opening-" + std::to_string(n);
            if (std::none_of(list.begin(), list.end(), [&](const auto& h) { return h.at("id") == id; })) break;
        }
        const J hole = {{"id", id}, {"face", face},
            {"shape", case_opening_shape == 0 ? "circle" : "rectangle"}, {"center_m", center_m},
            {"size_m", {case_opening_size_mm[0]/1000,
                (case_opening_shape == 0 ? case_opening_size_mm[0] : case_opening_size_mm[1])/1000}}};
        list.push_back(hole);
        document.draft = set_enclosure_openings(document.draft, case_id, list);
        enclosure_id = case_id;
        select_case_opening(hole);
        place_case_opening = false;
        message = "Opening added on " + face + ". Edit its size/position below; preview updates automatically.";
        return true;
    } catch (const std::exception& e) {
        case_opening_error = e.what();
        message = case_opening_error;
        return false;
    }
}
void ExperimentEditor::Impl::add_case_opening() {
    const auto mesh = editable_enclosures(document.draft).at(enclosure_id);
    const auto dims = mesh.at("inside_dimensions_m").get<std::array<double, 3>>();
    const int axis = case_opening_face[1] - 'x';
    const double w = case_opening_size_mm[0]/1000,
                 h = (case_opening_shape == 0 ? case_opening_size_mm[0] : case_opening_size_mm[1])/1000;
    if (!std::isfinite(w) || !std::isfinite(h) || w <= 0 || h <= 0) {
        case_opening_error = "Opening dimensions must be positive.";
        return;
    }
    auto fits = [&](std::array<double, 2> center) {
        auto candidate = mesh;
        candidate["openings"].push_back({{"id", "candidate-new-hole"}, {"face", case_opening_face},
            {"shape", case_opening_shape == 0 ? "circle" : "rectangle"},
            {"center_m", center}, {"size_m", {w, h}}});
        try { (void)parse_enclosure_openings(candidate); return true; }
        catch (const std::exception&) { return false; }
    };
    // Prefer the user's coordinates. If occupied, search only this face, not
    // a different wall or a silently smaller diameter. No solver trials.
    std::array<double, 2> target{case_opening_center_mm[0]/1000, case_opening_center_mm[1]/1000};
    if (!fits(target)) {
        const double u = (dims[(axis+1)%3]-w)/2, v = (dims[(axis+2)%3]-h)/2;
        bool found = false;
        for (int j = 0; j <= 12 && !found && u >= 0 && v >= 0; ++j)
            for (int i = 0; i <= 12; ++i) {
                target = {-u + 2*u*i/12, -v + 2*v*j/12};
                if (fits(target)) { found = true; break; }
            }
        if (!found) {
            case_opening_error = "No free spot of this size on the selected face. Choose another face or reduce the size.";
            return;
        }
    }
    (void)commit_case_opening(enclosure_id, case_opening_face, target);
}
void ExperimentEditor::Impl::place_opening_at(int wall_index, Vec3 point) {
    try {
        const auto& wall = preview_scenario.cells.at(wall_index);
        const auto meshes = editable_enclosures(document.draft);
        std::string case_id, face;
        for (const auto& [id, mesh] : meshes.items())
            for (int axis = 0; axis < 3; ++axis)
                for (int sign : {-1, 1})
                    if (wall.id == id + (sign > 0 ? "-plus" : "-minus") + "xyz"[axis]) {
                        case_id = id;
                        face = std::string(sign > 0 ? "+" : "-") + "xyz"[axis];
                    }
        if (case_id.empty())
            throw std::runtime_error("Click a case wall");
        const auto& mesh = meshes.at(case_id);
        const auto center = mesh.at("position_m").get<std::array<double, 3>>();
        const double p[] = {point.x, point.y, point.z};
        const int axis = face[1] - 'x', u = (axis + 1) % 3, v = (axis + 2) % 3;
        const std::array<double, 2> local{p[u] - center[u], p[v] - center[v]};
        auto list = mesh.value("openings", J::array());
        for (const auto& h : parse_enclosure_openings(mesh)) {
            if (h.face != face)
                continue;
            const double du = local[0] - h.center[0], dv = local[1] - h.center[1];
            const bool inside = h.circle ? du * du + dv * dv <= h.size[0] * h.size[0] / 4
                                         : std::abs(du) <= h.size[0] / 2 && std::abs(dv) <= h.size[1] / 2;
            if (inside) {
                if (case_opening_selection.creating) {
                    case_opening_error = "There is already an opening here. Click another spot on the wall.";
                    return;
                }
                enclosure_id = case_id;
                for (const auto& item : list)
                    if (item.at("id") == h.id)
                        select_case_opening(item);
                selected.clear();
                return;
            }
        }
        (void)commit_case_opening(case_id, face, local);
    } catch (const std::exception& e) {
        message = e.what();
        case_opening_error = e.what();
    }
}
} // namespace battery::ui
