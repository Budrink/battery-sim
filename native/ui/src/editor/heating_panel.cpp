#include "editor_internal.hpp"
#include "battery_native/contracts/experiment_editing.hpp"
#include "imgui.h"
#include <algorithm>
#include <set>

namespace battery::ui {
using J = nlohmann::json;
void ExperimentEditor::Impl::draw_heating() {
    ImGui::SeparatorText("Place a heater");
    ImGui::TextWrapped("Choose its location, then edit all parameters in the heater card below. "
                       "New and existing heaters use the same editor.");
    const auto& batteries = document.draft["assembly"]["batteries"];
    const auto is_battery = [&](const std::string& id) {
        return std::any_of(batteries.begin(), batteries.end(),
                           [&](const auto& b) { return b.at("id") == id; });
    };
    if (!is_battery(heater_target))
        heater_target = is_battery(selected) ? selected
                        : batteries.empty()  ? std::string{}
                                             : batteries.front().at("id").get<std::string>();
    if (ImGui::BeginCombo("Target battery", heater_target.c_str())) {
        for (const auto& b : batteries) {
            auto id = b.at("id").get<std::string>();
            if (ImGui::Selectable(id.c_str(), heater_target == id))
                heater_target = id;
        }
        ImGui::EndCombo();
    }
    bool cylinder = false;
    for (const auto& b : batteries)
        if (b.at("id") == heater_target)
            for (const auto& m : document.draft.at("battery_models"))
                if (m.at("id") == b.at("model_id") && m.at("revision") == b.at("model_revision"))
                    cylinder = m.at("geometry").at("kind") == "cylinder";
    int mounting = ImGui::GetStateStorage()->GetInt(ImGui::GetID("cylinder_mount"), 0);
    if (cylinder) {
        ImGui::TextWrapped("Heater mounting");
        ImGui::SetNextItemWidth(-1);
        if (ImGui::Combo("##mount", &mounting, "Wrap around side\0Top circular disk\0Bottom circular disk\0"))
            ImGui::GetStateStorage()->SetInt(ImGui::GetID("cylinder_mount"), mounting);
    } else
        ImGui::Combo("Contact face", &face, "+x\0-x\0+y\0-y\0+z\0-z\0");
    std::string created;
    ImGui::BeginDisabled(batteries.empty());
    if (ImGui::Button("Add external heater")) {
        try {
            const char* faces[]{"+x", "-x", "+y", "-y", "+z", "-z"};
            // Preserve the previous creation defaults, but expose their entire
            // protocol immediately in the same card used for imported heaters.
            const char* mounts[]{"wrap", "+z", "-z"};
            document.draft =
                add_external_heater(document.draft, heater_target, cylinder ? mounts[mounting] : faces[face],
                                    5., 500., 5., false, .1, 773.15, 0., true);
            created = selected = document.draft["assembly"]["passive_cells"].back().at("id");
            case_opening_selection.id.clear();
            message = "Heater added. Edit its power, contact, controller and cutoff in its card.";
        } catch (const std::exception& e) {
            message = e.what();
        }
    }
    ImGui::EndDisabled();

    std::string remove_heater, enable_control, remount_heater, remount_battery, remount_face;
    auto& assembly = document.draft["assembly"];
    auto& protocols = document.draft["conditions"]["heating_protocols"];
    for (auto& obj : assembly["passive_cells"]) {
        if (obj.value("material_id", "") != "external-resistive-heater" &&
            obj.value("material_id", "") != "heater" &&
            std::none_of(protocols.begin(), protocols.end(),
                         [&](const auto& p) { return p.at("heater_id") == obj.at("id"); }))
            continue;
        const auto id = obj.at("id").get<std::string>();
        ImGui::PushID(id.c_str());
        ImGui::SeparatorText(id.c_str());
        if (created == id)
            ImGui::SetScrollHereY(0.f);
        if (ImGui::Selectable("Select heater", selected == id)) {
            selected = id;
            case_opening_selection.id.clear();
        }
        if (ImGui::Button("Delete heater"))
            remove_heater = id;

        J* protocol = nullptr;
        for (auto& p : protocols)
            if (p.at("heater_id") == id)
                protocol = &p;
        J* contact = nullptr;
        std::vector<J*> contacts;
        std::set<std::string> own_ports;
        for (const auto& p : assembly["ports"])
            if (p.at("cell_id") == id)
                own_ports.insert(p.at("id"));
        for (auto& edge : assembly["interfaces"])
            if (own_ports.contains(edge.at("port_a")) || own_ports.contains(edge.at("port_b")))
                contacts.push_back(&edge);
        if (!contacts.empty())
            contact = contacts.front();
        double area = 0.;
        std::string attached;
        if (contact) {
            const auto target = contact->at(own_ports.contains(contact->at("port_a")) ? "port_b" : "port_a")
                                    .get<std::string>();
            attached = target.substr(0, target.find(':'));
            ImGui::TextWrapped("Attached to: %s (%d thermal contacts)", attached.c_str(),
                               int(contacts.size()));
            for (const auto& port : assembly["ports"])
                if (port.at("id") ==
                    contact->at(own_ports.contains(contact->at("port_a")) ? "port_a" : "port_b"))
                    area = port.at("area_m2");
        }
        for (const auto& b : assembly["batteries"])
            if (b.at("id") == attached)
                for (const auto& m : document.draft["battery_models"])
                    if (m.at("id") == b.at("model_id") && m.at("revision") == b.at("model_revision") &&
                        m.at("geometry").at("kind") == "cylinder") {
                        ImGui::TextWrapped("Cylinder mounting (keeps this controller and total power)");
                        const auto mounts =
                            document.draft.value("metadata", J::object()).value("heater_mounts", J::object());
                        const auto current =
                            mounts.contains(id) ? mounts.at(id).at("face").get<std::string>() : "legacy";
                        const char* names[]{"Wrap side", "Top disk", "Bottom disk"};
                        const char* faces[]{"wrap", "+z", "-z"};
                        for (int i = 0; i < 3; ++i) {
                            if (ImGui::RadioButton(names[i], current == faces[i])) {
                                remount_heater = id;
                                remount_battery = attached;
                                remount_face = faces[i];
                            }
                        }
                    }
        bool enabled = protocol != nullptr;
        if (ImGui::Checkbox("Temperature controller / cutoff", &enabled)) {
            if (enabled)
                enable_control = id;
            else {
                protocols.erase(std::remove_if(protocols.begin(), protocols.end(),
                                               [&](const auto& p) { return p.at("heater_id") == id; }),
                                protocols.end());
                protocol = nullptr;
            }
        }
        const auto old_law = contact ? contact->at("law") : J();
        draw_heater_parameters(obj, protocol, contact, area);
        // The card's common h scales every sector, not just the first quarter.
        if (contact && old_law != contact->at("law"))
            for (auto* other : contacts)
                if (other != contact)
                    for (const auto& [key, value] : old_law.items())
                        if (value.is_number() && value.get<double>() != 0 && other->at("law").contains(key)) {
                            const double ratio =
                                contact->at("law").at(key).get<double>() / value.get<double>();
                            other->at("law")[key] = other->at("law").at(key).get<double>() * ratio;
                        }
        ImGui::PopID();
    }
    // Replacement is deferred: no references into the draft survive these edits.
    if (!remove_heater.empty()) {
        selected = remove_heater;
        case_opening_selection.id.clear();
        delete_selected();
        return;
    }
    if (!enable_control.empty()) {
        try {
            document.draft = set_heater_cutoff(document.draft, enable_control, 773.15, 0.);
            message = "Controller enabled for this heater; constant power preserved (gain 0).";
        } catch (const std::exception& e) {
            message = e.what();
        }
    }
    if (!remount_heater.empty()) {
        try {
            document.draft = set_heater_mount(document.draft, remount_heater, remount_battery, remount_face);
            message = "Heater mounting changed. Same controller/power; contact area follows the cylinder.";
        } catch (const std::exception& e) {
            message = e.what();
        }
    }
    draw_cooling();
}
} // namespace battery::ui
