#include "editor/editor_internal.hpp"
#include "shared/form_fields.hpp"
#include "battery_native/contracts/experiment_editing.hpp"
#include "battery_native/contracts/enclosure_openings.hpp"
#include "imgui.h"
#include "misc/cpp/imgui_stdlib.h"

namespace battery::ui {
void ExperimentEditor::Impl::draw_enclosure() {
    using J = nlohmann::json;
    const auto cases = editable_enclosures(document.draft);
    if (!cases.empty()) {
        ImGui::SetNextItemWidth(-1);
        if (form::begin_combo("##case-selection", enclosure_id.c_str())) {
            for (const auto& [id, spec] : cases.items())
                if (ImGui::Selectable(id.c_str(), id == enclosure_id))
                    enclosure_id = id;
            ImGui::EndCombo();
        }
    }
    // The selected object's identifier is not a creation draft. Editing it
    // used to switch the whole panel to Add case and hide the hole tool.
    ImGui::BeginDisabled(!cases.empty());
    form::text_input("Case ID", &enclosure_id);
    ImGui::EndDisabled();
    draw_case_openings();
    const auto registry =
        document.draft.value("metadata", J::object()).value("editor_enclosures", J::object());
    const bool explicit_case =
        registry.contains(enclosure_id) && registry.at(enclosure_id).contains("explicit_mesh");
    const auto property_source =
        registry.contains(enclosure_id) ? registry.at(enclosure_id).at("properties") : J();
    if (enclosure_loaded != enclosure_id || enclosure_properties.is_null() ||
        enclosure_source != property_source) {
        if (enclosure_dirty) {
            commit_panel_edits();
            if (enclosure_dirty)
                ImGui::TextWrapped("Correct or discard the pending case input before changing case.");
            return;
        }
        enclosure_source = property_source;
        enclosure_loaded = enclosure_id;
        enclosure_properties = {{"inside_dimensions_m", {.12, .12, .08}},
                                {"position_m", {0, 0, 0}},
                                {"thickness_m", .002},
                                {"material_id", "demo-case"},
                                {"density_kg_m3", 1200},
                                {"specific_heat_j_kg_k", 1200},
                                {"conductivity_w_per_m_k", {.2, .2, .2}},
                                {"initial_temperature_k", 298.15},
                                {"ambient_temperature_k", 298.15},
                                {"gas_contact_h_w_m2_k", 10},
                                {"ambient_h_w_m2_k", 0},
                                {"damage_onset_k", 500},
                                {"damage_rate_per_s", 0},
                                {"damage_scale_k", 40},
                                {"gas_cell_size_m", .02},
                                {"solid_contact_h_w_m2_k", 10},
                                {"gas_discharge_coefficient", .7}};
        if (registry.contains(enclosure_id)) {
            enclosure_properties = registry.at(enclosure_id).at("properties");
            enclosure_gas = registry.at(enclosure_id).value("gas_compartment", "");
        }
        const J defaults = {
            {"gas_cell_size_m", .02}, {"solid_contact_h_w_m2_k", 10}, {"gas_discharge_coefficient", .7}};
        for (const auto& [key, value] : defaults.items())
            if (!enclosure_properties.contains(key))
                enclosure_properties[key] = value;
        if (!explicit_case)
            enclosure_properties["volumetric_heat_capacity_j_m3_k"] =
                enclosure_properties.at("density_kg_m3").get<double>() *
                enclosure_properties.at("specific_heat_j_kg_k").get<double>();
    }
    auto& p = enclosure_properties;
    const J fields_before = {p, enclosure_gas};
    auto number = [&](const char* label, const char* key, double scale = 1., double offset = 0.) {
        form::number(p, key, label, scale, offset);
    };
    auto vector = [&](const char* label, const char* key, double scale) {
        for (int k = 0; k < 3; ++k) {
            const auto title = std::string(label) + " " + "XYZ"[k];
            double x = p.at(key)[k].get<double>() * scale;
            if (form::input(title.c_str(), &x, "%.8g", key))
                p[key][k] = x / scale;
        }
    };
    ImGui::TextWrapped("Six finite-capacity walls. Gas contact and heat loss use the common connectivity "
                       "law. Dimensions are INTERNAL.");
    ImGui::BeginChild("case capabilities", {0, ImGui::GetTextLineHeightWithSpacing() * 3}, false);
    ImGui::TextWrapped(
        "%s", explicit_case
                  ? "Existing gas mesh: dimensions are locked. Openings and wall material remain editable."
                  : "Generated gas mesh: dimensions, openings and wall material are editable.");
    ImGui::EndChild();
    ImGui::BeginDisabled(explicit_case);
    vector("Inside mm", "inside_dimensions_m", 1000);
    vector("Center mm", "position_m", 1000);
    number("Wall thickness mm", "thickness_m", 1000);
    ImGui::EndDisabled();
    auto material = p.at("material_id").get<std::string>();
    if (form::text_input("Wall material ID", &material))
        p["material_id"] = material;
    number("Volumetric heat capacity J/(m3 K)", "volumetric_heat_capacity_j_m3_k");
    if (!explicit_case &&
        p.at("volumetric_heat_capacity_j_m3_k") != fields_before.at(0).at("volumetric_heat_capacity_j_m3_k"))
        p["specific_heat_j_kg_k"] =
            p.at("volumetric_heat_capacity_j_m3_k").get<double>() / p.at("density_kg_m3").get<double>();
    vector("Conductivity W/(m K)", "conductivity_w_per_m_k", 1);
    number("Initial C", "initial_temperature_k", 1, -273.15);
    ImGui::BeginDisabled(explicit_case);
    auto network_number = [&](const char* label, const char* key, double scale = 1., double offset = 0.) {
        if (explicit_case) {
            ImGui::TextWrapped("%s", label);
            ImGui::SetNextItemWidth(-1);
            ImGui::PushID(key);
            ImGui::Button("Defined by existing gas network", {-1, 0});
            ImGui::PopID();
        } else
            number(label, key, scale, offset);
    };
    network_number("Ambient C", "ambient_temperature_k", 1, -273.15);
    network_number("Gas-to-case heat transfer [W/(m2 K)]", "gas_contact_h_w_m2_k");
    network_number("Gas-to-solid heat transfer [W/(m2 K)]", "solid_contact_h_w_m2_k");
    network_number("Gas maximum cell size mm", "gas_cell_size_m", 1000);
    network_number("Gas passage discharge coefficient", "gas_discharge_coefficient");
    network_number("External case cooling [W/(m2 K)]", "ambient_h_w_m2_k");
    form::hint("0 = no external cooling; walls still store and exchange heat.");
    network_number("Damage onset C", "damage_onset_k", 1, -273.15);
    network_number("Damage rate 1/s (0 = rigid)", "damage_rate_per_s");
    network_number("Damage scale K", "damage_scale_k");
    ImGui::TextWrapped("Enclosed gas volume");
    ImGui::SetNextItemWidth(-1);
    if (ImGui::BeginCombo("##enclosed_gas",
                          explicit_case ? "Existing spatial gas network" : enclosure_gas.c_str())) {
        for (const auto& g : document.draft.at("assembly").at("gas_network").at("compartments")) {
            if (g.contains("contents_id") || g.contains("local_equilibrium"))
                continue;
            auto id = g.at("id").get<std::string>();
            if (ImGui::Selectable(id.c_str(), enclosure_gas == id))
                enclosure_gas = id;
        }
        ImGui::EndCombo();
    }
    ImGui::SeparatorText("Initial gas in the case");
    if (explicit_case) {
        network_number("Initial gas temperature [C]", "initial_temperature_k");
        network_number("Initial oxygen mass fraction [%]", "initial_oxidizer_mass_fraction");
        network_number("Initial gas mass [mg]", "initial_mass_kg");
        ImGui::TextWrapped(
            "Initial gas state belongs to the existing volumes. R3 chemistry is shared in Gas model / R3.");
    }
    for (auto& g : document.draft.at("assembly").at("gas_network").at("compartments"))
        if (g.at("id") == enclosure_gas) {
            form::number(g, "initial_temperature_k", "Initial gas temperature [C]", 1., -273.15);
            form::number(g, "initial_oxidizer_mass_fraction", "Initial oxygen mass fraction [%]", 100.);
            form::number(g, "initial_mass_kg", "Initial gas mass [mg]", 1e6);
            ImGui::TextWrapped(
                "Initial oxygen is local to this cavity. R3 chemistry is shared in Gas model / R3.");
        }
    ImGui::TextWrapped(
        "Spatial gas mesh is generated in free space at Preview / Run. Solids block passages; narrow gaps "
        "stay open. Openings connect only adjacent gas cells to outside air. No automatic paths through "
        "damaged case walls or wall-to-wall solid seam contacts are inferred.");
    ImGui::EndDisabled();
    if (registry.contains(enclosure_id) && registry.at(enclosure_id).contains("free_volume_m3"))
        ImGui::Text("Free gas volume: %.1f cm3",
                    registry.at(enclosure_id).at("free_volume_m3").get<double>() * 1e6);
    else
        ImGui::TextUnformatted("Free gas volume: defined by existing spatial volumes");
    const bool existing = registry.contains(enclosure_id);
    if (existing)
        enclosure_dirty |= fields_before != J{p, enclosure_gas};
    else if (ImGui::Button("Create case"))
        enclosure_dirty = true;
    ImGui::TextWrapped(existing ? "Valid wall changes apply automatically when you leave the field."
                                : "Creation adds walls and a gas mesh. Review dimensions before creating.");
}
} // namespace battery::ui
