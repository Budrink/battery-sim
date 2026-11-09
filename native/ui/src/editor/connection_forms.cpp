#include "editor_internal.hpp"
#include "form_fields.hpp"

namespace battery::ui {
namespace {
void conductance(nlohmann::json& law) {
    ImGui::PushID("conductance");
    form::number(law, "conductance_w_per_k", "Thermal conductance [W/K]");
    form::number(law, "background_conductance_w_per_k", "Background thermal conductance [W/K]");
    form::number(law, "active_conductance_w_per_k", "State-dependent thermal conductance [W/K]");
    form::number(law, "destroyed_background_conductance_w_per_k",
                 "Background conductance after destruction [W/K]");
    form::number(law, "destroyed_active_conductance_w_per_k",
                 "State-dependent conductance after destruction [W/K]");
    form::number(law, "geometry_reference_logit", "Reference connectivity bias (log-odds)");
    form::number(law, "endpoint_damage_weight", "Influence of endpoint damage on connectivity");
    ImGui::PopID();
}
} // namespace
void ExperimentEditor::Impl::draw_connections() {
    auto& assembly = document.draft.at("assembly");
    ImGui::TextWrapped(
        "Choose a connection to edit its physical properties. Values apply only to the selected "
        "connection. Gas reaction chemistry is global in Models.");
    ImGui::SeparatorText("Solid thermal contacts");
    if (auto* edge = form::record("Contact", assembly.at("interfaces"))) {
        form::text(*edge, "port_a", "First contact surface");
        form::text(*edge, "port_b", "Second contact surface");
        if (edge->contains("law"))
            conductance(edge->at("law"));
        if (edge->contains("radiation")) {
            auto& radiation = edge->at("radiation");
            form::number(radiation, "exchange_area_m2", "Radiating exchange area [mm2]", 1e6);
            form::number(radiation, "effective_emissivity", "Effective emissivity [0..1]");
        }
    }
    ImGui::SeparatorText("Contact surfaces");
    if (auto* port = form::record("Surface", assembly.at("ports"))) {
        ImGui::TextWrapped("Object: %s; face: %s", port->value("cell_id", "").c_str(),
                           port->value("surface", "").c_str());
        form::number(*port, "area_m2", "Contact area [mm2]", 1e6);
        form::vector(*port, "rectangle_m", "Contact dimensions [mm]", 1000.);
        form::vector(*port, "centroid_m", "Contact center [mm]", 1000.);
        form::vector(*port, "normal", "Outward normal (unit vector)");
    }
    ImGui::SeparatorText("Vent destinations");
    if (auto* vent = form::record("Vent connection", assembly.at("vent_connections"))) {
        ImGui::TextWrapped("Battery: %s; vent: %s", vent->value("battery_id", "").c_str(),
                           vent->value("vent_id", "").c_str());
        form::text(*vent, "to", "Destination gas region (outside_domain = environment)");
    }
    auto& gas = assembly.at("gas_network");
    ImGui::SeparatorText("Gas transport");
    ImGui::TextWrapped("%d declared gas regions; generated volumes are built automatically from the case. "
                       "No per-volume combustion editing is needed.",
                       int(gas.at("compartments").size()));
    form::number(gas, "mixing_diffusivity_m2_s", "Gas mixing diffusivity [m2/s]");
    form::number(gas, "mixing_flow_fraction", "Additional mixing relative to directed flow");
    form::number(gas, "thermal_stirring_coefficient", "Buoyancy-driven mixing multiplier");
    form::number(gas, "thermal_stirring_length_m", "Buoyancy mixing length [mm]", 1000.);
    form::number(gas, "mixing_length_cap_m", "Maximum mixing length [mm]", 1000.);
    form::number(gas, "dynamic_viscosity_pa_s", "Gas dynamic viscosity [Pa s]");
    form::number(gas, "cv_j_per_kg_k", "Gas specific heat at constant volume [J/(kg K)]");
    form::number(gas, "specific_r_j_per_kg_k", "Gas specific constant [J/(kg K)]");
    ImGui::TextWrapped(
        "Initial gas and ambient boundary conditions are configured with the case. "
        "Imported advanced connectivity laws are retained; JSON is only a technical fallback.");
}
void ExperimentEditor::Impl::draw_cooling() {
    ImGui::SeparatorText("External cooling");
    auto& boundaries = document.draft.at("conditions").at("boundaries");
    if (auto* boundary = form::record("Cooling boundary", boundaries)) {
        form::text(*boundary, "port_id", "Cooled contact surface");
        form::number(*boundary, "temperature_k", "Environment / coolant temperature [C]", 1., -273.15);
        form::number(*boundary, "conductance_w_per_k", "Boundary thermal conductance [W/K]");
        if (boundary->contains("connectivity"))
            conductance(boundary->at("connectivity"));
        if (boundary->contains("radiation")) {
            auto& r = boundary->at("radiation");
            form::number(r, "effective_emissivity", "Surface emissivity [0..1]");
            form::number(r, "area_fraction", "Fraction of surface exposed to radiation [0..1]");
        }
        if (boundary->contains("temperature_profile"))
            ImGui::TextWrapped("A measured coolant temperature profile is attached; it takes precedence over "
                               "a constant temperature.");
    }
}
} // namespace battery::ui
