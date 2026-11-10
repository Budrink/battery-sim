#include "editor/forms/model_properties.hpp"
#include "imgui.h"
#include "shared/form_fields.hpp"
#include <cmath>

namespace battery::ui {
namespace {
using J = nlohmann::json;
// All forms share width-safe labels, units and the same tooltip/reference source.
void number(J& object, const char* key, const char* label, double scale = 1., double offset = 0.) {
    form::number(object, key, label, scale, offset);
}
void vector(J& object, const char* key, const char* label) {
    form::vector(object, key, label);
}
bool section(const char* name) {
    ImGui::Spacing();
    return ImGui::CollapsingHeader(name);
}
} // namespace

void draw_model_properties(J& model) {
    auto& contents = model.at("contents");
    auto& thermal = contents.at("thermal");
    ImGui::SeparatorText("Heat capacity");
    number(thermal, "heat_capacity_j_per_k", "Initial contents heat capacity [J/K]");
    ImGui::TextWrapped("Energy needed to heat the entire contents by 1 degree. Not specific heat per kg. "
                       "The metal shell is additional and has its own heat capacity.");
    if (contents.contains("conversion")) {
        auto& c = contents.at("conversion");
        ImGui::PushID("capacity_components");
        ImGui::Spacing();
        ImGui::TextUnformatted("Reactive material");
        number(c, "mass_kg", "Reactive material mass [g]", 1000);
        number(c, "reactant_cp", "Reactive material specific heat [J/(kg K)]");
        const double reactive = c.value("mass_kg", 0.) * c.value("reactant_cp", 0.);
        ImGui::TextWrapped("Contribution: %.3f g x %.3f J/(kg K) / 1000 = %.3f J/K",
                           1000 * c.value("mass_kg", 0.), c.value("reactant_cp", 0.), reactive);
        double liquid = 0.;
        if (c.contains("volatile_phase")) {
            auto& v = c.at("volatile_phase");
            ImGui::Spacing();
            ImGui::TextUnformatted("Evaporating liquid");
            number(v, "initial_mass_kg", "Liquid mass [g]", 1000);
            number(v, "liquid_cp", "Liquid specific heat [J/(kg K)]");
            liquid = v.value("initial_mass_kg", 0.) * v.value("liquid_cp", 0.);
            ImGui::TextWrapped("Contribution: %.3f g x %.3f J/(kg K) / 1000 = %.3f J/K",
                               1000 * v.value("initial_mass_kg", 0.), v.value("liquid_cp", 0.), liquid);
        }
        const double total = thermal.at("heat_capacity_j_per_k");
        const double inert = total - reactive - liquid;
        ImGui::Spacing();
        ImGui::TextUnformatted("Heat capacity balance");
        ImGui::TextWrapped("Reactive material: %.3f J/K\nLiquid: %.3f J/K\n"
                           "Other inert contents (total minus components): %.3f J/K",
                           reactive, liquid, inert);
        if (inert <= 0.) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{1, .45f, .3f, 1});
            ImGui::TextWrapped("Cannot apply: total %.3f J/K must be greater than the components' "
                               "%.3f J/K. Increase the total above, or reduce the mass / specific heat "
                               "in the fields directly above. No values are adjusted automatically.",
                               total, reactive + liquid);
            ImGui::PopStyleColor();
        } else {
            ImGui::TextWrapped("Balance valid. Component heat capacities are included in the total, "
                               "not added on top of it. The shell is separate.");
        }
        ImGui::Spacing();
        ImGui::TextUnformatted("Heat storage after reaction");
        number(c, "activated_cp", "R2 precursor specific heat [J/(kg K)]");
        number(c, "residue_cp", "Reaction residue specific heat [J/(kg K)]");
        ImGui::TextWrapped("These two values determine heat capacity as the composition changes. "
                           "Changing component properties does not automatically change the total above.");
        ImGui::PopID();
    }
    if (section("Internal conduction")) {
        vector(thermal, "conductivity_w_per_m_k", "Contents thermal conductivity [W/(m K)]");
        if (contents.contains("conversion"))
            vector(contents.at("conversion"), "residue_conductivity",
                   "Residue / original conductivity ratio (dimensionless)");
        ImGui::TextWrapped("Local model axes. These values describe conduction, not heat storage. "
                           "State-dependent connectivity further modifies the heat flow.");
        if (thermal.contains("bulk_connectivity")) {
            auto& b = thermal.at("bulk_connectivity");
            number(b, "background_factor", "Background conduction multiplier");
            number(b, "active_factor", "State-dependent conduction multiplier");
            number(b, "destroyed_background_factor", "Background multiplier after destruction");
            number(b, "destroyed_active_factor", "State-dependent multiplier after destruction");
        }
    }
    if (contents.contains("conversion")) {
        auto& c = contents.at("conversion");
        ImGui::PushID("conversion");
        if (section("R1: degradation")) {
            ImGui::TextWrapped(
                "Thermally accelerated degradation produces R2 precursor. R1 is heat-neutral.");
            number(c, "degradation_rate_per_s", "Degradation rate at reference temperature [1/s]");
            number(c, "degradation_reference_k", "R1 reference temperature [C]", 1, -273.15);
            number(c, "degradation_activation_j_per_mol", "R1 Arrhenius activation energy [kJ/mol]", .001);
            number(c, "precursor_yield", "Fraction of degraded mass becoming R2 precursor [0..1]");
            number(c, "initial_precursor_fraction", "Initial degraded R2 precursor mass fraction [0..1]");
            number(c, "initial_charged_fraction", "Initial charged material mass fraction [0..1]");
        }
        if (section("R2: runaway")) {
            number(c, "reaction_rate_per_s", "R2 reaction rate at reference temperature [1/s]");
            number(c, "reaction_reference_k", "R2 reference temperature [C]", 1, -273.15);
            number(c, "reaction_activation_j_per_mol", "R2 Arrhenius activation energy [kJ/mol]", .001);
            number(c, "net_heat_j_per_kg", "Chemical energy per reacted mass [kJ/kg]", .001);
            number(c, "gas_yield", "Reacted mass converted to gas [kg/kg]");
            number(c, "gas_yield_max", "Maximum gas mass yield [kg/kg]");
            number(c, "gas_chemical_fraction", "Released chemical energy retained as gas fuel [0..1]");
            ImGui::TextWrapped("R3 combustion parameters are in Gas model / R3. "
                               "They apply to all gas volumes, including newly generated ones.");
        }
        if (c.contains("volatile_phase") && section("Liquid and evaporation")) {
            auto& v = c.at("volatile_phase");
            ImGui::TextWrapped("Liquid mass and specific heat are in Heat capacity at the top of this form.");
            number(v, "density_kg_m3", "Liquid density [kg/m3]");
            number(v, "latent_j_per_kg", "Latent heat of evaporation [kJ/kg]", .001);
            number(v, "boiling_temperature_k", "Reference boiling temperature [C]", 1, -273.15);
            number(v, "saturation_reference_pa", "Reference saturation pressure [kPa]", .001);
            number(v, "relaxation_per_s", "Evaporation / condensation relaxation rate [1/s]");
            ImGui::TextWrapped("The current gas model requires common liquid species properties across "
                               "the network. Validate after editing; mass may differ between batteries.");
        }
        ImGui::PopID();
    }
    if (contents.contains("electrical") && section("Electrical properties")) {
        auto& e = contents.at("electrical");
        number(e, "resistance_ohm", "Reference internal resistance [milliohm]", 1000);
        number(e, "nominal_capacity_coulomb", "Nominal electrical capacity [Ah]", 1. / 3600.);
        number(e, "charge_acceptance_fraction", "Charge acceptance fraction [0..1]");
        ImGui::TextWrapped("Electrical capacity (Ah) is not heat capacity (J/K). Wiring and supplied "
                           "current are configured in Circuit.");
    }
    if (model.contains("shell") && section("Shell material")) {
        auto& s = model.at("shell");
        number(s, "thickness_m", "Shell thickness [mm]", 1000);
        number(s, "density_kg_m3", "Shell material density [kg/m3]");
        number(s, "specific_heat_j_kg_k", "Shell material specific heat [J/(kg K)]");
        vector(s, "conductivity_w_per_m_k", "Shell conductivity [W/(m K)]");
        if (s.contains("contents_connectivity_per_m2")) {
            auto& contact = s.at("contents_connectivity_per_m2");
            ImGui::TextWrapped("Contents-to-shell heat exchange uses a state-dependent contact law:");
            number(contact, "background_conductance_w_per_k",
                   "Background contact conductance per area [W/(m2 K)]");
            number(contact, "active_conductance_w_per_k",
                   "State-dependent contact conductance amplitude [W/(m2 K)]");
            number(contact, "destroyed_background_conductance_w_per_k",
                   "Background contact conductance after destruction [W/(m2 K)]");
            number(contact, "destroyed_active_conductance_w_per_k",
                   "State-dependent amplitude after destruction [W/(m2 K)]");
        } else {
            number(s, "contents_contact_h", "Contents-to-shell contact conductance per area [W/(m2 K)]");
        }
        number(s, "seam_contact_h", "Between-shell-plates contact conductance per area [W/(m2 K)]");
        number(s, "damage_onset_k", "Thermal damage onset temperature [C]", 1, -273.15);
        number(s, "damage_rate_per_s", "Thermal damage rate at onset [1/s]");
        number(s, "damage_scale_k", "Temperature rise for an e-fold increase in damage rate [K]");
        ImGui::TextWrapped("Shell capacity = density x specific heat x plate volume. Face overrides "
                           "take precedence over these defaults. Edit individual faces, seams and valves "
                           "in Shell / vents. Stateful contact laws may override the simple contact value.");
    }
    ImGui::Spacing();
    ImGui::TextWrapped(
        "Shared shape and dimensions are below; selected-instance shape is also in Assembly. "
        "Face materials and rupture: Shell / vents. "
        "Additional imported laws remain intact. Parameter reference explains the exposed controls.");
}
} // namespace battery::ui
