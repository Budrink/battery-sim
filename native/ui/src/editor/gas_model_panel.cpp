#include "editor_internal.hpp"
#include "form_fields.hpp"
#include "gas_model_panel.hpp"
#include "battery_native/contracts/gas_combustion_model.hpp"

namespace battery::ui {
void ExperimentEditor::Impl::draw_gas_model() {
    draw_gas_combustion_model(document.draft.at("assembly").at("gas_network"), model_edits);
}
void draw_gas_combustion_model(const nlohmann::json& network, ModelEdits& model_edits) {
    ImGui::SeparatorText("Global R3 combustion");
    ImGui::TextWrapped(
        "One combustion law for the entire experiment: battery pores, free gas and all generated "
        "case volumes. Not limited to the selected battery. Oxygen inventory stays local.");
    const auto variants = gas_combustion_variants(network);
    if (variants.size() > 1)
        ImGui::TextWrapped(
            "Imported file contains %d different R3 laws. The first active law is shown as a starting point. "
            "Apply explicitly to replace all of them.",
            int(variants.size()));
    auto& edit = model_edits.edit_gas(network);
    ImGui::PushID("global_r3");
    auto& law = edit.value.at("combustion");
    form::number(law, "reaction_rate_per_s", "Combustion rate at reference temperature [1/s]");
    form::number(law, "reaction_reference_k", "Reference gas temperature [C]", 1., -273.15);
    form::number(law, "reaction_activation_j_per_mol",
                 "Temperature sensitivity: Arrhenius activation energy [kJ/mol]", .001);
    form::number(law, "oxidizer_heat_j_per_kg", "Energy released per consumed oxygen mass [MJ/kg O2]", 1e-6);
    form::number(edit.value, "vapor_fuel_heat_j_kg",
                 "Combustible energy carried by evaporated liquid [MJ/kg vapour]", 1e-6);
    ImGui::TextWrapped(
        "This common fuel energy also sets the chemical reserve of that liquid in every battery. "
        "It does not change heat capacity or evaporation latent heat.");
    ImGui::TextWrapped(
        "Oxygen energy is NOT fuel heating value. Actual heat is limited by both available fuel energy "
        "and oxygen. R2 fuel production is configured in the battery model below.");
    bool threshold = law.value("ignition_temperature_k", 0.) > 0;
    ImGui::TextWrapped("Optional ignition temperature cutoff");
    if (ImGui::Checkbox("Enable cutoff", &threshold))
        law["ignition_temperature_k"] = threshold ? law.at("reaction_reference_k") : nlohmann::json(0.);
    if (threshold)
        form::number(law, "ignition_temperature_k", "No R3 below this gas temperature [C]", 1., -273.15);
    ImGui::TextWrapped("Valid changes apply when you finish typing. Running simulation remains unchanged "
                       "until RUN SNAPSHOT.");
    if (ImGui::Button("Apply to all gas volumes"))
        model_edits.apply_global_gas();
    if (!edit.error.empty())
        ImGui::TextWrapped("Cannot apply: %s", edit.error.c_str());
    if ((edit.value != edit.source || !edit.error.empty()) && ImGui::Button("Discard R3 input"))
        model_edits.discard_gas();
    ImGui::PopID();
    ImGui::Spacing();
}
} // namespace battery::ui
