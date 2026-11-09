#include "editor_internal.hpp"
#include "form_fields.hpp"

namespace battery::ui {
void ExperimentEditor::Impl::draw_run_settings() {
    auto& settings = document.draft.at("numerics").at("simulation");
    ImGui::SeparatorText("Simulation time");
    form::number(settings, "duration_s", "Simulation duration [s]");
    form::number(settings, "max_dt_s", "Maximum physical timestep [s]");
    form::number(settings, "minimum_dt_s", "Minimum permitted timestep [s]");
    form::number(settings, "output_interval_s", "Recording interval [s]");
    form::number(settings, "step_interval_s", "Integration output interval [s]");
    form::number(settings, "stability_factor", "Explicit thermal stability safety factor [0..1]");
    ImGui::SeparatorText("Battery thermal grid");
    auto& grids = document.draft.at("numerics").at("batteries");
    ImGui::TextWrapped(
        "Grid belongs to each battery, not to its material model. Shell subdivisions must remain aligned.");
    ImGui::SetNextItemWidth(-1);
    std::string id = grids.contains(selected) ? selected
                     : grids.empty()          ? std::string{}
                                              : grids.begin().key();
    const auto state_id = ImGui::GetID("grid_selection");
    int index = ImGui::GetStateStorage()->GetInt(state_id, -1);
    if (index >= 0 && index < int(grids.size())) {
        auto it = grids.begin();
        std::advance(it, index);
        id = it.key();
    }
    if (ImGui::BeginCombo("##grid_battery", id.c_str())) {
        int k = 0;
        for (auto it = grids.begin(); it != grids.end(); ++it, ++k)
            if (ImGui::Selectable(it.key().c_str(), it.key() == id)) {
                id = it.key();
                ImGui::GetStateStorage()->SetInt(state_id, k);
            }
        ImGui::EndCombo();
    }
    if (!id.empty())
        form::vector(grids.at(id), "divisions", "Number of thermal zones along X / Y / Z", 1., true);
    ImGui::SeparatorText("Gas solver accuracy");
    auto& gas = document.draft.at("assembly").at("gas_network");
    ImGui::TextWrapped("Configured solver: %s. Live solver switching is in the viewer controls.",
                       gas.value("mass_backend", "cuda_global").c_str());
    form::number(gas, "coupled_transport_interval_s", "Gas / thermal coupling interval [s]");
    form::integer(gas, "averaged_pressure_iterations", "Pressure solver iteration limit");
    form::integer(gas, "averaged_payload_iterations", "Transport solver iteration limit");
    ImGui::SeparatorText("Experimental measurements");
    if (auto* m = form::record("Measurement", document.draft.at("measurements"))) {
        ImGui::TextWrapped("Quantity: %s [%s]", m->value("quantity", "").c_str(),
                           m->value("unit", "").c_str());
        auto& mapping = m->at("mapping");
        ImGui::TextWrapped("Location certainty: %s", mapping.value("status", "").c_str());
        form::text(mapping, "cell_id", "Observed battery / element");
        form::text(mapping, "notes", "Location notes");
        if (m->contains("data")) {
            auto& data = m->at("data");
            form::text(data, "artifact_id", "Measurement data artifact");
            form::text(data, "time_column", "Time column");
            form::text(data, "value_column", "Measured value column");
            form::number(data, "time_offset_s", "Experiment time offset [s]");
        }
    }
}
void ExperimentEditor::Impl::draw_circuit_properties() {
    auto& assembly = document.draft.at("assembly");
    auto& circuit = assembly.at("electrical_circuit");
    ImGui::SeparatorText("Edit an electrical branch");
    form::text(circuit, "reference_node", "Reference electrical node (ground)");
    if (auto* b = form::record("Branch", circuit.at("branches"))) {
        ImGui::TextWrapped("Branch type: %s", b->value("kind", "").c_str());
        form::text(*b, "from", "Positive / source node");
        form::text(*b, "to", "Negative / destination node");
        form::text(*b, "cell_id", "Battery connected to this branch");
        form::number(*b, "emf_v", "Battery electromotive force [V]");
        form::number(*b, "resistance_ohm", "Branch resistance [ohm]");
    }
    // Controllers are individual cards, never one shared parameter dropdown.
    ImGui::SeparatorText("Power supplies");
    auto& conditions = document.draft.at("conditions");
    if (conditions.contains("source_drives"))
        for (auto& drive : conditions.at("source_drives")) {
            const auto id = drive.value("source_id", std::string("source"));
            ImGui::PushID(id.c_str());
            ImGui::TextWrapped("Power supply: %s", id.c_str());
            form::number(drive, "current_a", "Applied current [A]");
            form::number(drive, "voltage_v", "Applied voltage [V]");
            form::number(drive, "start_s", "Power-on time [s]");
            form::number(drive, "stop_s", "Power-off time [s]");
            if (drive.contains("profile"))
                ImGui::TextWrapped("A measured supply profile is attached and preserved.");
            ImGui::PopID();
        }
    if (!conditions.at("electrical_drives").empty()) {
        ImGui::SeparatorText("Prescribed battery drives");
        if (auto* drive = form::record("Battery drive", conditions.at("electrical_drives"), "cell_id")) {
            form::number(*drive, "current_a", "Prescribed current [A]");
            form::number(*drive, "voltage_v", "Prescribed voltage [V]");
            ImGui::TextWrapped("Imported drive profiles are preserved; they are not replaced by the wiring "
                               "wizard unless requested.");
        }
    }
}
} // namespace battery::ui
