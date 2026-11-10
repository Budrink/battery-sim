#include "editor/editor_internal.hpp"
#include "battery_native/contracts/experiment_contract.hpp"
#include "battery_native/contracts/experiment_editing.hpp"
#include "battery_native/contracts/enclosure_openings.hpp"
#include "battery_native/scenario_document.hpp"
#include <algorithm>
namespace battery::ui {
using J = nlohmann::json;
void ExperimentEditor::Impl::wire() {
    auto next = document.draft;
    auto& a = next["assembly"];
    J circuit{{"mode", "solved_dc"},
              {"reference_node", "ground"},
              {"nodes", {"ground", "supply"}},
              {"branches", J::array()}};
    const auto count = a["batteries"].size();
    if (!count)
        throw std::runtime_error("Add a battery first");
    for (std::size_t i = 0; i < count; ++i) {
        const auto& b = a["batteries"][i];
        const auto bid = b.at("id").get<std::string>();
        const std::string from = parallel || i == 0 ? "supply" : "junction-" + std::to_string(i);
        const std::string to = parallel || i + 1 == count ? "ground" : "junction-" + std::to_string(i + 1);
        if (!parallel && i + 1 < count)
            circuit["nodes"].push_back(to);
        circuit["branches"].push_back({{"id", "branch-" + bid},
                                       {"kind", "battery"},
                                       {"cell_id", bid},
                                       {"from", from},
                                       {"to", to},
                                       {"emf_v", emf}});
        for (auto& model : next["battery_models"])
            if (model.at("id") == b.at("model_id") && model.at("revision") == b.at("model_revision"))
                if (!model["contents"].contains("electrical"))
                    model["contents"]["electrical"] = {{"resistance_ohm", .03}};
    }
    circuit["branches"].push_back({{"id", "source"},
                                   {"kind", voltage_source ? "voltage_source" : "current_source"},
                                   {"from", voltage_source ? "supply" : "ground"},
                                   {"to", voltage_source ? "ground" : "supply"}});
    a["electrical_circuit"] = circuit;
    next["metadata"]["editor_changes"].push_back(
        {{"kind", "rewire_dc"},
         {"emf_v", emf},
         {"assumption", "Fixed declared EMFs; missing electrical laws initialized to synthetic 0.03 ohm"}});
    if (next["provenance"]["status"] != "synthetic")
        next["provenance"]["status"] = "mixed";
    // Explicit replace-all action: the wizard connects every battery in this document.
    next["conditions"]["electrical_drives"] = J::array();
    next["conditions"]["source_drives"] = {
        {{"source_id", "source"}, {voltage_source ? "voltage_v" : "current_a", source_value}}};
    (void)compile_experiment(next);
    document.draft = std::move(next);
    message = "DC network created. Currents are solved from current material resistance; inspect assumptions "
              "before running.";
}
} // namespace battery::ui
