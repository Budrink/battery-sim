#include "battery_native/contracts/experiment_editing.hpp"
#include "battery_native/contracts/experiment_contract.hpp"
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace battery {
nlohmann::json set_pad_conductivity_preset(const nlohmann::json& source, const std::string& id,
                                           const std::string& kind) {
    if (kind != "insulation" && kind != "thermal_conductor")
        throw std::runtime_error("Unknown pad conductivity preset: " + kind);
    auto next = source;
    for (auto& object : next.at("assembly").at("passive_cells")) {
        if (object.at("id") != id)
            continue;
        // Starter insulation and the existing UPV TIM working prior, respectively.
        // These are editable defaults, not a calibrated product specification.
        const double k = kind == "insulation" ? .06 : 2.;
        object.at("thermal")["conductivity_w_per_m_k"] = {k, k, k};
        object["material_id"] = "editor-pad-" + id;
        next["metadata"]["visual_objects"][id] = kind;
        next["metadata"]["editor_changes"].push_back(
            {{"kind", "pad_conductivity_preset"},
             {"object_id", id},
             {"preset", kind},
             {"conductivity_w_per_m_k", k},
             {"assumption",
              "Editable conductivity default only; heat capacity, damage and contacts retained."}});
        if (next["provenance"]["status"] != "synthetic")
            next["provenance"]["status"] = "mixed";
        (void)compile_experiment(next);
        return next;
    }
    throw std::runtime_error("Passive pad not found: " + id);
}
} // namespace battery
