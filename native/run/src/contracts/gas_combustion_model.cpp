#include "battery_native/contracts/gas_combustion_model.hpp"
#include <cmath>
#include <stdexcept>
#include <array>

namespace battery {
using J = nlohmann::json;
J normalized_gas_combustion(const J& input, bool require_complete) {
    if (!input.is_object())
        throw std::invalid_argument("R3: expected a combustion model");
    const J defaults = {{"reaction_rate_per_s", 0.},
                        {"reaction_reference_k", 600.},
                        {"reaction_activation_j_per_mol", 0.},
                        {"oxidizer_heat_j_per_kg", 13.1e6},
                        {"ignition_temperature_k", 0.}};
    if (require_complete) {
        for (const auto& [key, value] : input.items())
            if (!defaults.contains(key))
                throw std::invalid_argument("R3: unknown parameter " + key);
        for (const char* key : {"reaction_rate_per_s", "reaction_reference_k",
                                "reaction_activation_j_per_mol", "oxidizer_heat_j_per_kg"})
            if (!input.contains(key))
                throw std::invalid_argument(std::string("R3: missing ") + key);
    }
    J result = defaults;
    for (const auto& [key, value] : defaults.items()) {
        if (input.contains(key))
            result[key] = input.at(key);
        if (!result.at(key).is_number())
            throw std::invalid_argument("R3: expected a number for " + key);
        const double v = result.at(key);
        if (!std::isfinite(v) || v < 0 ||
            ((key == "reaction_reference_k" || key == "oxidizer_heat_j_per_kg") && v <= 0))
            throw std::invalid_argument("R3: invalid " + key);
    }
    return result;
}
J gas_combustion_variants(const J& network) {
    if (network.contains("combustion"))
        return J::array({normalized_gas_combustion(network.at("combustion"), true)});
    J variants = J::array();
    for (const auto& compartment : network.at("compartments")) {
        const auto law = normalized_gas_combustion(compartment);
        bool found = false;
        for (const auto& previous : variants)
            found |= previous == law;
        if (!found)
            variants.push_back(law);
    }
    if (variants.empty())
        variants.push_back(normalized_gas_combustion(J::object()));
    return variants;
}
} // namespace battery
