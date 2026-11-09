#pragma once
#include <nlohmann/json.hpp>

namespace battery {
// One shared pseudo-species combustion law. Legacy per-volume laws remain
// untouched until the user explicitly adopts a common law.
nlohmann::json normalized_gas_combustion(const nlohmann::json&, bool require_complete = false);
nlohmann::json gas_combustion_variants(const nlohmann::json& network);
} // namespace battery
