#pragma once

#include <nlohmann/json_fwd.hpp>

namespace battery {
// Run/service adapter, NOT a physics API. Pure, deterministic and no file/network access.
// Produces the existing scenario document; never mutates catalog models or fits parameters.
nlohmann::json compile_experiment(const nlohmann::json& experiment);
// Explicit, lossless v1 -> v2 import. v2 documents are checked and returned unchanged.
// Never invents an electrical topology for legacy per-cell drives.
nlohmann::json upgrade_experiment(const nlohmann::json& experiment);
// UI-only sidecar validation; never forwarded to the simulation or checkpoint identity.
void validate_editor_document(const nlohmann::json& editor, const nlohmann::json& experiment);
} // namespace battery
