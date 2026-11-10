#pragma once

#include <nlohmann/json_fwd.hpp>

namespace battery {
// Run/service adapter, NOT a physics API. Pure, deterministic and no file/network access.
// Produces the existing scenario document; never mutates catalog models or fits parameters.
nlohmann::json compile_experiment(const nlohmann::json& experiment);
// Explicit, lossless v1 -> v2 import. v2 documents are checked and returned unchanged.
// Never invents an electrical topology for legacy per-cell drives.
nlohmann::json upgrade_experiment(const nlohmann::json& experiment);
// Import an explicit native R1/R2/R3 scenario into editable model/instance data.
// Checks a compile round trip; rejects any unrepresentable field instead of dropping it.
nlohmann::json experiment_from_native(const nlohmann::json& scenario);
// UI-only sidecar validation; never forwarded to the simulation or checkpoint identity.
void validate_editor_document(const nlohmann::json& editor, const nlohmann::json& experiment);
} // namespace battery
