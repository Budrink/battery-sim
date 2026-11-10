#pragma once
#include "editor/state/model_edits.hpp"
namespace battery::ui {
// Shared by the editor and UI regression tests. Viewing never changes network.
void draw_gas_combustion_model(const nlohmann::json& network, ModelEdits& edits);
} // namespace battery::ui
