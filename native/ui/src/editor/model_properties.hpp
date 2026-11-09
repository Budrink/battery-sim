#pragma once
#include <nlohmann/json.hpp>

namespace battery::ui {
// Edits only explicit contract properties; displaying a model never installs defaults.
void draw_model_properties(nlohmann::json& model);
} // namespace battery::ui
