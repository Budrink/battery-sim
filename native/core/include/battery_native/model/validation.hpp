#pragma once
#include "battery_native/model/model.hpp"

namespace battery {
// Shared host-only preflight: material inventories and compatible mechanisms.
// No allocation of meshes, no CUDA and no evolution of physical state.
// Called by both document adapters and direct-Scenario mesh construction.
void validate_material_models(const Scenario& scenario);
} // namespace battery
