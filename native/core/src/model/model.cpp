#include "battery_native/model/model.hpp"
#include <algorithm>
#include <stdexcept>

namespace battery {
const char* name(ShapeKind value) {
    switch (value) {
    case ShapeKind::Cylinder:
        return "cylinder";
    case ShapeKind::Prismatic:
        return "prismatic";
    case ShapeKind::Pouch:
        return "pouch";
    }
    return "unknown";
}

const char* name(TransitionKind value) {
    switch (value) {
    case TransitionKind::Constant:
        return "constant";
    case TransitionKind::Smooth:
        return "smooth";
    case TransitionKind::FirstOrder:
        return "first_order";
    case TransitionKind::SecondOrder:
        return "second_order";
    case TransitionKind::StateGeometry:
        return "state_geometry";
    case TransitionKind::DirectedGeometry:
        return "directed_geometry";
    }
    return "unknown";
}

const char* name(EdgePhase value) {
    switch (value) {
    case EdgePhase::Low:
        return "low";
    case EdgePhase::High:
        return "high";
    case EdgePhase::Destroyed:
        return "destroyed";
    }
    return "unknown";
}

const char* name(CellMode value) {
    switch (value) {
    case CellMode::Healthy:
        return "healthy";
    case CellMode::Runaway:
        return "runaway";
    case CellMode::Spent:
        return "spent";
    case CellMode::PartiallySpent:
        return "partially-spent";
    case CellMode::Quenched:
        return "quenched";
    }
    return "unknown";
}

void refine_environment_mesh(Scenario& scenario, int scale) {
    if (scale < 1 || scale > 8)
        throw std::runtime_error("mesh-scale must be 1..8");
    const bool passive = std::any_of(scenario.cells.begin(), scenario.cells.end(),
                                     [](const auto& c) { return !c.is_battery; });
    for (auto& c : scenario.cells)
        if (!passive || !c.is_battery) {
            if (!c.initial_temperature_field_k.empty() && scale != 1)
                throw std::runtime_error("Cannot refine an initial field without an interpolation policy");
            for (auto& n : c.thermal_divisions)
                n *= scale;
        }
}

} // namespace battery
