#pragma once
#include <vector>

namespace battery {
enum class CircuitKind { Battery, CurrentSource, VoltageSource };
// Node zero is the electrical reference. Positive current flows from -> to.
// Battery: Vfrom - Vto = emf + I R; positive I is charging.
struct CircuitBranch {
    CircuitKind kind = CircuitKind::Battery;
    int from = 0, to = 0, object = -1;
    double value = 0; // battery emf [V], source current [A] or source voltage [V]
};
struct ElectricalCircuit {
    int nodes = 0;
    std::vector<CircuitBranch> branches;
};
} // namespace battery
