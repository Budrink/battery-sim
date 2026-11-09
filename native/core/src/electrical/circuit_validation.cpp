#include "battery_native/electrical/circuit_validation.hpp"
#include "battery_native/electrical/circuit_solve.hpp"
#include <set>
#include <stdexcept>

namespace battery {
void validate_circuit(const Scenario& s) {
    const auto& c = s.electrical_circuit;
    if (c.branches.empty() && c.nodes == 0)
        return;
    auto fail = [] {
        throw std::invalid_argument(
            "Invalid DC circuit: topology, electrical law, independent drive or unsupported discharge");
    };
    if (c.nodes < 2 || c.nodes > 128 || c.branches.empty() || c.branches.size() > 512)
        fail();
    std::set<int> objects;
    std::vector<double> r(c.branches.size(), 0), current(r.size()), voltage(r.size());
    for (std::size_t k = 0; k < c.branches.size(); ++k) {
        const auto& b = c.branches[k];
        if (b.from < 0 || b.to < 0 || b.from >= c.nodes || b.to >= c.nodes || b.from == b.to ||
            !std::isfinite(b.value))
            fail();
        if (b.kind == CircuitKind::Battery) {
            if (b.object < 0 || b.object >= int(s.cells.size()) || !objects.insert(b.object).second ||
                b.value < 0)
                fail();
            const auto& cell = s.cells[b.object];
            if (!cell.is_battery || !cell.electrical.enabled || cell.electrical.voltage_limited_current ||
                cell.electrical.current_a != 0 || !cell.current_profile_a.empty() ||
                !cell.voltage_profile_v.empty() || b.value < cell.electrical.reversible_voltage_floor_v)
                fail();
            r[k] = cell.electrical.resistance_ohm;
        } else if (b.kind != CircuitKind::CurrentSource && b.kind != CircuitKind::VoltageSource)
            fail();
    }
    if (objects.empty())
        fail();
    const int n = circuit_unknowns(c.nodes, c.branches.data(), int(c.branches.size()));
    if (n < 1 || n > 128)
        fail();
    std::vector<double> matrix(n * n), rhs(n);
    if (!solve_circuit(c.nodes, c.branches.data(), int(r.size()), r.data(), matrix.data(), rhs.data(),
                       current.data(), voltage.data()))
        throw std::invalid_argument("DC circuit is singular or fails KCL/work balance: check floating nodes "
                                    "and conflicting ideal sources");
    for (std::size_t k = 0; k < c.branches.size(); ++k)
        if (c.branches[k].kind == CircuitKind::Battery && (current[k] < -1e-10 || voltage[k] < -1e-10))
            throw std::invalid_argument("DC circuit would discharge " + s.cells[c.branches[k].object].id +
                                        "; discharge is not implemented. Check source polarity, total series "
                                        "voltage and battery EMFs");
}
} // namespace battery
