#include "battery_native/electrical/circuit_solve.hpp"
#include "battery_native/simulation/simulation.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace battery;
void require(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}
Scenario fixture() {
    Scenario s;
    s.settings.duration_s = .1;
    s.settings.max_dt_s = .01;
    s.settings.output_interval_s = .01;
    for (int i = 0; i < 2; ++i) {
        CellConfig c;
        c.id = "B" + std::to_string(i);
        c.is_battery = true;
        c.dimensions_m = {.02, .02, .02};
        c.position_m.x = .04 * i;
        c.thermal_divisions = {1, 1, 1};
        c.heat_capacity_j_per_k = 100;
        c.initial_temperature_k = 300;
        c.electrical.enabled = true;
        c.electrical.resistance_ohm = i + 1;
        c.electrical.current_a = 0;
        c.electrical.max_resistance_ohm = 10;
        s.cells.push_back(c);
    }
    s.electrical_circuit = {2,
                            {{CircuitKind::Battery, 1, 0, 0, 3},
                             {CircuitKind::Battery, 1, 0, 1, 3},
                             {CircuitKind::CurrentSource, 0, 1, -1, 6}}};
    return s;
}
int main() {
    try {
        auto s = fixture();
        Simulation sim(s);
        sim.run_to_end();
        auto e = sim.electrical_states();
        require(std::abs(e[0].current_a - 4) < 1e-9 && std::abs(e[1].current_a - 2) < 1e-9,
                "Parallel unequal-R split wrong");
        require(std::abs(sim.joule_energy_j() - 2.4) < 1e-8, "Parallel I2R work wrong");
        require(std::abs(sim.energy_balance_residual_j()) < 1e-8, "Circuit heat ledger failed");
        s.electrical_circuit.branches[2] = {CircuitKind::VoltageSource, 1, 0, -1, 7};
        Simulation voltage(s);
        voltage.run_to_end();
        require(std::abs(voltage.electrical_states()[0].current_a - 4) < 1e-9, "Voltage drive wrong");
        s.electrical_circuit = {3,
                                {{CircuitKind::Battery, 1, 2, 0, 3},
                                 {CircuitKind::Battery, 2, 0, 1, 3},
                                 {CircuitKind::VoltageSource, 1, 0, -1, 12}}};
        Simulation series(s);
        series.run_to_end();
        require(std::abs(series.joule_energy_j() - 1.2) < 1e-8, "Series work wrong");
        require(std::abs(series.electrical_states()[0].current_a - 2) < 1e-9 &&
                    std::abs(series.electrical_states()[1].current_a - 2) < 1e-9,
                "Series currents differ");
        s = fixture();
        s.settings.duration_s = 2;
        s.cells[0].electrical.reference_temperature_k = 300;
        s.cells[0].electrical.temperature_log_slope_per_k = .1;
        Simulation dynamic(s);
        dynamic.run_to_end();
        e = dynamic.electrical_states();
        require(e[0].current_a < 3.99 && e[1].current_a > 2.01,
                "Circuit did not redistribute current with temperature");
        require(std::abs(e[0].current_a + e[1].current_a - 6) < 1e-9, "Dynamic KCL failed");
        auto fine = s;
        fine.settings.max_dt_s = .005;
        Simulation refined(fine);
        refined.run_to_end();
        require(std::abs(refined.cells()[0].temperature_k - dynamic.cells()[0].temperature_k) < 1e-3,
                "Frozen-R network fails step refinement");
        Simulation prefix(s);
        while (prefix.time_s() < .5 - 1e-9)
            prefix.step(.5 - prefix.time_s());
        auto bytes = prefix.save_checkpoint("network-test");
        Simulation restored(s);
        restored.restore_checkpoint(bytes, "network-test");
        prefix.run_to_end();
        restored.run_to_end();
        require(prefix.save_checkpoint("network-test") == restored.save_checkpoint("network-test"),
                "Circuit restart changed state");
        s = fixture();
        s.electrical_circuit.branches[2] = {CircuitKind::VoltageSource, 1, 0, -1, 2};
        bool rejected = false;
        try {
            Simulation reverse(s);
        } catch (const std::exception&) {
            rejected = true;
        }
        require(rejected, "Unsupported discharge silently injected energy");
        s = fixture();
        s.electrical_circuit.nodes = 3;
        rejected = false;
        try {
            Simulation floating(s);
        } catch (const std::exception&) {
            rejected = true;
        }
        require(rejected, "Floating node silently grounded");
        std::cout << "DC circuit: series/parallel, voltage/current, thermal redistribution, strict heat "
                     "balance, restart and rejection passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
