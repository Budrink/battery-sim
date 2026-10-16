#include "battery_native/playback.hpp"

#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace {
using namespace battery;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

bool near(double a, double b) { return std::abs(a - b) < 1.0e-8; }

Scenario fixture() {
    Scenario scenario;
    scenario.settings = {10.0, 0.25, 0.25, 0.2};
    for (const char* id : {"A", "B"}) {
        CellConfig cell;
        cell.id = id;
        cell.dimensions_m = {1, 1, 1};
        cell.thermal_divisions = {1, 1, 1};
        cell.position_m = {static_cast<double>(scenario.cells.size()) * 1.1, 0, 0};
        cell.heat_capacity_j_per_k = 1000;
        cell.runaway_energy_j = 1000;
        cell.runaway_duration_s = 1;
        scenario.cells.push_back(cell);
    }
    scenario.primary_failures = {{0, "A", 0.0}, {1, "B", 3.0}};
    return scenario;
}

PauseReason until_pause(Simulation& simulation, AutoPauseMode mode, int batch = 32) {
    for (int frame = 0; frame < 100000; ++frame) {
        const auto reason = advance_playback(simulation, batch, mode);
        if (reason != PauseReason::None) return reason;
    }
    throw std::runtime_error("Playback did not terminate");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 3, "Expected module and contained scenario paths");
        Simulation sequential(fixture());
        require(advance_playback(sequential, 32, AutoPauseMode::EachDestruction) ==
                    PauseReason::Destruction && near(sequential.time_s(), 1),
                "Each-destruction mode must break a batch at the first spent cell");
        require(sequential.cells()[1].mode == CellMode::Healthy,
                "Do not step past the pause into a later scheduled failure");
        require(advance_playback(sequential, 32, AutoPauseMode::EachDestruction) ==
                    PauseReason::AllCellsDestroyed && near(sequential.time_s(), 4),
                "Resume must consume only NEW events and stop at the last cell");
        const auto stopped_steps = sequential.step_count();
        require(advance_playback(sequential, 1, AutoPauseMode::AllCellsDestroyed) ==
                    PauseReason::None && sequential.step_count() == stopped_steps + 1,
                "Resume must allow cooldown without repeatedly pausing on all-spent state");
        require(until_pause(sequential, AutoPauseMode::AllCellsDestroyed) ==
                    PauseReason::TimeLimit && sequential.finished(),
                "Cooldown must end at the configured time limit");

        Simulation continuous(fixture());
        require(advance_playback(continuous, 32, AutoPauseMode::AllCellsDestroyed) ==
                    PauseReason::AllCellsDestroyed && near(continuous.time_s(), 4),
                "Continuous mode must ignore a quiet gap with surviving cells");
        for (int speed : {1, 2, 16, 32}) {
            Simulation simulation(fixture());
            require(until_pause(simulation, AutoPauseMode::AllCellsDestroyed, speed) ==
                        PauseReason::AllCellsDestroyed && near(simulation.time_s(), 4),
                    "Frame speed must not shift the stopping time");
        }

        Simulation switched(fixture());
        require(until_pause(switched, AutoPauseMode::EachDestruction) == PauseReason::Destruction,
                "First destruction expected");
        require(until_pause(switched, AutoPauseMode::AllCellsDestroyed) ==
                    PauseReason::AllCellsDestroyed && near(switched.time_s(), 4),
                "Switching mode while paused must not replay the previous event");
        switched.reset();
        require(until_pause(switched, AutoPauseMode::AllCellsDestroyed) ==
                    PauseReason::AllCellsDestroyed && near(switched.time_s(), 4),
                "Reset must clear previous stopping state");

        Simulation stepped(fixture());
        for (int i = 0; i < 4; ++i)
            advance_playback(stepped, 1, AutoPauseMode::EachDestruction);
        require(advance_playback(stepped, 1, AutoPauseMode::EachDestruction) == PauseReason::None,
                "Manual stepping across destruction must not leave a stale event");

        auto simultaneous_case = fixture();
        simultaneous_case.primary_failures[1].time_s = 0;
        Simulation simultaneous(simultaneous_case);
        require(until_pause(simultaneous, AutoPauseMode::EachDestruction) ==
                    PauseReason::AllCellsDestroyed && near(simultaneous.time_s(), 1),
                "Simultaneous destruction must produce a single pause");

        auto edge_case = fixture();
        edge_case.primary_failures.clear();
        InterfaceConfig edge;
        edge.id = "A-B";
        edge.cell_b = 1;
        edge.port_a = "a-face";
        edge.port_b = "b-face";
        edge_case.ports = {{"a-face", "A", 0, "contact", "+x", 1, {0.5,0,0}, {1,0,0}},
                           {"b-face", "B", 1, "contact", "-x", 1, {0.6,0,0}, {-1,0,0}}};
        edge.law.damage_onset_k = 298.15;
        edge.law.damage_rate_at_onset_per_s = 2;
        edge_case.interfaces.push_back(edge);
        Simulation destroyed_edge(edge_case);
        require(until_pause(destroyed_edge, AutoPauseMode::EachDestruction) ==
                    PauseReason::Destruction && near(destroyed_edge.time_s(), 0.5),
                "Per-destruction mode must also observe interface destruction");
        Simulation ignored_edge(edge_case);
        require(until_pause(ignored_edge, AutoPauseMode::AllCellsDestroyed) ==
                    PauseReason::TimeLimit && !all_cells_destroyed(ignored_edge),
                "Continuous mode must ignore edge destruction and retain surviving cells");

        auto phase_case = edge_case;
        phase_case.interfaces[0].law.damage_rate_at_onset_per_s = 0;
        phase_case.interfaces[0].law.kind = TransitionKind::FirstOrder;
        phase_case.interfaces[0].law.transition_temperature_k = 290;
        Simulation phase_only(phase_case);
        require(until_pause(phase_only, AutoPauseMode::EachDestruction) == PauseReason::TimeLimit,
                "Reversible interface phase transitions must not cause destruction pauses");

        // A controlled fully-triggered spatial fixture tests last-destruction
        // timing. The uncalibrated module is NOT required to destroy all cells.
        auto all_triggered = load_scenario(argv[1]);
        all_triggered.primary_failures.clear();
        all_triggered.settings.duration_s = 100;
        for (std::size_t i=0;i<all_triggered.cells.size();++i)
            all_triggered.primary_failures.push_back({i,all_triggered.cells[i].id,5.0});
        Simulation module(all_triggered);
        require(until_pause(module, AutoPauseMode::AllCellsDestroyed) ==
                    PauseReason::AllCellsDestroyed && !module.finished(),
                "Fully triggered 52-object fixture must pause on final destruction");
        double final_destruction_s = 0;
        for (const auto& event : module.events())
            if (event.kind == "runaway_complete") final_destruction_s = event.time_s;
        require(near(module.time_s(), final_destruction_s), "No overshoot on last destruction");
        require(std::abs(module.energy_balance_residual_j()) < 1.0e-5,
                "Playback policy must not change the energy balance");
        std::printf("52-cell stop: %.3f s, all spent\n", module.time_s());

        Simulation contained(load_scenario(argv[2]));
        require(until_pause(contained, AutoPauseMode::AllCellsDestroyed) ==
                    PauseReason::TimeLimit && !all_cells_destroyed(contained),
                "Contained case must reach the time limit with survivors");
        std::puts("Playback pause modes, resume, reset, step and frame-speed checks OK");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
