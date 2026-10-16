#pragma once

#include "battery_native/simulation.hpp"

#include <algorithm>

namespace battery {

enum class AutoPauseMode { AllCellsDestroyed, EachDestruction };
enum class PauseReason { None, Manual, SingleStep, Destruction, AllCellsDestroyed, TimeLimit };

inline bool all_cells_destroyed(const Simulation& simulation) {
    return simulation.all_batteries_destroyed();
}

inline const char* pause_description(PauseReason reason) {
    switch (reason) {
        case PauseReason::None: return "Running";
        case PauseReason::Manual: return "Paused manually";
        case PauseReason::SingleStep: return "Paused after single step";
        case PauseReason::Destruction: return "Paused: element destroyed";
        case PauseReason::AllCellsDestroyed: return "Paused: all batteries destroyed";
        case PauseReason::TimeLimit: return "Finished: time limit reached";
    }
    return "";
}

// Viewer-only stopping policy: no changes to the solver or headless runs.
// Check after EVERY physical substep, including those inside an outer interval.
inline PauseReason advance_playback(Simulation& simulation, int steps, AutoPauseMode mode) {
    if (simulation.finished()) return PauseReason::TimeLimit;
    for (int step = 0; step < steps; ++step) {
        bool all_destroyed_before = all_cells_destroyed(simulation);
        std::size_t first_new_event = simulation.events().size();
        PauseReason reason=PauseReason::None;
        simulation.advance([&]() {
            // Trigger once, on the transition; Resume can show the cooldown.
            if (!all_destroyed_before && all_cells_destroyed(simulation))
                reason=PauseReason::AllCellsDestroyed;
            else if (simulation.finished()) reason=PauseReason::TimeLimit;
            if (reason==PauseReason::None && mode==AutoPauseMode::EachDestruction) {
                const auto& events=simulation.events();
                for (std::size_t index=first_new_event;index<events.size();++index) {
                    if (events[index].kind=="runaway_complete" ||
                        events[index].kind=="thermal_volume_destroyed" ||
                        events[index].kind=="material_destroyed" ||
                        events[index].kind=="interface_destroyed") {
                        reason=PauseReason::Destruction;
                        break;
                    }
                }
            }
            first_new_event=simulation.events().size();
            all_destroyed_before=all_cells_destroyed(simulation);
            return reason!=PauseReason::None;
        });
        if(reason!=PauseReason::None)return reason;
    }
    return PauseReason::None;
}

}  // namespace battery
