#pragma once
#include "battery_ui/options.hpp"
#include "battery_native/data/visual_frame.hpp"
#include <memory>

namespace battery::ui {
enum class AutoPause { Never, EachRupture, EachSpentCell, AllSpent };
struct PlaybackControls {
    bool paused = true;
    float rate = 5;
    AutoPause auto_pause = AutoPause::Never;
};
struct SessionTiming {
    double solver_ms = 0, simulated_s = 0, solver_s = 0, wall_s = 0;
    std::size_t steps = 0;
    bool running = false;
};

// Application service (roughly a C# application-session class). Owns the live
// solver OR recording, never a window. Views issue commands and read snapshots;
// they cannot call Simulation::step or mutate the computational scenario.
class ViewerSession {
  public:
    explicit ViewerSession(const viewer::Options&);
    void advance(double elapsed_wall_s, double wall_budget_ms = 8);
    bool start_snapshot(const nlohmann::json&); // failed candidate leaves old session intact
    void restart();
    void seek(double seconds); // recordings only
    void next_rupture();
    void set_kawasaki(bool enabled);
    bool can_switch_gas() const;
    bool kawasaki() const;
    std::vector<ElectricalState> electrical_states() const;

    PlaybackControls playback;
    const Scenario& scenario() const {
        return scenario_;
    }
    const ThermalMesh& mesh() const {
        return mesh_;
    }
    const VisualGeometry& geometry() const {
        return geometry_;
    }
    const VisualFrame& frame() const {
        return frame_;
    }
    const std::vector<int>& batteries() const {
        return batteries_;
    }
    const std::vector<EventRecord>& events() const;
    const SessionTiming& timing() const {
        return timing_;
    }
    const std::string& error() const {
        return error_;
    }
    const std::string& source_path() const {
        return source_path_;
    }
    std::string source_label() const;
    bool recorded() const {
        return bool(replay_);
    }
    double time() const {
        return time_;
    }
    double duration() const {
        return duration_;
    }
    std::size_t generation() const {
        return generation_;
    }

  private:
    void refresh_frame();
    void advance_live(double elapsed_wall_s, double wall_budget_ms);
    void advance_replay(double elapsed_wall_s);
    RunConfiguration run_;
    Scenario scenario_;
    ThermalMesh mesh_;
    VisualGeometry geometry_;
    VisualFrame frame_;
    std::unique_ptr<Simulation> simulation_;
    std::unique_ptr<VisualReplay> replay_;
    std::vector<int> batteries_;
    SessionTiming timing_;
    std::string source_path_, error_;
    double time_ = 0, duration_ = 0, last_spent_time_ = -1;
    bool all_spent_paused_ = false;
    std::size_t generation_ = 0;
};
} // namespace battery::ui
