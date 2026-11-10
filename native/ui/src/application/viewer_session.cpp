#include "battery_ui/viewer_session.hpp"
#include "battery_native/scenario_document.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <set>

namespace battery::ui {
namespace {
using Clock = std::chrono::steady_clock;
std::vector<int> battery_indices(const Scenario& scenario) {
    std::vector<int> result;
    for (std::size_t i = 0; i < scenario.cells.size(); ++i)
        if (scenario.cells[i].is_battery)
            result.push_back(int(i));
    if (result.empty())
        throw std::runtime_error("Scene has no batteries");
    return result;
}
bool stops_at(AutoPause mode, const EventRecord& event) {
    return (mode == AutoPause::EachRupture && event.kind == "shell_rupture") ||
           (mode == AutoPause::EachSpentCell && event.kind == "runaway_complete");
}
} // namespace
ViewerSession::ViewerSession(const viewer::Options& options)
    : run_(options.run), source_path_(options.scenario_path) {
    playback.paused = options.paused;
    if (!options.replay_path.empty()) {
        replay_ = std::make_unique<VisualReplay>(options.replay_path);
        source_path_ = replay_->scenario_path();
    }
    const auto document = load_scenario_document(source_path_);
    scenario_ = configure_run(scenario_from_json(document), run_);
    batteries_ = battery_indices(scenario_);
    geometry_ = visual_geometry_from_json(document, scenario_);
    if (replay_)
        mesh_ = build_thermal_mesh(scenario_);
    else {
        simulation_ = std::make_unique<Simulation>(scenario_);
        mesh_ = simulation_->thermal_mesh();
    }
    duration_ = replay_ ? replay_->times().back() : scenario_.settings.duration_s;
    time_ = replay_ ? std::clamp(options.start_time_s, replay_->times().front(), duration_)
                    : simulation_->time_s();
    if (replay_) {
        std::set<std::string> completed;
        for (const auto& event : replay_->events())
            if (event.kind == "runaway_complete") {
                completed.insert(event.entity_id);
                last_spent_time_ = std::max(last_spent_time_, event.time_s);
            }
        if (completed.size() != batteries_.size())
            last_spent_time_ = -1;
    }
    refresh_frame();
}
void ViewerSession::refresh_frame() {
    frame_ = replay_ ? replay_->sample(time_, scenario_, mesh_) : capture_visual_frame(*simulation_);
}
bool ViewerSession::start_snapshot(const nlohmann::json& document) {
    try {
        auto next = configure_run(scenario_from_json(document), run_);
        auto indices = battery_indices(next);
        auto geometry = visual_geometry_from_json(document, next);
        auto simulation = std::make_unique<Simulation>(next);
        auto frame = capture_visual_frame(*simulation);
        auto mesh = simulation->thermal_mesh();
        scenario_ = std::move(next);
        geometry_ = std::move(geometry);
        mesh_ = std::move(mesh);
        simulation_ = std::move(simulation);
        frame_ = std::move(frame);
        batteries_ = std::move(indices);
        replay_.reset();
        time_ = 0;
        duration_ = scenario_.settings.duration_s;
        last_spent_time_ = -1;
        all_spent_paused_ = false;
        timing_ = {};
        error_.clear();
        playback.paused = false;
        ++generation_;
        return true;
    } catch (const std::exception& e) {
        error_ = e.what();
        playback.paused = true;
        return false;
    }
}
void ViewerSession::advance(double elapsed_wall_s, double wall_budget_ms) {
    if (playback.paused) {
        timing_.running = false;
        return;
    }
    if (replay_)
        advance_replay(elapsed_wall_s);
    else
        advance_live(elapsed_wall_s, wall_budget_ms);
    if (frame_.time_s != time_)
        refresh_frame();
}
void ViewerSession::advance_replay(double elapsed_wall_s) {
    const double before = time_;
    time_ = std::min(duration_, time_ + std::min(.1, elapsed_wall_s) * playback.rate);
    for (const auto& event : replay_->events())
        if (stops_at(playback.auto_pause, event) && event.time_s > before + 1e-8 && event.time_s <= time_) {
            time_ = event.time_s;
            playback.paused = true;
            break;
        }
    if (playback.auto_pause == AutoPause::AllSpent && last_spent_time_ > before + 1e-8 &&
        last_spent_time_ <= time_) {
        time_ = last_spent_time_;
        playback.paused = true;
    }
    if (time_ >= duration_)
        playback.paused = true;
}
void ViewerSession::advance_live(double elapsed_wall_s, double wall_budget_ms) {
    const auto begin = Clock::now();
    const auto steps = simulation_->step_count();
    const auto event_count = simulation_->events().size();
    const double before = time_;
    try {
        // Budget the UI batch, never change or interpolate physical steps.
        do {
            if (!simulation_->step()) {
                playback.paused = true;
                break;
            }
            if (playback.auto_pause == AutoPause::EachRupture ||
                playback.auto_pause == AutoPause::EachSpentCell)
                for (std::size_t k = event_count; k < simulation_->events().size(); ++k)
                    if (stops_at(playback.auto_pause, simulation_->events()[k]))
                        playback.paused = true;
            if (playback.auto_pause == AutoPause::AllSpent && !all_spent_paused_ &&
                simulation_->all_batteries_destroyed()) {
                playback.paused = true;
                all_spent_paused_ = true;
            }
        } while (!playback.paused &&
                 std::chrono::duration<double, std::milli>(Clock::now() - begin).count() < wall_budget_ms);
    } catch (const std::exception& e) {
        error_ = e.what();
        playback.paused = true;
    }
    time_ = simulation_->time_s();
    timing_.solver_ms = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
    timing_.simulated_s += time_ - before;
    timing_.solver_s += timing_.solver_ms / 1000;
    timing_.wall_s += timing_.running ? elapsed_wall_s : timing_.solver_ms / 1000;
    timing_.steps += simulation_->step_count() - steps;
    timing_.running = !playback.paused;
}
void ViewerSession::restart() {
    if (replay_)
        time_ = replay_->times().front();
    else {
        simulation_->reset();
        time_ = simulation_->time_s();
    }
    timing_ = {};
    playback.paused = true;
    all_spent_paused_ = false;
    error_.clear();
    refresh_frame();
}
void ViewerSession::seek(double seconds) {
    if (!replay_)
        return;
    time_ = std::clamp(seconds, replay_->times().front(), duration_);
    playback.paused = true;
    refresh_frame();
}
void ViewerSession::next_rupture() {
    if (!replay_)
        return;
    for (const auto& event : replay_->events())
        if (event.kind == "shell_rupture" && event.time_s > time_ + .01) {
            seek(event.time_s);
            break;
        }
}
void ViewerSession::set_kawasaki(bool enabled) {
    if (!can_switch_gas())
        return;
    simulation_->set_colored_kawasaki(enabled);
    timing_ = {};
}
bool ViewerSession::can_switch_gas() const {
    return simulation_ && simulation_->can_switch_gas_transport();
}
bool ViewerSession::kawasaki() const {
    return simulation_ && simulation_->colored_kawasaki();
}
std::vector<ElectricalState> ViewerSession::electrical_states() const {
    return simulation_ ? simulation_->electrical_states() : std::vector<ElectricalState>{};
}
const std::vector<EventRecord>& ViewerSession::events() const {
    return replay_ ? replay_->events() : simulation_->events();
}
std::string ViewerSession::source_label() const {
    return replay_ ? replay_->source_label() : scenario_.name;
}
} // namespace battery::ui
