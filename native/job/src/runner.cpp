#include "battery_job/job.hpp"
#include "battery_native/simulation/simulation.hpp"
#include "battery_native/io/checkpoint_io.hpp"
#include "artifacts.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace battery::job {
namespace {
using nlohmann::json;
using Clock = std::chrono::steady_clock;
json cell_snapshot(const Simulation& sim) {
    json cells = json::array();
    for (std::size_t i = 0; i < sim.cells().size(); ++i) {
        const auto& c = sim.cells()[i];
        cells.push_back({{"id", sim.scenario().cells[i].id},
                         {"temperature_k", c.temperature_k},
                         {"damage", c.damage},
                         {"mode", name(c.mode)},
                         {"stored_energy_j", c.stored_energy_j},
                         {"chemical_energy_remaining_j", c.chemical_energy_remaining_j}});
    }
    return {{"schema_version", schema_version}, {"time_s", sim.time_s()}, {"cells", cells}};
}
json expectations(const Simulation& sim) {
    const auto& scenario = sim.scenario();
    json checks = json::object();
    if (scenario.has_expected_propagation)
        checks["propagation_matches"] = sim.propagated() == scenario.expected_propagation;
    if (!scenario.expected_runaway_order.empty()) {
        std::vector<std::string> actual;
        for (auto i : sim.runaway_order())
            actual.push_back(scenario.cells[i].id);
        checks["runaway_order_matches"] = actual == scenario.expected_runaway_order;
    }
    bool passed = true;
    for (const auto& value : checks)
        passed &= value.get<bool>();
    return {{"status", !sim.finished()  ? "not_evaluated"
                       : checks.empty() ? "not_requested"
                       : passed         ? "passed"
                                        : "failed"},
            {"checks", checks}};
}
void save_restart(const std::filesystem::path& dir, const Request& request, const Simulation& sim,
                  const std::string& contract, std::size_t number) {
    const std::string file = "checkpoint-" + std::to_string(number) + ".bin";
    const auto bytes = sim.save_checkpoint(contract);
    atomic_write(dir / file, bytes);
    // Publish pointer AFTER the immutable payload. An interrupted write leaves
    // the previous checkpoint.json usable, rather than a mismatched pair.
    atomic_json(dir / "checkpoint.json", {{"schema_version", schema_version},
                                          {"build_id", build_id()},
                                          {"job_id", request.job_id},
                                          {"attempt_id", request.attempt_id},
                                          {"time_s", sim.time_s()},
                                          {"state_file", file},
                                          {"checksum_fnv1a64", checksum(bytes)}});
}
std::string read_restart(const Request& request, const std::string& contract) {
    const auto metadata = read_json(request.restart_from);
    if (metadata.at("schema_version") != schema_version || metadata.at("build_id") != build_id())
        throw std::invalid_argument("Checkpoint schema/build mismatch");
    const auto file = std::filesystem::path(metadata.at("state_file").get<std::string>());
    if (file.empty() || file != file.filename() || file == "." || file == "..")
        throw std::invalid_argument("Checkpoint state_file must be a basename");
    const auto bytes = read_bytes(request.restart_from.parent_path() / file, 4ULL * 1024 * 1024 * 1024);
    if (metadata.at("checksum_fnv1a64") != checksum(bytes))
        throw std::invalid_argument("Checkpoint checksum mismatch");
    checkpoint::Reader reader(bytes);
    if (reader.text() != "battery-simulation-v1" || reader.text() != contract)
        throw std::invalid_argument("Checkpoint physics/topology contract mismatch");
    return bytes;
}
} // namespace

int execute(const std::filesystem::path& request_path, const std::filesystem::path& output_directory,
            std::ostream& events, const std::function<bool()>& stop) {
    const auto started = Clock::now();
    auto elapsed = [&] { return std::chrono::duration<double>(Clock::now() - started).count(); };
    json identity = {{"schema_version", schema_version}, {"job_id", nullptr}, {"attempt_id", nullptr}};
    std::uint64_t sequence = 0;
    auto emit = [&](json message) {
        message.update(identity);
        message["sequence"] = sequence++;
        message["wall_s"] = elapsed();
        events << message.dump() << '\n' << std::flush;
        if (!events)
            throw std::runtime_error("Cannot write worker event stream");
    };
    bool owns_output = false;
    json last_confirmed_time = nullptr;
    std::string stage = "output";
    try {
        // Never reuse a directory, even for the same job/attempt. The queue owns
        // deduplication; the worker cannot silently overwrite a previous result.
        if (!std::filesystem::exists(output_directory) && !std::filesystem::is_symlink(output_directory))
            owns_output = std::filesystem::create_directory(output_directory);
        if (!owns_output) {
            emit({{"type", "terminal"},
                  {"status", "rejected"},
                  {"reason", "output_exists"},
                  {"exit_code", 3}});
            return 3;
        }
        stage = "validation";
        const auto input = read_json(request_path, 1024 * 1024);
        const auto request = parse_request(input, std::filesystem::absolute(request_path).parent_path());
        identity["job_id"] = request.job_id;
        identity["attempt_id"] = request.attempt_id;
        const auto scenario_json = read_json(request.scenario_path);
        if (!scenario_json.is_object())
            throw std::invalid_argument("Scenario must be an object");
        const auto contract = physics_contract(request, scenario_json);
        atomic_json(output_directory / "request.json", input);
        atomic_json(output_directory / "effective-request.json", effective_request(request));
        atomic_json(output_directory / "scenario.json", scenario_json);
        auto scenario = load_run((output_directory / "scenario.json").string(), request.run);
        if (!std::isfinite(scenario.settings.duration_s) || scenario.settings.duration_s <= 0. ||
            !std::isfinite(scenario.settings.max_dt_s) || scenario.settings.max_dt_s <= 0. ||
            !std::isfinite(scenario.settings.output_interval_s) || scenario.settings.output_interval_s <= 0.)
            throw std::invalid_argument("Invalid scenario time settings");
        atomic_json(output_directory / "provenance.json",
                    {{"schema_version", schema_version},
                     {"build_id", build_id()},
                     {"scenario_checksum_fnv1a64", checksum(scenario_json.dump())},
                     {"physics_contract", json::parse(contract)},
                     {"checkpoint_portability", "compatible build only; not a public interchange format"}});
        stage = "restart";
        const auto restart_bytes =
            request.restart_from.empty() ? std::string{} : read_restart(request, contract);
        stage = "initialization";
        Simulation sim(std::move(scenario));
        stage = "restart";
        if (!request.restart_from.empty())
            sim.restore_checkpoint(restart_bytes, contract);
        stage = "execution";
        last_confirmed_time = sim.time_s();
        const double started_at_s = sim.time_s();
        const auto first_step = sim.step_count();
        std::size_t event_cursor = request.restart_from.empty() ? 0 : sim.events().size();
        std::size_t checkpoint_number = 0;
        double last_observation = -1., last_progress = -request.progress_interval_wall_s;
        double last_checkpoint_wall = elapsed();
        std::ofstream observations, physical_events;
        observations.exceptions(std::ios::failbit | std::ios::badbit);
        physical_events.exceptions(std::ios::failbit | std::ios::badbit);
        observations.open(output_directory / "observations.ndjson");
        physical_events.open(output_directory / "events.ndjson");
        auto observe = [&](bool force) {
            if (sim.time_s() == last_observation)
                return;
            if (force || sim.time_s() >= last_observation + request.observation_interval_s) {
                observations << cell_snapshot(sim).dump() << '\n' << std::flush;
                last_observation = sim.time_s();
            }
        };
        auto progress = [&](bool force) {
            if (force || elapsed() - last_progress >= request.progress_interval_wall_s) {
                emit({{"type", "progress"},
                      {"status", "running"},
                      {"time_s", sim.time_s()},
                      {"target_time_s", sim.scenario().settings.duration_s},
                      {"steps", sim.step_count()}});
                last_progress = elapsed();
            }
        };
        emit({{"type", "started"},
              {"status", "running"},
              {"build_id", build_id()},
              {"device", sim.device_name()},
              {"started_at_s", started_at_s}});
        observe(true);
        progress(true);
        std::string status = "completed", reason = "duration_reached";
        int exit_code = 0;
        while (!sim.finished()) {
            if (stop && stop()) {
                status = "cancelled";
                reason = "cancel_requested";
                exit_code = 130;
                break;
            }
            if ((request.wall_time_limit_s > 0. && elapsed() >= request.wall_time_limit_s) ||
                (request.step_limit > 0 && sim.step_count() - first_step >= request.step_limit)) {
                status = "limit_reached";
                reason = "resource_limit";
                exit_code = 124;
                break;
            }
            // Exactly the same stepping as run_to_end(). Observation does not
            // cap dt, change output_interval_s, or interrupt an outer interval.
            sim.advance();
            last_confirmed_time = sim.time_s();
            while (event_cursor < sim.events().size()) {
                const auto& e = sim.events()[event_cursor++];
                json event = {{"schema_version", schema_version},
                              {"type", "physical_event"},
                              {"time_s", e.time_s},
                              {"kind", e.kind},
                              {"entity_id", e.entity_id}};
                physical_events << event.dump() << '\n' << std::flush;
                emit(event);
            }
            observe(false);
            progress(false);
            if (request.checkpoint_interval_wall_s > 0. &&
                elapsed() - last_checkpoint_wall >= request.checkpoint_interval_wall_s) {
                save_restart(output_directory, request, sim, contract, checkpoint_number++);
                last_checkpoint_wall = elapsed();
                emit({{"type", "checkpoint"}, {"time_s", sim.time_s()}, {"manifest", "checkpoint.json"}});
            }
        }
        observe(true);
        const double residual = sim.energy_balance_residual_j();
        const double chemical_residual = sim.chemical_balance_residual_j();
        if (!std::isfinite(residual) || !std::isfinite(chemical_residual))
            throw std::runtime_error("Nonfinite final balance");
        save_restart(output_directory, request, sim, contract, checkpoint_number++);
        observations.close();
        physical_events.close();
        json order = json::array();
        for (auto i : sim.runaway_order())
            order.push_back(sim.scenario().cells[i].id);
        json result = identity;
        result.update({{"status", status},
                       {"reason", reason},
                       {"exit_code", exit_code},
                       {"build_id", build_id()},
                       {"started_at_s", started_at_s},
                       {"time_s", sim.time_s()},
                       {"target_time_s", sim.scenario().settings.duration_s},
                       {"steps", sim.step_count()},
                       {"attempt_steps", sim.step_count() - first_step},
                       {"wall_s", elapsed()},
                       {"runaway_order", order},
                       {"max_temperature_k", sim.max_temperature_k()},
                       {"validation",
                        {{"scenario_expectations", expectations(sim)},
                         {"energy_balance_residual_j", residual},
                         {"chemical_balance_residual_j", chemical_residual}}},
                       {"artifacts",
                        {{"observations", "observations.ndjson"},
                         {"events", "events.ndjson"},
                         {"checkpoint", "checkpoint.json"},
                         {"provenance", "provenance.json"}}}});
        // Last publication: consumers must not infer success from file presence
        // or a peak in a partial trajectory. Read terminal status explicitly.
        atomic_json(output_directory / "result.json", result);
        // Once committed, result.json is authoritative even if the supervisor
        // disconnects before receiving the terminal notification.
        try {
            emit({{"type", "terminal"},
                  {"status", status},
                  {"reason", reason},
                  {"time_s", sim.time_s()},
                  {"result", "result.json"},
                  {"exit_code", exit_code}});
        } catch (...) {
        }
        return exit_code;
    } catch (const std::exception& e) {
        const bool io_failure = dynamic_cast<const std::ios_base::failure*>(&e) ||
                                dynamic_cast<const std::filesystem::filesystem_error*>(&e);
        const bool invalid = !io_failure && (stage == "validation" || stage == "restart");
        const int code = invalid ? 2 : 1;
        json error = identity;
        error.update({{"status", invalid ? "invalid_request" : "failed"},
                      {"stage", stage},
                      {"error", e.what()},
                      {"last_confirmed_time_s", last_confirmed_time},
                      {"exit_code", code},
                      {"build_id", build_id()}});
        // A failed solver may not be at a valid commit point. Do not manufacture
        // a checkpoint here; only previously published checkpoints are usable.
        if (owns_output) {
            try {
                atomic_json(output_directory / "result.json", error);
            } catch (...) {
            }
        }
        try {
            error["type"] = "terminal";
            emit(error);
        } catch (...) {
        }
        return code;
    }
}
} // namespace battery::job
