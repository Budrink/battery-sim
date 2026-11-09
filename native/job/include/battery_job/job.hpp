#pragma once

#include "battery_native/run_configuration.hpp"
#include <cstdint>
#include <filesystem>
#include <functional>
#include <nlohmann/json.hpp>
#include <ostream>
#include <string>

namespace battery::job {
// This protocol belongs to the process adapter, never to the numerical solver.
inline constexpr int schema_version = 1;
struct Request {
    std::string job_id, attempt_id;
    std::filesystem::path scenario_path, restart_from;
    RunConfiguration run;
    double observation_interval_s = 1.;
    double progress_interval_wall_s = 1.;
    double checkpoint_interval_wall_s = 0.; // zero: final/cancel/limit only
    double wall_time_limit_s = 0.;
    std::uint64_t step_limit = 0; // additional accepted substeps in this attempt
};

Request parse_request(const nlohmann::json&, const std::filesystem::path& base);
nlohmann::json effective_request(const Request&);
std::string build_id();
std::string physics_contract(const Request&, const nlohmann::json& scenario);

// output_directory must not exist. stdout is NDJSON; diagnostics go to stderr
// in the executable. Stop is checked BETWEEN outer intervals, never mid-step.
// Exit: 0 completed (even if expectations differ), 1 failed, 2 invalid_request,
// 3 output_exists, 124 limit_reached, 130 cancelled.
int execute(const std::filesystem::path& request_path, const std::filesystem::path& output_directory,
            std::ostream& events, const std::function<bool()>& stop = {});
} // namespace battery::job
