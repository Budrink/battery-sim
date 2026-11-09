#include "battery_job/job.hpp"
#include "battery_job_build.hpp"
#include <cmath>
#include <regex>
#include <stdexcept>

namespace battery::job {
namespace {
using nlohmann::json;
void fields(const json& value, std::initializer_list<const char*> allowed) {
    if (!value.is_object())
        throw std::invalid_argument("Expected a JSON object");
    for (const auto& [key, ignored] : value.items()) {
        bool known = false;
        for (const auto* candidate : allowed)
            known |= key == candidate;
        if (!known)
            throw std::invalid_argument("Unknown job field: " + key);
    }
}
double number(const json& value, const char* key, double fallback, bool allow_zero) {
    const double result = value.value(key, fallback);
    if (!std::isfinite(result) || result < 0. || (!allow_zero && result == 0.))
        throw std::invalid_argument(std::string("Invalid ") + key);
    return result;
}
std::filesystem::path path(const json& value, const char* key, const std::filesystem::path& base) {
    const auto text = value.at(key).get<std::string>();
    if (text.empty() || text.find('\0') != std::string::npos)
        throw std::invalid_argument(std::string("Invalid path: ") + key);
    return std::filesystem::absolute(base / text).lexically_normal();
}
} // namespace

Request parse_request(const json& value, const std::filesystem::path& base) {
    fields(value,
           {"schema_version", "job_id", "attempt_id", "scenario_path", "run_configuration", "observations",
            "progress_interval_wall_s", "checkpoint_interval_wall_s", "limits", "restart_from"});
    if (!value.at("schema_version").is_number_integer() || value.at("schema_version") != schema_version)
        throw std::invalid_argument("Unsupported job schema_version");
    Request r;
    r.job_id = value.at("job_id").get<std::string>();
    r.attempt_id = value.at("attempt_id").get<std::string>();
    const std::regex id("[A-Za-z0-9][A-Za-z0-9_.-]{0,127}");
    if (!std::regex_match(r.job_id, id) || !std::regex_match(r.attempt_id, id))
        throw std::invalid_argument("Invalid job_id or attempt_id");
    r.scenario_path = path(value, "scenario_path", base);
    if (value.contains("restart_from"))
        r.restart_from = path(value, "restart_from", base);
    const auto run = value.value("run_configuration", json::object());
    fields(run, {"environment_mesh_scale", "isothermal_batteries"});
    if (run.contains("environment_mesh_scale") &&
        (!run.at("environment_mesh_scale").is_number_integer() || run.at("environment_mesh_scale") < 1 ||
         run.at("environment_mesh_scale") > 8))
        throw std::invalid_argument("environment_mesh_scale must be an integer in 1..8");
    r.run.environment_mesh_scale = run.value("environment_mesh_scale", 1);
    r.run.isothermal_batteries = run.value("isothermal_batteries", false);
    validate_run_configuration(r.run);
    const auto obs = value.value("observations", json::object());
    fields(obs, {"interval_s"});
    r.observation_interval_s = number(obs, "interval_s", 1., false);
    r.progress_interval_wall_s = number(value, "progress_interval_wall_s", 1., false);
    r.checkpoint_interval_wall_s = number(value, "checkpoint_interval_wall_s", 0., true);
    const auto limits = value.value("limits", json::object());
    fields(limits, {"wall_time_s", "max_steps"});
    r.wall_time_limit_s = number(limits, "wall_time_s", 0., true);
    if (limits.contains("max_steps")) {
        if (!limits.at("max_steps").is_number_integer() || limits.at("max_steps") < 0)
            throw std::invalid_argument("max_steps must be a nonnegative integer");
        r.step_limit = limits.at("max_steps").get<std::uint64_t>();
    }
    return r;
}

json effective_request(const Request& r) {
    json result = {{"schema_version", schema_version},
                   {"job_id", r.job_id},
                   {"attempt_id", r.attempt_id},
                   {"scenario_path", r.scenario_path.string()},
                   {"run_configuration",
                    {{"environment_mesh_scale", r.run.environment_mesh_scale},
                     {"isothermal_batteries", r.run.isothermal_batteries}}},
                   {"observations", {{"interval_s", r.observation_interval_s}}},
                   {"progress_interval_wall_s", r.progress_interval_wall_s},
                   {"checkpoint_interval_wall_s", r.checkpoint_interval_wall_s},
                   {"limits", {{"wall_time_s", r.wall_time_limit_s}, {"max_steps", r.step_limit}}}};
    if (!r.restart_from.empty())
        result["restart_from"] = r.restart_from.string();
    return result;
}
std::string build_id() {
    return BATTERY_JOB_BUILD_ID;
}
std::string physics_contract(const Request& r, const json& scenario) {
    // Exact canonical content is checked by Simulation::restore_checkpoint.
    // IDs, paths, observations and operational limits do not change physics.
    return json({{"format", "battery-job-physics-v1"},
                 {"build_id", build_id()},
                 {"scenario", scenario},
                 {"run_configuration", effective_request(r).at("run_configuration")}})
        .dump();
}
} // namespace battery::job
