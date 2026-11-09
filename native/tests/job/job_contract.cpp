#include "battery_job/job.hpp"
#include "battery_native/simulation/simulation.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

namespace {
using nlohmann::json;
namespace fs = std::filesystem;
void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}
json read(const fs::path& path) {
    std::ifstream in(path);
    json value;
    in >> value;
    return value;
}
void write(const fs::path& path, const json& value) {
    std::ofstream out(path);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out << value.dump(2);
}
std::string bytes(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
json last_sample(const fs::path& path) {
    std::ifstream in(path);
    json result;
    std::string line;
    while (std::getline(in, line))
        result = json::parse(line);
    return result;
}
void compare(const battery::Simulation& sim, const json& sample) {
    require(sim.time_s() == sample.at("time_s"), "Different final time");
    require(sim.cells().size() == sample.at("cells").size(), "Different topology");
    for (std::size_t i = 0; i < sim.cells().size(); ++i) {
        const auto& c = sim.cells()[i];
        const auto& actual = sample.at("cells").at(i);
        require(c.temperature_k == actual.at("temperature_k"), "Different temperature");
        require(c.damage == actual.at("damage"), "Different damage");
        require(c.stored_energy_j == actual.at("stored_energy_j"), "Different stored energy");
        require(c.chemical_energy_remaining_j == actual.at("chemical_energy_remaining_j"),
                "Different chemical inventory");
        require(std::string(battery::name(c.mode)) == actual.at("mode"), "Different cell mode");
    }
}
} // namespace
int main(int argc, char** argv) {
    try {
        require(argc == 2, "Expected synthetic scenario path");
        char pattern[] = "/tmp/battery-job-test-XXXXXX";
        const char* directory = mkdtemp(pattern);
        require(directory, "Cannot create test directory");
        const fs::path root(directory);
        auto scenario = read(argv[1]);
        scenario["simulation"]["duration_s"] = .03;
        scenario["simulation"]["step_interval_s"] = .01;
        write(root / "scenario.json", scenario);
        json request = {{"schema_version", 1},
                        {"job_id", "test"},
                        {"attempt_id", "base"},
                        {"scenario_path", "scenario.json"},
                        {"observations", {{"interval_s", .005}}}};
        write(root / "job.json", request);
        auto parsed = battery::job::parse_request(request, root);
        require(parsed.run.environment_mesh_scale == 1 && !parsed.run.isothermal_batteries,
                "Defaults change physics");
        for (auto mutation :
             {json{{"schema_version", 2}}, json{{"schema_version", 1.0}}, json{{"unknown", true}},
              json{{"job_id", "../escape"}}, json{{"observations", {{"interval_s", 0}}}},
              json{{"limits", {{"max_steps", -1}}}}, json{{"limits", {{"max_steps", 1.2}}}},
              json{{"run_configuration", {{"environment_mesh_scale", 1.5}}}}}) {
            auto invalid = request;
            invalid.update(mutation);
            bool rejected = false;
            try {
                battery::job::parse_request(invalid, root);
            } catch (const std::exception&) {
                rejected = true;
            }
            require(rejected, "Invalid envelope accepted");
        }
        battery::Simulation baseline(battery::load_run((root / "scenario.json").string()));
        baseline.run_to_end();
        auto run = [&](const std::string& attempt, json value, const std::function<bool()>& stop = {}) {
            value["attempt_id"] = attempt;
            write(root / "job.json", value);
            std::ostringstream events;
            const int code = battery::job::execute(root / "job.json", root / attempt, events, stop);
            std::istringstream input(events.str());
            std::string line;
            std::uint64_t sequence = 0;
            while (std::getline(input, line))
                require(json::parse(line).at("sequence") == sequence++, "Invalid event sequence");
            require(sequence > 0, "No machine-readable status");
            return code;
        };
        require(run("full", request) == 0, "Full worker run failed");
        const auto full = read(root / "full/result.json");
        require(full.at("status") == "completed", "Full run status");
        require(full.at("steps") == baseline.step_count(), "Worker changed step boundaries");
        require(full.at("validation").at("energy_balance_residual_j") == baseline.energy_balance_residual_j(),
                "Worker changed energy ledger");
        compare(baseline, last_sample(root / "full/observations.ndjson"));
        const auto original_result = bytes(root / "full/result.json");
        require(run("full", request) == 3, "Duplicate attempt was not rejected");
        require(bytes(root / "full/result.json") == original_result, "Duplicate overwrote result");

        auto sparse = request;
        sparse["observations"]["interval_s"] = 100.;
        require(run("sparse", sparse) == 0, "Sparse observation run failed");
        compare(baseline, last_sample(root / "sparse/observations.ndjson"));
        require(read(root / "sparse/result.json").at("steps") == baseline.step_count(),
                "Observation frequency changed dt");

        int boundaries = 0;
        require(run("cancelled", request, [&] { return ++boundaries == 3; }) == 130, "Cancellation failed");
        require(read(root / "cancelled/result.json").at("time_s") == .02,
                "Cancellation not at outer boundary");
        auto resumed = request;
        resumed["restart_from"] = "cancelled/checkpoint.json";
        require(run("resumed", resumed) == 0, "Resume failed");
        const auto resumed_result = read(root / "resumed/result.json");
        require(resumed_result.at("steps") == full.at("steps"), "Resume changed step count");
        compare(baseline, last_sample(root / "resumed/observations.ndjson"));
        // Published checkpoint must itself restore the same committed state.
        battery::Simulation restored(battery::load_run((root / "scenario.json").string()));
        const auto checkpoint = read(root / "resumed/checkpoint.json");
        restored.restore_checkpoint(bytes(root / "resumed" / checkpoint.at("state_file").get<std::string>()),
                                    battery::job::physics_contract(parsed, scenario));
        compare(restored, last_sample(root / "full/observations.ndjson"));

        auto limited = request;
        limited["limits"] = {{"max_steps", 1}};
        require(run("limited", limited) == 124, "Step budget ignored");
        limited["limits"] = {{"wall_time_s", 1e-12}};
        require(run("wall-limited", limited) == 124, "Wall budget ignored");
        auto wrong_physics = resumed;
        wrong_physics["run_configuration"] = {{"isothermal_batteries", true}};
        require(run("wrong-physics", wrong_physics) == 2, "Incompatible physics resumed");
        auto wrong_build = read(root / "cancelled/checkpoint.json");
        wrong_build["build_id"] = "different-build";
        write(root / "cancelled/wrong-build.json", wrong_build);
        resumed["restart_from"] = "cancelled/wrong-build.json";
        require(run("wrong-build", resumed) == 2, "Incompatible build resumed");
        auto wrong_checksum = read(root / "cancelled/checkpoint.json");
        wrong_checksum["checksum_fnv1a64"] = "0000000000000000";
        write(root / "cancelled/wrong-checksum.json", wrong_checksum);
        resumed["restart_from"] = "cancelled/wrong-checksum.json";
        require(run("wrong-checksum", resumed) == 2, "Corrupt checkpoint resumed");
        auto malformed = request;
        malformed["schema_version"] = 99;
        require(run("invalid", malformed) == 2, "Invalid request status");
        require(read(root / "invalid/result.json").at("status") == "invalid_request",
                "Invalid result status");

        scenario["observations"] = {{"propagated", !baseline.propagated()}};
        write(root / "mismatch.json", scenario);
        auto mismatch = request;
        mismatch["scenario_path"] = "mismatch.json";
        require(run("mismatch", mismatch) == 0, "Expectation mismatch treated as solver failure");
        require(
            read(root / "mismatch/result.json").at("validation").at("scenario_expectations").at("status") ==
                "failed",
            "Expectation mismatch hidden");
        std::cout << "Worker contract: exact direct/resume trajectories, cancellation, limits, provenance,\n"
                     "duplicate protection, incompatible checkpoints and separate expectations PASS\n"
                  << "Artifacts retained at " << root << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
