#include "battery_ui/options.hpp"
#include <cmath>
#include <stdexcept>

namespace battery::viewer {
Options parse_options(int argc, char** argv) {
    Options result;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        const auto next = [&]() {
            if (i + 1 >= argc)
                throw std::invalid_argument("Missing argument for " + argument);
            return std::string(argv[++i]);
        };
        if (argument == "--replay")
            result.replay_path = next();
        else if (argument == "--at")
            result.start_time_s = std::stod(next());
        else if (argument == "--screenshot")
            result.screenshot_path = next();
        else if (argument == "--frames")
            result.frame_limit = std::stoi(next());
        else if (argument == "--play")
            result.paused = false;
        else if (argument == "--hidden")
            result.hidden = true;
        else if (argument == "--editor")
            result.editor = true;
        else if (argument == "--mesh-scale")
            result.run.environment_mesh_scale = std::stoi(next());
        else if (argument == "--isothermal")
            result.run.isothermal_batteries = true;
        else if (argument.starts_with("--"))
            throw std::invalid_argument("Unknown option: " + argument);
        else
            result.scenario_path = argument;
    }
    validate_run_configuration(result.run);
    if (!std::isfinite(result.start_time_s))
        throw std::invalid_argument("Replay time must be finite");
    if (result.frame_limit < 0)
        throw std::invalid_argument("Frame limit must be nonnegative");
    if (!result.replay_path.empty() &&
        (result.run.environment_mesh_scale != 1 || result.run.isothermal_batteries))
        throw std::invalid_argument("Cannot change physics/topology of a recorded trajectory");
    return result;
}
} // namespace battery::viewer
