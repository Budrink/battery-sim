#pragma once

#include "battery_native/run_configuration.hpp"
#include <string>

namespace battery::viewer {
struct Options {
    std::string scenario_path = "examples/contracts/module-4x4.json";
    std::string replay_path;
    std::string screenshot_path;
    double start_time_s = 0.;
    int frame_limit = 0;
    bool paused = true;
    bool hidden = false;
    bool editor = false;
    RunConfiguration run;
};

// Parsing and validation do not initialize CUDA, GLFW, or an ImGui context.
Options parse_options(int argc, char** argv);
} // namespace battery::viewer
