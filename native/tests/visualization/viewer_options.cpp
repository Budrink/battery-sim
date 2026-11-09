#include "battery_ui/options.hpp"
#include <iostream>
#include <stdexcept>
#include <vector>

battery::viewer::Options parse(std::vector<std::string> args) {
    std::vector<char*> pointers;
    for (auto& arg : args)
        pointers.push_back(arg.data());
    return battery::viewer::parse_options(static_cast<int>(pointers.size()), pointers.data());
}
int main() {
    try {
        if (parse({"viewer"}).scenario_path != "examples/contracts/module-4x4.json")
            throw std::runtime_error("Default laboratory module lost");
        const auto options = parse({"viewer", "scene.json", "--play", "--hidden", "--frames", "2"});
        if (options.paused || !options.hidden || options.frame_limit != 2 ||
            options.scenario_path != "scene.json")
            throw std::runtime_error("Viewer arguments lost");
        if (!parse({"viewer", "--editor"}).editor)
            throw std::runtime_error("Editor launch flag lost");
        for (const auto& args :
             std::vector<std::vector<std::string>>{{"viewer", "--replay", "record.json", "--isothermal"},
                                                   {"viewer", "--replay", "record.json", "--mesh-scale", "2"},
                                                   {"viewer", "--frames", "-1"},
                                                   {"viewer", "--at", "nan"},
                                                   {"viewer", "--frames"},
                                                   {"viewer", "--unknown"}}) {
            bool rejected = false;
            try {
                parse(args);
            } catch (const std::exception&) {
                rejected = true;
            }
            if (!rejected)
                throw std::runtime_error("Invalid viewer arguments accepted");
        }
        std::cout << "Viewer options: replay cannot change physics; no GPU/window needed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
