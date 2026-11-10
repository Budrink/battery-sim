#include "battery_ui/viewer_session.hpp"
#include "battery_native/scenario_document.hpp"
#include <nlohmann/json.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace battery;
using namespace battery::ui;
using J = nlohmann::json;
void require(bool ok, const char* why) {
    if (!ok)
        throw std::runtime_error(why);
}
// Headless test of the UI application service: no GLFW, ImGui, renderer or GL.
int main(int argc, char** argv) {
    try {
        require(argc == 2, "Expected scenario");
        viewer::Options options;
        options.scenario_path = argv[1];
        ViewerSession session(options);
        const auto document = load_scenario_document(argv[1]);
        Simulation reference(scenario_from_json(document));
        session.advance(.1, 0);
        require(session.time() == 0, "Paused session advanced the solver");
        session.playback.paused = false;
        for (int i = 0; i < 5; ++i) {
            reference.step();
            session.advance(.01, 0); // one committed physical step
            require(session.time() == reference.time_s(), "UI batching changed physical time");
            for (std::size_t j = 0; j < reference.cells().size(); ++j)
                require(session.frame().cells[j].temperature_k == reference.cells()[j].temperature_k &&
                            session.frame().cells[j].damage == reference.cells()[j].damage,
                        "Application session changed solver trajectory");
        }
        require(session.timing().steps == 5, "Wrong session step counter");
        const auto before = session.time();
        require(!session.start_snapshot(J::object()) && session.time() == before && session.generation() == 0,
                "Failed snapshot destroyed the running session");
        require(session.playback.paused && !session.error().empty(), "Invalid snapshot not reported");
        session.restart();
        require(session.time() == 0 && session.frame().time_s == 0 && session.timing().steps == 0,
                "Restart leaves stale displayed time or counters");
        require(session.start_snapshot(document) && session.generation() == 1 && !session.playback.paused,
                "Valid snapshot did not replace session");

        const auto directory = std::filesystem::temp_directory_path() /
                               ("battery-viewer-session-" +
                                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(directory);
        struct Cleanup {
            std::filesystem::path path;
            ~Cleanup() {
                std::error_code ec;
                std::filesystem::remove_all(path, ec);
            }
        } cleanup{directory};
        std::ofstream(directory / "scenario.json") << document.dump();
        J manifest{{"format", "BATTVIEW2"},
                   {"scenario_path", (directory / "scenario.json").string()},
                   {"source_label", "session regression"},
                   {"time_s", {0., 1., 2.}},
                   {"objects", J::array()},
                   {"channels", J::array()},
                   {"gas", J::array()},
                   {"channel_fields", 8},
                   {"gas_fields", 6},
                   {"events", J::array()}};
        for (const auto& c : session.scenario().cells)
            manifest["objects"].push_back(c.id);
        for (const auto& c : session.scenario().gas_channels)
            manifest["channels"].push_back(c.id);
        for (const auto& g : session.scenario().gas_compartments)
            manifest["gas"].push_back(g.id);
        manifest["events"].push_back({{"time_s", .75}, {"kind", "shell_rupture"}, {"id", "test-shell"}});
        for (int i : session.batteries())
            manifest["events"].push_back(
                {{"time_s", 1.25}, {"kind", "runaway_complete"}, {"id", session.scenario().cells[i].id}});
        std::ofstream(directory / "manifest.json") << manifest.dump();
        std::ofstream binary(directory / "frames.bin", std::ios::binary);
        for (int t = 0; t < 3; ++t) {
            std::vector<float> values;
            for (const auto& c : session.scenario().cells) {
                (void)c;
                values.insert(values.end(), {300.f + t, 0, 0, 0});
            }
            values.resize(values.size() + 8 * session.scenario().gas_channels.size(), 0);
            for (const auto& g : session.scenario().gas_compartments) {
                (void)g;
                values.insert(values.end(), {300, 0, 0, .001, 100000, 0});
            }
            binary.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
        }
        binary.close();
        options.replay_path = (directory / "manifest.json").string();
        ViewerSession replay(options);
        replay.playback.rate = 100;
        replay.playback.auto_pause = AutoPause::EachRupture;
        replay.playback.paused = false;
        replay.advance(.1);
        require(replay.time() == .75 && replay.playback.paused, "Replay missed exact rupture pause");
        replay.seek(1.8);
        require(replay.frame().time_s == 1.8, "Seek did not update displayed frame");
        replay.restart();
        replay.playback.auto_pause = AutoPause::AllSpent;
        replay.playback.paused = false;
        replay.advance(.1);
        require(replay.time() == 1.25 && replay.playback.paused, "Replay all-spent pause changed");
        require(!replay.start_snapshot(J::object()) && replay.recorded(), "Failed load discarded recording");
        require(replay.start_snapshot(document) && !replay.recorded(),
                "Cannot switch recording to live snapshot");
        std::cout << "Session: identical CUDA steps, pause/reset, atomic snapshot replacement and exact "
                     "replay events pass\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
