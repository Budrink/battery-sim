#include "battery_ui/application.hpp"
#include "battery_ui/options.hpp"
#include "battery_ui/viewer_session.hpp"
#include "battery_ui/experiment_editor.hpp"
#include "platform/desktop_host.hpp"
#include "viewer/viewer_window.hpp"
#include "editor/views/editor_desktop.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
namespace battery::ui {
int run(int argc, char** argv) {
    try {
        const auto options = viewer::parse_options(argc, argv);
        ViewerSession session(options);
        DesktopHost host(options.hidden);
        EditorDesktop editor_desktop(host, options.hidden);
        auto& editor = editor_desktop.editor();
        editor.try_load(session.source_path());
        editor.open = options.editor;
        ViewerWindow view(session, editor, options.hidden);
        using Clock = std::chrono::steady_clock;
        auto tick = Clock::now();
        const auto started = tick;
        auto capture = options.screenshot_path;
        int frames = 0;
        while (!host.should_close()) {
            host.poll_events();
            if (auto snapshot = editor.take_run())
                if (session.start_snapshot(*snapshot)) {
                    view.new_snapshot();
                    editor.open = false;
                    host.focus();
                }
            const auto now = Clock::now();
            const double elapsed = std::chrono::duration<double>(now - tick).count();
            tick = now;
            session.advance(elapsed);
            host.begin_frame();
            view.draw(std::min(.1, elapsed));
            ++frames;
            const bool capture_now =
                !capture.empty() && (!options.frame_limit || frames >= options.frame_limit);
            host.present(capture_now ? capture : std::string{});
            editor_desktop.draw(capture_now && options.editor ? capture + ".editor.ppm" : std::string{});
            if (capture_now)
                capture.clear();
            if (options.frame_limit && frames >= options.frame_limit)
                break;
        }
        std::printf("Viewer: %d frames in %.3f s (includes vsync); mode=%s; t=%.3f\n", frames,
                    std::chrono::duration<double>(Clock::now() - started).count(),
                    session.recorded() ? "recorded" : "live", session.time());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "battery viewer: %s\n", e.what());
        return 1;
    }
}
} // namespace battery::ui
