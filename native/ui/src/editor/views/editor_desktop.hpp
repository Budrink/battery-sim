#pragma once
#include "platform/desktop_host.hpp"
#include "battery_ui/experiment_editor.hpp"

namespace battery::ui {
// Native editor window. Owns its GL renderer/context and input independently
// of the simulation window, but shares the application document/session commands.
class EditorDesktop {
  public:
    EditorDesktop(DesktopHost& main, bool hidden);
    ~EditorDesktop();
    ExperimentEditor& editor() {
        return *editor_;
    }
    void draw(const std::string& capture = {});
    void show();
    GLFWwindow* native_window() const {
        return window_.native_window();
    }

  private:
    DesktopHost& main_;
    DesktopHost window_;
    std::unique_ptr<ExperimentEditor> editor_;
    bool visible_ = false;
};
} // namespace battery::ui
