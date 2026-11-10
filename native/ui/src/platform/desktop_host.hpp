#pragma once
#include <string>
struct GLFWwindow;
struct ImGuiContext;
struct ImPlotContext;
namespace battery::ui {
// Owns GLFW/ImGui/ImPlot lifetime. Declared before renderers so their GPU
// resources are released while the graphics context is still alive.
class DesktopHost {
  public:
    explicit DesktopHost(bool hidden, DesktopHost* parent = nullptr);
    ~DesktopHost();
    DesktopHost(const DesktopHost&) = delete;
    DesktopHost& operator=(const DesktopHost&) = delete;
    bool should_close() const;
    void poll_events();
    void begin_frame();
    void present(const std::string& screenshot_path = {});
    void activate();
    void show();
    void hide();
    void focus();
    GLFWwindow* native_window() const {
        return window_;
    }

  private:
    void close();
    void install_input_callbacks();
    static void route_input(GLFWwindow*);
    GLFWwindow* window_ = nullptr;
    ImGuiContext* imgui_ = nullptr;
    ImPlotContext* implot_ = nullptr;
    DesktopHost* parent_ = nullptr;
    bool hidden_ = false;
    bool glfw_ready_ = false, contexts_ready_ = false, glfw_backend_ = false, gl_backend_ = false;
};
} // namespace battery::ui
