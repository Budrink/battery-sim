#include "platform/desktop_host.hpp"
#include <GLFW/glfw3.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>
namespace battery::ui {
namespace {
void style() {
    ImGui::StyleColorsDark();
    auto& s = ImGui::GetStyle();
    s.WindowPadding = {16, 14};
    s.ItemSpacing = {8, 9};
    s.FramePadding = {10, 7};
    s.WindowRounding = 0;
    s.FrameRounding = 5;
    s.GrabRounding = 4;
    s.WindowBorderSize = 0;
    s.Colors[ImGuiCol_WindowBg] = {.035f, .046f, .060f, 1};
    s.Colors[ImGuiCol_ChildBg] = {.025f, .033f, .045f, 1};
    s.Colors[ImGuiCol_FrameBg] = {.08f, .105f, .135f, 1};
    s.Colors[ImGuiCol_Button] = {.10f, .16f, .21f, 1};
    s.Colors[ImGuiCol_ButtonHovered] = {.13f, .29f, .36f, 1};
    s.Colors[ImGuiCol_ButtonActive] = {.12f, .42f, .46f, 1};
    s.Colors[ImGuiCol_CheckMark] = {.30f, .88f, .80f, 1};
    s.Colors[ImGuiCol_SliderGrab] = {.30f, .88f, .80f, 1};
    s.Colors[ImGuiCol_Text] = {.88f, .92f, .95f, 1};
    s.Colors[ImGuiCol_TextDisabled] = {.46f, .57f, .66f, 1};
}

void screenshot(GLFWwindow* window, const std::string& path) {
    int w, h;
    glfwGetFramebufferSize(window, &w, &h);
    std::vector<unsigned char> pixels(w * h * 3);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadBuffer(GL_BACK);
    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    std::ofstream out(path, std::ios::binary);
    out << "P6\n" << w << " " << h << "\n255\n";
    for (int y = h - 1; y >= 0; --y)
        out.write(reinterpret_cast<const char*>(pixels.data() + y * w * 3), w * 3);
    if (!out)
        throw std::runtime_error("Cannot write screenshot");
}

} // namespace
DesktopHost::DesktopHost(bool hidden, DesktopHost* parent) : parent_(parent), hidden_(hidden) {
    try {
        glfwSetErrorCallback(
            [](int code, const char* msg) { std::fprintf(stderr, "GLFW %d: %s\n", code, msg); });
        if (!parent_ && !glfwInit())
            throw std::runtime_error("GLFW initialization failed");
        glfw_ready_ = !parent_;
        glfwDefaultWindowHints();
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        if (hidden || parent_)
            glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        window_ = glfwCreateWindow(
            parent_ ? 1440 : 1600, 1000,
            parent_ ? "BatterySim | Experiment editor" : "BatterySim | Virtual laboratory", nullptr, nullptr);
        if (!window_)
            throw std::runtime_error("Cannot create OpenGL window");
        glfwMakeContextCurrent(window_);
        glfwSetWindowSizeLimits(window_, 720, 540, GLFW_DONT_CARE, GLFW_DONT_CARE);
        glfwSwapInterval(parent_ ? 0 : 1); // only one vsync wait per application frame
        IMGUI_CHECKVERSION();
        imgui_ = ImGui::CreateContext();
        ImGui::SetCurrentContext(imgui_);
        ImGui::GetIO().ConfigWindowsMoveFromTitleBarOnly = true;
        implot_ = ImPlot::CreateContext();
        ImPlot::SetCurrentContext(implot_);
        contexts_ready_ = true;
        style();
        ImGui::GetIO().IniFilename = nullptr;
        const char* font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf";
        if (std::filesystem::exists(font))
            ImGui::GetIO().Fonts->AddFontFromFileTTF(font, 16);
        // GLFW polls all windows together. Route each event to its owning ImGui
        // context; backend auto-callbacks assume one current context and cannot do this.
        glfw_backend_ = ImGui_ImplGlfw_InitForOpenGL(window_, false);
        if (!glfw_backend_)
            throw std::runtime_error("Cannot initialize ImGui GLFW backend");
        gl_backend_ = ImGui_ImplOpenGL3_Init("#version 330");
        if (!gl_backend_)
            throw std::runtime_error("Cannot initialize ImGui OpenGL backend");
        install_input_callbacks();

    } catch (...) {
        close();
        throw;
    }
}
DesktopHost::~DesktopHost() {
    close();
}
void DesktopHost::close() {
    activate();
    if (gl_backend_)
        ImGui_ImplOpenGL3_Shutdown();
    if (glfw_backend_)
        ImGui_ImplGlfw_Shutdown();
    if (contexts_ready_) {
        ImPlot::DestroyContext();
        ImGui::DestroyContext();
    }
    if (window_)
        glfwDestroyWindow(window_);
    if (glfw_ready_)
        glfwTerminate();
    if (parent_)
        parent_->activate();
}
bool DesktopHost::should_close() const {
    return glfwWindowShouldClose(window_);
}
void DesktopHost::poll_events() {
    glfwPollEvents();
    activate();
}
void DesktopHost::begin_frame() {
    activate();
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}
void DesktopHost::present(const std::string& capture) {
    activate();
    ImGui::Render();
    int width, height;
    glfwGetFramebufferSize(window_, &width, &height);
    glViewport(0, 0, width, height);
    glClearColor(.02f, .025f, .035f, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    if (!capture.empty())
        screenshot(window_, capture);
    glfwSwapBuffers(window_);
}
void DesktopHost::activate() {
    if (window_)
        glfwMakeContextCurrent(window_);
    ImGui::SetCurrentContext(imgui_);
    ImPlot::SetCurrentContext(implot_);
}
void DesktopHost::show() {
    glfwSetWindowShouldClose(window_, GLFW_FALSE);
    if (!hidden_)
        glfwShowWindow(window_);
}
void DesktopHost::hide() {
    glfwHideWindow(window_);
}
void DesktopHost::focus() {
    if (!hidden_)
        glfwFocusWindow(window_);
}
void DesktopHost::route_input(GLFWwindow* window) {
    const auto* host = static_cast<DesktopHost*>(glfwGetWindowUserPointer(window));
    ImGui::SetCurrentContext(host->imgui_);
}
void DesktopHost::install_input_callbacks() {
    glfwSetWindowUserPointer(window_, this);
    glfwSetWindowFocusCallback(window_, [](GLFWwindow* w, int v) {
        route_input(w);
        ImGui_ImplGlfw_WindowFocusCallback(w, v);
    });
    glfwSetCursorEnterCallback(window_, [](GLFWwindow* w, int v) {
        route_input(w);
        ImGui_ImplGlfw_CursorEnterCallback(w, v);
    });
    glfwSetCursorPosCallback(window_, [](GLFWwindow* w, double x, double y) {
        route_input(w);
        ImGui_ImplGlfw_CursorPosCallback(w, x, y);
    });
    glfwSetMouseButtonCallback(window_, [](GLFWwindow* w, int b, int a, int m) {
        route_input(w);
        ImGui_ImplGlfw_MouseButtonCallback(w, b, a, m);
    });
    glfwSetScrollCallback(window_, [](GLFWwindow* w, double x, double y) {
        route_input(w);
        ImGui_ImplGlfw_ScrollCallback(w, x, y);
    });
    glfwSetKeyCallback(window_, [](GLFWwindow* w, int k, int s, int a, int m) {
        route_input(w);
        ImGui_ImplGlfw_KeyCallback(w, k, s, a, m);
    });
    glfwSetCharCallback(window_, [](GLFWwindow* w, unsigned c) {
        route_input(w);
        ImGui_ImplGlfw_CharCallback(w, c);
    });
}
} // namespace battery::ui
