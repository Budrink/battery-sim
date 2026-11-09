#include "viewport_input.hpp"
#include "toolbar.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

void require(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}
int main() {
    ImGui::CreateContext();
    try {
        auto& io = ImGui::GetIO();
        io.DisplaySize = {1000, 800};
        io.DeltaTime = 1.f / 60;
        io.IniFilename = nullptr;
        io.ConfigInputTrickleEventQueue = false;
        io.ConfigWindowsMoveFromTitleBarOnly = true;
        unsigned char* pixels;
        int width, height;
        io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
        ImVec2 position;
        auto frame = [&] {
            ImGui::NewFrame();
            ImGui::SetNextWindowPos({100, 100}, ImGuiCond_Once);
            ImGui::SetNextWindowSize({500, 500}, ImGuiCond_Once);
            ImGui::Begin("Editor");
            position = ImGui::GetWindowPos();
            ImGui::BeginChild("Preview", {400, 400});
            const auto input = battery::ui::orbit_viewport(ImTextureID(1), {300, 300});
            ImGui::EndChild();
            ImGui::End();
            ImGui::Render();
            return input;
        };
        frame();
        io.AddMousePosEvent(200, 200);
        frame();
        io.AddMouseButtonEvent(0, true);
        frame();
        io.AddMousePosEvent(260, 230);
        auto drag = frame();
        require(drag.dx > 50 && drag.dy > 20, "Scene did not capture orbit drag");
        require(position.x == 100 && position.y == 100, "Orbit moved the editor window");
        io.AddMousePosEvent(700, 650);
        drag = frame();
        require(drag.dx > 0, "Orbit lost capture outside viewport");
        io.AddMouseButtonEvent(0, false);
        require(!frame().clicked, "Orbit release selected an object");
        require(position.x == 100 && position.y == 100, "Orbit release moved window");
        io.AddMousePosEvent(200, 200);
        frame();
        io.AddMouseButtonEvent(0, true);
        frame();
        io.AddMouseButtonEvent(0, false);
        require(frame().clicked, "Stationary click cannot select object");
        io.AddMousePosEvent(200, 110);
        frame();
        io.AddMouseButtonEvent(0, true);
        frame();
        io.AddMousePosEvent(250, 140);
        frame();
        require(position.x > 140 && position.y > 120, "Title-bar dragging no longer works");
        io.AddMouseButtonEvent(0, false);
        // Repeated frames and narrow windows must not hide Refresh off the
        // right edge or let scrolling properties remove the toolbar.
        for (const float toolbar_width : {350.f, 600.f, 900.f}) {
            for (int repeat = 0; repeat < 4; ++repeat) {
                ImGui::NewFrame();
                ImGui::SetNextWindowPos({0, 0});
                ImGui::SetNextWindowSize({toolbar_width, 500});
                ImGui::Begin("Toolbar test");
                ImGui::BeginChild("toolbar", {0, 0}, ImGuiChildFlags_AutoResizeY,
                                  ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
                const float right = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
                for (const char* label :
                     {"Save to path", "Undo", "Redo", "Validate", "Refresh 3D preview", "RUN SNAPSHOT"}) {
                    battery::ui::toolbar_button(label);
                    require(ImGui::GetItemRectMax().x <= right + 1, "Toolbar button clipped horizontally");
                    if (repeat > 1)
                        require(ImGui::IsItemVisible(), "Toolbar button disappeared");
                }
                ImGui::NewLine();
                ImGui::EndChild();
                ImGui::BeginChild("properties", {0, 0});
                for (int row = 0; row < 100; ++row)
                    ImGui::Text("Property %d", row);
                ImGui::SetScrollY(500);
                ImGui::EndChild();
                ImGui::End();
                ImGui::Render();
            }
        }
        ImGui::DestroyContext();
        std::cout << "Viewport: orbit capture, fixed window, click selection and title-bar move passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        ImGui::DestroyContext();
        return 1;
    }
}
