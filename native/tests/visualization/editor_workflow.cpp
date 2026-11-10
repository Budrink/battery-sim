#include "editor/views/editor_desktop.hpp"
#include "editor/editor_internal.hpp"
#include "battery_native/contracts/enclosure_openings.hpp"
#include "imgui.h"
#include "imgui_internal.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include <stdexcept>

namespace battery::ui {
struct ExperimentEditorTestAccess {
    static auto& get(ExperimentEditor& e) {
        return *e.impl_;
    }
};
} // namespace battery::ui
namespace {
void require(bool ok, const std::string& message) {
    if (!ok)
        throw std::runtime_error(message);
}
} // namespace
int main(int argc, char** argv) {
    try {
        require(argc == 2, "Expected starter contract");
        using namespace battery::ui;
        DesktopHost main(true);
        auto* main_context = ImGui::GetCurrentContext();
        {
            EditorDesktop desktop(main, true);
            auto& editor = desktop.editor();
            auto& p = ExperimentEditorTestAccess::get(editor);
            editor.try_load(argv[1]);
            require(p.import_error.empty(), p.import_error);
            const auto original = p.document.draft;
            editor.show();
            auto frame = [&] {
                main.poll_events();
                main.begin_frame();
                ImGui::Begin("Simulation test");
                ImGui::TextUnformatted("Simulation");
                ImGui::End();
                main.present();
                desktop.draw();
                require(ImGui::GetCurrentContext() == main_context &&
                            glfwGetCurrentContext() == main.native_window(),
                        "Editor did not restore the main window context");
                require(glGetError() == GL_NO_ERROR, "GL error across native windows");
            };
            frame();
            frame();
            require(desktop.native_window() != main.native_window(), "Editor is not a native window");
            // GLFW dispatches all windows in one poll. Simulate events delivered
            // while a different ImGui context is current, as happens in real use.
            const auto main_queue = main_context->InputEventsQueue.Size;
            auto character = glfwSetCharCallback(desktop.native_window(), nullptr);
            glfwSetCharCallback(desktop.native_window(), character);
            require(character != nullptr, "Editor input routing not installed");
            character(desktop.native_window(), 'x');
            auto* editor_context = ImGui::GetCurrentContext();
            require(editor_context != main_context, "Editor keyboard event reached simulation context");
            require(main_context->InputEventsQueue.Size == main_queue, "Editor modified simulation input");
            main.activate();
            auto mouse = glfwSetMouseButtonCallback(desktop.native_window(), nullptr);
            glfwSetMouseButtonCallback(desktop.native_window(), mouse);
            mouse(desktop.native_window(), GLFW_MOUSE_BUTTON_LEFT, GLFW_RELEASE, 0);
            require(ImGui::GetCurrentContext() == editor_context, "Mouse event routed to wrong context");
            main.activate();

            // Every section must fit a narrow content pane and preserve data on view.
            for (const auto size : {ImVec2{1440, 1000}, ImVec2{1000, 800}, ImVec2{760, 640}}) {
                glfwSetWindowSize(desktop.native_window(), int(size.x), int(size.y));
                for (int section = 0; section < int(std::size(editor_sections)); ++section) {
                    p.section = EditorSection(section);
                    frame();
                    frame();
                    for (auto* window : editor_context->Windows) {
                        const std::string name = window->Name;
                        if (!window->Active || name.find("section content") == std::string::npos)
                            continue;
                        require(window->DC.CursorMaxPos.x <= window->WorkRect.Max.x + 2.f,
                                std::string("Clipped section: ") + editor_sections[section].title +
                                    " width=" + std::to_string(size.x));
                    }
                    require(p.document.draft == original,
                            std::string("Viewing changed document: ") + editor_sections[section].title);
                }
            }
            // A shell field is committed even after leaving its section; no Apply.
            p.section = EditorSection::Shell;
            frame();
            p.face_material["thickness_m"] = p.face_material.at("thickness_m").get<double>() * 1.1;
            p.shell_dirty = true;
            p.section = EditorSection::Run;
            frame();
            require(!p.shell_dirty && p.panel_errors.empty() && p.document.draft != original,
                    "Shell input was not automatically committed across section change");
            const auto valid = p.document.draft;
            p.section = EditorSection::Shell;
            frame();
            p.face_material["thickness_m"] = -1.;
            p.shell_dirty = true;
            p.section = EditorSection::Run;
            frame();
            require(p.shell_dirty && !p.panel_errors.empty() && p.document.draft == valid,
                    "Invalid shell input poisoned the document or silently disappeared");
            auto& duration = p.document.draft["numerics"]["simulation"]["duration_s"];
            duration = duration.get<double>() + 10.;
            frame();
            require(p.document.draft != valid, "Rejected shell input blocked unrelated editing");
            require(!editor.take_run(), "Editing implicitly launched a simulation");
            // Native close hides the editor but retains its unsaved draft.
            const auto unsaved = p.document.draft;
            glfwSetWindowShouldClose(desktop.native_window(), GLFW_TRUE);
            frame();
            require(!editor.open, "Editor native close did not hide it");
            require(!main.should_close(), "Closing editor closed simulation");
            editor.show();
            frame();
            require(editor.open && p.document.draft == unsaved, "Reopening discarded unsaved edits");
        }
        main.activate();
        require(glGetString(GL_VERSION) != nullptr, "Editor destruction destroyed main GL context");
        std::cout << "Independent windows/input; all section widths; no-op preservation; auto-commit; "
                     "invalid input isolation; close/reopen OK\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
