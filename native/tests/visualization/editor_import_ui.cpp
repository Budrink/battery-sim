#include "editor/editor_internal.hpp"
#include "imgui.h"
#include "imgui_internal.h"
#include <GLFW/glfw3.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <unistd.h>
namespace battery::ui {
struct ExperimentEditorTestAccess {
    static auto& get(ExperimentEditor& e) {
        return *e.impl_;
    }
};
} // namespace battery::ui
void require(bool ok, const std::string& why) {
    if (!ok)
        throw std::runtime_error(why);
}
int main(int argc, char** argv) {
    try {
        require(argc == 3, "Expected two contracts");
        require(glfwInit(), "GLFW failed");
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        auto* window = glfwCreateWindow(1400, 1000, "Import test", nullptr, nullptr);
        require(window, "GL failed");
        glfwMakeContextCurrent(window);
        ImGui::CreateContext();
        auto& io = ImGui::GetIO();
        io.DisplaySize = {1400, 1000};
        io.DeltaTime = 1.f / 60;
        io.IniFilename = nullptr;
        io.ConfigInputTrickleEventQueue = false;
        unsigned char* pixels;
        int w, h;
        io.Fonts->GetTexDataAsRGBA32(&pixels, &w, &h);
        {
            battery::ui::ExperimentEditor editor;
            auto& p = battery::ui::ExperimentEditorTestAccess::get(editor);
            editor.try_load(argv[1]);
            require(p.import_error.empty(), p.import_error);
            const auto first = p.document.draft.at("id");
            p.refresh_preview();
            p.browser = true;
            p.browse_path = std::filesystem::path(argv[2]).parent_path();
            auto browser_frame = [&] {
                ImGui::NewFrame();
                p.draw_browser();
                ImGui::Render();
            };
            browser_frame();
            browser_frame();
            std::vector<std::filesystem::path> files;
            for (const auto& e : std::filesystem::directory_iterator(p.browse_path))
                if (e.is_directory() || (e.path().extension() == ".json" &&
                                         !e.path().filename().string().ends_with(".editor.json")))
                    files.push_back(e.path());
            std::sort(files.begin(), files.end());
            auto found = std::find(files.begin(), files.end(), std::filesystem::path(argv[2]));
            require(found != files.end(), "Target file not listed");
            auto* browser = ImGui::FindWindowByName("Import experiment JSON");
            ImGuiWindow* list = nullptr;
            for (auto* win : ImGui::GetCurrentContext()->Windows)
                if (win->ParentWindow == browser)
                    list = win;
            require(list, "Browser list missing");
            const auto origin = list->DC.CursorStartPos;
            const ImVec2 target{origin.x + 40,
                                origin.y +
                                    float(found - files.begin()) * ImGui::GetTextLineHeightWithSpacing() + 5};
            require(target.y < list->InnerRect.Max.y, "Test file outside visible list");
            io.AddMousePosEvent(target.x, target.y);
            browser_frame();
            io.AddMouseButtonEvent(0, true);
            browser_frame();
            io.AddMouseButtonEvent(0, false);
            browser_frame();
            require(p.import_requested && !p.browser, "File click did not request import");
            editor.open = true;
            auto frame = [&] {
                ImGui::NewFrame();
                editor.draw();
                ImGui::Render();
            };
            frame();
            require(p.import_error.empty(), p.import_error);
            require(p.document.draft.at("id") != first, "File selection left previous document loaded");
            p.refresh_preview();
            const auto imported = p.document.draft;
            require(p.preview_source == imported, "Preview retained old contract");
            require(!editor.take_run(), "Import unexpectedly started/replaced running simulation");
            // Exercise the same Import dispatch with a companion view file,
            // not merely EditorDocument::load in isolation.
            const auto sidecar = std::string(argv[2]) + ".editor.json";
            if (std::filesystem::exists(sidecar)) {
                p.file_path = sidecar;
                p.import_requested = true;
                frame();
                require(p.import_error.empty(), p.import_error);
                require(p.document.draft == imported && p.document.path == std::filesystem::path(argv[2]),
                        "View-file import did not load its experiment or retained the wrong save path");
                p.refresh_preview();
                require(p.preview_source == imported, "View-file import did not reach preview");
            }
            p.file_path = "/tmp/battery-sim-definitely-missing-import.json";
            p.import_requested = true;
            frame();
            require(!p.import_error.empty() && p.document.draft == imported,
                    "Failed import destroyed draft or hid error");
            // A valid JSON with invalid physics must also be rejected transactionally.
            char tmp[] = "/tmp/battery-import-invalid-XXXXXX";
            const int fd = mkstemp(tmp);
            require(fd >= 0, "Temporary file failed");
            close(fd);
            auto invalid = imported;
            invalid["battery_models"][0]["contents"]["thermal"]["heat_capacity_j_per_k"] = -1;
            {
                std::ofstream out(tmp);
                out << invalid.dump();
            }
            editor.try_load(tmp);
            std::filesystem::remove(tmp);
            require(!p.import_error.empty() && p.document.draft == imported,
                    "Invalid graph replaced valid draft");
            p.document.dirty = true;
            p.file_path = argv[1];
            p.import_requested = true;
            frame();
            require(p.document.draft == imported, "Unsaved draft discarded without confirmation");
            require(ImGui::GetCurrentContext()->OpenPopupStack.Size > 0,
                    "Unsaved import confirmation not shown");
            std::cout << "Browser mouse click imports; preview switches; bad import is atomic; unsaved draft "
                         "protected\n";
        }
        ImGui::DestroyContext();
        glfwDestroyWindow(window);
        glfwTerminate();
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
