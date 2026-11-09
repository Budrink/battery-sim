#include "model_properties.hpp"
#include "battery_ui/editor_document.hpp"
#include "battery_native/scenario_document.hpp"
#include "imgui.h"
#include "imgui_internal.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}
void open_sections() {
    auto* storage = ImGui::GetStateStorage();
    for (const char* title : {"Internal conduction", "Electrical properties", "Shell material"})
        storage->SetInt(ImGui::GetID(title), 1);
    ImGui::PushID("conversion");
    for (const char* title :
         {"Component heat storage", "R1: degradation", "R2: runaway", "Liquid and evaporation"})
        storage->SetInt(ImGui::GetID(title), 1);
    ImGui::PopID();
}
} // namespace
int main(int argc, char** argv) {
    try {
        require(argc == 2, "Expected starter contract");
        battery::ui::EditorDocument document;
        document.load(argv[1]);
        const auto original = document.draft;
        auto& model = document.draft.at("battery_models").at(0);
        ImGui::CreateContext();
        auto& io = ImGui::GetIO();
        io.IniFilename = nullptr;
        io.DisplaySize = {1400, 1600};
        io.DeltaTime = 1.f / 60;
        unsigned char* pixels;
        int width, height;
        io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
        auto frame = [&](float pane_width, bool focus = false) {
            ImGui::NewFrame();
            ImGui::SetNextWindowPos({0, 0});
            ImGui::SetNextWindowSize({pane_width, 1500});
            ImGui::Begin("Models", nullptr, ImGuiWindowFlags_NoSavedSettings);
            open_sections();
            if (focus)
                ImGui::SetKeyboardFocusHere();
            battery::ui::draw_model_properties(model);
            const auto* window = ImGui::GetCurrentWindow();
            require(window->DC.CursorMaxPos.x <= window->WorkRect.Max.x + 1.f,
                    "Model form exceeds pane width");
            ImGui::End();
            ImGui::Render();
        };
        for (float pane : {280.f, 420.f, 620.f}) {
            frame(pane);
            frame(pane);
        }
        require(document.draft == original, "Viewing properties changed contract values");
        frame(420, true);
        frame(420);
        io.AddKeyEvent(ImGuiMod_Ctrl, true);
        io.AddKeyEvent(ImGuiKey_A, true);
        frame(420);
        io.AddKeyEvent(ImGuiKey_A, false);
        io.AddKeyEvent(ImGuiMod_Ctrl, false);
        io.AddInputCharactersUTF8("60");
        frame(420);
        io.AddKeyEvent(ImGuiKey_Enter, true);
        frame(420);
        io.AddKeyEvent(ImGuiKey_Enter, false);
        frame(420);
        require(model.at("contents").at("thermal").at("heat_capacity_j_per_k") == 60.,
                "Heat capacity field is not editable");
        auto expected = original;
        expected["battery_models"][0]["contents"]["thermal"]["heat_capacity_j_per_k"] = 60.;
        require(document.draft == expected, "Capacity edit affected unrelated parameters");
        const auto scene = battery::scenario_from_json(document.compile());
        int matched = 0;
        for (const auto& cell : scene.cells)
            if (cell.is_battery) {
                require(std::abs(cell.heat_capacity_j_per_k - 60.) < 1e-10,
                        "Shared model capacity did not reach compiled batteries");
                ++matched;
            }
        require(matched == 16, "Expected 16 batteries sharing the starter model");
        document.record(original);
        document.undo();
        require(document.draft == original, "Capacity Undo failed");
        document.redo();
        require(document.draft == expected, "Capacity Redo failed");
        ImGui::DestroyContext();
        std::cout
            << "Models: narrow layout, no-op preservation, capacity input, compilation and Undo/Redo OK\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
