#include "editor/forms/heater_controller_panel.hpp"
#include "battery_native/contracts/experiment_editing.hpp"
#include <nlohmann/json.hpp>
#include "imgui.h"
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <cmath>

int main(int argc, char** argv) {
    try {
        using J = nlohmann::json;
        J a = {{"id", "control-a"},
               {"heater_id", "heater-a"},
               {"sensor_id", "B00"},
               {"initial_temperature_k", 298.15},
               {"ramp_k_per_s", .1},
               {"feedforward_w", 0.},
               {"proportional_w_per_k", .2},
               {"maximum_power_w", 30.},
               {"cutoff_temperature_k", 773.15},
               {"cutoff_rate_k_per_s", 10.}};
        J b = a;
        b["id"] = "control-b";
        b["heater_id"] = "heater-b";
        b["cutoff_temperature_k"] = 423.15;
        b["maximum_power_w"] = 5.;
        const auto original_a = a, original_b = b;
        if (argc != 2)
            throw std::runtime_error("Expected starter contract");
        J source;
        std::ifstream(argv[1]) >> source;
        const std::string bid = source.at("assembly").at("batteries")[0].at("id");
        auto created =
            battery::add_external_heater(source, bid, "+x", 5., 500., 5., false, .1, 773.15, 0., true);
        auto imported = J::parse(created.dump());
        const auto original_created = created;
        ImGui::CreateContext();
        auto& io = ImGui::GetIO();
        io.IniFilename = nullptr;
        io.DisplaySize = {1000, 1600};
        io.DeltaTime = 1.f / 60;
        unsigned char* pixels;
        int width, height;
        io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
        ImGui::NewFrame();
        ImGui::SetNextWindowSize({950, 1500});
        ImGui::Begin("Heaters");
        ImGui::PushID("heater-a");
        auto first = ImGui::GetID("Cutoff temperature C");
        battery::ui::draw_heater_controller(a);
        ImGui::PopID();
        ImGui::PushID("heater-b");
        auto second = ImGui::GetID("Cutoff temperature C");
        battery::ui::draw_heater_controller(b);
        ImGui::PopID();
        const auto card = [](J& doc, const char* scope) {
            auto& obj = doc["assembly"]["passive_cells"].back();
            auto& control = doc["conditions"]["heating_protocols"].back();
            auto& coupling = doc["assembly"]["interfaces"].back();
            const double area = doc["assembly"]["ports"].back().at("area_m2");
            ImGui::PushID(scope);
            const float top = ImGui::GetCursorPosY();
            battery::ui::draw_heater_parameters(obj, &control, &coupling, area);
            const float height = ImGui::GetCursorPosY() - top;
            ImGui::PopID();
            return height;
        };
        const float created_height = card(created, "created");
        const float imported_height = card(imported, "imported");
        ImGui::End();
        ImGui::Render();
        ImGui::DestroyContext();
        if (first == second || a != original_a || b != original_b)
            throw std::runtime_error("Heater cards share widget identity or mutate values on display");
        if (created != original_created || imported != original_created ||
            std::abs(created_height - imported_height) > .01f)
            throw std::runtime_error("New/imported heater parameter forms differ or reset values");
        std::cout << "Unified new/imported heater cards: identical form, distinct IDs, preserved settings\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
