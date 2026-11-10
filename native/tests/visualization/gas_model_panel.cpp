#include "editor/panels/gas_model_panel.hpp"
#include "battery_native/contracts/experiment_contract.hpp"
#include "battery_native/contracts/gas_combustion_model.hpp"
#include "battery_native/scenario_document.hpp"
#include "battery_native/simulation/simulation.hpp"
#include "battery_ui/editor_document.hpp"
#include "imgui.h"
#include "imgui_internal.h"
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
} // namespace
int main(int argc, char** argv) {
    try {
        require(argc == 2, "Expected a test contract path");
        using namespace battery;
        using namespace battery::ui;
        EditorDocument document;
        document.load(argv[1]);
        const auto original = document.draft;
        ModelEdits edits;
        ImGui::CreateContext();
        auto& io = ImGui::GetIO();
        io.IniFilename = nullptr;
        io.DisplaySize = {1200, 1800};
        io.DeltaTime = 1.f / 60;
        unsigned char* pixels;
        int w, h;
        io.Fonts->GetTexDataAsRGBA32(&pixels, &w, &h);
        auto frame = [&](float width, bool focus = false) {
            ImGui::NewFrame();
            ImGui::SetNextWindowPos({0, 0});
            ImGui::SetNextWindowSize({width, 1600});
            ImGui::Begin("R3 form", nullptr, ImGuiWindowFlags_NoSavedSettings);
            if (focus)
                ImGui::SetKeyboardFocusHere();
            draw_gas_combustion_model(document.draft.at("assembly").at("gas_network"), edits);
            const auto* win = ImGui::GetCurrentWindow();
            require(win->DC.CursorMaxPos.x <= win->WorkRect.Max.x + 1,
                    "R3 captions exceed narrow panel width");
            ImGui::End();
            ImGui::Render();
        };
        for (float width : {280.f, 420.f, 620.f}) {
            frame(width);
            frame(width);
        }
        edits.commit_valid(document.draft);
        require(document.draft == original && !edits.pending(), "Viewing R3 rewrote the imported document");
        frame(420, true);
        frame(420);
        io.AddKeyEvent(ImGuiMod_Ctrl, true);
        io.AddKeyEvent(ImGuiKey_A, true);
        frame(420);
        io.AddKeyEvent(ImGuiKey_A, false);
        io.AddKeyEvent(ImGuiMod_Ctrl, false);
        io.AddInputCharactersUTF8("0.75");
        frame(420);
        io.AddKeyEvent(ImGuiKey_Enter, true);
        frame(420);
        io.AddKeyEvent(ImGuiKey_Enter, false);
        frame(420);
        require(edits.edit_gas(document.draft.at("assembly").at("gas_network"))
                        .value["combustion"]["reaction_rate_per_s"] == .75,
                "R3 rate input is not editable");
        edits.apply_global_gas();
        edits.commit_valid(document.draft);
        require(!edits.pending(), "Valid shared R3 failed to apply");
        auto scene = scenario_from_json(compile_experiment(document.draft));
        require(scene.gas_compartments.size() > original["assembly"]["gas_network"]["compartments"].size(),
                "Fixture did not generate a case mesh");
        for (const auto& g : scene.gas_compartments)
            require(g.reaction_rate_per_s == .75, "Generated/internal gas missed the global R3 law");
        require(document.draft["battery_models"] == original["battery_models"],
                "Global kinetics changed battery chemistry/materials");
        const auto valid = document.draft;
        edits.edit_gas(document.draft["assembly"]["gas_network"]).value["combustion"]["reaction_rate_per_s"] =
            -1.;
        edits.commit_valid(document.draft);
        require(edits.pending() && document.draft == valid, "Invalid R3 poisoned the accepted document");
        document.draft["name"] = "Other edits still work";
        edits.commit_valid(document.draft);
        require(document.draft.at("name") == "Other edits still work", "Invalid R3 blocks unrelated editing");
        edits.discard_gas();
        require(!edits.pending(), "Discard left R3 pending");
        edits.edit_gas(document.draft["assembly"]["gas_network"]).value["vapor_fuel_heat_j_kg"] = 4.2e6;
        edits.commit_valid(document.draft);
        if (edits.pending())
            throw std::runtime_error("Global fuel energy cannot be edited consistently: " +
                                     edits.edit_gas(document.draft["assembly"]["gas_network"]).error);
        scene = scenario_from_json(compile_experiment(document.draft));
        for (const auto& cell : scene.cells)
            if (cell.conversion.volatile_phase.initial_mass_kg > 0)
                require(cell.conversion.volatile_phase.chemical_j_per_kg == 4.2e6,
                        "Liquid and gas chemical reserves disagree");
        require(document.draft["battery_models"] == original["battery_models"],
                "Global fuel model rewrote stored battery templates");
        auto roundtrip = nlohmann::json::parse(document.draft.dump());
        require(compile_experiment(roundtrip) == compile_experiment(document.draft),
                "Shared law did not survive JSON roundtrip");
        Simulation sim(scene);
        sim.step(.005);
        const auto d = sim.process_diagnostics();
        require(std::abs(d.mass_residual_kg) < 1e-10 && std::abs(d.thermal_residual_j) < 1e-5 &&
                    std::abs(d.chemical_residual_j) < 1e-5,
                "Global R3 snapshot breaks CUDA balances");
        auto mixed = original;
        mixed["assembly"]["gas_network"]["compartments"][0]["reaction_rate_per_s"] = 23.;
        ModelEdits mixed_edits;
        auto& candidate = mixed_edits.edit_gas(mixed["assembly"]["gas_network"]);
        candidate.value["combustion"]["reaction_rate_per_s"] = .3;
        mixed_edits.commit_valid(mixed);
        require(!mixed["assembly"]["gas_network"].contains("combustion"),
                "Heterogeneous legacy laws silently homogenized");
        mixed_edits.apply_global_gas();
        mixed_edits.commit_valid(mixed);
        require(!mixed_edits.pending() &&
                    mixed["assembly"]["gas_network"]["combustion"]["reaction_rate_per_s"] == .3,
                "Explicit shared-law replacement failed");
        auto bad = document.draft;
        bad["assembly"]["gas_network"]["combustion"]["unknown_parameter"] = 1;
        bool rejected = false;
        try {
            (void)compile_experiment(bad);
        } catch (const std::exception&) {
            rejected = true;
        }
        require(rejected, "Global combustion accepts misspelled parameters");
        ImGui::DestroyContext();
        std::cout << "Global R3: narrow forms, actual keyboard edit, no mutation on view, all generated "
                     "volumes, invalid-input isolation, legacy confirmation, fuel reserve, roundtrip and "
                     "CUDA balance pass\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
