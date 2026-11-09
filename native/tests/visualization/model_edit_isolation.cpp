#include "model_edits.hpp"
#include "battery_ui/editor_document.hpp"
#include "battery_native/contracts/experiment_editing.hpp"
#include "battery_native/scenario_document.hpp"
#include "battery_native/simulation/simulation.hpp"
#include <iostream>
#include <stdexcept>

namespace {
void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}
} // namespace
int main(int argc, char** argv) {
    try {
        require(argc == 2, "Expected starter contract");
        battery::ui::EditorDocument doc;
        doc.load(argv[1]);
        const auto original = doc.draft;
        // Reproduce the old failure: one invalid heat capacity blocks the entire
        // preview, including otherwise valid changes to unrelated elements.
        auto invalid = original;
        invalid["battery_models"][0]["contents"]["thermal"]["heat_capacity_j_per_k"] = 10.;
        doc.draft = invalid;
        bool rejected = false;
        try {
            (void)battery::scenario_from_json(doc.compile());
        } catch (const std::exception& e) {
            rejected = true;
            std::cout << "Reproduced global rejection: " << e.what() << '\n';
        }
        require(rejected, "Invalid capacity unexpectedly accepted");
        doc.draft = original;
        battery::ui::ModelEdits edits;
        auto& buffer = edits.edit(0, doc.draft.at("battery_models")[0]);
        buffer.value = invalid["battery_models"][0];
        edits.commit_valid(doc.draft);
        require(doc.draft == original && edits.pending() && !buffer.error.empty(),
                "Invalid model poisoned the shared document or lost its error");
        // The invalid form stays open, while another element can still update.
        doc.draft = battery::set_pad_conductivity_preset(doc.draft, "ISS-B00-B01", "thermal_conductor");
        edits.commit_valid(doc.draft);
        require(edits.pending(), "Unrelated edit discarded pending model input");
        auto scene = battery::scenario_from_json(doc.compile());
        bool pad_changed = false;
        for (const auto& cell : scene.cells)
            if (cell.id == "ISS-B00-B01")
                pad_changed = cell.conductivity_w_per_m_k[0] == 2.;
        require(pad_changed, "Pad edit did not reach the compiled preview");
        {
            battery::Simulation preview(scene);
            require(!preview.thermal_mesh().volumes.empty(), "CUDA preview did not build");
        }
        const auto before_capacity = doc.draft;
        buffer.value["contents"]["thermal"]["heat_capacity_j_per_k"] = 60.;
        edits.commit_valid(doc.draft);
        require(!edits.pending() && buffer.error.empty(), "Valid correction not accepted");
        require(doc.draft["assembly"] == before_capacity["assembly"], "Model correction erased pad edit");
        {
            battery::Simulation preview(battery::scenario_from_json(doc.compile()));
            require(!preview.thermal_mesh().volumes.empty(), "Corrected CUDA preview did not build");
        }
        doc.record(before_capacity);
        doc.undo();
        edits.synchronize(doc.draft);
        require(!edits.pending() && edits.edit(0, doc.draft["battery_models"][0]).value ==
                                        before_capacity["battery_models"][0],
                "Undo left stale model input that could overwrite restored values");
        std::cout << "Invalid capacity isolated; pad edit and corrected capacity rebuild on CUDA; Undo OK\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
