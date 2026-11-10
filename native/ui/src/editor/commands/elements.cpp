#include "editor/editor_internal.hpp"
#include "battery_native/contracts/experiment_contract.hpp"
#include "battery_native/contracts/experiment_editing.hpp"
#include "battery_native/contracts/enclosure_openings.hpp"
#include "battery_native/scenario_document.hpp"
#include <algorithm>
namespace battery::ui {
using J = nlohmann::json;
void ExperimentEditor::Impl::delete_selected() {
    try {
        if (selected.empty() && !case_opening_selection.id.empty()) {
            auto list = editable_enclosures(document.draft).at(enclosure_id).value("openings", J::array());
            list.erase(std::remove_if(list.begin(), list.end(),
                                      [&](const auto& h) { return h.at("id") == case_opening_selection.id; }),
                       list.end());
            document.draft = set_enclosure_openings(document.draft, enclosure_id, list);
            case_opening_selection.id.clear();
            message = "Opening deleted; Undo restores it.";
            return;
        }
        const auto id = selected;
        document.draft = delete_element(document.draft, id);
        selected.clear();
        message = "Deleted " + id +
                  " and dependent contacts/routes. Undo restores the edit. "
                  "Running simulation unchanged; review wiring and heater controls before Run simulation.";
    } catch (const std::exception& e) {
        message = e.what();
    }
}
} // namespace battery::ui
