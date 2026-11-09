#pragma once
#include <nlohmann/json.hpp>
#include <algorithm>
#include <string>

namespace battery::ui {
// Keeps creation intent distinct from an initially empty/stale selection.
// The source snapshot also lets property fields follow Undo/Redo, without
// overwriting an unfinished (possibly temporarily invalid) numeric edit.
struct OpeningSelection {
    std::string id, owner;
    nlohmann::json source;
    bool creating = false;

    void select(const std::string& case_id, const nlohmann::json& hole) {
        owner = case_id;
        id = hole.at("id");
        source = hole;
        creating = false;
    }
    void begin_new() {
        id.clear();
        source = nullptr;
        creating = true;
    }
    // Returns true only when fields need reloading from the document.
    bool sync(const std::string& case_id, const nlohmann::json& holes) {
        if (owner != case_id) {
            *this = {};
            owner = case_id;
        }
        if (creating)
            return false;
        auto found =
            std::find_if(holes.begin(), holes.end(), [&](const auto& h) { return h.at("id") == id; });
        if (found == holes.end())
            found = holes.begin();
        if (found == holes.end()) {
            id.clear();
            source = nullptr;
            return false;
        }
        if (id == found->at("id") && source == *found)
            return false;
        select(case_id, *found);
        return true;
    }
};
} // namespace battery::ui
