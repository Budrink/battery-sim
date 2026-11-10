#include "battery_ui/editor_document.hpp"
#include "battery_native/contracts/experiment_contract.hpp"
#include "battery_native/contracts/enclosure_openings.hpp"
#include <fstream>
#include <algorithm>
#include <set>
#include <stdexcept>
#include <unistd.h>

namespace battery::ui {
namespace {
nlohmann::json read(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file)
        throw std::runtime_error("Cannot open " + path.string());
    return nlohmann::json::parse(file);
}
void atomic_write(const std::filesystem::path& path, const nlohmann::json& j) {
    auto name = path.string() + ".tmp-XXXXXX";
    std::vector<char> bytes(name.begin(), name.end());
    bytes.push_back(0);
    const int fd = mkstemp(bytes.data());
    if (fd < 0)
        throw std::runtime_error("Cannot create temporary file beside " + path.string());
    bool closed = false;
    try {
        const auto data = j.dump(2) + "\n";
        std::size_t offset = 0;
        while (offset < data.size()) {
            const auto n = ::write(fd, data.data() + offset, data.size() - offset);
            if (n <= 0)
                throw std::runtime_error("Cannot write experiment");
            offset += std::size_t(n);
        }
        if (fsync(fd) != 0)
            throw std::runtime_error("Cannot flush experiment");
        const int result = ::close(fd);
        closed = true;
        if (result != 0)
            throw std::runtime_error("Cannot close experiment");
        std::filesystem::rename(bytes.data(), path);
    } catch (...) {
        if (!closed)
            ::close(fd);
        std::error_code ec;
        std::filesystem::remove(bytes.data(), ec);
        throw;
    }
}
} // namespace
void EditorDocument::load(const std::filesystem::path& source) {
    auto input = source;
    auto next = read(input);
    std::set<std::filesystem::path> visited;
    while (next.is_object() && next.contains("job_id") && next.contains("scenario_path")) {
        if (visited.size() >= 16 || !visited.insert(std::filesystem::weakly_canonical(input)).second)
            throw std::runtime_error("Cyclic or excessively nested scenario_path in " + source.string());
        const std::filesystem::path target = next.at("scenario_path").get<std::string>();
        input = (input.parent_path() / target).lexically_normal();
        next = read(input);
    }
    nlohmann::json selected_view;
    if (!next.is_object())
        throw std::runtime_error("Expected an experiment JSON object in " + source.string());
    if (next.value("format", "") == "battery-sim.editor") {
        // A Save produces two adjacent JSON files. Selecting the view file must
        // open its exact companion, not treat camera settings as a scenario.
        const std::string suffix = ".editor.json";
        const auto name = input.string();
        if (!name.ends_with(suffix))
            throw std::runtime_error("This file contains editor view settings, not an experiment. Select the "
                                     "experiment JSON instead: " +
                                     name);
        input = name.substr(0, name.size() - suffix.size());
        if (!std::filesystem::is_regular_file(input))
            throw std::runtime_error("Editor view selected, but its experiment file is missing: " +
                                     input.string());
        selected_view = next;
        next = read(input);
        if (!next.is_object())
            throw std::runtime_error("Expected an experiment JSON object in " + input.string());
    }
    const bool compiled_input = next.value("format", "") != "battery-sim.experiment";
    if (next.value("format", "") != "battery-sim.experiment") {
        if (next.contains("metadata") && next["metadata"].contains("experiment_contract"))
            next = next["metadata"]["experiment_contract"].get<nlohmann::json>();
        else if (next.contains("cells") && next.contains("simulation"))
            next = experiment_from_native(next);
        else {
            const auto format = next.value("format", "");
            std::string reason;
            if (format == "battery-sim.battery")
                reason = "This is a battery material model, not a complete experiment.";
            else if (next.contains("scenario_path"))
                reason = "This is a worker job. Open the experiment referenced by scenario_path: " +
                         next.at("scenario_path").dump();
            else
                reason = "This JSON is not a complete experiment (expected format: battery-sim.experiment). "
                         "Material presets and result files cannot replace an experiment.";
            throw std::runtime_error(input.string() + ": " + reason);
        }
    }
    next = upgrade_experiment(next);
    next = register_imported_enclosures(next);
    if (!selected_view.is_null())
        validate_editor_document(selected_view, next);
    nlohmann::json view{{"format", "battery-sim.editor"}, {"version", 1}, {"experiment_id", next.at("id")}};
    const auto view_path = input.string() + ".editor.json";
    if (std::filesystem::exists(view_path)) {
        try {
            auto saved = read(view_path);
            validate_editor_document(saved, next);
            view = std::move(saved);
        } catch (const std::exception&) {
        } // view is optional, never discard valid physics
    }
    draft = std::move(next);
    sidecar = std::move(view);
    path = compiled_input ? input.parent_path() / (input.stem().string() + ".experiment.json") : input;
    dirty = false;
    undo_.clear();
    redo_.clear();
}
nlohmann::json EditorDocument::compile() const {
    return compile_experiment(draft);
}
void EditorDocument::save(const std::filesystem::path& target) {
    (void)compile();
    auto view = sidecar;
    view["experiment_id"] = draft.at("id");
    // View preferences are not physics/history. Discard references to deleted elements
    // only when saving, so Undo can still restore their previous visibility.
    std::set<std::string> objects;
    for (const auto* key : {"batteries", "passive_cells"})
        for (const auto& object : draft.at("assembly").value(key, nlohmann::json::array()))
            objects.insert(object.at("id"));
    if (view.contains("hidden_objects")) {
        auto& hidden = view["hidden_objects"];
        hidden.erase(std::remove_if(hidden.begin(), hidden.end(),
                                    [&](const auto& id) { return !objects.contains(id); }),
                     hidden.end());
    }
    if (view.contains("layout") && !objects.contains(view["layout"].value("selected_object", "")))
        view["layout"].erase("selected_object");
    validate_editor_document(view, draft);
    atomic_write(target, draft);
    path = target;
    dirty = false;
    atomic_write(target.string() + ".editor.json", view);
    sidecar = std::move(view);
}
void EditorDocument::record(const nlohmann::json& before) {
    if (before == draft)
        return;
    undo_.push_back(before);
    if (undo_.size() > 32)
        undo_.erase(undo_.begin());
    redo_.clear();
    dirty = true;
}
void EditorDocument::undo() {
    if (undo_.empty())
        return;
    redo_.push_back(draft);
    draft = std::move(undo_.back());
    undo_.pop_back();
    dirty = true;
}
void EditorDocument::redo() {
    if (redo_.empty())
        return;
    undo_.push_back(draft);
    draft = std::move(redo_.back());
    redo_.pop_back();
    dirty = true;
}
} // namespace battery::ui
