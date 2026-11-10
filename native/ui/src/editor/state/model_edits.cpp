#include "editor/state/model_edits.hpp"
#include "battery_native/contracts/experiment_contract.hpp"
#include "battery_native/scenario_document.hpp"
#include "battery_native/contracts/gas_combustion_model.hpp"
#include "battery_native/contracts/experiment_editing.hpp"
#include <cmath>
#include <set>

namespace battery::ui {
namespace {
// Keep the model geometry and its numerical surrogate consistent in one edit.
void prepare_shape(nlohmann::json& candidate, const nlohmann::json& old_document, std::size_t index) {
    const auto& model = candidate.at("battery_models").at(index);
    const auto& old_model = old_document.at("battery_models").at(index);
    if (model.at("geometry") == old_model.at("geometry"))
        return;
    const auto& geometry = model.at("geometry");
    auto box = geometry.at("dimensions_m").get<std::vector<double>>();
    for (double x : box)
        if (!std::isfinite(x) || x <= 0)
            throw std::invalid_argument("Dimensions must be positive");
    const bool cylinder = geometry.at("kind") == "cylinder";
    if (cylinder)
        box = {std::sqrt(std::acos(-1.)) * box.at(0), std::sqrt(std::acos(-1.)) * box.at(0), box.at(1)};
    std::set<std::string> changed;
    for (const auto& b : candidate.at("assembly").at("batteries")) {
        if (b.at("model_id") != model.at("id") || b.at("model_revision") != model.at("revision"))
            continue;
        const auto bid = b.at("id").get<std::string>();
        changed.insert(bid);
        auto& mesh = candidate["numerics"]["batteries"][bid];
        const auto old_box = mesh.contains("geometry_approximation")
                                 ? mesh.at("geometry_approximation")
                                       .at("geometry")
                                       .at("dimensions_m")
                                       .get<std::vector<double>>()
                                 : old_model.at("geometry").at("dimensions_m").get<std::vector<double>>();
        if (cylinder)
            mesh["geometry_approximation"] = {
                {"geometry", {{"kind", "prismatic"}, {"dimensions_m", box}}},
                {"reason", "Model editor: equal-volume thermal prism; explicit cylindrical heater contacts"}};
        else
            mesh.erase("geometry_approximation");
        for (auto& p : candidate["assembly"]["ports"]) {
            if (p.at("cell_id") != bid || p.value("frame", "world") != "local")
                continue;
            int axis = -1;
            for (int k = 0; k < 3; ++k)
                if (std::abs(p.at("normal")[k].get<double>()) > 1 - 1e-9)
                    axis = k;
            if (axis < 0)
                continue;
            for (int k = 0; k < 3; ++k)
                p["centroid_m"][k] = p["centroid_m"][k].get<double>() * box[k] / old_box.at(k);
            const int u = (axis + 1) % 3, v = (axis + 2) % 3;
            p["area_m2"] = p["area_m2"].get<double>() * box[u] * box[v] / old_box.at(u) / old_box.at(v);
            if (p.contains("rectangle_m")) {
                p["rectangle_m"][0] = p["rectangle_m"][0].get<double>() * box[u] / old_box.at(u);
                p["rectangle_m"][1] = p["rectangle_m"][1].get<double>() * box[v] / old_box.at(v);
            }
        }
    }
    const auto mounts = candidate.value("metadata", nlohmann::json::object())
                            .value("heater_mounts", nlohmann::json::object());
    for (const auto& [hid, mount] : mounts.items())
        if (changed.contains(mount.at("battery_id"))) {
            if (!cylinder)
                throw std::invalid_argument(
                    "Remove cylindrical heaters before changing this model to a prism");
            candidate = set_heater_mount(candidate, hid, mount.at("battery_id"), mount.at("face"));
        }
    // A legacy all-battery visual override must not hide an explicitly edited model shape.
    candidate["metadata"].erase("packed_cylinder_geometry");
}
} // namespace
ModelEdits::Entry& ModelEdits::edit(std::size_t index, const nlohmann::json& model) {
    auto& entry = entries_[index];
    if (entry.source != model)
        entry = {model, model, {}, {}};
    return entry;
}
ModelEdits::Entry& ModelEdits::edit_gas(const nlohmann::json& network) {
    using J = nlohmann::json;
    const auto variants = gas_combustion_variants(network);
    const J signature = {variants, network.contains("combustion"), network.value("vapor_fuel_heat_j_kg", 0.)};
    if (!gas_ || gas_signature_ != signature) {
        J law = variants.front();
        for (const auto& candidate : variants)
            if (candidate.at("reaction_rate_per_s").get<double>() > 0) {
                law = candidate;
                break;
            }
        J value = {{"combustion", law}, {"vapor_fuel_heat_j_kg", network.value("vapor_fuel_heat_j_kg", 0.)}};
        gas_ = Entry{value, value, {}, {}};
        gas_signature_ = signature;
        gas_apply_ = false;
    }
    return *gas_;
}
void ModelEdits::synchronize(const nlohmann::json& document) {
    if (gas_)
        edit_gas(document.at("assembly").at("gas_network"));
    const auto& models = document.at("battery_models");
    for (auto it = entries_.begin(); it != entries_.end();) {
        if (it->first >= models.size()) {
            it = entries_.erase(it);
        } else {
            edit(it->first, models[it->first]);
            ++it;
        }
    }
}
bool ModelEdits::pending() const {
    if (gas_ && (gas_apply_ || gas_->value != gas_->source))
        return true;
    for (const auto& [index, entry] : entries_)
        if (entry.value != entry.source)
            return true;
    return false;
}
void ModelEdits::commit_valid(nlohmann::json& document) {
    synchronize(document);
    if (gas_ && (gas_apply_ || gas_->value != gas_->source)) {
        auto candidate = document;
        auto& network = candidate.at("assembly").at("gas_network");
        network["combustion"] = gas_->value.at("combustion");
        network["vapor_fuel_heat_j_kg"] = gas_->value.at("vapor_fuel_heat_j_kg");
        // A heterogeneous legacy file is not silently homogenized while typing.
        const bool mixed = gas_combustion_variants(document.at("assembly").at("gas_network")).size() > 1;
        if (mixed && !gas_apply_)
            gas_->error = "Imported R3 laws differ. Use Unify imported R3 laws to confirm replacement.";
        else if (candidate != gas_->attempted) {
            gas_->attempted = candidate;
            try {
                (void)scenario_from_json(compile_experiment(candidate));
                auto& target = document.at("assembly").at("gas_network");
                target["combustion"] = network.at("combustion");
                target["vapor_fuel_heat_j_kg"] = network.at("vapor_fuel_heat_j_kg");
                gas_.reset();
                gas_apply_ = false;
            } catch (const std::exception& e) {
                gas_->error = e.what();
            }
        }
    }
    for (auto& [index, entry] : entries_) {
        if (entry.value == entry.source) {
            entry.error.clear();
            continue;
        }
        auto candidate = document;
        candidate.at("battery_models").at(index) = entry.value;
        if (candidate == entry.attempted)
            continue;
        entry.attempted = candidate;
        try {
            const bool shape_changed = entry.value.at("geometry") != entry.source.at("geometry");
            prepare_shape(candidate, document, index);
            // Host preflight includes mass/heat-capacity consistency. No CUDA
            // allocation or physical advance is required to validate a field.
            (void)scenario_from_json(compile_experiment(candidate));
            // Preserve the parent document and references held by other forms.
            document.at("battery_models").at(index) = entry.value;
            if (shape_changed) {
                document["numerics"] = candidate["numerics"];
                document["assembly"] = candidate["assembly"];
                document["metadata"] = candidate["metadata"];
            }
            entry.source = entry.value;
            entry.error.clear();
            entry.attempted = nullptr;
        } catch (const std::exception& e) {
            entry.error = e.what();
        }
    }
}
} // namespace battery::ui
