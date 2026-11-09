#include "battery_native/contracts/experiment_editing.hpp"
#include "battery_native/contracts/experiment_contract.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace battery {
nlohmann::json set_battery_geometry(const nlohmann::json& input, const std::string& bid,
                                    const std::string& kind, const std::vector<double>& dimensions) {
    using J = nlohmann::json;
    const bool cylinder = kind == "cylinder";
    if ((!cylinder && kind != "prismatic") || dimensions.size() != (cylinder ? 2u : 3u) ||
        std::any_of(dimensions.begin(), dimensions.end(),
                    [](double d) { return !std::isfinite(d) || d <= 0; }))
        throw std::invalid_argument("Geometry requires positive finite dimensions");
    auto e = upgrade_experiment(input);
    auto& batteries = e["assembly"]["batteries"];
    auto selected =
        std::find_if(batteries.begin(), batteries.end(), [&](const auto& b) { return b.at("id") == bid; });
    if (selected == batteries.end())
        throw std::invalid_argument("Select a battery to reshape");
    auto& models = e["battery_models"];
    auto source = std::find_if(models.begin(), models.end(), [&](const auto& m) {
        return m.at("id") == selected->at("model_id") && m.at("revision") == selected->at("model_revision");
    });
    if (source == models.end())
        throw std::invalid_argument("Battery model is missing");
    J model = *source;
    auto& mesh = e["numerics"]["batteries"][bid];
    const auto old_dimensions =
        mesh.contains("geometry_approximation")
            ? mesh.at("geometry_approximation").at("geometry").at("dimensions_m").get<std::vector<double>>()
            : model.at("geometry").at("dimensions_m").get<std::vector<double>>();
    auto box = dimensions;
    if (cylinder) {
        const double side = std::sqrt(std::acos(-1.)) * dimensions[0];
        box = {side, side, dimensions[1]};
        mesh["geometry_approximation"] = {
            {"geometry", {{"kind", "prismatic"}, {"dimensions_m", box}}},
            {"reason", "Editor: equal-volume square prism; cylinder rendered from original geometry. Contact "
                       "areas require review."}};
    } else {
        mesh.erase("geometry_approximation");
    }
    model["geometry"] = {{"kind", kind}, {"dimensions_m", dimensions}};
    if (model["provenance"]["status"] != "synthetic")
        model["provenance"]["status"] = "mixed";
    const auto references = std::count_if(batteries.begin(), batteries.end(), [&](const auto& b) {
        return b.at("model_id") == source->at("id") && b.at("model_revision") == source->at("revision");
    });
    // Copy-on-write: resizing B must not silently reshape every battery of its model.
    if (references == 1 && model.at("revision") == "editor-geometry") {
        *source = model;
    } else {
        std::string id = bid + "-geometry";
        for (int suffix = 1;
             std::any_of(models.begin(), models.end(), [&](const auto& m) { return m.at("id") == id; });
             ++suffix)
            id = bid + "-geometry-" + std::to_string(suffix);
        model["id"] = id;
        model["revision"] = "editor-geometry";
        selected->at("model_id") = id;
        selected->at("model_revision") = "editor-geometry";
        models.push_back(model);
    }
    // Keep attached local, axis-aligned surface ports on the same fractional face.
    // World ports and neighbouring apparatus/routes are deliberately not inferred.
    if (old_dimensions.size() == 3)
        for (auto& p : e["assembly"]["ports"]) {
            if (p.at("cell_id") != bid || p.value("frame", "world") != "local")
                continue;
            int axis = -1;
            for (int i = 0; i < 3; ++i)
                if (std::abs(p.at("normal")[i].get<double>()) > 1 - 1e-9)
                    axis = i;
            if (axis < 0)
                continue;
            for (int i = 0; i < 3; ++i)
                p["centroid_m"][i] = p["centroid_m"][i].get<double>() * box[i] / old_dimensions[i];
            const int u = (axis + 1) % 3, v = (axis + 2) % 3;
            const double su = box[u] / old_dimensions[u], sv = box[v] / old_dimensions[v];
            p["area_m2"] = p.at("area_m2").get<double>() * su * sv;
            if (p.contains("rectangle_m")) {
                p["rectangle_m"][0] = p["rectangle_m"][0].get<double>() * su;
                p["rectangle_m"][1] = p["rectangle_m"][1].get<double>() * sv;
            }
        }
    e["metadata"]["editor_changes"].push_back(
        {{"kind", "battery_geometry"},
         {"battery_id", bid},
         {"geometry", model["geometry"]},
         {"assumption", "Mass, heat capacity, chemistry, vents and external gas/contact routes retained; "
                        "review apparatus after resizing."}});
    if (e["provenance"]["status"] != "synthetic")
        e["provenance"]["status"] = "mixed";
    (void)compile_experiment(e);
    const auto mounts = e.value("metadata", J::object()).value("heater_mounts", J::object());
    for (const auto& [hid, mount] : mounts.items())
        if (mount.at("battery_id") == bid) {
            if (!cylinder)
                throw std::invalid_argument(
                    "Remove cylindrical heaters before changing this battery to a prism");
            e = set_heater_mount(e, hid, bid, mount.at("face"));
        }
    return e;
}
} // namespace battery
