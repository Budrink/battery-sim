#include "battery_native/contracts/experiment_editing.hpp"
#include "battery_native/contracts/experiment_contract.hpp"
#include "battery_native/scenario_document.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

namespace battery {
nlohmann::json set_shell_face(const nlohmann::json& input, const std::string& bid, const std::string& face,
                              const nlohmann::json& material, const nlohmann::json& channel,
                              const std::string& destination) {
    using J = nlohmann::json;
    auto e = upgrade_experiment(input);
    const auto s = scenario_from_json(compile_experiment(e));
    const std::string suffix = "case:" + face + ":outer";
    const auto p = std::find_if(s.ports.begin(), s.ports.end(),
                                [&](const auto& port) { return port.id == bid + ":" + suffix; });
    if (p == s.ports.end())
        throw std::invalid_argument("Select an existing shell face");
    auto& batteries = e["assembly"]["batteries"];
    auto b = std::find_if(batteries.begin(), batteries.end(),
                          [&](const auto& item) { return item.at("id") == bid; });
    if (b == batteries.end())
        throw std::invalid_argument("Unknown battery");
    auto& models = e["battery_models"];
    auto source = std::find_if(models.begin(), models.end(), [&](const auto& m) {
        return m.at("id") == b->at("model_id") && m.at("revision") == b->at("model_revision");
    });
    J model = *source;
    model["shell"]["face_materials"][face] = material;
    std::set<std::string> removed;
    J vents = J::array();
    for (const auto& v : model["vents"])
        if (v.at("port_suffix") == suffix)
            removed.insert(v.at("id"));
        else
            vents.push_back(v);
    auto& routes = e["assembly"]["vent_connections"];
    routes.erase(std::remove_if(routes.begin(), routes.end(),
                                [&](const auto& r) {
                                    return r.at("battery_id") == bid &&
                                           removed.contains(r.at("vent_id").template get<std::string>());
                                }),
                 routes.end());
    if (!channel.is_null()) {
        if (destination.empty())
            throw std::invalid_argument("Opening requires a gas destination");
        J law = channel;
        if (law.contains("damage_opening")) {
            const auto& opening = law.at("damage_opening");
            if (opening.contains("valve")) {
                const auto& valve = opening.at("valve");
                const auto center = valve.at("center_m").get<std::array<double, 2>>();
                const double radius = valve.at("radius_m");
                for (int axis = 0; axis < 2; ++axis)
                    if (!std::isfinite(center[axis]) || !std::isfinite(radius) || radius <= 0 ||
                        std::abs(center[axis]) + radius > p->rectangle_m[axis] / 2)
                        throw std::invalid_argument("Valve must fit completely on the selected face");
            }
            const double seam = opening.value("seam_width_m", 0.);
            if (!std::isfinite(seam) || seam < 0 || seam > std::min(p->rectangle_m[0], p->rectangle_m[1]) / 2)
                throw std::invalid_argument("Seam width exceeds the selected face");
        }
        law["kind"] = "rupture";
        law["area_m2"] = 0.; // derive from the rebuilt plate, not the pre-edit geometry
        const std::string vid = "editor-" + std::string(face[0] == '+' ? "plus" : "minus") + face[1];
        vents.push_back({{"id", vid}, {"port_suffix", suffix}, {"channel", law}});
        routes.push_back(
            {{"id", bid + "-" + vid}, {"battery_id", bid}, {"vent_id", vid}, {"to", destination}});
    }
    model["vents"] = vents;
    const auto references = std::count_if(batteries.begin(), batteries.end(), [&](const auto& other) {
        return other.at("model_id") == source->at("id") &&
               other.at("model_revision") == source->at("revision");
    });
    if (model["provenance"]["status"] != "synthetic")
        model["provenance"]["status"] = "mixed";
    if (references == 1 && model.at("revision") == "editor-shell")
        *source = model;
    else {
        std::string id = bid + "-shell";
        for (int k = 1;
             std::any_of(models.begin(), models.end(), [&](const auto& m) { return m.at("id") == id; }); ++k)
            id = bid + "-shell-" + std::to_string(k);
        model["id"] = id;
        model["revision"] = "editor-shell";
        (*b)["model_id"] = id;
        (*b)["model_revision"] = "editor-shell";
        models.push_back(model);
    }
    e["metadata"]["editor_changes"].push_back({{"kind", "shell_face"}, {"battery_id", bid}, {"face", face}});
    if (e["provenance"]["status"] != "synthetic")
        e["provenance"]["status"] = "mixed";
    const auto rebuilt = scenario_from_json(compile_experiment(e));
    for (const auto& plate : rebuilt.cells) {
        if (plate.shell_contents_id != bid) continue;
        for (const auto& other : rebuilt.cells) {
            if (&plate == &other) continue;
            const double p0[]{plate.position_m.x, plate.position_m.y, plate.position_m.z};
            const double p1[]{other.position_m.x, other.position_m.y, other.position_m.z};
            double overlap = 1.;
            for (int d = 0; d < 3; ++d)
                overlap *= std::max(0., std::min(p0[d] + plate.dimensions_m[d]/2, p1[d] + other.dimensions_m[d]/2) -
                                         std::max(p0[d] - plate.dimensions_m[d]/2, p1[d] - other.dimensions_m[d]/2));
            if (overlap > 1e-16)
                throw std::invalid_argument("Plate thickness overlaps " + other.id + "; adjust the assembly clearance first");
        }
    }
    return e;
}
} // namespace battery
