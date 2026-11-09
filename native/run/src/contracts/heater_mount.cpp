#include "battery_native/contracts/experiment_editing.hpp"
#include "battery_native/contracts/experiment_contract.hpp"
#include "battery_native/scenario_document.hpp"
#include "battery_native/thermal/thermal_mesh.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

namespace battery {
nlohmann::json set_heater_mount(const nlohmann::json& input, const std::string& hid, const std::string& bid,
                                const std::string& face) {
    using J = nlohmann::json;
    auto e = upgrade_experiment(input);
    auto& a = e.at("assembly");
    const auto instance = std::find_if(a["batteries"].begin(), a["batteries"].end(),
                                       [&](const auto& b) { return b.at("id") == bid; });
    if (instance == a["batteries"].end())
        throw std::invalid_argument("Unknown heater target battery");
    const auto model =
        std::find_if(e["battery_models"].begin(), e["battery_models"].end(), [&](const auto& m) {
            return m.at("id") == instance->at("model_id") &&
                   m.at("revision") == instance->at("model_revision");
        });
    const bool cylinder = model->at("geometry").at("kind") == "cylinder";
    const bool wrap = face == "wrap";
    if ((wrap && !cylinder) || (!wrap && face != "+z" && face != "-z"))
        throw std::invalid_argument("Cylinder heater mounting must be wrap, +z or -z");
    if (!cylinder)
        throw std::invalid_argument("Wrap/disk heater requires a cylindrical battery");
    const auto scene = scenario_from_json(compile_experiment(e));
    const auto body =
        std::find_if(scene.cells.begin(), scene.cells.end(), [&](const auto& c) { return c.id == bid; });
    if (body->orientation_xyzw != std::array<double, 4>{0, 0, 0, 1})
        throw std::invalid_argument("Heater mounting currently requires an axis-aligned battery");
    auto heater = std::find_if(a["passive_cells"].begin(), a["passive_cells"].end(),
                               [&](const auto& c) { return c.at("id") == hid; });
    if (heater == a["passive_cells"].end())
        throw std::invalid_argument("Unknown heater");
    std::set<std::string> heater_ports;
    for (const auto& p : a["ports"])
        if (p.at("cell_id") == hid)
            heater_ports.insert(p.at("id"));
    J law;
    double old_area = 0.;
    for (const auto& edge : a["interfaces"]) {
        const auto pa = edge.at("port_a").template get<std::string>();
        const auto pb = edge.at("port_b").template get<std::string>();
        if (!heater_ports.contains(pa) && !heater_ports.contains(pb))
            continue;
        const auto port = std::find_if(scene.ports.begin(), scene.ports.end(), [&](const auto& p) {
            return p.id == (heater_ports.contains(pa) ? pa : pb);
        });
        if (port == scene.ports.end())
            throw std::invalid_argument("Missing heater contact port");
        if (law.is_null()) {
            law = edge.at("law");
            old_area = port->area_m2;
        }
        // A common per-area contact law is required, rather than silently flattening heterogeneity.
        auto normalized = edge.at("law"), reference = law;
        for (const char* key :
             {"conductance_w_per_k", "background_conductance_w_per_k", "active_conductance_w_per_k",
              "destroyed_background_conductance_w_per_k", "destroyed_active_conductance_w_per_k"}) {
            if (normalized.contains(key))
                normalized[key] = normalized[key].template get<double>() / port->area_m2;
            if (reference.contains(key))
                reference[key] = reference[key].template get<double>() / old_area;
        }
        for (const auto& [key, value] : reference.items()) {
            if (value.is_number() && normalized.contains(key) && normalized[key].is_number()) {
                if (std::abs(value.template get<double>() - normalized[key].template get<double>()) >
                    1e-9 * std::max(1., std::abs(value.template get<double>())))
                    throw std::invalid_argument(
                        "Different heater contact laws; cannot remount automatically");
            } else if (!normalized.contains(key) || normalized[key] != value)
                throw std::invalid_argument("Different heater contact laws; cannot remount automatically");
        }
    }
    if (law.is_null() || old_area <= 0)
        throw std::invalid_argument("Heater needs an existing thermal contact");
    const auto geometry = model->at("geometry").at("dimensions_m").template get<std::vector<double>>();
    const double shell =
        model->at("shell").is_null() ? 0. : model->at("shell").at("thickness_m").template get<double>();
    const double radius = geometry[0] + shell, height = geometry[1] + 2 * shell;
    const double pi = std::acos(-1.), area = wrap ? 2 * pi * radius * height : pi * radius * radius;
    constexpr double thickness = .0002;
    // A single lumped thermal reservoir is represented by a thin equal-volume
    // slab. It is not an extra solid cylinder occupying the battery's interior.
    // All contact surfaces below are explicit; renderer supplies the wrap/disk.
    const double side = wrap ? area / height : std::sqrt(area);
    std::array<double, 3> dims =
        wrap ? std::array<double, 3>{thickness, side, height} : std::array<double, 3>{side, side, thickness};
    std::array<double, 3> center{body->position_m.x, body->position_m.y, body->position_m.z};
    if (wrap)
        center[0] -= body->dimensions_m[0] / 2 + shell + thickness / 2;
    else
        center[2] += (face == "+z" ? 1 : -1) * (height / 2 + thickness / 2);
    // Folded reservoir stays compact; no unfolded sheet may block neighbouring gas routes.
    if (wrap)
        dims[1] = body->dimensions_m[1] + 2 * shell;
    (*heater)["geometry"] = {{"kind", "prismatic"}, {"dimensions_m", dims}};
    (*heater)["pose"] = {{"position_m", center}, {"orientation_xyzw", {0, 0, 0, 1}}};
    auto& edges = a["interfaces"];
    edges.erase(std::remove_if(edges.begin(), edges.end(),
                               [&](const auto& edge) {
                                   return heater_ports.contains(edge.at("port_a")) ||
                                          heater_ports.contains(edge.at("port_b"));
                               }),
                edges.end());
    auto& ports = a["ports"];
    ports.erase(
        std::remove_if(ports.begin(), ports.end(), [&](const auto& p) { return p.at("cell_id") == hid; }),
        ports.end());
    const std::vector<std::string> faces =
        wrap ? std::vector<std::string>{"+x", "-x", "+y", "-y"} : std::vector<std::string>{face};
    std::size_t index = 0;
    for (const auto& f : faces) {
        const auto target = std::find_if(scene.ports.begin(), scene.ports.end(), [&](const auto& p) {
            return p.id == bid + ":case:" + f + ":outer";
        });
        if (target == scene.ports.end())
            throw std::invalid_argument("Cylinder heater requires shell face ports");
        const std::string suffix = index == 0 ? "" : "-" + std::to_string(index);
        const std::string pid = hid + "-contact" + suffix;
        // All source ports address the same one-zone heater, while destinations
        // cover each entire shell face. Conductance owns the true cylinder area.
        const int axis = wrap ? 0 : 2;
        std::array<double, 3> p{}, normal{};
        p[axis] = dims[axis] / 2;
        normal[axis] = 1;
        const int u = (axis + 1) % 3, v = (axis + 2) % 3;
        const double contact_height = dims[v];
        const double contact_width = area / faces.size() / contact_height;
        ports.push_back({{"id", pid},
                         {"cell_id", hid},
                         {"frame", "local"},
                         {"surface", wrap ? "+x" : "+z"},
                         {"centroid_m", p},
                         {"normal", normal},
                         {"rectangle_m", {contact_width, contact_height}},
                         {"area_m2", area / faces.size()}});
        auto part = law;
        for (const char* key :
             {"conductance_w_per_k", "background_conductance_w_per_k", "active_conductance_w_per_k",
              "destroyed_background_conductance_w_per_k", "destroyed_active_conductance_w_per_k"})
            if (part.contains(key))
                part[key] = part[key].template get<double>() * area / faces.size() / old_area;
        edges.push_back({{"id", hid + "-coupling" + suffix},
                         {"port_a", pid},
                         {"port_b", target->id},
                         {"law", part},
                         {"lumped_heater", true}});
        ++index;
    }
    e["metadata"]["heater_mounts"][hid] = {{"battery_id", bid},  {"kind", wrap ? "wrap" : "disk"},
                                           {"face", face},       {"radius_m", radius},
                                           {"height_m", height}, {"film_thickness_m", thickness}};
    e["metadata"]["editor_changes"].push_back(
        {{"kind", "heater_mount"},
         {"id", hid},
         {"face", face},
         {"contact_area_m2", area},
         {"assumption", "One thermal reservoir; cylindrical-area conductance across aligned shell faces; no "
                        "extra power sources"}});
    e["provenance"]["status"] = "mixed";
    e["provenance"]["notes"] =
        e["provenance"].value("notes", std::string{}) + " Heater " + hid + " remounted as " +
        (wrap ? "lateral wrap" : "end disk") +
        "; cylindrical contact area changed, controller unchanged; requires a fresh run.";
    (void)build_thermal_mesh(scenario_from_json(compile_experiment(e)));
    return e;
}
} // namespace battery
