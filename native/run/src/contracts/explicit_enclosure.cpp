#include "battery_native/contracts/enclosure_openings.hpp"
#include "battery_native/contracts/experiment_contract.hpp"
#include "battery_native/scenario_document.hpp"
#include "explicit_enclosure_thermal.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>

namespace battery {
namespace {
using J = nlohmann::json;
using V = std::array<double, 3>;
std::string wall_id(const std::string& id, const EnclosureOpening& h) {
    return id + (h.sign > 0 ? "-plus" : "-minus") + "xyz"[h.axis];
}
void visual_openings(J& e, const std::string& id, const J& spec) {
    auto& list = e["metadata"]["case_openings"];
    if (list.is_null()) list = J::array();
    list.erase(std::remove_if(list.begin(), list.end(), [&](const J& h) {
        return h.at("case_id") == id;
    }), list.end());
    const auto origin = spec.at("position_m").get<V>();
    const auto size = spec.at("inside_dimensions_m").get<V>();
    for (const auto& h : parse_enclosure_openings(spec)) {
        const auto wid = wall_id(id, h);
        const auto& cells = e.at("assembly").at("passive_cells");
        const auto wall = std::find_if(cells.begin(), cells.end(), [&](const J& c) {return c.at("id") == wid;});
        if (wall == cells.end()) throw std::invalid_argument("Missing case wall: " + wid);
        auto position = origin;
        position[h.axis] += h.sign * (size[h.axis] / 2 + wall->at("geometry").at("dimensions_m")[h.axis].get<double>());
        position[(h.axis + 1) % 3] += h.center[0];
        position[(h.axis + 2) % 3] += h.center[1];
        list.push_back({{"id", h.id}, {"case_id", id}, {"wall_id", wid}, {"axis", h.axis},
                        {"sign", h.sign}, {"circle", h.circle}, {"center_m", position},
                        {"size_m", h.size}, {"area_m2", h.area()}});
    }
}
struct Patch {
    std::string gas;
    std::array<double, 2> lo{}, hi{};
};
// Partition the opening by actual gas-volume bounds, not wall tile rectangles:
// one thermal tile can contact several different gas volumes.
std::vector<Patch> patches(const J& boxes, int axis, int sign, double plane, const V& origin) {
    std::vector<Patch> result;
    for (const auto& [gas, box] : boxes.items()) {
        const auto lo = box.at("low_m").get<V>(), hi = box.at("high_m").get<V>();
        if (std::abs((sign > 0 ? hi[axis] : lo[axis]) - plane) > 1e-10) continue;
        Patch patch;
        patch.gas = gas;
        for (int d = 0; d < 2; ++d) {
            const int a = (axis + 1 + d) % 3;
            patch.lo[d] = lo[a] - origin[a];
            patch.hi[d] = hi[a] - origin[a];
        }
        result.push_back(patch);
    }
    return result;
}
} // namespace

J editable_enclosures(const J& e) {
    auto result = e.at("assembly").value("gas_meshes", J::object());
    const auto registry = e.value("metadata", J::object()).value("editor_enclosures", J::object());
    for (const auto& [id, entry] : registry.items())
        if (entry.contains("explicit_mesh")) result[id] = entry.at("explicit_mesh");
    return result;
}

J register_explicit_enclosure(const J& input, const std::string& id, const J& openings,
                              const J& route_ids, const J& passage_law) {
    auto e = input;
    const auto& cells = e.at("assembly").at("passive_cells");
    J walls = J::array();
    V low{}, high{};
    for (int axis = 0; axis < 3; ++axis)
        for (int sign : {-1, 1}) {
            const std::string wid = id + std::string(sign > 0 ? "-plus" : "-minus") + "xyz"[axis];
            const auto it = std::find_if(cells.begin(), cells.end(), [&](const J& c) {return c.at("id") == wid;});
            if (it == cells.end()) throw std::invalid_argument("Missing case wall: " + wid);
            const double edge = it->at("pose").at("position_m")[axis].get<double>() -
                sign * it->at("geometry").at("dimensions_m")[axis].get<double>() / 2;
            (sign > 0 ? high[axis] : low[axis]) = edge;
            walls.push_back(*it);
        }
    J routes = J::array();
    const auto requested = route_ids.get<std::set<std::string>>();
    for (const auto& ch : e.at("assembly").at("gas_network").at("channels"))
        if (requested.contains(ch.at("id"))) {
            routes.push_back(ch);
        }
    if (routes.size() != requested.size()) throw std::invalid_argument("Missing original opening route");
    V size{}, center{};
    for (int d = 0; d < 3; ++d) { size[d] = high[d] - low[d]; center[d] = (high[d] + low[d]) / 2; }
    J spec = {{"inside_dimensions_m", size}, {"position_m", center},
               {"discharge_coefficient", passage_law.at("discharge_coefficient")}, {"openings", openings}};
    (void)parse_enclosure_openings(spec);
    J contacts = J::array();
    std::set<std::string> wall_ports;
    for (const auto& p : e.at("assembly").at("ports"))
        for (const auto& w : walls)
            if (p.at("cell_id") == w.at("id")) wall_ports.insert(p.at("id"));
    for (const auto& c : e.at("assembly").at("gas_network").at("thermal_contacts"))
        if (wall_ports.contains(c.value("port_id", ""))) contacts.push_back(c);
    // Immutable baseline for reversible edits; opening/closing must not compound
    // capacity or conductance reductions. Raw contract remains the runtime source.
    auto& entry = e["metadata"]["editor_enclosures"][id];
    entry = {{"explicit_mesh", spec}, {"reference_walls", walls}, {"reference_contacts", contacts},
              {"reference_routes", routes}, {"reference_openings", spec.at("openings")},
              {"owned_channels", J::array()}, {"passage_law", passage_law}};
    const auto dims = walls[0].at("geometry").at("dimensions_m").get<V>();
    const auto& thermal = walls[0].at("thermal");
    // Only volumetric capacity is identified by the archived model. Do not
    // invent a measured density/specific-heat split for this imported case.
    entry["properties"] = {{"inside_dimensions_m", size}, {"position_m", center},
        {"thickness_m", dims[0]}, {"material_id", walls[0].at("material_id")},
        {"volumetric_heat_capacity_j_m3_k", thermal.at("heat_capacity_j_per_k").get<double>() / (dims[0]*dims[1]*dims[2])},
        {"conductivity_w_per_m_k", thermal.at("conductivity_w_per_m_k")},
        {"initial_temperature_k", thermal.at("initial_temperature_k")}};
    for (const auto& ch : routes) entry["owned_channels"].push_back(ch.at("id"));
    e["metadata"]["visual_materials"][walls[0].at("material_id").get<std::string>()] = "enclosure";
    visual_openings(e, id, spec);
    return e; // No physical field, route, mass, material or controller changed.
}

J edit_explicit_enclosure_properties(const J& input, const std::string& id, const J& p) {
    auto e = input;
    auto& entry = e["metadata"]["editor_enclosures"][id];
    const auto before = entry.at("properties");
    for (const auto* key : {"inside_dimensions_m", "position_m", "thickness_m"})
        if (p.at(key) != before.at(key))
            throw std::invalid_argument("Imported case geometry is bound to its explicit gas graph");
    const double capacity = p.at("volumetric_heat_capacity_j_m3_k"), temperature = p.at("initial_temperature_k");
    const auto k = p.at("conductivity_w_per_m_k").get<V>();
    if (!std::isfinite(capacity) || capacity <= 0 || !std::isfinite(temperature) || temperature <= 0 ||
        std::any_of(k.begin(), k.end(), [](double v){return !std::isfinite(v) || v <= 0;}))
        throw std::invalid_argument("Case heat capacity, temperature and conductivity must be positive");
    const double ratio = capacity / before.at("volumetric_heat_capacity_j_m3_k").get<double>();
    auto update = [&](J& wall) {
        wall["material_id"] = p.at("material_id");
        auto& t = wall["thermal"];
        t["heat_capacity_j_per_k"] = t.at("heat_capacity_j_per_k").get<double>() * ratio;
        t["conductivity_w_per_m_k"] = k;
        t["initial_temperature_k"] = temperature;
    };
    for (auto& ref : entry["reference_walls"]) {
        for (auto& wall : e["assembly"]["passive_cells"]) if (wall.at("id") == ref.at("id")) update(wall);
        update(ref);
    }
    entry["properties"] = p;
    e["metadata"]["visual_materials"][p.at("material_id").get<std::string>()] = "enclosure";
    e["provenance"]["status"] = "mixed";
    (void)compile_experiment(e);
    return e;
}

J edit_explicit_enclosure_openings(const J& input, const std::string& id, const J& openings) {
    auto e = input;
    auto& entry = e["metadata"]["editor_enclosures"][id];
    auto& spec = entry.at("explicit_mesh");
    if (spec.at("openings") == openings) return e;
    spec["openings"] = openings;
    const auto holes = parse_enclosure_openings(spec);
    const auto scene = scenario_from_json(compile_experiment(input));
    const auto origin = spec.at("position_m").get<V>();
    auto& net = e["assembly"]["gas_network"];
    const auto owned = entry.at("owned_channels").get<std::set<std::string>>();
    auto& channels = net["channels"];
    channels.erase(std::remove_if(channels.begin(), channels.end(), [&](const J& ch) {
        return owned.contains(ch.at("id"));
    }), channels.end());
    entry["owned_channels"] = J::array();
    // Restoring the imported opening restores its exact historical transport too.
    const bool restore = openings == entry.at("reference_openings");
    if (restore) {
        for (const auto& ch : entry.at("reference_routes")) {
            channels.push_back(ch);
            entry["owned_channels"].push_back(ch.at("id"));
        }
    } else for (const auto& h : holes) {
        const auto wid = wall_id(id, h);
        const auto wall = std::find_if(scene.cells.begin(), scene.cells.end(), [&](const auto& c) {return c.id == wid;});
        if (wall == scene.cells.end()) throw std::invalid_argument("Missing case wall: " + wid);
        const auto& boxes = e.at("metadata").at("gas_volume_geometry");
        const double plane = origin[h.axis] + h.sign * spec.at("inside_dimensions_m")[h.axis].get<double>() / 2;
        const auto inside = patches(boxes, h.axis, h.sign, plane, origin);
        const auto outside = patches(boxes, h.axis, -h.sign, plane + h.sign * wall->dimensions_m[h.axis], origin);
        std::map<std::pair<std::string, std::string>, double> pairs;
        double total = 0;
        for (const auto& a : inside) for (const auto& b : outside) {
            const double overlap = h.overlap({std::max(a.lo[0], b.lo[0]), std::max(a.lo[1], b.lo[1])},
                                             {std::min(a.hi[0], b.hi[0]), std::min(a.hi[1], b.hi[1])});
            if (overlap <= 1e-16) continue;
            if (a.gas == b.gas) throw std::invalid_argument("Same gas volume on both sides of case wall");
            pairs[{a.gas, b.gas}] += overlap;
            total += overlap;
        }
        if (std::abs(total - h.area()) > std::max(1e-12, h.area() * 1e-7))
            throw std::invalid_argument("Incomplete gas contact coverage at opening " + h.id +
                ": assigned=" + std::to_string(total) + ", expected=" + std::to_string(h.area()) +
                ", inner patches=" + std::to_string(inside.size()) + ", outer patches=" + std::to_string(outside.size()));
        int n = 0;
        for (const auto& [pair, area] : pairs) {
            auto ch = entry.at("passage_law");
            ch["id"] = id + "-opening-" + h.id + "-" + std::to_string(n++);
            ch["from"] = pair.first; ch["to"] = pair.second;
            ch["area_m2"] = area; ch["mixing_axis"] = h.axis;
            ch["mixing_length_m"] = wall->dimensions_m[h.axis];
            ch["discharge_coefficient"] = spec.at("discharge_coefficient");
            channels.push_back(ch);
            entry["owned_channels"].push_back(ch.at("id"));
        }
    }
    // Keep the archived model byte-for-byte until an actual opening edit. That
    // model retained solid behind the hole; edited walls remove its thermal area.
    explicit_case_detail::rebuild_contacts(e, id, scene, restore);
    for (const auto& ref : entry.at("reference_walls"))
        for (auto& wall : e["assembly"]["passive_cells"]) if (wall.at("id") == ref.at("id")) {
            double fraction = 1;
            if (!restore) for (const auto& h : holes) if (wall.at("id") == wall_id(id, h)) {
                const auto dims = wall.at("geometry").at("dimensions_m").get<V>();
                fraction -= h.area() / (dims[(h.axis+1)%3]*dims[(h.axis+2)%3]);
            }
            if (fraction <= 1e-8) throw std::invalid_argument("Opening removes the whole wall");
            wall["thermal"]["heat_capacity_j_per_k"] = ref.at("thermal").at("heat_capacity_j_per_k").get<double>() * fraction;
        }
    visual_openings(e, id, spec);
    e["provenance"]["status"] = "mixed";
    (void)compile_experiment(e);
    return e;
}
} // namespace battery
