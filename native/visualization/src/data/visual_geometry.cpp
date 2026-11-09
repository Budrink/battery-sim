#include "battery_native/data/visual_frame.hpp"
#include "battery_native/scenario_document.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <fstream>
#include <limits>
#include <map>

namespace battery {
VisualGeometry load_visual_geometry(const std::string& path, const Scenario& scenario) {
    return visual_geometry_from_json(load_scenario_document(path), scenario);
}
VisualGeometry visual_geometry_from_json(const nlohmann::json& root, const Scenario& scenario) {
    const auto meta = root.value("metadata", nlohmann::json::object());
    const auto boxes = meta.value("gas_volume_geometry", nlohmann::json::object());
    const auto reconstruction = meta.value("geometry_reconstruction", nlohmann::json::object());
    const auto polygons = reconstruction.value("polygon_xy_m", nlohmann::json::array());
    VisualGeometry result;
    for (const auto& h : meta.value("case_openings", nlohmann::json::array())) {
        CaseOpeningVisual hole;
        const auto wall = std::find_if(scenario.cells.begin(), scenario.cells.end(),
                                       [&](const auto& c) { return c.id == h.at("wall_id"); });
        if (wall == scenario.cells.end())
            throw std::runtime_error("Unknown opening wall");
        hole.wall = int(wall - scenario.cells.begin());
        hole.id = h.at("id");
        hole.case_id = h.at("case_id");
        hole.axis = h.at("axis");
        hole.sign = h.at("sign");
        hole.circle = h.at("circle");
        const auto p = h.at("center_m").get<std::array<double, 3>>();
        hole.center = {p[0], p[1], p[2]};
        hole.size = h.at("size_m").get<std::array<double, 2>>();
        result.case_openings.push_back(hole);
    }
    result.solid_proxies = scenario.cells;
    result.hidden_planar_shells.resize(scenario.cells.size(), false);
    result.solid_materials.resize(scenario.cells.size(), VisualMaterialKind::Unspecified);
    result.heater_shapes.resize(scenario.cells.size(), 0);
    const auto materials = meta.value("visual_materials", nlohmann::json::object());
    const auto objects = meta.value("visual_objects", nlohmann::json::object());
    for (std::size_t i = 0; i < scenario.cells.size(); ++i) {
        const auto& cell = scenario.cells[i];
        if (cell.is_battery || !cell.shell_contents_id.empty())
            continue;
        if (!objects.contains(cell.id) &&
            (cell.material_id == "external-resistive-heater" || cell.material_id == "heater" ||
             std::any_of(scenario.heating_protocols.begin(), scenario.heating_protocols.end(),
                         [&](const auto& p) { return p.heater_id == cell.id; }))) {
            result.solid_materials[i] = VisualMaterialKind::Heater;
            continue;
        }
        if (!objects.contains(cell.id) && !materials.contains(cell.material_id))
            continue;
        const auto kind = (objects.contains(cell.id) ? objects.at(cell.id) : materials.at(cell.material_id))
                              .get<std::string>();
        if (kind == "insulation")
            result.solid_materials[i] = VisualMaterialKind::Insulation;
        else if (kind == "thermal_conductor")
            result.solid_materials[i] = VisualMaterialKind::ThermalConductor;
        else if (kind == "enclosure")
            result.solid_materials[i] = VisualMaterialKind::Enclosure;
        else if (kind == "heater")
            result.solid_materials[i] = VisualMaterialKind::Heater;
        else if (kind != "unspecified")
            throw std::runtime_error("Unknown visual material kind: " + kind);
    }
    const auto cylinders = meta.value("packed_cylinder_geometry", nlohmann::json::object());
    const auto contract = root.value("format", "") == "battery-sim.experiment"
                              ? root
                              : meta.value("experiment_contract", nlohmann::json::object());
    if (contract.contains("assembly") && contract.contains("battery_models")) {
        for (const auto& instance : contract.at("assembly").at("batteries"))
            for (const auto& model : contract.at("battery_models"))
                if (model.at("id") == instance.at("model_id") &&
                    model.at("revision") == instance.at("model_revision") &&
                    model.at("geometry").at("kind") == "cylinder")
                    for (auto& cell : result.solid_proxies)
                        if (cell.id == instance.at("id")) {
                            cell.shape = ShapeKind::Cylinder;
                            cell.dimensions_m =
                                model.at("geometry").at("dimensions_m").get<std::vector<double>>();
                            // Renderer uploads a vec3 even for cylinders (radius, height, unused).
                            cell.dimensions_m.resize(3, 0.);
                        }
        for (std::size_t i = 0; i < scenario.cells.size(); ++i)
            for (const auto& cell : result.solid_proxies)
                if (cell.shape == ShapeKind::Cylinder && scenario.cells[i].shell_contents_id == cell.id)
                    result.hidden_planar_shells[i] = true;
    }
    if (contract.empty() && cylinders.contains("diameter_m") && cylinders.contains("height_m")) {
        const double diameter = cylinders.at("diameter_m"), height = cylinders.at("height_m");
        if (!std::isfinite(diameter) || !std::isfinite(height) || diameter <= 0 || height <= 0)
            throw std::runtime_error("Invalid visual cylinder dimensions");
        for (auto& cell : result.solid_proxies)
            if (cell.is_battery) {
                cell.shape = ShapeKind::Cylinder;
                cell.dimensions_m = {diameter / 2, height, 0};
            }
        for (std::size_t i = 0; i < scenario.cells.size(); ++i) {
            const auto& shell = scenario.cells[i];
            result.hidden_planar_shells[i] =
                !shell.shell_contents_id.empty() &&
                std::any_of(result.solid_proxies.begin(), result.solid_proxies.end(), [&](const auto& c) {
                    return c.is_battery && c.shape == ShapeKind::Cylinder && c.id == shell.shell_contents_id;
                });
        }
    }
    const auto heater_mounts = meta.value("heater_mounts", nlohmann::json::object());
    for (const auto& [id, mount] : heater_mounts.items()) {
        const auto heater = std::find_if(result.solid_proxies.begin(), result.solid_proxies.end(),
                                         [&](const auto& c) { return c.id == id; });
        const auto body = std::find_if(result.solid_proxies.begin(), result.solid_proxies.end(),
                                       [&](const auto& c) { return c.id == mount.at("battery_id"); });
        if (heater == result.solid_proxies.end() || body == result.solid_proxies.end())
            continue;
        const bool wrap = mount.at("kind") == "wrap";
        const double radius = mount.at("radius_m"), height = mount.at("height_m");
        const double film = mount.at("film_thickness_m");
        if (radius <= 0 || height <= 0 || film <= 0)
            throw std::runtime_error("Invalid heater mount dimensions");
        const auto index = std::size_t(heater - result.solid_proxies.begin());
        result.heater_shapes[index] = wrap ? 1 : 2;
        result.solid_materials[index] = VisualMaterialKind::Heater;
        heater->shape = ShapeKind::Cylinder;
        heater->dimensions_m = {radius + film / 2, wrap ? height : film, 0};
        heater->position_m = body->position_m;
        heater->orientation_xyzw = body->orientation_xyzw;
        if (!wrap)
            heater->position_m.z += (mount.at("face") == "+z" ? 1 : -1) * (height + film) / 2;
    }
    for (std::size_t i = 0; i < scenario.gas_compartments.size(); ++i) {
        const auto& node = scenario.gas_compartments[i];
        // Interior mobile inventory shares a solid numerical cell; do not draw
        // it a second time as an exterior plume.
        if (node.owner_cell >= 0 || node.thermal_cell >= 0)
            continue;
        GasVisualRegion region;
        region.compartment = int(i);
        if (boxes.contains(node.id)) {
            auto lo = boxes.at(node.id).at("low_m").get<std::array<double, 3>>();
            auto hi = boxes.at(node.id).at("high_m").get<std::array<double, 3>>();
            region.center = {(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2};
            region.size = {hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]};
        } else if (!polygons.empty() && (node.id.starts_with("body-") || node.id.starts_with("top-") ||
                                         node.id.starts_with("base-"))) {
            const int n = std::stoi(node.id.substr(node.id.find('-') + 1));
            if (n < 0 || n >= int(polygons.size()))
                throw std::runtime_error("Invalid gas polygon index");
            const std::string id = "B" + (n < 10 ? std::string("0") : std::string{}) + std::to_string(n);
            auto cell = std::find_if(scenario.cells.begin(), scenario.cells.end(),
                                     [&](auto& c) { return c.id == id; });
            if (cell == scenario.cells.end())
                throw std::runtime_error("Missing gas polygon battery");
            double xmin = 1e10, xmax = -1e10, ymin = 1e10, ymax = -1e10;
            for (const auto& p : polygons[n]) {
                const double x = p[0], y = p[1];
                region.polygon.push_back({float(x), float(y)});
                xmin = std::min(xmin, x);
                xmax = std::max(xmax, x);
                ymin = std::min(ymin, y);
                ymax = std::max(ymax, y);
            }
            // Dimensions are read from the reconstructed scene, not another
            // visual geometry fit. Use the actual outer can and ABS inner face.
            auto outer = cell->dimensions_m;
            double floor = 0, roof = 0;
            for (const auto& c : scenario.cells) {
                if (c.shell_contents_id == id) {
                    outer[0] = std::max(outer[0], 2 * std::abs(c.position_m.x - cell->position_m.x) +
                                                      c.dimensions_m[0]);
                    outer[1] = std::max(outer[1], 2 * std::abs(c.position_m.y - cell->position_m.y) +
                                                      c.dimensions_m[1]);
                    outer[2] = std::max(outer[2], 2 * std::abs(c.position_m.z - cell->position_m.z) +
                                                      c.dimensions_m[2]);
                }
                if (c.id == "ABS:+z")
                    roof = c.position_m.z - c.dimensions_m[2] / 2;
                if (c.id == "ABS:-z")
                    floor = c.position_m.z + c.dimensions_m[2] / 2;
            }
            double low = cell->position_m.z - outer[2] / 2, high = cell->position_m.z + outer[2] / 2;
            if (node.id.starts_with("top-")) {
                low = high;
                high = roof;
            } else if (node.id.starts_with("base-")) {
                high = low;
                low = floor;
            } else {
                region.excluded_center = cell->position_m;
                region.excluded_size = {outer[0], outer[1], outer[2]};
            }
            if (high <= low)
                throw std::runtime_error("Invalid gas cavity height");
            region.center = {(xmin + xmax) / 2, (ymin + ymax) / 2, (low + high) / 2};
            region.size = {xmax - xmin, ymax - ymin, high - low};
        } else {
            ++result.unresolved_gas_regions;
            continue;
        }
        for (const auto& c : result.solid_proxies)
            if (c.is_battery && c.shape == ShapeKind::Cylinder) {
                // This packed fixture is axial. Exclude the real circular cross
                // section, not its solver's equal-volume square, from the plume.
                if (std::abs(c.orientation_xyzw[0]) + std::abs(c.orientation_xyzw[1]) > 1e-8)
                    throw std::runtime_error("Tilted visual cylinder gas exclusion unsupported");
                const double r = c.dimensions_m[0], h = c.dimensions_m[1];
                if (std::abs(c.position_m.x - region.center.x) <= region.size.x / 2 + r &&
                    std::abs(c.position_m.y - region.center.y) <= region.size.y / 2 + r &&
                    std::abs(c.position_m.z - region.center.z) <= region.size.z / 2 + h / 2)
                    region.excluded_cylinders.push_back({c.position_m, r, h});
            }
        if (!region.excluded_cylinders.empty())
            region.excluded_size = {};
        if (boxes.contains(node.id) && boxes.at(node.id).contains("fluid_fragments")) {
            // A partially occupied lattice slot draws only its fluid support,
            // never a fog-filled battery or a visual bridge through a pad.
            for (const auto& fragment : boxes.at(node.id).at("fluid_fragments")) {
                auto piece = region;
                const auto lo = fragment.at("low_m").get<std::array<double, 3>>();
                const auto hi = fragment.at("high_m").get<std::array<double, 3>>();
                piece.center = {(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2};
                piece.size = {hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]};
                result.gas.push_back(std::move(piece));
            }
        } else
            result.gas.push_back(std::move(region));
    }
    // Same bounds for generated and imported explicit cases. Cell size is not
    // a criterion for whether gas is inside the case.
    auto cases = contract.value("assembly", nlohmann::json::object())
                         .value("gas_meshes", nlohmann::json::object());
    const auto case_registry = meta.value("editor_enclosures", nlohmann::json::object());
    for (const auto& [id, entry] : case_registry.items())
        if (entry.contains("explicit_mesh")) cases[id] = entry.at("explicit_mesh");
    std::vector<VisualBox> interiors;
    for (const auto& [id, spec] : cases.items()) {
        const auto center = spec.at("position_m").get<std::array<double,3>>();
        const auto size = spec.at("inside_dimensions_m").get<std::array<double,3>>();
        VisualBox box;
        for (int k=0;k<3;++k) {box.low[k]=center[k]-size[k]/2;box.high[k]=center[k]+size[k]/2;}
        interiors.push_back(box);
    }
    // Older native recordings have six explicit wall objects but no editor
    // descriptor. Recover bounds from those planes, never from gas-cell size.
    if (interiors.empty()) {
        std::map<std::string, std::array<const CellConfig*, 6>> walls;
        for (const auto& cell : scenario.cells) {
            if (cell.is_battery || !cell.shell_contents_id.empty() || cell.shape != ShapeKind::Prismatic)
                continue;
            for (int axis=0; axis<3; ++axis) for (int sign=0; sign<2; ++sign)
                for (const auto& suffix : {std::string(sign ? ":+" : ":-")+"xyz"[axis],
                                          std::string(sign ? "-plus" : "-minus")+"xyz"[axis]})
                    if (cell.id.ends_with(suffix))
                        walls[cell.id.substr(0,cell.id.size()-suffix.size())][2*axis+sign]=&cell;
        }
        for (const auto& [id, faces] : walls) {
            if (std::any_of(faces.begin(),faces.end(),[](auto* p){return !p;})) continue;
            VisualBox box; bool valid=true;
            for (int axis=0;axis<3;++axis) {
                const auto& a=*faces[2*axis]; const auto& b=*faces[2*axis+1];
                const double pa[]={a.position_m.x,a.position_m.y,a.position_m.z};
                const double pb[]={b.position_m.x,b.position_m.y,b.position_m.z};
                box.low[axis]=pa[axis]+a.dimensions_m[axis]/2;
                box.high[axis]=pb[axis]-b.dimensions_m[axis]/2;
                valid &= box.high[axis]>box.low[axis] &&
                         a.orientation_xyzw==std::array<double,4>{0,0,0,1} &&
                         b.orientation_xyzw==std::array<double,4>{0,0,0,1};
            }
            if(valid)interiors.push_back(box);
        }
    }
    for (auto& gas : result.gas) clip_gas_to_cases(gas, interiors);
    return result;
}
VisualShellPoint map_visual_shell_point(const Scenario& s, const VisualGeometry& g, int shell,
                                        Vec3 p, Vec3 normal, double offset) {
    const auto finish = [&]() -> VisualShellPoint {
        return {{p.x + normal.x * offset, p.y + normal.y * offset, p.z + normal.z * offset}, normal};
    };
    if (shell < 0 || g.solid_proxies.empty()) return finish();
    const auto& panel = s.cells.at(shell);
    const auto parent = std::find_if(g.solid_proxies.begin(), g.solid_proxies.end(), [&](const auto& c) {
        return c.id == panel.shell_contents_id && c.is_battery && c.shape == ShapeKind::Cylinder;
    });
    if (parent == g.solid_proxies.end())
        return finish();
    const double x = p.x - parent->position_m.x, y = p.y - parent->position_m.y;
    const double radius = parent->dimensions_m[0], height = parent->dimensions_m[1];
    const auto& d = panel.dimensions_m;
    // Concentric square-to-disk map: no collapsed corner triangles on the
    // lids, and the identical boundary map wraps the four lateral sectors.
    // Numerical shell coordinates and fracture areas remain untouched.
    const bool cap = d[2] < d[0] && d[2] < d[1];
    const double half_x = cap ? d[0] / 2 : radius;
    const double half_y = cap ? d[1] / 2 : radius;
    double a = std::clamp(x / half_x, -1.0, 1.0);
    double b = std::clamp(y / half_y, -1.0, 1.0);
    if (!cap) {
        if (d[0] < d[1]) {
            a = panel.position_m.x > parent->position_m.x ? 1 : -1;
            b = std::clamp(y / (d[1] / 2), -1.0, 1.0);
        } else {
            b = panel.position_m.y > parent->position_m.y ? 1 : -1;
            a = std::clamp(x / (d[0] / 2), -1.0, 1.0);
        }
    }
    double dx = 0, dy = 0;
    constexpr double quarter_pi = 0.7853981633974483;
    if (std::abs(a) > 1e-14 || std::abs(b) > 1e-14) {
        const bool x_major = std::abs(a) > std::abs(b);
        const double r = x_major ? a : b;
        const double angle = x_major ? quarter_pi * b / a : 2 * quarter_pi - quarter_pi * a / b;
        dx = r * std::cos(angle);
        dy = r * std::sin(angle);
    }
    p.x = parent->position_m.x + radius * dx;
    p.y = parent->position_m.y + radius * dy;
    if (cap) {
        normal = {0, 0, panel.position_m.z > parent->position_m.z ? 1.0 : -1.0};
        p.z = parent->position_m.z + normal.z * height / 2;
    } else {
        normal = {dx, dy, 0};
        p.z = std::clamp(p.z, parent->position_m.z - height / 2, parent->position_m.z + height / 2);
    }
    return finish();
}
Vec3 project_visual_shell_point(const Scenario& s, const VisualGeometry& g, int shell, Vec3 p) {
    return map_visual_shell_point(s, g, shell, p, {}).position;
}
} // namespace battery
