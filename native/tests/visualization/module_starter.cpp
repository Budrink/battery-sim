#include "battery_native/contracts/experiment_contract.hpp"
#include "battery_native/contracts/experiment_editing.hpp"
#include "battery_native/contracts/enclosure_openings.hpp"
#include "battery_native/scenario_document.hpp"
#include "battery_native/data/visual_frame.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <set>

void require(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}
#include "module_surface_law_checks.hpp"
#include "gap_geometry_checks.hpp"
#include "shell_grid_checks.hpp"
void opening_checks(const nlohmann::json& doc, const nlohmann::json& compiled) {
    using J = nlohmann::json;
    const double disk_area = std::acos(-1.) * .01 * .01;
    const auto holes = battery::parse_enclosure_openings(doc["assembly"]["gas_meshes"]["case"]);
    require(holes.size() == 1 && std::abs(holes[0].overlap({-.1, -.1}, {0, 0}) - disk_area / 4) < 1e-15,
            "Circle/rectangle area integral is incorrect");
    auto area = [](const J& scene) {
        double sum = 0;
        for (const auto& g : scene["gas_network"]["compartments"])
            if (g.contains("ambient_boundary"))
                sum += g["ambient_boundary"]["area_m2"].get<double>();
        return sum;
    };
    require(std::abs(area(compiled) - disk_area) < 1e-12,
            "Default aperture area lost at gas-cell boundaries");
    const auto& boxes = compiled["metadata"]["gas_volume_geometry"];
    for (const auto& g : compiled["gas_network"]["compartments"])
        if (g.contains("ambient_boundary")) {
            require(std::abs(boxes.at(g["id"]).at("high_m")[2].get<double>() - .04) < 1e-12,
                    "Opening attached to a remote gas cell");
            require(g["ambient_boundary"]["renewal_flow_m3_s"] == 0, "Hole invented forced ventilation");
        }
    const auto closed = battery::set_enclosure_openings(doc, "case", J::array());
    const auto closed_scene = battery::compile_experiment(closed);
    require(area(closed_scene) == 0, "Deleted opening still vents");
    const double contact_diff =
        closed_scene["metadata"]["resolved_gas_meshes"]["case"]["solid_contact_area_m2"].get<double>() -
        compiled["metadata"]["resolved_gas_meshes"]["case"]["solid_contact_area_m2"].get<double>();
    require(std::abs(contact_diff - disk_area) < 1e-12, "Removed plastic still exchanges heat with gas");
    auto capacity = [](const J& scene) {
        for (const auto& c : scene["cells"])
            if (c["id"] == "case-plusz")
                return c["thermal"]["heat_capacity_j_per_k"].get<double>();
        throw std::runtime_error("Missing lid");
    };
    require(std::abs(capacity(closed_scene) - capacity(compiled) - disk_area * .002 * 1200 * 1200) < 1e-10,
            "Removed material heat capacity is not accounted for");
    auto list = doc["assembly"]["gas_meshes"]["case"]["openings"];
    list.push_back({{"id", "rect"},
                    {"face", "+z"},
                    {"shape", "rectangle"},
                    {"center_m", {.03, .03}},
                    {"size_m", {.008, .012}}});
    list.push_back({{"id", "side"},
                    {"face", "-x"},
                    {"shape", "circle"},
                    {"center_m", {0., 0.}},
                    {"size_m", {.012, .012}}});
    auto multiple = battery::set_enclosure_openings(doc, "case", list);
    const double expected = disk_area + .008 * .012 + std::acos(-1.) * .006 * .006;
    for (double step : {.04, .02, .01}) {
        auto refined = multiple;
        refined["assembly"]["gas_meshes"]["case"]["maximum_cell_size_m"] = step;
        require(std::abs(area(battery::compile_experiment(refined)) - expected) < 1e-12,
                "Refinement changes aggregate aperture conductance area");
    }
    const auto& props = doc["metadata"]["editor_enclosures"]["case"]["properties"];
    require(battery::set_enclosure(multiple, "case", "cavity",
                                   props)["assembly"]["gas_meshes"]["case"]["openings"] == list,
            "Case update erased custom openings");
    auto invalid = list;
    invalid[1]["center_m"] = {0., 0.};
    bool rejected = false;
    try {
        (void)battery::set_enclosure_openings(doc, "case", invalid);
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "Overlapping holes accepted");
    invalid = list;
    invalid[1]["size_m"] = {1., 1.};
    rejected = false;
    try {
        (void)battery::set_enclosure_openings(doc, "case", invalid);
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "Out-of-face hole accepted");
    for (double pressure_ratio : {.5, 2.}) {
        auto scene = battery::scenario_from_json(compiled);
        scene.heating_protocols.clear();
        for (auto& cell : scene.cells)
            cell.electrical.current_a = 0;
        for (auto& gas : scene.gas_compartments) {
            gas.reaction_rate_per_s = 0;
            if (gas.ambient.enabled)
                gas.initial_mass_kg *= pressure_ratio;
        }
        scene.settings.duration_s = .01;
        battery::Simulation simulation(scene);
        simulation.run_to_end();
        const auto d = simulation.process_diagnostics();
        double inward = 0, outward = 0;
        for (const auto& g : d.gas) {
            inward += g.ambient_gross_in_kg;
            outward += g.ambient_gross_out_kg;
        }
        require(pressure_ratio < 1 ? inward > 0 : outward > 0,
                "Real case hole did not exchange mass with ambient");
        require(std::abs(d.mass_residual_kg) < 1e-12 && std::abs(d.oxidizer_residual_kg) < 1e-12 &&
                    std::abs(d.thermal_residual_j) < 1e-6,
                "Open case violates conservation ledgers");
    }
}
int main(int argc, char** argv) {
    try {
        require(argc == 2, "Expected module contract");
        nlohmann::json doc;
        std::ifstream(argv[1]) >> doc;
        const auto& a = doc.at("assembly");
        require(a.at("batteries").size() == 16 && a.at("passive_cells").size() == 31 &&
                    a.at("interfaces").size() == 49,
                "Wrong 4x4 layout counts");
        const auto compiled = battery::compile_experiment(doc);
        module_surface_law_checks(doc);
        gap_geometry_checks(doc);
        shell_grid_checks(doc);
        opening_checks(doc, compiled);
        auto s = battery::scenario_from_json(compiled);
        require(s.gas_model.mixing_diffusivity_m2_s > 0 && s.gas_model.mixing_flow_fraction > 0 &&
                    s.gas_model.thermal_stirring_coefficient > 0 && s.gas_model.joint_gas_surface_mixing,
                "Gas mixing or simultaneous surface heating silently disabled");
        for (std::size_t i = 0; i < s.cells.size(); ++i)
            for (std::size_t j = i + 1; j < s.cells.size(); ++j) {
                const auto& a = s.cells[i];
                const auto& b = s.cells[j];
                const double pa[] = {a.position_m.x, a.position_m.y, a.position_m.z};
                const double pb[] = {b.position_m.x, b.position_m.y, b.position_m.z};
                double overlap = 1;
                for (int d = 0; d < 3; ++d)
                    overlap *= std::max(
                        0., std::min(pa[d] + a.dimensions_m[d] / 2, pb[d] + b.dimensions_m[d] / 2) -
                                std::max(pa[d] - a.dimensions_m[d] / 2, pb[d] - b.dimensions_m[d] / 2));
                require(overlap < 1e-15, "Heater, casing or insulation overlaps another solid");
            }
        auto geometry = battery::visual_geometry_from_json(compiled, s);
        // CLI/GUI file startup must resolve exactly the same geometry as Run
        // Snapshot, not read the uncompiled contract after building physics.
        require(battery::load_scenario_document(argv[1]) == compiled,
                "File startup and compiled snapshot differ");
        const auto from_file = battery::load_visual_geometry(argv[1], s);
        require(!from_file.gas.empty() && from_file.unresolved_gas_regions == 0 &&
                    from_file.gas.size() == geometry.gas.size(),
                "Direct file startup loses generated gas geometry");
        // Two pads may share a material label; changing one must change its
        // actual conductivity, not the shared colour or any neighbouring pad.
        const auto pad_id = a.at("passive_cells")[0].at("id").get<std::string>();
        const auto without_pad = battery::delete_element(doc, pad_id);
        require(without_pad["assembly"]["passive_cells"].size() == 30 &&
                    without_pad["assembly"]["interfaces"].size() == 47,
                "Deleting a pad did not remove exactly its two thermal contacts");
        require(without_pad["battery_models"] == doc["battery_models"] &&
                    without_pad["assembly"]["batteries"] == a["batteries"],
                "Deleting a pad changed batteries/materials");
        const auto without_battery = battery::delete_element(doc, "B05");
        const auto deleted_scene = battery::scenario_from_json(battery::compile_experiment(without_battery));
        require(without_battery["assembly"]["batteries"].size() == 15 &&
                    a["gas_network"]["compartments"].size() -
                            without_battery["assembly"]["gas_network"]["compartments"].size() ==
                        27,
                "Deleting a battery must remove all 27 pore volumes");
        for (const auto& c : deleted_scene.cells)
            require(c.id != "B05" && c.shell_contents_id != "B05", "Deleted battery/shell survived");
        require(without_battery["conditions"]["heating_protocols"].empty(),
                "Deleted sensor left a heater controller");
        std::string heater_id;
        for (const auto& cell : a["passive_cells"])
            if (cell.value("material_id", "") == "external-resistive-heater")
                heater_id = cell.at("id");
        require(!heater_id.empty(), "Missing heater fixture");
        const auto without_heater = battery::delete_element(doc, heater_id);
        require(without_heater["assembly"]["passive_cells"].size() == 30 &&
                    without_heater["assembly"]["interfaces"].size() == 47,
                "Deleting heater must remove both its battery and pad contacts");
        const auto without_wall = battery::delete_element(doc, "case-plusx");
        require(without_wall["assembly"]["passive_cells"].size() == 30, "Deleting case wall failed");
        const auto duplicated_pad = battery::duplicate_element(doc, pad_id);
        require(duplicated_pad["assembly"]["passive_cells"].size() == 32 &&
                    duplicated_pad["assembly"]["interfaces"] == a["interfaces"] &&
                    duplicated_pad["assembly"]["passive_cells"].back()["thermal"] ==
                        a["passive_cells"][0]["thermal"],
                "Pad copy changed properties or invented neighbour contacts");
        const auto duplicated_battery = battery::duplicate_element(doc, "B05");
        require(duplicated_battery["assembly"]["gas_network"]["compartments"].size() -
                        a["gas_network"]["compartments"].size() ==
                    27,
                "Battery copy lost local pore geometry");
        const auto roundtrip = battery::delete_element(duplicated_battery, "B05_1");
        require(roundtrip["assembly"] == a && roundtrip["numerics"] == doc["numerics"],
                "Duplicate/delete battery did not restore physical topology");
        const auto conductive = battery::set_pad_conductivity_preset(doc, pad_id, "thermal_conductor");
        require(conductive.at("metadata").at("visual_materials") == doc.at("metadata").at("visual_materials"),
                "Pad preset mutates shared appearance");
        const auto& before_pad = a.at("passive_cells")[0];
        const auto& after_pad = conductive.at("assembly").at("passive_cells")[0];
        auto expected = before_pad;
        expected["thermal"]["conductivity_w_per_m_k"] = {2., 2., 2.};
        expected["material_id"] = "editor-pad-" + pad_id;
        require(after_pad == expected, "Pad preset changed capacity, geometry or damage unexpectedly");
        for (std::size_t i = 1; i < a.at("passive_cells").size(); ++i)
            require(conductive.at("assembly").at("passive_cells")[i] == a.at("passive_cells")[i],
                    "Pad preset changed a sibling");
        require(conductive.at("assembly").at("interfaces") == a.at("interfaces"),
                "Pad preset changed contact laws");
        const auto conductive_compiled = battery::compile_experiment(conductive);
        const auto conductive_scene = battery::scenario_from_json(conductive_compiled);
        const auto pad = std::find_if(conductive_scene.cells.begin(), conductive_scene.cells.end(),
                                      [&](const auto& c) { return c.id == pad_id; });
        require(pad != conductive_scene.cells.end() &&
                    pad->conductivity_w_per_m_k == std::array<double, 3>{2., 2., 2.},
                "Conductive pad preset did not reach solver properties");
        const auto conductive_visual =
            battery::visual_geometry_from_json(conductive_compiled, conductive_scene);
        require(std::count(conductive_visual.solid_materials.begin(), conductive_visual.solid_materials.end(),
                           battery::VisualMaterialKind::ThermalConductor) == 1 &&
                    std::count(conductive_visual.solid_materials.begin(),
                               conductive_visual.solid_materials.end(),
                               battery::VisualMaterialKind::Insulation) == 23,
                "Changing one pad recoloured siblings");
        const auto insulating = battery::set_pad_conductivity_preset(conductive, pad_id, "insulation");
        require(insulating.at("assembly").at("passive_cells")[0].at("thermal").at("conductivity_w_per_m_k") ==
                    nlohmann::json({.06, .06, .06}),
                "Insulation preset not restored");
        auto visual_only = compiled;
        visual_only["metadata"]["visual_objects"][pad_id] = "thermal_conductor";
        const auto local_visual = battery::visual_geometry_from_json(visual_only, s);
        require(std::count(local_visual.solid_materials.begin(), local_visual.solid_materials.end(),
                           battery::VisualMaterialKind::ThermalConductor) == 1,
                "Object appearance must override a shared material only for that object");
        require(std::count(geometry.solid_materials.begin(), geometry.solid_materials.end(),
                           battery::VisualMaterialKind::Heater) == 1,
                "Heater has no dedicated visible style");
        std::set<std::string> valves;
        for (const auto& ch : s.gas_channels)
            if (ch.rupture && ch.opening.valve_radius_m > 0) {
                require(ch.port_id.ends_with(":case:+z:outer") || ch.port_id.ends_with(":case:-z:outer"),
                        "Default valve is not on a lid");
                valves.insert(s.cells.at(ch.shell_object).id);
            }
        require(valves.size() == 32, "Every default battery must have top and bottom rupture valves");
        for (const auto& c : s.cells)
            if (!c.shell_contents_id.empty())
                require(c.damage_onset_k == (c.id.ends_with("z") ? 450. : 900.),
                        "Weak lid material leaked to side plates");
        require(s.primary_failures.empty(), "Demo must not inject scheduled battery failures");
        auto missing_pore = doc;
        auto& nodes = missing_pore["assembly"]["gas_network"]["compartments"];
        for (auto& node : nodes)
            if (node.at("id") == "inside-B05-22")
                node.erase("local_equilibrium");
        bool rejected_missing_pore = false;
        try {
            (void)battery::compile_experiment(missing_pore);
        } catch (const std::exception&) {
            rejected_missing_pore = true;
        }
        require(rejected_missing_pore, "Missing pore silently reroutes a valve through the whole battery");
        require(s.heating_protocols.size() == 1 && s.heating_protocols[0].sensor_id == "B05:case:+x" &&
                    s.heating_protocols[0].feedforward_w == 0 &&
                    s.heating_protocols[0].maximum_power_w == 30 &&
                    std::abs(s.heating_protocols[0].proportional_w_per_k - .19767037747050903) < 1e-15,
                "FSRI30 heater protocol changed");
        require(compiled.at("metadata").at("resolved_gas_meshes").at("case").at("cells").get<int>() > 16,
                "Case gas stayed lumped");
        require(geometry.unresolved_gas_regions == 0, "Generated free gas cannot be rendered");
        require(std::none_of(s.gas_compartments.begin(), s.gas_compartments.end(),
                             [](const auto& g) { return g.id == "cavity"; }),
                "Global mixing reservoir survived meshing");
        const auto& report = compiled.at("metadata").at("resolved_gas_meshes").at("case");
        const auto& boxes = compiled.at("metadata").at("gas_volume_geometry");
        double summed_volume = 0;
        std::array<double, 3> lattice_width{};
        require(report.at("layout") == "uniform_masked", "Case is not on a regular masked lattice");
        for (const auto& [id, b] : boxes.items()) {
            for (int d = 0; d < 3; ++d) {
                const double width = b.at("high_m")[d].get<double>() - b.at("low_m")[d].get<double>();
                if (lattice_width[d] == 0)
                    lattice_width[d] = width;
                require(std::abs(width - lattice_width[d]) < 1e-12, "Nonuniform gas lattice spacing");
            }
            double fluid_volume = 0.;
            for (const auto& fragment : b.at("fluid_fragments")) {
                const auto lo = fragment.at("low_m").get<std::array<double, 3>>(),
                           hi = fragment.at("high_m").get<std::array<double, 3>>();
                double vol = 1;
                for (int d = 0; d < 3; ++d) {
                    require(hi[d] - lo[d] <= .02 + 1e-12, "Gas cell exceeds requested maximum size");
                    vol *= hi[d] - lo[d];
                }
                summed_volume += vol;
                fluid_volume += vol;
                for (const auto& c : s.cells) {
                    const double pos[] = {c.position_m.x, c.position_m.y, c.position_m.z};
                    double overlap = 1;
                    for (int d = 0; d < 3; ++d)
                        overlap *= std::max(0., std::min(hi[d], pos[d] + c.dimensions_m[d] / 2) -
                                                    std::max(lo[d], pos[d] - c.dimensions_m[d] / 2));
                    require(overlap < 1e-15, "Gas intersects a solid or thin insulation barrier");
                }
            }
            for (const auto& gas : s.gas_compartments)
                if (gas.id == id)
                    require(std::abs(gas.volume_m3 - fluid_volume) < 1e-15, "Masked gas volume mismatch");
        }
        require(std::abs(summed_volume - report.at("free_volume_m3").get<double>()) < 1e-12,
                "Gas volume ledger mismatch");
        for (const auto& channel : s.gas_channels) {
            if (channel.from < 0 || channel.to < 0)
                continue;
            const auto& a = s.gas_compartments[channel.from].id;
            const auto& b = s.gas_compartments[channel.to].id;
            if (!boxes.contains(a) || !boxes.contains(b))
                continue;
            int adjacent_axes = 0;
            for (int d = 0; d < 3; ++d) {
                const double delta = std::abs(boxes.at(a).at("low_m")[d].get<double>() -
                                              boxes.at(b).at("low_m")[d].get<double>());
                if (delta > 1e-12) {
                    ++adjacent_axes;
                    require(std::abs(delta - lattice_width[d]) < 1e-12, "Gas skips a lattice cell");
                }
            }
            require(adjacent_axes <= 1, "Gas skips a lattice face");
            const auto& ba = boxes.at(a);
            const auto& bb = boxes.at(b);
            const int axis = channel.mixing_axis;
            const double distance = std::abs(ba.at("fluid_centroid_m")[axis].get<double>() -
                                             bb.at("fluid_centroid_m")[axis].get<double>());
            require(distance > 0 && std::abs(channel.mixing_length_m - distance) < 1e-12 &&
                        std::abs(channel.inertial_length_m - distance) < 1e-12,
                    "Gas transport uses empty lattice space instead of fluid distance");
            const bool touch =
                std::abs(ba.at("fluid_high_m")[axis].get<double>() - bb.at("fluid_low_m")[axis].get<double>()) < 1e-12 ||
                std::abs(bb.at("fluid_high_m")[axis].get<double>() - ba.at("fluid_low_m")[axis].get<double>()) < 1e-12;
            require(touch, "Gas connects nonadjacent subvolumes");
        }
        std::cout << "Uniform lattice: " << report.at("lattice_divisions") << ", " << report.at("cells")
                  << " fluid components, " << report.at("passages") << " open faces\n";
        auto coarse = doc;
        coarse["assembly"]["gas_meshes"]["case"]["maximum_cell_size_m"] = .04;
        const auto coarse_scene = battery::compile_experiment(coarse);
        const auto& cr = coarse_scene.at("metadata").at("resolved_gas_meshes").at("case");
        require(cr.at("cells").get<int>() < report.at("cells").get<int>(), "Mesh resolution has no effect");
        require(std::abs(cr.at("free_volume_m3").get<double>() - summed_volume) < 1e-12 &&
                    std::abs(cr.at("solid_contact_area_m2").get<double>() -
                             report.at("solid_contact_area_m2").get<double>()) < 1e-12,
                "Gas refinement changes free volume or solid contact area");
        for (const auto& channel : s.gas_channels)
            if (channel.rupture) {
                require(channel.to >= 0 && s.gas_compartments[channel.to].id.starts_with("case-air-"),
                        "Valve bypasses the local case grid");
                require(channel.shell_thermal_volume >= 0, "Valve is not tied to local shell damage");
                const auto& source = s.gas_compartments.at(channel.from);
                require(source.thermal_cell >= 0 &&
                            source.thermal_volume == (channel.port_id.ends_with(":case:+z:outer") ? 18 : 0) + channel.shell_thermal_volume,
                        "Lid vent bypasses the adjacent material pore");
            }
        require(std::count(geometry.solid_materials.begin(), geometry.solid_materials.end(),
                           battery::VisualMaterialKind::Insulation) == 24,
                "Insulation material metadata was lost");
        require(std::count(geometry.solid_materials.begin(), geometry.solid_materials.end(),
                           battery::VisualMaterialKind::Enclosure) == 6,
                "Missing case walls");
        const auto& settings = doc.at("metadata").at("editor_enclosures").at("case").at("properties");
        const auto updated = battery::set_enclosure(doc, "case", "cavity", settings);
        require(updated.at("conditions").at("boundaries").empty() && s.boundaries.empty(),
                "Adiabatic default case leaks heat to the environment");
        auto cooled_settings = settings;
        cooled_settings["ambient_h_w_m2_k"] = 5;
        const auto cooled = battery::set_enclosure(updated, "case", "cavity", cooled_settings);
        require(cooled.at("conditions").at("boundaries").size() == 6, "Cannot enable optional case cooling");
        const auto insulated = battery::set_enclosure(cooled, "case", "cavity", settings);
        require(insulated.at("conditions").at("boundaries").empty(),
                "Disabling cooling left stale boundaries");
        require(insulated.at("assembly").at("passive_cells") == updated.at("assembly").at("passive_cells"),
                "Cooling switch changes wall material/heat capacity");
        require(updated.at("assembly").at("passive_cells").size() == 31, "Case update duplicates walls");
        require(
            std::abs(
                updated.at("metadata").at("editor_enclosures").at("case").at("free_volume_m3").get<double>() -
                doc.at("metadata").at("editor_enclosures").at("case").at("free_volume_m3").get<double>()) <
                1e-12,
            "Case displaced-volume calculation differs from generator");
        for (const auto& interface : a.at("interfaces")) {
            auto get = [&](const char* key) -> const battery::PortConfig& {
                auto it = std::find_if(s.ports.begin(), s.ports.end(),
                                       [&](const auto& p) { return p.id == interface.at(key); });
                require(it != s.ports.end(), "Missing physical pad contact");
                return *it;
            };
            const auto& p = get("port_a");
            const auto& q = get("port_b");
            require(std::abs(p.centroid_m.x - q.centroid_m.x) + std::abs(p.centroid_m.y - q.centroid_m.y) +
                            std::abs(p.centroid_m.z - q.centroid_m.z) <
                        1e-12,
                    "Pad contact does not touch the casing");
            require(p.area_m2 <= q.area_m2 * (1 + 1e-12), "Pad exceeds casing face");
        }
        // At t=0 the ramp controller has zero error; include its next update.
        s.settings.duration_s = .2;
        battery::Simulation idle(s);
        idle.run_to_end();
        require(idle.joule_energy_j() > 0, "Default central heater is off");
        require(std::abs(idle.energy_balance_residual_j()) < 1e-7, "Module idle thermal balance failed");
        auto isolated = doc;
        isolated["conditions"]["heating_protocols"] = nlohmann::json::array();
        for (auto& c : isolated["assembly"]["passive_cells"])
            if (c.contains("electrical"))
                c["electrical"]["current_a"] = 0;
        auto heated = battery::add_external_heater(isolated, "B00", "-x", 5, 500, 5, false, .1);
        s = battery::scenario_from_json(battery::compile_experiment(heated));
        s.settings.duration_s = .1;
        battery::Simulation heating(s);
        heating.run_to_end();
        require(std::abs(heating.joule_energy_j() - .5) < 1e-7 &&
                    std::abs(heating.energy_balance_residual_j()) < 1e-7,
                "Module external heater balance failed");
        auto pressurized = battery::scenario_from_json(compiled);
        for (auto& gas : pressurized.gas_compartments)
            if (gas.thermal_cell >= 0 && pressurized.cells[gas.thermal_cell].id == "B05")
                gas.initial_mass_kg *= 10;
        pressurized.settings.duration_s = .02;
        battery::Simulation opening(pressurized);
        opening.run_to_end();
        require(std::abs(opening.energy_balance_residual_j()) < 1e-6,
                "Opening on generated grid loses energy");
        const auto process = opening.process_diagnostics();
        require(std::any_of(process.channels.begin(), process.channels.end(),
                            [](const auto& c) { return c.opening_fraction > 0 && c.transferred_kg > 0; }),
                "Pressure did not open a local patch and discharge gas");
        require(std::abs(process.mass_residual_kg) < 1e-12, "Local discharge loses gas mass");
        std::set<std::string> opened_lids;
        for (std::size_t k = 0; k < pressurized.gas_channels.size(); ++k) {
            const auto& channel = pressurized.gas_channels[k];
            if (!channel.rupture || channel.shell_object < 0 || process.channels[k].opening_fraction <= 0)
                continue;
            const auto& plate = pressurized.cells[channel.shell_object];
            require(plate.id.ends_with(":+z") || plate.id.ends_with(":-z"),
                    "Pressure rupture leaked from a lid into a side wall");
            if (plate.shell_contents_id == "B05")
                opened_lids.insert(plate.id);
        }
        require(opened_lids.size() == 2, "Pressurized battery did not open both declared lids");
        const auto& opening_states = opening.thermal_states();
        const auto& opening_mesh = opening.thermal_mesh();
        for (std::size_t v = 0; v < opening_states.size(); ++v) {
            const auto& cell = pressurized.cells[opening_mesh.volumes[v].object];
            if (cell.shell_contents_id == "B05" && !cell.id.ends_with("z"))
                require(opening_states[v].damage < 1e-12,
                        "Opening a lid damages the neighbouring side-wall tile");
        }
        // Isolated hot-gas pulse in the REAL generated cavity: no heater and
        // no R3. Initially equal pressure, so counterexchange must carry heat
        // to other volumes and the neighbouring shell without a forced jet.
        auto pulse_scene = battery::scenario_from_json(compiled);
        pulse_scene.heating_protocols.clear();
        for (auto& c : pulse_scene.cells)
            c.electrical.current_a = 0;
        int hot_gas = -1, shell = -1;
        for (const auto& contact : pulse_scene.gas_contacts)
            for (const auto& port : pulse_scene.ports)
                if (contact.port_id == port.id && port.cell_id == "B06:case:+z") {
                    hot_gas = contact.compartment;
                    shell = int(port.cell_index);
                    break;
                }
        require(hot_gas >= 0 && shell >= 0, "Neighbour shell has no gas contact");
        for (auto& g : pulse_scene.gas_compartments)
            g.reaction_rate_per_s = 0;
        auto& hot = pulse_scene.gas_compartments[hot_gas];
        hot.initial_mass_kg *= hot.initial_temperature_k / 800.;
        hot.initial_temperature_k = 800.;
        pulse_scene.settings.duration_s = .1;
        battery::Simulation pulse(pulse_scene);
        const double shell_initial = pulse.cells()[shell].stored_energy_j;
        pulse.run_to_end();
        const auto pulse_process = pulse.process_diagnostics();
        double exchanged = 0, neighbour_rise = 0;
        for (const auto& channel : pulse_process.channels)
            exchanged += channel.mixed_mass_kg;
        for (std::size_t i = 0; i < pulse_process.gas.size(); ++i)
            if (int(i) != hot_gas && pulse_scene.gas_compartments[i].thermal_cell < 0)
                neighbour_rise =
                    std::max(neighbour_rise, pulse_process.gas[i].temperature_k -
                                                 pulse_scene.gas_compartments[i].initial_temperature_k);
        const double shell_gain = pulse.cells()[shell].stored_energy_j - shell_initial;
        require(exchanged > 0 && neighbour_rise > .01 && shell_gain > 1e-7,
                "Restored mixing does not heat adjacent gas and shell");
        require(std::abs(pulse_process.mass_residual_kg) < 1e-12 &&
                    std::abs(pulse.energy_balance_residual_j()) < 1e-6,
                "Hot gas pulse violates mass or energy balance");
        battery::Simulation mc(pulse_scene);
        mc.step();
        const auto before_switch = mc.process_diagnostics();
        const double switch_time = mc.time_s();
        mc.set_colored_kawasaki(true);
        require(mc.time_s() == switch_time &&
                    mc.process_diagnostics().gas[hot_gas].mass_kg == before_switch.gas[hot_gas].mass_kg,
                "Live switch resets gas or time");
        const auto restart = mc.save_checkpoint("uniform-kawasaki-pulse");
        battery::Simulation resumed(pulse_scene);
        resumed.restore_checkpoint(restart, "uniform-kawasaki-pulse");
        require(resumed.colored_kawasaki(), "Full restart lost Kawasaki mode");
        mc.run_to_end();
        resumed.run_to_end();
        const auto mc_process = mc.process_diagnostics();
        require(std::abs(mc_process.mass_residual_kg) < 1e-12 &&
                    std::abs(mc.energy_balance_residual_j()) < 1e-6,
                "Uniform Kawasaki + films violates mass or energy balance");
        require(mc.cells()[shell].stored_energy_j > shell_initial + 1e-7,
                "Kawasaki silently disabled gas-solid heat exchange");
        require(mc.cells()[shell].stored_energy_j == resumed.cells()[shell].stored_energy_j,
                "Full Kawasaki restart changed trajectory");
        mc.set_colored_kawasaki(false);
        battery::Simulation mc_opening(pressurized);
        mc_opening.set_colored_kawasaki(true);
        mc_opening.run_to_end();
        const auto mc_open = mc_opening.process_diagnostics();
        std::cout << "Kawasaki rupture residuals: mass=" << mc_open.mass_residual_kg
                  << " energy=" << mc_opening.energy_balance_residual_j() << '\n';
        require(std::abs(mc_open.mass_residual_kg) < 1e-12 &&
                    std::abs(mc_opening.energy_balance_residual_j()) < 1e-6,
                "Kawasaki rupture + gas + chemistry balance failed");
        require(std::any_of(mc_open.channels.begin(), mc_open.channels.end(),
                            [](const auto& c) { return c.opening_fraction > 0 && c.transferred_kg > 0; }),
                "Kawasaki does not discharge through a broken barrier");
        auto no_mix_scene = pulse_scene;
        no_mix_scene.gas_model.mixing_diffusivity_m2_s = 0;
        no_mix_scene.gas_model.mixing_flow_fraction = 0;
        no_mix_scene.gas_model.thermal_stirring_coefficient = 0;
        battery::Simulation no_mix(no_mix_scene);
        no_mix.run_to_end();
        double no_mix_rise = 0;
        const auto no_mix_process = no_mix.process_diagnostics();
        for (std::size_t i = 0; i < no_mix_process.gas.size(); ++i)
            if (int(i) != hot_gas && no_mix_scene.gas_compartments[i].thermal_cell < 0)
                no_mix_rise =
                    std::max(no_mix_rise, no_mix_process.gas[i].temperature_k -
                                              no_mix_scene.gas_compartments[i].initial_temperature_k);
        require(neighbour_rise > no_mix_rise + .01,
                "Mixing has no effect compared with pressure-only transport");
        std::cout << "Hot gas pulse (0.1 s): each-way sum " << exchanged << " kg, neighbouring gas +"
                  << neighbour_rise << " K (mixing OFF: +" << no_mix_rise << " K), B06 cap +" << shell_gain
                  << " J\n";
        std::cout
            << "4x4: 24 pads, case, central heater, aligned contacts, visual materials, CUDA heater balance "
               "passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
