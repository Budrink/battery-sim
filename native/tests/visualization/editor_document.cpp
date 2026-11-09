#include "battery_ui/editor_document.hpp"
#include "battery_ui/preview_updates.hpp"
#include "battery_ui/opening_selection.hpp"
#include "battery_native/contracts/enclosure_openings.hpp"
#include "battery_native/contracts/experiment_contract.hpp"
#include "battery_native/contracts/experiment_editing.hpp"
#include "battery_native/scenario_document.hpp"
#include "battery_native/data/visual_frame.hpp"
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <unistd.h>
using J = nlohmann::json;
void require(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}
int main(int argc, char** argv) {
    try {
        if (argc != 2 && argc != 3)
            throw std::runtime_error("Expected experiment [cylinder-preview-output]");
        battery::ui::EditorDocument doc;
        doc.load(argv[1]);
        const auto original = doc.draft;
        battery::ui::PreviewUpdates updates;
        require(updates.observe(original, 0) && updates.ready(0), "Initial preview not automatic");
        updates.completed(original);
        auto edited = original;
        edited["name"] = "Changed";
        require(updates.observe(edited, 1) && !updates.ready(1), "Edit not throttled");
        edited["name"] = "Changed again";
        updates.observe(edited, 1.1);
        require(updates.ready(1.16), "Continuous editing postponed preview indefinitely");
        updates.attempted(); // including a rejected intermediate draft
        require(!updates.observe(edited, 2) && !updates.ready(2), "Invalid preview retried every frame");
        require(updates.observe(original, 3) && updates.ready(3.16), "Undo did not refresh preview");
        updates.reset();
        require(updates.observe(original, 4) && updates.ready(4), "Import did not reset preview");
        edited = original;
        edited["metadata"]["visual_objects"]["pad"] = "thermal_conductor";
        require(battery::ui::appearance_only_change(original, edited), "Appearance edit recompiles physics");
        edited["conditions"]["ambient_temperature_k"] = 300;
        require(!battery::ui::appearance_only_change(original, edited), "Physical edit bypasses compilation");
        const std::string bid = original.at("assembly").at("batteries")[0].at("id");
        for (bool ramp_control : {false, true}) {
            const auto heated = battery::add_external_heater(original, bid, "+x", 7, 500, 5,
                                                              ramp_control, .1, 423.15, 0., true);
            const auto id = heated["assembly"]["passive_cells"].back().at("id").get<std::string>();
            auto scene = battery::scenario_from_json(battery::compile_experiment(heated));
            const auto& p = scene.heating_protocols.back();
            require(p.cutoff_temperature_k == 423.15 && p.cutoff_rate_k_per_s == 0. && p.sensor_id == bid,
                    "Custom cutoff lost during heater creation/compilation");
            if (!ramp_control)
                require(p.feedforward_w == 7 && p.proportional_w_per_k == 0 && p.ramp_k_per_s == 0,
                        "Constant heater gained temperature feedback");
            const auto edited_cutoff = battery::set_heater_cutoff(heated, id, 473.15, 2.);
            scene = battery::scenario_from_json(battery::compile_experiment(edited_cutoff));
            require(scene.heating_protocols.back().cutoff_temperature_k == 473.15 &&
                        scene.heating_protocols.back().cutoff_rate_k_per_s == 2.,
                    "Existing cutoff edit lost");
            require(edited_cutoff["assembly"] == heated["assembly"], "Cutoff edit changed physical apparatus");
            battery::ui::EditorDocument cutoff_doc;
            cutoff_doc.draft = edited_cutoff;cutoff_doc.record(heated);cutoff_doc.undo();
            require(cutoff_doc.draft == heated, "Cutoff Undo failed");
            cutoff_doc.redo();require(cutoff_doc.draft == edited_cutoff, "Cutoff Redo failed");
        }
        const auto fixed_heater = battery::add_external_heater(original, bid, "+x", 7, 500, 5, false, .1);
        const auto fixed_id = fixed_heater["assembly"]["passive_cells"].back().at("id").get<std::string>();
        const auto with_cutoff = battery::set_heater_cutoff(fixed_heater, fixed_id, 423.15, 0.);
        const auto fixed_scene = battery::scenario_from_json(battery::compile_experiment(with_cutoff));
        require(std::abs(fixed_scene.heating_protocols.back().feedforward_w - 7.) < 1e-12 &&
                    fixed_scene.heating_protocols.back().proportional_w_per_k == 0.,
                "Attaching cutoff changed existing constant power");
        for (const auto invalid : {std::pair{-1., 0.}, std::pair{400., -1.}}) {
            bool rejected = false;
            try { (void)battery::set_heater_cutoff(fixed_heater, fixed_id, invalid.first, invalid.second); }
            catch (const std::exception&) { rejected = true; }
            require(rejected, "Invalid cutoff accepted");
        }
        // Apparatus deletion must target the heater, not its battery, and be
        // reversible for both fixed-power and regulated external heaters.
        for (bool controlled : {false, true}) {
            const auto heated = battery::add_external_heater(original, bid, "+x", 5, 500, 5, controlled, .1);
            const std::string heater = heated["assembly"]["passive_cells"].back().at("id");
            const auto two = battery::add_external_heater(heated, bid, "-x", 7, 500, 5, true, .1);
            const auto sibling = two["assembly"]["passive_cells"].back();
            const auto individually_edited = battery::set_heater_cutoff(two, heater, 453.15, 3.);
            for (const auto& before : two["conditions"]["heating_protocols"])
                if (before["heater_id"] != heater) {
                    const auto& list = individually_edited["conditions"]["heating_protocols"];
                    const auto same = std::find_if(list.begin(), list.end(),
                                                  [&](const auto& p) { return p["id"] == before["id"]; });
                    require(same != list.end() && *same == before,
                            "Editing one heater changed another controller");
                }
            const auto removed = battery::delete_element(two, heater);
            require(removed["assembly"]["batteries"] == original["assembly"]["batteries"],
                    "Deleting heater changed its target battery");
            require(removed["assembly"]["passive_cells"].back() == sibling,
                    "Deleting heater changed another apparatus");
            for (const auto& p : removed["assembly"]["ports"])
                require(p["cell_id"] != heater, "Deleted heater retained a port");
            for (const auto& c : removed["conditions"]["heating_protocols"])
                require(c["heater_id"] != heater, "Deleted heater retained its controller");
            const auto scene = battery::scenario_from_json(battery::compile_experiment(removed));
            for (const auto& cell : scene.cells)
                require(cell.id != heater, "Deleted heater survived compilation");
            // Same face can be used again after deletion.
            (void)battery::add_external_heater(removed, bid, "+x", 5, 500, 5, controlled, .1);
            battery::ui::EditorDocument transaction;
            transaction.draft = removed;
            transaction.record(two);
            transaction.undo();
            require(transaction.draft == two, "Heater deletion Undo lost contacts/controller");
            transaction.redo();
            require(transaction.draft == removed, "Heater deletion Redo failed");
        }
        auto duplicated = battery::duplicate_battery(original, bid);
        const std::string copied_id = duplicated["assembly"]["batteries"].back().at("id");
        auto observed_copy = duplicated;
        observed_copy["measurements"].push_back(
            {{"id", "copy-temperature"},
             {"quantity", "temperature"},
             {"unit", "K"},
             {"mapping", {{"status", "confirmed"}, {"cell_id", copied_id}}}});
        const auto deleted = battery::delete_element(observed_copy, copied_id);
        require(deleted["measurements"].back()["mapping"]["status"] == "unknown" &&
                    !deleted["measurements"].back()["mapping"].contains("cell_id"),
                "Delete lost observation data or retained a dangling mapping");
        doc.draft = deleted;
        doc.record(observed_copy);
        doc.undo();
        require(doc.draft == observed_copy, "Delete undo lost topology or observations");
        doc.redo();
        require(doc.draft == deleted, "Delete redo failed");
        doc.load(argv[1]);
        bool rejected_last = false;
        try {
            (void)battery::delete_element(original, bid);
        } catch (const std::exception&) {
            rejected_last = true;
        }
        require(rejected_last, "Deleting the last battery should explain the contract constraint");
        J face = {{"material_id", "test-top"}, {"conductivity_w_per_m_k", {.3, .3, .3}},
                  {"thickness_m", .0002}, {"density_kg_m3", 1200}, {"specific_heat_j_kg_k", 1300},
                  {"damage_onset_k", 443.15},  {"damage_rate_per_s", .02},
                  {"damage_scale_k", 30},      {"destroyed_conductance_factor", .05}};
        J opening = {{"kind", "rupture"},
                     {"opening_pressure_pa", 80000},
                     {"strength_reference_k", 300},
                     {"strength_temperature_scale_k", 300},
                     {"discharge_coefficient", .7},
                     {"damage_opening",
                      {{"mode", "deterministic_aperture"},
                       {"area_rate_per_s", 1},
                       {"pressure_exponent", 1},
                       {"max_damage_step", .005},
                       {"fracture_energy_j_per_m2", 200},
                       {"valve", {{"radius_m", .001}, {"center_m", {0, 0}}, {"pressure_factor", .4}}}}}};
        const auto faced = battery::set_shell_face(duplicated, bid, "+z", face, opening, "cavity");
        require(faced["assembly"]["batteries"].back() == duplicated["assembly"]["batteries"].back(),
                "Face edit modified sibling");
        const auto faced_scene = battery::scenario_from_json(battery::compile_experiment(faced));
        const auto panel = std::find_if(faced_scene.cells.begin(), faced_scene.cells.end(),
                                        [&](const auto& c) { return c.id == bid + ":case:+z"; });
        require(panel != faced_scene.cells.end() && panel->damage_onset_k == 443.15 &&
                    panel->damage_rate_at_onset_per_s == .02 && panel->material_id == "test-top",
                "Face material/damage did not reach kernel config");
        const double plate_volume = panel->dimensions_m[0] * panel->dimensions_m[1] * panel->dimensions_m[2];
        require(panel->dimensions_m[2] == .0002 &&
                    std::abs(panel->heat_capacity_j_per_k - plate_volume * 1200 * 1300) < 1e-12,
                "Face thickness/density/specific heat did not reach physical plate");
        auto bad_opening = opening;
        bad_opening["damage_opening"]["valve"]["radius_m"] = 1;
        bool rejected_valve = false;
        try {
            (void)battery::set_shell_face(duplicated, bid, "+z", face, bad_opening, "cavity");
        } catch (const std::exception&) {
            rejected_valve = true;
        }
        require(rejected_valve, "Oversized valve accepted");
        auto bad_face = face;
        bad_face["damage_rate_per_s"] = -1;
        bool rejected_face = false;
        try {
            (void)battery::set_shell_face(duplicated, bid, "+z", bad_face, opening, "cavity");
        } catch (const std::exception&) {
            rejected_face = true;
        }
        require(rejected_face, "Negative face damage rate accepted");
        J case_properties = {{"inside_dimensions_m", {.12, .12, .08}},
                             {"position_m", {0, 0, 0}},
                             {"thickness_m", .002},
                             {"material_id", "test-case"},
                             {"density_kg_m3", 1200},
                             {"specific_heat_j_kg_k", 1200},
                             {"conductivity_w_per_m_k", {.2, .2, .2}},
                             {"initial_temperature_k", 298.15},
                             {"ambient_temperature_k", 298.15},
                             {"gas_contact_h_w_m2_k", 10},
                             {"ambient_h_w_m2_k", 5},
                             {"damage_onset_k", 500},
                             {"damage_rate_per_s", 0},
                             {"damage_scale_k", 40}};
        auto boxed = battery::set_enclosure(original, "test-case", "cavity", case_properties);
        // Regression: the Case panel originally showed editable fields without
        // selecting the existing hole, silently changing only a future template.
        battery::ui::OpeningSelection selection;
        auto holes = boxed["assembly"]["gas_meshes"]["test-case"]["openings"];
        require(selection.sync("test-case", holes) && selection.id == "top-opening" && !selection.creating,
                "Existing opening not automatically selected on first panel visit");
        require(!selection.sync("test-case", holes), "Unchanged source resets unfinished numeric input");
        selection.begin_new();
        require(!selection.sync("test-case", holes) && selection.id.empty() && selection.creating,
                "New opening mode unexpectedly edits an existing hole");
        selection.select("test-case", holes.front());
        holes[0]["size_m"] = {.03, .03};
        battery::ui::EditorDocument hole_doc;
        hole_doc.draft = battery::set_enclosure_openings(boxed, "test-case", holes);
        hole_doc.record(boxed);
        require(selection.sync("test-case", holes) && selection.source["size_m"][0] == .03,
                "Diameter edit not reflected in selection fields");
        auto resized_compiled = hole_doc.compile();
        auto resized_geometry = battery::visual_geometry_from_json(
            resized_compiled, battery::scenario_from_json(resized_compiled));
        require(resized_geometry.case_openings.size() == 1 && resized_geometry.case_openings[0].circle &&
                    resized_geometry.case_openings[0].size == std::array<double, 2>{.03, .03},
                "Diameter edit did not reach compiled visual opening");
        const auto diameter_draft = hole_doc.draft;
        holes[0]["shape"] = "rectangle";
        holes[0]["size_m"] = {.03, .012};
        hole_doc.draft = battery::set_enclosure_openings(hole_doc.draft, "test-case", holes);
        hole_doc.record(diameter_draft);
        require(selection.sync("test-case", holes) && selection.source["shape"] == "rectangle",
                "Shape edit did not update selection");
        const auto rectangular = hole_doc.compile();
        const auto rectangle_geometry =
            battery::visual_geometry_from_json(rectangular, battery::scenario_from_json(rectangular));
        require(!rectangle_geometry.case_openings[0].circle &&
                    rectangle_geometry.case_openings[0].size == std::array<double, 2>{.03, .012},
                "Shape edit did not reach compiled visual opening");
        double aperture_area = 0;
        for (const auto& gas : rectangular["gas_network"]["compartments"])
            if (gas.contains("ambient_boundary"))
                aperture_area += gas["ambient_boundary"]["area_m2"].get<double>();
        require(std::abs(aperture_area - .03 * .012) < 1e-12,
                "Shape/size edit changed only display, not physical gas aperture");
        hole_doc.undo();
        require(
            selection.sync("test-case", hole_doc.draft["assembly"]["gas_meshes"]["test-case"]["openings"]) &&
                selection.source["shape"] == "circle",
            "Undo leaves stale shape fields");
        hole_doc.redo();
        require(
            selection.sync("test-case", hole_doc.draft["assembly"]["gas_meshes"]["test-case"]["openings"]) &&
                selection.source["shape"] == "rectangle",
            "Redo leaves stale shape fields");
        selection.begin_new();
        require(selection.sync("another-case", holes) && !selection.creating,
                "Switching cases retains stale create/selection state");
        const auto count = boxed["assembly"]["passive_cells"].size();
        require(count == original["assembly"].value("passive_cells", J::array()).size() + 6,
                "Case must add six walls");
        case_properties["inside_dimensions_m"] = {.14, .14, .08};
        auto resized = battery::set_enclosure(boxed, "test-case", "cavity", case_properties);
        require(resized["assembly"]["passive_cells"].size() == count, "Resizing case duplicates walls");
        auto density = [](const J& d) {
            for (const auto& g : d["assembly"]["gas_network"]["compartments"])
                if (g["id"] == "cavity")
                    return g["initial_mass_kg"].get<double>() / g["volume_m3"].get<double>();
            throw std::runtime_error("Missing gas");
        };
        require(std::abs(density(resized) - density(original)) < 1e-12,
                "Case resize changes initial gas density/pressure");
        case_properties["inside_dimensions_m"] = {.001, .001, .001};
        bool rejected_case = false;
        try {
            (void)battery::set_enclosure(original, "test-case", "cavity", case_properties);
        } catch (const std::exception&) {
            rejected_case = true;
        }
        require(rejected_case, "Case smaller than contents accepted");
        const auto sibling = duplicated.at("assembly").at("batteries").back();
        const auto models_before = duplicated.at("battery_models");
        auto cylinder = battery::set_battery_geometry(duplicated, bid, "cylinder", {.009, .065});
        require(cylinder.at("assembly").at("batteries").back() == sibling, "Shape edit modified sibling");
        for (std::size_t i = 0; i < models_before.size(); ++i)
            require(cylinder.at("battery_models")[i] == models_before[i], "Shape edit mutated shared model");
        auto cylinder_scene = battery::scenario_from_json(battery::compile_experiment(cylinder));
        const auto visual = battery::visual_geometry_from_json(cylinder, cylinder_scene);
        require(visual.solid_proxies[0].shape == battery::ShapeKind::Cylinder &&
                    visual.solid_proxies[0].dimensions_m == std::vector<double>({.009, .065, 0.}),
                "Cylinder visual proxy has wrong shape/dimensions");
        const auto& dims = cylinder_scene.cells[0].dimensions_m;
        require(std::abs(dims[0] * dims[1] * dims[2] - std::acos(-1.) * .009 * .009 * .065) < 1e-15,
                "Cylinder approximation changed volume");
        auto prism = battery::set_battery_geometry(cylinder, bid, "prismatic", {.02, .03, .07});
        require(!prism["numerics"]["batteries"][bid].contains("geometry_approximation"),
                "Stale cylinder approximation");
        require(prism["battery_models"].size() == cylinder["battery_models"].size(),
                "Every shape edit leaks a model");
        const auto prism_scene = battery::scenario_from_json(battery::compile_experiment(prism));
        require(prism_scene.cells[0].dimensions_m == std::vector<double>({.02, .03, .07}), "Prism size lost");
        bool bad_geometry = false;
        try {
            (void)battery::set_battery_geometry(prism, bid, "cylinder", {-.009, .065});
        } catch (const std::exception&) {
            bad_geometry = true;
        }
        require(bad_geometry, "Negative radius accepted");
        doc.draft = cylinder;
        doc.record(original);
        doc.undo();
        require(doc.draft == original, "Shape undo failed");
        doc.redo();
        require(doc.draft == cylinder, "Shape redo failed");
        doc.undo();
        if (argc == 3) {
            std::ofstream preview(argv[2]);
            preview << cylinder.dump(2);
            require(bool(preview), "Cannot export cylinder preview fixture");
        }
        doc.draft["name"] = "Edited";
        doc.record(original);
        doc.undo();
        require(doc.draft == original, "Undo lost data");
        doc.redo();
        require(doc.draft["name"] == "Edited", "Redo failed");
        char tmp[] = "/tmp/battery-editor-test-XXXXXX";
        const auto dir = mkdtemp(tmp);
        if (!dir)
            throw std::runtime_error("mkdtemp failed");
        const auto path = std::filesystem::path(dir) / "experiment.json";
        battery::ui::EditorDocument cutoff_saved;
        cutoff_saved.load(argv[1]);
        cutoff_saved.draft = with_cutoff;
        const auto cutoff_path = std::filesystem::path(dir) / "heater-cutoff.json";
        cutoff_saved.save(cutoff_path);
        battery::ui::EditorDocument cutoff_loaded;
        cutoff_loaded.load(cutoff_path);
        require(cutoff_loaded.draft == with_cutoff, "Save/load changed heater cutoff");
        doc.sidecar["hidden_objects"] = {"deleted-object"};
        doc.sidecar["layout"]["selected_object"] = "deleted-object";
        doc.save(path);
        require(doc.sidecar["hidden_objects"].empty() && !doc.sidecar["layout"].contains("selected_object"),
                "Deleted UI references prevent saving");
        battery::ui::EditorDocument loaded;
        loaded.load(path);
        require(loaded.draft == doc.draft, "Editor save/load lost fields");
        const auto good = loaded.draft;
        const auto bad = std::filesystem::path(dir) / "invalid.json";
        {
            std::ofstream f(bad);
            f << "{}";
        }
        bool rejected = false;
        try {
            loaded.load(bad);
        } catch (const std::exception&) {
            rejected = true;
        }
        require(rejected && loaded.draft == good, "Failed import replaced the open project");
        {
            std::ofstream f(path.string() + ".editor.json");
            f << "{}";
        }
        loaded.load(path);
        require(loaded.draft == good, "Bad optional view prevented loading physics");
        const auto native = std::filesystem::path(dir) / "scenario.json";
        {
            std::ofstream f(native);
            f << battery::compile_experiment(good).dump();
        }
        loaded.load(native);
        require(loaded.path != native, "Compiled native input would be overwritten by editor save");
        loaded.draft["version"] = 999;
        rejected = false;
        try {
            loaded.save(path);
        } catch (const std::exception&) {
            rejected = true;
        }
        require(rejected, "Invalid draft saved over valid project");
        loaded.load(path);
        require(loaded.draft == good, "Failed save corrupted existing project");
        for (const auto& entry : std::filesystem::directory_iterator(dir))
            std::filesystem::remove(entry.path());
        std::filesystem::remove(dir);
        std::cout << "Editor document: transactional import/save, unknown fields, undo/redo, sidecar and "
                     "native protection passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
