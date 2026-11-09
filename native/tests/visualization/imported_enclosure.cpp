#include "battery_ui/editor_document.hpp"
#include "battery_native/contracts/enclosure_openings.hpp"
#include "battery_native/contracts/experiment_contract.hpp"
#include "battery_native/scenario_document.hpp"
#include "battery_native/simulation/simulation.hpp"
#include "battery_native/data/visual_frame.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <functional>
#include <iostream>
#include <cmath>
#include <set>
#include <stdexcept>

using J = nlohmann::json;
void require(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
int main(int argc, char** argv) {
    try {
        using namespace battery;
        require(argc == 2, "Expected imported contract");
        J original; std::ifstream(argv[1]) >> original;
        ui::EditorDocument doc; doc.load(argv[1]);
        const auto registered = doc.draft;
        require(register_imported_enclosures(registered) == registered, "Import adapter is not idempotent");
        const auto a = compile_experiment(original), b = doc.compile();
        int shared_fronts = 0;
        for (const auto& source : original.at("assembly").at("gas_network").at("channels")) {
            const auto& channels = a.at("gas_network").at("channels");
            auto compiled = std::find_if(channels.begin(), channels.end(), [&](const auto& ch) {
                return ch.at("id") == source.at("id");
            });
            require(compiled != channels.end(), "Import lost an original gas route");
            auto unchanged = *compiled;
            if (source.contains("damage_opening") &&
                source.at("damage_opening").value("mode", "") == "deterministic_aperture") {
                require(unchanged["damage_opening"].value("whole_plate", false),
                        "Imported complete face still uses isolated tile rupture");
                ++shared_fronts;
                if (!source.at("damage_opening").contains("whole_plate"))
                    unchanged["damage_opening"].erase("whole_plate");
            }
            require(unchanged == source, "Common front changes valve, strength, work or gas routing");
        }
        require(shared_fronts == 90, "Expected three connected lid patches per cylinder");
        // Explicit native/legacy choices remain reproducible; do not invent
        // gas paths for a face that declares only a limited local aperture.
        auto local_only = original;
        for (auto& ch : local_only["assembly"]["gas_network"]["channels"])
            if (ch.contains("damage_opening")) ch["damage_opening"]["whole_plate"] = false;
        const auto legacy = compile_experiment(local_only);
        for (const auto& ch : legacy.at("gas_network").at("channels"))
            if (ch.contains("damage_opening"))
                require(!ch.at("damage_opening").value("whole_plate", false), "Explicit legacy choice lost");
        for (const auto* key : {"cells", "shells", "ports", "interfaces", "gas_network", "boundaries", "heating_protocols", "simulation"})
            require(a.at(key) == b.at(key), std::string("Import changed physics: ") + key);
        const auto cases = editable_enclosures(doc.draft);
        require(cases.size() == 1 && cases.contains("ABS"), "Existing case not registered");
        auto holes = cases.at("ABS").at("openings");
        require(holes[0].at("size_m")[0] == .0254, "Lost original central opening");
        auto check = [&](const J& e, double area, int count) {
            const auto& entry = e.at("metadata").at("editor_enclosures").at("ABS");
            const auto ids = entry.at("owned_channels").get<std::set<std::string>>();
            double actual = 0;
            for (const auto& ch : e.at("assembly").at("gas_network").at("channels"))
                if (ids.contains(ch.at("id"))) {
                    actual += ch.at("area_m2").get<double>();
                    require(ch.at("from").get<std::string>().starts_with("aligned-inner-"), "Hole has wrong inside source");
                    require(ch.at("to").get<std::string>().starts_with("air-"), "Hole bypasses external gas field");
                    require(ch.at("hydraulic_resistance_factor") == 25, "Lost fitted gas resistance");
                }
            require(std::abs(actual - area) < 1e-11, "Hole area duplicated or lost");
            const auto compiled = compile_experiment(e);
            const auto scene = scenario_from_json(compiled);
            const auto geometry = visual_geometry_from_json(compiled, scene);
            require(int(geometry.case_openings.size()) == count, "Visual openings out of sync");
            require(compiled.at("gas_network").at("compartments") == a.at("gas_network").at("compartments"), "Gas mass/state changed");
            require(compiled.at("heating_protocols") == a.at("heating_protocols") &&
                    compiled.at("shells") == a.at("shells"), "Heater or battery shell changed");
            if (entry.at("explicit_mesh").at("openings") != entry.at("reference_openings")) {
                double removed = 0, conductance = 0;
                for (const auto& h : parse_enclosure_openings(entry.at("explicit_mesh")))
                    if (h.face == "+z") removed += h.area();
                std::set<std::string> inner_ports;
                for (const auto& p : scene.ports)
                    if (p.cell_id == "ABS-plusz" && p.surface == "-z") inner_ports.insert(p.id);
                for (const auto& c : compiled.at("gas_network").at("thermal_contacts"))
                    if (inner_ports.contains(c.value("port_id", ""))) {
                        const auto& law = c.at("connectivity");
                        conductance += law.at("background_conductance_w_per_k").get<double>() +
                                       .5*law.at("active_conductance_w_per_k").get<double>();
                    }
                const auto size = entry.at("explicit_mesh").at("inside_dimensions_m").get<std::array<double,3>>();
                require(std::abs(conductance - 8*(size[0]*size[1]-removed)) < 1e-9,
                        "Opening incorrectly scales gas-wall thermal contact area");
            }
            std::set<std::string> original_owned;
            for (const auto& route : entry.at("reference_routes")) original_owned.insert(route.at("id"));
            for (const auto& ch : a.at("gas_network").at("channels")) if (!original_owned.contains(ch.at("id"))) {
                const auto& channels = compiled.at("gas_network").at("channels");
                require(std::find(channels.begin(), channels.end(), ch) != channels.end(), "Unrelated gas channel changed");
            }
        };
        check(registered, std::acos(-1.)*.0254*.0254/4, 1);
        holes[0]["size_m"] = {.03, .03};
        const auto larger = set_enclosure_openings(registered, "ABS", holes);
        check(larger, std::acos(-1.)*.03*.03/4, 1);
        require(set_enclosure_openings(larger, "ABS", holes) == larger, "Repeated edit changes physics");
        for (const char* face : {"+x", "-x", "+y", "-y", "+z", "-z"}) {
            holes[0]["shape"] = "rectangle"; holes[0]["face"] = face;
            holes[0]["center_m"] = {.005, .004}; holes[0]["size_m"] = {.012, .01};
            std::cout << "Checking " << face << std::endl;
            const auto moved = set_enclosure_openings(larger, "ABS", holes);
            check(moved, .012*.01, 1);
        }
        auto multiple = holes;
        multiple[0]["face"] = "+z";
        multiple.push_back({{"id", "side"}, {"face", "+x"}, {"shape", "circle"},
                            {"center_m", {0., 0.}}, {"size_m", {.008, .008}}});
        doc.draft = set_enclosure_openings(larger, "ABS", multiple);
        check(doc.draft, .012*.01 + std::acos(-1.)*.008*.008/4, 2);
        const auto changed = doc.draft;
        doc.record(registered); doc.undo(); require(doc.draft == registered, "Undo lost case");
        doc.redo(); require(doc.draft == changed, "Redo lost case");
        auto closed = set_enclosure_openings(changed, "ABS", J::array());
        check(closed, 0, 0);
        auto restored = set_enclosure_openings(closed, "ABS", cases.at("ABS").at("openings"));
        check(restored, std::acos(-1.)*.0254*.0254/4, 1);
        require(restored.at("assembly").at("passive_cells") == registered.at("assembly").at("passive_cells"), "Wall capacity drift after restore");
        auto invalid = multiple; invalid[0]["size_m"] = {1., 1.};
        bool rejected = false;
        try { (void)set_enclosure_openings(registered, "ABS", invalid); } catch (const std::exception&) {rejected = true;}
        require(rejected && doc.draft == changed, "Invalid opening poisoned draft");
        // Same editor/backend on an unrelated case ID/material: no FSRI branch
        // may be required to add an aperture to an explicit enclosure.
        std::function<J(const J&)> rename = [&](const J& value) -> J {
            if (value.is_string()) {
                auto s = value.get<std::string>();
                std::size_t pos = 0;
                while ((pos = s.find("ABS", pos)) != std::string::npos) {
                    s.replace(pos, 3, "custom-case"); pos += 11;
                }
                return s;
            }
            if (value.is_array()) { auto result = J::array(); for (const auto& v : value) result.push_back(rename(v)); return result; }
            if (value.is_object()) { auto result = J::object(); for (const auto& [k,v] : value.items()) result[rename(J(k)).get<std::string>()] = rename(v); return result; }
            return value;
        };
        auto generic = rename(closed);
        auto law = generic["metadata"]["editor_enclosures"]["custom-case"]["passage_law"];
        generic["metadata"].erase("editor_enclosures"); generic["metadata"].erase("editor_import");
        generic = register_explicit_enclosure(generic, "custom-case", J::array(), J::array(), law);
        generic = set_enclosure_openings(generic, "custom-case", multiple);
        require(editable_enclosures(generic).at("custom-case").at("openings") == multiple,
                "Non-FSRI initially closed case cannot create holes");
        (void)compile_experiment(generic);
        const auto scene = scenario_from_json(compile_experiment(changed));
        Simulation sim(scene); sim.step(.005);
        const auto d = sim.process_diagnostics();
        require(std::abs(d.mass_residual_kg) < 1e-10 && std::abs(d.thermal_residual_j) < 1e-4 &&
                std::abs(d.chemical_residual_j) < 1e-4, "CUDA ledger failed after aperture edits");
        std::cout << "Imported case: unchanged import, all six faces, area conservation, closed/restore, undo/redo, CUDA passed\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
