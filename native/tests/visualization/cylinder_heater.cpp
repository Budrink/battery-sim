#include "battery_native/contracts/experiment_editing.hpp"
#include "battery_native/contracts/experiment_contract.hpp"
#include "battery_native/scenario_document.hpp"
#include "battery_native/render/renderer.hpp"
#include "editor/state/model_edits.hpp"
#include <nlohmann/json.hpp>
#include <GLFW/glfw3.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <set>
#include <stdexcept>

namespace {
void require(bool ok, const std::string& why) {
    if (!ok)
        throw std::runtime_error(why);
}
bool equivalent(const nlohmann::json& a, const nlohmann::json& b) {
    if (a.is_number() && b.is_number())
        return std::abs(a.get<double>() - b.get<double>()) <= 1e-12 * std::max(1., std::abs(a.get<double>()));
    if (a.type() != b.type() || a.size() != b.size())
        return false;
    if (a.is_object()) {
        for (const auto& [k, v] : a.items())
            if (!b.contains(k) || !equivalent(v, b[k]))
                return false;
        return true;
    }
    if (a.is_array()) {
        for (std::size_t i = 0; i < a.size(); ++i)
            if (!equivalent(a[i], b[i]))
                return false;
        return true;
    }
    return a == b;
}
} // namespace
int main(int argc, char** argv) {
    try {
        using namespace battery;
        using J = nlohmann::json;
        require(argc == 2, "Expected laboratory contract");
        J original;
        std::ifstream(argv[1]) >> original;
        auto e = upgrade_experiment(original);
        const std::string bid = e["assembly"]["batteries"][0]["id"];
        ui::ModelEdits edits;
        edits.edit(0, e["battery_models"][0]).value["geometry"] = {{"kind", "cylinder"},
                                                                   {"dimensions_m", {.01, .02}}};
        edits.commit_valid(e);
        require(!edits.pending(),
                "Shared model shape failed: " + edits.edit(0, e["battery_models"][0]).error);
        auto wrapped = add_external_heater(e, bid, "wrap", 5, 500, 5, false, .1, 298.15, 0, true);
        const std::string hid = wrapped["assembly"]["passive_cells"].back()["id"];
        const auto controls = wrapped["conditions"]["heating_protocols"];
        auto verify = [&](const J& doc, const std::string& mount, int count) {
            auto compiled = compile_experiment(doc);
            const auto scene = scenario_from_json(compiled);
            const auto geometry = visual_geometry_from_json(compiled, scene);
            int heater = -1;
            for (int i = 0; i < int(scene.cells.size()); ++i)
                if (scene.cells[i].id == hid)
                    heater = i;
            require(heater >= 0, "Missing heater");
            require(geometry.heater_shapes[heater] == (mount == "wrap" ? 1 : 2),
                    "Incorrect heater render mesh");
            const auto& info = doc["metadata"]["heater_mounts"][hid];
            const double r = info["radius_m"], h = info["height_m"];
            const double expected = mount == "wrap" ? 2 * std::acos(-1.) * r * h : std::acos(-1.) * r * r;
            std::set<std::string> destinations;
            double area = 0, conductance = 0;
            for (const auto& edge : doc["assembly"]["interfaces"])
                if (edge["id"].get<std::string>().starts_with(hid + "-coupling")) {
                    destinations.insert(edge["port_b"]);
                    conductance += edge["law"]["background_conductance_w_per_k"].get<double>() +
                                   .5 * edge["law"]["active_conductance_w_per_k"].get<double>();
                    for (const auto& port : doc["assembly"]["ports"])
                        if (port["id"] == edge["port_a"])
                            area += port["area_m2"].get<double>();
                }
            require(destinations.size() == std::size_t(count), "Wrong number of independently heated faces");
            require(std::abs(area / expected - 1) < 1e-12, "Cylinder contact area not conserved");
            require(std::abs(conductance / (500 * expected) - 1) < 1e-12,
                    "Per-area contact conductance wrong");
            require(doc["conditions"]["heating_protocols"] == controls, "Remount changed power or cutoff");
            const auto mesh = build_thermal_mesh(scene);
            for (std::size_t j = 0; j < scene.interfaces.size(); ++j) {
                const auto& edge = scene.interfaces[j];
                if (!edge.id.starts_with(hid + "-coupling"))
                    continue;
                double weights = 0.;
                std::set<int> donors;
                for (const auto& link : mesh.links)
                    if (link.interface_index == int(j)) {
                        weights += link.weight;
                        donors.insert(link.a);
                        require(link.series_resistance_a_k_per_w == 0.,
                                "Isothermal heater has fictitious internal resistance");
                    }
                require(std::abs(weights - 1.) < 1e-12 && donors.size() == 1,
                        "Heater conductance/power duplicated by mesh subdivision");
            }
            return std::make_pair(scene, geometry);
        };
        auto [scene, geometry] = verify(wrapped, "wrap", 4);
        auto refined = wrapped;
        auto divisions = refined["numerics"]["batteries"][bid]["divisions"].get<std::vector<int>>();
        for (auto& n : divisions)
            n *= 2;
        refined["numerics"]["batteries"][bid]["shell_surface_divisions"] = divisions;
        auto [fine_scene, fine_geometry] = verify(refined, "wrap", 4);
        auto effective_g = [&](const Scenario& s) {
            const auto mesh = build_thermal_mesh(s);
            double total = 0;
            for (const auto& link : mesh.links)
                if (link.interface_index >= 0 &&
                    s.interfaces[link.interface_index].id.starts_with(hid + "-coupling")) {
                    const auto& law = s.interfaces[link.interface_index].law;
                    const double g =
                        (law.background_conductance_w_per_k + .5 * law.active_conductance_w_per_k) *
                        link.weight;
                    total += 1. / (1. / g + link.series_resistance_k_per_w);
                }
            return total;
        };
        require(std::abs(effective_g(fine_scene) / effective_g(scene) - 1) < 1e-12,
                "Heater conductance depends on shell surface subdivision");
        const auto top = set_heater_mount(wrapped, hid, bid, "+z");
        verify(top, "+z", 1);
        const auto bottom = set_heater_mount(top, hid, bid, "-z");
        verify(bottom, "-z", 1);
        const auto again = set_heater_mount(bottom, hid, bid, "wrap");
        verify(again, "wrap", 4);
        require(equivalent(again["assembly"]["interfaces"], wrapped["assembly"]["interfaces"]),
                "Remount drifts contacts");
        auto fresh_disk = add_external_heater(e, bid, "-z", 5, 500, 5, false, .1, 298.15, 0, true);
        require(fresh_disk["metadata"]["heater_mounts"].begin().value()["kind"] == "disk",
                "New cylinder end heater is not a disk");
        Simulation sim(scene);
        auto before = capture_visual_frame(sim);
        sim.step(.005);
        auto after = capture_visual_frame(sim);
        require(!before.heaters.back().stopped && after.heaters.back().stopped,
                "Frame does not capture accepted cutoff");
        const auto d = sim.process_diagnostics();
        require(std::abs(d.mass_residual_kg) < 1e-10 && std::abs(d.thermal_residual_j) < 1e-5 &&
                    std::abs(d.chemical_residual_j) < 1e-5,
                "Wrap heater violates CUDA conservation");
        // Same one heater remains the sole electrical source after mounting.
        auto hot = wrapped;
        hot["conditions"]["heating_protocols"].back()["cutoff_temperature_k"] = 2000.;
        hot["assembly"]["passive_cells"].back()["thermal"]["initial_temperature_k"] = 398.15;
        Simulation transfer(scenario_from_json(compile_experiment(hot)));
        const auto cold = transfer.cells();
        for (int step = 0; step < 10; ++step)
            transfer.step(.005);
        std::set<std::string> heated_faces;
        for (std::size_t i = 0; i < transfer.cells().size(); ++i) {
            const auto& cell = transfer.scenario().cells[i];
            if (cell.shell_contents_id == bid &&
                (cell.id.ends_with(":+x") || cell.id.ends_with(":-x") || cell.id.ends_with(":+y") ||
                 cell.id.ends_with(":-y")) &&
                transfer.cells()[i].temperature_k > cold[i].temperature_k + 1e-5)
                heated_faces.insert(cell.id);
        }
        require(heated_faces.size() == 4, "Hot wrap does not heat all four lateral shell faces on CUDA");
        require(std::abs(transfer.process_diagnostics().thermal_residual_j) < 1e-5,
                "Active heater loses energy");
        require(glfwInit(), "GLFW initialization failed");
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        auto* window = glfwCreateWindow(400, 400, "Heater regression", nullptr, nullptr);
        require(window != nullptr, "No GL context");
        glfwMakeContextCurrent(window);
        {
            SceneRenderer renderer;
            require(renderer.init(), "Renderer initialization failed");
            OrbitCamera camera;
            camera.distance = 1.4;
            camera.yaw = -.8;
            camera.pitch = .5;
            VisualSettings settings;
            settings.show_gas = false;
            auto frame = capture_visual_frame(sim);
            auto image = [&] {
                auto texture = renderer.render(400, 400, scene, sim.thermal_mesh(), frame, geometry, camera,
                                               settings, -1);
                std::vector<unsigned char> pixels(400 * 400 * 4);
                glBindTexture(GL_TEXTURE_2D, texture);
                glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
                require(glGetError() == GL_NO_ERROR, "Heater OpenGL error");
                return pixels;
            };
            frame.heaters.back().stopped = false;
            frame.heaters.back().commanded_power_w = 0;
            const auto enabled = image();
            frame.heaters.back().stopped = true;
            const auto stopped = image();
            settings.show_heaters = false;
            const auto hidden = image();
            require(enabled != stopped && stopped != hidden && enabled != hidden,
                    "Cutoff/visibility do not affect heater rendering");
            int dark = 0, yellow = 0;
            for (std::size_t i = 0; i < enabled.size(); i += 4) {
                if (enabled[i] > 180 && enabled[i + 1] > 150 && enabled[i + 2] < 70) {
                    ++yellow;
                    if (stopped[i] < 30 && stopped[i + 1] < 30 && stopped[i + 2] < 30)
                        ++dark;
                }
            }
            require(yellow > 20 && dark > 10, "Yellow heater outline does not turn black on cutoff");
        }
        glfwDestroyWindow(window);
        glfwTerminate();
        auto removed = delete_element(wrapped, hid);
        require(!removed["metadata"]["heater_mounts"].contains(hid),
                "Deleted heater leaves mounting metadata");
        std::cout << "Cylinder heaters: shared model shape, wrap/disk area and contacts, fixed total power, "
                     "remount, CUDA balance/cutoff, GL visibility/yellow-to-black passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
