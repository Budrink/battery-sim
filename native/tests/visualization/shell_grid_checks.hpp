#pragma once
#include "battery_native/thermal/object_grid.hpp"

void shell_grid_checks(const nlohmann::json& doc) {
    auto refined = doc;
    refined["numerics"]["batteries"]["B05"]["shell_surface_divisions"] = {9, 9, 3};
    const auto scene = battery::scenario_from_json(battery::compile_experiment(refined));
    const auto mesh = battery::build_thermal_mesh(scene);
    double area = 0.;
    int count = 0, valves = 0;
    for (const auto& ch : scene.gas_channels) {
        if (ch.shell_object < 0)
            continue;
        const auto& shell = scene.cells[ch.shell_object];
        if (shell.id != "B05:case:+z")
            continue;
        const auto q = battery::grid_coordinates(shell, ch.shell_thermal_volume);
        const auto& pore = scene.gas_compartments[ch.from];
        require(pore.thermal_volume == q[0] / 3 + 3 * (q[1] / 3 + 3 * 2),
                "Refined aperture discharges from a remote contents pore");
        const auto& tile = mesh.volumes[mesh.object_offsets[ch.shell_object] + ch.shell_thermal_volume];
        const double actual_area = tile.size_m.x * tile.size_m.y;
        require(std::abs(battery::gas_channel_area(scene, ch) - actual_area) < 1e-15,
                "Refined aperture area differs from its tile");
        area += actual_area;
        ++count;
        valves += ch.opening.valve_radius_m > 0;
    }
    require(count == 81 && valves == 1 && std::abs(area - .02 * .02) < 1e-14,
            "Refined face lost area, patches or its local valve");
    refined["numerics"]["batteries"]["B05"]["shell_surface_divisions"] = {4, 3, 3};
    bool rejected = false;
    try {
        (void)battery::compile_experiment(refined);
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "Nonaligned shell subdivisions accepted");
}
