#pragma once
#include "battery_native/gas/mixing/gas_thermal_contact.hpp"

// Generator, compiler and editor must preserve the complete calibrated shape,
// not only its endpoint temperature table. Compare on identical exposed areas.
void module_surface_law_checks(const nlohmann::json& doc) {
    auto old = doc;
    old["assembly"]["gas_meshes"]["case"].erase("battery_surface_connectivity_per_m2");
    const auto before = battery::scenario_from_json(battery::compile_experiment(old));
    const auto after = battery::scenario_from_json(battery::compile_experiment(doc));
    const auto a = battery::build_thermal_mesh(before), b = battery::build_thermal_mesh(after);
    require(a.gas_patches.size() == b.gas_patches.size(), "Surface law changed contact geometry");
    require(a.volumes.size() == b.volumes.size(), "Surface law changed solid geometry");
    for (std::size_t i = 0; i < a.volumes.size(); ++i)
        require(a.volumes[i].capacity_j_per_k == b.volumes[i].capacity_j_per_k,
                "Surface law changed heat capacity");
    for (std::size_t i = 0; i < a.gas_patches.size(); ++i) {
        const auto& p = a.gas_patches[i];
        const auto& q = b.gas_patches[i];
        const bool casing = !after.cells[b.volumes[q.volume].object].shell_contents_id.empty();
        require(p.volume == q.volume && p.compartment == q.compartment &&
                    p.geometry_weight == q.geometry_weight,
                "Surface law moved a contact");
        for (double t : {298.15, 473.15, 773.15, 1200.}) {
            const double actual = battery::gas_contact_conductance(q, t, 298.15, 0.);
            double expected = battery::gas_contact_conductance(p, t, 298.15, 0.);
            if (casing) {
                const double response = battery::directional_log_metric(q.law.endpoint_a, q.geometry_axis, t, 0.);
                // Current generic h=10; fitted cold h=20, logit=-ln(59).
                expected = p.conductance_w_per_k / 10. *
                           (.2 + 1188. / (1. + 59. * std::exp(-response)));
            }
            require(std::abs(actual - expected) < 1e-11, "Gas surface law differs from reference");
        }
    }
    const auto& props = doc["metadata"]["editor_enclosures"]["case"]["properties"];
    const auto edited = battery::set_enclosure(doc, "case", "cavity", props);
    require(edited["assembly"]["gas_meshes"]["case"]["battery_surface_connectivity_per_m2"] ==
                doc["assembly"]["gas_meshes"]["case"]["battery_surface_connectivity_per_m2"],
            "Editing enclosure erased battery surface law");
    auto total = [](const nlohmann::json& document) {
        const auto scene = battery::scenario_from_json(battery::compile_experiment(document));
        const auto mesh = battery::build_thermal_mesh(scene);
        double g = 0.;
        for (const auto& p : mesh.gas_patches)
            if (!scene.cells[mesh.volumes[p.volume].object].shell_contents_id.empty())
                g += battery::gas_contact_conductance(p, 773.15, 298.15, 0.);
        return g;
    };
    const double reference = total(doc);
    for (double dx : {.04, .01}) {
        auto refined = doc;
        refined["assembly"]["gas_meshes"]["case"]["maximum_cell_size_m"] = dx;
        require(std::abs(total(refined) - reference) < 1e-10,
                "Refinement changes integrated battery surface conductance");
    }
}
