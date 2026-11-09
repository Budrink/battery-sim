#pragma once

// Removing a thin pad must create its actual volume, not a well-mixed shortcut
// around the neighbouring battery. Distances belong to fluid, not empty cubes.
void gap_geometry_checks(const nlohmann::json& doc) {
    const auto bare = battery::delete_element(doc, "ISS-B05-B06");
    const auto original = battery::compile_experiment(doc);
    for (double dx : {.04, .02, .01}) {
        auto input = bare;
        input["assembly"]["gas_meshes"]["case"]["maximum_cell_size_m"] = dx;
        const auto compiled = battery::compile_experiment(input);
        const auto& boxes = compiled["metadata"]["gas_volume_geometry"];
        const auto scene = battery::scenario_from_json(compiled);
        int narrow = 0;
        for (const auto& [id, b] : boxes.items()) {
            double box_volume = 1., fragments = 0.;
            for (int d = 0; d < 3; ++d)
                box_volume *= b["fluid_high_m"][d].get<double>() - b["fluid_low_m"][d].get<double>();
            for (const auto& f : b["fluid_fragments"]) {
                double v = 1.;
                for (int d = 0; d < 3; ++d) v *= f["high_m"][d].get<double>() - f["low_m"][d].get<double>();
                fragments += v;
            }
            require(std::abs(box_volume - fragments) < 1e-15,
                    "Nonrectangular cavity was collapsed into one mixed gas volume");
        }
        for (const auto& ch : scene.gas_channels) {
            if (ch.from < 0 || ch.to < 0 || ch.mixing_axis != 0) continue;
            const auto a = scene.gas_compartments[ch.from].id, b = scene.gas_compartments[ch.to].id;
            if (!boxes.contains(a) || !boxes.contains(b)) continue;
            const auto& ca = boxes[a]["fluid_centroid_m"];
            const auto& cb = boxes[b]["fluid_centroid_m"];
            if (std::abs(ca[0].get<double>()) < .001 && std::abs(cb[0].get<double>()) < .001 &&
                std::abs(ca[1].get<double>() + .012) < .011 && std::abs(ca[2].get<double>()) < .021) {
                ++narrow;
                require(ch.mixing_length_m < .002, "2 mm gap still has a background-grid transport length");
            }
        }
        if (dx == .02 || dx == .01) require(narrow > 0, "Missing cross-gap gas exchange");
        require(std::abs(compiled["metadata"]["resolved_gas_meshes"]["case"]["free_volume_m3"].get<double>() -
                         original["metadata"]["resolved_gas_meshes"]["case"]["free_volume_m3"].get<double>() -
                         .002 * .022 * .042) < 1e-12, "Deleted pad leaves an incorrect gas volume");
    }
}
