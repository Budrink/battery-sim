// Migration only: old FSRI contracts omitted the case editor descriptor and
// internal gas display boxes. No scenario-specific branch belongs in the UI or
// the generic opening editor. Reconstruct the ORIGINAL grid, never remesh it.
#include "battery_native/contracts/enclosure_openings.hpp"
#include <nlohmann/json.hpp>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace battery {
nlohmann::json register_imported_enclosures(const nlohmann::json& input) {
    using J = nlohmann::json;
    const auto meta = input.value("metadata", J::object());
    if (!meta.contains("editor_import") || !meta.contains("aligned_inner_gas") ||
        meta.value("editor_enclosures", J::object()).contains("ABS")) return input;
    auto e = input;
    const auto outer = meta.at("geometry_reconstruction").at("enclosure_dimensions_m").get<std::array<double, 3>>();
    const double t = meta.at("geometry_budget").at("abs_thickness_m");
    const double step = meta.at("aligned_inner_gas").at("max_step_m");
    std::array<std::vector<double>, 3> axes;
    for (int d = 0; d < 2; ++d) {
        const double size = outer[d] - 2*t;
        const int count = std::max(1, int(std::ceil(size / step - 1e-12)));
        for (int n = 0; n <= count; ++n) axes[d].push_back(-size/2 + size*n/count);
    }
    axes[2] = meta.at("aligned_inner_gas").at("axial_boundaries_m").get<std::vector<double>>();
    for (const auto& node : e.at("assembly").at("gas_network").at("compartments")) {
        const std::string id = node.at("id");
        if (!id.starts_with("aligned-inner-")) continue;
        int index[3];
        if (std::sscanf(id.c_str(), "aligned-inner-%d-%d-%d", &index[0], &index[1], &index[2]) != 3)
            throw std::invalid_argument("Invalid archived gas grid ID");
        std::array<double, 3> lo{}, hi{};
        for (int d = 0; d < 3; ++d) {
            if (index[d] < 0 || index[d] + 1 >= int(axes[d].size()))
                throw std::invalid_argument("Archived gas cell outside original grid");
            lo[d] = axes[d][index[d]]; hi[d] = axes[d][index[d]+1];
        }
        e["metadata"]["gas_volume_geometry"][id] = {{"low_m", lo}, {"high_m", hi}};
    }
    J routes = J::array(), law;
    double area = 0;
    for (const auto& ch : e.at("assembly").at("gas_network").at("channels"))
        if (ch.at("id").get<std::string>().starts_with("aligned-vent-")) {
            routes.push_back(ch.at("id")); law = ch;
            area += ch.at("area_m2").get<double>();
        }
    const double diameter = meta.at("aligned_inner_gas").at("aperture_diameter_m");
    if (routes.empty() || std::abs(area - std::acos(-1.)*diameter*diameter/4) > 1e-10)
        throw std::invalid_argument("Archived case opening does not match gas routes");
    const J holes = {{{"id", "top-opening"}, {"face", "+z"}, {"shape", "circle"},
                      {"center_m", {0., 0.}}, {"size_m", {diameter, diameter}}}};
    e = register_explicit_enclosure(e, "ABS", holes, routes, law);
    // Historical generator subtracted the aperture only from inner wall films.
    e["metadata"]["editor_enclosures"]["ABS"]["reference_outer_contacts_uncut"] = true;
    return e;
}
} // namespace battery
