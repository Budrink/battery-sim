#include "battery_native/contracts/enclosure_openings.hpp"
#include "battery_native/contracts/experiment_contract.hpp"
#include "contract_internal.hpp"
#include <algorithm>
#include <cmath>
#include <set>

namespace battery {
namespace {
// Signed integral of a disk over [0,x] x [0,y]. Inclusion/exclusion gives
// exact circle/rectangle areas even when an opening straddles many gas cells.
double quadrant(double x, double y, double r) {
    const double sign = (x < 0 ? -1. : 1.) * (y < 0 ? -1. : 1.);
    x = std::min(std::abs(x), r);
    y = std::min(std::abs(y), r);
    const double split = std::sqrt(std::max(0., r * r - y * y));
    auto integral = [r](double t) {
        return .5 * (t * std::sqrt(std::max(0., r * r - t * t)) + r * r * std::asin(t / r));
    };
    return sign * (y * std::min(x, split) + (x > split ? integral(x) - integral(split) : 0.));
}
} // namespace
double EnclosureOpening::area() const {
    return circle ? std::acos(-1.) * size[0] * size[0] / 4 : size[0] * size[1];
}
double EnclosureOpening::overlap(std::array<double, 2> lo, std::array<double, 2> hi) const {
    for (int d = 0; d < 2; ++d) {
        lo[d] = std::max(lo[d] - center[d], -size[d] / 2);
        hi[d] = std::min(hi[d] - center[d], size[d] / 2);
        if (hi[d] <= lo[d])
            return 0;
    }
    if (!circle)
        return (hi[0] - lo[0]) * (hi[1] - lo[1]);
    const double r = size[0] / 2;
    return std::max(0., quadrant(hi[0], hi[1], r) - quadrant(lo[0], hi[1], r) - quadrant(hi[0], lo[1], r) +
                            quadrant(lo[0], lo[1], r));
}
std::vector<EnclosureOpening> parse_enclosure_openings(const nlohmann::json& mesh) {
    using namespace contract_detail;
    const auto list = mesh.value("openings", J::array());
    array(list, "case/openings");
    const auto dims = mesh.at("inside_dimensions_m").get<std::array<double, 3>>();
    std::vector<EnclosureOpening> result;
    std::set<std::string> ids;
    int counts[6]{};
    for (const auto& item : list) {
        object(item, "case/opening", {"id", "face", "shape", "center_m", "size_m"},
               {"id", "face", "shape", "center_m", "size_m"});
        EnclosureOpening h;
        h.id = id(item.at("id"), "opening/id");
        h.face = text(item.at("face"), "opening/face");
        if (!ids.insert(h.id).second || h.face.size() != 2 || (h.face[0] != '+' && h.face[0] != '-') ||
            std::string("xyz").find(h.face[1]) == std::string::npos)
            fail("opening", "Invalid face or duplicate ID");
        h.axis = h.face[1] - 'x';
        h.sign = h.face[0] == '+' ? 1 : -1;
        if (++counts[2 * h.axis + (h.sign > 0)] > 64)
            fail("openings", "Maximum 64 openings per face");
        const auto shape = text(item.at("shape"), "opening/shape");
        if (shape != "circle" && shape != "rectangle")
            fail("opening", "Expected circle or rectangle");
        h.circle = shape == "circle";
        for (const auto* key : {"center_m", "size_m"})
            if (!item.at(key).is_array() || item.at(key).size() != 2)
                fail("opening", "Expected 2D coordinates/size");
        h.center = item.at("center_m").get<std::array<double, 2>>();
        h.size = item.at("size_m").get<std::array<double, 2>>();
        for (int d = 0; d < 2; ++d) {
            if (!std::isfinite(h.center[d]) || !std::isfinite(h.size[d]) || h.size[d] <= 0 ||
                std::abs(h.center[d]) + h.size[d] / 2 > dims[(h.axis + 1 + d) % 3] / 2 + 1e-12)
                fail("opening", "Opening must fit entirely inside the case face");
        }
        if (h.circle && std::abs(h.size[0] - h.size[1]) > 1e-12)
            fail("opening", "Circle sizes must equal diameter");
        for (const auto& other : result) {
            if (other.face != h.face)
                continue;
            bool overlap = false;
            if (h.circle && other.circle)
                overlap = std::hypot(h.center[0] - other.center[0], h.center[1] - other.center[1]) <
                          (h.size[0] + other.size[0]) / 2 - 1e-12;
            else {
                const auto& shape = h.circle ? h : other;
                const auto& rect = h.circle ? other : h;
                overlap = shape.overlap(
                              {rect.center[0] - rect.size[0] / 2, rect.center[1] - rect.size[1] / 2},
                              {rect.center[0] + rect.size[0] / 2, rect.center[1] + rect.size[1] / 2}) > 1e-16;
            }
            if (overlap)
                fail("opening", "Openings overlap; move or resize the opening");
        }
        result.push_back(h);
    }
    return result;
}
nlohmann::json set_enclosure_openings(const nlohmann::json& input, const std::string& id,
                                      const nlohmann::json& openings) {
    auto next = upgrade_experiment(input);
    const auto registry = next.value("metadata", nlohmann::json::object())
                             .value("editor_enclosures", nlohmann::json::object());
    if (registry.contains(id) && registry.at(id).contains("explicit_mesh"))
        return edit_explicit_enclosure_openings(next, id, openings);
    auto& meshes = next.at("assembly").at("gas_meshes");
    if (!meshes.contains(id))
        throw std::invalid_argument("Create a case before placing openings");
    meshes[id]["openings"] = openings;
    (void)compile_experiment(next);
    if (next["provenance"]["status"] != "synthetic")
        next["provenance"]["status"] = "mixed";
    return next;
}
} // namespace battery
