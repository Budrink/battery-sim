#include "contract_internal.hpp"
#include <array>
#include <cmath>

namespace battery::contract_detail {
namespace {
using Vec = std::array<double, 3>;
Vec vector(const J& value, const std::string& where) {
    if (!value.is_array() || value.size() != 3)
        fail(where, "expected three coordinates");
    Vec result{};
    for (int i = 0; i < 3; ++i)
        result[i] = number(value[i], where, -1e9, 1e9);
    return result;
}
Vec rotate(const Vec& v, const J& q) {
    if (!q.is_array() || q.size() != 4)
        fail("port owner/pose", "expected quaternion xyzw");
    std::array<double, 4> r{};
    double norm = 0;
    for (int i = 0; i < 4; ++i) {
        r[i] = number(q[i], "orientation_xyzw", -1, 1);
        norm += r[i] * r[i];
    }
    if (std::abs(norm - 1) > 1e-9)
        fail("port owner/pose", "quaternion must be normalized");
    const Vec t{2 * (r[1] * v[2] - r[2] * v[1]), 2 * (r[2] * v[0] - r[0] * v[2]),
                2 * (r[0] * v[1] - r[1] * v[0])};
    return {v[0] + r[3] * t[0] + r[1] * t[2] - r[2] * t[1], v[1] + r[3] * t[1] + r[2] * t[0] - r[0] * t[2],
            v[2] + r[3] * t[2] + r[0] * t[1] - r[1] * t[0]};
}
} // namespace

void lower_local_ports(J& assembly) {
    std::map<std::string, J> owners;
    for (const auto* list : {"batteries", "passive_cells"}) {
        if (!assembly.contains(list))
            continue;
        array(assembly.at(list), list);
        for (const auto& cell : assembly.at(list))
            if (!owners.emplace(text(cell.at("id"), "port owner/id"), cell).second)
                fail("assembly", "duplicate port owner");
    }
    array(assembly.at("ports"), "ports");
    for (auto& port : assembly["ports"]) {
        object(
            port, "port",
            {"id", "cell_id", "frame", "kind", "surface", "area_m2", "rectangle_m", "centroid_m", "normal"},
            {"id", "cell_id", "frame", "area_m2", "centroid_m", "normal"});
        const auto frame = text(port.at("frame"), "port/frame");
        auto center = vector(port.at("centroid_m"), "port/centroid_m");
        auto normal = vector(port.at("normal"), "port/normal");
        const double norm = normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
        if (std::abs(norm - 1) > 1e-9)
            fail("port/normal", "normal must be unit length");
        number(port.at("area_m2"), "port/area_m2", 1e-20, 1e12);
        if (port.contains("rectangle_m")) {
            const auto& rect = port.at("rectangle_m");
            if (!rect.is_array() || rect.size() != 2)
                fail("port/rectangle_m", "expected two lengths");
            for (const auto& x : rect)
                number(x, "port/rectangle_m", 0, 1e9);
        }
        if (frame == "local") {
            const auto owner = text(port.at("cell_id"), "port/cell_id");
            if (!owners.contains(owner))
                fail(owner, "local port must belong to an explicit battery or passive object");
            const auto pose = owners.at(owner).value(
                "pose", J{{"position_m", {0, 0, 0}}, {"orientation_xyzw", {0, 0, 0, 1}}});
            const auto translation = vector(pose.at("position_m"), "pose/position_m");
            center = rotate(center, pose.at("orientation_xyzw"));
            normal = rotate(normal, pose.at("orientation_xyzw"));
            for (int i = 0; i < 3; ++i)
                center[i] += translation[i];
            port["centroid_m"] = center;
            port["normal"] = normal;
        } else if (frame != "world")
            fail("port/frame", "expected local or world");
        port.erase("frame");
    }
}
} // namespace battery::contract_detail
