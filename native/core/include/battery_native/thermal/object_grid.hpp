#pragma once
#include "battery_native/model/model.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace battery {
// Local-coordinate edges of the host-built Cartesian mesh. Casing tiles inherit
// the contents' tangential cuts (or integer refinements), not the expanded outer
// rectangle's uniform cuts. Only the edge tiles own the extra corner strips.
// This preserves the closed shell's mass without crossing an interior cut.
using ObjectGrid = std::array<std::vector<double>, 3>;
inline ObjectGrid object_grid(const Scenario& scene, const CellConfig& cell) {
    const auto core = std::find_if(scene.cells.begin(), scene.cells.end(), [&](const auto& c) {
        return !cell.shell_contents_id.empty() && c.id == cell.shell_contents_id;
    });
    if (!cell.shell_contents_id.empty() && core == scene.cells.end())
        throw std::runtime_error("Missing shell contents: " + cell.id);
    const double origin[]{cell.position_m.x, cell.position_m.y, cell.position_m.z};
    ObjectGrid grid;
    for (int d = 0; d < 3; ++d) {
        const int n = cell.thermal_divisions[d];
        if (n < 1 || n > 256)
            throw std::runtime_error("thermal_divisions must be in [1,256]: " + cell.id);
        double width = cell.dimensions_m.at(d), offset = 0.;
        if (core != scene.cells.end()) {
            const double center[]{core->position_m.x, core->position_m.y, core->position_m.z};
            const double touching_distance = (core->dimensions_m[d] + cell.dimensions_m[d]) / 2;
            const bool tangent = std::abs(origin[d] - center[d]) < touching_distance * (1. - 1e-12);
            if (tangent) {
                if (core->thermal_divisions[d] < 1 || n % core->thermal_divisions[d] != 0)
                    throw std::runtime_error(
                        "Shell surface grid must be an integer refinement of contents: " + cell.id);
                width = core->dimensions_m[d];
                offset = center[d] - origin[d];
            } else if (n != 1)
                throw std::runtime_error("Shell must have one layer through thickness: " + cell.id);
        }
        auto& edges = grid[d];
        edges.resize(n + 1);
        for (int i = 0; i <= n; ++i)
            edges[i] = offset + width * (double(i) / n - .5);
        edges.front() = -cell.dimensions_m[d] / 2;
        edges.back() = cell.dimensions_m[d] / 2;
        for (int i = 0; i < n; ++i)
            if (!(edges[i + 1] > edges[i]))
                throw std::runtime_error("Invalid aligned object grid: " + cell.id);
    }
    return grid;
}
inline std::array<int, 3> grid_coordinates(const CellConfig& cell, int local) {
    const auto n = cell.thermal_divisions;
    return {local % n[0], (local / n[0]) % n[1], local / (n[0] * n[1])};
}
} // namespace battery
