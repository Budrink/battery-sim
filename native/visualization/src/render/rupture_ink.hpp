#pragma once
#include <glm/glm.hpp>
#include <algorithm>
#include <cmath>
#include <vector>
namespace battery::rupture_ink {
struct Site {
    glm::vec2 center, half_size;
    float fraction;
};
// Display-only smooth union of the accepted local opening fractions. This is
// NOT a resolved crack outline and never changes the physical aperture area.
// Noise is fixed in plate coordinates: paused frames do not flicker.
inline std::vector<Site> prepare(const std::vector<Site>& sites) {
    auto result = sites;
    for (std::size_t i = 0; i < sites.size(); ++i) {
        const auto& site = sites[i];
        const float size = std::sqrt(site.fraction);
        // Join the active partial region to an already opened neighbouring tile.
        for (const auto& other : sites) {
            if (other.fraction < .999f)
                continue;
            const auto gap = other.center - site.center;
            for (int axis = 0; axis < 2; ++axis) {
                const int tangent = 1 - axis;
                if (std::abs(std::abs(gap[axis]) - other.half_size[axis] - site.half_size[axis]) < 1e-6f &&
                    std::abs(gap[tangent]) < other.half_size[tangent] + site.half_size[tangent] - 1e-6f) {
                    result[i].center[axis] += std::copysign(site.half_size[axis] * (1 - size), gap[axis]);
                    break;
                }
            }
        }
    }
    return result;
}
inline glm::vec4 sample(glm::vec2 p, glm::vec2 plate_size, const std::vector<Site>& sites) {
    float field = 0;
    for (const auto& site : sites) {
        if (site.fraction <= 0)
            continue;
        const auto q = (p - site.center) /
                       glm::max(site.half_size * (1.1f * std::sqrt(site.fraction)), glm::vec2(1e-8f));
        const float x = q.x * q.x, y = q.y * q.y;
        field += std::exp(-.7f * (x * x + y * y));
    }
    const auto q = p / glm::max(plate_size, glm::vec2(1e-8f));
    const float rough = .075f * std::sin(73 * q.x + 19 * q.y) * std::sin(97 * q.y - 31 * q.x) +
                        .035f * std::sin(181 * q.x + 127 * q.y);
    const float a = std::clamp((field + rough - .16f) / .65f, 0.f, 1.f);
    const float smooth = a * a * (3 - 2 * a);
    const float shade = .025f + .10f * (1 - smooth);
    // Visible dark core; the same smooth contour still fades to transparency.
    constexpr float core_opacity = .65f;
    return {shade, shade * .95f, shade * .9f, core_opacity * smooth};
}
} // namespace battery::rupture_ink
