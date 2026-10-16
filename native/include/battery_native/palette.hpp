#pragma once

#include "battery_native/thermal_lut.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace battery {

enum class FieldLayer { Temperature, Degradation };
using Rgb = std::array<float, 3>;

// Cobalt blue #397DFF is reserved for destroyed material, not scalar values.
inline constexpr Rgb destroyed_rgb{57.0f / 255.0f, 125.0f / 255.0f, 1.0f};

inline Rgb mix_rgb(const Rgb& a, const Rgb& b, float t) {
    return {a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t};
}

inline float palette_position(float value) {
    return std::isnan(value) ? 0.0f : std::clamp(value, 0.0f, 1.0f);
}

inline Rgb temperature_rgb(float normalized) {
    const float position = palette_position(normalized) * (thermal_lut.size() - 1);
    const auto lower = static_cast<std::size_t>(position);
    const auto upper = std::min(lower + 1, thermal_lut.size() - 1);
    return mix_rgb(thermal_lut[lower], thermal_lut[upper], position - lower);
}

struct PaletteStop { float position; Rgb color; };
inline constexpr std::array<PaletteStop, 5> degradation_stops = {{
        {0.00f, {0.035f, 0.48f, 0.16f}},
        {0.35f, {0.30f, 0.72f, 0.10f}},
        {0.58f, {0.92f, 0.78f, 0.06f}},
        {0.78f, {0.96f, 0.32f, 0.035f}},
        {1.00f, {0.72f, 0.015f, 0.025f}},
}};
inline Rgb degradation_rgb(float normalized) {
    const auto& stops = degradation_stops;
    const float value = palette_position(normalized);
    std::size_t upper = 1;
    while (upper < stops.size() - 1 && value > stops[upper].position) ++upper;
    const auto& low = stops[upper - 1];
    const auto& high = stops[upper];
    return mix_rgb(low.color, high.color,
                   (value - low.position) / (high.position - low.position));
}

// A hot destroyed element retains its true temperature in the thermal layer.
// Its reserved state color is used on the outline, or as fill in degradation.
inline Rgb field_rgb(FieldLayer layer, float normalized, bool destroyed = false) {
    if (layer == FieldLayer::Temperature) return temperature_rgb(normalized);
    return destroyed ? destroyed_rgb : degradation_rgb(normalized);
}

}  // namespace battery
