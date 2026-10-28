#include "battery_native/palette.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

float distance_squared(const battery::Rgb& a, const battery::Rgb& b) {
    float result = 0.0f;
    for (std::size_t channel = 0; channel < a.size(); ++channel)
        result += (a[channel] - b[channel]) * (a[channel] - b[channel]);
    return result;
}
}  // namespace

int main() {
    try {
        using namespace battery;
        require(temperature_rgb(0)[2]>.5f && temperature_rgb(0)[0]<.1f,"Cold endpoint must be blue");
        require(temperature_rgb(1)[0]>.9f && temperature_rgb(1)[1]<.05f,"Hot endpoint must be saturated red");
        require(temperature_rgb(.6f)[0]>.95f && temperature_rgb(.6f)[1]>.85f && temperature_rgb(.6f)[2]<.1f,"Intermediate heat must be yellow");
        require(temperature_rgb(-1) == temperature_rgb(0), "Clamp cold values");
        require(temperature_rgb(2) == temperature_rgb(1), "Clamp hot values");
        require(degradation_rgb(-1) == degradation_rgb(0), "Clamp healthy values");
        require(degradation_rgb(2) == degradation_rgb(1), "Clamp critical values");
        require(temperature_rgb(std::numeric_limits<float>::quiet_NaN()) ==
                    temperature_rgb(0), "Nonfinite input must not index outside LUT");
        require(temperature_rgb(std::numeric_limits<float>::infinity()) ==
                    temperature_rgb(1), "Infinite input must clamp");
        for (std::size_t i = 0; i < thermal_lut.size(); ++i)
            require(distance_squared(temperature_rgb(static_cast<float>(i) / 255),
                                     thermal_lut[i]) < 1.0e-10f,
                    "CPU and CUDA must use the same palette samples");
        for (int i = 0; i <= 4096; ++i) {
            const float value = i / 4096.0f;
            const auto heat = temperature_rgb(value);
            for (auto layer : {FieldLayer::Temperature, FieldLayer::Degradation}) {
                const auto color = field_rgb(layer, value);
                for (float channel : color)
                    require(std::isfinite(channel) && channel >= 0 && channel <= 1,
                            "Palette channels must be finite and in range");
                require(distance_squared(color, destroyed_rgb) > 0.10f,
                        "Destroyed purple must remain outside both scalar palettes");
            }
            require(field_rgb(FieldLayer::Temperature, value, true) == heat,
                    "Destruction must not hide residual temperature");
            require(field_rgb(FieldLayer::Degradation, value, true) == destroyed_rgb,
                    "Destroyed degradation fill must use the reserved color");
        }
        std::puts("Thermal/degradation palettes and destruction semantics OK");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
