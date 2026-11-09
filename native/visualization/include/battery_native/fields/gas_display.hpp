#pragma once
#include "battery_native/data/visual_frame.hpp"
namespace battery {
struct GasDisplaySample {
    Vec3 temperature_gradient, displacement;
    float minimum_k = 298.15f, maximum_k = 298.15f;
    // Local accepted R3 release / physical observation interval, not heat
    // advected into this region and not an inference from its temperature.
    double combustion_power_w = 0, combustion_power_density_w_m3 = 0;
};
struct GasPuff {
    Vec3 position;
    float radius_m, temperature_k, opacity;
    Vec3 direction; // display heading; not a CFD velocity
};
struct GasDisplayState {
    std::vector<GasDisplaySample> regions;
    std::vector<GasPuff> puffs;
};
bool in_gas_region(const GasVisualRegion&, Vec3 point);
// Read-only display reconstruction: gradients are limited to connected gas
// neighbours. Markers follow accepted net transfer AND each-way mixing,
// never arbitrary wall time. Mixing rate uses consecutive physical samples;
// the first frame after a seek has no inferred mixing rate.
class GasDisplayModel {
  public:
    GasDisplayModel(const Scenario&, const ThermalMesh&, const VisualGeometry&);
    GasDisplayState sample(const VisualFrame&, bool include_markers = true);

  private:
    struct Link {
        int channel, from, to;
        Vec3 a, b;
        double area;
        bool rupture;
        double display_mass_kg = .0005;
    };
    const VisualGeometry& geometry_;
    std::vector<int> gas_to_region_;
    std::vector<Link> links_;
    double previous_time_s_ = -1;
    std::vector<double> previous_mixed_kg_, mixing_rates_kg_s_;
    std::vector<bool> previous_known_;
    double combustion_time_s_ = -1;
    std::vector<double> previous_release_j_, combustion_power_w_, reference_volume_m3_;
};
} // namespace battery
