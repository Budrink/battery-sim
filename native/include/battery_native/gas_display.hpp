#pragma once
#include "battery_native/visual_frame.hpp"
namespace battery {
struct GasDisplaySample {
    Vec3 temperature_gradient, displacement;
    float minimum_k=298.15f,maximum_k=298.15f;
};
struct GasPuff {Vec3 position;float radius_m,temperature_k,opacity;};
struct GasDisplayState {
    std::vector<GasDisplaySample> regions;
    std::vector<GasPuff> puffs;
};
bool in_gas_region(const GasVisualRegion&,Vec3 point);
// Read-only display reconstruction: gradients are limited to connected gas
// neighbours. Tracers follow accepted net transfer, not arbitrary wall time.
class GasDisplayModel {
public:
    GasDisplayModel(const Scenario&,const ThermalMesh&,const VisualGeometry&);
    GasDisplayState sample(const VisualFrame&) const;
private:
    struct Link {int channel,from,to;Vec3 a,b;double area;bool rupture;double display_mass_kg=.0005;};
    const VisualGeometry& geometry_;
    std::vector<int> gas_to_region_;
    std::vector<Link> links_;
};
}
