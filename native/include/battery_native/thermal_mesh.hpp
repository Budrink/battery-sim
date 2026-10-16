#pragma once
#include "battery_native/model.hpp"
#include <vector>

namespace battery {
struct GasThermalPatch {
    int volume, compartment;
    double conductance_w_per_k;
    // channel < 0: ordinary compartment/solid contact. Otherwise heat is
    // taken from the channel's arriving enthalpy, NOT added to gas discharge.
    int channel = -1, barrier_volume = -1;
    double aperture_area_fraction = 0., seam_fraction = 0.;
};
// Numerical regions, NOT a claim about physical casing thickness or material.
enum class ThermalRegion { Core, Shell };
enum class ThermalLinkRegion { CoreCore, CoreShell, ShellShell, MaterialContact, Boundary };
inline ThermalRegion volume_region(int x,int y,int z,const std::array<int,3>& n) {
    return x==0 || y==0 || z==0 || x==n[0]-1 || y==n[1]-1 || z==n[2]-1
        ? ThermalRegion::Shell : ThermalRegion::Core;
}
// Host-built topology; all evolving fields live in CUDA device memory.
struct ThermalVolume {
    int object = 0;
    Vec3 local_center_m;
    Vec3 size_m;
    double capacity_j_per_k = 0;
    double initial_temperature_k = 0;
    ThermalRegion region = ThermalRegion::Shell;
};
struct ThermalLink {
    int a = 0, b = -1; // b=-1 is a prescribed-temperature boundary
    int interface_index = -1;
    ConnectivityLaw law;
    InterfaceState initial;
    double weight = 1.0;
    double series_resistance_k_per_w = 0;
    double series_resistance_a_k_per_w = 0; // endpoint-a share; b is total minus a
    double boundary_temperature_k = 298.15;
    double maximum_g = 0;
    int boundary_index = -1;
    int geometry_axis = 0; // direction in the aligned local material frames
    ThermalLinkRegion region = ThermalLinkRegion::ShellShell;
};
struct ThermalMesh {
    std::vector<GasThermalPatch> gas_patches;
    std::vector<ThermalVolume> volumes;
    std::vector<ThermalLink> links;
    std::vector<int> object_offsets;
    std::vector<int> adjacency_offsets, adjacency_links;
    std::vector<int> interface_offsets, interface_links;
};
ThermalMesh build_thermal_mesh(const Scenario& scenario);
// Fixed vents use their orifice area. Progressive rupture exposes the actual
// panel face; it has no independent fitted maximum aperture.
double gas_channel_area(const Scenario&, const GasChannel&);
// Static visibility only; opening and transport are evaluated on the GPU.
void append_aperture_surface_links(const Scenario&, ThermalMesh&);
} // namespace battery
