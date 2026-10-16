#pragma once
#include "battery_native/thermal_mesh.hpp"

namespace battery {
// A separate geometric graph: gas nodes and pressure channels do NOT inherit
// thermal conductance. This map only establishes local phase co-occupancy.
struct MassBinding {
    int thermal_volume = -1;
    double initial_reactant_kg = 0., reactant_density = 0., residue_density = 0.;
    double geometric_volume_m3 = 0.;
};
struct MassMesh {
    std::vector<MassBinding> bindings;
    std::vector<int> gas_of_thermal_volume;
    std::vector<int> source_gas; // local source overrides legacy lumped plenum
};
MassMesh build_mass_mesh(const Scenario&, const ThermalMesh&);
}
