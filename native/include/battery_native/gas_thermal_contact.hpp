#pragma once
#include "battery_native/thermal_mesh.hpp"
#include "battery_native/state_geometry.hpp"

namespace battery {
// Turn a supplied reference scale into the SAME directed contact law used by
// solids. No runtime scalar-film fallback exists. Reference state S=0 gives Gref.
inline ConnectivityLaw reference_gas_connectivity(double reference_g) {
    return reference_directed_connectivity(reference_g);
}
#ifdef __CUDACC__
__host__ __device__
#endif
inline double gas_contact_conductance(const GasThermalPatch& patch,double gas_temperature,
    double solid_temperature,double solid_damage,int direction=0) {
    return evaluate_directed_geometry(patch.law,{},patch.geometry_axis,
        gas_temperature,solid_temperature,0.,solid_damage,0.,patch.geometry_weight,0.,1.,direction).conductance_w_per_k;
}
}
