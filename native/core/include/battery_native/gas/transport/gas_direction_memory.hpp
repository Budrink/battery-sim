#pragma once
#include <cmath>
#ifdef __CUDACC__
#define GDM_HD __host__ __device__
#else
#define GDM_HD
#endif

// Reduced staggered momentum network; not a resolved jet/CFD model.
namespace battery::gas_direction {
struct Memory {
    double flow_kg_s = 0.; // signed A -> B, last ACCEPTED backward-Euler flow
    double kinetic_j = 0.; // paid by the two adjacent thermal volumes
};
struct Direction {
    double x = 0., y = 0., z = 0.; // unit A -> B
    double length_m = 0.;          // physical dual-channel length
};
struct Arrival {
    double weight = 0., x = 0., y = 0., z = 0.; // incoming momentum-flux weights
};
struct Operator {
    double mobility = 0., bias = 0.; // q = mobility * (pA-pB) + bias
};
GDM_HD inline double turn_loss(const Arrival& in, const Direction& out, bool forward) {
    if (!(in.weight > 0.))
        return 0.;
    const double projection = (in.x * out.x + in.y * out.y + in.z * out.z) / in.weight;
    // Inelastic, equal-speed redirection closure: |u*n_in-u*n_out|^2/u^2.
    // Straight=0, right angle=2, reversal=4. Opposing jets still dissipate.
    return 2. * (1. - fmin(1., fmax(-1., forward ? projection : -projection)));
}
GDM_HD inline double laminar_resistance(double viscosity, double length, double area, double density) {
    // Poiseuille loss of an equivalent circular passage: explicit geometric
    // approximation, not the exact drag of an arbitrary cavity.
    return 8. * 3.14159265358979323846 * viscosity * length / (density * area * area);
}
GDM_HD inline Operator implicit_flow(double old_flow, double dp, double dt, double inertance,
                                     double linear_drag, double quadratic_drag) {
    // I*(q-q0)/dt + R*q + K*q*|q| = dp. Analytic monotone scalar solve;
    // no physical substeps even when dt >> the physical relaxation time.
    const double history = inertance / dt;
    const double linear = history + linear_drag;
    const double force = dp + history * old_flow;
    const double magnitude =
        2. * fabs(force) / (linear + sqrt(linear * linear + 4. * quadratic_drag * fabs(force)));
    const double mobility = 1. / (linear + quadratic_drag * magnitude);
    return {mobility, mobility * history * old_flow};
}
GDM_HD inline double kinetic_energy(double flow, double length, double area, double density) {
    return .5 * length / (density * area) * flow * flow;
}
GDM_HD inline double relaxation_time(double inertance, double linear_drag, double quadratic_drag,
                                     double flow) {
    return inertance / (linear_drag + 2. * quadratic_drag * fabs(flow));
}
} // namespace battery::gas_direction
#undef GDM_HD
