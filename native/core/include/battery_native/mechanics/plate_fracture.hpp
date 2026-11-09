#pragma once
#include <cmath>
#ifdef __CUDACC__
#define PLATE_HD __host__ __device__
#else
#define PLATE_HD
#endif
namespace battery {
// Reduced connected-front model, not an elastic crack-tip solver. Access owns
// the planar adjacency and local pressure/strength. Opening is an AREA fraction;
// thermal damage is separate. No averaging of the load across the plate.
// A tile fills before the front can cross it into a face-adjacent tile.
// Several frontier branches may compete when their local loads change.
// Rates refer to the remaining area of the WHOLE plate, not to a mesh tile.
// Event integration can cross several tiles in one physical step, so advancing
// the front is not artificially limited to one neighbour per solver iteration.
template <class Access> PLATE_HD void advance_plate_front(Access& a, double dt) {
    double total = 0., remaining = 0.;
    for (int i = 0; i < a.size(); ++i) {
        total += a.area(i);
        remaining += a.area(i) * (1. - a.open(i));
    }
    if (!(total > 0.) || !(dt > 0.))
        return;
    int order = 0;
    // Each tile has at most two events: weak region and the surrounding matrix.
    for (int event = 0; event < 2 * a.size() + 1 && dt > 0.; ++event) {
        bool started = false;
        for (int i = 0; i < a.size(); ++i) {
            started |= a.open(i) > 0.;
        }
        int selected = -1;
        double best = 1.;
        for (int i = 0; i < a.size(); ++i) {
            const double opened = a.open(i);
            if (opened >= 1.)
                continue;
            const bool eligible = !started || opened > 0. || a.touches_open(i);
            const double load = a.load_ratio(i);
            if (eligible && load > best) {
                best = load;
                selected = i;
            }
        }
        if (selected < 0)
            break; // unloading arrests the crack, never heals it
        const int i = selected;
        const double rate = a.rate(i);
        if (!(rate > 0.) || !(remaining > 0.))
            break;
        const double opened = a.open(i), weak = a.weak_fraction(i);
        const double target = opened < weak ? weak : 1.;
        const double capacity = (target - opened) * a.area(i);
        const double amount = remaining * (-expm1(-rate * dt));
        if (amount < capacity) {
            a.advance(i, amount / a.area(i), order);
            break;
        }
        const double crossing = capacity < remaining ? -log1p(-capacity / remaining) / rate : dt;
        a.advance(i, target - opened, order++);
        remaining = fmax(0., remaining - capacity);
        dt = fmax(0., dt - crossing);
    }
}
} // namespace battery
#undef PLATE_HD
