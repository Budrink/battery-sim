#include "battery_native/fields/gas_display.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace battery {
namespace {
// Presentation constants only. They never enter the solver or a calibration.
Vec3 add(Vec3 a, Vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
Vec3 sub(Vec3 a, Vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
Vec3 mul(Vec3 a, double k) {
    return {a.x * k, a.y * k, a.z * k};
}
double dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
double length(Vec3 a) {
    return std::sqrt(dot(a, a));
}
Vec3 anchor(const GasVisualRegion& g) {
    if (in_gas_region(g, g.center))
        return g.center;
    Vec3 best = g.center;
    double distance = 1e30;
    for (int x = 0; x < 9; ++x)
        for (int y = 0; y < 9; ++y)
            for (int z = 0; z < 3; ++z) {
                const Vec3 p =
                    add(g.center, {g.size.x * ((x + .5) / 9. - .5), g.size.y * ((y + .5) / 9. - .5),
                                   g.size.z * ((z + .5) / 3. - .5)});
                const double d = length(sub(p, g.center));
                if (d < distance && in_gas_region(g, p)) {
                    distance = d;
                    best = p;
                }
            }
    if (distance == 1e30)
        throw std::runtime_error("No visible free gas support");
    return best;
}
float hash(unsigned n) {
    n ^= n >> 16;
    n *= 0x7feb352d;
    n ^= n >> 15;
    n *= 0x846ca68b;
    n ^= n >> 16;
    return (n & 0xffffff) / float(0x1000000);
}
} // namespace
bool in_gas_region(const GasVisualRegion& g, Vec3 p) {
    const auto d = sub(p, g.center);
    if (std::abs(d.x) > g.size.x * .5 + 1e-9 || std::abs(d.y) > g.size.y * .5 + 1e-9 ||
        std::abs(d.z) > g.size.z * .5 + 1e-9)
        return false;
    const auto e = sub(p, g.excluded_center);
    for (const auto& c : g.excluded_cylinders) {
        const auto q = sub(p, c.center);
        if (q.x * q.x + q.y * q.y < c.radius_m * c.radius_m && std::abs(q.z) < c.height_m * .5)
            return false;
    }
    if (g.excluded_size.x > 0 && std::abs(e.x) < g.excluded_size.x * .5 &&
        std::abs(e.y) < g.excluded_size.y * .5 && std::abs(e.z) < g.excluded_size.z * .5)
        return false;
    bool positive = false, negative = false;
    for (std::size_t i = 0; i < g.polygon.size(); ++i) {
        const auto a = g.polygon[i], b = g.polygon[(i + 1) % g.polygon.size()];
        const double v = (b[0] - a[0]) * (p.y - a[1]) - (b[1] - a[1]) * (p.x - a[0]);
        positive |= v > 1e-9;
        negative |= v < -1e-9;
    }
    return !(positive && negative);
}
GasDisplayModel::GasDisplayModel(const Scenario& s, const ThermalMesh& mesh, const VisualGeometry& geometry)
    : geometry_(geometry) {
    for (const auto& gas : s.gas_compartments) reference_volume_m3_.push_back(gas.volume_m3);
    // Cache topology once, independently of changing temperatures and flows.
    gas_to_region_.assign(s.gas_compartments.size(), -1);
    std::vector<std::vector<int>> pieces(s.gas_compartments.size());
    std::vector<Vec3> anchors;
    for (std::size_t i = 0; i < geometry.gas.size(); ++i) {
        gas_to_region_[geometry.gas[i].compartment] = int(i);
        pieces[geometry.gas[i].compartment].push_back(int(i));
        anchors.push_back(anchor(geometry.gas[i]));
    }
    for (std::size_t k = 0; k < s.gas_channels.size(); ++k) {
        const auto& c = s.gas_channels[k];
        if (c.from < 0 || c.to < 0)
            continue;
        int a = gas_to_region_[c.from], b = gas_to_region_[c.to];
        // Cut-cell fragments share one inventory. Select local supports for
        // flow annotations, not whichever fragment happened to be listed last.
        double best = std::numeric_limits<double>::infinity();
        for (int i : pieces[c.from])
            for (int j : pieces[c.to]) {
                const double distance = length(sub(anchors[i], anchors[j]));
                if (distance < best) {
                    best = distance;
                    a = i;
                    b = j;
                }
            }
        if (a >= 0 && b >= 0 && c.area_m2 > 0)
            links_.push_back({int(k), a, b, anchors[a], anchors[b], c.area_m2, c.rupture});
        else if (c.rupture && b >= 0 && c.shell_object >= 0 && c.shell_thermal_volume >= 0) {
            const auto port =
                std::find_if(s.ports.begin(), s.ports.end(), [&](auto& p) { return p.id == c.port_id; });
            if (port == s.ports.end())
                continue;
            const auto& patch = mesh.volumes.at(mesh.object_offsets[c.shell_object] + c.shell_thermal_volume);
            auto p = add(s.cells[c.shell_object].position_m, patch.local_center_m);
            if (std::abs(port->normal.x) > .5)
                p.x = port->centroid_m.x + port->normal.x * 1e-6;
            if (std::abs(port->normal.y) > .5)
                p.y = port->centroid_m.y + port->normal.y * 1e-6;
            if (std::abs(port->normal.z) > .5)
                p.z = port->centroid_m.z + port->normal.z * 1e-6;
            p = project_visual_shell_point(s, geometry, c.shell_object, p);
            for (int j : pieces[c.to]) {
                const double distance = length(sub(p, anchors[j]));
                if (distance < best) {
                    best = distance;
                    b = j;
                }
            }
            links_.push_back({int(k), -1, b, p, anchors[b], gas_channel_area(s, c), true});
        }
    }
    for (auto& link : links_) {
        const double b = s.gas_compartments[geometry.gas[link.to].compartment].initial_mass_kg;
        const double a =
            link.from >= 0 ? s.gas_compartments[geometry.gas[link.from].compartment].initial_mass_kg : 0;
        const double reference = a > 0 && b > 0 ? std::min(a, b) : std::max(a, b);
        // A fixed reference inventory per link, not a changing current mass:
        // tiny near-cell cavities should not need a whole 0.5g to show motion.
        // Fixed normalization also avoids cloud jumps when the inventory drops.
        if (reference > 0)
            link.display_mass_kg = std::clamp(reference, 5e-6, 5e-4);
    }
}
GasDisplayState GasDisplayModel::sample(const VisualFrame& frame, bool include_markers) {
    // Reaction ledgers stay with the control volume, unlike transported heat.
    // Never use the cumulative total itself as flame brightness. On a seek or
    // the first observation the rate is unknown; do not fabricate ignition.
    const auto gases = frame.process.gas.size();
    const bool reset = combustion_time_s_ < 0 || frame.time_s < combustion_time_s_ ||
                       previous_release_j_.size() != gases;
    if (reset) {
        previous_release_j_.resize(gases);
        combustion_power_w_.assign(gases, 0);
    }
    if (reset || frame.time_s > combustion_time_s_) {
        const double dt = frame.time_s - combustion_time_s_;
        for (std::size_t i = 0; i < gases; ++i) {
            const double energy = frame.process.gas[i].released_energy_j;
            combustion_power_w_[i] = reset ? 0 : std::max(0., energy - previous_release_j_[i]) / dt;
            previous_release_j_[i] = energy;
        }
        combustion_time_s_ = frame.time_s;
    }
    // Mixing has a cumulative EACH-WAY ledger, not a net instantaneous flux.
    // Derive display rates from physical sample times. Pause preserves the last
    // frame; seeking backward/first sample starts a new observation interval.
    const auto count_channels = frame.process.channels.size();
    if (previous_time_s_ < 0 || frame.time_s < previous_time_s_ ||
        previous_mixed_kg_.size() != count_channels) {
        previous_time_s_ = frame.time_s;
        previous_mixed_kg_.resize(count_channels);
        previous_known_.resize(count_channels);
        mixing_rates_kg_s_.assign(count_channels, 0.);
        for (std::size_t i = 0; i < count_channels; ++i) {
            previous_mixed_kg_[i] = frame.process.channels[i].mixed_mass_kg;
            previous_known_[i] = frame.known_channels.empty() || frame.known_channels[i];
        }
    } else if (frame.time_s > previous_time_s_) {
        const double dt = frame.time_s - previous_time_s_;
        for (std::size_t i = 0; i < count_channels; ++i) {
            const bool known = frame.known_channels.empty() || frame.known_channels[i];
            const double mass = frame.process.channels[i].mixed_mass_kg;
            mixing_rates_kg_s_[i] =
                known && previous_known_[i] ? std::max(0., mass - previous_mixed_kg_[i]) / dt : 0.;
            previous_mixed_kg_[i] = mass;
            previous_known_[i] = known;
        }
        previous_time_s_ = frame.time_s;
    }
    GasDisplayState out;
    out.regions.resize(geometry_.gas.size());
    struct Sum {
        Vec3 gradient;
        double weights[3]{};
    };
    std::vector<Sum> sums(geometry_.gas.size());
    for (std::size_t i = 0; i < geometry_.gas.size(); ++i) {
        const auto index = geometry_.gas[i].compartment;
        const auto& gas = frame.process.gas.at(index);
        const double volume = gas.free_volume_m3 > 0 ? gas.free_volume_m3 : reference_volume_m3_.at(index);
        out.regions[i].combustion_power_w = combustion_power_w_.at(index);
        out.regions[i].combustion_power_density_w_m3 = volume > 0 ? combustion_power_w_.at(index) / volume : 0;
        out.regions[i].minimum_k = out.regions[i].maximum_k =
            float(frame.process.gas.at(geometry_.gas[i].compartment).temperature_k);
    }
    for (const auto& link : links_) {
        const auto& c = frame.process.channels.at(link.channel);
        if (link.rupture && c.effective_area_m2 <= 0)
            continue;
        auto& to = out.regions[link.to];
        const auto& gas_b = frame.process.gas[geometry_.gas[link.to].compartment];
        // Reconstruction uses connected neighbours only. Final shader samples
        // are clamped to these bounds, so display interpolation cannot create
        // temperatures outside the input field's local range.
        if (link.from >= 0) {
            const auto& gas_a = frame.process.gas[geometry_.gas[link.from].compartment];
            const auto d = sub(geometry_.gas[link.to].center, geometry_.gas[link.from].center);
            const double w = 1. / std::max(dot(d, d), 1e-12), dt = gas_b.temperature_k - gas_a.temperature_k;
            for (int i : {link.from, link.to}) {
                sums[i].gradient = add(sums[i].gradient, mul(d, dt * w));
                sums[i].weights[0] += d.x * d.x * w;
                sums[i].weights[1] += d.y * d.y * w;
                sums[i].weights[2] += d.z * d.z * w;
                out.regions[i].minimum_k = std::min(
                    {out.regions[i].minimum_k, float(gas_a.temperature_k), float(gas_b.temperature_k)});
                out.regions[i].maximum_k = std::max(
                    {out.regions[i].maximum_k, float(gas_a.temperature_k), float(gas_b.temperature_k)});
            }
        }
        if (!frame.known_channels.empty() && !frame.known_channels[link.channel])
            continue;
        const double flow = c.mass_flow_kg_s;
        // Parcel markers use an explicit DISPLAY reference-inventory scale,
        // not an assertion of resolved gas velocity. Cumulative accepted mass
        // gives deterministic pause/seek and the correct reversal direction.
        const double phase = c.transferred_kg / link.display_mass_kg;
        const auto direction = sub(link.b, link.a);
        const double route_length = length(direction);
        const auto drift = mul(direction, phase);
        to.displacement = add(to.displacement, mul(drift, .5));
        if (link.from >= 0)
            out.regions[link.from].displacement = add(out.regions[link.from].displacement, mul(drift, .5));
        if (!include_markers)
            continue; // Keep volume motion/gradients; no parcels along graph edges.
        const float radius = std::clamp(
            float(std::sqrt(std::max(c.effective_area_m2, link.area) / 3.14159265) * .4), .0006f, .0035f);
        auto emit = [&](double phase, double rate, double heading, int count, double lane) {
            if (rate < 1e-10 || route_length < 1e-12)
                return;
            const float strength = std::clamp(float(std::log1p(rate * 1e6) / 6.), 0.f, 1.f);
            Vec3 across = std::abs(direction.z) < route_length * .9 ? Vec3{-direction.y, direction.x, 0}
                                                                    : Vec3{1, 0, 0};
            across = mul(across, lane * radius * .65 / std::max(length(across), 1e-12));
            for (int k = 0; k < count; ++k) {
                const double raw = phase + k / double(count), u = raw - std::floor(raw);
                auto p = add(link.a, mul(direction, u));
                const unsigned seed = unsigned(link.channel) * 7919u + unsigned(k) * 109u;
                const double jitter = std::min(double(radius), route_length * .12);
                const auto displaced =
                    add(add(p, across), {(hash(seed) - .5) * jitter, (hash(seed + 1) - .5) * jitter,
                                         (hash(seed + 2) - .5) * jitter});
                auto valid = [&](Vec3 x) {
                    return in_gas_region(geometry_.gas[link.to], x) ||
                           (link.from >= 0 && in_gas_region(geometry_.gas[link.from], x));
                };
                if (valid(displaced))
                    p = displaced;
                else if (!valid(p))
                    continue;
                const double ta = link.from >= 0
                                      ? frame.process.gas[geometry_.gas[link.from].compartment].temperature_k
                                      : gas_b.temperature_k;
                const float temperature = float(ta + (gas_b.temperature_k - ta) * u);
                out.puffs.push_back({p, radius, temperature,
                                     (.07f + strength * .30f) * float(std::sin(3.14159265 * u)),
                                     mul(direction, heading / route_length)});
            }
        };
        emit(phase, std::abs(flow), flow >= 0 ? 1 : -1, link.rupture ? 18 : 8, 0);
        const double mixing_phase = c.mixed_mass_kg / link.display_mass_kg;
        const double mixing_rate = mixing_rates_kg_s_[link.channel];
        emit(mixing_phase, mixing_rate, 1, 4, 1);
        emit(-mixing_phase + .5, mixing_rate, -1, 4, -1);
    }
    for (std::size_t i = 0; i < out.regions.size(); ++i) {
        auto& g = out.regions[i].temperature_gradient;
        const auto& sum = sums[i];
        g = {sum.gradient.x / std::max(sum.weights[0], 1.), sum.gradient.y / std::max(sum.weights[1], 1.),
             sum.gradient.z / std::max(sum.weights[2], 1.)};
    }
    return out;
}
} // namespace battery
