#include "battery_native/fields/gas_display.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool ok, const char* why) {
    if (!ok)
        throw std::runtime_error(why);
}
double distance(battery::Vec3 a, battery::Vec3 b) {
    return std::hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}
} // namespace
int main() {
    try {
        using namespace battery;
        Scenario s;
        s.gas_compartments.resize(2);
        GasChannel channel;
        channel.from = 0;
        channel.to = 1;
        channel.area_m2 = .0001;
        s.gas_channels.push_back(channel);
        ThermalMesh mesh;
        VisualGeometry geometry;
        GasVisualRegion a;
        a.compartment = 0;
        a.center = {-.01, 0, 0};
        a.size = {.02, .02, .02};
        GasVisualRegion b = a;
        b.compartment = 1;
        b.center = {.01, 0, 0};
        geometry.gas = {a, b};
        VisualFrame frame;
        frame.process.gas.resize(2);
        frame.process.channels.resize(1);
        frame.process.gas[0].temperature_k = 800;
        frame.process.gas[1].temperature_k = 400;
        auto& flow = frame.process.channels[0];
        flow.effective_area_m2 = .0001;
        flow.mass_flow_kg_s = .0001;
        flow.transferred_kg = .0000125;
        GasDisplayModel model(s, mesh, geometry);
        const auto base = model.sample(frame), paused = model.sample(frame);
        const auto volumes_only = model.sample(frame, false);
        require(volumes_only.puffs.empty(), "Volume-only rendering still generates flow markers");
        for (std::size_t i = 0; i < base.regions.size(); ++i)
            require(distance(base.regions[i].displacement, volumes_only.regions[i].displacement) == 0 &&
                        distance(base.regions[i].temperature_gradient, volumes_only.regions[i].temperature_gradient) == 0,
                    "Removing markers changed volume animation or temperature gradient");
        require(!base.puffs.empty(), "Accepted hot flow must be visible");
        require(base.puffs.size() == paused.puffs.size(), "Pause changes particle count");
        for (std::size_t i = 0; i < base.puffs.size(); ++i) {
            require(distance(base.puffs[i].position, paused.puffs[i].position) == 0,
                    "Paused particles moved");
            require(base.puffs[i].temperature_k >= 400 && base.puffs[i].temperature_k <= 800,
                    "Flow temperature overshoot");
            require(in_gas_region(a, base.puffs[i].position) || in_gas_region(b, base.puffs[i].position),
                    "Particle outside gas support");
        }
        for (const auto& r : base.regions) {
            require(r.minimum_k == 400 && r.maximum_k == 800, "Connected gradient bounds lost");
            require(r.temperature_gradient.x < 0 && r.temperature_gradient.y == 0,
                    "Temperature gradient has wrong direction");
        }
        flow.transferred_kg += .000005;
        const auto forward = model.sample(frame);
        require(forward.puffs.front().position.x > base.puffs.front().position.x,
                "Accepted transfer did not advance flow");
        require(forward.regions[0].displacement.x > base.regions[0].displacement.x, "Cloud did not advect");
        flow.transferred_kg -= .00001;
        flow.mass_flow_kg_s = -.0001;
        const auto reverse = model.sample(frame);
        require(reverse.puffs.front().position.x < base.puffs.front().position.x,
                "Reversed transfer did not reverse motion");
        frame.known_channels = {false};
        require(model.sample(frame).puffs.empty(), "Unrecorded channels invented gas flow");
        frame.known_channels = {true};
        flow.mass_flow_kg_s = 0;
        require(model.sample(frame).puffs.empty(), "Zero accepted flow animated particles");
        require(distance(model.sample(frame).regions[0].displacement, reverse.regions[0].displacement) == 0,
                "Stopped cloud jumped backwards");
        frame.time_s = 1.;
        flow.mixed_mass_kg = .00001;
        frame.process.gas[0].temperature_k = frame.process.gas[1].temperature_k = 298.15;
        const auto mixing = model.sample(frame);
        require(!mixing.puffs.empty(), "Equal-pressure cold counterexchange is invisible");
        bool forward_heading = false, backward_heading = false;
        for (const auto& p : mixing.puffs) {
            forward_heading |= p.direction.x > 0;
            backward_heading |= p.direction.x < 0;
            require(p.temperature_k == 298.15f, "Cold flow invented heating");
        }
        require(forward_heading && backward_heading, "Counterexchange lacks opposite directions");
        const auto mixing_pause = model.sample(frame);
        require(mixing_pause.puffs.size() == mixing.puffs.size(), "Paused mixing disappeared");
        for (std::size_t i = 0; i < mixing.puffs.size(); ++i)
            require(distance(mixing_pause.puffs[i].position, mixing.puffs[i].position) == 0,
                    "Pause moved mixing");
        frame.time_s = 2.;
        require(model.sample(frame).puffs.empty(),
                "Old cumulative mixing keeps animating after exchange stops");
        frame.time_s = .5;
        flow.mixed_mass_kg = .000005;
        require(model.sample(frame).puffs.empty(), "Backward seek invented a mixing rate");
        frame.time_s = .6;
        flow.mixed_mass_kg = .000006;
        require(!model.sample(frame).puffs.empty(), "Mixing not restored after seek");
        frame.time_s = .7;
        frame.known_channels = {false};
        flow.mixed_mass_kg = .000007;
        require(model.sample(frame).puffs.empty(), "Unknown mixing is shown");
        frame.time_s = .8;
        frame.known_channels = {true};
        flow.mixed_mass_kg = .000008;
        require(model.sample(frame).puffs.empty(), "Missing mixing interval was fabricated");
        flow.mass_flow_kg_s = .0001;
        require(!model.sample(frame).puffs.empty(), "Cold advected gas is invisible");
        frame.process.gas[0].temperature_k = 800;
        frame.process.gas[1].temperature_k = 400;
        s.gas_channels[0].rupture = true;
        flow.effective_area_m2 = 0;
        GasDisplayModel barrier(s, mesh, geometry);
        const auto closed = barrier.sample(frame);
        require(closed.puffs.empty() && closed.regions[0].minimum_k == 800 &&
                    closed.regions[1].maximum_k == 400,
                "Closed barrier leaks interpolation/flow");
        a.excluded_center = a.center;
        a.excluded_size = {.01, .01, .01};
        require(!in_gas_region(a, a.center), "Solid occupied region treated as gas");
        require(!in_gas_region(a, {1, 1, 1}), "Outside volume treated as gas");
        a.excluded_size = {};
        a.polygon = {{{-.02f, -.01f}}, {{0, -.01f}}, {{-.02f, .01f}}};
        require(!in_gas_region(a, {-.001, .009, 0}), "Outside polygon treated as gas");
        // Hot transported gas is not burning gas. Only the local reaction
        // ledger delta can activate the separate flame layer.
        s.gas_compartments[0].volume_m3=s.gas_compartments[1].volume_m3=1e-5;
        GasDisplayModel burning(s,mesh,geometry);
        frame.time_s=0;frame.process.gas[0].released_energy_j=100;
        frame.process.gas[1].released_energy_j=200;
        require(burning.sample(frame,false).regions[0].combustion_power_w==0,
                "First sample invents a flame from historical release");
        frame.time_s=2;frame.process.gas[0].released_energy_j=104;
        frame.process.gas[0].free_volume_m3=2e-5;
        auto flame=burning.sample(frame,false);
        require(std::abs(flame.regions[0].combustion_power_w-2)<1e-12 &&
                std::abs(flame.regions[0].combustion_power_density_w_m3-1e5)<1e-8,
                "R3 optical rate ignores physical time or free gas volume");
        require(flame.regions[1].combustion_power_w==0,"R3 glow leaks to a hot neighbour");
        require(burning.sample(frame,false).regions[0].combustion_power_w==2,"Pause loses flame rate");
        frame.time_s=3;
        require(burning.sample(frame,false).regions[0].combustion_power_w==0,"Spent gas keeps burning visually");
        frame.time_s=1;
        require(burning.sample(frame,false).regions[0].combustion_power_w==0,"Backward seek reuses future flame");
        std::cout << "Gas display: pause, signed accepted motion, bounded gradients, unknown flows and "
                     "solid/barrier masking passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
