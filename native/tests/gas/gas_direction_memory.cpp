#include "battery_native/gas/transport/gas_direction_memory.hpp"
#include "battery_native/gas/transport/cuda_gas_transport.hpp"
#include "battery_native/gas/transport/kawasaki_gas.hpp"
#include "battery_native/scenario_io.hpp"
#include "battery_native/simulation/simulation.hpp"
#include "battery_native/io/checkpoint_io.hpp"
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace battery;
namespace gd = battery::gas_direction;
void require(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}
void checked(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}
template <class T> struct Buffer {
    T* p = nullptr;
    std::size_t n;
    explicit Buffer(std::size_t count) : n(count) {
        if (n)
            checked(cudaMalloc(&p, n * sizeof(T)));
    }
    ~Buffer() {
        if (p)
            cudaFree(p);
    }
    void put(const std::vector<T>& v) {
        require(v.size() == n, "buffer size");
        if (n)
            checked(cudaMemcpy(p, v.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> get() const {
        std::vector<T> v(n);
        if (n)
            checked(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
        return v;
    }
};
void analytic_checks() {
    const gd::Arrival incoming{1, 1, 0, 0};
    require(gd::turn_loss(incoming, {1, 0, 0}, true) == 0., "straight momentum");
    require(gd::turn_loss(incoming, {0, 1, 0}, true) == 2., "side redirection");
    require(gd::turn_loss(incoming, {1, 0, 0}, false) == 4., "reverse redirection");
    require(gd::turn_loss({2, 0, 0, 0}, {1, 0, 0}, true) == 2., "opposing jets dissipate");
    require(gd::turn_loss({}, {1, 0, 0}, true) == 0., "rest invents anisotropy");
    for (double dt : {.00001, .01, 1., 100.}) {
        const double q0 = .7, dp = -.1, inertia = 2., resistance = 3., quadratic = 4.;
        const auto op = gd::implicit_flow(q0, dp, dt, inertia, resistance, quadratic);
        const double q = op.mobility * dp + op.bias;
        require(std::abs(inertia * (q - q0) / dt + resistance * q + quadratic * q * std::abs(q) - dp) < 1e-10,
                "momentum balance");
        const double deltaK = .5 * inertia * (q * q - q0 * q0);
        const double drag = dt * (resistance * q * q + quadratic * std::abs(q * q * q));
        const double numerical = .5 * inertia * (q - q0) * (q - q0);
        require(std::abs(dp * q * dt - deltaK - drag - numerical) < 1e-11, "discrete kinetic work balance");
    }
    const auto coast = gd::implicit_flow(1., 0., .1, 2., 4., 0.);
    require(std::abs(coast.bias - 1. / 1.2) < 1e-15, "inertia vanishes at zero pressure gap");
    require(gd::relaxation_time(2., 4., 0., 1.) == .5, "derived lifetime");
    double previous_error = 1.;
    for (int n : {10, 20, 40}) {
        double q = 1.;
        for (int i = 0; i < n; ++i)
            q = gd::implicit_flow(q, 0., 1. / n, 2., 4., 0.).bias;
        const double error = std::abs(q - std::exp(-2.));
        require(error < previous_error, "physical-time refinement");
        previous_error = error;
    }
    require(gd::laminar_resistance(2e-5, .01, 1e-4, 1.) > 0., "viscous drag");
}
Scenario branch(bool inertia) {
    Scenario s;
    s.gas_model.cuda_network_transport = true;
    s.gas_model.averaged_network_transport = true;
    s.gas_model.directional_inertia = inertia;
    s.gas_model.dynamic_viscosity_pa_s = 2e-5;
    s.gas_model.coupled_transport_interval_s = .001;
    s.gas_model.averaged_payload_iterations = 64;
    s.settings.minimum_dt_s = 0.;
    s.gas_compartments.resize(4);
    // Source -> junction -> straight / side, identical hydraulic exits.
    for (int i = 0; i < 4; ++i) {
        auto& g = s.gas_compartments[i];
        g.id = "gas" + std::to_string(i);
        g.volume_m3 = 1e-6;
        g.initial_temperature_k = 400;
        g.initial_mass_kg = (i == 0 ? 3e5 : 1e5) * g.volume_m3 / (287 * 400);
    }
    for (int i = 0; i < 3; ++i) {
        GasChannel e;
        e.id = "edge" + std::to_string(i);
        e.from = i == 0 ? 0 : 1;
        e.to = i == 0 ? 1 : i + 1;
        e.flow_direction = i == 2 ? Vec3{0, 1, 0} : Vec3{1, 0, 0};
        e.discharge_coefficient = .7;
        e.inertial_length_m = .01;
        s.gas_channels.push_back(e);
    }
    return s;
}
struct Fixture {
    Scenario scene;
    ThermalMesh thermal;
    MassMesh mass;
    std::unique_ptr<CudaGasTransport> solver;
    Buffer<GasState> input, output;
    Buffer<GasChannelState> channels;
    std::vector<GasState> initial;
    std::vector<GasChannelState> initial_channels;
    explicit Fixture(bool inertia) : Fixture(branch(inertia)) {}
    explicit Fixture(Scenario config)
        : scene(std::move(config)), input(scene.gas_compartments.size()),
          output(scene.gas_compartments.size()), channels(scene.gas_channels.size()) {
        mass.bindings.resize(scene.gas_compartments.size());
        solver = std::make_unique<CudaGasTransport>(scene, thermal, mass);
        for (const auto& g : scene.gas_compartments) {
            GasState v;
            v.mass_kg = g.initial_mass_kg;
            v.temperature_k = g.initial_temperature_k;
            v.free_volume_m3 = g.volume_m3;
            v.pressure_pa = v.mass_kg * 287 * v.temperature_k / v.free_volume_m3;
            v.energy_j = v.mass_kg * 1000 * (v.temperature_k - 298.15);
            v.chemical_j = v.mass_kg * 1e6;
            v.oxidizer_kg = v.mass_kg * .2;
            v.vapor_kg = v.mass_kg * .1;
            v.latent_j = v.vapor_kg * 2e5;
            initial.push_back(v);
        }
        initial_channels.resize(scene.gas_channels.size());
        for (auto& c : initial_channels) {
            c.open = true;
            c.effective_area_m2 = 2e-8;
        }
        reset_inputs();
    }
    void reset_inputs() {
        input.put(initial);
        channels.put(initial_channels);
    }
    void prepare(double dt = .001) {
        solver->prepare(input.p, output.p, nullptr, channels.p, dt);
    }
    void step(double dt = .001) {
        prepare(dt);
        solver->commit();
        std::swap(input.p, output.p);
    }
};
void balances(const Fixture& f) {
    const auto end = f.input.get();
    const auto edges = f.channels.get();
    const auto momentum = f.solver->directional_memory();
    double total[6]{};
    for (std::size_t i = 0; i < end.size(); ++i) {
        const auto& a = f.initial[i];
        const auto& b = end[i];
        require(b.mass_kg > 0 && b.energy_j > 0 && b.chemical_j >= 0 && b.oxidizer_kg >= 0,
                "negative inventory");
        const double delta[]{b.mass_kg - a.mass_kg,       b.energy_j - a.energy_j,
                             b.chemical_j - a.chemical_j, b.oxidizer_kg - a.oxidizer_kg,
                             b.vapor_kg - a.vapor_kg,     b.latent_j - a.latent_j};
        double ledger[6]{};
        for (std::size_t e = 0; e < edges.size(); ++e) {
            const auto& c = edges[e];
            const auto& topology = f.scene.gas_channels[e];
            const double sign = topology.from == int(i) ? -1 : topology.to == int(i) ? 1 : 0;
            const double x[]{c.transferred_kg,          c.transferred_energy_j, c.transferred_chemical_j,
                             c.transferred_oxidizer_kg, c.transferred_vapor_kg, c.transferred_latent_j};
            for (int k = 0; k < 6; ++k)
                ledger[k] += sign * x[k];
            if (sign != 0. && !momentum.empty())
                ledger[1] -= .5 * momentum[e].kinetic_j;
        }
        const double scale[]{a.mass_kg, a.energy_j, a.chemical_j, a.oxidizer_kg, a.vapor_kg, a.latent_j};
        for (int k = 0; k < 6; ++k) {
            require(std::abs(delta[k] - ledger[k]) < 1e-9 * scale[k], "directional channel ledger");
            total[k] += delta[k];
        }
    }
    for (const auto& m : momentum)
        total[1] += m.kinetic_j;
    for (double value : total)
        require(std::abs(value) < 1e-10, "global inventory balance");
}
void transport_checks() {
    Fixture off(false), on(true);
    for (int i = 0; i < 40; ++i) {
        off.step();
        on.step();
    }
    balances(off);
    balances(on);
    const auto a = off.input.get(), b = on.input.get();
    const auto m = on.solver->directional_memory();
    require(std::abs(a[2].mass_kg - a[3].mass_kg) < 1e-18, "symmetric baseline");
    require(b[2].mass_kg > b[3].mass_kg * 1.001, "incoming stream does not favor straight exit");
    require(m[1].flow_kg_s > m[2].flow_kg_s && m[1].kinetic_j > 0., "momentum or side loss missing");
    std::cout << "branch added mass, straight/side: off="
              << (a[2].mass_kg - off.initial[2].mass_kg) / (a[3].mass_kg - off.initial[3].mass_kg)
              << ", on=" << (b[2].mass_kg - on.initial[2].mass_kg) / (b[3].mass_kg - on.initial[3].mass_kg)
              << '\n';

    const auto checkpoint = on.solver->save_state();
    const auto before = on.input.get();
    const auto before_channels = on.channels.get();
    on.prepare();
    require(on.solver->save_state() == checkpoint, "uncommitted prepare changed memory");
    const auto first = on.output.get();
    on.channels.put(before_channels);
    on.prepare();
    const auto second = on.output.get();
    for (std::size_t i = 0; i < first.size(); ++i)
        require(first[i].mass_kg == second[i].mass_kg && first[i].energy_j == second[i].energy_j,
                "retry accumulated memory");
    on.solver->commit();
    const auto committed = on.solver->save_state();
    Fixture restored(true);
    restored.solver->restore_state(checkpoint);
    restored.input.put(before);
    restored.channels.put(before_channels);
    restored.step();
    require(restored.solver->save_state() == committed, "checkpoint continuation lost memory");
    bool rejected = false;
    try {
        restored.solver->restore_state(off.solver->save_state());
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "historyless checkpoint silently accepted");

    // Close all channels: no transfer; kinetic energy returns to thermal energy.
    auto closed = restored.channels.get();
    for (auto& c : closed) {
        c.open = false;
        c.effective_area_m2 = 0;
    }
    restored.channels.put(closed);
    const auto start = restored.input.get();
    const auto memory = restored.solver->directional_memory();
    restored.step(.005);
    const auto finish = restored.input.get();
    const auto faded = restored.solver->directional_memory();
    for (std::size_t i = 0; i < start.size(); ++i)
        require(start[i].mass_kg == finish[i].mass_kg, "closed edge leaks");
    for (std::size_t i = 0; i < memory.size(); ++i)
        require(faded[i].flow_kg_s == 0. && faded[i].kinetic_j == 0., "closed graph retains momentum");
    double released = 0., heating = 0.;
    for (auto m : memory)
        released += m.kinetic_j;
    for (std::size_t i = 0; i < start.size(); ++i)
        heating += finish[i].energy_j - start[i].energy_j;
    require(std::abs(released - heating) < 1e-12, "closure loses kinetic energy");
    restored.solver->reset();
    for (auto value : restored.solver->directional_memory())
        require(value.flow_kg_s == 0 && value.kinetic_j == 0, "reset momentum");
}
void convergence_checks() {
    std::vector<std::vector<GasState>> end;
    for (int divisions : {1, 2, 4}) {
        Fixture f(true);
        for (int i = 0; i < 40 * divisions; ++i)
            f.step(.001 / divisions);
        balances(f);
        end.push_back(f.input.get());
    }
    double error[2]{};
    for (int j = 0; j < 2; ++j)
        for (std::size_t i = 0; i < end[j].size(); ++i)
            error[j] =
                std::max(error[j], std::abs(end[j][i].mass_kg - end[2][i].mass_kg) / end[2][i].mass_kg);
    require(error[1] < error[0] && error[0] < .01, "direction memory time convergence");
    std::cout << "time refinement, max relative mass error: " << error[0] << " -> " << error[1] << '\n';
}
void orientation_checks() {
    auto reversed = branch(true);
    for (auto& e : reversed.gas_channels) {
        std::swap(e.from, e.to);
        e.flow_direction.x = -e.flow_direction.x;
        e.flow_direction.y = -e.flow_direction.y;
        e.flow_direction.z = -e.flow_direction.z;
    }
    Fixture original(true), flipped(reversed);
    for (int i = 0; i < 40; ++i) {
        original.step();
        flipped.step();
    }
    const auto a = original.input.get(), b = flipped.input.get();
    const auto ma = original.solver->directional_memory(), mb = flipped.solver->directional_memory();
    for (std::size_t i = 0; i < a.size(); ++i)
        require(std::abs(a[i].mass_kg - b[i].mass_kg) < 1e-13 * a[i].mass_kg,
                "edge labeling changes transport");
    for (std::size_t i = 0; i < ma.size(); ++i)
        require(std::abs(ma[i].flow_kg_s + mb[i].flow_kg_s) < 1e-13 &&
                    std::abs(ma[i].kinetic_j - mb[i].kinetic_j) < 1e-13,
                "edge labeling loses direction memory");
    balances(flipped);
}
void coast_and_reaction_checks() {
    auto config = branch(true);
    config.gas_channels.resize(1);
    config.gas_compartments.resize(2);
    Fixture coast(config);
    coast.step(.0001);
    const double first_flow = coast.solver->directional_memory()[0].flow_kg_s;
    require(first_flow > 0., "pressure does not accelerate gas");
    auto uniform = coast.input.get();
    // Controlled isolated initial condition: same pressure, still moving gas.
    for (auto& g : uniform) {
        g.temperature_k = 400.;
        g.pressure_pa = 1e5;
        g.mass_kg = 1e5 * g.free_volume_m3 / (287. * 400.);
        g.energy_j = g.mass_kg * 1000. * (400. - 298.15);
        g.chemical_j = g.mass_kg * 1e6;
        g.oxidizer_kg = g.mass_kg * .2;
        g.vapor_kg = g.mass_kg * .1;
        g.latent_j = g.vapor_kg * 2e5;
    }
    coast.input.put(uniform);
    coast.step(.000001);
    const double continued = coast.solver->directional_memory()[0].flow_kg_s;
    require(continued > 0. && continued < first_flow, "zero-dp gas does not coast and slow down");
    require(coast.input.get()[1].mass_kg > uniform[1].mass_kg, "inertial transport absent");

    for (bool colored : {false, true})
        for (bool reactive_transit : {false, true}) {
            auto s = branch(true);
            s.gas_model.reactive_transit = reactive_transit;
            s.gas_model.max_temperature_step_k = 100.;
            for (auto& g : s.gas_compartments)
                g.reaction_rate_per_s = 2.;
            Fixture f(s);
            f.solver->set_colored_kawasaki(colored);
            for (int i = 0; i < 20; ++i)
                f.step();
            double balance = 0., oxygen = 0., heat = 0.;
            const auto finish = f.input.get();
            for (std::size_t i = 0; i < finish.size(); ++i) {
                const auto& g = finish[i];
                const auto& start = f.initial[i];
                balance += g.energy_j + g.chemical_j - start.energy_j - start.chemical_j;
                oxygen += g.oxidizer_kg + g.consumed_oxidizer_kg - start.oxidizer_kg;
                heat += g.released_energy_j;
            }
            for (auto m : f.solver->directional_memory())
                balance += m.kinetic_j;
            require(heat > 0. && std::abs(balance) < 1e-10 && std::abs(oxygen) < 1e-14,
                    "reactive transport lost kinetic/chemical/oxygen balance");
        }

    Fixture original(true);
    original.step();
    auto changed = branch(true);
    changed.gas_channels[0].inertial_length_m *= 2.;
    Fixture incompatible(changed);
    const auto saved = incompatible.solver->save_state();
    bool rejected = false;
    try {
        incompatible.solver->restore_state(original.solver->save_state());
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected && incompatible.solver->save_state() == saved, "restart silently changed inertance");
    Fixture intact(true);
    const auto intact_state = intact.solver->save_state();
    // Reader owns a view; retain its backing bytes for the whole read.
    const auto valid_bytes = original.solver->save_state();
    checkpoint::Reader source(valid_bytes);
    checkpoint::Writer malformed;
    malformed.text(source.text());
    malformed.pod(source.pod<bool>());
    malformed.pod(source.pod<bool>());
    source.vector<kawasaki::Clock>();
    malformed.vector(std::vector<kawasaki::Clock>{});
    malformed.pod(source.pod<double>());
    malformed.text(source.text());
    malformed.vector(source.vector<gd::Memory>());
    rejected = false;
    try {
        intact.solver->restore_state(malformed.finish());
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected && intact.solver->save_state() == intact_state,
            "malformed restart modified committed momentum");
    changed = branch(true);
    changed.gas_compartments[0].ambient.enabled = true;
    rejected = false;
    try {
        Fixture invalid(changed);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "scalar ambient silently discards momentum");
}
Scenario cube(bool inertia) {
    auto s = branch(inertia);
    const auto node = s.gas_compartments[1];
    s.gas_compartments.clear();
    s.gas_channels.clear();
    constexpr int n = 8;
    for (int z = 0; z < n; ++z)
        for (int y = 0; y < n; ++y)
            for (int x = 0; x < n; ++x) {
                auto g = node;
                g.id = "gas" + std::to_string(s.gas_compartments.size());
                if (x == 0)
                    g.initial_mass_kg *= 3;
                s.gas_compartments.push_back(g);
                const int index = x + n * (y + n * z);
                for (int axis = 0; axis < 3; ++axis) {
                    if ((axis == 0 ? x : axis == 1 ? y : z) == 0)
                        continue;
                    GasChannel e;
                    e.id = "edge" + std::to_string(s.gas_channels.size());
                    e.from = index - (axis == 0 ? 1 : axis == 1 ? n : n * n);
                    e.to = index;
                    e.flow_direction = {axis == 0 ? 1. : 0., axis == 1 ? 1. : 0., axis == 2 ? 1. : 0.};
                    e.inertial_length_m = .01;
                    s.gas_channels.push_back(e);
                }
            }
    return s;
}
void benchmark() {
    // Synthetic 512-node/1344-edge graph, not the battery calibration runtime.
    // Warm up both modes; alternate their order to reduce GPU clock bias.
    Fixture off(cube(false)), on(cube(true));
    for (int i = 0; i < 5; ++i) {
        off.step();
        on.step();
    }
    std::vector<double> timings[2];
    for (int repeat = 0; repeat < 6; ++repeat)
        for (int order = 0; order < 2; ++order) {
            const int mode = (repeat + order) % 2;
            auto& f = mode ? on : off;
            f.reset_inputs();
            f.solver->reset();
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < 50; ++i)
                f.step();
            const double ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() /
                50;
            timings[mode].push_back(ms);
            balances(f);
        }
    for (auto& t : timings)
        std::sort(t.begin(), t.end());
    const double off_ms = (timings[0][2] + timings[0][3]) / 2, on_ms = (timings[1][2] + timings[1][3]) / 2;
    std::cout << "512 nodes / 1344 edges, median adapter ms/step: off=" << off_ms << ", on=" << on_ms
              << ", ratio=" << on_ms / off_ms << '\n';
}
void parser_checks(const char* source) {
    namespace fs = std::filesystem;
    const auto path = fs::temp_directory_path() /
                      ("gas-memory-" +
                       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".json");
    struct Cleanup {
        fs::path p;
        ~Cleanup() {
            std::error_code ec;
            fs::remove(p, ec);
        }
    } cleanup{path};
    nlohmann::json original;
    std::ifstream(source) >> original;
    auto root = original;
    root["gas_network"]["mass_backend"] = "cuda_network";
    root["gas_network"]["network_integrator"] = "averaged";
    root["gas_network"]["directional_memory"] = {{"enabled", true}};
    root["gas_network"]["dynamic_viscosity_pa_s"] = 2e-5;
    for (auto& c : root["gas_network"]["channels"]) {
        c["flow_direction"] = {2, 0, 0};
        c["inertial_length_m"] = .01;
    }
    const auto load = [&](const nlohmann::json& j) {
        {
            std::ofstream out(path);
            out << j;
        }
        return load_scenario(path.string());
    };
    const auto s = load(root);
    require(s.gas_model.directional_inertia, "momentum parser");
    require(!s.gas_channels.empty() && s.gas_channels[0].flow_direction.x == 1, "signed geometry parser");
    for (int kind = 0; kind < 8; ++kind) {
        auto bad = root;
        if (kind == 0)
            bad["gas_network"]["directional_memory"]["strength"] = 1.;
        if (kind == 1)
            bad["gas_network"]["directional_memory"]["relaxation_time_s"] = 0.;
        if (kind == 2)
            bad["gas_network"]["channels"][0].erase("flow_direction");
        if (kind == 3)
            bad["gas_network"]["directional_memory"]["typo"] = 1.;
        if (kind == 4)
            bad["gas_network"]["network_integrator"] = "pairwise";
        if (kind == 5)
            bad["gas_network"]["transport"] = "legacy_coupled";
        if (kind == 6)
            bad["gas_network"]["dynamic_viscosity_pa_s"] = 0.;
        if (kind == 7)
            bad["gas_network"]["channels"][0].erase("inertial_length_m");
        bool rejected = false;
        try {
            load(bad);
        } catch (const std::exception&) {
            rejected = true;
        }
        require(rejected, "invalid memory config accepted");
    }
    require(!load(original).gas_model.directional_inertia, "legacy scenario changed");
    // Exercise the complete thermal/fracture/transport driver, not just the adapter.
    auto smoke = s;
    smoke.settings.duration_s = .05;
    Simulation sim(smoke);
    sim.step();
    const auto restart = sim.save_checkpoint("direction-memory-smoke");
    Simulation resumed(smoke);
    resumed.restore_checkpoint(restart, "direction-memory-smoke");
    sim.run_to_end();
    resumed.run_to_end();
    require(std::abs(sim.energy_balance_residual_j()) < 1e-6, "full solver kinetic ledger");
    for (std::size_t i = 0; i < sim.cells().size(); ++i)
        require(sim.cells()[i].temperature_k == resumed.cells()[i].temperature_k,
                "full solver checkpoint continuation");
}
void colored_checks() {
    for (bool inertia : {false, true}) {
        Fixture f(inertia);
        for (int i = 0; i < 10; ++i)
            f.step();
        const auto original = f.solver->directional_memory();
        f.solver->set_colored_kawasaki(true);
        const auto switched = f.solver->directional_memory();
        require(original.size() == switched.size(), "Switch loses directional history");
        for (std::size_t i = 0; i < original.size(); ++i)
            require(original[i].flow_kg_s == switched[i].flow_kg_s &&
                        original[i].kinetic_j == switched[i].kinetic_j,
                    "Switch resets directional history");
        for (int i = 0; i < 40; ++i)
            f.step();
        balances(f);
        const auto checkpoint = f.solver->save_state();
        const auto before = f.input.get();
        const auto channels = f.channels.get();
        f.prepare();
        const auto first = f.output.get();
        require(f.solver->save_state() == checkpoint, "Kawasaki trial commits RNG or momentum");
        f.channels.put(channels);
        f.prepare();
        const auto retry = f.output.get();
        for (std::size_t i = 0; i < first.size(); ++i)
            require(first[i].mass_kg == retry[i].mass_kg && first[i].energy_j == retry[i].energy_j,
                    "Kawasaki retry is not reproducible");
        Fixture restored(inertia);
        restored.solver->restore_state(checkpoint);
        require(restored.solver->colored_kawasaki(), "Checkpoint lost numerical method");
        restored.input.put(before);
        restored.channels.put(channels);
        restored.step();
        const auto after = restored.input.get();
        for (std::size_t i = 0; i < first.size(); ++i)
            require(first[i].mass_kg == after[i].mass_kg && first[i].energy_j == after[i].energy_j,
                    "Kawasaki restart is not reproducible");
        f.channels.put(channels); // discard the adapter trial's caller-owned channel output
        f.solver->set_colored_kawasaki(false);
        for (int i = 0; i < 10; ++i)
            f.step();
        balances(f);
        auto closed = restored.channels.get();
        for (auto& c : closed) {
            c.open = false;
            c.effective_area_m2 = 0.;
        }
        restored.channels.put(closed);
        restored.step();
        const auto stopped = restored.input.get();
        for (std::size_t i = 0; i < after.size(); ++i)
            require(after[i].mass_kg == stopped[i].mass_kg, "Kawasaki crosses closed barrier");
        for (const auto& m : restored.solver->directional_memory())
            require(m.kinetic_j == 0. && m.flow_kg_s == 0., "Kawasaki closure retains momentum");
    }
    std::cout << "Kawasaki: live switching, six inventory ledgers, barriers, retry and checkpoint passed\n";
}
void colored_time_checks() {
    auto config = branch(false);
    config.gas_compartments.resize(2);
    config.gas_channels.resize(1);
    Fixture reference(config);
    for (int i = 0; i < 1000; ++i)
        reference.step(.00002);
    const double target = reference.input.get()[1].mass_kg;
    const double transferred = target - reference.initial[1].mass_kg;
    for (int steps : {20, 40, 80}) {
        double sum = 0., sumsq = 0.;
        constexpr int samples = 32;
        for (int seed = 0; seed < samples; ++seed) {
            config.gas_model.kawasaki_seed = 937 + seed;
            Fixture f(config);
            f.solver->set_colored_kawasaki(true);
            for (int i = 0; i < steps; ++i)
                f.step(.02 / steps);
            const double delta = f.input.get()[1].mass_kg - target;
            sum += delta;
            sumsq += delta * delta;
            balances(f);
        }
        const double bias = sum / samples;
        const double sem = std::sqrt(std::max(0., sumsq / samples - bias * bias) / (samples - 1));
        std::cout << "Kawasaki physical time dt=" << .02 / steps
                  << ": transfer bias=" << 100 * bias / transferred
                  << "%; standard error=" << 100 * sem / transferred << "%\n";
        require(std::abs(bias) < .05 * transferred + 4 * sem, "Kawasaki physical-time transfer mismatch");
    }
}
void colored_benchmark() {
    Fixture deterministic(cube(true)), mc(cube(true));
    mc.solver->set_colored_kawasaki(true);
    for (int i = 0; i < 5; ++i) {
        deterministic.step();
        mc.step();
    }
    for (auto* f : {&deterministic, &mc}) {
        std::vector<double> times;
        for (int repeat = 0; repeat < 3; ++repeat) {
            f->reset_inputs();
            f->solver->reset();
            for (int i = 0; i < 50; ++i) {
                const auto start = std::chrono::steady_clock::now();
                f->step();
                times.push_back(
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                        .count());
            }
            balances(*f);
        }
        std::sort(times.begin(), times.end());
        std::cout << "512-node identical initial state, " << (f == &mc ? "Kawasaki" : "network")
                  << ": median=" << times[times.size() / 2] << " ms; p95=" << times[times.size() * 95 / 100]
                  << " ms, physical dt=1 ms\n";
    }
}
int main(int argc, char** argv) {
    try {
        analytic_checks();
        colored_checks();
        colored_time_checks();
        colored_benchmark();
        transport_checks();
        convergence_checks();
        orientation_checks();
        coast_and_reaction_checks();
        benchmark();
        if (argc > 1)
            parser_checks(argv[1]);
        std::cout
            << "Directional memory: analytical, CUDA conservation, rollback, restart, parser checks passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
