// GPU storage, static topology upload and internal stage declarations. No run-file parsing.
// Private implementation fragment, included only by cuda_solver.cu.
struct CudaSolver::Impl {
    const Scenario& scenario;
    const ThermalMesh& mesh;
    MassMesh mass_mesh;
    ElectricalMesh electrical_mesh;
    SolverProfile profiling;
    std::unique_ptr<LocalGasTransport> cpu_gas;
    std::unique_ptr<CudaGasTransport> cuda_gas;
    std::vector<GasState> host_gas;
    std::vector<ThermalState> host_solid;
    std::vector<GasChannelState> host_channels;
    std::string device;
    Buffer<DeviceVolume> volumes;
    Buffer<DeviceLink> links;
    std::unique_ptr<Buffer<ConnectivityLaw>> laws;
    Buffer<Material> materials;
    std::vector<Material> reference_materials;
    std::unique_ptr<Buffer<CurrentPoint>> heater_voltage_points;
    std::unique_ptr<Buffer<CurrentPoint>> current_points, voltage_points;
    Buffer<ThermalState> state_a, state_b, local_state;
    Buffer<InterfaceState> edge_state, edge_next, macro_edges;
    std::unique_ptr<Buffer<int>> conversion_ids, legacy_ids;
    std::unique_ptr<Buffer<int>> film_aging_ids;
    bool has_volatile = false;
    std::array<std::unique_ptr<Buffer<int>>, 4> edge_ids;
    Buffer<double> local_heat;
    Buffer<GasTransfer> source_transfer, channel_transfer, gas_reaction;
    Buffer<CellState> macro_cells;
    Buffer<int> offsets, adjacency, object_offsets, interface_offsets, interface_ids;
    Buffer<double> flux, dt, constant_g, cfl, hard_dt;
    Buffer<StepResult> result;
    Buffer<int> electrical_objects, electrical_edge_offsets, electrical_offsets, electrical_adjacency;
    Buffer<ElectricalEdge> electrical_edges;
    Buffer<double> electrical_g, electrical_sigma, electrical_diagonal, electrical_r, electrical_z,
        electrical_p, electrical_ap;
    Buffer<double> electrical_potential_a, electrical_potential_b, electrical_power;
    Buffer<ElectricalState> electrical_a, electrical_b;
    std::unique_ptr<Buffer<int>> circuit_mask;
    std::unique_ptr<Buffer<CircuitBranch>> circuit_branches;
    std::unique_ptr<Buffer<double>> circuit_r, circuit_matrix, circuit_rhs, circuit_i, circuit_v;
    std::unique_ptr<Buffer<CurrentPoint>> circuit_voltage;
    bool has_compliant_drive = false;
    Buffer<int> profile_offsets;
    std::unique_ptr<Buffer<ProfilePoint>> profile_points;
    Buffer<GasState> gas_a, gas_b;
    std::unique_ptr<Buffer<GasState>> gas_stage;
    std::unique_ptr<Buffer<MassBinding>> mass_bindings;
    std::unique_ptr<Buffer<double>> mass_dt;
    Buffer<DeviceGas> gas_config;
    Buffer<DeviceChannel> channels;
    Buffer<GasChannelState> channel_state, channel_next;
    Buffer<GasThermalPatch> gas_patches;
    Buffer<double> gas_flux, gas_conductance, channel_mass_cap;
    std::unique_ptr<Buffer<int>> aperture_patch_ids;
    std::unique_ptr<Incidence> aperture_channel_rows;
    std::unique_ptr<Buffer<DeviceShellPatch>> shell_patches;
    std::unique_ptr<Buffer<ShellDamageState>> shell_damage, shell_next, shell_rates;
    std::unique_ptr<Buffer<double>> shell_work_power;
    // Transient connected-front proposals; never part of committed/checkpoint state.
    std::unique_ptr<Buffer<double>> plate_delta, fracture_budget_scale;
    std::unique_ptr<Buffer<int>> plate_order;
    std::unique_ptr<Buffer<GasState>> shell_event_pressure_snapshot;
    // Scratch inputs for the leading aperture operator. Committed buffers are
    // never modified by prepare/retry; pointer swaps are restored before commit.
    std::unique_ptr<Buffer<ThermalState>> prefix_solid;
    std::unique_ptr<Buffer<GasState>> prefix_gas;
    std::unique_ptr<Buffer<GasChannelState>> prefix_channels;
    std::unique_ptr<Buffer<ShellDamageState>> prefix_shell;
    std::vector<DeviceShellPatch> shell_geometry;
    bool prefix_inputs_active = false, prefix_applied = false;
    std::unique_ptr<Incidence> volume_patches, gas_patches_rows, gas_channels_rows, gas_sources_rows,
        shell_channels_rows;
    double cached_interface_scale = -1, cached_ambient_scale = -1, cached_safety = -1;
    bool has_isothermal = false, has_spatial = false;
    bool prepared = false;
    int work_threads;
    int work_blocks(std::size_t n) const {
        return static_cast<int>((n + work_threads - 1) / work_threads);
    }
    Impl(const Scenario& s, const ThermalMesh& m, int tile)
        : scenario(s), mesh(m), mass_mesh(build_mass_mesh(s, m)),
          electrical_mesh(build_electrical_mesh(s, m)), volumes(m.volumes.size()), links(m.links.size()),
          materials(s.cells.size()), state_a(m.volumes.size()), state_b(m.volumes.size()),
          local_state(m.volumes.size()), edge_state(m.links.size()), edge_next(m.links.size()),
          macro_edges(s.interfaces.size()), local_heat(m.volumes.size()), source_transfer(m.volumes.size()),
          channel_transfer(s.gas_channels.size()), gas_reaction(s.gas_compartments.size()),
          macro_cells(s.cells.size()), offsets(m.adjacency_offsets.size()),
          adjacency(m.adjacency_links.size()), object_offsets(m.object_offsets.size()),
          interface_offsets(m.interface_offsets.size()), interface_ids(m.interface_links.size()),
          flux(m.links.size()), dt(1), constant_g(m.links.size()), cfl(1), hard_dt(1), result(1),
          electrical_objects(electrical_mesh.objects.size()),
          electrical_edge_offsets(electrical_mesh.edge_offsets.size()),
          electrical_offsets(electrical_mesh.offsets.size()),
          electrical_adjacency(electrical_mesh.adjacency.size()),
          electrical_edges(electrical_mesh.edges.size()), electrical_g(electrical_mesh.edges.size()),
          electrical_sigma(electrical_objects.n ? m.volumes.size() : 0),
          electrical_diagonal(electrical_sigma.n), electrical_r(electrical_sigma.n),
          electrical_z(electrical_sigma.n), electrical_p(electrical_sigma.n),
          electrical_ap(electrical_sigma.n), electrical_potential_a(electrical_sigma.n),
          electrical_potential_b(electrical_sigma.n), electrical_power(electrical_sigma.n),
          electrical_a(s.cells.size()), electrical_b(s.cells.size()),
          profile_offsets(s.boundaries.size() + 1), gas_a(s.gas_compartments.size()),
          gas_b(s.gas_compartments.size()), gas_config(s.gas_compartments.size()),
          channels(s.gas_channels.size()), channel_state(s.gas_channels.size()),
          channel_next(s.gas_channels.size()), gas_patches(m.gas_patches.size()),
          gas_flux(m.gas_patches.size()), gas_conductance(m.gas_patches.size()),
          channel_mass_cap(s.gas_channels.size()), work_threads(tile) {
        if (tile != 64 && tile != 128 && tile != 256)
            throw std::invalid_argument("Unsupported CUDA work tile size");
        for (const auto& c : s.cells)
            has_compliant_drive |= c.electrical.enabled && c.electrical.voltage_limited_current;
        const double floor = s.settings.minimum_dt_s;
        if (!std::isfinite(floor) || floor < 0. ||
            (floor > 0. && (floor < .005 || floor > s.settings.max_dt_s)))
            throw std::invalid_argument("Invalid minimum_dt_s: coarse integration requires >=0.005 s");
        if (floor > 0. && !s.gas_compartments.empty() &&
            !(s.gas_model.split_transport && s.gas_model.cuda_network_transport &&
              s.gas_model.coupled_gas_films))
            throw std::invalid_argument("Coarse gas integration requires CUDA network + coupled films");
        electrical_objects.upload(electrical_mesh.objects);
        electrical_edge_offsets.upload(electrical_mesh.edge_offsets);
        electrical_offsets.upload(electrical_mesh.offsets);
        electrical_adjacency.upload(electrical_mesh.adjacency);
        electrical_edges.upload(electrical_mesh.edges);
        if (s.gas_model.directional_inertia &&
            !(s.gas_model.split_transport && s.gas_model.cuda_kawasaki_transport &&
              s.gas_model.cuda_network_transport && s.gas_model.averaged_network_transport))
            throw std::invalid_argument("Directional memory requires the split averaged CUDA adapter");
        if (s.gas_model.cpu_local_transport)
            cpu_gas = std::make_unique<LocalGasTransport>(s, m, mass_mesh);
        if (s.gas_model.cuda_kawasaki_transport)
            cuda_gas = std::make_unique<CudaGasTransport>(scenario, m, mass_mesh);
        cudaDeviceProp info{};
        int id = 0;
        check(cudaGetDevice(&id), "CUDA device");
        check(cudaGetDeviceProperties(&info, id), "CUDA device properties");
        device = info.name;
        std::vector<DeviceVolume> device_volumes;
        for (const auto& v : m.volumes)
            device_volumes.push_back({v.object, v.capacity_j_per_k});
        volumes.upload(device_volumes);
        offsets.upload(m.adjacency_offsets);
        adjacency.upload(m.adjacency_links);
        object_offsets.upload(m.object_offsets);
        interface_offsets.upload(m.interface_offsets);
        interface_ids.upload(m.interface_links);
        std::vector<Material> host;
        std::vector<CurrentPoint> drive, volts;
        for (const auto& c : s.cells)
            for (const auto& q : c.current_profile_a)
                drive.push_back({q[0], q[1]});
        for (const auto& c : s.cells)
            for (const auto& q : c.voltage_profile_v)
                volts.push_back({q[0], q[1]});
        current_points = std::make_unique<Buffer<CurrentPoint>>(drive.size());
        current_points->upload(drive);
        voltage_points = std::make_unique<Buffer<CurrentPoint>>(volts.size());
        voltage_points->upload(volts);
        int current_offset = 0, voltage_offset = 0;
        for (const auto& c : s.cells) {
            host.push_back({c.heat_capacity_j_per_k,
                            c.runaway_energy_j,
                            c.runaway_duration_s,
                            c.damage_onset_k,
                            c.damage_rate_at_onset_per_s,
                            c.damage_temperature_scale_k,
                            1 + c.aging_damage_multiplier * c.aging_fraction,
                            c.isothermal,
                            c.is_battery,
                            c.heat_capacity_law,
                            c.runaway_pulse,
                            boundary_phase(c),
                            c.reaction_kinetics,
                            c.reaction_structure,
                            c.energy_outlet,
                            c.conversion,
                            s.gas_model,
                            c.electrical,
                            {current_points->p ? current_points->p + current_offset : nullptr,
                             static_cast<int>(c.current_profile_a.size())},
                            {voltage_points->p ? voltage_points->p + voltage_offset : nullptr,
                             static_cast<int>(c.voltage_profile_v.size())}});
            current_offset += c.current_profile_a.size();
            voltage_offset += c.voltage_profile_v.size();
            if (c.isothermal && c.heat_capacity_law.factor.count)
                throw std::runtime_error("Variable capacity requires finite_volume transfer");
            has_isothermal |= c.isothermal;
            has_spatial |= !c.isothermal;
        }
        materials.upload(host);
        reference_materials = host;
        if (!s.electrical_circuit.branches.empty()) {
            validate_circuit(s);
            const auto& circuit = s.electrical_circuit;
            const int count = int(circuit.branches.size());
            const int n = circuit_unknowns(circuit.nodes, circuit.branches.data(), count);
            circuit_branches = std::make_unique<Buffer<CircuitBranch>>(count);
            circuit_branches->upload(circuit.branches);
            circuit_mask = std::make_unique<Buffer<int>>(s.cells.size());
            std::vector<int> mask(s.cells.size(), 0);
            for (const auto& b : circuit.branches)
                if (b.kind == CircuitKind::Battery) mask[b.object] = 1;
            circuit_mask->upload(mask);
            circuit_r = std::make_unique<Buffer<double>>(count);
            circuit_i = std::make_unique<Buffer<double>>(count);
            circuit_v = std::make_unique<Buffer<double>>(count);
            circuit_matrix = std::make_unique<Buffer<double>>(n*n);
            circuit_rhs = std::make_unique<Buffer<double>>(n);
            circuit_voltage = std::make_unique<Buffer<CurrentPoint>>(s.cells.size());
        }
        heater_voltage_points = std::make_unique<Buffer<CurrentPoint>>(s.cells.size());
        if (s.gas_model.split_transport) {
            gas_stage = std::make_unique<Buffer<GasState>>(s.gas_compartments.size());
            mass_bindings = std::make_unique<Buffer<MassBinding>>(mass_mesh.bindings.size());
            mass_bindings->upload(mass_mesh.bindings);
            mass_dt = std::make_unique<Buffer<double>>(2);
        }
        std::vector<int> conversion, legacy;
        for (const auto& c : s.cells)
            has_volatile |= c.conversion.volatile_phase.initial_mass_kg > 0.;
        for (std::size_t i = 0; i < m.volumes.size(); ++i)
            (s.cells[m.volumes[i].object].conversion.enabled ? conversion : legacy)
                .push_back(static_cast<int>(i));
        conversion_ids = std::make_unique<Buffer<int>>(conversion.size());
        conversion_ids->upload(conversion);
        legacy_ids = std::make_unique<Buffer<int>>(legacy.size());
        legacy_ids->upload(legacy);
        if (s.gas_model.coupled_gas_films) {
            std::vector<int> ids;
            for (const auto& p : m.gas_patches)
                ids.push_back(p.volume);
            std::sort(ids.begin(), ids.end());
            ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
            film_aging_ids = std::make_unique<Buffer<int>>(ids.size());
            film_aging_ids->upload(ids);
        }
        gas_patches.upload(m.gas_patches);
        std::vector<DeviceGas> gas_data;
        std::vector<int> aperture_ids;
        std::vector<std::vector<int>> aperture_rows(s.gas_channels.size());
        for (std::size_t j = 0; j < m.gas_patches.size(); ++j)
            if (m.gas_patches[j].channel >= 0) {
                aperture_ids.push_back(static_cast<int>(j));
                aperture_rows[m.gas_patches[j].channel].push_back(static_cast<int>(j));
            }
        aperture_patch_ids = std::make_unique<Buffer<int>>(aperture_ids.size());
        aperture_patch_ids->upload(aperture_ids);
        aperture_channel_rows = std::make_unique<Incidence>(aperture_rows);
        for (std::size_t i = 0; i < s.gas_compartments.size(); ++i) {
            const auto& g = s.gas_compartments[i];
            gas_data.push_back({g.volume_m3, gas_combustion_law(g), mass_mesh.bindings[i].thermal_volume});
        }
        gas_config.upload(gas_data);
        std::vector<DeviceChannel> channel_data;
        for (const auto& ch : s.gas_channels)
            channel_data.push_back({ch.from, ch.to, ch.shell_object, gas_channel_area(s, ch),
                                    ch.discharge_coefficient, ch.opening_pressure_pa, ch.outside_pressure_pa,
                                    ch.strength_reference_k, ch.strength_temperature_scale_k, ch.rupture,
                                    ch.opening, ch.shell_thermal_volume});
        channels.upload(channel_data);
        if (std::any_of(s.gas_channels.begin(), s.gas_channels.end(),
                        [](const auto& c) { return c.opening.enabled && discrete_shell_events(c.opening); }))
            shell_event_pressure_snapshot = std::make_unique<Buffer<GasState>>(s.gas_compartments.size());
        if (std::any_of(s.gas_channels.begin(), s.gas_channels.end(), [](const auto& c) {
                return c.opening.enabled && c.opening.fracture_energy_j_per_m2 > 0;
            }))
            shell_work_power = std::make_unique<Buffer<double>>(s.gas_compartments.size());
        if (std::any_of(s.gas_channels.begin(), s.gas_channels.end(),
                        [](const auto& c) { return c.opening.enabled; })) {
            std::vector<DeviceShellPatch> patches(m.volumes.size());
            for (std::size_t j = 0; j < s.gas_channels.size(); ++j) {
                const auto& ch = s.gas_channels[j];
                if (!ch.opening.enabled)
                    continue;
                const auto& panel = s.cells[ch.shell_object];
                const auto port = std::find_if(s.ports.begin(), s.ports.end(),
                                               [&](const auto& p) { return p.id == ch.port_id; });
                const int axis = port->surface[1] - 'x', u = (axis + 1) % 3, v = (axis + 2) % 3;
                for (int k = m.object_offsets[ch.shell_object]; k < m.object_offsets[ch.shell_object + 1];
                     ++k) {
                    if (ch.shell_thermal_volume >= 0 &&
                        k != m.object_offsets[ch.shell_object] + ch.shell_thermal_volume)
                        continue;
                    const auto& tv = m.volumes[k];
                    const double center[3]{tv.local_center_m.x, tv.local_center_m.y, tv.local_center_m.z};
                    const double size[3]{tv.size_m.x, tv.size_m.y, tv.size_m.z};
                    patches[k] = {static_cast<int>(j),
                                  shell_seam_fraction(center[u], center[v], size[u], size[v],
                                                      panel.dimensions_m[u], panel.dimensions_m[v],
                                                      ch.opening.seam_width_m)};
                    patches[k].area_m2 = size[u] * size[v];
                    patches[k].rectangle =
                        opening_rectangle(center[u], center[v], size[u], size[v], panel.dimensions_m[u],
                                          panel.dimensions_m[v], ch.opening.seam_width_m);
                    patches[k].weak_fraction = patches[k].seam_fraction;
                    patches[k].weak_factor = ch.opening.seam_strength_factor;
                    if (ch.opening.valve_radius_m > 0) {
                        if (ch.opening.deterministic_aperture && patches[k].seam_fraction > 0)
                            throw std::runtime_error(
                                "A local patch cannot declare both a valve and a weak seam");
                        patches[k].weak_fraction =
                            clipped_disk_area(size[u], size[v], ch.opening.valve_u_m, ch.opening.valve_v_m,
                                              ch.opening.valve_radius_m) /
                            patches[k].area_m2;
                        patches[k].weak_factor = ch.opening.valve_pressure_factor;
                    }
                    patches[k].random_key = pixel_mix(
                        ch.opening.rupture_seed ^ pixel_mix(pixel_channel_key(ch.id)) ^
                        pixel_mix(static_cast<std::uint64_t>(k - m.object_offsets[ch.shell_object])));
                    if (ch.opening.stochastic_pixels)
                        patches[k].failure_hazard = pixel_failure_hazard(
                            ch.opening.rupture_seed, pixel_channel_key(ch.id),
                            static_cast<std::uint64_t>(k - m.object_offsets[ch.shell_object]));
                }
            }
            bool has_plate_front = false;
            for (std::size_t object = 0; object < s.cells.size(); ++object) {
                const int begin = m.object_offsets[object], end = m.object_offsets[object + 1];
                bool grouped = false;
                for (int k = begin; k < end; ++k)
                    grouped |= patches[k].channel >= 0 &&
                               s.gas_channels[patches[k].channel].opening.whole_plate;
                if (!grouped) continue;
                has_plate_front = true;
                const auto& panel = s.cells[object];
                if (panel.shell_contents_id.empty())
                    throw std::runtime_error("whole_plate requires a physical planar shell face");
                const int axis = panel.id.back() - 'x';
                if (axis < 0 || axis > 2 || panel.thermal_divisions[axis] != 1)
                    throw std::runtime_error("whole_plate requires one thermal layer through thickness");
                const int u = (axis + 1) % 3, v = (axis + 2) % 3;
                for (int k = begin; k < end; ++k) {
                    if (patches[k].channel < 0)
                        throw std::runtime_error("whole_plate requires a gas channel for every face tile");
                    const auto& ch = s.gas_channels[patches[k].channel];
                    if (!ch.opening.whole_plate || !ch.opening.deterministic_aperture ||
                        ch.shell_thermal_volume != k - begin)
                        throw std::runtime_error("whole_plate cannot mix local/legacy rupture on one face");
                    const auto& a = m.volumes[k];
                    const double ac[]{a.local_center_m.x, a.local_center_m.y, a.local_center_m.z};
                    const double as[]{a.size_m.x, a.size_m.y, a.size_m.z};
                    int neighbors = 0;
                    for (int j = begin; j < end; ++j) {
                        if (j == k) continue;
                        const auto& b = m.volumes[j];
                        const double bc[]{b.local_center_m.x, b.local_center_m.y, b.local_center_m.z};
                        const double bs[]{b.size_m.x, b.size_m.y, b.size_m.z};
                        const double du = std::abs(ac[u] - bc[u]), dv = std::abs(ac[v] - bc[v]);
                        const bool edge = (std::abs(du - .5 * (as[u] + bs[u])) < 1e-10 &&
                                           dv < .5 * (as[v] + bs[v]) - 1e-10) ||
                                          (std::abs(dv - .5 * (as[v] + bs[v])) < 1e-10 &&
                                           du < .5 * (as[u] + bs[u]) - 1e-10);
                        if (!edge) continue;
                        if (neighbors == 4)
                            throw std::runtime_error("whole_plate requires a conforming planar grid");
                        patches[k].neighbors[neighbors++] = j;
                    }
                }
            }
            if (has_plate_front) {
                plate_delta = std::make_unique<Buffer<double>>(patches.size());
                plate_order = std::make_unique<Buffer<int>>(patches.size());
                fracture_budget_scale = std::make_unique<Buffer<double>>(gas_a.n);
            }
            shell_patches = std::make_unique<Buffer<DeviceShellPatch>>(patches.size());
            shell_patches->upload(patches);
            shell_geometry = patches;
            shell_damage = std::make_unique<Buffer<ShellDamageState>>(patches.size());
            shell_next = std::make_unique<Buffer<ShellDamageState>>(patches.size());
            shell_rates = std::make_unique<Buffer<ShellDamageState>>(patches.size());
            if (std::any_of(s.gas_channels.begin(), s.gas_channels.end(),
                            [](const auto& c) { return c.opening.deterministic_aperture; })) {
                prefix_solid = std::make_unique<Buffer<ThermalState>>(state_a.n);
                prefix_gas = std::make_unique<Buffer<GasState>>(gas_a.n);
                prefix_channels = std::make_unique<Buffer<GasChannelState>>(channel_state.n);
                prefix_shell = std::make_unique<Buffer<ShellDamageState>>(patches.size());
            }
        }
        std::vector<std::vector<int>> vp(m.volumes.size()), gp(s.gas_compartments.size()),
            gc(s.gas_compartments.size()), gs(s.gas_compartments.size()), sc(s.cells.size());
        for (std::size_t j = 0; j < m.gas_patches.size(); ++j) {
            vp[m.gas_patches[j].volume].push_back(j);
            gp[m.gas_patches[j].compartment].push_back(j);
        }
        for (std::size_t j = 0; j < s.gas_channels.size(); ++j) {
            const auto& c = s.gas_channels[j];
            gc[c.from].push_back(j);
            if (c.to >= 0)
                gc[c.to].push_back(j);
            if (c.rupture)
                sc[c.shell_object].push_back(j);
        }
        for (std::size_t j = 0; j < m.volumes.size(); ++j) {
            if (mass_mesh.source_gas[j] >= 0)
                gs[mass_mesh.source_gas[j]].push_back(j);
        }
        volume_patches = std::make_unique<Incidence>(vp);
        gas_patches_rows = std::make_unique<Incidence>(gp);
        gas_channels_rows = std::make_unique<Incidence>(gc);
        gas_sources_rows = std::make_unique<Incidence>(gs);
        shell_channels_rows = std::make_unique<Incidence>(sc);
        std::vector<int> starts{0};
        std::vector<ProfilePoint> points;
        for (const auto& b : s.boundaries) {
            for (const auto& p : b.temperature_profile)
                points.push_back({p[0], p[1]});
            starts.push_back(static_cast<int>(points.size()));
        }
        profile_offsets.upload(starts);
        profile_points = std::make_unique<Buffer<ProfilePoint>>(points.size());
        profile_points->upload(points);
        std::vector<double> constants;
        std::vector<DeviceLink> device_links;
        std::vector<ConnectivityLaw> unique_laws;
        std::map<decltype(law_key(ConnectivityLaw{})), int> law_ids;
        std::array<std::vector<int>, 4> edge_groups;
        for (const auto& e : m.links) {
            const auto& l = e.law;
            bool fixed = l.kind == TransitionKind::Constant && l.damage_rate_at_onset_per_s == 0 &&
                         l.endpoint_damage_weight == 0 && l.active_conductance_w_per_k == 0 &&
                         e.initial.damage == 0 && e.initial.phase == EdgePhase::Low &&
                         e.initial.l_m == l.l_low_m;
            constants.push_back(fixed ? l.background_conductance_w_per_k * e.weight : -1);
            const int kind = fixed                                        ? 0
                             : l.kind == TransitionKind::DirectedGeometry ? 1
                             : l.kind == TransitionKind::StateGeometry    ? 2
                                                                          : 3;
            edge_groups[kind].push_back(static_cast<int>(device_links.size()));
            int law_id = -1;
            if (!fixed) {
                auto [entry, inserted] = law_ids.emplace(law_key(l), static_cast<int>(unique_laws.size()));
                law_id = entry->second;
                if (inserted)
                    unique_laws.push_back(l);
            }
            device_links.push_back({e.a, e.b, e.interface_index, law_id, e.weight,
                                    e.series_resistance_k_per_w, e.boundary_temperature_k, e.maximum_g,
                                    e.boundary_index, e.geometry_axis, e.series_resistance_a_k_per_w});
        }
        constant_g.upload(constants);
        links.upload(device_links);
        laws = std::make_unique<Buffer<ConnectivityLaw>>(unique_laws.size());
        laws->upload(unique_laws);
        for (int k = 0; k < 4; ++k) {
            edge_ids[k] = std::make_unique<Buffer<int>>(edge_groups[k].size());
            edge_ids[k]->upload(edge_groups[k]);
        }
    }
    template <int Kind> void propose_edges(RuntimeParameters runtime, double time) {
        const auto& ids = *edge_ids[Kind];
        if (!ids.n)
            return;
        edge_kernel<Kind><<<work_blocks(ids.n), work_threads>>>(
            ids.n, ids.p, links.p, edge_state.p, edge_next.p, state_a.p, runtime, dt.p, flux.p, constant_g.p,
            laws->p, time, profile_offsets.p, profile_points->p, volumes.p, materials.p);
    }
    void propose_edges(RuntimeParameters runtime, double time) {
        propose_edges<0>(runtime, time);
        propose_edges<1>(runtime, time);
        propose_edges<2>(runtime, time);
        propose_edges<3>(runtime, time);
    }
    void select_solid_step(double time, double cap, double safety, const RuntimeParameters&);
    void apply_step_floor(double cap);
    void solve_electrical(double time, double horizon) {
        if (electrical_objects.n)
            electrical_kernel<<<electrical_objects.n, 128>>>(
                electrical_objects.n, electrical_objects.p, object_offsets.p, electrical_edge_offsets.p,
                volumes.p, materials.p, state_a.p, {electrical_offsets.p, electrical_adjacency.p},
                electrical_edges.p, electrical_g.p, electrical_sigma.p, electrical_diagonal.p, electrical_r.p,
                electrical_z.p, electrical_p.p, electrical_ap.p, electrical_potential_a.p,
                electrical_potential_b.p, electrical_power.p, electrical_a.p, electrical_b.p, time, horizon,
                circuit_mask ? circuit_mask->p : nullptr);
        if (circuit_branches)
            circuit_kernel<<<1, 1>>>(scenario.electrical_circuit.nodes, circuit_branches->p,
                int(circuit_branches->n), circuit_r->p, circuit_matrix->p, circuit_rhs->p,
                circuit_i->p, circuit_v->p, circuit_voltage->p, materials.p, electrical_b.p);
    }
    void propose_gas_transport(double time);
    void select_gas_step(double time, double safety, const RuntimeParameters&);
    void propose_processes(double time, const RuntimeParameters&);
    void propose_local_chemistry(double time, const RuntimeParameters&);
    void gather_solid_candidate(double time, const RuntimeParameters&);
    void gather_candidate(double time, const RuntimeParameters&);
    double close_and_validate_candidate(double time, const RuntimeParameters&);
    double prepare_split_step(double time, double cap, double safety, const RuntimeParameters&);
    bool prepare_fracture_candidate(double time, double& cap, double safety, double retry_floor,
                                    const RuntimeParameters&);
    void prepare_thermal_candidate(double time, const RuntimeParameters&);
    double prepare_cuda_mass_candidate(double time, const RuntimeParameters&);
    double prepare_cpu_mass_candidate(double time, const RuntimeParameters&);
    bool prepare_explicit_mass_candidate(double time, double& cap, double safety, double retry_floor,
                                         const RuntimeParameters&, double& accepted_dt);
    void begin_prefix_inputs();
    void restore_prefix_inputs() noexcept;
    void apply_leading_shell(double time, const RuntimeParameters&);
    void equilibrate(ThermalState* solid, GasState* gas, double speed) {
        if (gas_a.n)
            mixture_equilibrium_kernel<<<work_blocks(gas_a.n), work_threads>>>(
                gas_a.n, mass_bindings->p, gas_config.p, volumes.p, materials.p, solid, gas,
                scenario.gas_model, speed);
    }
    void aggregate() {
        macro_kernel<<<scenario.cells.size(), 128>>>(volumes.p, materials.p, state_a.p, object_offsets.p,
                                                     macro_cells.p);
        if (macro_edges.n)
            interface_macro_kernel<<<macro_edges.n, 128>>>(
                macro_edges.n, links.p, edge_state.p, interface_offsets.p, interface_ids.p, macro_edges.p);
        check(cudaGetLastError(), "CUDA aggregate launch");
    }
};
