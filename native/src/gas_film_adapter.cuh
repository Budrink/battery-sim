// Included in the CUDA mixture adapter's anonymous namespace.
// Film heat records have one writer, then one fixed-order gather per solid.
struct FilmLayout {
    std::vector<GasThermalPatch> patches;
    std::vector<int> gas_offsets{0},gas_ids,solid_offsets{0},solid_ids;
    std::vector<double> capacities;
    int nodes=0;
};
FilmLayout film_layout(const Scenario& s,const ThermalMesh& mesh,const MassMesh& mass) {
    FilmLayout l;l.nodes=s.gas_compartments.size();l.patches=mesh.gas_patches;
    std::vector<std::vector<int>> gr(l.nodes),sr(mesh.volumes.size());
    std::vector<bool> bound(mesh.volumes.size());
    for(const auto& b:mass.bindings)if(b.thermal_volume>=0)bound[b.thermal_volume]=true;
    for(const auto& v:mesh.volumes)l.capacities.push_back(v.capacity_j_per_k);
    for(int i=0;i<int(l.patches.size());++i) {
        const auto& p=l.patches[i];const auto& c=s.cells[mesh.volumes[p.volume].object];
        if(p.channel>=0 || bound[p.volume] || mass.bindings[p.compartment].thermal_volume>=0 ||
            c.conversion.enabled || c.heat_capacity_law.factor.count || c.runaway_energy_j>0 || c.isothermal)
            throw std::runtime_error("Coupled films need unbound gas and constant-capacity passive surfaces");
        gr[p.compartment].push_back(i);sr[p.volume].push_back(i);
    }
    for(const auto& row:gr){l.gas_ids.insert(l.gas_ids.end(),row.begin(),row.end());l.gas_offsets.push_back(l.gas_ids.size());}
    for(const auto& row:sr){l.solid_ids.insert(l.solid_ids.end(),row.begin(),row.end());l.solid_offsets.push_back(l.solid_ids.size());}
    return l;
}
__global__ void import_films(int count,GasFilmView view,const kg::Node* nodes,
    kg::Settings settings,GasFilmState* films) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    films[i]=evaluate_gas_surface_contacts(i,kg::temperature(nodes[i],settings),view);
}
__global__ void gather_film_heat(int count,const int* offsets,const int* ids,const double* heat,
    const double* capacity,ThermalState* solid) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count || offsets[i]==offsets[i+1])return;
    double q=0.;for(int j=offsets[i];j<offsets[i+1];++j)q+=heat[ids[j]];
    solid[i].stored_energy_j+=q;solid[i].temperature_k=solid[i].stored_energy_j/capacity[i];
}
struct FilmAdapter {
    DeviceArray<GasFilmState> states;
    DeviceArray<GasThermalPatch> patches;
    DeviceArray<int> gas_offsets,gas_ids,solid_offsets,solid_ids;
    DeviceArray<double> capacities,heat,conductance;
    explicit FilmAdapter(const FilmLayout& l):states(l.nodes),patches(l.patches.size()),
        gas_offsets(l.gas_offsets.size()),gas_ids(l.gas_ids.size()),solid_offsets(l.solid_offsets.size()),
        solid_ids(l.solid_ids.size()),capacities(l.capacities.size()),heat(l.patches.size()),conductance(l.patches.size()) {
        patches.upload(l.patches);gas_offsets.upload(l.gas_offsets);gas_ids.upload(l.gas_ids);
        solid_offsets.upload(l.solid_offsets);solid_ids.upload(l.solid_ids);capacities.upload(l.capacities);
    }
    GasFilmView view(const ThermalState* solid) const {
        return {patches.p,gas_offsets.p,gas_ids.p,solid,capacities.p,heat.p,conductance.p,
            solid_offsets.p,solid_ids.p,int(capacities.n)};
    }
    void prepare(const ThermalState* solid,const kg::Node* nodes,kg::Settings settings) {
        if(heat.n)checked(cudaMemsetAsync(heat.p,0,heat.n*sizeof(double)));
        if(states.n)import_films<<<(states.n+127)/128,128>>>(states.n,view(solid),nodes,settings,states.p);
    }
    void gather(ThermalState* solid) {
        if(capacities.n)gather_film_heat<<<(capacities.n+127)/128,128>>>(capacities.n,solid_offsets.p,solid_ids.p,
            heat.p,capacities.p,solid);
    }
};
