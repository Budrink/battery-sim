// Read-only snapshot audit of the ACTUAL production CUDA pressure operator.
// No reaction/film evolution: this isolates a transport window, not a replay
// of the historical coupled trajectory or a replacement CPU gas solver.
#include "../src/batched_gas.cuh"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <numeric>
#include <cstdlib>
#include <stdexcept>
using J=nlohmann::json;
namespace kg=battery::kawasaki;
namespace bg=battery::batched_gas;
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T> struct Buffer {
    T* p=nullptr;std::size_t n;
    explicit Buffer(std::size_t count):n(count){if(n)check(cudaMalloc(&p,n*sizeof(T)));}
    ~Buffer(){if(p)cudaFree(p);}
    void put(const std::vector<T>& v){if(v.size()!=n)throw std::runtime_error("Buffer size");if(n)check(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> get(){std::vector<T> v(n);if(n)check(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
struct Result {
    std::vector<kg::Node> nodes;
    std::vector<bg::Proposal> proposals;
    std::vector<bg::Budget> budgets;
    std::vector<battery::GasChannelState> channels;
    std::vector<bg::AverageState> average;
    double scale=1.;unsigned backtracks=0;
    int error=0;bg::Control control{};
    double wall_ms=0.;
};
Result advance(const std::vector<kg::Node>& input,const std::vector<kg::Edge>& es,
    kg::Settings settings,battery::GasModel model,double dt,int block=128) {
    if(dt<.005 || !std::isfinite(dt))throw std::runtime_error("Audit window must be >=5 ms");
    model.cuda_network_transport=true;model.cuda_batched_transport=true;
    model.cuda_deterministic_transport=false;model.minimum_step_s=.005;
    model.coupled_gas_films=false;
    Buffer<kg::Node> nodes(input.size()),scratch(input.size());nodes.put(input);
    Buffer<kg::Edge> edges(es.size());edges.put(es);
    Buffer<kg::Clock> clocks(es.size());clocks.put(std::vector<kg::Clock>(es.size()));
    Buffer<bg::Proposal> proposals(es.size());Buffer<bg::Budget> budgets(input.size());budgets.put(std::vector<bg::Budget>(input.size()));
    Buffer<battery::GasChannelState> channels(es.size());
    std::vector<battery::GasChannelState> initial(es.size());
    std::vector<std::vector<int>> adjacency(input.size());
    for(int e=0;e<int(es.size());++e){initial[e].open=es[e].area>0;initial[e].effective_area_m2=es[e].area;
        adjacency[es[e].a].push_back(e);adjacency[es[e].b].push_back(e);}
    channels.put(initial);std::vector<int> flat,offset{0};
    for(const auto& row:adjacency){flat.insert(flat.end(),row.begin(),row.end());offset.push_back(flat.size());}
    Buffer<int> ids(flat.size()),offsets(offset.size()),error(1);ids.put(flat);offsets.put(offset);error.put({0});
    Buffer<bg::Control> control(1);control.put({bg::Control{}});
    Buffer<bg::AverageState> average_a(model.averaged_network_transport?input.size():0),average_b(average_a.n);
    Buffer<bg::MixSolveState> mix_work(model.implicit_network_mixing?input.size():0);
    int blocks=0,repeats=1;
    if(const char* p=std::getenv("BATTERY_GAS_BLOCKS"))blocks=std::stoi(p);
    if(const char* p=std::getenv("BATTERY_GAS_THREADS"))block=std::stoi(p);
    if(const char* p=std::getenv("BATTERY_GAS_REPEATS"))repeats=std::stoi(p);
    Buffer<kg::Node> saved(input.size());saved.put(input);
    Buffer<battery::GasChannelState> saved_channels(es.size());saved_channels.put(initial);
    cudaEvent_t begin,end;check(cudaEventCreate(&begin));check(cudaEventCreate(&end));check(cudaEventRecord(begin));
    for(int i=0;i<repeats;++i){
        check(cudaMemcpyAsync(nodes.p,saved.p,input.size()*sizeof(kg::Node),cudaMemcpyDeviceToDevice));
        if(es.size())check(cudaMemcpyAsync(channels.p,saved_channels.p,es.size()*sizeof(battery::GasChannelState),cudaMemcpyDeviceToDevice));
        check(bg::launch(input.size(),es.size(),offsets.p,ids.p,edges.p,nodes.p,scratch.p,proposals.p,budgets.p,
            clocks.p,channels.p,settings,model,dt,error.p,control.p,block,blocks,{},average_a.p,average_b.p,mix_work.p));}
    check(cudaEventRecord(end));check(cudaEventSynchronize(end));float elapsed;check(cudaEventElapsedTime(&elapsed,begin,end));
    check(cudaEventDestroy(begin));check(cudaEventDestroy(end));
    check(cudaDeviceSynchronize());
    const int code=error.get()[0];
    if((code && code!=23 && code!=24) || control.get()[0].reaction_intervals!=1)throw std::runtime_error("Single-window CUDA audit failed: "+std::to_string(code));
    const auto c=control.get()[0];
    return {code==23?scratch.get():nodes.get(),proposals.get(),budgets.get(),channels.get(),average_a.get(),c.averaged_scale,c.averaged_backtracks,code,c,elapsed/repeats};
}
J audit(const std::vector<kg::Node>& nodes,const std::vector<kg::Edge>& edges,
    const std::vector<std::string>& names,const std::vector<std::string>& edge_names,
    kg::Settings settings,battery::GasModel model,double dt) {
    const auto saved_model=model;
    model.mixing_diffusivity_m2_s=0.;model.mixing_flow_fraction=0.;
    const auto pressure=advance(nodes,edges,settings,model,dt);
    if(pressure.error){
        J attempts=J::array(),rejected=J::array(),packets=J::array();
        for(int i=0;pressure.error==23 && i<8;++i)attempts.push_back({{"scale",pressure.control.averaged_attempt_scale[i]},
            {"invalid_mask",pressure.control.averaged_invalid_mask[i]},
            {"entropy_change_j_per_k",pressure.control.averaged_entropy[i]}});
        for(int i=0;pressure.error==23 && i<int(nodes.size());++i){const auto& n=pressure.nodes[i];
            rejected.push_back({{"id",names[i]},{"mass_kg",n.mass},{"energy_j",n.energy},
                {"temperature_k",kg::temperature(n,settings)},{"pressure_pa",kg::pressure(n,settings)},
                {"entropy_change_j_per_k",kg::entropy(n,settings)-kg::entropy(nodes[i],settings)},
                {"chemical_j",n.chemical},{"oxidizer_kg",n.oxidizer},{"vapor_kg",n.vapor},{"latent_j",n.latent}});}
        for(int e=0;pressure.error==23 && e<int(edges.size());++e){const auto& p=pressure.proposals[e];
            packets.push_back({{"id",edge_names[e]},{"donor",names[p.donor]},
                {"from",names[edges[e].a]},{"to",names[edges[e].b]},
                {"mass_kg",p.mass},{"energy_j",p.energy},{"conductance_kg_s_pa",p.cap}});}
        J worst=J::array();
        if(pressure.error==24) {
            std::vector<int> order(nodes.size());std::iota(order.begin(),order.end(),0);
            const auto error=[&](int i){const auto& a=pressure.average[i];
                return std::abs(a.residual)/(a.compliance*std::max(a.pressure,1.));};
            std::sort(order.begin(),order.end(),[&](int a,int b){return error(a)>error(b);});
            for(int k=0;k<std::min(10,int(order.size()));++k){const int i=order[k];const auto& a=pressure.average[i];
                worst.push_back({{"id",names[i]},{"relative_residual",error(i)},{"residual_kg",a.residual},
                    {"compliance",a.compliance},{"pressure_pa",a.pressure},{"correction_pa",a.correction},
                    {"mass_kg",nodes[i].mass},{"volume_m3",nodes[i].volume}});}
        }
        return {{"error_code",pressure.error},{"window_s",dt},{"attempts",attempts},{"worst_nodes",worst},
            {"pressure_iterations_used",pressure.control.averaged_pressure_used},
            {"pressure_relative_residual",pressure.control.averaged_pressure_residual},
            {"rejected_nodes",rejected},{"packets",packets}};
    }
    const auto mixed=advance(nodes,edges,settings,saved_model,dt);
    std::vector<double> requests(edges.size()),scales(nodes.size(),1.),out(nodes.size());
    for(int e=0;e<int(edges.size());++e){const auto& edge=edges[e];
        requests[e]=model.implicit_network_mixing?dt*battery::mixing_conductance(pressure.nodes[edge.a],pressure.nodes[edge.b],edge,settings,
            saved_model.mixing_diffusivity_m2_s,saved_model.mixing_flow_fraction,pressure.proposals[e].mass/dt):
            battery::mixing_request(pressure.nodes[edge.a],pressure.nodes[edge.b],edge,settings,
            saved_model.mixing_diffusivity_m2_s,saved_model.mixing_flow_fraction,pressure.proposals[e].mass/dt,dt);
        out[edge.a]+=requests[e];out[edge.b]+=requests[e];}
    for(int i=0;i<int(nodes.size());++i)if(out[i]>0. && !model.implicit_network_mixing)scales[i]=std::min(1.,.5*pressure.nodes[i].mass/out[i]);
    J rows=J::array(),budgets=J::array();double mass0=0,mass1=0,energy0=0,energy1=0;
    for(int i=0;i<int(nodes.size());++i){mass0+=nodes[i].mass;mass1+=mixed.nodes[i].mass;
        energy0+=nodes[i].energy;energy1+=mixed.nodes[i].energy;
        budgets.push_back({{"id",names[i]},{"pressure_scale",pressure.budgets[i].scale},
            {"mixing_scale",scales[i]},{"mass_kg",nodes[i].mass},
            {"post_temperature_k",kg::temperature(mixed.nodes[i],settings)},
            {"post_chemical_j",mixed.nodes[i].chemical},{"post_oxidizer_kg",mixed.nodes[i].oxidizer}});}
    for(int e=0;e<int(edges.size());++e){const auto& p=pressure.proposals[e];const auto& edge=edges[e];
        const auto& d=nodes[p.donor];const int receiver=p.donor==edge.a?edge.b:edge.a;
        const double raw=p.rate*dt;
        double pair=0.;
        if(p.rate>0 && p.cap>0){const double gamma=1+settings.R/settings.cv;
            if(kg::pressure(nodes[receiver],settings)/kg::pressure(d,settings)<=std::pow(2/(gamma+1),gamma/(gamma-1)))pair=std::min(p.cap,raw);
            else {const double f=std::min(1.,raw/(2*p.cap));pair=p.cap*f*(2-f);}}
        const double mix=requests[e]*std::min(scales[edge.a],scales[edge.b]);
        const double tolerance=1e-12*std::max(1e-12,mix);
        if(!model.implicit_network_mixing && std::abs(mix-mixed.proposals[e].mixing.mass_each_way)>tolerance)throw std::runtime_error("Host diagnostic does not match GPU mixing budget");
        if(!model.averaged_network_transport && std::abs(p.mass-pair*std::min(pressure.budgets[edge.a].scale,pressure.budgets[edge.b].scale))>1e-12*std::max(1e-12,p.mass))
            throw std::runtime_error("Pressure stages do not reconstruct accepted packet");
        rows.push_back({{"id",edge_names[e]},{"from",names[edge.a]},{"to",names[edge.b]},
            {"donor",names[p.donor]},{"pressure_gap_pa",std::abs(kg::pressure(nodes[edge.a],settings)-kg::pressure(nodes[edge.b],settings))},
            {"raw_mass_kg",raw},{"pair_cap_kg",model.averaged_network_transport?J(nullptr):J(p.cap)},
            {"pair_request_kg",model.averaged_network_transport?p.mass:pair},{"accepted_mass_kg",p.mass},
            {"node_scale",model.averaged_network_transport?pressure.scale:std::min(pressure.budgets[edge.a].scale,pressure.budgets[edge.b].scale)},
            {"mix_request_each_way_kg",requests[e]},{"mix_accepted_each_way_kg",mixed.proposals[e].mixing.mass_each_way},
            {"mix_energy_from_to_j",mixed.proposals[e].mixing.energy},
            {"pressure_energy_from_to_j",pressure.channels[e].transferred_energy_j}});}
    const double dm=mass1-mass0,de=energy1-energy0;
    if(std::abs(dm)>1e-11*mass0 || std::abs(de)>1e-11*energy0)throw std::runtime_error("Snapshot mass/energy does not close");
    return {{"window_s",dt},{"reaction_and_films",false},{"single_production_cuda_window",true},
        {"integrator",model.averaged_network_transport?"averaged":"pairwise"},
        {"net_mass_scale",pressure.scale},{"entropy_backtracks",pressure.backtracks},
        {"pressure_iterations_used",pressure.control.averaged_pressure_used},
        {"pressure_relative_residual",pressure.control.averaged_pressure_residual},
        {"mixing_iterations_used",mixed.control.mixing_iterations},{"mixing_relative_residual",mixed.control.mixing_residual},
        {"window_wall_ms",mixed.wall_ms},{"pressure_wall_ms",pressure.wall_ms},
        {"mass_residual_kg",dm},{"energy_residual_j",de},{"edges",rows},{"nodes",budgets}};
}
kg::Node node(double pressure,double volume){const double m=pressure*volume/(287.*500.);return {m,m*1000.*500.,volume};}
J throughflow(bool averaged=false){
    J rows=J::array();
    for(double volume:{1e-7,1e-6,1e-5})for(double dt:{.005,.01,.02})for(double area_factor:{1.,10.,1000.}){
        std::vector<kg::Node> nodes{node(102325.,1.),node(101825.,volume),node(101325.,1.)};
        kg::Settings s;std::vector<kg::Edge> edges{{0,1,1e-5*area_factor,.7},{1,2,1e-5*area_factor,.7}};
        // Match inlet/outlet initial mass flow: the middle inventory need not
        // grow or shrink, despite a large gross flux through a tiny volume.
        edges[1].area*=kg::flow(nodes[0],nodes[1],edges[0],s)/kg::flow(nodes[1],nodes[2],edges[1],s);
        battery::GasModel model;model.network_pressure_fraction=1.;
        model.averaged_network_transport=averaged;
        const auto result=audit(nodes,edges,{"source","small_transit","sink"},{"inlet","outlet"},s,model,dt);
        const double inlet=result["edges"][0]["accepted_mass_kg"],outlet=result["edges"][1]["accepted_mass_kg"];
        rows.push_back({{"volume_m3",volume},{"dt_s",dt},{"area_factor",area_factor},
            {"raw_flow_kg_s",kg::flow(nodes[0],nodes[1],edges[0],s)},
            {"accepted_inlet_kg_s",inlet/dt},{"accepted_outlet_kg_s",outlet/dt},
            {"node_scale",result["edges"][0]["node_scale"]},
            {"mass_residual_kg",result["mass_residual_kg"]},{"energy_residual_j",result["energy_residual_j"]}});
    }return {{"matched_inlet_outlet_fixture",true},{"integrator",averaged?"averaged":"pairwise"},{"rows",rows}};
}
int main(int argc,char** argv){try{
    if(argc>=2 && std::string(argv[1])=="--mix-chain"){
        kg::Settings settings;std::vector<kg::Node> nodes;
        std::vector<std::string> names;for(int i=0;i<5;++i){const double t=i==0?1000.:300.,v=(i==0 || i==4)?1e-2:1e-7;
            const double mass=1e5*v/(settings.R*t);nodes.push_back({mass,mass*settings.cv*t,v});names.push_back("node-"+std::to_string(i));}
        nodes[0].chemical=nodes[0].mass*1e6;nodes[0].oxidizer=nodes[0].mass*.232;
        std::vector<kg::Edge> edges;std::vector<std::string> ids;for(int i=0;i<4;++i){edges.push_back({i,i+1,1e-4,.7,.02});ids.push_back("edge-"+std::to_string(i));}
        battery::GasModel model;model.averaged_network_transport=true;model.mixing_diffusivity_m2_s=.1;
        model.implicit_network_mixing=argc>=3 && std::string(argv[2])=="implicit";
        auto result=audit(nodes,edges,names,ids,settings,model,.005);J input=J::array();
        for(const auto& n:nodes)input.push_back({{"mass_kg",n.mass},{"temperature_k",kg::temperature(n,settings)},
            {"chemical_j",n.chemical},{"oxidizer_kg",n.oxidizer}});
        result["initial_nodes"]=input;std::cout<<result.dump()<<'\n';return 0;
    }
    const bool averaged=argc==3 && std::string(argv[2])=="averaged";
    if(argc>=2 && std::string(argv[1])=="--throughflow"){std::cout<<throughflow(averaged).dump()<<'\n';return 0;}
    if(argc!=2 && !averaged)throw std::runtime_error("Usage: gas_limiter_snapshot SNAPSHOT.json | --throughflow [averaged]");
    J input;std::ifstream file(argv[1]);file>>input;
    const char* override_scene=std::getenv("BATTERY_GAS_SCENARIO");
    const auto scenario=battery::load_scenario(override_scene?override_scene:input.at("scenario_path").get<std::string>());
    kg::Settings settings;settings.cv=scenario.gas_model.cv_j_per_kg_k;settings.R=scenario.gas_model.specific_r_j_per_kg_k;
    settings.thermal_stirring_coefficient=scenario.gas_model.thermal_stirring_coefficient;
    settings.mixing_length_cap_m=scenario.gas_model.mixing_length_cap_m;
    settings.thermal_stirring_length_m=scenario.gas_model.thermal_stirring_length_m;
    if(scenario.has_gas_mixing_connectivity)settings.mixing_connectivity=scenario.gas_mixing_connectivity;
    settings.mixing_connectivity.endpoint_a=settings.mixing_connectivity.endpoint_b=scenario.gas_surface_connectivity;
    std::vector<kg::Node> nodes;std::vector<kg::Edge> edges;std::vector<std::string> names,ids;
    for(const auto& g:input["nodes"]){kg::Node n;n.mass=g.at("mass_kg");n.volume=g.at("free_volume_m3");
        const double capacity=g.at("heat_capacity_j_per_k");n.solid_capacity=std::max(0.,capacity-settings.cv*n.mass);
        n.energy=g.value("energy_j",capacity*g.at("temperature_k").get<double>());n.chemical=g.at("chemical_j");n.oxidizer=g.at("oxidizer_kg");
        n.vapor=g.at("vapor_kg");n.latent=g.at("latent_j");nodes.push_back(n);names.push_back(g.at("id"));}
    for(int e=0;e<int(scenario.gas_channels.size());++e){const auto& c=scenario.gas_channels[e];
        if(input["channels"][e]["id"]!=c.id)throw std::runtime_error("Channel order changed");
        edges.push_back({c.from,c.to,input["channels"][e].at("effective_area_m2"),c.discharge_coefficient,c.mixing_length_m,c.mixing_axis});
        edges.back().hydraulic_resistance_factor=c.hydraulic_resistance_factor;
        edges.back().molecular_mixing=battery::internal_porous_mixing(scenario.gas_model.free_gas_response_only,c.rupture,
            scenario.gas_compartments[c.from].thermal_cell,scenario.gas_compartments[c.to].thermal_cell);
        ids.push_back(c.id);}
    auto model=scenario.gas_model;if(averaged)model.averaged_network_transport=true;
    if(const auto* value=std::getenv("BATTERY_AVERAGE_PRESSURE_ITERATIONS"))model.averaged_pressure_iterations=std::stoi(value);
    if(const auto* value=std::getenv("BATTERY_AVERAGE_PAYLOAD_ITERATIONS"))model.averaged_payload_iterations=std::stoi(value);
    std::cout<<audit(nodes,edges,names,ids,settings,model,input.at("window_s")).dump()<<'\n';
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
