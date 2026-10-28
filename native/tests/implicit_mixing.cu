#include "../src/batched_gas.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include <random>
#include <stdexcept>
#include <vector>
namespace kg=battery::kawasaki;
namespace bg=battery::batched_gas;
void require(bool v,const char* s){if(!v)throw std::runtime_error(s);}
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>struct Buffer{
    T* p=nullptr;size_t n;
    explicit Buffer(size_t size):n(size){if(n)check(cudaMalloc(&p,n*sizeof(T)));}
    ~Buffer(){if(p)cudaFree(p);}
    void put(std::vector<T> v){require(n==v.size(),"size");if(n)check(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> get(){std::vector<T> v(n);if(n)check(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
void opened_boundary_response() {
    require(battery::internal_porous_mixing(true,false,0,0),"internal pores lost their calibrated reference");
    require(!battery::internal_porous_mixing(true,true,0,-1),"open shell was classified as an internal pore");
    require(!battery::internal_porous_mixing(true,false,0,1),"different objects share an internal pore law");
    require(!battery::internal_porous_mixing(true,false,-1,-1),"free gas became an internal pore");
    require(!battery::internal_porous_mixing(false,false,0,0),"disabled molecular selector still applies");
    kg::Settings settings;settings.mixing_connectivity=battery::reference_directed_connectivity(1000.);
    const kg::Node a{.001,.001*settings.cv*600.,.001},b{.001,.001*settings.cv*300.,.0005};
    kg::Edge edge{0,1,1e-6,.7,.001};edge.molecular_mixing=true;
    const double internal=battery::mixing_conductance(a,b,edge,settings,2e-5,.25,0.);
    edge.molecular_mixing=battery::internal_porous_mixing(true,true,0,-1);
    const double opened=battery::mixing_conductance(a,b,edge,settings,2e-5,.25,0.);
    require(std::abs(opened/internal-1000.)<1e-9,"opened hole ignores free-gas S at zero pressure difference");
    edge.area=0.;require(battery::mixing_conductance(a,b,edge,settings,2e-5,.25,0.)==0.,"closed shell leaks mixing heat");
    std::cout<<"opened boundary: internal reference retained, exterior S active at equal pressure, closed aperture blocked\n";
}
double throughflow_film(int blocks) {
    const double dt=.005,cs=.2,g=10.;kg::Settings settings;
    auto gas=[&](double t,double volume){double m=1e5*volume/(settings.R*t);return kg::Node{m,m*settings.cv*t,volume};};
    const std::vector<kg::Node> input{gas(1000.,.01),gas(300.,1e-7)};
    const kg::Edge edge{0,1,1e-4,.7,.02};
    const double w=dt*settings.cv*battery::mixing_conductance(input[0],input[1],edge,settings,.1,0.,0.);
    const double c0=input[0].mass*settings.cv,c1=input[1].mass*settings.cv,f=dt*g;
    const double t1=(c1*300.+w*c0*1000./(c0+w)+f*cs*300./(cs+f))/(c1+w+f-w*w/(c0+w)-f*f/(cs+f));
    const double t0=(c0*1000.+w*t1)/(c0+w),ts=(cs*300.+f*t1)/(cs+f);
    Buffer<kg::Node> nodes(2),scratch(2);nodes.put(input);
    Buffer<kg::Edge> edges(1);edges.put({edge});Buffer<kg::Clock> clocks(1);clocks.put({{}});
    Buffer<bg::Proposal> proposals(1);Buffer<bg::Budget> budgets(2);
    Buffer<battery::GasChannelState> channels(1);battery::GasChannelState ch;ch.open=true;ch.effective_area_m2=edge.area;channels.put({ch});
    Buffer<int> offsets(3),ids(2),error(1);offsets.put({0,1,2});ids.put({0,0});error.put({0});
    Buffer<bg::Control> control(1);control.put({{}});Buffer<bg::AverageState>a(2),b(2);Buffer<bg::MixSolveState> work(3);
    battery::GasThermalPatch patch;patch.volume=0;patch.compartment=1;patch.law=battery::reference_gas_connectivity(g);
    patch.geometry_weight=1.;
    Buffer<battery::GasThermalPatch> patches(1);patches.put({patch});
    Buffer<int> go(3),gi(1),so(2),si(1);go.put({0,0,1});gi.put({0});so.put({0,1});si.put({0});
    Buffer<double> capacity(1),heat(1),conductance(1);capacity.put({cs});heat.put({0.});conductance.put({g});
    battery::ThermalState solid;solid.temperature_k=300.;solid.stored_energy_j=cs*300.;Buffer<battery::ThermalState> solids(1);solids.put({solid});
    Buffer<battery::GasCombustionLaw> laws(2);laws.put({{}, {}});Buffer<battery::GasFilmState> films(2);
    battery::GasFilmView view{patches.p,go.p,gi.p,solids.p,capacity.p,heat.p,conductance.p,so.p,si.p,1};
    battery::GasModel model;model.cuda_network_transport=model.averaged_network_transport=model.implicit_network_mixing=true;
    model.joint_gas_surface_mixing=model.coupled_gas_films=true;model.minimum_step_s=.005;
    model.mixing_diffusivity_m2_s=.1;model.coupled_transport_interval_s=dt;
    check(bg::launch(2,1,offsets.p,ids.p,edges.p,nodes.p,scratch.p,proposals.p,budgets.p,clocks.p,channels.p,
        settings,model,dt,error.p,control.p,128,blocks,{laws.p,false,films.p,view},a.p,b.p,work.p));
    check(cudaDeviceSynchronize());require(error.get()[0]==0,"joint gas/surface kernel failed");
    const auto output=nodes.get();const double q=heat.get()[0];
    const double err=std::max({std::abs(kg::temperature(output[0],settings)-t0),std::abs(kg::temperature(output[1],settings)-t1),std::abs(300.+q/cs-ts)});
    std::cout<<"joint throughflow film: blocks="<<blocks<<" max_error_K="<<err<<" heat_J="<<q<<" resident_heat_J="<<c1*700.<<'\n';
    require(err<.002,"joint film differs from dense backward-Euler solution");
    require(q>10.*c1*700.,"heat uptake still capped by resident gas");
    require(std::abs(output[0].energy+output[1].energy+q-input[0].energy-input[1].energy)<1e-10,"joint heat not conservative");
    require(output[0].mass==input[0].mass && output[1].mass==input[1].mass,"counterexchange changes net mass");
    return q;
}
__global__ void species_kernel(int count,int edge_count,const int* offsets,const int* ids,
    const kg::Edge* edges,kg::Node* nodes,bg::Proposal* flux,battery::GasChannelState* channels,
    kg::Settings settings,battery::GasModel model,double dt,bg::MixSolveState* work,bg::Control* control,int* error) {
    __shared__ double shared[128];
    const bool ok=bg::implicit_mix_snapshot(cooperative_groups::this_grid(),blockIdx.x*blockDim.x+threadIdx.x,
        gridDim.x*blockDim.x,count,edge_count,offsets,ids,edges,nodes,flux,channels,
        settings,model,dt,work,control,shared,error);
    if(!ok && blockIdx.x==0 && threadIdx.x==0)for(int i=0;i<count;++i) {
        auto n=nodes[i];
        for(int j=offsets[i];j<offsets[i+1];++j) {
            const int e=ids[j];const double s=edges[e].a==i?-1.:1.;const auto& p=flux[e].mixing;
            n.energy+=s*p.energy;n.chemical+=s*p.chemical;n.oxidizer+=s*p.oxidizer;n.vapor+=s*p.vapor;n.latent+=s*p.latent;
        }
        if(n.chemical<0 || n.oxidizer<0 || n.vapor<0 || n.latent<0 || n.oxidizer+n.vapor>n.mass+1e-14)
            printf("Rejected row %d m=%.17g chemical=%.17g O2=%.17g vapor=%.17g latent=%.17g\n",i,n.mass,n.chemical,n.oxidizer,n.vapor,n.latent);
    }
}
void nearly_pure_species(int grid_blocks) {
    std::mt19937 rng(412);std::uniform_real_distribution<double> uniform(0.,1.);
    const int count=40,edge_count=count-1;double dt=.02;kg::Settings settings;
    battery::GasModel model;model.mixing_diffusivity_m2_s=.1;
    model.implicit_temperature_tolerance_k=.02;model.implicit_tracer_relative_tolerance=.001;
    for(int trial=0;trial<32;++trial) {
        std::vector<kg::Node> input(count);std::vector<kg::Edge> topology;
        for(auto& n:input) {
            n.mass=pow(10.,-14.+11.*uniform(rng));n.solid_capacity=settings.cv*pow(10.,-10.+10.*uniform(rng));
            const double t=300.+1700.*uniform(rng);n.energy=(settings.cv*n.mass+n.solid_capacity)*t;
            n.volume=n.mass*settings.R*t/1e5;
            n.chemical=n.mass*1e6*(uniform(rng)<.5?0.:1.);
            n.oxidizer=n.mass*(uniform(rng)<.5?0.:.232);
            n.vapor=n.oxidizer>0.?0.:n.mass;n.latent=n.vapor*5e5;
        }
        std::vector<int> starts{0},indices;
        for(int i=0;i<count;++i) {
            if(i)indices.push_back(i-1);if(i<edge_count)indices.push_back(i);
            starts.push_back(indices.size());
            if(i<edge_count) {
                kg::Edge e{i,i+1,1.,.7,.01};
                const double weight=pow(10.,-17.+15.*uniform(rng));
                e.area=weight/(dt*battery::mixing_conductance(input[i],input[i+1],e,settings,.1,0.,0.));
                topology.push_back(e);
            }
        }
        Buffer<kg::Node> nodes(count);nodes.put(input);Buffer<kg::Edge> edges(edge_count);edges.put(topology);
        Buffer<int> offsets(count+1),ids(indices.size()),error(1);offsets.put(starts);ids.put(indices);error.put({0});
        Buffer<bg::Proposal> flux(edge_count);flux.put(std::vector<bg::Proposal>(edge_count));
        Buffer<battery::GasChannelState> channels(edge_count);std::vector<battery::GasChannelState> channel(edge_count);
        for(int i=0;i<edge_count;++i){channel[i].open=true;channel[i].effective_area_m2=topology[i].area;}channels.put(channel);
        Buffer<bg::MixSolveState> work(count);Buffer<bg::Control> control(1);control.put({{}});
        void* args[]={const_cast<int*>(&count),const_cast<int*>(&edge_count),&offsets.p,&ids.p,&edges.p,&nodes.p,
            &flux.p,&channels.p,&settings,&model,&dt,&work.p,&control.p,&error.p};
        check(cudaLaunchCooperativeKernel((void*)species_kernel,grid_blocks,128,args));check(cudaDeviceSynchronize());
        if(error.get()[0])std::cerr<<"mix solve trial="<<trial<<" code="<<error.get()[0]
            <<" iterations="<<control.get()[0].mixing_iterations<<" residual="<<control.get()[0].mixing_residual<<'\n';
        require(error.get()[0]==0,"bounded composition solve failed");
        const auto result=nodes.get();double mass=0.,energy=0.,chemical=0.,oxygen=0.,vapor=0.,latent=0.;
        for(int i=0;i<count;++i){const auto& n=result[i];
            if(!(n.chemical>=0 && n.oxidizer>=0 && n.vapor>=0 && n.latent>=0 && n.oxidizer+n.vapor<=n.mass+1e-14)) {
                std::cerr<<"species bound: trial="<<trial<<" node="<<i<<" blocks="<<grid_blocks
                    <<" m="<<n.mass<<" O2="<<n.oxidizer<<" vapor="<<n.vapor<<'\n';
                throw std::runtime_error("counterexchange violates local component inventory");
            }
            mass+=n.mass-input[i].mass;energy+=n.energy-input[i].energy;chemical+=n.chemical-input[i].chemical;
            oxygen+=n.oxidizer-input[i].oxidizer;vapor+=n.vapor-input[i].vapor;latent+=n.latent-input[i].latent;
        }
        require(mass==0. && fabs(energy)<1e-7 && fabs(chemical)<1e-8 && fabs(latent)<1e-8 &&
            fabs(oxygen)<1e-13 && fabs(vapor)<1e-13,"bounded solve loses extensive inventory");
    }
    std::cout<<"near-pure species: 32 heterogeneous graphs, blocks="<<grid_blocks<<", bounded and conservative\n";
}
__global__ void frozen_residual(int count,int gas_count,const int* offsets,const int* ids,
    const kg::Edge* edges,bg::Proposal* flux,kg::Node* nodes,bg::MixSolveState* work,
    bg::Control* control,int* error,battery::GasModel model,kg::Settings settings,
    battery::GasFilmView films,double dt,int perturb) {
    __shared__ double shared[128];const int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid==0 && perturb>=0)work[perturb].x[0]+=.001;
    cooperative_groups::this_grid().sync();
    const double r=bg::mix_true_residual(cooperative_groups::this_grid(),tid,blockDim.x*gridDim.x,
        count,gas_count,work,offsets,ids,edges,flux,films,dt/settings.cv,model,shared,control);
    if(tid==0)control->mixing_residual=r;
    if(perturb>=0) {if(tid==0)*error=r>1.?0:1;return;}
    const bool bounded=bg::mix_stage_bounded_flux(cooperative_groups::this_grid(),tid,blockDim.x*gridDim.x,
        count,gas_count,offsets[gas_count]/2,work,offsets,ids,edges,nodes,flux,settings,films,true,dt,shared,control);
    if(tid==0)*error=r<=1. && bounded?0:1;
}
void frozen_matrix(const nlohmann::json& json,bool admissible=true) {
    int gas_count=json.at("gas_count"),count=json.at("rows").size();double dt=json.at("step_s");
    kg::Settings settings;settings.cv=json.at("cv_j_kg_k");battery::GasModel model;
    model.implicit_temperature_tolerance_k=json.at("temperature_tolerance_k");
    model.implicit_tracer_relative_tolerance=json.at("tracer_relative_tolerance");
    std::vector<bg::MixSolveState> values(count);std::vector<kg::Node> input(gas_count);
    int perturb=0;
    for(int i=0;i<count;++i) {
        auto& v=values[i];const auto& row=json["rows"][i];v.mobile=row["mobile"];v.thermal=row["thermal"];
        for(int f=0;f<5;++f){v.x[f]=row["x"][f];v.origin[f]=row["origin"][f];}
        if(v.thermal>values[perturb].thermal)perturb=i;
        if(i<gas_count){auto& n=input[i];n.mass=v.mobile;n.solid_capacity=(v.thermal-v.mobile)*settings.cv;
            n.energy=v.thermal*settings.cv*v.origin[0]*1000.;n.chemical=v.mobile*1e6*v.origin[1];
            n.oxidizer=v.mobile*v.origin[2];n.vapor=v.mobile*v.origin[3];n.latent=v.mobile*1e6*v.origin[4];}
    }
    std::vector<kg::Edge> edge;std::vector<bg::Proposal> proposals;
    std::vector<std::vector<int>> adjacency(gas_count),gas_films(gas_count),solid_films(count-gas_count);
    for(const auto& row:json["gas_edges"]){int a=row[0],b=row[1],e=edge.size();edge.push_back({a,b,1.,1.,1.});
        bg::Proposal p;p.mixing.mass_each_way=row[2];proposals.push_back(p);adjacency[a].push_back(e);adjacency[b].push_back(e);}
    std::vector<battery::GasThermalPatch> patch;std::vector<double> conductance;
    for(const auto& row:json["film_edges"]){int a=row[0],b=row[1],j=patch.size();battery::GasThermalPatch p;
        p.compartment=a;p.volume=b-gas_count;patch.push_back(p);conductance.push_back(double(row[2])*settings.cv/dt);
        gas_films[a].push_back(j);solid_films[b-gas_count].push_back(j);}
    auto flatten=[](const auto& rows){std::pair<std::vector<int>,std::vector<int>> r{{0},{}};
        for(const auto& row:rows){r.second.insert(r.second.end(),row.begin(),row.end());r.first.push_back(r.second.size());}return r;};
    auto [offset,ids]=flatten(adjacency);auto [go,gi]=flatten(gas_films);auto [so,si]=flatten(solid_films);
    Buffer<bg::MixSolveState> work(count);Buffer<kg::Node> nodes(gas_count);nodes.put(input);
    Buffer<kg::Edge> edges(edge.size());edges.put(edge);Buffer<bg::Proposal> flux(proposals.size());
    Buffer<int> starts(offset.size()),incidence(ids.size()),gstart(go.size()),gids(gi.size()),sstart(so.size()),sids(si.size());
    starts.put(offset);incidence.put(ids);gstart.put(go);gids.put(gi);sstart.put(so);sids.put(si);
    Buffer<battery::GasThermalPatch> patches(patch.size());patches.put(patch);
    Buffer<double> weights(conductance.size());weights.put(conductance);
    Buffer<bg::Control> control(1);Buffer<int> error(1);
    battery::GasFilmView films{};films.patches=patches.p;films.offsets=gstart.p;films.ids=gids.p;
    films.solid_offsets=sstart.p;films.solid_ids=sids.p;films.conductance=weights.p;
    for(int blocks:{1,2})for(bool bad:{false,true}) {
        work.put(values);flux.put(proposals);control.put({{}});error.put({0});int change=bad?perturb:-1;
        void* args[]={&count,&gas_count,&starts.p,&incidence.p,&edges.p,&flux.p,&nodes.p,&work.p,
            &control.p,&error.p,&model,&settings,&films,&dt,&change};
        check(cudaLaunchCooperativeKernel((void*)frozen_residual,blocks,128,args));check(cudaDeviceSynchronize());
        auto result=control.get()[0];std::cout<<"frozen mixing: blocks="<<blocks<<" perturbed="<<bad
            <<" residual="<<result.mixing_residual<<" raw="<<result.mixing_raw_residual<<'\n';
        require(error.get()[0]==((bad || admissible)?0:1),bad?
            "roundoff hides a resolvable temperature error":"frozen composition admissibility differs");
    }
}
void subnormal_edge_flux() {
    // Reduced form of the exhaust failure: an empty chemical inventory with
    // a subnormal Krylov potential. The OLD gather transfers ~1e-319 J out of
    // that empty node and rejects an otherwise converged solve (error 26).
    nlohmann::json j={{"gas_count",2},{"step_s",.1},{"cv_j_kg_k",1000.},
        {"temperature_tolerance_k",.02},{"tracer_relative_tolerance",.001},
        {"rows",nlohmann::json::array()}, {"gas_edges",{{0,1,1e-5}}},
        {"film_edges",nlohmann::json::array()}};
    for(int i=0;i<2;++i)j["rows"].push_back({{"mobile",1.},{"thermal",1.},
        {"origin",{.3,0.,.232,0.,0.}}, {"x",{.3,i==0?1e-320:0.,.232,0.,0.}}});
    frozen_matrix(j);
    j["rows"][0]["x"][1]=1e-100;frozen_matrix(j,false);
    for(double result:{-1e-250,-1.,double(NAN),-double(INFINITY),0.,1.})
        require(bg::mixing_underflow_factor(result,0.,1.)==1.,"real error or valid inventory was silently changed");
    require(bg::mixing_underflow_factor(-1e-320,0.,1e-320)==0.,"empty donor still exports a subnormal tracer");
    // A normal outgoing flux can legitimately need subnormal incoming flux.
    // Never zero the incoming edge independently (the previous fix did).
    j["gas_count"]=3;j["rows"]=nlohmann::json::array();
    for(double q:{8e-308,4e-308,0.})j["rows"].push_back({{"mobile",1.},{"thermal",1.},
        {"origin",{.3,q*1e-6,.232,0.,0.}}, {"x",{.3,q,.232,0.,0.}}});
    j["gas_edges"]={{0,1,2.5e-7},{1,2,1.125e-6}};frozen_matrix(j);
    std::cout<<"subnormal edge budgets: bounded, paired; normal errors and incoming flux retained\n";
}
int main(int argc,char** argv){try{opened_boundary_response();const auto a=throughflow_film(1),b=throughflow_film(2);require(std::abs(a-b)<1e-9,"CUDA tile changes joint heat");
    nearly_pure_species(1);nearly_pure_species(2);subnormal_edge_flux();
    for(int i=1;i<argc;++i){nlohmann::json json;std::ifstream file(argv[i]);file>>json;frozen_matrix(json);}}
catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
