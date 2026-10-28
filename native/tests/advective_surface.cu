// Full production CUDA transport against a dense simultaneous enthalpy oracle.
// The oracle consumes accepted carrier masses, not an alternate mass solver.
#include "../src/batched_gas.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace kg=battery::kawasaki;
namespace bg=battery::batched_gas;
void require(bool value,const char* why){if(!value)throw std::runtime_error(why);}
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>struct Buffer {
    T* p=nullptr;std::size_t n;
    explicit Buffer(std::size_t count):n(count){if(n)check(cudaMalloc(&p,n*sizeof(T)));}
    ~Buffer(){if(p)cudaFree(p);}
    void put(const std::vector<T>& v){require(v.size()==n,"Buffer size");if(n)check(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> get(){std::vector<T> v(n);if(n)check(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
void fixture(double dt,double g,double mixing,int blocks,bool reactive=false) {
    kg::Settings settings;const double cs=10.;
    auto gas=[&](double t,double v,double p){const double m=p*v/(settings.R*t);return kg::Node{m,m*settings.cv*t,v};};
    std::vector<kg::Node> input{gas(1000.,.01,2e5),gas(300.,1e-7,1.5e5),gas(300.,.01,1e5)};
    if(reactive)for(auto& n:input){n.chemical=n.mass*1e6;n.oxidizer=n.mass*.232;}
    const std::vector<kg::Edge> es{{0,1,1e-5,.7,.02},{1,2,1e-5,.7,.02}};
    Buffer<kg::Node> nodes(3),scratch(3);nodes.put(input);Buffer<kg::Edge> edges(2);edges.put(es);
    Buffer<kg::Clock> clocks(2);clocks.put({{}, {}});Buffer<bg::Proposal> proposals(2);Buffer<bg::Budget> budgets(3);
    Buffer<battery::GasChannelState> channels(2);battery::GasChannelState ch;ch.open=true;ch.effective_area_m2=1e-5;channels.put({ch,ch});
    Buffer<int> offsets(4),ids(4),error(1);offsets.put({0,1,3,4});ids.put({0,0,1,1});error.put({0});
    Buffer<bg::Control> control(1);control.put({{}});Buffer<bg::AverageState>a(3),b(3);Buffer<bg::MixSolveState> work(4);
    battery::GasThermalPatch patch;patch.volume=0;patch.compartment=1;patch.law=battery::reference_gas_connectivity(g);patch.geometry_weight=1.;
    Buffer<battery::GasThermalPatch> patches(1);patches.put({patch});
    Buffer<int> go(4),gi(1),so(2),si(1);go.put({0,0,1,1});gi.put({0});so.put({0,1});si.put({0});
    Buffer<double> capacity(1),heat(1),conductance(1);capacity.put({cs});heat.put({0.});conductance.put({g});
    battery::ThermalState solid;solid.temperature_k=300.;solid.stored_energy_j=cs*300.;Buffer<battery::ThermalState> solids(1);solids.put({solid});
    Buffer<battery::GasCombustionLaw> laws(3);battery::GasCombustionLaw law;law.rate_per_s=reactive?50.:0.;
    laws.put({law,law,law});Buffer<battery::GasFilmState> films(3);
    battery::GasFilmView view{patches.p,go.p,gi.p,solids.p,capacity.p,heat.p,conductance.p,so.p,si.p,1};
    battery::GasModel model;model.cuda_network_transport=model.averaged_network_transport=model.implicit_network_mixing=true;
    model.joint_gas_surface_mixing=model.coupled_gas_films=true;model.minimum_step_s=.005;
    model.mixing_diffusivity_m2_s=0.;model.mixing_flow_fraction=mixing;model.coupled_transport_interval_s=dt;
    model.averaged_pressure_iterations=2048;model.averaged_payload_iterations=128;
    model.reactive_transit=reactive;
    check(bg::launch(3,2,offsets.p,ids.p,edges.p,nodes.p,scratch.p,proposals.p,budgets.p,clocks.p,channels.p,
        settings,model,dt,error.p,control.p,128,blocks,{laws.p,false,films.p,view},a.p,b.p,work.p));
    check(cudaDeviceSynchronize());require(error.get()[0]==0,"Advective surface kernel failed");
    const auto output=nodes.get();const auto flux=proposals.get();const auto ledger=channels.get();const double q=heat.get()[0];
    long double matrix[4][4]{},rhs[4]{};
    for(int i=0;i<3;++i){matrix[i][i]=output[i].mass*settings.cv;
        rhs[i]=input[i].energy+output[i].reaction_released_j-input[i].reaction_released_j;}
    matrix[3][3]=cs+dt*g;rhs[3]=cs*300.;matrix[1][1]+=dt*g;matrix[1][3]=matrix[3][1]=-dt*g;
    for(int e=0;e<2;++e){const auto& p=flux[e];const int donor=p.donor,receiver=es[e].a==donor?es[e].b:es[e].a;
        const long double h=p.mass*(settings.cv+settings.R),w=p.mixing.mass_each_way*settings.cv;
        matrix[donor][donor]+=h;matrix[receiver][donor]-=h;
        matrix[donor][donor]+=w;matrix[receiver][receiver]+=w;matrix[donor][receiver]-=w;matrix[receiver][donor]-=w;
        const double sign=donor==es[e].a?1.:-1.;
        const double expected=sign*(p.energy+p.advective_energy_delta-p.mass*settings.cv*model.reference_temperature_k)+p.mixing.energy;
        require(std::abs(ledger[e].transferred_energy_j-expected)<1e-10,"Channel ledger lost enthalpy correction");}
    for(int i=0;i<4;++i)for(int j=i+1;j<4;++j){const long double f=matrix[j][i]/matrix[i][i];
        for(int k=i;k<4;++k)matrix[j][k]-=f*matrix[i][k];rhs[j]-=f*rhs[i];}
    for(int i=3;i>=0;--i){for(int j=i+1;j<4;++j)rhs[i]-=matrix[i][j]*rhs[j];rhs[i]/=matrix[i][i];}
    double worst=std::abs(300.+q/cs-double(rhs[3])),mass=0.,energy=q,oxygen=0.;
    for(int i=0;i<3;++i){worst=std::max(worst,std::abs(kg::temperature(output[i],settings)-double(rhs[i])));
        mass+=output[i].mass-input[i].mass;
        energy+=output[i].energy+output[i].chemical-input[i].energy-input[i].chemical;
        oxygen+=output[i].oxidizer-input[i].oxidizer+(output[i].reaction_released_j-input[i].reaction_released_j)/law.oxidizer_heat_j_per_kg;}
    require(worst<.03,"Gas/surface heat differs from simultaneous dense reference");
    require(std::abs(mass)<1e-12 && std::abs(energy)<1e-9,"Joint pressure/film heat not conserved");
    require(std::abs(oxygen)<1e-12,"Coupled enthalpy changes reaction oxygen budget");
    if(g>=1.)require(q>10.*input[1].mass*settings.cv*700.,"Film heat capped by resident gas");
    std::cout<<"Advective surface: dt="<<dt<<" G="<<g<<" mixing="<<mixing<<" blocks="<<blocks<<" reactive="<<reactive
        <<" error_K="<<worst<<" heat_J="<<q<<" iterations="<<control.get()[0].enthalpy_iterations<<'\n';
}
int main(){try{for(int blocks:{1,2})for(double dt:{.005,.01,.025})
    for(double g:{0.,1.,10.})for(double mixing:{0.,.72})fixture(dt,g,mixing,blocks);
    for(int blocks:{1,2})fixture(.01,10.,.72,blocks,true);
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
