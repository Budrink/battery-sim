#include "battery_native/volatile_phase.hpp"
#include "battery_native/gas_combustion.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace battery;
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void close(double a,double b,double tol,const char* why){require(std::abs(a-b)<=tol,why);}
struct Pair {ThermalState s;GasState g;};
ConversionLaw law() {
    ConversionLaw c;c.enabled=true;c.mass_kg=.01;c.net_heat_j_per_kg=1e5;
    c.volatile_phase.initial_mass_kg=.002;c.volatile_phase.chemical_j_per_kg=2e6;return c;
}
Pair initial(double temperature=600.) {
    Pair a;a.s.reactant_kg=.01;a.s.liquid_kg=.002;a.s.temperature_k=temperature;
    a.s.stored_energy_j=100.*temperature;
    a.g.mass_kg=1e5*1e-6/(287.*temperature);a.g.free_volume_m3=1e-6;
    a.g.temperature_k=temperature;a.g.pressure_pa=1e5;
    a.g.energy_j=a.g.mass_kg*1000.*(temperature-298.15);return a;
}
double total(const Pair& a){return a.s.stored_energy_j+a.g.energy_j+a.g.latent_j;}
double chemical(const ConversionLaw& c,const Pair& a){return conversion_chemical(c,a.s)+a.g.chemical_j;}
void balances(const ConversionLaw& c,const Pair& a,const Pair& b) {
    close(total(a),total(b),2e-10,"Phase exchange loses energy");
    close(a.s.liquid_kg+a.g.mass_kg,b.s.liquid_kg+b.g.mass_kg,1e-17,"Phase exchange loses mass");
    close(chemical(c,a),chemical(c,b),1e-10,"Phase exchange loses chemical energy");
    require(b.s.liquid_kg>=0. && b.g.vapor_kg>=0. && b.g.mass_kg>0. && b.g.latent_j>=0.,"Negative phase inventory");
    require(b.s.temperature_k>0. && b.s.temperature_k==b.g.temperature_k,"Phase LTE invalid");
}
__global__ void phase(Pair* p,ConversionLaw c,double duration) {
    advance_volatile(c,p->s,p->g,100.,1.,{},duration);
}
Pair evolve(Pair a,const ConversionLaw& c,int steps,double duration=.1) {
    for(int j=0;j<steps;++j)advance_volatile(c,a.s,a.g,100.,1.,{},duration/steps);
    return a;
}
int main(){try {
    auto c=law();auto a=initial();const auto b=evolve(a,c,1);
    balances(c,a,b);require(b.s.liquid_kg<a.s.liquid_kg && b.s.temperature_k<a.s.temperature_k,"Hot liquid fails to evaporate/cool");
    require(b.g.chemical_j>0. && b.g.latent_j>0.,"Vapor owns no fuel/latent inventory");
    Pair* device=nullptr;require(cudaMalloc(&device,sizeof(Pair))==cudaSuccess,"CUDA allocation");
    cudaMemcpy(device,&a,sizeof(a),cudaMemcpyHostToDevice);phase<<<1,1>>>(device,c,.1);
    Pair gpu;require(cudaMemcpy(&gpu,device,sizeof(gpu),cudaMemcpyDeviceToHost)==cudaSuccess,"CUDA phase kernel");cudaFree(device);
    balances(c,a,gpu);close(gpu.s.temperature_k,b.s.temperature_k,1e-10,"CUDA/host phase disagreement");
    close(gpu.s.liquid_kg,b.s.liquid_kg,1e-15,"CUDA phase inventory disagreement");
    auto cooled=b;cooled.s.temperature_k=cooled.g.temperature_k=300.;
    cooled.s.stored_energy_j=100.*298.15+conversion_capacity(c,cooled.s,100.,1.)*(300.-298.15);
    cooled.g.energy_j=cooled.g.mass_kg*1000.*(300.-298.15);
    const auto condensed=evolve(cooled,c,10);balances(c,cooled,condensed);
    require(condensed.s.liquid_kg>cooled.s.liquid_kg && condensed.s.temperature_k>300.,"Condensation fails to return latent heat");
    const auto coarse=evolve(a,c,10),fine=evolve(a,c,20),ref=evolve(a,c,320);
    const double e0=std::abs(coarse.s.temperature_k-ref.s.temperature_k),e1=std::abs(fine.s.temperature_k-ref.s.temperature_k);
    require(e1<e0*.7,"Implicit phase exchange fails time convergence");
    // Fuel heat is already owned by the vapor; R3 cannot burn without oxygen.
    kawasaki::Settings settings;settings.vapor_fuel_heat_j_kg=c.volatile_phase.chemical_j_per_kg;
    kawasaki::Node n{b.g.mass_kg,b.g.mass_kg*1000.*600.,1e-4,0.,b.g.chemical_j,0.};
    n.vapor=b.g.vapor_kg;n.latent=b.g.latent_j;
    const auto fuel=n;GasCombustionLaw burn{10.,0.,13.1e6};
    require(burn_gas(n,burn,settings,.1),"R3 integration");require(n.chemical==fuel.chemical && n.vapor==fuel.vapor,"Oxygen-free vapor burned");
    n.oxidizer=.1*n.mass;const double energy=n.energy+n.chemical+n.latent;
    require(burn_gas(n,burn,settings,.1),"Vapor combustion");
    require(n.vapor<fuel.vapor && n.chemical<fuel.chemical,"Burned vapor remains condensable");
    close(n.energy+n.chemical+n.latent,energy,1e-10,"Vapor combustion creates energy");
    const auto empty=evolve(b,c,1000,100.);balances(c,b,empty);
    std::cout<<"volatile phase: finite inventory, heat/chemical/mass closure, condensation, CUDA agreement, refinement "<<e0<<" -> "<<e1<<" K pass\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
