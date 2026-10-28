#include "battery_native/radiation.hpp"
#include "battery_native/simulation.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace battery;
struct Case {double coefficient,a,b,resistance;};
struct PairCase {double coefficient,a,b,ra,rb;};
__global__ void evaluate_pairs(int n,const PairCase* input,double* result) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) {
        const auto c=input[i];result[i]=radiative_pair_conductance(c.coefficient,c.a,c.b,c.ra,c.rb);
    }
}
__global__ void evaluate(int n,const Case* input,double* result) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) {
        const auto c=input[i];result[i]=radiative_conductance(c.coefficient,c.a,c.b,c.resistance);
    }
}
void require(bool x,const char* message){if(!x)throw std::runtime_error(message);}
double independent(const Case& c) {
    if(c.coefficient==0.)return 0.;
    double low=std::min(c.a,c.b),high=std::max(c.a,c.b);
    for(int i=0;i<120;++i) {
        double x=(low+high)/2.,q=c.coefficient*(std::pow(x,4)-std::pow(c.b,4));
        if(x-c.a+c.resistance*q>0.)high=x;else low=x;
    }
    double x=(low+high)/2.;double g=c.coefficient*(x+c.b)*(x*x+c.b*c.b);
    return g/(1+c.resistance*g);
}
Scenario fixture(double emissivity,double initial=900.) {
    Scenario s;s.settings={10.,.01,1.,.25};
    CellConfig c;c.id="ABS";c.is_battery=false;c.dimensions_m={.0045,.01,.01};
    c.thermal_divisions={1,1,1};c.heat_capacity_j_per_k=1.;c.initial_temperature_k=initial;
    c.conductivity_w_per_m_k={.18,.18,.18};s.cells={c};
    s.ports.push_back({"outside","ABS",0,"contact","+x",.0001,{.00225,0,0},{1,0,0}});
    AmbientBoundary b;b.id="radiation";b.port_id="outside";b.temperature_k=300.;
    b.radiative=true;b.effective_emissivity=emissivity;s.boundaries={b};return s;
}
void finite_pairs() {
    std::vector<PairCase> input;
    for(double a:{100.,300.,900.,2000.,3999.})for(double b:{100.,300.,900.,3999.})
        for(double ra:{0.,.001,1.,1e3,1e6})for(double rb:{0.,.01,100.,1e6})
            input.push_back({5.670374419e-11,a,b,ra,rb});
    PairCase* device;double* result;cudaMalloc(&device,input.size()*sizeof(PairCase));cudaMalloc(&result,input.size()*sizeof(double));
    cudaMemcpy(device,input.data(),input.size()*sizeof(PairCase),cudaMemcpyHostToDevice);
    evaluate_pairs<<<(input.size()+127)/128,128>>>(input.size(),device,result);
    std::vector<double> gpu(input.size());
    require(cudaMemcpy(gpu.data(),result,gpu.size()*sizeof(double),cudaMemcpyDeviceToHost)==cudaSuccess,"CUDA finite radiative pair failed");
    cudaFree(device);cudaFree(result);
    for(std::size_t i=0;i<input.size();++i) {
        const auto c=input[i];long double q=0.,total=c.ra+c.rb;
        if(total>0. && c.a!=c.b) {
            long double lo=std::min(0.L,(c.a-c.b)/total),hi=std::max(0.L,(c.a-c.b)/total);
            for(int k=0;k<150;++k) {
                q=(lo+hi)/2;long double a=c.a-c.ra*q,b=c.b+c.rb*q;
                if(q-c.coefficient*(a*a*a*a-b*b*b*b)>0.)hi=q;else lo=q;
            }
        }
        const long double a=c.a-c.ra*q,b=c.b+c.rb*q,g=c.coefficient*(a+b)*(a*a+b*b);
        const double oracle=g/(1.+total*g),cpu=radiative_pair_conductance(c.coefficient,c.a,c.b,c.ra,c.rb);
        const double reverse=radiative_pair_conductance(c.coefficient,c.b,c.a,c.rb,c.ra);
        require(std::abs(gpu[i]-oracle)<1e-9*std::max(1e-9,oracle),"Pair differs from independent extended-precision bisection");
        require(std::abs(cpu-gpu[i])<1e-12*std::max(1.,oracle),"Pair CPU/CUDA mismatch");
        require(std::abs(reverse-cpu)<1e-10*std::max(1e-9,oracle),"Pair reversal violates reciprocity");
    }
    auto s=fixture(0.);s.boundaries.clear();s.settings.duration_s=2.;
    auto receiver=s.cells[0];receiver.id="receiver";receiver.position_m={.01,0,0};receiver.initial_temperature_k=300.;
    receiver.heat_capacity_j_per_k=2.;s.cells.push_back(receiver);
    s.ports.push_back({"receiver","receiver",1,"contact","-x",.0001,{.00775,0,0},{-1,0,0}});
    InterfaceConfig e;e.id="finite-pair";e.port_a="outside";e.port_b="receiver";e.cell_a=0;e.cell_b=1;
    e.radiative=true;e.effective_emissivity=.8;e.radiation_exchange_area_m2=.00008;s.interfaces={e};
    Simulation coupled(s);coupled.run_to_end();
    require(coupled.cells()[0].temperature_k<900. && coupled.cells()[1].temperature_k>300.,"Finite radiative pair did not transfer heat");
    require(std::abs(coupled.cells()[0].temperature_k+2.*coupled.cells()[1].temperature_k-1500.)<1e-8,"Finite-pair energy not conserved");
    require(coupled.boundary_energy_out_j()==0.,"Finite pair incorrectly heats an infinite reservoir");
    s.interfaces[0].effective_emissivity=0.;Simulation dark(s);dark.run_to_end();
    require(dark.cells()[0].temperature_k==900. && dark.cells()[1].temperature_k==300.,"Zero finite emissivity changes state");
    s.interfaces[0].radiation_exchange_area_m2=.00011;
    bool rejected=false;try {Simulation invalid(s);}catch(const std::runtime_error&){rejected=true;}
    require(rejected,"View factor above unity was accepted");
    std::cout<<input.size()<<" finite-pair CUDA/oracle, reciprocity, finite capacities and geometry guards OK\n";
}
int main(){try {
    finite_pairs();
    std::vector<Case> cases;
    for(double a:{100.,298.15,300.,600.,900.,2000.,3999.})
        for(double b:{100.,298.15,900.,2000.,3999.})
            for(double r:{0.,.001,1.,1000.,1e6})
                for(double e:{0.,.1,.9})cases.push_back({stefan_boltzmann_w_m2_k4*.001*e,a,b,r});
    Case* device;double* results;cudaMalloc(&device,cases.size()*sizeof(Case));cudaMalloc(&results,cases.size()*sizeof(double));
    cudaMemcpy(device,cases.data(),cases.size()*sizeof(Case),cudaMemcpyHostToDevice);
    evaluate<<<(cases.size()+127)/128,128>>>(cases.size(),device,results);
    std::vector<double> values(cases.size());
    require(cudaMemcpy(values.data(),results,values.size()*sizeof(double),cudaMemcpyDeviceToHost)==cudaSuccess,"CUDA radiative edge failed");
    cudaFree(device);cudaFree(results);
    for(std::size_t i=0;i<cases.size();++i) {
        const auto c=cases[i];double ref=independent(c),cpu=radiative_conductance(c.coefficient,c.a,c.b,c.resistance);
        if(std::abs(values[i]-ref)>1e-9*std::max(1.,ref) || std::abs(cpu-ref)>1e-9*std::max(1.,ref))
            throw std::runtime_error("Radiative surface does not match independent bisection");
        require(values[i]>=0. && (c.resistance==0. || values[i]*c.resistance<=1.+1e-12),"Radiation violated passivity/material resistance");
    }
    require(std::isnan(radiative_conductance(1e-8,4001.,300.,0.)),"Numerical domain must fail, not clamp heat");
    auto scene=fixture(.5);Simulation radiating(scene),dark(fixture(0.));radiating.run_to_end();dark.run_to_end();
    require(radiating.cells()[0].temperature_k<899. && radiating.cells()[0].temperature_k>300.,"Radiation failed to cool finite mass");
    require(dark.cells()[0].temperature_k==900. && dark.boundary_energy_out_j()==0.,"Zero emissivity changed the field");
    require(std::abs(radiating.energy_balance_residual_j())<1e-8,"Radiative boundary energy ledger failed");
    // Independent fine RK4 reference is a TEST oracle, not a production CPU solver.
    const Case c{stefan_boltzmann_w_m2_k4*.5*.0001,0.,300.,.0045/(2*.18*.0001)};
    auto rate=[&](double t){auto p=c;p.a=t;return -independent(p)*(t-300.);};
    double t=900.,dt=.001;
    for(int i=0;i<10000;++i){double a=rate(t),b=rate(t+dt*a/2),d=rate(t+dt*b/2),e=rate(t+dt*d);t+=dt*(a+2*b+2*d+e)/6;}
    require(std::abs(radiating.cells()[0].temperature_k-t)<.02,"Radiative cooling trajectory differs from RK4 reference");
    Simulation warming(fixture(.5,200.));warming.run_to_end();
    require(warming.cells()[0].temperature_k>200. && warming.cells()[0].temperature_k<300. && warming.boundary_energy_out_j()<0.,"Hot reservoir cannot warm the solid conservatively");
    require(std::abs(warming.energy_balance_residual_j())<1e-8,"Incoming radiative heat ledger failed");
    std::cout<<cases.size()<<" radiative constitutive CPU/CUDA cases + finite-wall cooling, heating, zero emissivity, domain guard and energy ledger OK\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
