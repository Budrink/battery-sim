#include "battery_native/simulation.hpp"
#include <iostream>
#include <stdexcept>
using namespace battery;
Scenario fixture(double e1,double e2) {
    Scenario s;CellConfig c;c.id="B";c.dimensions_m={.02,.02,.02};
    c.heat_capacity_j_per_k=10.;c.initial_temperature_k=300.;
    auto& p=c.conversion;p.enabled=true;p.mass_kg=.004;
    p.reactant_cp=p.activated_cp=p.residue_cp=1000.;
    p.degradation_rate_per_s=.01;p.degradation_reference_k=400.;p.degradation_activation_j_per_mol=e1;
    p.reaction_rate_per_s=1.;p.reaction_reference_k=450.;p.reaction_activation_j_per_mol=e2;
    p.net_heat_j_per_kg=200000.;p.gas_yield=.2;p.gas_yield_max=.6;p.gas_chemical_fraction=.25;p.gas_compartment=0;
    s.cells={c};GasCompartment g;g.id="inside";g.owner_cell=0;g.volume_m3=1e-6;
    g.initial_temperature_k=300.;g.initial_mass_kg=1e-6;s.gas_compartments={g};return s;
}
void reject(const Scenario& s) {
    bool rejected=false;try{build_thermal_mesh(s);}catch(const std::exception&){rejected=true;}
    if(!rejected)throw std::runtime_error("Invalid kinetic input accepted");
}
int main(){try {
    for(auto pair:{std::pair<double,double>{160000.,110000.},{110000.,110000.},{40000.,136000.}})
        build_thermal_mesh(fixture(pair.first,pair.second));
    reject(fixture(-1.,110000.));reject(fixture(110000.,-1.));
    auto s=fixture(160000.,110000.);s.cells[0].conversion.precursor_yield=1.01;reject(s);
    s=fixture(160000.,110000.);s.cells[0].conversion.gas_yield=1.01;reject(s);
    s=fixture(160000.,110000.);s.cells[0].conversion.degradation_rate_per_s=-1.;reject(s);
    std::cout<<"Independent R1/R2 activation ordering accepted; physical input guards retained\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
