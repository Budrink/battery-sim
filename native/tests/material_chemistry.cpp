#include "battery_native/conversion.hpp"
#include "battery_native/joule.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace battery;
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void close(double a,double b,double tolerance,const char* why){require(std::abs(a-b)<=tolerance,why);}
ConversionLaw law(double r,double b) {
    ConversionLaw p;p.enabled=true;p.mass_kg=1.;p.net_heat_j_per_kg=200000.;
    p.degradation_reference_k=p.reaction_reference_k=300.;p.degradation_rate_per_s=r;p.reaction_rate_per_s=b;
    p.degradation_activation_j_per_mol=p.reaction_activation_j_per_mol=0.;return p;
}
ThermalState initial() {
    ThermalState s;s.temperature_k=300.;s.reactant_kg=.6;s.activated_kg=.3;s.residue_kg=.1;
    s.healthy_charged_kg=.2;s.precursor_charge_energy_j=45000.;return s;
}
void charge_is_not_degradation() {
    auto p=law(0.,0.);ThermalState s;s.temperature_k=300.;s.reactant_kg=1.;
    p.precursor_yield=.01;GasModel g;
    const double old=conversion_chemical(p,s);
    const double heat=advance_conversion(p,g,s,1.,10.,1.,0.,.1);refresh_conversion(p,s,1.,1.);
    close(s.healthy_charged_kg,1-std::exp(-1.),1e-15,"Charging inventory is wrong");
    require(heat==0. && s.damage==0. && s.residue_kg==0.,"Charging creates degradation or heat outside I2R");
    close(charge_capacity_fraction(p,s,1.),1.,1e-15,"Charging reduces capacity");
    close(conversion_chemical(p,s)-old,s.electrical_chemical_input_j,1e-10,"Charge work is not stored");
    ElectricalLaw e;e.product_conductivity_ratio=.2;
    close(electrical_mixture_factor(e,p,s.reactant_kg,s.activated_kg,s.residue_kg,1.,s.healthy_charged_kg),1.,1e-15,"Charging raises base mixture resistance");
}
void degradation_ages_charged_material() {
    auto p=law(.1,0.);p.precursor_yield=.25;auto s=initial();const double energy=conversion_chemical(p,s);
    const double heat=advance_conversion(p,{},s,1.,2.,1.,0.);refresh_conversion(p,s,1.,1.);
    close(s.degraded_mass_kg,.8*(1-std::exp(-.2)),1e-15,"R1 fails to degrade healthy charged material");
    close(charge_capacity_fraction(p,s,1.),.8*std::exp(-.2),1e-15,"R1 capacity loss is incorrect");
    require(heat==0. && s.generated_gas_kg==0.,"R1 releases heat/gas");
    close(conversion_chemical(p,s)+s.sterilized_chemical_j,energy,1e-10,"R1 loses chemical ledger");
    ElectricalLaw e;e.product_conductivity_ratio=.2;
    auto before=initial();
    require(electrical_mixture_factor(e,p,s.reactant_kg,s.activated_kg,s.residue_kg,1.,s.healthy_charged_kg)<
        electrical_mixture_factor(e,p,before.reactant_kg,before.activated_kg,before.residue_kg,1.,before.healthy_charged_kg),"R1 does not reduce conductivity");
}
void exact_flow_and_ledgers() {
    for(double r:{0.,1e-12,.1,1.,100.})for(double c:{0.,1e-12,.1,1.,100.})
    for(double b:{0.,1e-12,.1,1.,100.})for(double phi:{0.,.27,1.}) {
        auto p=law(r,b);auto one=initial(),many=one;const double before=conversion_chemical(p,one);
        auto flow=advance_material_inventory(p,one,.7,1.,c,phi);
        double reacted=flow.reacted_kg,released=flow.released_j;
        one.residue_kg+=reacted;
        close(one.reactant_kg+one.activated_kg+one.residue_kg,1.,2e-14,"Exact mass ledger");
        close(conversion_chemical(p,one)+one.sterilized_chemical_j+released,
            before+one.electrical_chemical_input_j,2e-8,"Exact chemical/electrical ledger");
        require(one.healthy_charged_kg<=one.activated_kg+1e-14 && one.degraded_mass_kg<=.8+1e-14,"Inventory bounds");
        require(one.precursor_charge_energy_j+1e-9>=p.net_heat_j_per_kg*one.healthy_charged_kg,"Unpaid healthy charged stock");
        for(int i=0;i<100;++i) {
            const auto part=advance_material_inventory(p,many,.007,1.,c,phi);many.residue_kg+=part.reacted_kg;
        }
        for(auto pair:{std::pair{one.reactant_kg,many.reactant_kg},{one.activated_kg,many.activated_kg},
                       {one.healthy_charged_kg,many.healthy_charged_kg},{one.degraded_mass_kg,many.degraded_mass_kg},
                       {one.residue_kg,many.residue_kg}})close(pair.first,pair.second,2e-13,"Frozen-rate flow depends on dt partition");
        close(one.precursor_charge_energy_j,many.precursor_charge_energy_j,1e-7,"Stored electrical energy depends on dt partition");
    }
}
void r2_only_and_quenching() {
    auto p=law(0.,1.);p.gas_yield=.3;p.gas_chemical_fraction=.4;auto s=initial();
    advance_conversion(p,{},s,1.,1.,1.,0.);refresh_conversion(p,s,1.,1.);
    require(s.generated_gas_kg>0. && s.released_energy_j>0.,"R2 produces no heat/gas");
    require(s.damage==0. && s.degraded_mass_kg==0.,"R2 mislabeled as R1 degradation");
    p.reaction_activation_j_per_mol=150000.;p.reaction_reference_k=400.;
    require(completion_rate(p,300.)<completion_rate(p,400.)*1e-6,"Cooling does not suppress R2");
}
int main(){try {
    charge_is_not_degradation();degradation_ages_charged_material();exact_flow_and_ledgers();r2_only_and_quenching();
    std::cout<<"Separated charging/R1/R2: exact flow, neutral aging, capacity, resistance and ledgers pass\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
