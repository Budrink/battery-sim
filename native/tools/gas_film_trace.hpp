#pragma once
#include "battery_native/simulation.hpp"
#include "battery_native/gas_thermal_contact.hpp"
#include <nlohmann/json.hpp>
#include <cmath>
#include <map>
#include <set>

// Optional host-side observation only: sum ACTUAL accepted per-patch heat.
// Does not recompute heat from sparsely sampled temperatures, add an energy
// ledger to device state, alter the solver step or change checkpoint ABI.
class GasFilmTrace {
    struct Contact {int gas,object;double conductance=0.,capacity=0.;std::vector<int> patches;};
    bool enabled_;
    const battery::Scenario& scene_;
    const battery::ThermalMesh& mesh_;
    std::vector<double> cumulative_;
    std::vector<double> positive_,negative_;
    std::map<std::string,Contact> contacts_;
public:
    GasFilmTrace(bool enabled,const battery::Scenario& s,const battery::ThermalMesh& m):
        enabled_(enabled),scene_(s),mesh_(m),cumulative_(enabled?m.gas_patches.size():0),
        positive_(cumulative_.size()),negative_(cumulative_.size()) {
        if(!enabled)return;
        if(!s.gas_model.coupled_gas_films)throw std::runtime_error("Film trace requires coupled films");
        std::map<std::string,std::set<int>> volumes;
        for(std::size_t i=0;i<m.gas_patches.size();++i) {
            const auto& p=m.gas_patches[i];const int object=m.volumes[p.volume].object;
            const auto key=s.gas_compartments[p.compartment].id+" -> "+s.cells[object].id;
            auto& c=contacts_[key];c.gas=p.compartment;c.object=object;c.conductance+=p.conductance_w_per_k;
            c.patches.push_back(i);
            if(volumes[key].insert(p.volume).second)c.capacity+=m.volumes[p.volume].capacity_j_per_k;
        }
    }
    void collect(const battery::Simulation& sim) {
        if(!enabled_)return;
        const auto q=sim.last_gas_film_heat_j();
        if(q.size()!=cumulative_.size())throw std::runtime_error("Film trace patch count mismatch");
        for(std::size_t i=0;i<q.size();++i) {
            if(!std::isfinite(q[i]))throw std::runtime_error("Nonfinite accepted film heat");
            cumulative_[i]+=q[i];
            positive_[i]+=std::max(q[i],0.);negative_[i]+=std::min(q[i],0.);
        }
    }
    void sample(nlohmann::json& out,const std::vector<battery::ThermalState>& field,
        const std::vector<battery::GasState>& gas)const {
        if(!enabled_)return;
        for(const auto& [key,c]:contacts_) {
            double q=0.,power=0.,positive=0.,negative=0.,conductance=0.;
            for(int i:c.patches) {
                const auto& p=mesh_.gas_patches[i];q+=cumulative_[i];
                positive+=positive_[i];negative+=negative_[i];
                const auto& solid=field[p.volume];
                const double g=battery::gas_contact_conductance(p,gas[c.gas].temperature_k,
                    solid.temperature_k,solid.damage);
                conductance+=g;power+=g*(gas[c.gas].temperature_k-solid.temperature_k);
            }
            auto& row=out["gas_film_trace"][key];row["gas"]=scene_.gas_compartments[c.gas].id;
            row["solid"]=scene_.cells[c.object].id;
            row["reference_conductance_w_per_k"]=c.conductance;
            row["instantaneous_conductance_w_per_k"].push_back(conductance);
            row["receiving_reference_capacity_j_k"]=c.capacity;
            row["heat_to_solid_j"].push_back(q);row["instantaneous_power_w"].push_back(power);
            row["positive_step_heat_j"].push_back(positive);row["negative_step_heat_j"].push_back(negative);
        }
    }
};
