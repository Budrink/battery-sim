#include "battery_native/simulation.hpp"
#include <nlohmann/json.hpp>
#include <array>
#include <chrono>
#include <iostream>
using J=nlohmann::json;
int main(int argc,char** argv){try {
    if(argc<2)throw std::runtime_error("usage: battery_sim_reaction_probe SCENARIO [--max-dt S] [--fraction F] [--temperature-step K]");
    auto s=battery::load_scenario(argv[1]);
    for(int i=2;i<argc;i+=2) {
        if(i+1>=argc)throw std::runtime_error("Missing option value");const std::string key=argv[i];const double value=std::stod(argv[i+1]);
        if(key=="--max-dt")s.settings.max_dt_s=value;
        else if(key=="--fraction") {s.gas_model.max_fraction_per_step=value;for(auto& c:s.cells) {c.reaction_kinetics.max_fraction_per_step=value;c.conversion.max_fraction_per_step=value;}}
        else if(key=="--temperature-step") {s.gas_model.max_temperature_step_k=value;for(auto& c:s.cells) {c.reaction_kinetics.max_temperature_rise_per_step_k=value;c.conversion.max_temperature_step_k=value;}}
        else throw std::runtime_error("Unknown option: "+key);
    }
    battery::Simulation sim(s);const auto start=std::chrono::steady_clock::now();
    J out={{"name",s.name},{"calibration_status",s.calibration_status},{"samples",J::array()}};
    double worst_thermal=0,worst_chemical=0,worst_mass=0;
    auto sample=[&] {
        J row={{"time_s",sim.time_s()},{"objects",J::array()}};const auto& field=sim.thermal_states();const auto& mesh=sim.thermal_mesh();
        for(std::size_t i=0;i<s.cells.size();++i)if(s.cells[i].is_battery || !s.gas_compartments.empty()) {
            const auto& c=sim.cells()[i];std::array<double,3> lo{1e300,1e300,1e300},hi{0,0,0};
            for(int j=mesh.object_offsets[i];j<mesh.object_offsets[i+1];++j)for(int axis=0;axis<3;++axis) {
                lo[axis]=std::min(lo[axis],field[j].conductivity_factor[axis]);hi[axis]=std::max(hi[axis],field[j].conductivity_factor[axis]);
            }
            row["objects"].push_back({{"id",s.cells[i].id},{"mode",battery::name(c.mode)},{"temperature_k",c.temperature_k},
                {"min_temperature_k",c.min_temperature_k},{"max_temperature_k",c.max_temperature_k},{"damage",c.damage},
                {"remaining_j",c.chemical_energy_remaining_j},{"released_j",c.released_energy_j},{"joule_energy_j",c.joule_energy_j},{"vent_heat_j",c.vent_heat_out_j},
                {"ejected_chemical_j",c.ejected_chemical_energy_j},{"chemical_power_w",c.chemical_power_w},
                {"structure_min_xyz",lo},{"structure_max_xyz",hi},{"heat_capacity_j_per_k",c.heat_capacity_j_per_k}});
            if(s.cells[i].conversion.enabled) {
                double mr=0,ma=0,mp=0,gross=0;
                for(int j=mesh.object_offsets[i];j<mesh.object_offsets[i+1];++j) {
                    const auto& v=field[j];mr+=v.reactant_kg;ma+=v.activated_kg;mp+=v.residue_kg;gross+=v.gross_reaction_energy_j;
                }
                row["objects"].back().update({{"reactant_kg",mr},{"activated_kg",ma},{"residue_kg",mp},{"gross_reaction_j",gross},
                    {"heat_capacity_j_per_k",c.heat_capacity_j_per_k}});
            }
        }
        if(!s.gas_compartments.empty()) {
            auto p=sim.process_diagnostics();row["gas"]=J::array();row["channels"]=J::array();
            for(std::size_t i=0;i<p.gas.size();++i) {const auto& g=p.gas[i];row["gas"].push_back({{"id",s.gas_compartments[i].id},{"mass_kg",g.mass_kg},{"energy_j",g.energy_j},{"chemical_j",g.chemical_j},{"temperature_k",g.temperature_k},{"pressure_pa",g.pressure_pa}});}
            for(std::size_t i=0;i<p.gas.size();++i) {
                const auto& g=p.gas[i];auto& result=row["gas"][i];
                result["free_volume_m3"]=g.free_volume_m3;
                result["heat_capacity_j_per_k"]=g.mass_kg*s.gas_model.cv_j_per_kg_k;
                result["released_energy_j"]=g.released_energy_j;result["oxidizer_kg"]=g.oxidizer_kg;
                const auto& config=s.gas_compartments[i];
                if(config.thermal_cell>=0) {
                    result["thermal_cell_id"]=s.cells[config.thermal_cell].id;
                    result["thermal_volume"]=config.thermal_volume;
                    result["material_temperature_k"]=field[mesh.object_offsets[config.thermal_cell]+config.thermal_volume].temperature_k;
                }
            }
            for(std::size_t i=0;i<p.channels.size();++i) {const auto& c=p.channels[i];row["channels"].push_back({{"id",s.gas_channels[i].id},{"open",c.open},{"opened_s",c.opened_s},{"mass_flow_kg_s",c.mass_flow_kg_s},{"transferred_kg",c.transferred_kg},{"transferred_energy_j",c.transferred_energy_j},{"transferred_chemical_j",c.transferred_chemical_j},{"opening_fraction",c.opening_fraction},{"effective_area_m2",c.effective_area_m2},{"max_bulk_damage",c.max_bulk_damage},{"max_seam_damage",c.max_seam_damage}});}
            row["escaped_mass_kg"]=p.escaped_mass_kg;row["escaped_energy_j"]=p.escaped_energy_j;row["escaped_chemical_j"]=p.escaped_chemical_j;
            row["fracture_work_j"]=p.fracture_work_j;
            for(std::size_t i=0;i<p.channels.size();++i) {
                const auto& channel=p.channels[i];
                row["channels"][i]["surface_energy_j"]=channel.surface_energy_j;
                row["channels"][i]["fracture_work_j"]=channel.fracture_work_j;
                row["channels"][i]["delivered_gas_energy_j"]=channel.transferred_energy_j-channel.surface_energy_j;
            }
        }
        out["samples"].push_back(row);
    };
    sample();for(double next=s.settings.output_interval_s;sim.time_s()<s.settings.duration_s-1e-10;next+=s.settings.output_interval_s) {
        const double end=std::min(next,s.settings.duration_s);
        while(sim.time_s()<end-1e-10) {sim.step(end-sim.time_s());
            worst_thermal=std::max(worst_thermal,std::abs(sim.energy_balance_residual_j()));
            worst_chemical=std::max(worst_chemical,std::abs(sim.chemical_balance_residual_j()));
            if(!s.gas_compartments.empty())worst_mass=std::max(worst_mass,std::abs(sim.process_diagnostics().mass_residual_kg));}
        sample();
    }
    out["events"]=J::array();for(auto& e:sim.events())out["events"].push_back({{"time_s",e.time_s},{"kind",e.kind},{"id",e.entity_id}});
    out["max_thermal_residual_j"]=worst_thermal;out["max_chemical_residual_j"]=worst_chemical;
    out["max_mass_residual_kg"]=worst_mass;
    out["steps"]=sim.step_count();out["wall_s"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::cout<<out.dump()<<'\n';
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
