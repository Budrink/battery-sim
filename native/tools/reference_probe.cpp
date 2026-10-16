// Narada numerical-reference calibration driver. All time integration and heat
// transport run in the native CUDA core; stdin only supplies material parameters.
#include "battery_native/simulation.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <chrono>
#include <cmath>
using J=nlohmann::json;
int main(int argc,char**argv) {try {
 if(argc!=2)throw std::runtime_error("usage: battery_sim_reference_probe SCENARIO (JSON requests on stdin)");
 const auto base=battery::load_scenario(argv[1]);
 std::string line;
 while(std::getline(std::cin,line)) {try {
  J p=J::parse(line); auto s=base;
  s.settings.duration_s=p.value("duration",1400.);
  s.settings.max_dt_s=p.value("dt",.2);
  if(p.value("apply_fit",true)) {
  for(auto&c:s.cells) {
   if(c.is_battery){c.isothermal=p.value("isothermal",false);continue;}
   bool broad=c.material_id.find("broad")!=std::string::npos;
   if(p.value("coarse",true))c.thermal_divisions=broad?std::array<int,3>{2,2,2}:std::array<int,3>{4,2,2};
   double k=p.value(broad?"kb":"kx",broad?5.:50.);
   c.conductivity_w_per_m_k={.2,.2,.2};c.conductivity_w_per_m_k[broad?1:0]=k;
   c.heat_capacity_j_per_k*=p.value("capacity",1.);
   c.has_bulk_connectivity=p.value("bulk",false);
   c.bulk_connectivity.endpoint_damage_weight=p.value("endpoint",0.);
  }
  for(auto&e:s.interfaces){bool b=s.cells[e.cell_b].material_id.find("broad")!=std::string::npos;
   auto&l=e.law;double g=p.value(b?"gb":"gx",1.);
   l.background_conductance_w_per_k*=p.value(b?"coldb":"coldx",g);
   l.active_conductance_w_per_k*=g;
   l.destroyed_background_conductance_w_per_k*=p.value(b?"db":"dx",1.);
   l.destroyed_active_conductance_w_per_k*=p.value(b?"db":"dx",1.);
   l.transition_temperature_k=p.value(b?"tb":"tx",425.);
   l.damage_onset_k=p.value(b?"damageb":"damagex",850.);
   l.endpoint_damage_weight=p.value("endpoint",0.);
  }
  }
  battery::refine_environment_mesh(s,p.value("mesh_scale",1));
  battery::Simulation sim(s); auto start=std::chrono::steady_clock::now();
  double complete=-1;
  while(sim.advance([&]{if(complete<0&&sim.all_batteries_destroyed())complete=sim.time_s();return false;})){}
  J out={{"parameters",p},{"onsets",J::object()},{"complete_s",complete},{"steps",sim.step_count()},
  {"wall_s",std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()},
  {"volumes",sim.thermal_mesh().volumes.size()},{"links",sim.thermal_mesh().links.size()},
  {"final_time_s",sim.time_s()},{"max_temperature_k",sim.max_temperature_k()},
  {"residual_j",sim.energy_balance_residual_j()},{"released_j",sim.released_energy_j()}};
  for(size_t i=0;i<s.cells.size();++i)if(s.cells[i].is_battery&&sim.cells()[i].runaway_started_s>=0)
   out["onsets"][s.cells[i].id]=sim.cells()[i].runaway_started_s;
  std::cout<<out.dump()<<std::endl;
 }catch(const std::exception&e){std::cout<<J({{"error",e.what()}}).dump()<<std::endl;}}
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
