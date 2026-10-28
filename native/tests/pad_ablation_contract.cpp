#include "../tools/pad_ablation.hpp"
#include <iostream>
using J=nlohmann::json;
void require(bool value){if(!value)throw std::runtime_error("Pad ablation contract regression");}
int main(){try {
    J original={{"cells",J::array({{{"id","ISS1"},{"material_id","ISS"}},{{"id","C2"},{"material_id","battery"}}})},
        {"ports",J::array({{{"id","pad"},{"cell_id","ISS1"}},{{"id","cell"},{"cell_id","C2"}}})},
        {"interfaces",J::array({{{"port_a","pad"},{"port_b","cell"},{"law",{{"kind","directed_geometry"},
            {"background_conductance_w_per_k",1.},{"active_conductance_w_per_k",2.},
            {"destroyed_background_conductance_w_per_k",3.},{"destroyed_active_conductance_w_per_k",4.}}}}})},
        {"gas_network",{{"thermal_contacts",J::array()}}},{"boundaries",J::array()}};
    J target=original;target["metadata"]["pad_ablation"]["removed"]=J::array({"ISS1"});
    const J permission={{"diagnostic_pad_ablation",true}};
    const auto normalized=expected_pad_ablation(original,target,permission);
    require(normalized["cells"]==original["cells"] && normalized["interfaces"][0]["law"]["active_conductance_w_per_k"]==0.);
    bool rejected=false;try{expected_pad_ablation(original,target,J::object());}catch(const std::runtime_error&){rejected=true;}require(rejected);
    auto bad=target;bad["metadata"]["pad_ablation"]["removed"]=J::array({"C2"});rejected=false;
    try{expected_pad_ablation(original,bad,permission);}catch(const std::runtime_error&){rejected=true;}require(rejected);
    bad=target;bad["gas_network"]["thermal_contacts"].push_back({{"port_id","pad"}});rejected=false;
    try{expected_pad_ablation(original,bad,permission);}catch(const std::runtime_error&){rejected=true;}require(rejected);
    bad=target;bad["boundaries"].push_back({{"port_id","pad"}});rejected=false;
    try{expected_pad_ablation(original,bad,permission);}catch(const std::runtime_error&){rejected=true;}require(rejected);
    std::cout<<"pad ablation contract: constrained contacts and escrow isolation pass\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
