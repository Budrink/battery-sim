#pragma once
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>

// Explicit diagnostic permission: only incident pad contacts may be zeroed.
// No change to battery/material/chemistry/electrical properties is permitted.
inline nlohmann::json expected_pad_ablation(nlohmann::json original,
    const nlohmann::json& target,const nlohmann::json& request) {
    using J=nlohmann::json;
    const auto config=target.value("metadata",J::object()).value("pad_ablation",J());
    if(config.is_null()) {
        if(request.value("diagnostic_pad_ablation",false))throw std::runtime_error("Missing pad ablation manifest");
        return original;
    }
    if(!request.value("diagnostic_pad_ablation",false))throw std::runtime_error("Explicit pad ablation permission required");
    std::set<std::string> removed;
    for(const auto& value:config.at("removed")) {
        const auto name=value.get<std::string>();
        if(!removed.insert(name).second)throw std::runtime_error("Duplicate removed pad");
        bool found=false;
        for(const auto& c:original.at("cells"))if(c.at("id")==name &&
            (c.at("material_id")=="ISS" || c.at("material_id")=="TIM"))found=true;
        if(!found)throw std::runtime_error("Ablation can only remove ISS/TIM objects");
    }
    std::set<std::string> disabled_ports;
    for(const auto& p:original.at("ports"))if(removed.count(p.at("cell_id").get<std::string>()))
        disabled_ports.insert(p.at("id"));
    for(auto& edge:original.at("interfaces"))if(disabled_ports.count(edge.at("port_a")) || disabled_ports.count(edge.at("port_b"))) {
        if(edge.at("law").at("kind")!="directed_geometry")throw std::runtime_error("Unexpected ablated contact law");
        for(const auto* k:{"background_conductance_w_per_k","active_conductance_w_per_k",
            "destroyed_background_conductance_w_per_k","destroyed_active_conductance_w_per_k"})edge["law"][k]=0.;
    }
    // Escrows must have no gas film, boundary or extra heat source.
    std::set<std::string> all_disabled=disabled_ports;
    for(const auto& p:target.at("ports"))if(removed.count(p.at("cell_id").get<std::string>()))all_disabled.insert(p.at("id"));
    for(const auto& c:target.at("gas_network").at("thermal_contacts"))
        if(all_disabled.count(c.at("port_id")))throw std::runtime_error("Gas heats an ablated pad escrow");
    for(const auto& b:target.at("boundaries"))if(all_disabled.count(b.at("port_id")))
        throw std::runtime_error("Boundary heats an ablated pad escrow");
    return original;
}
