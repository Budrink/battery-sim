#pragma once
// Explicit diagnostic suffix continuation. Only unresolved HOT gas response
// laws may differ; geometry, cold references and all solid/chemical physics stay.
inline void check_gas_response_restart(const J& original,const J& trial,bool transport_probe=false) {
    J a=restart_identity(original,"",1.,0.),b=restart_identity(trial,"",1.,0.);
    // An explicitly declared numerical coarse floor is not a material refit.
    a["simulation"].erase("minimum_dt_s");b["simulation"].erase("minimum_dt_s");
    for(const auto* key:{"surface_connectivity","mixing_connectivity","mixing_integrator","free_gas_response_only"}) {
        a["gas_network"].erase(key);b["gas_network"].erase(key);
    }
    auto reference=[](const J& law){return law.at("background_conductance_w_per_k").template get<double>()+
        law.at("active_conductance_w_per_k").template get<double>()/
        (1.+std::exp(-law.value("geometry_reference_logit",0.)));};
    auto& ca=a["gas_network"]["thermal_contacts"];auto& cb=b["gas_network"]["thermal_contacts"];
    if(ca.size()!=cb.size())throw std::runtime_error("Gas-response restart changes contact topology");
    for(std::size_t i=0;i<ca.size();++i){
        const double ga=reference(ca[i].at("connectivity")),gb=reference(cb[i].at("connectivity"));
        if(std::abs(ga-gb)>1e-12*std::max(1.,std::abs(ga)))
            throw std::runtime_error("Gas-response restart changes cold surface reference");
        if(ca[i].at("port_id").template get<std::string>().rfind("chamber-wall:",0)==0 && ca[i]!=cb[i])
            throw std::runtime_error("Gas-response restart changes chamber wall");
        ca[i].erase("connectivity");cb[i].erase("connectivity");
        ca[i].erase("gas_surface_connectivity");cb[i].erase("gas_surface_connectivity");
        ca[i].erase("solid_surface_connectivity");cb[i].erase("solid_surface_connectivity");
    }
    if(transport_probe){
        auto& aa=a["gas_network"]["channels"];auto& bb=b["gas_network"]["channels"];
        if(aa.size()!=bb.size())throw std::runtime_error("Transport probe changes graph topology");
        for(std::size_t i=0;i<aa.size();++i)if(aa[i]!=bb[i]){
            if(aa[i].value("kind",std::string{})!="passage" || bb[i].value("kind",std::string{})!="passage")
                throw std::runtime_error("Transport probe may not change a material barrier");
            const double scale=bb[i].at("area_m2").get<double>()/aa[i].at("area_m2").get<double>();
            const double length=aa[i].at("mixing_length_m").get<double>()/bb[i].at("mixing_length_m").get<double>();
            if(!(scale>=1.-1e-12 && scale<=1000.*(1.+1e-12) && std::abs(scale-length)<1e-10*scale) ||
                bb[i].at("discharge_coefficient").get<double>()!=1.)
                throw std::runtime_error("Invalid explicitly declared passage saturation probe");
            for(const auto* k:{"area_m2","mixing_length_m","discharge_coefficient"}){aa[i].erase(k);bb[i].erase(k);}
        }
    }
    if(a!=b)throw std::runtime_error("Gas-response restart changes frozen geometry/solid/chemistry/transport physics");
}
