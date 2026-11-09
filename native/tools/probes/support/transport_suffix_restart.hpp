#pragma once
#include <map>
#include <set>

// Explicit CONDITIONED transport experiment, not a physical warmup replay.
// Same gas/solid volumes and complete rupture state: restore without remapping,
// reinitializing fuel, closing holes, or changing temperatures/pressure.
inline void check_transport_suffix_restart(const J& original,const J& trial,bool numerical_scheme_change=false,
    bool closed_exhaust_diagnostic=false) {
    J a=restart_identity(original,"",1.,0.),b=restart_identity(trial,"",1.,0.);
    auto& ga=a.at("gas_network");auto& gb=b.at("gas_network");
    if(numerical_scheme_change){
        if(!gb.value("reactive_transit",false))throw std::runtime_error("Numerical trial must enable reactive transit");
        ga.erase("reactive_transit");gb.erase("reactive_transit");
    }
    if(ga.at("compartments")!=gb.at("compartments"))
        throw std::runtime_error("Conditioned transport suffix requires IDENTICAL gas volumes, initial inventory and reaction laws");
    auto& ea=ga.at("channels");auto& eb=gb.at("channels");
    if(ea.size()!=eb.size())throw std::runtime_error("Transport suffix changes channel topology");
    std::size_t cuts=0;
    for(std::size_t i=0;i<ea.size();++i)if(closed_exhaust_diagnostic &&
        ea[i].value("kind",std::string{})=="passage" &&
        ea[i].at("from").get<std::string>().rfind("top-",0)==0 &&
        (ea[i].at("to")=="hood" || ea[i].at("to").get<std::string>().rfind("plume-",0)==0)) {
        if(eb[i].at("area_m2").get<double>()!=0.)
            throw std::runtime_error("Closed-exhaust diagnostic must block EVERY module exit");
        ++cuts;
    }
    if(closed_exhaust_diagnostic && cuts==0)throw std::runtime_error("No explicit module exits to close");
    for(std::size_t i=0;i<ea.size();++i)if(ea[i]!=eb[i]) {
        if(ea[i].at("kind")!="passage" || eb[i].at("kind")!="passage")
            throw std::runtime_error("Transport suffix changes a pore or material barrier");
        const double ratio=eb[i].at("area_m2").get<double>()/ea[i].at("area_m2").get<double>();
        const bool exterior_cut=closed_exhaust_diagnostic && ratio==0. &&
            ea[i].at("from").get<std::string>().rfind("top-",0)==0 &&
            (ea[i].at("to")=="hood" || ea[i].at("to").get<std::string>().rfind("plume-",0)==0);
        if(!exterior_cut && !(ratio>=.05-1e-12 && ratio<=4.+1e-12))throw std::runtime_error("Transport suffix passage scale outside declared prior");
        ea[i].erase("area_m2");eb[i].erase("area_m2");
    }
    const double mix=gb.at("mixing_flow_fraction").get<double>();
    if(!std::isfinite(mix) || mix<.05 || mix>.9)throw std::runtime_error("Transport suffix mixing outside declared prior");
    ga.erase("mixing_flow_fraction");gb.erase("mixing_flow_fraction");
    auto& ca=ga.at("thermal_contacts");auto& cb=gb.at("thermal_contacts");
    auto passive=[](const J& contacts){J result=J::array();for(const auto& c:contacts)
        if(c.at("port_id").get<std::string>().rfind("B",0)!=0)result.push_back(c);return result;};
    if(passive(ca)!=passive(cb))throw std::runtime_error("Transport suffix changes passive/ABS surfaces");
    using Contacts=std::map<std::string,std::pair<double,double>>;
    auto collect=[&](const J& contacts,const J& references) {
        Contacts totals;std::map<std::string,J> permitted;
        for(const auto& c:references)permitted.emplace(c.at("port_id").get<std::string>(),c);
        for(const auto& c:contacts) {
            const auto port=c.at("port_id").get<std::string>();
            if(port.rfind("B",0)!=0)continue;
            const auto found=permitted.find(port);
            if(found==permitted.end())throw std::runtime_error("Transport suffix adds a previously unexposed surface");
            const auto& ref=found->second;
            if(port.rfind("B",0)!=0 || port.find(":case:")==std::string::npos) {
                if(c!=ref)throw std::runtime_error("Transport suffix changes a non-battery surface");
            }else {
                const auto gas=c.at("compartment").get<std::string>();
                const auto oldgas=ref.at("compartment").get<std::string>();
                if(gas!=oldgas) {
                    if(oldgas.rfind("body-",0)!=0 || gas.rfind("body-",0)!=0)
                        throw std::runtime_error("Transport suffix creates a gas/shell shortcut through a barrier");
                    const int own=std::stoi(oldgas.substr(5)),other=std::stoi(gas.substr(5));
                    bool neighbours=false;
                    for(const auto& pair:original.at("metadata").at("neighbour_pairs"))
                        if((pair[0]==own && pair[1]==other)||(pair[0]==other && pair[1]==own))neighbours=true;
                    if(!neighbours)throw std::runtime_error("Transport suffix exposes a non-neighbour gas volume");
                    const auto split=port.find(":case:")+6;
                    const int axis=port.at(split+1)-'x';const double sign=port.at(split)=='+'?1.:-1.;
                    if(axis<0 || axis>1)throw std::runtime_error("Only lateral gap exposure may change");
                    const auto& positions=original.at("metadata").at("approximate_positions_m");
                    const auto key=[](int i){return std::string("B")+(i<10?"0":"")+std::to_string(i);};
                    const auto& pa=positions.at(key(own));const auto& pb=positions.at(key(other));
                    double distance2=0.;for(int k=0;k<3;++k){double d=pb[k].get<double>()-pa[k].get<double>();distance2+=d*d;}
                    if(sign*(pb[axis].get<double>()-pa[axis].get<double>())<.5*std::sqrt(distance2)-1e-12)
                        throw std::runtime_error("Transport suffix gas volume is behind receiving surface");
                }
            }
            J shape=c,reference=ref;
            for(const auto* key:{"compartment","connectivity"}){shape.erase(key);reference.erase(key);}
            if(shape!=reference)throw std::runtime_error("Transport suffix changes surface state response or patch selection");
            const auto& l=c.at("connectivity");const auto& r=ref.at("connectivity");
            const double span=(l.at("active_conductance_w_per_k").get<double>()/l.at("background_conductance_w_per_k").get<double>())/
                (r.at("active_conductance_w_per_k").get<double>()/r.at("background_conductance_w_per_k").get<double>());
            if(!(span>=1.-1e-10 && span<=30.+1e-10))throw std::runtime_error("Transport suffix hot-contact span outside prior");
            J ls=l,rs=r;
            for(const auto* k:{"geometry_reference_logit","background_conductance_w_per_k","active_conductance_w_per_k",
                              "destroyed_background_conductance_w_per_k","destroyed_active_conductance_w_per_k"}){ls.erase(k);rs.erase(k);}
            if(ls!=rs)throw std::runtime_error("Transport suffix changes contact damage/constitutive law");
            const double cold=l.at("background_conductance_w_per_k").get<double>()+
                l.at("active_conductance_w_per_k").get<double>()/(1.+std::exp(-l.value("geometry_reference_logit",0.)));
            totals[port].first+=cold;
            totals[port].second+=l.at("destroyed_background_conductance_w_per_k").get<double>();
        }
        return totals;
    };
    const auto ta=collect(ca,ca),tb=collect(cb,ca);
    if(ta.size()!=tb.size())throw std::runtime_error("Transport suffix removes a surface");
    for(const auto& [port,values]:ta) {
        const auto& other=tb.at(port);
        if(std::abs(values.first-other.first)>1e-12 || std::abs(values.second-other.second)>1e-12)
            throw std::runtime_error("Transport suffix duplicates exposed area/cold conductance");
    }
    ga.erase("thermal_contacts");gb.erase("thermal_contacts");
    if(a!=b)throw std::runtime_error("Transport suffix changes frozen solids, chemistry, heater, shell or numerics");
}
