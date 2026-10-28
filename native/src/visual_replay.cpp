#include "battery_native/visual_frame.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <bit>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <unordered_map>

namespace battery {
using Json=nlohmann::json;
VisualReplay::VisualReplay(const std::string& path) {
    std::ifstream input(path);const auto j=Json::parse(input);
    if((j.at("format")!="BATTVIEW1" && j.at("format")!="BATTVIEW2") || std::endian::native!=std::endian::little)
        throw std::runtime_error("Unsupported visual recording format");
    scenario_path_=j.at("scenario_path");source_label_=j.at("source_label");
    times_=j.at("time_s").get<std::vector<double>>();
    objects_=j.at("objects").get<std::vector<std::string>>();
    channels_=j.at("channels").get<std::vector<std::string>>();
    gas_=j.at("gas").get<std::vector<std::string>>();
    if(times_.size()<2)throw std::runtime_error("Empty visual recording");
    for(std::size_t i=0;i<times_.size();++i)
        if(!std::isfinite(times_[i]) || (i && times_[i]<=times_[i-1]))
            throw std::runtime_error("Non-monotone visual recording");
    for(const auto& e:j.at("events"))events_.push_back({e.at("time_s"),e.at("kind"),e.at("id")});
    std::stable_sort(events_.begin(),events_.end(),[](auto& a,auto& b){return a.time_s<b.time_s;});
    channel_fields_=j.value("channel_fields",8);gas_fields_=j.value("gas_fields",3);
    if((channel_fields_!=8 && channel_fields_!=12) || (gas_fields_!=3 && gas_fields_!=6))throw std::runtime_error("Unknown recording fields");
    stride_=objects_.size()*4+channels_.size()*channel_fields_+gas_.size()*gas_fields_;
    const auto file=std::filesystem::path(path).parent_path()/"frames.bin";
    const auto expected=times_.size()*stride_*sizeof(float);
    const auto actual=std::filesystem::file_size(file);
    // A running recorder publishes an atomic manifest after flushing frames.
    // Read only its committed prefix; never guess times for appended frames.
    if(actual<expected || (!j.value("partial",false) && actual!=expected))
        throw std::runtime_error("Truncated visual recording");
    if(j.value("partial",false))source_label_+=" / PARTIAL capture";
    frames_.resize(times_.size()*stride_);std::ifstream binary(file,std::ios::binary);
    binary.read(reinterpret_cast<char*>(frames_.data()),frames_.size()*sizeof(float));
    if(!binary || !std::all_of(frames_.begin(),frames_.end(),[](float x){return std::isfinite(x);}))
        throw std::runtime_error("Invalid visual frame data");
}
VisualFrame VisualReplay::sample(double time,const Scenario& s,const ThermalMesh& mesh) const {
    VisualFrame f;f.time_s=std::clamp(time,times_.front(),times_.back());
    const auto high=std::clamp<std::size_t>(std::upper_bound(times_.begin(),times_.end(),f.time_s)-times_.begin(),1,times_.size()-1);
    const auto low=high-1;const float t=(f.time_s-times_[low])/(times_[high]-times_[low]);
    auto value=[&](std::size_t offset){return std::lerp(frames_[low*stride_+offset],frames_[high*stride_+offset],t);};
    std::unordered_map<std::string,std::size_t> object_map,channel_map,gas_map;
    for(std::size_t i=0;i<objects_.size();++i)object_map[objects_[i]]=i;
    for(std::size_t i=0;i<channels_.size();++i)channel_map[channels_[i]]=i;
    for(std::size_t i=0;i<gas_.size();++i)gas_map[gas_[i]]=i;
    f.cells.resize(s.cells.size());f.interfaces.resize(s.interfaces.size());
    for(std::size_t i=0;i<s.cells.size();++i) {
        const auto it=object_map.find(s.cells[i].id);
        if(it==object_map.end())throw std::runtime_error("Unrecorded solid: "+s.cells[i].id);
        auto& c=f.cells[i];const auto o=it->second*4;
        c.temperature_k=c.min_temperature_k=c.max_temperature_k=value(o);
        c.damage=std::clamp<double>(value(o+1),0,1);c.chemical_power_w=value(o+2);
    }
    std::unordered_map<std::string,double> opened;
    for(const auto& e:events_) {
        // A discrete event is NEVER interpolated backwards before its real time.
        if(e.time_s>f.time_s)break;
        if(e.kind=="shell_rupture") {opened[e.entity_id]=e.time_s;continue;}
        const auto it=std::find_if(s.cells.begin(),s.cells.end(),[&](auto& c){return c.id==e.entity_id;});
        if(it==s.cells.end())continue;
        auto& c=f.cells[it-s.cells.begin()];
        if(e.kind=="runaway_complete") {c.mode=CellMode::Spent;c.spent_fraction=1;}
        else {c.runaway_started_s=e.time_s;c.mode=CellMode::Runaway;}
    }
    f.samples.resize(mesh.volumes.size());
    for(std::size_t i=0;i<mesh.volumes.size();++i) {
        const auto& c=f.cells[mesh.volumes[i].object];
        f.samples[i]={float(c.temperature_k),float(c.damage),c.mode==CellMode::Spent?1.f:0.f,0};
    }
    f.process.channels.resize(s.gas_channels.size());
    f.known_channels.assign(s.gas_channels.size(),false);
    for(std::size_t i=0;i<s.gas_channels.size();++i) {
        auto it=channel_map.find(s.gas_channels[i].id);if(it==channel_map.end())continue;
        f.known_channels[i]=true;
        auto& c=f.process.channels[i];const auto o=objects_.size()*4+it->second*channel_fields_;
        if(channel_fields_==12) {
            c.max_thermal_damage=value(o+10);c.max_bulk_damage=value(o+11);
        }
        if(s.gas_channels[i].rupture && !opened.contains(s.gas_channels[i].id))continue;
        c.open=true;c.opened_s=opened.contains(s.gas_channels[i].id)?opened.at(s.gas_channels[i].id):0.;
        c.opening_fraction=value(o);c.effective_area_m2=value(o+1);
        c.mass_flow_kg_s=value(o+2);c.aperture_radius_m=value(o+3);
        for(int k=0;k<4;++k)c.aperture_strips_m[k]=value(o+4+k);
        if(channel_fields_==12) {
            c.transferred_kg=value(o+8);c.mixed_mass_kg=value(o+9);
            // A sparse recording stores the instantaneous rate only at sample
            // endpoints. Use the accepted interval mass / interval duration
            // for tracers, consistent with their interpolated cumulative mass.
            // Otherwise an endpoint sign reversal could make the glyph say
            // "left" while its recorded parcel actually moves to the right.
            c.mass_flow_kg_s=(double(frames_[high*stride_+o+8])-frames_[low*stride_+o+8])/
                (times_[high]-times_[low]);
        }
    }
    f.process.gas.resize(s.gas_compartments.size());
    for(std::size_t i=0;i<s.gas_compartments.size();++i) {
        auto it=gas_map.find(s.gas_compartments[i].id);
        if(it==gas_map.end())throw std::runtime_error("Unrecorded gas region");
        const auto o=objects_.size()*4+channels_.size()*channel_fields_+it->second*gas_fields_;
        auto& g=f.process.gas[i];g.temperature_k=value(o);g.chemical_j=value(o+1);g.oxidizer_kg=value(o+2);
        if(gas_fields_==6) {g.mass_kg=value(o+3);g.pressure_pa=value(o+4);g.released_energy_j=value(o+5);}
    }
    return f;
}
}
