#pragma once
// Read-only export: no extra physical steps, uploads or solver dependencies.
class GasFieldObserver {
    std::vector<std::pair<std::string,std::vector<std::size_t>>> groups;
    std::ofstream frames;
    std::size_t frame_count=0;
    std::string frame_path;
public:
    GasFieldObserver(const J& root,const J& request,const battery::Scenario& scenario) {
        const auto selection=root.value("gas_history_groups",J::object());
        if(!selection.is_object())throw std::runtime_error("Gas history groups must be an object");
        for(const auto& item:selection.items()) {
            if(!item.value().is_array() || item.value().empty())throw std::runtime_error("Empty gas group selectors");
            for(const auto& prefix:item.value())if(!prefix.is_string() || prefix.get<std::string>().empty())
                throw std::runtime_error("Invalid gas group prefix");
            std::vector<std::size_t> indices;
            for(std::size_t i=0;i<scenario.gas_compartments.size();++i) {
                const auto& id=scenario.gas_compartments[i].id;
                for(const auto& prefix:item.value())if(id.rfind(prefix.get<std::string>(),0)==0) {
                    indices.push_back(i);break;
                }
            }
            if(indices.empty())throw std::runtime_error("Unmatched gas history group");
            groups.emplace_back(item.key(),std::move(indices));
        }
        frame_path=request.value("gas_frames_path",std::string{});
        if(!frame_path.empty()) {
            // v1: native little-endian IEEE-754; 8-byte magic + u64 N;
            // each frame: f64 t, then N*(f32 Kelvin, f32 fuel J, f32 oxygen kg).
            const std::uint16_t endian=1;
            if(*reinterpret_cast<const unsigned char*>(&endian)!=1 || sizeof(float)!=4 || sizeof(double)!=8)
                throw std::runtime_error("Unsupported gas frame representation");
            frames.open(frame_path,std::ios::binary);
            const std::uint64_t count=scenario.gas_compartments.size();
            frames.write("BATTGAS1",8);frames.write(reinterpret_cast<const char*>(&count),sizeof(count));
            if(!frames)throw std::runtime_error("Cannot create gas field export");
        }
    }
    template<class GasStates>
    void sample(J& out,const GasStates& gas,double time) {
        for(const auto& group:groups) {
            double mass=0.,sensible=0.,chemical=0.,oxygen=0.,released=0.,tmin=1e300,tmax=0.,pmax=0.;
            for(auto i:group.second) {
                const auto& g=gas[i];mass+=g.mass_kg;sensible+=g.energy_j;chemical+=g.chemical_j;
                oxygen+=g.oxidizer_kg;released+=g.released_energy_j;
                tmin=std::min(tmin,g.temperature_k);tmax=std::max(tmax,g.temperature_k);pmax=std::max(pmax,g.pressure_pa);
            }
            auto& v=out["gas_groups"][group.first];
            v["mass_kg"].push_back(mass);v["sensible_energy_j"].push_back(sensible);
            v["chemical_j"].push_back(chemical);v["oxidizer_kg"].push_back(oxygen);
            v["released_energy_j"].push_back(released);v["minimum_temperature_k"].push_back(tmin);
            v["maximum_temperature_k"].push_back(tmax);v["maximum_pressure_pa"].push_back(pmax);
        }
        if(frames.is_open()) {
            frames.write(reinterpret_cast<const char*>(&time),sizeof(time));
            for(const auto& g:gas) {
                const float values[]={float(g.temperature_k),float(g.chemical_j),float(g.oxidizer_kg)};
                frames.write(reinterpret_cast<const char*>(values),sizeof(values));
            }
            frames.flush();if(!frames)throw std::runtime_error("Gas field export failed");++frame_count;
        }
    }
    void describe(J& out,const battery::Scenario& scenario) const {
        if(frame_path.empty())return;
        auto& v=out["gas_fields"];v={{"format","BATTGAS1"},{"path",frame_path},{"frames",frame_count},
            {"byte_order","little-endian"},{"fields",{"temperature_k","chemical_j","oxidizer_kg"}},
            {"node_ids",J::array()}};
        for(const auto& c:scenario.gas_compartments)v["node_ids"].push_back(c.id);
    }
};
