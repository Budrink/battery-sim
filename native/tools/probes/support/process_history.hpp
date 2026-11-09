#pragma once
// Read-only serialization filter. All volumes/channels still participate in
// physics, pressure observers and global conservation checks.
class ProcessHistorySelection {
    J settings;
public:
    explicit ProcessHistorySelection(J value):settings(std::move(value)) {
        if(!settings.is_object())throw std::runtime_error("Process history selection must be an object");
        for(const auto& item:settings.items()) {
            if(item.key()!="gas_prefixes" && item.key()!="channel_prefixes")
                throw std::runtime_error("Unknown process history selector");
            if(!item.value().is_array())throw std::runtime_error("History prefixes must be arrays");
            for(const auto& prefix:item.value())
                if(!prefix.is_string() || prefix.get<std::string>().empty())
                    throw std::runtime_error("History prefixes must be nonempty strings");
        }
    }
    bool includes(const char* key,const std::string& id) const {
        if(!settings.contains(key))return true;
        for(const auto& prefix:settings.at(key))
            if(id.rfind(prefix.get<std::string>(),0)==0)return true;
        return false;
    }
};
