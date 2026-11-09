#include "battery_native/data/visual_frame.hpp"
#include <algorithm>

namespace battery {
namespace {
bool intersection(const VisualBox& a, const VisualBox& b, VisualBox& result) {
    for (int k=0; k<3; ++k) {
        result.low[k]=std::max(a.low[k], b.low[k]);
        result.high[k]=std::min(a.high[k], b.high[k]);
        if (result.high[k]-result.low[k] <= 1e-12) return false;
    }
    return true;
}
// Disjoint pieces prevent double opacity in overlapping/nested cases.
void subtract(VisualBox box, const VisualBox& removed, std::vector<VisualBox>& out) {
    VisualBox overlap;
    if (!intersection(box, removed, overlap)) {out.push_back(box); return;}
    for (int k=0; k<3; ++k) {
        if (box.low[k]<overlap.low[k]) {
            auto part=box; part.high[k]=overlap.low[k]; out.push_back(part);
            box.low[k]=overlap.low[k];
        }
        if (box.high[k]>overlap.high[k]) {
            auto part=box; part.low[k]=overlap.high[k]; out.push_back(part);
            box.high[k]=overlap.high[k];
        }
    }
}
}
void clip_gas_to_cases(GasVisualRegion& gas, const std::vector<VisualBox>& cases) {
    gas.case_clipped=!cases.empty();
    gas.case_clips.clear();
    const VisualBox support{{gas.center.x-gas.size.x/2,gas.center.y-gas.size.y/2,gas.center.z-gas.size.z/2},
                            {gas.center.x+gas.size.x/2,gas.center.y+gas.size.y/2,gas.center.z+gas.size.z/2}};
    for (const auto& enclosure:cases) {
        VisualBox part;
        if (!intersection(support,enclosure,part)) continue;
        std::vector<VisualBox> pieces{part};
        for (const auto& accepted:gas.case_clips) {
            std::vector<VisualBox> remainder;
            for (const auto& p:pieces) subtract(p,accepted,remainder);
            pieces=std::move(remainder);
        }
        gas.case_clips.insert(gas.case_clips.end(),pieces.begin(),pieces.end());
    }
}
}
