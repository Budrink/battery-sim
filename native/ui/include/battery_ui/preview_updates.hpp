#pragma once
#include <nlohmann/json.hpp>

namespace battery::ui {
// UI-only throttling. Coalesce continuous edits without rebuilding CUDA every
// frame; retry an invalid draft only when it changes, or on an explicit retry.
class PreviewUpdates {
  public:
    bool observe(const nlohmann::json& draft, double now) {
        if (draft == observed_)
            return false;
        observed_ = draft;
        if (!pending_)
            due_ = attempted_ ? now + .15 : now;
        pending_ = true;
        return true;
    }
    bool ready(double now) const {
        return pending_ && now >= due_;
    }
    void attempted() {
        pending_ = false;
        attempted_ = true;
    }
    void completed(const nlohmann::json& draft) {
        observed_ = draft;
        attempted();
    }
    void reset() {
        observed_ = nullptr;
        pending_ = false;
        attempted_ = false;
    }

  private:
    nlohmann::json observed_;
    double due_ = 0;
    bool pending_ = false, attempted_ = false;
};

inline bool appearance_only_change(nlohmann::json a, nlohmann::json b) {
    for (auto* doc : {&a, &b}) {
        if (doc->contains("metadata")) {
            (*doc)["metadata"].erase("visual_materials");
            (*doc)["metadata"].erase("visual_objects");
            if ((*doc)["metadata"].empty())
                doc->erase("metadata");
        }
    }
    return a == b;
}
} // namespace battery::ui
