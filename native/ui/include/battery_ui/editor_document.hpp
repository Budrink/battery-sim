#pragma once
#include <nlohmann/json.hpp>
#include <filesystem>
#include <string>
#include <vector>
namespace battery::ui {
class EditorDocument {
  public:
    nlohmann::json draft, sidecar;
    std::filesystem::path path;
    bool dirty = false;
    void load(const std::filesystem::path&);
    void save(const std::filesystem::path&);
    nlohmann::json compile() const;
    void record(const nlohmann::json& before);
    void undo();
    void redo();

  private:
    std::vector<nlohmann::json> undo_, redo_;
};
} // namespace battery::ui
