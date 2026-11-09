#pragma once
#include "battery_ui/editor_document.hpp"
#include <memory>
#include <optional>
namespace battery::ui {
class ExperimentEditor {
  public:
    ExperimentEditor();
    ~ExperimentEditor();
    bool open = false;
    void try_load(const std::string& path);
    void select_object(const std::string& id);
    void draw();
    std::optional<nlohmann::json> take_run();

  private:
    friend struct ExperimentEditorTestAccess;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace battery::ui
