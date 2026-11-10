#pragma once
#include "battery_ui/experiment_editor.hpp"
#include "battery_native/render/renderer.hpp"
#include "battery_ui/preview_updates.hpp"
#include "battery_ui/opening_selection.hpp"
#include "editor/forms/heater_controller_panel.hpp"
#include "editor/state/model_edits.hpp"
#include "editor/help/editor_sections.hpp"
#include <optional>

namespace battery::ui {
struct ExperimentEditor::Impl {
    // Document and uncommitted inputs. Never the running simulation's state.
    EditorDocument document;
    std::string file_path, message, selected;
    std::string heater_target;
    std::string model_selected_battery;
    std::size_t model_index = 0;
    ModelEdits model_edits;
    EditorSection section = EditorSection::Assembly;
    bool show_preview = true;
    std::string help_search;
    bool shell_dirty = false, enclosure_dirty = false, shape_dirty = false, opening_dirty = false;
    std::map<std::string, std::string> panel_errors;
    nlohmann::json panel_attempt;
    std::optional<nlohmann::json> pending;
    // File browser and import flow (views/ and commands/import.cpp).
    bool browser = false, import_requested = false;
    std::string import_error;
    std::filesystem::path browse_path = ".";
    // Panel drafts: editing a field must not change the compiled scenario yet.
    bool parallel = true, voltage_source = true;
    float source_value = 4.2f, emf = 3.65f;
    int face = 0;
    int shell_face = 5, opening_kind = 0;
    std::string enclosure_id = "case", enclosure_gas = "cavity", enclosure_loaded;
    nlohmann::json enclosure_properties, enclosure_source;
    bool place_case_opening = false;
    int case_opening_shape = 0;
    OpeningSelection case_opening_selection;
    std::string case_opening_face = "+z", case_opening_error;
    double case_opening_size_mm[2]{20, 20}, case_opening_center_mm[2]{0, 0};
    nlohmann::json shell_source, face_material, face_channel;
    std::string shell_owner, gas_destination;
    double opening_pressure_kpa = 80, area_rate = 1, fracture_energy = 200;
    double weak_factor = .4, weak_size_mm = 1, valve_center_mm[2]{0, 0};
    std::string shape_owner;
    nlohmann::json shape_source;
    int shape_kind = 0;
    double shape_dimensions_mm[3]{20, 20, 65};
    // Isolated initial-state preview (preview/). It never advances physics.
    SceneRenderer preview_renderer;
    bool renderer_ready = false;
    PreviewUpdates preview_updates;
    std::string preview_error;
    unsigned preview_texture = 0;
    int preview_width = 0, preview_height = 0, preview_selected = -2;
    OrbitCamera rendered_camera;
    std::unique_ptr<Simulation> preview;
    Scenario preview_scenario;
    ThermalMesh preview_mesh;
    VisualFrame preview_frame;
    VisualGeometry preview_geometry;
    OrbitCamera camera;
    VisualSettings visual;
    VisualSettings rendered_visual;
    nlohmann::json preview_source, preview_compiled;
    // Implementations are grouped by responsibility, not one large draw method.
    void refresh_preview();
    void update_preview();
    void draw_preview();
    void draw_browser();
    void draw_reference();
    void commit_panel_edits();
    void draw_assembly();
    void delete_selected();
    bool draw_battery_shape();
    void draw_circuit();
    void draw_heating();
    void draw_models();
    void draw_gas_model();
    void draw_run_settings();
    void draw_connections();
    void draw_cooling();
    void draw_circuit_properties();
    void draw_shell();
    void draw_enclosure();
    void draw_case_openings();
    void place_opening_at(int wall, Vec3 point);
    void select_case_opening(const nlohmann::json&);
    void add_case_opening();
    bool commit_case_opening(const std::string& case_id, const std::string& face,
                             std::array<double, 2> center_m);
    void wire();
};
} // namespace battery::ui
