#pragma once
#include "battery_ui/viewer_session.hpp"
#include "battery_ui/experiment_editor.hpp"
#include "battery_native/render/renderer.hpp"
namespace battery::ui {
// Presentation-only state: selection, camera, layers and cached texture.
// No solver ownership and no physical-step logic here.
struct ViewerState {
    VisualSettings visual;
    OrbitCamera orbit, initial_camera;
    bool inspector = true, auto_orbit = false;
    int selected = 0;
};
struct ViewerLayout {
    float side = 310, top = 100, bottom = 165;
};
class ViewerWindow {
  public:
    ViewerWindow(ViewerSession&, ExperimentEditor&, bool verify_gl = false);
    ~ViewerWindow();
    ViewerWindow(const ViewerWindow&) = delete;
    ViewerWindow& operator=(const ViewerWindow&) = delete;
    void draw(double wall_dt);
    void new_snapshot();

  private:
    void draw_header();
    void draw_inspector();
    void draw_scene();
    void draw_transport();
    ViewerSession& session;
    ExperimentEditor& editor;
    ViewerState state;
    ViewerLayout layout;
    SceneRenderer renderer;
    bool hidden;
    unsigned scene_texture = 0;
    int rendered_w = 0, rendered_h = 0, rendered_selected = -1;
    double rendered_time = -1;
    OrbitCamera rendered_camera;
    VisualSettings rendered_settings;
};
} // namespace battery::ui
