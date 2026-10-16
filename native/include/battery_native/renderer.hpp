#pragma once

#include "battery_native/model.hpp"
#include "battery_native/palette.hpp"
#include "battery_native/simulation.hpp"

#include <memory>
#include <vector>

namespace battery {

struct OrbitCamera {
    float yaw = -0.82f;
    float pitch = 0.34f;
    float distance = 1.75f;
};

struct VisualSettings {
    FieldLayer field_layer = FieldLayer::Temperature;
    float temperature_min_k = 298.15f;
    float temperature_max_k = 700.0f;
    float field_opacity = 0.82f;
    float shell_opacity = 0.11f;
    bool show_interfaces = true;
    bool show_ports = false;
    float cutaway_fraction = 1.0f;
};

class SceneRenderer {
public:
    SceneRenderer();
    ~SceneRenderer();
    SceneRenderer(const SceneRenderer&) = delete;
    SceneRenderer& operator=(const SceneRenderer&) = delete;

    bool init();
    unsigned int render(int width, int height, const Scenario& scenario,
                        const Simulation& simulation, const OrbitCamera& camera,
                        const VisualSettings& settings, int selected_cell,
                        const std::vector<int>& probed_cells);
    int pick_cell(float x_px, float y_px, int width, int height,
                  const Scenario& scenario, const OrbitCamera& camera) const;
    void destroy();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace battery
