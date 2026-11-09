#pragma once
#include "battery_native/simulation/simulation.hpp"
#include <nlohmann/json_fwd.hpp>
#include <array>
#include <string>
#include <vector>

namespace battery {
// Read-only presentation state. Playback never uploads these values to a solver.
struct VisualFrame {
    double time_s = 0;
    std::vector<CellState> cells;
    std::vector<InterfaceState> interfaces;
    ProcessDiagnostics process;
    std::vector<std::array<float, 4>> samples;
    const ThermalState* device_samples = nullptr;
    // Missing recorded channels are unknown, NOT measured zero flow.
    std::vector<bool> known_channels;
    // Committed controller state, not inferred from temperature or zero power.
    // Empty on older recordings: cutoff is unknown, not guessed.
    std::vector<HeatingProtocolState> heaters;
};
inline VisualFrame capture_visual_frame(const Simulation& simulation) {
    VisualFrame frame;
    frame.time_s = simulation.time_s();
    frame.cells = simulation.cells();
    frame.interfaces = simulation.interfaces();
    frame.process = simulation.process_diagnostics();
    frame.device_samples = simulation.cuda_field();
    frame.heaters = simulation.heating_protocol_states();
    return frame;
}
struct VisualCylinder {
    Vec3 center;
    double radius_m = 0, height_m = 0;
};
struct VisualBox {
    std::array<double, 3> low{}, high{};
};
struct GasVisualRegion {
    int compartment = -1;
    Vec3 center, size;
    // Convex XY polygon; optional exclusion for the solid battery inside it.
    std::vector<std::array<float, 2>> polygon;
    Vec3 excluded_center, excluded_size;
    std::vector<VisualCylinder> excluded_cylinders;
    // Display-only clipping. Full compartment/gradient remains unchanged.
    bool case_clipped = false;
    std::vector<VisualBox> case_clips;
};
void clip_gas_to_cases(GasVisualRegion&, const std::vector<VisualBox>&);
enum class VisualMaterialKind { Unspecified, Insulation, ThermalConductor, Enclosure, Heater };
struct CaseOpeningVisual {
    std::string id, case_id;
    int wall = -1, axis = 2, sign = 1;
    bool circle = true;
    Vec3 center;
    std::array<double, 2> size{};
};
struct VisualGeometry {
    // Per-solid mounting: 0 = plate/legacy, 1 = lateral sleeve, 2 = end disk.
    std::vector<int> heater_shapes;
    std::vector<CaseOpeningVisual> case_openings;
    std::vector<GasVisualRegion> gas;
    std::size_t unresolved_gas_regions = 0;
    // Presentation-only proxies. Never pass these to a solver/thermal mesher.
    std::vector<CellConfig> solid_proxies;
    std::vector<bool> hidden_planar_shells;
    std::vector<VisualMaterialKind> solid_materials;
};
VisualGeometry load_visual_geometry(const std::string& scenario_path, const Scenario&);
VisualGeometry visual_geometry_from_json(const nlohmann::json&, const Scenario&);
Vec3 project_visual_shell_point(const Scenario&, const VisualGeometry&, int shell, Vec3 point);
struct VisualShellPoint { Vec3 position, normal; };
// Same face coordinates/rupture field for all shapes; only the surface mapping
// changes. Apply the display offset after projection to avoid depth fighting.
VisualShellPoint map_visual_shell_point(const Scenario&, const VisualGeometry&, int shell,
                                        Vec3 point, Vec3 normal, double offset_m = 0);

class VisualReplay {
  public:
    explicit VisualReplay(const std::string& manifest);
    const std::string& scenario_path() const {
        return scenario_path_;
    }
    const std::vector<double>& times() const {
        return times_;
    }
    const std::vector<EventRecord>& events() const {
        return events_;
    }
    const std::string& source_label() const {
        return source_label_;
    }
    VisualFrame sample(double time, const Scenario&, const ThermalMesh&) const;

  private:
    std::string scenario_path_, source_label_;
    std::vector<std::string> objects_, channels_, gas_;
    std::vector<double> times_;
    std::vector<EventRecord> events_;
    std::vector<float> frames_;
    std::size_t stride_ = 0;
    std::size_t channel_fields_ = 8, gas_fields_ = 3;
};
} // namespace battery
