#pragma once
#include "battery_native/simulation.hpp"
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
    std::vector<std::array<float,4>> samples;
    const ThermalState* device_samples = nullptr;
    // Missing recorded channels are unknown, NOT measured zero flow.
    std::vector<bool> known_channels;
};
inline VisualFrame capture_visual_frame(const Simulation& simulation) {
    VisualFrame frame;
    frame.time_s=simulation.time_s();frame.cells=simulation.cells();
    frame.interfaces=simulation.interfaces();frame.process=simulation.process_diagnostics();
    frame.device_samples=simulation.cuda_field();return frame;
}
struct VisualCylinder {
    Vec3 center;
    double radius_m=0, height_m=0;
};
struct GasVisualRegion {
    int compartment=-1;
    Vec3 center, size;
    // Convex XY polygon; optional exclusion for the solid battery inside it.
    std::vector<std::array<float,2>> polygon;
    Vec3 excluded_center, excluded_size;
    std::vector<VisualCylinder> excluded_cylinders;
};
struct VisualGeometry {
    std::vector<GasVisualRegion> gas;
    std::size_t unresolved_gas_regions=0;
    // Presentation-only proxies. Never pass these to a solver/thermal mesher.
    std::vector<CellConfig> solid_proxies;
    std::vector<bool> hidden_planar_shells;
};
VisualGeometry load_visual_geometry(const std::string& scenario_path,const Scenario&);
Vec3 project_visual_shell_point(const Scenario&,const VisualGeometry&,int shell,Vec3 point);

class VisualReplay {
public:
    explicit VisualReplay(const std::string& manifest);
    const std::string& scenario_path() const { return scenario_path_; }
    const std::vector<double>& times() const { return times_; }
    const std::vector<EventRecord>& events() const { return events_; }
    const std::string& source_label() const { return source_label_; }
    VisualFrame sample(double time,const Scenario&,const ThermalMesh&) const;
private:
    std::string scenario_path_,source_label_;
    std::vector<std::string> objects_,channels_,gas_;
    std::vector<double> times_;
    std::vector<EventRecord> events_;
    std::vector<float> frames_;
    std::size_t stride_=0;
    std::size_t channel_fields_=8,gas_fields_=3;
};
}
