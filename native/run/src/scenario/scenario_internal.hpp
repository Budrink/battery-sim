#pragma once
// Parsing only: declaration order is cells -> shells -> thermal links -> gas and chemistry.
// Indexes are rebuilt after shell expansion; no CUDA or simulation runs here.
#include "battery_native/model/model.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
namespace battery::scenario_detail {
using json = nlohmann::json;
using CellIndex = std::unordered_map<std::string, std::size_t>;
Vec3 vec3(const json&, const std::string& where);
ShapeKind shape_kind(const std::string&);
TransitionKind transition_kind(const std::string&);
EdgePhase edge_phase(const std::string&);
ScalarTable parse_table(const json&, bool positive);
SurfaceConnectivity parse_surface(const json&);
ConnectivityLaw parse_law(const json&);
void parse_cells(const json&, Scenario&, CellIndex&);
void parse_shells(const json&, Scenario&, CellIndex&);
void parse_circuit(const json&, Scenario&, const CellIndex&);
CellIndex parse_thermal_links(const json&, Scenario&, const CellIndex&);
void parse_gas_network(const json&, Scenario&, const CellIndex&, const CellIndex& port_index);
} // namespace battery::scenario_detail
