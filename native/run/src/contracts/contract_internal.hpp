#pragma once
// Host-only compilation helpers. These never evolve the simulation or allocate CUDA state.
#include <nlohmann/json.hpp>
#include <initializer_list>
#include <map>
#include <set>
#include <string>
#include <utility>

namespace battery::contract_detail {
using J = nlohmann::json;
using Keys = std::initializer_list<const char*>;
using ModelCatalog = std::map<std::pair<std::string, std::string>, J>;
using Instances = std::map<std::string, J>;
[[noreturn]] void fail(const std::string& where, const std::string& reason);
void object(const J&, const std::string& where, Keys allowed, Keys required = {});
void array(const J&, const std::string& where);
std::string text(const J&, const std::string& where);
std::string id(const J&, const std::string& where);
double number(const J&, const std::string& where, double lo, double hi);
void finite_tree(const J&);
void grid(const J&, const std::string& where);
double volume(const J&);
void provenance(const J&);
void model(const J&);
void measurements(const J&, const std::set<std::string>& cells);
void append_instances(const J& assembly, const J& numerics, const ModelCatalog& models, J& output,
                      std::set<std::string>& cells, Instances& instances);
void apply_electrical_drives(const J& conditions, const std::set<std::string>& cells, J& output);
void connect_vents(const J& assembly, const Instances& instances, J& output);
void resolve_vent_patches(J& output);
void compile_gas_meshes(J& output, const J& meshes);
J lower_experiment_v2(const J& experiment);
J compile_dc_circuit(const J& assembly, const J& conditions);
void lower_local_ports(J& assembly);
void validate_electrical_circuit(const J& assembly, const J& conditions);
} // namespace battery::contract_detail
