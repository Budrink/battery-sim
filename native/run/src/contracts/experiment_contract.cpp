#include "battery_native/contracts/experiment_contract.hpp"
#include "battery_native/scenario_document.hpp"
#include "contract_internal.hpp"

namespace battery {
using namespace contract_detail;

namespace {
J compile_normalized(const J& e, const J& network = nullptr) {
    object(e, "document",
           {"format", "version", "id", "name", "provenance", "battery_models", "assembly", "conditions",
            "numerics", "measurements", "metadata"},
           {"format", "version", "id", "name", "provenance", "battery_models", "assembly", "conditions",
            "numerics", "measurements"});
    if (e.at("format") != "battery-sim.experiment" || e.at("version") != 1 ||
        !e.at("version").is_number_integer())
        fail("document", "unsupported format/version");
    id(e.at("id"), "experiment/id");
    text(e.at("name"), "experiment/name");
    provenance(e.at("provenance"));
    array(e.at("battery_models"), "battery_models");
    ModelCatalog models;
    for (const auto& m : e.at("battery_models")) {
        model(m);
        if (!models
                 .emplace(std::make_pair(m.at("id").get<std::string>(), m.at("revision").get<std::string>()),
                          m)
                 .second)
            fail("battery_models", "duplicate id/revision");
    }
    const auto& a = e.at("assembly");
    object(a, "assembly",
           {"batteries", "passive_cells", "ports", "interfaces", "gas_network", "gas_meshes", "vent_connections"},
           {"batteries", "ports", "interfaces", "gas_network", "vent_connections"});
    const auto& n = e.at("numerics");
    object(n, "numerics", {"simulation", "batteries"}, {"simulation", "batteries"});
    object(n.at("simulation"), "numerics/simulation",
           {"duration_s", "max_dt_s", "minimum_dt_s", "step_interval_s", "output_interval_s",
            "stability_factor"},
           {"duration_s", "max_dt_s", "output_interval_s"});
    for (const auto* key : {"duration_s", "max_dt_s", "output_interval_s"})
        number(n.at("simulation").at(key), key, 1e-12, 1e15);
    if (!n.at("batteries").is_object())
        fail("numerics/batteries", "expected per-instance settings");
    const auto& c = e.at("conditions");
    object(c, "conditions", {"boundaries", "heating_protocols", "electrical_drives"},
           {"boundaries", "heating_protocols", "electrical_drives"});
    for (const auto* key : {"batteries", "ports", "interfaces", "vent_connections"})
        array(a.at(key), key);
    for (const auto* key : {"boundaries", "heating_protocols", "electrical_drives"})
        array(c.at(key), key);
    if (a.at("batteries").empty())
        fail("assembly", "at least one battery required");
    J out{{"name", e.at("name")},
          {"source", "experiment-contract-v1:" + e.at("id").get<std::string>()},
          {"simulation", n.at("simulation")},
          {"cells", J::array()},
          {"shells", J::array()},
          {"ports", a.at("ports")},
          {"interfaces", a.at("interfaces")},
          {"gas_network", a.at("gas_network")},
          {"boundaries", c.at("boundaries")},
          {"heating_protocols", c.at("heating_protocols")},
          {"primary_failures", J::array()}};
    std::set<std::string> cells;
    Instances instances;
    append_instances(a, n, models, out, cells, instances);
    apply_electrical_drives(c, cells, out);
    connect_vents(a, instances, out);
    resolve_vent_patches(out);
    measurements(e.at("measurements"), cells);
    if (e.contains("metadata") && !e.at("metadata").is_object())
        fail("metadata", "expected object");
    out["metadata"] = e.value("metadata", J::object());
    if (out["metadata"].contains("experiment_contract"))
        fail("metadata", "experiment_contract is reserved");
    // Entire immutable source snapshot: reusable model revisions, assumptions and measurement mapping.
    // Observation records are metadata, NEVER imposed boundary conditions or ignition triggers.
    out["metadata"]["experiment_contract"] = e;
    if (!network.is_null())
        out["electrical_circuit"] = network;
    compile_gas_meshes(out, a.value("gas_meshes", J::object()));
    (void)scenario_from_json(out); // existing native topology/material parser, without CUDA or simulation
    return out;
}
} // namespace
J compile_experiment(const J& e) {
    finite_tree(e);
    if (e.is_object() && e.value("format", "") == "battery-sim.experiment" && e.contains("version") &&
        e.at("version").is_number_integer() && e.at("version") == 2) {
        auto lowered = lower_experiment_v2(e);
        J network;
        if (e.at("assembly").at("electrical_circuit").value("mode", "") == "solved_dc")
            network = compile_dc_circuit(e.at("assembly"), e.at("conditions"));
        auto out = compile_normalized(lowered, network);
        out["source"] = "experiment-contract-v2:" + e.at("id").get<std::string>();
        out["metadata"]["experiment_contract"] = e;
        return out;
    }
    return compile_normalized(e);
}
} // namespace battery
