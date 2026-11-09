#include "contract_internal.hpp"
#include <cmath>

namespace battery::contract_detail {
void append_instances(const J& a, const J& n, const ModelCatalog& models, J& out,
                      std::set<std::string>& cells, Instances& instances) {
    for (const auto& b : a.at("batteries")) {
        object(b, "battery instance",
               {"id", "model_id", "model_revision", "pose", "initial_state", "gas_compartment"},
               {"id", "model_id", "model_revision", "pose", "initial_state", "gas_compartment"});
        const auto bid = id(b.at("id"), "battery/id");
        if (!cells.insert(bid).second)
            fail(bid, "duplicate instance ID");
        const auto key =
            std::make_pair(id(b.at("model_id"), "model_id"), id(b.at("model_revision"), "model_revision"));
        if (!models.contains(key))
            fail(bid, "unknown model id/revision");
        const auto& m = models.at(key);
        if (!n.at("batteries").contains(bid))
            fail(bid, "missing numerical settings");
        const auto& mesh = n.at("batteries").at(bid);
        object(mesh, "numerics/" + bid,
               {"divisions", "shell_surface_divisions", "internal_heat_transfer", "geometry_approximation",
                "conversion", "electrical"},
               {"divisions", "internal_heat_transfer"});
        grid(mesh.at("divisions"), bid + "/divisions");
        J geometry = m.at("geometry");
        if (mesh.contains("geometry_approximation")) {
            const auto& approx = mesh.at("geometry_approximation");
            object(approx, "geometry_approximation", {"geometry", "reason"}, {"geometry", "reason"});
            text(approx.at("reason"), "geometry_approximation/reason");
            if (std::abs(volume(approx.at("geometry")) / volume(geometry) - 1.) > 1e-6)
                fail(bid, "geometry approximation must preserve contents volume (1e-6 relative)");
            geometry = approx.at("geometry");
        }
        if (geometry.at("kind") == "cylinder")
            fail(bid, "native cylinder mesh unavailable; declare a volume-preserving geometry_approximation");
        const auto& state = b.at("initial_state");
        object(
            state, "initial_state",
            {"temperature_k", "shell_temperature_k", "damage_fraction", "aging_fraction",
             "precursor_fraction", "charged_fraction"},
            {"temperature_k", "damage_fraction", "aging_fraction", "precursor_fraction", "charged_fraction"});
        number(state.at("temperature_k"), "temperature_k", 1e-12, 1e9);
        for (const auto* field :
             {"damage_fraction", "aging_fraction", "precursor_fraction", "charged_fraction"})
            number(state.at(field), field, 0, 1);
        if (state.at("precursor_fraction").get<double>() + state.at("charged_fraction").get<double>() > 1.)
            fail(bid, "precursor + charged fractions exceed one");
        object(b.at("pose"), "pose", {"position_m", "orientation_xyzw"}, {"position_m", "orientation_xyzw"});
        const auto& pos = b.at("pose").at("position_m");
        const auto& rot = b.at("pose").at("orientation_xyzw");
        if (!pos.is_array() || pos.size() != 3 || !rot.is_array() || rot.size() != 4)
            fail(bid, "invalid pose dimensions");
        for (const auto& x : pos)
            number(x, "position_m", -1e9, 1e9);
        double norm = 0;
        for (const auto& x : rot) {
            const double v = number(x, "orientation_xyzw", -1, 1);
            norm += v * v;
        }
        if (std::abs(norm - 1.) > 1e-9)
            fail(bid, "orientation quaternion must be normalized");
        J cell = m.at("contents");
        cell.update(J{{"id", bid},
                      {"role", "battery"},
                      {"pose", b.at("pose")},
                      {"geometry", geometry},
                      {"runaway", {{"energy_j", 0}, {"duration_s", 1}}}});
        cell["thermal"].update(J{{"initial_temperature_k", state.at("temperature_k")},
                                 {"divisions", mesh.at("divisions")},
                                 {"internal_heat_transfer", mesh.at("internal_heat_transfer")}});
        if (!cell.contains("degradation"))
            cell["degradation"] = J::object();
        cell["degradation"].update(J{{"initial_damage", state.at("damage_fraction")},
                                     {"aging_fraction", state.at("aging_fraction")}});
        cell["conversion"].update(J{{"initial_precursor_fraction", state.at("precursor_fraction")},
                                    {"initial_charged_fraction", state.at("charged_fraction")},
                                    {"gas_compartment", text(b.at("gas_compartment"), "gas_compartment")}});
        for (const auto* block : {"conversion", "electrical"})
            if (mesh.contains(block)) {
                if (!cell.contains(block))
                    fail(bid, std::string("numerics specified for absent ") + block);
                object(mesh.at(block), "numerics/" + std::string(block),
                       std::string(block) == "conversion"
                           ? Keys{"max_fraction_per_step", "max_temperature_step_k"}
                           : Keys{"max_temperature_step_k"});
                cell[block].update(mesh.at(block));
            }
        // An explicit zero drive keeps the material resistance available without energizing it.
        if (cell.contains("electrical"))
            cell["electrical"]["current_a"] = 0.;
        out["cells"].push_back(cell);
        if (!m.at("shell").is_null()) {
            if (!state.contains("shell_temperature_k"))
                fail(bid, "shell requires explicit initial temperature");
            number(state.at("shell_temperature_k"), "shell_temperature_k", 1e-12, 1e9);
            J shell = m.at("shell");
            shell.update(J{{"contents_id", bid}, {"initial_temperature_k", state.at("shell_temperature_k")}});
            if (mesh.contains("shell_surface_divisions")) {
                grid(mesh.at("shell_surface_divisions"), "shell_surface_divisions");
                shell["surface_divisions"] = mesh.at("shell_surface_divisions");
            }
            out["shells"].push_back(shell);
        } else if (state.contains("shell_temperature_k") || mesh.contains("shell_surface_divisions"))
            fail(bid, "shell state/mesh without shell");
        instances[bid] = J{{"model", m}, {"gas_compartment", b.at("gas_compartment")}};
    }
    for (const auto& [key, value] : n.at("batteries").items())
        if (!instances.contains(key))
            fail(key, "numerics for unknown battery");
    if (a.contains("passive_cells"))
        array(a.at("passive_cells"), "passive_cells");
    for (const auto& passive : a.value("passive_cells", J::array())) {
        const auto pid = id(passive.at("id"), "passive/id");
        if (passive.value("role", "") != "passive" || !cells.insert(pid).second)
            fail(pid, "expected unique passive cell");
        out["cells"].push_back(passive);
    }
}

void apply_electrical_drives(const J& c, const std::set<std::string>& cells, J& out) {
    std::set<std::string> driven;
    for (const auto& drive : c.at("electrical_drives")) {
        object(drive, "electrical_drive",
               {"cell_id", "current_a", "current_profile_a", "voltage_profile_v", "current_input"},
               {"cell_id"});
        const auto target = text(drive.at("cell_id"), "drive/cell_id");
        if (!cells.contains(target) || !driven.insert(target).second)
            fail(target, "unknown or duplicate electrical drive");
        if (drive.contains("current_a") == drive.contains("current_profile_a"))
            fail(target, "specify current_a OR current_profile_a");
        for (auto& cell : out["cells"])
            if (cell.at("id") == target) {
                if (!cell.contains("electrical"))
                    fail(target, "drive without electrical material law");
                auto& law = cell["electrical"];
                law.erase("current_a");
                law.erase("current_profile_a");
                for (const auto& [key, value] : drive.items())
                    if (key != "cell_id")
                        law[key] = value;
            }
    }
}

void connect_vents(const J& a, const Instances& instances, J& out) {
    auto& gas = out["gas_network"];
    if (!gas.is_object() || !gas.contains("compartments"))
        fail("gas_network", "explicit gas compartments required");
    array(gas.at("compartments"), "gas compartments");
    if (!gas.contains("channels"))
        gas["channels"] = J::array();
    array(gas.at("channels"), "gas channels");
    std::map<std::string, J> compartments;
    for (const auto& g : gas.at("compartments"))
        if (!compartments.emplace(text(g.at("id"), "gas/id"), g).second)
            fail("gas_network", "duplicate compartment");
    for (const auto& [bid, inst] : instances) {
        const auto gid = inst.at("gas_compartment").get<std::string>();
        if (!compartments.contains(gid) || compartments.at(gid).value("contents_id", "") != bid)
            fail(bid, "gas compartment must exist and belong to this battery");
    }
    std::set<std::pair<std::string, std::string>> connected;
    for (const auto& route : a.at("vent_connections")) {
        object(route, "vent_connection", {"id", "battery_id", "vent_id", "to"},
               {"id", "battery_id", "vent_id", "to"});
        const auto bid = text(route.at("battery_id"), "vent_connection/battery_id");
        const auto vid = text(route.at("vent_id"), "vent_connection/vent_id");
        if (!instances.contains(bid) || !connected.emplace(bid, vid).second)
            fail(bid, "unknown battery or duplicate vent connection");
        const auto& inst = instances.at(bid);
        bool found = false;
        for (const auto& vent : inst.at("model").at("vents"))
            if (vent.at("id") == vid) {
                J channel = vent.at("channel");
                channel.update(J{{"id", text(route.at("id"), "channel/id")},
                                 {"from", inst.at("gas_compartment")},
                                 {"to", route.at("to")},
                                 {"port_id", bid + ":" + vent.at("port_suffix").get<std::string>()}});
                gas["channels"].push_back(channel);
                found = true;
            }
        if (!found)
            fail(vid, "unknown vent");
    }
    for (const auto& [bid, inst] : instances)
        for (const auto& vent : inst.at("model").at("vents"))
            if (!connected.contains({bid, vent.at("id").get<std::string>()}))
                fail(bid, "every declared vent must have an explicit destination");
    std::set<std::string> channel_ids;
    for (const auto& channel : gas.at("channels"))
        if (!channel_ids.insert(text(channel.at("id"), "channel/id")).second)
            fail("gas_network/channels", "duplicate channel ID");
}
} // namespace battery::contract_detail
