#include "contract_internal.hpp"

namespace battery::contract_detail {
J compile_dc_circuit(const J& assembly, const J& conditions) {
    const auto& c = assembly.at("electrical_circuit");
    object(c, "electrical_circuit", {"mode", "nodes", "branches", "reference_node"},
           {"mode", "nodes", "branches", "reference_node"});
    if (c.at("mode") != "solved_dc")
        fail("circuit", "expected solved_dc");
    array(c.at("nodes"), "nodes");
    array(c.at("branches"), "branches");
    array(conditions.at("source_drives"), "source_drives");
    array(conditions.at("electrical_drives"), "electrical_drives");
    std::set<std::string> batteries, connected, branch_ids;
    for (const auto& b : assembly.at("batteries"))
        batteries.insert(id(b.at("id"), "battery/id"));
    std::map<std::string, J> drives;
    for (const auto& d : conditions.at("source_drives")) {
        object(d, "DC source drive", {"source_id", "current_a", "voltage_v"}, {"source_id"});
        if (!drives.emplace(id(d.at("source_id"), "source_id"), d).second)
            fail("source_drives", "duplicate source");
    }
    J out{{"nodes", c.at("nodes")}, {"reference_node", c.at("reference_node")}, {"branches", J::array()}};
    for (const auto& b : c.at("branches")) {
        object(b, "DC branch", {"id", "kind", "from", "to", "cell_id", "emf_v"},
               {"id", "kind", "from", "to"});
        const auto bid = id(b.at("id"), "branch/id");
        if (!branch_ids.insert(bid).second)
            fail(bid, "duplicate branch");
        J branch = b;
        const auto kind = text(b.at("kind"), "branch/kind");
        if (kind == "battery") {
            const auto cell = id(b.at("cell_id"), "branch/cell_id");
            if (!batteries.contains(cell) || !connected.insert(cell).second)
                fail(cell, "unknown/duplicate battery");
            number(b.at("emf_v"), "emf_v", 0, 1e6);
        } else {
            if ((kind != "current_source" && kind != "voltage_source") || b.contains("cell_id") ||
                b.contains("emf_v"))
                fail(bid, "unsupported DC branch");
            if (!drives.contains(bid))
                fail(bid, "missing source drive");
            auto d = drives.at(bid);
            const char* field = kind == "current_source" ? "current_a" : "voltage_v";
            if (d.size() != 2 || !d.contains(field))
                fail(bid, "source requires its matching constant drive");
            branch[field] = number(d.at(field), field, -1e9, 1e9);
            drives.erase(bid);
        }
        out["branches"].push_back(branch);
    }
    if (!drives.empty())
        fail("source_drives", "unknown source");
    for (const auto& d : conditions.at("electrical_drives"))
        if (connected.contains(text(d.at("cell_id"), "cell_id")))
            fail("electrical_drives", "network battery cannot also have an imposed current/voltage");
    return out;
}
} // namespace battery::contract_detail
