#include "scenario_internal.hpp"
#include <map>
#include <set>

namespace battery::scenario_detail {
void parse_circuit(const json& root, Scenario& scenario, const CellIndex& cells) {
    if (!root.contains("electrical_circuit"))
        return;
    const auto& c = root.at("electrical_circuit");
    const auto ref = c.at("reference_node").get<std::string>();
    std::map<std::string, int> nodes;
    if (!c.at("nodes").is_array() || !c.at("branches").is_array())
        throw std::invalid_argument("DC circuit expects node and branch arrays");
    for (const auto& node : c.at("nodes")) {
        auto name = node.get<std::string>();
        if (name.empty() || !nodes.emplace(name, -1).second)
            throw std::invalid_argument("Duplicate/empty circuit node");
    }
    if (!nodes.contains(ref))
        throw std::invalid_argument("Unknown circuit reference node");
    nodes[ref] = 0;
    int next = 1;
    for (auto& [name, index] : nodes)
        if (name != ref)
            index = next++;
    scenario.electrical_circuit.nodes = next;
    std::set<std::string> ids;
    for (const auto& b : c.at("branches")) {
        if (!ids.insert(b.at("id").get<std::string>()).second)
            throw std::invalid_argument("Duplicate circuit branch");
        CircuitBranch branch;
        branch.from = nodes.at(b.at("from").get<std::string>());
        branch.to = nodes.at(b.at("to").get<std::string>());
        const auto kind = b.at("kind").get<std::string>();
        if (kind == "battery") {
            branch.kind = CircuitKind::Battery;
            branch.object = int(cells.at(b.at("cell_id").get<std::string>()));
            branch.value = b.at("emf_v");
        } else if (kind == "current_source") {
            branch.kind = CircuitKind::CurrentSource;
            branch.value = b.at("current_a");
        } else if (kind == "voltage_source") {
            branch.kind = CircuitKind::VoltageSource;
            branch.value = b.at("voltage_v");
        } else
            throw std::invalid_argument("Unknown circuit branch kind");
        scenario.electrical_circuit.branches.push_back(branch);
    }
}
} // namespace battery::scenario_detail
