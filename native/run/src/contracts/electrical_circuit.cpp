#include "contract_internal.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace battery::contract_detail {
namespace {
// Piecewise-linear input currents, with the same constant endpoint tails as native drives.
using Profile = std::vector<std::pair<double, double>>;
Profile current(const J& drive, std::set<double>& knots) {
    if (drive.contains("current_a") == drive.contains("current_profile_a"))
        fail("circuit drive", "specify current_a OR current_profile_a");
    if (drive.contains("current_a"))
        return {{0, number(drive.at("current_a"), "current_a", -1e12, 1e12)}};
    const auto& values = drive.at("current_profile_a");
    array(values, "current_profile_a");
    if (values.empty())
        fail("current_profile_a", "empty profile");
    Profile result;
    for (const auto& point : values) {
        if (!point.is_array() || point.size() != 2)
            fail("current_profile_a", "expected [time_s, current_a]");
        const double t = number(point[0], "time_s", 0, 1e15);
        const double i = number(point[1], "current_a", -1e12, 1e12);
        if (!result.empty() && t <= result.back().first)
            fail("current_profile_a", "times must strictly increase");
        result.emplace_back(t, i);
        knots.insert(t);
    }
    return result;
}
double sample(const Profile& p, double t) {
    if (t <= p.front().first)
        return p.front().second;
    for (std::size_t k = 1; k < p.size(); ++k)
        if (t <= p[k].first) {
            const double f = (t - p[k - 1].first) / (p[k].first - p[k - 1].first);
            return p[k - 1].second + f * (p[k].second - p[k - 1].second);
        }
    return p.back().second;
}
struct Branch {
    std::string from, to;
    Profile current;
};
} // namespace

void validate_electrical_circuit(const J& assembly, const J& conditions) {
    const auto& circuit = assembly.at("electrical_circuit");
    object(circuit, "electrical_circuit", {"mode", "nodes", "branches"}, {"mode", "nodes", "branches"});
    array(circuit.at("nodes"), "circuit/nodes");
    array(circuit.at("branches"), "circuit/branches");
    array(conditions.at("source_drives"), "source_drives");
    array(conditions.at("electrical_drives"), "electrical_drives");
    const auto mode = text(circuit.at("mode"), "circuit/mode");
    if (mode == "unmapped_drives") {
        if (!circuit.at("nodes").empty() || !circuit.at("branches").empty() ||
            !conditions.at("source_drives").empty())
            fail("electrical_circuit", "unmapped_drives cannot claim a circuit or sources");
        return;
    }
    if (mode != "prescribed_currents")
        fail("electrical_circuit", "unsupported mode; coupled network solve is not implemented");
    std::set<std::string> nodes, battery_ids, branch_ids, connected_batteries, source_ids, used_nodes;
    for (const auto& node : circuit.at("nodes"))
        if (!nodes.insert(id(node, "circuit/node")).second)
            fail("circuit/nodes", "duplicate node");
    for (const auto& battery : assembly.at("batteries"))
        battery_ids.insert(text(battery.at("id"), "battery/id"));
    std::map<std::string, J> drives, sources;
    for (const auto& drive : conditions.at("electrical_drives")) {
        const auto target = text(drive.at("cell_id"), "drive/cell_id");
        if (!battery_ids.contains(target) || !drives.emplace(target, drive).second)
            fail(target, "unknown or duplicate battery drive");
        if (drive.value("current_input", "measured") != "measured")
            fail(target, "prescribed circuit requires measured currents; independent limiting breaks KCL");
    }
    for (const auto& drive : conditions.at("source_drives")) {
        object(drive, "source_drive", {"source_id", "current_a", "current_profile_a"}, {"source_id"});
        if (!sources.emplace(id(drive.at("source_id"), "source_id"), drive).second)
            fail("source_drives", "duplicate source drive");
    }
    std::set<double> knots{0};
    std::vector<Branch> branches;
    for (const auto& b : circuit.at("branches")) {
        object(b, "circuit/branch", {"id", "kind", "from", "to", "cell_id"}, {"id", "kind", "from", "to"});
        const auto bid = id(b.at("id"), "branch/id");
        const auto from = id(b.at("from"), "branch/from"), to = id(b.at("to"), "branch/to");
        if (!branch_ids.insert(bid).second || !nodes.contains(from) || !nodes.contains(to) || from == to)
            fail(bid, "duplicate branch, unknown node or shorted terminals");
        used_nodes.insert(from);
        used_nodes.insert(to);
        const auto kind = text(b.at("kind"), "branch/kind");
        Profile p;
        if (kind == "battery") {
            const auto cell = id(b.at("cell_id"), "branch/cell_id");
            if (!battery_ids.contains(cell) || !connected_batteries.insert(cell).second)
                fail(bid, "unknown or multiply connected battery");
            // No drive means explicitly unenergized, as in the existing native adapter.
            p = current(drives.contains(cell) ? drives.at(cell) : J{{"current_a", 0}}, knots);
        } else if (kind == "current_source") {
            if (b.contains("cell_id") || !sources.contains(bid))
                fail(bid, "source needs its own source_drive and cannot own a battery");
            source_ids.insert(bid);
            p = current(sources.at(bid), knots);
        } else
            fail(bid, "unsupported branch kind; wires are shared nodes, not resistive branches");
        branches.push_back({from, to, std::move(p)});
    }
    if (nodes != used_nodes)
        fail("circuit/nodes", "unconnected nodes");
    for (const auto& [cell, drive] : drives)
        if (!connected_batteries.contains(cell))
            fail(cell, "drive not represented in circuit");
    for (const auto& [source, drive] : sources)
        if (!source_ids.contains(source))
            fail(source, "drive for unknown source");
    // A linear sum is zero between all union knots iff it is zero at every knot.
    for (double t : knots) {
        std::map<std::string, double> balance, magnitude;
        for (const auto& branch : branches) {
            const double i = sample(branch.current, t);
            balance[branch.from] += i;
            balance[branch.to] -= i;
            magnitude[branch.from] += std::abs(i);
            magnitude[branch.to] += std::abs(i);
        }
        for (const auto& node : nodes)
            if (std::abs(balance[node]) > 1e-9 * std::max(1., magnitude[node]))
                fail("circuit/" + node, "Kirchhoff current imbalance at t=" + std::to_string(t));
    }
}
} // namespace battery::contract_detail
