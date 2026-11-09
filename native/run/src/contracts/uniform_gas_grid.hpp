#pragma once

// Included inside enclosure_gas_mesh.cpp's anonymous namespace. Geometry stays
// local to the contract compiler: the solver receives volumes and open faces.
struct UniformGasGrid {
    std::vector<Box> fragments;
    std::vector<int> voxel;
    std::vector<Box> supports;
    std::array<int, 3> divisions{};
};

UniformGasGrid uniform_gas_grid(const Box& enclosure, const Scenario& scene, double step) {
    UniformGasGrid result;
    for (int d = 0; d < 3; ++d) {
        const double count = std::ceil((enclosure.hi[d] - enclosure.lo[d]) / step - 1e-9);
        if (count > 5000)
            fail("gas_mesh", "Too many uniform cells along an axis");
        result.divisions[d] = std::max(1, int(count));
    }
    const auto& n = result.divisions;
    if (double(n[0]) * n[1] * n[2] > 5000)
        fail("gas_mesh", "Too many uniform gas cells (limit 5000)");
    for (const auto& cell : scene.cells)
        if (cell.orientation_xyzw != std::array<double, 4>{0, 0, 0, 1})
            fail("gas_mesh", "Rotated solids require a different mesher");
    for (int z = 0; z < n[2]; ++z)
        for (int y = 0; y < n[1]; ++y)
            for (int x = 0; x < n[0]; ++x) {
                const int ijk[]{x, y, z};
                Box support;
                for (int d = 0; d < 3; ++d) {
                    const double width = (enclosure.hi[d] - enclosure.lo[d]) / n[d];
                    support.lo[d] = enclosure.lo[d] + width * ijk[d];
                    support.hi[d] = enclosure.lo[d] + width * (ijk[d] + 1);
                }
                const int slot = int(result.supports.size());
                result.supports.push_back(support);
                std::vector<Box> free{support};
                for (const auto& solid : scene.cells) {
                    std::vector<Box> next;
                    for (const auto& part : free)
                        subtract(part, box(solid), next);
                    free = std::move(next);
                }
                for (const auto& part : free) {
                    result.fragments.push_back(part);
                    result.voxel.push_back(slot);
                }
                if (result.fragments.size() > 12000)
                    fail("gas_mesh", "Too many obstacle fragments (limit 12000)");
            }
    return result;
}

// Keep a regular background lattice, but not a single mixed reservoir around
// every obstacle. Only rectangular unions are well represented by one gas
// temperature and one centroid. In particular, a narrow gap must not instantly
// mix around a corner into the large volume above/below the battery.
void merge_uniform_gas(J& out, const UniformGasGrid& grid, const std::string& id) {
    const auto name = [&](int i) { return id + "-air-" + std::to_string(i); };
    auto& net = out["gas_network"];
    std::map<std::string, int> index;
    std::vector<int> parent(grid.fragments.size());
    auto fluid_bounds = grid.fragments;
    for (int i = 0; i < int(parent.size()); ++i) {
        parent[i] = i;
        index[name(i)] = i;
    }
    const auto root = [&](int i) {
        while (parent[i] != i)
            i = parent[i];
        return i;
    };
    bool changed;
    do {
        changed = false;
        for (const auto& ch : net["channels"]) {
            auto a = index.find(ch.at("from")), b = index.find(ch.at("to"));
            if (a == index.end() || b == index.end() || grid.voxel[a->second] != grid.voxel[b->second])
                continue;
            const int ra = root(a->second), rb = root(b->second);
            if (ra == rb) continue;
            const int axis = ch.at("mixing_axis");
            const auto& ba = fluid_bounds[ra];
            const auto& bb = fluid_bounds[rb];
            bool rectangular = std::abs(ba.hi[axis] - bb.lo[axis]) < eps ||
                               std::abs(bb.hi[axis] - ba.lo[axis]) < eps;
            for (int d = 0; d < 3; ++d)
                if (d != axis)
                    rectangular &= std::abs(ba.lo[d] - bb.lo[d]) < eps &&
                                   std::abs(ba.hi[d] - bb.hi[d]) < eps;
            if (!rectangular) continue;
            Box joined;
            for (int d = 0; d < 3; ++d) {
                joined.lo[d] = std::min(ba.lo[d], bb.lo[d]);
                joined.hi[d] = std::max(ba.hi[d], bb.hi[d]);
            }
            parent[rb] = ra;
            fluid_bounds[ra] = joined;
            changed = true;
        }
    } while (changed);
    std::map<int, int> compact;
    std::map<std::string, std::string> remap;
    std::map<std::string, J> merged;
    auto& geometry = out["metadata"]["gas_volume_geometry"];
    for (int i = 0; i < int(parent.size()); ++i) {
        const int r = root(i);
        if (!compact.contains(r))
            compact[r] = int(compact.size());
        remap[name(i)] = name(compact.at(r));
        geometry.erase(name(i));
    }
    J nodes = J::array();
    for (const auto& old : net["compartments"]) {
        const auto key = old.at("id").get<std::string>();
        if (!remap.contains(key)) {
            nodes.push_back(old);
            continue;
        }
        const auto target = remap.at(key);
        const int i = index.at(key), slot = grid.voxel[i];
        if (!merged.contains(target)) {
            merged[target] = old;
            merged[target]["id"] = target;
            const auto& support = grid.supports[slot];
            geometry[target] = {{"low_m", support.lo},
                                {"high_m", support.hi},
                                {"fluid_low_m", fluid_bounds[root(i)].lo},
                                {"fluid_high_m", fluid_bounds[root(i)].hi},
                                {"fluid_centroid_m", center(fluid_bounds[root(i)])},
                                {"lattice_index", slot},
                                {"fluid_fragments", J::array()}};
        } else {
            for (const char* field : {"volume_m3", "initial_mass_kg", "initial_chemical_j"})
                if (old.contains(field))
                    merged[target][field] =
                        merged[target].at(field).get<double>() + old.at(field).get<double>();
            if (old.contains("ambient_boundary")) {
                if (!merged[target].contains("ambient_boundary"))
                    merged[target]["ambient_boundary"] = old.at("ambient_boundary");
                else
                    merged[target]["ambient_boundary"]["area_m2"] =
                        merged[target]["ambient_boundary"].at("area_m2").get<double>() +
                        old.at("ambient_boundary").at("area_m2").get<double>();
            }
        }
        geometry[target]["fluid_fragments"].push_back(
            {{"low_m", grid.fragments[i].lo}, {"high_m", grid.fragments[i].hi}});
    }
    for (const auto& [key, node] : merged)
        nodes.push_back(node);
    net["compartments"] = std::move(nodes);
    J channels = J::array();
    std::map<std::pair<std::string, std::string>, std::size_t> passages;
    for (auto ch : net["channels"]) {
        const auto old_a = ch.at("from").get<std::string>(), old_b = ch.at("to").get<std::string>();
        const bool internal = remap.contains(old_a) && remap.contains(old_b);
        for (const char* end : {"from", "to"})
            if (remap.contains(ch.at(end)))
                ch[end] = remap.at(ch.at(end));
        if (ch.at("from") == ch.at("to"))
            continue;
        if (internal) {
            const auto key = std::make_pair(ch.at("from").get<std::string>(), ch.at("to").get<std::string>());
            const int axis = ch.at("mixing_axis");
            const auto& a = fluid_bounds[root(index.at(old_a))];
            const auto& b = fluid_bounds[root(index.at(old_b))];
            const double length = std::abs(center(a)[axis] - center(b)[axis]);
            ch["mixing_length_m"] = length;
            ch["inertial_length_m"] = length;
            if (passages.contains(key)) {
                auto& previous = channels[passages.at(key)];
                previous["area_m2"] = previous.at("area_m2").get<double>() + ch.at("area_m2").get<double>();
                continue;
            }
            passages[key] = channels.size();
        }
        channels.push_back(ch);
    }
    net["channels"] = std::move(channels);
    for (auto& contact : net["thermal_contacts"])
        if (remap.contains(contact.at("compartment")))
            contact["compartment"] = remap.at(contact.at("compartment"));
    auto& report = out["metadata"]["resolved_gas_meshes"][id];
    report["cells"] = merged.size();
    report["passages"] = passages.size();
    report["layout"] = "uniform_masked";
    report["subvolume_merge"] = "rectangular_only";
    report["lattice_divisions"] = grid.divisions;
    report["lattice_slots"] = grid.supports.size();
    report["obstacle_fragments"] = grid.fragments.size();
}
