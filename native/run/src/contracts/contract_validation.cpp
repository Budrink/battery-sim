#include "contract_internal.hpp"
#include <cmath>
#include <stdexcept>

namespace battery::contract_detail {
[[noreturn]] void fail(const std::string& where, const std::string& reason) {
    throw std::invalid_argument("experiment contract: " + where + ": " + reason);
}

void object(const J& j, const std::string& where, Keys allowed, Keys required) {
    if (!j.is_object())
        fail(where, "expected object");
    const std::set<std::string> keys(allowed.begin(), allowed.end());
    for (const auto& [key, value] : j.items())
        if (!keys.contains(key))
            fail(where + "/" + key, "unknown or misplaced field");
    for (const auto* key : required)
        if (!j.contains(key))
            fail(where, "missing " + std::string(key));
}

void array(const J& j, const std::string& where) {
    if (!j.is_array())
        fail(where, "expected array");
}

std::string text(const J& j, const std::string& where) {
    if (!j.is_string() || j.get_ref<const std::string&>().empty())
        fail(where, "expected nonempty string");
    return j.get<std::string>();
}

std::string id(const J& j, const std::string& where) {
    const auto s = text(j, where);
    if (s.size() > 128 ||
        s.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-") !=
            std::string::npos)
        fail(where, "ID must contain 1..128 ASCII letters, digits, '_' or '-'");
    return s;
}

double number(const J& j, const std::string& where, double lo, double hi) {
    if (!j.is_number())
        fail(where, "expected number");
    const double x = j.get<double>();
    if (!std::isfinite(x) || x < lo || x > hi)
        fail(where, "number outside permitted range");
    return x;
}

void finite_tree(const J& j) {
    if (j.is_number_float() && !std::isfinite(j.get<double>()))
        fail("document", "nonfinite number");
    if (j.is_structured())
        for (const auto& v : j)
            finite_tree(v);
}

void grid(const J& j, const std::string& where) {
    if (!j.is_array() || j.size() != 3)
        fail(where, "expected three integer divisions");
    for (const auto& n : j)
        if (!n.is_number_integer() || n.get<double>() < 1 || n.get<double>() > 256)
            fail(where, "divisions must be integers in [1,256]");
}

double volume(const J& g) {
    object(g, "geometry", {"kind", "dimensions_m"}, {"kind", "dimensions_m"});
    const auto kind = text(g.at("kind"), "geometry/kind");
    const auto& d = g.at("dimensions_m");
    const bool cylinder = kind == "cylinder";
    if (!cylinder && kind != "prismatic" && kind != "pouch")
        fail("geometry", "unknown shape");
    if (!d.is_array() || d.size() != (cylinder ? 2 : 3))
        fail("geometry", "wrong dimension count");
    for (const auto& x : d)
        number(x, "geometry/dimensions_m", 1e-12, 1e6);
    return cylinder ? std::acos(-1.) * d[0].get<double>() * d[0].get<double>() * d[1].get<double>()
                    : d[0].get<double>() * d[1].get<double>() * d[2].get<double>();
}

void provenance(const J& p) {
    object(p, "provenance", {"status", "notes", "sources"}, {"status", "notes"});
    const std::set<std::string> states{"measured", "calibrated", "assumed", "mixed", "synthetic"};
    if (!states.contains(text(p.at("status"), "provenance/status")))
        fail("provenance", "unknown status");
    text(p.at("notes"), "provenance/notes");
    if (p.contains("sources")) {
        array(p.at("sources"), "provenance/sources");
        for (const auto& s : p.at("sources"))
            text(s, "provenance/source");
    }
}

void model(const J& m) {
    object(m, "battery_model",
           {"format", "version", "id", "revision", "provenance", "geometry", "contents", "shell", "vents"},
           {"format", "version", "id", "revision", "provenance", "geometry", "contents", "shell", "vents"});
    if (m.at("format") != "battery-sim.battery" || m.at("version") != 1 ||
        !m.at("version").is_number_integer())
        fail("battery_model", "unsupported format/version");
    id(m.at("id"), "model/id");
    id(m.at("revision"), "model/revision");
    provenance(m.at("provenance"));
    volume(m.at("geometry"));
    const auto& c = m.at("contents");
    object(c, "contents", {"material_id", "thermal", "degradation", "conversion", "electrical"},
           {"material_id", "thermal", "conversion"});
    text(c.at("material_id"), "contents/material_id");
    object(c.at("thermal"), "contents/thermal",
           {"heat_capacity_j_per_k", "conductivity_w_per_m_k", "surface_connectivity", "bulk_connectivity",
            "bulk_response", "heat_capacity_law", "bulk_failure_transient", "bulk_destruction_on_boundary"},
           {"heat_capacity_j_per_k", "conductivity_w_per_m_k"});
    object(c.at("conversion"), "contents/conversion",
           {"chemistry",
            "mass_kg",
            "reactant_cp",
            "activated_cp",
            "residue_cp",
            "net_heat_j_per_kg",
            "degradation_rate_per_s",
            "degradation_reference_k",
            "degradation_activation_j_per_mol",
            "reaction_rate_per_s",
            "reaction_reference_k",
            "reaction_activation_j_per_mol",
            "charge_conversion_per_nominal_charge",
            "precursor_yield",
            "precursor_yield_temperature_per_k",
            "precursor_temperature_ref_k",
            "runaway_reporting_rate_per_s",
            "gas_yield",
            "gas_yield_max",
            "gas_yield_reference_k",
            "gas_yield_temperature_per_k",
            "gas_yield_density_exponent",
            "gas_chemical_fraction",
            "residue_conductivity",
            "volatile_phase"},
           {"chemistry", "mass_kg", "reactant_cp", "activated_cp", "residue_cp", "net_heat_j_per_kg",
            "degradation_rate_per_s", "degradation_reference_k", "degradation_activation_j_per_mol",
            "reaction_rate_per_s", "reaction_reference_k", "reaction_activation_j_per_mol", "gas_yield",
            "gas_chemical_fraction", "residue_conductivity"});
    if (c.at("conversion").at("chemistry") != "r1-r2-r3")
        fail("conversion", "only the existing R1/R2/R3 model is supported");
    if (c.contains("degradation"))
        object(c.at("degradation"), "contents/degradation",
               {"onset_k", "rate_at_onset_per_s", "temperature_scale_k", "aging_damage_multiplier"});
    if (c.contains("electrical"))
        object(c.at("electrical"), "contents/electrical",
               {"resistance_ohm", "reference_temperature_k", "temperature_log_slope_per_k",
                "temperature_log_response", "product_conductivity_ratio", "terminal_axis",
                "reversible_voltage_floor_v", "min_resistance_ohm", "max_resistance_ohm",
                "charge_acceptance_fraction", "nominal_capacity_coulomb", "charge_log_slope",
                "charge_state_input"},
               {"resistance_ohm"});
    if (!m.at("shell").is_null())
        object(m.at("shell"), "shell",
               {"material_id", "thickness_m", "density_kg_m3", "specific_heat_j_kg_k",
                "conductivity_w_per_m_k", "conductivity_frame", "contents_contact_h", "seam_contact_h",
                "surface_connectivity", "damage_onset_k", "damage_rate_per_s", "damage_scale_k",
                "destroyed_conductance_factor", "contents_surface_connectivity",
                "contents_connectivity_per_m2", "face_materials"},
               {"material_id", "thickness_m", "density_kg_m3", "specific_heat_j_kg_k",
                "conductivity_w_per_m_k", "contents_contact_h", "seam_contact_h"});
    array(m.at("vents"), "vents");
    std::set<std::string> vents;
    for (const auto& v : m.at("vents")) {
        object(v, "vent", {"id", "port_suffix", "channel"}, {"id", "port_suffix", "channel"});
        if (!vents.insert(id(v.at("id"), "vent/id")).second)
            fail("vents", "duplicate local ID");
        text(v.at("port_suffix"), "vent/port_suffix");
        if (!v.at("channel").is_object())
            fail("vent/channel", "expected native rupture channel law");
        for (const auto* field : {"id", "from", "to", "port_id"})
            if (v.at("channel").contains(field))
                fail("vent/channel", "routing belongs to assembly");
        if (v.at("channel").value("kind", "") != "rupture")
            fail("vent/channel", "expected kind=rupture");
    }
}

void measurements(const J& list, const std::set<std::string>& cells) {
    array(list, "measurements");
    std::set<std::string> ids;
    for (const auto& m : list) {
        object(m, "measurement", {"id", "quantity", "unit", "mapping", "data", "uncertainty", "notes"},
               {"id", "quantity", "unit", "mapping"});
        if (!ids.insert(id(m.at("id"), "measurement/id")).second)
            fail("measurements", "duplicate ID");
        const auto q = text(m.at("quantity"), "measurement/quantity");
        const auto u = text(m.at("unit"), "measurement/unit");
        if (!((q == "temperature" && (u == "K" || u == "degC")) || (q == "pressure_absolute" && u == "Pa") ||
              (q == "pressure_gauge" && u == "Pa") || (q == "current" && u == "A") ||
              (q == "voltage" && u == "V")))
            fail("measurement", "quantity/unit mismatch");
        const auto& mapping = m.at("mapping");
        object(mapping, "measurement/mapping", {"status", "cell_id", "notes"}, {"status"});
        const auto status = text(mapping.at("status"), "measurement/mapping/status");
        if (status != "unknown" && status != "confirmed" && status != "hypothesis")
            fail("measurement/mapping", "unknown status");
        if (status == "unknown" && mapping.contains("cell_id"))
            fail("measurement/mapping", "unknown mapping must not name a cell");
        if (status != "unknown" &&
            (!mapping.contains("cell_id") || !cells.contains(mapping.at("cell_id").get<std::string>())))
            fail("measurement/mapping", "expected known cell_id");
        if (m.contains("data")) {
            object(m.at("data"), "measurement/data",
                   {"artifact_id", "sha256", "time_column", "value_column", "time_offset_s"},
                   {"artifact_id", "sha256", "time_column", "value_column", "time_offset_s"});
            const auto& d = m.at("data");
            for (const auto* key : {"artifact_id", "time_column", "value_column"})
                text(d.at(key), key);
            const auto hash = text(d.at("sha256"), "measurement/data/sha256");
            if (hash.size() != 64 || hash.find_first_not_of("0123456789abcdef") != std::string::npos)
                fail("measurement/data", "expected lowercase SHA256");
            number(d.at("time_offset_s"), "time_offset_s", -1e15, 1e15);
        }
        if (m.contains("uncertainty")) {
            object(m.at("uncertainty"), "measurement/uncertainty", {"absolute", "burst_absolute", "notes"},
                   {"absolute"});
            number(m.at("uncertainty").at("absolute"), "uncertainty/absolute", 0, 1e15);
            if (m.at("uncertainty").contains("burst_absolute"))
                number(m.at("uncertainty").at("burst_absolute"), "uncertainty/burst_absolute", 0, 1e15);
        }
    }
}
} // namespace battery::contract_detail
