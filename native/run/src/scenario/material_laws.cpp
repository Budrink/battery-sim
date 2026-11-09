#include "scenario_internal.hpp"
#include "battery_native/thermal/constitutive.hpp"
#include "battery_native/thermal/state_geometry.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

namespace battery::scenario_detail {
Vec3 vec3(const json& value, const std::string& where) {
    if (!value.is_array() || value.size() != 3) {
        throw std::runtime_error(where + " must contain three numbers");
    }
    return {value.at(0).get<double>(), value.at(1).get<double>(), value.at(2).get<double>()};
}

ShapeKind shape_kind(const std::string& value) {
    if (value == "cylinder")
        return ShapeKind::Cylinder;
    if (value == "prismatic")
        return ShapeKind::Prismatic;
    if (value == "pouch")
        return ShapeKind::Pouch;
    throw std::runtime_error("unknown cell shape: " + value);
}

TransitionKind transition_kind(const std::string& value) {
    if (value == "constant")
        return TransitionKind::Constant;
    if (value == "smooth")
        return TransitionKind::Smooth;
    if (value == "first_order")
        return TransitionKind::FirstOrder;
    if (value == "second_order")
        return TransitionKind::SecondOrder;
    if (value == "state_geometry")
        return TransitionKind::StateGeometry;
    if (value == "directed_geometry")
        return TransitionKind::DirectedGeometry;
    throw std::runtime_error("unknown transition kind: " + value);
}

EdgePhase edge_phase(const std::string& value) {
    if (value == "low")
        return EdgePhase::Low;
    if (value == "high")
        return EdgePhase::High;
    if (value == "destroyed")
        return EdgePhase::Destroyed;
    throw std::runtime_error("unknown edge phase: " + value);
}

ScalarTable parse_table(const json& value, bool positive) {
    if (!value.is_array() || value.size() < 2 || value.size() > 8)
        throw std::runtime_error("Constitutive table requires 2..8 [temperature_k,value] knots");
    ScalarTable t;
    t.count = static_cast<int>(value.size());
    for (int i = 0; i < t.count; ++i) {
        if (!value[i].is_array() || value[i].size() != 2)
            throw std::runtime_error("Invalid constitutive knot");
        t.temperature_k[i] = value[i][0].get<double>();
        t.value[i] = value[i][1].get<double>();
        if (!std::isfinite(t.temperature_k[i]) || t.temperature_k[i] <= 0 || !std::isfinite(t.value[i]) ||
            (positive && t.value[i] <= 0) || (i && t.temperature_k[i] <= t.temperature_k[i - 1]))
            throw std::runtime_error("Constitutive knots must be finite, ordered, with positive capacities");
    }
    return t;
}
SurfaceConnectivity parse_surface(const json& value) {
    SurfaceConnectivity r;
    r.reference_temperature_k = value.value("reference_temperature_k", 298.15);
    r.temperature_scale_k = value.value("temperature_scale_k", 100.);
    r.linear = vec3(value.value("linear", json::array({0., 0., 0.})), "surface_connectivity.linear");
    r.quadratic = vec3(value.value("quadratic", json::array({0., 0., 0.})), "surface_connectivity.quadratic");
    r.damage = vec3(value.value("damage", json::array({0., 0., 0.})), "surface_connectivity.damage");
    r.temperature_contrast = vec3(value.value("temperature_contrast", json::array({0., 0., 0.})),
                                  "surface_connectivity.temperature_contrast");
    r.damage_contrast = vec3(value.value("damage_contrast", json::array({0., 0., 0.})),
                             "surface_connectivity.damage_contrast");
    if (value.contains("temperature_log_response")) {
        r.temperature_log_response = parse_table(value.at("temperature_log_response"), false);
        for (double v : {r.linear.x, r.linear.y, r.linear.z, r.quadratic.x, r.quadratic.y, r.quadratic.z})
            if (v != 0.)
                throw std::runtime_error(
                    "Tabulated response cannot be combined with a temperature polynomial");
    }
    return r;
}

ConnectivityLaw parse_law(const json& value) {
    ConnectivityLaw law;
    law.kind = transition_kind(value.value("kind", "constant"));
    law.destroyed_geometry_reference_logit = value.value("destroyed_geometry_reference_logit", 0.);
    law.interface_damage_log_response = value.value("interface_damage_log_response", 0.);
    if (law.kind == TransitionKind::DirectedGeometry &&
        (value.contains("l0_m") || value.contains("destroyed_l_m") || value.contains("tensor_only")))
        throw std::runtime_error("Directed contacts use only tensor geometry; obsolete length/compatibility "
                                 "fields are not accepted");
    law.l0_m = law.kind == TransitionKind::DirectedGeometry ? 0. : value.value("l0_m", 0.);
    law.l_low_m = value.value("l_low_m", law.l0_m);
    law.l_high_m = value.value("l_high_m", law.l_low_m);
    law.transition_temperature_k = value.value("transition_temperature_k", 400.0);
    law.transition_width_k = value.value("transition_width_k", 10.0);
    law.hysteresis_k = value.value("hysteresis_k", 0.0);
    law.relaxation_time_s = value.value("relaxation_time_s", 0.0);
    law.background_conductance_w_per_k = value.value("background_conductance_w_per_k", 0.0);
    law.active_conductance_w_per_k = value.value("active_conductance_w_per_k", 0.0);
    law.activation_width_m = value.value("activation_width_m", 1.0e-5);
    law.geometry_reference_logit = value.value("geometry_reference_logit", 0.);
    law.damage_onset_k = value.value("damage_onset_k", 450.0);
    law.damage_rate_at_onset_per_s = value.value("damage_rate_at_onset_per_s", 0.0);
    law.damage_temperature_scale_k = value.value("damage_temperature_scale_k", 25.0);
    law.damage_l_shift_m = value.value("damage_l_shift_m", 0.0);
    law.damage_conductance_factor_at_one = value.value("damage_conductance_factor_at_one", 1.0);
    law.endpoint_damage_weight = value.value("endpoint_damage_weight", 0.0);
    law.destruction_damage = value.value("destruction_damage", 1.0);
    law.destroyed_l0_m = value.value("destroyed_l0_m", law.l0_m);
    law.destroyed_l_m = value.value("destroyed_l_m", law.l_low_m);
    law.destroyed_background_conductance_w_per_k =
        value.value("destroyed_background_conductance_w_per_k", law.background_conductance_w_per_k);
    law.destroyed_active_conductance_w_per_k =
        value.value("destroyed_active_conductance_w_per_k", law.active_conductance_w_per_k);
    if (law.activation_width_m <= 0.0 || law.transition_width_k <= 0.0 ||
        law.damage_temperature_scale_k <= 0.0) {
        throw std::runtime_error("interface law scales must be positive");
    }
    return law;
}

} // namespace battery::scenario_detail
