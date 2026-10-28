#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace battery {
// Experimental apparatus, NOT a material/chemistry law or a temperature clamp.
struct HeatingProtocol {
    std::string id, heater_id, sensor_id;
    int heater_object=-1, sensor_object=-1;
    double initial_temperature_k=298.15, ramp_k_per_s=.1;
    double feedforward_w=4.5, proportional_w_per_k=1., maximum_power_w=30.;
    double cutoff_temperature_k=773.15, cutoff_rate_k_per_s=10.;
};
struct HeatingProtocolState {
    double previous_time_s=0., previous_temperature_k=298.15;
    double commanded_power_w=0., cutoff_time_s=-1.;
    bool stopped=false;
};
inline void validate_heating_protocol(const HeatingProtocol& p) {
    for(double x:{p.initial_temperature_k,p.ramp_k_per_s,p.proportional_w_per_k,
                  p.maximum_power_w,p.cutoff_temperature_k,p.cutoff_rate_k_per_s})
        if(!std::isfinite(x)||x<=0.)throw std::invalid_argument("Invalid heating protocol");
    if(!std::isfinite(p.feedforward_w)||p.feedforward_w<0.||p.feedforward_w>p.maximum_power_w)
        throw std::invalid_argument("Invalid heating feedforward");
}
// Pure proposed state: retrying a thermal prepare cannot consume controller time
// or latch a cutoff. Commit this state only with the same accepted physical step.
inline HeatingProtocolState heating_command(const HeatingProtocol& p,HeatingProtocolState state,
                                             double time,double sensor_temperature) {
    if(!std::isfinite(time)||time<state.previous_time_s || !std::isfinite(sensor_temperature)||sensor_temperature<=0.)
        throw std::invalid_argument("Nonphysical heating observation");
    const double dt=time-state.previous_time_s;
    const double rate=dt>0.?(sensor_temperature-state.previous_temperature_k)/dt:0.;
    if(!state.stopped && sensor_temperature>=p.cutoff_temperature_k && rate>=p.cutoff_rate_k_per_s) {
        state.stopped=true;state.cutoff_time_s=time;
    }
    const double target=p.initial_temperature_k+p.ramp_k_per_s*time;
    state.commanded_power_w=state.stopped?0.:std::clamp(
        p.feedforward_w+p.proportional_w_per_k*(target-sensor_temperature),0.,p.maximum_power_w);
    state.previous_time_s=time;state.previous_temperature_k=sensor_temperature;
    return state;
}
} // namespace battery
