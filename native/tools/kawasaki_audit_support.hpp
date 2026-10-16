#pragma once
#include "battery_native/kawasaki_gas_math.hpp"
#include <nlohmann/json.hpp>
#include <chrono>
#include <string>

namespace kawasaki_audit {
namespace kg=battery::kawasaki;
using J=nlohmann::json;
using Timer=std::chrono::steady_clock;
constexpr int samples=64;
struct Reference {std::vector<std::vector<kg::Node>> samples;double wall=0;std::uint64_t steps=0;};
double seconds(Timer::time_point);
kg::Node node(double p,double t,double volume=1e-6);
kg::Network fixture(const std::string&);
kg::Network grid(int side);
Reference reference(const kg::Network&,double duration,double step_cap);
J ensemble(kg::Network,const Reference&,double duration,double fraction,int substeps,int replicas);
void tail_sweep(const char* path,int replicas);
}
