#pragma once
// Source compatibility only. The adapter also runs deterministic network transport.
#include "battery_native/gas/transport/cuda_gas_transport.hpp"
namespace battery {
using CudaKawasakiTransport = CudaGasTransport;
} // namespace battery
