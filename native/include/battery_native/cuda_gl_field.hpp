#pragma once
#include "battery_native/cuda_solver.hpp"
#include "battery_native/palette.hpp"
#include <memory>
#include <vector>

namespace battery {
// Owns registration, not the GL buffer. Destroy/detach before deleting the
// GL buffer/context. Fields stay on the GPU; only rendered colors are packed.
class CudaGlField {
public:
    CudaGlField();
    ~CudaGlField();
    void attach(unsigned int buffer,const std::vector<int>& volume_ids,const std::vector<float>& aging);
    void detach();
    void update(const ThermalState* state,FieldLayer layer,float tmin,float tmax,float opacity);
    // Scalar volume rendering: T, combined degradation, destruction, unused.
    // Interpolate the physical scalar before looking up its display color.
    void update_samples(const ThermalState* state);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
