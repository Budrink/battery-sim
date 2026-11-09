# Physics and numerical algorithms. No scenario loader or UI dependency.
add_library(battery_native_core STATIC
    native/core/src/electrical/circuit_validation.cpp
    native/core/src/mechanics/shell.cpp
    native/core/src/simulation/simulation.cpp
    native/core/src/simulation/observations.cpp
    native/core/src/model/model.cpp
    native/core/src/model/validation.cpp
    native/core/src/io/checkpoint.cpp
    native/core/src/thermal/thermal_mesh.cpp
    native/core/src/mechanics/aperture_geometry.cpp
    native/core/src/gas/transport/mass_mesh.cpp
    native/core/src/gas/transport/local_gas_transport.cpp
    native/core/src/gas/transport/implicit_gas_transport.cpp
    native/core/src/gas/transport/kawasaki_gas.cpp
    native/core/src/gas/transport/cuda_gas_transport.cu
    native/core/src/simulation/cuda_solver.cu
)
target_include_directories(battery_native_core PUBLIC native/core/include)
add_library(battery::core ALIAS battery_native_core)
target_link_libraries(battery_native_core PUBLIC CUDA::cudart PRIVATE nlohmann_json::nlohmann_json)
target_compile_options(battery_native_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)

