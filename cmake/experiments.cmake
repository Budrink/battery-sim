# Isolated transport experiments: never selected by the production solver.
add_library(battery_gas_experiment STATIC native/experiments/src/gas_experiment.cpp native/experiments/src/gas_experiment.cu)
target_include_directories(battery_gas_experiment PUBLIC native/experiments/include)
target_link_libraries(battery_gas_experiment PUBLIC battery_native_core)
target_compile_options(battery_gas_experiment PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)
add_library(battery_kawasaki_gas STATIC native/experiments/src/kawasaki_gas.cu)
target_link_libraries(battery_kawasaki_gas PUBLIC battery_native_core)
target_compile_options(battery_kawasaki_gas PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)
