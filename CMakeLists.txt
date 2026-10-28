cmake_minimum_required(VERSION 3.20)
project(battery_sim_native LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
find_package(CUDAToolkit REQUIRED)

include(FetchContent)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

add_library(battery_native_core STATIC
    native/src/scenario.cpp
    native/src/shell.cpp
    native/src/simulation.cpp
    native/src/checkpoint.cpp
    native/src/thermal_mesh.cpp
    native/src/aperture_geometry.cpp
    native/src/mass_mesh.cpp
    native/src/local_gas_transport.cpp
    native/src/implicit_gas_transport.cpp
    native/src/kawasaki_gas.cpp
    native/src/kawasaki_transport.cu
    native/src/cuda_solver.cu
)
target_include_directories(battery_native_core PUBLIC native/include)
target_link_libraries(battery_native_core PUBLIC nlohmann_json::nlohmann_json CUDA::cudart)
target_compile_options(battery_native_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)

add_executable(battery_sim_headless native/tests/headless.cpp)
target_link_libraries(battery_sim_headless PRIVATE battery_native_core)
add_executable(battery_sim_benchmark native/tools/benchmark.cpp)
target_link_libraries(battery_sim_benchmark PRIVATE battery_native_core)
add_executable(battery_sim_reference_probe native/tools/reference_probe.cpp)
target_link_libraries(battery_sim_reference_probe PRIVATE battery_native_core)
add_executable(battery_sim_experiment_probe native/tools/experiment_probe.cpp)
target_link_libraries(battery_sim_experiment_probe PRIVATE battery_native_core)
add_executable(battery_sim_profile_probe native/tools/battery_profile_probe.cpp)
target_link_libraries(battery_sim_profile_probe PRIVATE battery_native_core)
add_executable(battery_sim_failure_probe native/tools/failure_probe.cpp)
target_link_libraries(battery_sim_failure_probe PRIVATE battery_native_core)
add_executable(battery_sim_reaction_probe native/tools/reaction_probe.cpp)
target_link_libraries(battery_sim_reaction_probe PRIVATE battery_native_core)
add_executable(battery_sim_casing_probe native/tools/casing_probe.cpp)
target_link_libraries(battery_sim_casing_probe PRIVATE battery_native_core)
add_executable(battery_sim_record_visual native/tools/record_visual.cpp)
target_link_libraries(battery_sim_record_visual PRIVATE battery_native_core)
add_executable(battery_sim_gas_transport_audit native/tools/gas_transport_audit.cpp)
target_link_libraries(battery_sim_gas_transport_audit PRIVATE battery_native_core)
add_executable(battery_sim_gas_limiter_snapshot native/tools/gas_limiter_snapshot.cu)
target_link_libraries(battery_sim_gas_limiter_snapshot PRIVATE battery_native_core)
add_executable(battery_sim_implicit_mixing_test native/tests/implicit_mixing.cu)
target_link_libraries(battery_sim_implicit_mixing_test PRIVATE battery_native_core)
add_executable(battery_sim_pressure_solver_test native/tests/pressure_solver.cu)
target_link_libraries(battery_sim_pressure_solver_test PRIVATE battery_native_core)
add_executable(battery_sim_advective_surface_test native/tests/advective_surface.cu)
target_link_libraries(battery_sim_advective_surface_test PRIVATE battery_native_core)

# Isolated transport experiments: never selected by the production solver.
add_library(battery_gas_experiment STATIC native/src/gas_experiment.cpp native/src/gas_experiment.cu)
target_link_libraries(battery_gas_experiment PUBLIC battery_native_core)
target_compile_options(battery_gas_experiment PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)
add_executable(battery_sim_gas_patch_benchmark native/tools/gas_patch_benchmark.cpp)
target_link_libraries(battery_sim_gas_patch_benchmark PRIVATE battery_gas_experiment)

add_library(battery_kawasaki_gas STATIC native/src/kawasaki_gas.cu)
target_link_libraries(battery_kawasaki_gas PUBLIC battery_native_core)
target_compile_options(battery_kawasaki_gas PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)
add_executable(battery_sim_kawasaki_gas_audit native/tools/kawasaki_gas_audit.cpp native/tools/kawasaki_tail_audit.cpp)
target_link_libraries(battery_sim_kawasaki_gas_audit PRIVATE battery_kawasaki_gas battery_gas_experiment)

option(BATTERY_SIM_BUILD_VIEWER "Build the realtime Dear ImGui viewer" ON)
if(BATTERY_SIM_BUILD_VIEWER)
    FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG        3.4
        GIT_SHALLOW    TRUE
    )
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(glfw)

    FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG        v1.91.8
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(imgui)

    add_library(imgui_lib STATIC
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_demo.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
    )
    target_include_directories(imgui_lib PUBLIC
        ${imgui_SOURCE_DIR}
        ${imgui_SOURCE_DIR}/backends
    )
    target_link_libraries(imgui_lib PUBLIC glfw)

    FetchContent_Declare(
        implot
        GIT_REPOSITORY https://github.com/epezent/implot.git
        GIT_TAG        v0.16
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(implot)

    add_library(implot_lib STATIC
        ${implot_SOURCE_DIR}/implot.cpp
        ${implot_SOURCE_DIR}/implot_items.cpp
    )
    target_include_directories(implot_lib PUBLIC ${implot_SOURCE_DIR})
    target_link_libraries(implot_lib PUBLIC imgui_lib)

    FetchContent_Declare(
        glm
        GIT_REPOSITORY https://github.com/g-truc/glm.git
        GIT_TAG        1.0.1
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(glm)

    find_package(OpenGL REQUIRED)
    add_executable(battery_sim_calibration_monitor native/tools/calibration_monitor.cpp)
    target_link_libraries(battery_sim_calibration_monitor PRIVATE implot_lib OpenGL::GL nlohmann_json::nlohmann_json)
    add_library(battery_cuda_visual STATIC native/src/cuda_gl_field.cu native/src/gas_display.cpp native/src/visual_geometry.cpp)
    target_link_libraries(battery_cuda_visual PUBLIC battery_native_core OpenGL::GL)
    add_executable(battery_sim_viewer
        native/src/main.cpp
        native/src/renderer.cpp
        native/src/visual_replay.cpp
    )
    target_include_directories(battery_sim_viewer PRIVATE native/include ${glm_SOURCE_DIR})
    target_compile_definitions(battery_sim_viewer PRIVATE GL_GLEXT_PROTOTYPES GLM_ENABLE_EXPERIMENTAL)
    target_compile_options(battery_sim_viewer PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(battery_sim_viewer PRIVATE
        battery_native_core
        battery_cuda_visual
        implot_lib
        OpenGL::GL
    )
    add_executable(battery_sim_visual_replay_test native/tests/visual_replay.cpp
        native/src/visual_replay.cpp native/src/visual_geometry.cpp)
    target_link_libraries(battery_sim_visual_replay_test PRIVATE battery_native_core)
    add_executable(battery_sim_gas_display_test native/tests/gas_display.cpp native/src/gas_display.cpp native/src/visual_geometry.cpp)
    target_link_libraries(battery_sim_gas_display_test PRIVATE battery_native_core)
    add_executable(battery_sim_visual_cylinders_test native/tests/visual_cylinders.cpp
        native/src/visual_geometry.cpp native/src/gas_display.cpp)
    target_link_libraries(battery_sim_visual_cylinders_test PRIVATE battery_native_core)
    add_executable(battery_sim_render_benchmark native/tools/render_benchmark.cpp native/src/renderer.cpp)
    target_include_directories(battery_sim_render_benchmark PRIVATE native/include ${glm_SOURCE_DIR})
    target_compile_definitions(battery_sim_render_benchmark PRIVATE GL_GLEXT_PROTOTYPES GLM_ENABLE_EXPERIMENTAL)
    target_link_libraries(battery_sim_render_benchmark PRIVATE battery_cuda_visual glfw OpenGL::GL)
    add_executable(battery_sim_cuda_visual_test native/tests/cuda_visual.cpp)
    target_compile_definitions(battery_sim_cuda_visual_test PRIVATE GL_GLEXT_PROTOTYPES)
    target_link_libraries(battery_sim_cuda_visual_test PRIVATE battery_cuda_visual glfw OpenGL::GL)
    add_executable(battery_sim_render_continuity_test native/tests/render_continuity.cpp native/src/renderer.cpp)
    target_include_directories(battery_sim_render_continuity_test PRIVATE native/include ${glm_SOURCE_DIR})
    target_compile_definitions(battery_sim_render_continuity_test PRIVATE GL_GLEXT_PROTOTYPES GLM_ENABLE_EXPERIMENTAL)
    target_link_libraries(battery_sim_render_continuity_test PRIVATE battery_cuda_visual glfw OpenGL::GL)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(battery_sim_heating_protocol_test native/tests/heating_protocol.cpp)
    target_link_libraries(battery_sim_heating_protocol_test PRIVATE battery_native_core)
    add_test(NAME cuda_heating_protocol COMMAND battery_sim_heating_protocol_test)
    add_executable(battery_sim_pressure_controls_test native/tests/pressure_controls.cpp)
    target_link_libraries(battery_sim_pressure_controls_test PRIVATE battery_native_core)
    add_test(NAME pressure_iteration_controls COMMAND battery_sim_pressure_controls_test
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/reaction-chain-demo.json)
    add_test(NAME cuda_implicit_gas_surface_mixing COMMAND battery_sim_implicit_mixing_test)
    add_test(NAME cuda_pressure_solver COMMAND battery_sim_pressure_solver_test)
    add_test(NAME cuda_advective_surface COMMAND battery_sim_advective_surface_test)
    add_executable(battery_sim_reaction_step_bounds_test native/tests/reaction_step_bounds.cpp)
    target_include_directories(battery_sim_reaction_step_bounds_test PRIVATE native/include)
    add_test(NAME reaction_step_bounds COMMAND battery_sim_reaction_step_bounds_test)
    add_executable(battery_sim_gas_local_reactor_test native/tests/gas_local_reactor.cu)
    target_link_libraries(battery_sim_gas_local_reactor_test PRIVATE battery_native_core)
    add_test(NAME cuda_gas_local_reactor COMMAND battery_sim_gas_local_reactor_test)
    add_executable(battery_sim_reactive_transit_test native/tests/reactive_transit.cu)
    target_link_libraries(battery_sim_reactive_transit_test PRIVATE battery_native_core)
    add_test(NAME cuda_reactive_transit COMMAND battery_sim_reactive_transit_test)
    add_executable(battery_sim_gas_ambient_test native/tests/gas_ambient.cu)
    target_link_libraries(battery_sim_gas_ambient_test PRIVATE battery_native_core)
    add_test(NAME cuda_gas_ambient COMMAND battery_sim_gas_ambient_test)
    add_executable(battery_sim_radiation_test native/tests/radiation.cu)
    target_link_libraries(battery_sim_radiation_test PRIVATE battery_native_core)
    add_test(NAME cuda_radiation COMMAND battery_sim_radiation_test)
    add_executable(battery_sim_gas_film_audit native/tests/gas_film_audit.cpp)
    target_link_libraries(battery_sim_gas_film_audit PRIVATE battery_native_core)
    add_executable(battery_sim_volatile_phase_test native/tests/volatile_phase.cu)
    target_link_libraries(battery_sim_volatile_phase_test PRIVATE battery_native_core)
    add_test(NAME cuda_volatile_phase COMMAND battery_sim_volatile_phase_test)
    add_executable(battery_sim_kawasaki_gas_test native/tests/kawasaki_gas.cpp)
    target_link_libraries(battery_sim_kawasaki_gas_test PRIVATE battery_kawasaki_gas)
    add_test(NAME cuda_kawasaki_gas COMMAND battery_sim_kawasaki_gas_test)
    add_executable(battery_sim_batched_gas_test native/tests/batched_gas.cu)
    target_link_libraries(battery_sim_batched_gas_test PRIVATE battery_kawasaki_gas)
    add_test(NAME cuda_batched_gas COMMAND battery_sim_batched_gas_test)
    add_executable(battery_sim_gas_combustion_test native/tests/gas_combustion.cu)
    target_link_libraries(battery_sim_gas_combustion_test PRIVATE battery_native_core)
    add_test(NAME cuda_gas_combustion COMMAND battery_sim_gas_combustion_test)
    add_executable(battery_sim_implicit_gas_test native/tests/implicit_gas.cpp)
    target_link_libraries(battery_sim_implicit_gas_test PRIVATE battery_native_core)
    add_test(NAME cpu_implicit_gas COMMAND battery_sim_implicit_gas_test)
    add_executable(battery_sim_gas_experiment_test native/tests/gas_experiment.cpp)
    target_link_libraries(battery_sim_gas_experiment_test PRIVATE battery_gas_experiment)
    add_test(NAME cpu_gas_experiment COMMAND battery_sim_gas_experiment_test)
    add_test(NAME cuda_gas_experiment COMMAND battery_sim_gas_experiment_test --cuda)
    add_executable(battery_sim_split_transport_test native/tests/split_transport.cpp)
    target_link_libraries(battery_sim_split_transport_test PRIVATE battery_native_core)
    add_test(NAME cuda_split_transport COMMAND battery_sim_split_transport_test)
    add_test(NAME cuda_kawasaki_split COMMAND battery_sim_split_transport_test --cuda-kawasaki)
    add_test(NAME cuda_batched_split COMMAND battery_sim_split_transport_test --cuda-batched)
    add_test(NAME hybrid_split_transport COMMAND battery_sim_split_transport_test --cpu-local)
    add_test(NAME implicit_split_transport COMMAND battery_sim_split_transport_test --cpu-local --implicit)
    add_executable(battery_sim_aperture_test native/tests/aperture.cpp)
    target_link_libraries(battery_sim_aperture_test PRIVATE battery_native_core)
    add_test(NAME cuda_aperture_connections COMMAND battery_sim_aperture_test)
    add_executable(battery_sim_shell_opening_test native/tests/shell_opening.cpp)
    target_link_libraries(battery_sim_shell_opening_test PRIVATE battery_native_core)
    add_test(NAME cuda_shell_opening COMMAND battery_sim_shell_opening_test)
    add_executable(battery_sim_stochastic_shell_test native/tests/stochastic_shell.cpp)
    target_link_libraries(battery_sim_stochastic_shell_test PRIVATE battery_native_core)
    add_test(NAME cuda_stochastic_shell COMMAND battery_sim_stochastic_shell_test)
    add_executable(battery_sim_area_rupture_test native/tests/area_rupture.cpp)
    target_link_libraries(battery_sim_area_rupture_test PRIVATE battery_native_core)
    add_test(NAME cuda_area_rupture COMMAND battery_sim_area_rupture_test)
    add_executable(battery_sim_leading_aperture_test native/tests/leading_aperture.cpp)
    target_link_libraries(battery_sim_leading_aperture_test PRIVATE battery_native_core)
    add_test(NAME cuda_leading_aperture COMMAND battery_sim_leading_aperture_test)
    add_executable(battery_sim_material_barrier_test native/tests/material_barrier.cpp)
    target_link_libraries(battery_sim_material_barrier_test PRIVATE battery_native_core)
    add_test(NAME material_barrier_geometry COMMAND battery_sim_material_barrier_test --geometry-only)
    add_test(NAME cuda_material_barrier COMMAND battery_sim_material_barrier_test)
    add_test(NAME cuda_coarse_aperture_retry COMMAND battery_sim_leading_aperture_test --coarse-retry)
    add_test(NAME cuda_kawasaki_fracture COMMAND battery_sim_leading_aperture_test --cuda-kawasaki)
    add_test(NAME cuda_batched_fracture COMMAND battery_sim_leading_aperture_test --cuda-batched)
    add_test(NAME hybrid_leading_aperture COMMAND battery_sim_leading_aperture_test --cpu-local)
    add_test(NAME implicit_leading_aperture COMMAND battery_sim_leading_aperture_test --cpu-local --implicit)
    add_executable(battery_sim_mechanism_test native/tests/mechanism_consistency.cpp)
    target_link_libraries(battery_sim_mechanism_test PRIVATE battery_native_core)
    add_test(NAME cuda_mechanism_consistency COMMAND battery_sim_mechanism_test)
    add_executable(battery_sim_transaction_test native/tests/transaction.cpp)
    target_link_libraries(battery_sim_transaction_test PRIVATE battery_native_core)
    add_test(NAME cuda_transaction COMMAND battery_sim_transaction_test)
    add_executable(battery_sim_process_test native/tests/process_network.cpp)
    target_link_libraries(battery_sim_process_test PRIVATE battery_native_core)
    add_test(NAME cuda_process_network COMMAND battery_sim_process_test)
    add_executable(battery_sim_reaction_chain_test native/tests/reaction_chain.cpp)
    target_link_libraries(battery_sim_reaction_chain_test PRIVATE battery_native_core)
    add_test(NAME cuda_reaction_chain COMMAND battery_sim_reaction_chain_test)
    add_executable(battery_sim_kinetic_validation_test native/tests/kinetic_validation.cpp)
    target_link_libraries(battery_sim_kinetic_validation_test PRIVATE battery_native_core)
    add_test(NAME r123_kinetic_validation COMMAND battery_sim_kinetic_validation_test)
    add_executable(battery_sim_joule_test native/tests/joule.cpp)
    target_link_libraries(battery_sim_joule_test PRIVATE battery_native_core)
    add_test(NAME cuda_joule COMMAND battery_sim_joule_test)
    add_executable(battery_sim_shell_test native/tests/shell.cpp)
    target_link_libraries(battery_sim_shell_test PRIVATE battery_native_core)
    add_test(NAME cuda_physical_shell COMMAND battery_sim_shell_test)
    add_executable(battery_sim_insulation_damage_test native/tests/insulation_damage.cpp)
    target_link_libraries(battery_sim_insulation_damage_test PRIVATE battery_native_core)
    add_test(NAME irreversible_insulation_damage COMMAND battery_sim_insulation_damage_test)
    add_executable(battery_sim_insulation_damage_cuda_test native/tests/insulation_damage_cuda.cpp)
    target_link_libraries(battery_sim_insulation_damage_cuda_test PRIVATE battery_native_core)
    add_test(NAME cuda_irreversible_insulation_damage COMMAND battery_sim_insulation_damage_cuda_test)
    add_executable(battery_sim_reaction_test native/tests/reaction.cpp)
    target_link_libraries(battery_sim_reaction_test PRIVATE battery_native_core)
    add_test(NAME cuda_reaction_process COMMAND battery_sim_reaction_test)
    add_executable(battery_sim_failure_test native/tests/failure.cpp)
    target_link_libraries(battery_sim_failure_test PRIVATE battery_native_core)
    add_test(NAME cuda_failure_state COMMAND battery_sim_failure_test)
    add_executable(battery_sim_constitutive_test native/tests/constitutive.cpp)
    target_link_libraries(battery_sim_constitutive_test PRIVATE battery_native_core)
    add_test(NAME cuda_constitutive COMMAND battery_sim_constitutive_test)
    add_executable(battery_sim_core_shell_test native/tests/core_shell.cpp)
    target_link_libraries(battery_sim_core_shell_test PRIVATE battery_native_core)
    add_test(NAME cuda_core_shell COMMAND battery_sim_core_shell_test)
    add_executable(battery_sim_directed_geometry_test native/tests/directed_geometry.cpp)
    target_link_libraries(battery_sim_directed_geometry_test PRIVATE battery_native_core)
    add_test(NAME cuda_directed_geometry COMMAND battery_sim_directed_geometry_test)
    add_executable(battery_sim_state_geometry_test native/tests/state_geometry.cpp)
    target_link_libraries(battery_sim_state_geometry_test PRIVATE battery_native_core)
    add_test(NAME cuda_state_geometry COMMAND battery_sim_state_geometry_test)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME native_reference_fit COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/scripts/calibrate_narada_reference.py
            --probe $<TARGET_FILE:battery_sim_reference_probe> --check)
    endif()
    add_executable(battery_sim_cuda_test native/tests/cuda_thermal.cpp)
    target_link_libraries(battery_sim_cuda_test PRIVATE battery_native_core)
    add_test(NAME cuda_thermal COMMAND battery_sim_cuda_test)
    add_executable(battery_sim_cooling_wall_test native/tests/cooling_wall.cpp)
    target_link_libraries(battery_sim_cooling_wall_test PRIVATE battery_native_core)
    add_test(NAME cuda_cooling_wall COMMAND battery_sim_cooling_wall_test)
    add_executable(battery_sim_multiscale_test native/tests/multiscale.cpp)
    target_link_libraries(battery_sim_multiscale_test PRIVATE battery_native_core)
    add_test(NAME cuda_multiscale COMMAND battery_sim_multiscale_test)
    add_executable(battery_sim_isothermal_test native/tests/isothermal.cpp)
    target_link_libraries(battery_sim_isothermal_test PRIVATE battery_native_core)
    add_executable(battery_sim_playback_test native/tests/playback.cpp)
    target_link_libraries(battery_sim_playback_test PRIVATE battery_native_core)
    target_compile_options(battery_sim_playback_test PRIVATE -Wall -Wextra -Wpedantic)
    add_executable(battery_sim_palette_test native/tests/palette.cpp)
    target_include_directories(battery_sim_palette_test PRIVATE native/include)
    target_compile_options(battery_sim_palette_test PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME native_palette COMMAND battery_sim_palette_test)
    if(TARGET battery_sim_gas_display_test)
        add_test(NAME native_gas_display COMMAND battery_sim_gas_display_test)
    endif()
    add_executable(battery_sim_material_chemistry_test native/tests/material_chemistry.cpp)
    target_include_directories(battery_sim_material_chemistry_test PRIVATE native/include)
    add_test(NAME material_chemistry COMMAND battery_sim_material_chemistry_test)
endif()
