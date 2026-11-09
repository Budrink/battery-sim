cmake_minimum_required(VERSION 3.20)
project(battery_sim_native LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
find_package(CUDAToolkit REQUIRED)

include(FetchContent)
include(CTest)

option(BATTERY_SIM_BUILD_RUNTIME "Build scenario/job adapters and headless applications" ON)
option(BATTERY_SIM_BUILD_VIEWER "Build the optional desktop ImGui UI" ON)
option(BATTERY_SIM_BUILD_TOOLS "Build calibration, recording and research tools" ON)
if(NOT BATTERY_SIM_BUILD_RUNTIME AND
   (BATTERY_SIM_BUILD_VIEWER OR BATTERY_SIM_BUILD_TOOLS OR BUILD_TESTING))
    message(FATAL_ERROR "Core-only build requires VIEWER, TOOLS and BUILD_TESTING=OFF (see preset core)")
endif()

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

# Layers are ordered by dependency: algorithm -> run/data adapters -> applications.
include(cmake/core.cmake)
if(BATTERY_SIM_BUILD_RUNTIME)
    include(cmake/run.cmake)
    include(cmake/job.cmake)
    include(cmake/apps.cmake)
endif()
# CPU-only recording/geometry adapters are useful for tools and tests too.
# A minimal server has no visualization target, source or dependency at all.
if(BATTERY_SIM_BUILD_VIEWER OR BATTERY_SIM_BUILD_TOOLS OR BUILD_TESTING)
    include(cmake/visual_data.cmake)
endif()
if(BATTERY_SIM_BUILD_VIEWER)
    include(cmake/visualization.cmake)
endif()
if(BATTERY_SIM_BUILD_TOOLS OR BUILD_TESTING)
    include(cmake/experiments.cmake)
endif()
if(BATTERY_SIM_BUILD_TOOLS)
    include(cmake/tools.cmake)
endif()
if(BUILD_TESTING)
    include(cmake/tests.cmake)
endif()
