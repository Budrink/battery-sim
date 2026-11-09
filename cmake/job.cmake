# A process contract around the solver; no service/HTTP or renderer dependency.
# Bind checkpoints to source AND toolchain/configuration, including dirty worktrees.
file(GLOB_RECURSE BATTERY_JOB_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/native/core/*" "${CMAKE_SOURCE_DIR}/native/run/*"
    "${CMAKE_SOURCE_DIR}/native/job/*" "${CMAKE_SOURCE_DIR}/native/apps/worker/*")
list(APPEND BATTERY_JOB_SOURCES "${CMAKE_SOURCE_DIR}/CMakeLists.txt"
    "${CMAKE_SOURCE_DIR}/cmake/core.cmake" "${CMAKE_SOURCE_DIR}/cmake/run.cmake"
    "${CMAKE_SOURCE_DIR}/cmake/job.cmake")
list(APPEND BATTERY_JOB_SOURCES "${CMAKE_SOURCE_DIR}/cmake/apps.cmake")
set(BATTERY_JOB_ID_INPUT "${CMAKE_CXX_COMPILER_ID}|${CMAKE_CXX_COMPILER_VERSION}|${CMAKE_CUDA_COMPILER_VERSION}|${CMAKE_CUDA_ARCHITECTURES}|${CMAKE_BUILD_TYPE}|${CMAKE_CXX_FLAGS}|${CMAKE_CUDA_FLAGS}")
string(TOUPPER "${CMAKE_BUILD_TYPE}" BATTERY_JOB_CONFIG)
string(APPEND BATTERY_JOB_ID_INPUT "|${CMAKE_CXX_FLAGS_${BATTERY_JOB_CONFIG}}|${CMAKE_CUDA_FLAGS_${BATTERY_JOB_CONFIG}}")
string(APPEND BATTERY_JOB_ID_INPUT "|${CMAKE_CUDA_ARCHITECTURES_NATIVE}|${CMAKE_SYSTEM_NAME}|${CMAKE_SYSTEM_PROCESSOR}|${CMAKE_SIZEOF_VOID_P}")
foreach(source IN LISTS BATTERY_JOB_SOURCES)
    file(SHA256 "${source}" source_hash)
    file(RELATIVE_PATH relative_source "${CMAKE_SOURCE_DIR}" "${source}")
    string(APPEND BATTERY_JOB_ID_INPUT "|${relative_source}:${source_hash}")
endforeach()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${BATTERY_JOB_SOURCES})
string(SHA256 BATTERY_JOB_BUILD_ID "${BATTERY_JOB_ID_INPUT}")
configure_file("${CMAKE_SOURCE_DIR}/native/job/build.hpp.in"
    "${CMAKE_BINARY_DIR}/generated/battery_job_build.hpp" @ONLY)
add_library(battery_job STATIC native/job/src/contract.cpp native/job/src/artifacts.cpp native/job/src/runner.cpp)
add_library(battery::job ALIAS battery_job)
target_include_directories(battery_job PUBLIC native/job/include PRIVATE "${CMAKE_BINARY_DIR}/generated")
target_link_libraries(battery_job PUBLIC battery_run nlohmann_json::nlohmann_json)
target_compile_options(battery_job PRIVATE -Wall -Wextra -Wpedantic)
add_executable(battery_sim_worker native/apps/worker/main.cpp)
target_link_libraries(battery_sim_worker PRIVATE battery_job)
