# Only defaults: explicit -D values or a CMakeUserPresets.json take precedence.
# CUDA needs the real host compiler, not a broken launcher inherited from a shell.
if(NOT CMAKE_CXX_COMPILER AND NOT DEFINED ENV{CXX})
    find_program(BATTERY_GXX NAMES g++ REQUIRED)
    set(CMAKE_CXX_COMPILER "${BATTERY_GXX}" CACHE FILEPATH "Host C++ compiler")
endif()
if(NOT CMAKE_CUDA_HOST_COMPILER AND NOT DEFINED ENV{CUDAHOSTCXX})
    find_program(BATTERY_CUDA_GXX NAMES g++ REQUIRED)
    set(CMAKE_CUDA_HOST_COMPILER "${BATTERY_CUDA_GXX}" CACHE FILEPATH "CUDA host compiler")
endif()
if(NOT CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
    find_program(BATTERY_NVCC NAMES nvcc
        HINTS "$ENV{CUDA_HOME}/bin" "$ENV{CUDA_PATH}/bin" /usr/local/cuda/bin REQUIRED)
    set(CMAKE_CUDA_COMPILER "${BATTERY_NVCC}" CACHE FILEPATH "CUDA compiler")
endif()
