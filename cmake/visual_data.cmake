# Read-only adapters for live states and recorded trajectories; no window needed.
add_library(battery_visual_data STATIC
    native/visualization/src/fields/gas_display.cpp
    native/visualization/src/data/visual_geometry.cpp
    native/visualization/src/data/gas_case_clipping.cpp
    native/visualization/src/data/visual_replay.cpp
)
target_include_directories(battery_visual_data PUBLIC native/visualization/include)
target_link_libraries(battery_visual_data PUBLIC battery_native_core nlohmann_json::nlohmann_json)
# File import resolves experiment contracts through the host run adapter.
# Neither the numerical core nor a headless server gains any UI dependency.
target_link_libraries(battery_visual_data PRIVATE battery_run)
