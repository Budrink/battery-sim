# Included only for BATTERY_SIM_BUILD_VIEWER. No graphics discovery in server/core builds.
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
    ${imgui_SOURCE_DIR}/misc/cpp/imgui_stdlib.cpp
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
add_executable(battery_sim_calibration_monitor native/apps/calibration_monitor/main.cpp)
target_link_libraries(battery_sim_calibration_monitor PRIVATE implot_lib OpenGL::GL nlohmann_json::nlohmann_json)
add_library(battery_cuda_visual STATIC native/visualization/src/render/cuda_gl_field.cu)
target_link_libraries(battery_cuda_visual PUBLIC battery_visual_data OpenGL::GL)
add_library(battery_renderer STATIC native/visualization/src/render/renderer.cpp)
target_include_directories(battery_renderer PRIVATE ${glm_SOURCE_DIR})
target_compile_definitions(battery_renderer PRIVATE GL_GLEXT_PROTOTYPES GLM_ENABLE_EXPERIMENTAL)
target_link_libraries(battery_renderer PUBLIC battery_cuda_visual PRIVATE glfw)
add_library(battery_viewer_options STATIC native/ui/src/application/options.cpp)
target_include_directories(battery_viewer_options PUBLIC native/ui/include)
target_link_libraries(battery_viewer_options PUBLIC battery_run)
# Optional application module. The executable is only an entry point.
add_library(battery_editor_document STATIC
    native/ui/src/editor/state/document.cpp
    native/ui/src/editor/state/model_edits.cpp)
target_include_directories(battery_editor_document PUBLIC native/ui/include PRIVATE native/ui/src)
target_link_libraries(battery_editor_document PUBLIC battery_run)
# Window-independent application session; tested without creating GLFW/ImGui.
add_library(battery_viewer_session STATIC native/ui/src/application/viewer_session.cpp)
target_include_directories(battery_viewer_session PUBLIC native/ui/include)
target_link_libraries(battery_viewer_session PUBLIC battery_viewer_options battery_visual_data battery_run)
add_library(battery_ui STATIC
    native/ui/src/application/application.cpp
    native/ui/src/platform/desktop_host.cpp
    native/ui/src/viewer/viewer_window.cpp
    native/ui/src/viewer/widgets.cpp
    native/ui/src/viewer/scene_view.cpp
    native/ui/src/viewer/panels/header_panel.cpp
    native/ui/src/viewer/panels/inspector_panel.cpp
    native/ui/src/viewer/panels/transport_panel.cpp
    native/ui/src/editor/experiment_editor.cpp
    native/ui/src/editor/commands/import.cpp
    native/ui/src/editor/commands/panel_edits.cpp
    native/ui/src/editor/views/parameter_reference.cpp
    native/ui/src/editor/views/file_browser.cpp
    native/ui/src/editor/views/editor_window.cpp
    native/ui/src/editor/views/editor_desktop.cpp
    native/ui/src/editor/panels/assembly_panel.cpp
    native/ui/src/editor/commands/elements.cpp
    native/ui/src/editor/commands/circuit.cpp
    native/ui/src/editor/panels/circuit_panel.cpp
    native/ui/src/editor/preview/preview.cpp
    native/ui/src/editor/panels/case_openings_panel.cpp
    native/ui/src/editor/panels/connection_forms.cpp
    native/ui/src/editor/panels/enclosure_panel.cpp
    native/ui/src/editor/panels/gas_model_panel.cpp
    native/ui/src/editor/panels/geometry_panel.cpp
    native/ui/src/editor/panels/heating_panel.cpp
    native/ui/src/editor/panels/models_panel.cpp
    native/ui/src/editor/panels/run_forms.cpp
    native/ui/src/editor/panels/shell_panel.cpp
    native/ui/src/editor/forms/heater_controller_panel.cpp
    native/ui/src/editor/forms/model_properties.cpp)
add_library(battery::ui ALIAS battery_ui)
target_include_directories(battery_ui PUBLIC native/ui/include PRIVATE native/ui/src ${glm_SOURCE_DIR})
target_compile_definitions(battery_ui PRIVATE GL_GLEXT_PROTOTYPES GLM_ENABLE_EXPERIMENTAL)
target_compile_options(battery_ui PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(battery_ui PUBLIC battery_viewer_options PRIVATE battery_viewer_session battery_renderer implot_lib battery_editor_document)
add_executable(battery_sim_viewer native/apps/viewer/main.cpp)
target_link_libraries(battery_sim_viewer PRIVATE battery::ui)

if(BATTERY_SIM_BUILD_TOOLS)
    add_executable(battery_sim_render_benchmark native/tools/benchmarks/render_benchmark.cpp)
    target_include_directories(battery_sim_render_benchmark PRIVATE ${glm_SOURCE_DIR})
    target_compile_definitions(battery_sim_render_benchmark PRIVATE GL_GLEXT_PROTOTYPES GLM_ENABLE_EXPERIMENTAL)
    target_link_libraries(battery_sim_render_benchmark PRIVATE battery_renderer glfw OpenGL::GL battery_run nlohmann_json::nlohmann_json battery_visual_data)
endif()

