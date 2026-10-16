#include "battery_native/renderer.hpp"
#include <GLFW/glfw3.h>
#include <nlohmann/json.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

int main(int argc,char** argv) {try {
    if(argc<2)throw std::runtime_error("usage: battery_sim_render_benchmark SCENARIO [SCALE] [OUTPUT]");
    auto scenario=battery::load_scenario(argv[1]);int scale=argc>2?std::stoi(argv[2]):1;
    if(scale<1||scale>4)throw std::runtime_error("Scale must be 1..4");
    battery::refine_environment_mesh(scenario,scale);
    battery::Simulation simulation(scenario);
    if(!glfwInit())throw std::runtime_error("GLFW unavailable");
    glfwWindowHint(GLFW_VISIBLE,GLFW_FALSE);glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
    auto window=glfwCreateWindow(1280,720,"battery-sim render benchmark",nullptr,nullptr);
    if(!window)throw std::runtime_error("OpenGL context unavailable");
    glfwMakeContextCurrent(window);glfwSwapInterval(0);
    battery::SceneRenderer renderer;if(!renderer.init())throw std::runtime_error("Renderer initialization failed");
    battery::OrbitCamera camera;camera.pitch=0.92f;camera.distance=1.35f;
    battery::VisualSettings visual;visual.field_opacity=0.65f;visual.temperature_max_k=1050;
    using Clock=std::chrono::steady_clock;
    double compute_ms=0,draw_ms=0;constexpr int frames=60,warmup=10,steps_per_frame=16;
    for(int frame=-warmup;frame<frames;++frame) {
        auto begin=Clock::now();for(int i=0;i<steps_per_frame;++i)simulation.step();auto computed=Clock::now();
        renderer.render(1280,720,scenario,simulation,camera,visual,0,{});glFinish();
        auto drawn=Clock::now();
        if(frame>=0) {compute_ms+=std::chrono::duration<double,std::milli>(computed-begin).count();draw_ms+=std::chrono::duration<double,std::milli>(drawn-computed).count();}
    }
    nlohmann::json report={{"device",simulation.device_name()},
        {"gl_renderer",reinterpret_cast<const char*>(glGetString(GL_RENDERER))},{"scale",scale},
        {"volumes",simulation.thermal_mesh().volumes.size()},{"frames",frames},{"steps_per_frame",steps_per_frame},
        {"width",1280},{"height",720},{"compute_ms_per_frame",compute_ms/frames},
        {"scene_ms_per_frame",draw_ms/frames},{"combined_fps_no_ui_no_vsync",frames*1000/(compute_ms+draw_ms)},
        {"relative_energy_residual",std::abs(simulation.energy_balance_residual_j())/simulation.initial_stored_energy_j()}};
    std::cout<<report.dump(2)<<'\n';
    if(argc>3) {std::ofstream out(argv[3]);out<<report.dump(2)<<'\n';if(!out)throw std::runtime_error("Cannot write report");}
    renderer.destroy();glfwDestroyWindow(window);glfwTerminate();
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
