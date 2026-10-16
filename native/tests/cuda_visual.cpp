#include "battery_native/cuda_gl_field.hpp"
#include "battery_native/simulation.hpp"
#include <GLFW/glfw3.h>
#include <cmath>
#include <iostream>
#include <stdexcept>

int main() {try {
    using namespace battery;
    Scenario s;s.settings={1,0.01,0.1,0.2};CellConfig c;c.id="palette-test";
    c.dimensions_m={1,1,1};c.thermal_divisions={4,1,1};c.initial_temperature_field_k={250,300,500,1200};
    c.aging_fraction=0.25;c.initial_damage=0.4;c.runaway_energy_j=0;c.runaway_duration_s=1;
    s.cells.push_back(c);s.primary_failures={{0,"palette-test",0,3}};
    Simulation sim(s);sim.step();
    if(!glfwInit())throw std::runtime_error("GLFW init");glfwWindowHint(GLFW_VISIBLE,GLFW_FALSE);
    auto window=glfwCreateWindow(64,64,"CUDA color test",nullptr,nullptr);
    if(!window)throw std::runtime_error("GL context");glfwMakeContextCurrent(window);
    GLuint buffer=0;glGenBuffers(1,&buffer);glBindBuffer(GL_ARRAY_BUFFER,buffer);
    glBufferData(GL_ARRAY_BUFFER,4*4*sizeof(float),nullptr,GL_DYNAMIC_DRAW);
    CudaGlField interop;interop.attach(buffer,{0,1,2,3},{0.25f,0.25f,0.25f,0.25f});
    for(auto layer:{FieldLayer::Temperature,FieldLayer::Degradation}) {
        interop.update(sim.cuda_field(),layer,300,1000,0.65f);
        float colors[16]{};glGetBufferSubData(GL_ARRAY_BUFFER,0,sizeof(colors),colors);
        for(int i=0;i<4;++i) {
            const auto& state=sim.thermal_states()[i];
            float value=layer==FieldLayer::Temperature?static_cast<float>((state.temperature_k-300)/700):static_cast<float>(1-0.75*(1-state.damage));
            auto expected=field_rgb(layer,value,state.mode==CellMode::Spent);
            for(int ch=0;ch<3;++ch)if(std::abs(expected[ch]-colors[4*i+ch])>2e-6)throw std::runtime_error("GPU palette differs from CPU legend");
            if(std::abs(colors[4*i+3]-0.65f)>1e-7)throw std::runtime_error("GPU alpha mismatch");
        }
    }
    interop.update_samples(sim.cuda_field());
    float samples[16]{};glGetBufferSubData(GL_ARRAY_BUFFER,0,sizeof(samples),samples);
    for(int i=0;i<4;++i) {
        const auto& state=sim.thermal_states()[i];
        if(std::abs(samples[4*i]-state.temperature_k)>1e-3)throw std::runtime_error("Volume sample temperature mismatch");
        if(std::abs(samples[4*i+1]-(1-.75*(1-state.damage)))>1e-6)throw std::runtime_error("Volume sample degradation mismatch");
        if(samples[4*i+2]!=(state.mode==CellMode::Spent?1.f:0.f))throw std::runtime_error("Volume sample destruction mismatch");
    }
    interop.detach();glDeleteBuffers(1,&buffer);glfwDestroyWindow(window);glfwTerminate();
    std::cout<<"CUDA/OpenGL colors match CPU palettes, including destroyed volume: PASS\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
