#include "battery_native/renderer.hpp"
#include <GLFW/glfw3.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
constexpr int width=360,height=280;
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
std::vector<unsigned char> capture(battery::SceneRenderer& renderer,int divisions,bool gradient=false,float cut=1) {
    battery::Scenario scenario;
    battery::CellConfig cell;cell.id="one-battery";cell.dimensions_m={.174,.072,.207};
    cell.thermal_divisions={divisions,divisions,divisions};cell.initial_temperature_k=550;
    if(gradient)for(int z=0;z<divisions;++z)for(int y=0;y<divisions;++y)for(int x=0;x<divisions;++x)
        cell.initial_temperature_field_k.push_back(x==0?350:850);
    scenario.cells.push_back(cell);battery::Simulation simulation(scenario);
    battery::OrbitCamera camera;camera.yaw=-.8;camera.pitch=.5;camera.distance=1.4;
    battery::VisualSettings visual;visual.cutaway_fraction=cut;visual.temperature_max_k=900;
    const auto texture=renderer.render(width,height,scenario,simulation,camera,visual,-1,{});
    std::vector<unsigned char> pixels(width*height*4);
    glBindTexture(GL_TEXTURE_2D,texture);glGetTexImage(GL_TEXTURE_2D,0,GL_RGBA,GL_UNSIGNED_BYTE,pixels.data());
    require(glGetError()==GL_NO_ERROR,"OpenGL error in continuous renderer");return pixels;
}
}
int main(){try {
    require(glfwInit(),"GLFW unavailable");glfwWindowHint(GLFW_VISIBLE,GLFW_FALSE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
    auto* window=glfwCreateWindow(width,height,"continuous field regression",nullptr,nullptr);
    require(window,"GL context unavailable");glfwMakeContextCurrent(window);
    battery::SceneRenderer renderer;require(renderer.init(),"Renderer initialization failed");
    auto one=capture(renderer,1),eight=capture(renderer,2),core_shell=capture(renderer,3),many=capture(renderer,4);
    int lit=0;for(std::size_t i=0;i<one.size();i+=4)lit+=one[i]>25;
    require(lit>1000,"Empty render must not pass the continuity test");
    for(std::size_t i=0;i<one.size();++i) {
        require(std::abs(int(one[i])-int(eight[i]))<=1,"Eight zones changed the battery silhouette/opacity/color");
        require(std::abs(int(one[i])-int(core_shell[i]))<=1,"27 zones changed the battery silhouette/opacity/color");
        require(std::abs(int(one[i])-int(many[i]))<=1,"64 zones changed the battery silhouette/opacity/color");
    }
    auto gradient=capture(renderer,2,true),cut=capture(renderer,2,true,.5);
    int changed=0,cut_changed=0;
    for(std::size_t i=0;i<one.size();i+=4) {
        changed+=std::abs(int(gradient[i])-int(eight[i]))>10;
        cut_changed+=std::abs(int(gradient[i])-int(cut[i]))>10;
    }
    require(changed>1000,"Nonuniform field was replaced with one macro temperature");
    require(cut_changed>1000,"Cutaway failed to expose the continuous inner field");
    renderer.destroy();glfwDestroyWindow(window);glfwTerminate();
    std::cout<<"Continuous macro field: identical uniform 1/8/27/64-zone render; gradient and cutaway retained\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
