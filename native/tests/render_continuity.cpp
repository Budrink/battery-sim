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
std::vector<unsigned char> capture(battery::SceneRenderer& renderer,int divisions,bool gradient=false,float cut=1,float surface=0) {
    battery::Scenario scenario;
    battery::CellConfig cell;cell.id="one-battery";cell.dimensions_m={.174,.072,.207};
    cell.thermal_divisions={divisions,divisions,divisions};cell.initial_temperature_k=550;
    if(gradient)for(int z=0;z<divisions;++z)for(int y=0;y<divisions;++y)for(int x=0;x<divisions;++x)
        cell.initial_temperature_field_k.push_back(x==0?350:850);
    scenario.cells.push_back(cell);battery::Simulation simulation(scenario);
    battery::OrbitCamera camera;camera.yaw=-.8;camera.pitch=.5;camera.distance=1.4;
    battery::VisualSettings visual;visual.cutaway_fraction=cut;visual.temperature_max_k=900;
    visual.battery_surface_opacity=surface;
    const auto texture=renderer.render(width,height,scenario,simulation,camera,visual,-1,{});
    std::vector<unsigned char> pixels(width*height*4);
    glBindTexture(GL_TEXTURE_2D,texture);glGetTexImage(GL_TEXTURE_2D,0,GL_RGBA,GL_UNSIGNED_BYTE,pixels.data());
    require(glGetError()==GL_NO_ERROR,"OpenGL error in continuous renderer");return pixels;
}
void gas_regression(battery::SceneRenderer& renderer) {
    using namespace battery;
    Scenario s;CellConfig cell;cell.id="battery";cell.dimensions_m={.174,.072,.207};s.cells.push_back(cell);
    const auto mesh=build_thermal_mesh(s);
    s.gas_compartments.resize(2);
    GasChannel channel;channel.from=0;channel.to=1;channel.area_m2=.0001;s.gas_channels.push_back(channel);
    VisualGeometry geometry;
    GasVisualRegion a;a.compartment=0;a.center={-.05,0,.15};a.size={.10,.10,.09};
    auto b=a;b.compartment=1;b.center.x=.05;geometry.gas={a,b};
    VisualFrame state;state.cells.resize(1);state.cells[0].temperature_k=550;
    state.samples.assign(mesh.volumes.size(),{550,0,0,0});
    state.process.gas.resize(2);state.process.gas[0].temperature_k=850;state.process.gas[1].temperature_k=400;
    state.process.channels.resize(1);auto& flow=state.process.channels[0];
    flow.effective_area_m2=.0001;flow.mass_flow_kg_s=.0001;flow.transferred_kg=.0000125;
    OrbitCamera camera;camera.yaw=-.8;camera.pitch=.5;camera.distance=1.4;
    VisualSettings visual;visual.battery_surface_opacity=.9;visual.gas_opacity=.8;
    visual.show_far_gas=true;visual.temperature_max_k=900;
    auto image=[&] {
        const auto texture=renderer.render(width,height,s,mesh,state,geometry,camera,visual,-1);
        std::vector<unsigned char> pixels(width*height*4);
        glBindTexture(GL_TEXTURE_2D,texture);glGetTexImage(GL_TEXTURE_2D,0,GL_RGBA,GL_UNSIGNED_BYTE,pixels.data());
        require(glGetError()==GL_NO_ERROR,"OpenGL error in gas depth/cloud/puff passes");return pixels;
    };
    const auto gas=image(),pause=image();require(gas==pause,"Gas changes during paused redraw");
    flow.transferred_kg+=.0001;const auto moved=image();
    visual.show_gas=false;const auto hidden=image();
    int gas_pixels=0,moved_pixels=0;
    for(std::size_t i=0;i<gas.size();i+=4) {
        gas_pixels+=std::abs(int(gas[i])-int(hidden[i]))>5;
        moved_pixels+=std::abs(int(gas[i])-int(moved[i]))>2;
    }
    require(gas_pixels>100,"Volumetric gas pass is invisible");
    require(moved_pixels>50,"Accepted mass does not move rendered gas");
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
    auto dense=capture(renderer,1,false,1,.9f),dense_many=capture(renderer,4,false,1,.9f);
    for(std::size_t i=0;i<dense.size();++i)
        require(std::abs(int(dense[i])-int(dense_many[i]))<=1,"Dense surface exposed numerical divisions");
    auto dense_gradient=capture(renderer,2,true,1,.9f);
    int dense_changed=0;
    for(std::size_t i=0;i<dense.size();i+=4)dense_changed+=std::abs(int(dense_gradient[i])-int(dense[i]))>10;
    require(dense_changed>1000,"Dense surface lost actual spatial thermal field");
    gas_regression(renderer);
    renderer.destroy();glfwDestroyWindow(window);glfwTerminate();
    std::cout<<"Continuous macro field: identical uniform 1/8/27/64-zone render; gradient/cutaway and moving gas verified\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
