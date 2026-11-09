#include "battery_native/render/renderer.hpp"
#include "battery_native/scenario_document.hpp"
#include <nlohmann/json.hpp>
#include <GLFW/glfw3.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <cstdlib>
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
void shell_damage_regression(battery::SceneRenderer& renderer) {
    using namespace battery;
    Scenario s;CellConfig c;c.id="B";c.dimensions_m={.04,.04,.08};c.heat_capacity_j_per_k=100;s.cells={c};
    ShellConfig shell;shell.contents_id="B";shell.material_id="steel";shell.thickness_m=.001;
    shell.density_kg_m3=7800;shell.specific_heat_j_kg_k=500;shell.contents_contact_h=100;shell.seam_contact_h=100;
    add_physical_shell(s,shell);Simulation sim(s);const auto& mesh=sim.thermal_mesh();
    auto frame=capture_visual_frame(sim);frame.device_samples=nullptr;VisualGeometry geometry;
    frame.samples.assign(mesh.volumes.size(),{298.15f,0.f,0.f,0.f});
    GasChannel diagnostic;diagnostic.shell_object=6;s.gas_channels.push_back(diagnostic);frame.process.channels.resize(1);
    OrbitCamera camera;camera.yaw=-.8;camera.pitch=.5;camera.distance=1.4;
    VisualSettings visual;visual.show_gas=false;visual.show_ruptures=false;visual.battery_surface_opacity=.9;
    visual.show_damage=true;
    auto image=[&]{auto texture=renderer.render(width,height,s,mesh,frame,geometry,camera,visual,-1);
        std::vector<unsigned char> pixels(width*height*4);glBindTexture(GL_TEXTURE_2D,texture);
        glGetTexImage(GL_TEXTURE_2D,0,GL_RGBA,GL_UNSIGNED_BYTE,pixels.data());return pixels;};
    const auto before=image();frame.process.channels[0].max_thermal_damage=1.;
    require(before==image(),"One shell channel paints damage onto the entire battery contents silhouette");
    geometry.hidden_planar_shells.assign(s.cells.size(),false);
    geometry.hidden_planar_shells[6]=true;
    const auto sides_before=image();
    for(int i=mesh.object_offsets[6];i<mesh.object_offsets[7];++i)frame.samples[i][1]=.9f;
    require(sides_before==image(),"Lid damage changes the rendering of other faces");
    geometry.hidden_planar_shells[6]=false;
    for(int i=mesh.object_offsets[6];i<mesh.object_offsets[7];++i)frame.samples[i][1]=.9f;
    const auto damaged=image();int changed=0;
    for(size_t i=0;i<before.size();i+=4)changed+=std::abs(int(before[i])-int(damaged[i]))+std::abs(int(before[i+1])-int(damaged[i+1]))+std::abs(int(before[i+2])-int(damaged[i+2]))>3;
    require(changed>25,"Physical plate damage is not visible on the plate");
    const auto shell_image = image();
    frame.cells[0].damage=1.;frame.cells[0].mode=CellMode::Spent;
    for(int i=mesh.object_offsets[0];i<mesh.object_offsets[1];++i) {
        frame.samples[i][1]=1.f;frame.samples[i][2]=1.f;
    }
    require(shell_image==image(),"Contents degradation/spent state repaints the battery");
    // A rupture is a soft translucent mark: visible, stable, not opaque black.
    auto& channel=s.gas_channels[0];channel.rupture=true;
    channel.port_id="B:case:+z:outer";
    channel.shell_thermal_volume=(mesh.object_offsets[7]-mesh.object_offsets[6])/2;
    auto& aperture=frame.process.channels[0];aperture.open=true;
    aperture.effective_area_m2=.00005;aperture.aperture_radius_m=.004;
    aperture.opening_fraction=.2;
    visual.show_ruptures=true;
    const auto opened=image();int black=0,marked=0;
    for(size_t i=0;i<opened.size();i+=4)
    {   black+=opened[i]<3&&opened[i+1]<3&&opened[i+2]<3&&
               (shell_image[i]>15||shell_image[i+1]>15||shell_image[i+2]>15);
        marked+=std::abs(int(opened[i])-int(shell_image[i]))+
                std::abs(int(opened[i+1])-int(shell_image[i+1]))+
                std::abs(int(opened[i+2])-int(shell_image[i+2]))>3;
    }
    require(black==0,"Rupture paints opaque black pixels");
    require(marked>10,"Translucent shell rupture is invisible");
    require(opened==image(),"Rupture boundary flickers while paused");

    // Cylinder proxies hide only planar solids, NEVER their physical rupture
    // or valve state. Exercise the real renderer, not just projection helpers.
    geometry.solid_proxies=s.cells;
    geometry.solid_proxies[0].shape=ShapeKind::Cylinder;
    geometry.solid_proxies[0].dimensions_m={.02,.08,0};
    for(std::size_t i=1;i<s.cells.size();++i)geometry.hidden_planar_shells[i]=true;
    for(const auto& face:std::vector<std::pair<int,const char*>>{{6,"B:case:+z:outer"},{2,"B:case:+x:outer"}}) {
        channel.shell_object=face.first;channel.port_id=face.second;
        channel.shell_thermal_volume=(mesh.object_offsets[face.first+1]-mesh.object_offsets[face.first])/2;
        aperture.effective_area_m2=.0002;aperture.opening_fraction=.7;
        visual.show_ruptures=false;const auto intact=image();
        visual.show_ruptures=true;const auto torn=image();
        int pixels=0;
        for(std::size_t i=0;i<torn.size();i+=4)
            pixels+=std::abs(int(torn[i])-int(intact[i]))+std::abs(int(torn[i+1])-int(intact[i+1]))+
                    std::abs(int(torn[i+2])-int(intact[i+2]))>3;
        require(pixels>10,"Cylinder rupture invisible on lid or curved side");
        require(torn==image(),"Cylinder rupture flickers while paused");
    }
    channel.shell_object=6;channel.port_id="B:case:+z:outer";
    channel.opening.valve_radius_m=.005;aperture.effective_area_m2=0;
    visual.show_ruptures=false;const auto no_ring=image();
    visual.show_ruptures=true;require(no_ring!=image(),"Closed cylinder valve is hidden");
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
    state.process.gas[0].mass_kg=state.process.gas[1].mass_kg=.001;
    state.process.channels.resize(1);auto& flow=state.process.channels[0];
    flow.effective_area_m2=.0001;flow.mass_flow_kg_s=.0001;flow.transferred_kg=.0000125;
    OrbitCamera camera;camera.yaw=-.8;camera.pitch=.5;camera.distance=1.4;
    VisualSettings visual;visual.battery_surface_opacity=.9;visual.gas_opacity=.8;
    visual.show_far_gas=true;visual.temperature_max_k=900;
    auto image=[&] {
        const auto texture=renderer.render(width,height,s,mesh,state,geometry,camera,visual,-1);
        std::vector<unsigned char> pixels(width*height*4);
        glBindTexture(GL_TEXTURE_2D,texture);glGetTexImage(GL_TEXTURE_2D,0,GL_RGBA,GL_UNSIGNED_BYTE,pixels.data());
        require(glGetError()==GL_NO_ERROR,"OpenGL error in volumetric gas passes");return pixels;
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
    visual.show_gas=true;visual.show_flux=false;
    state.process.gas[0].temperature_k=state.process.gas[1].temperature_k=298.15;
    visual.hot_gas_only=false;const auto cold=image();
    visual.hot_gas_only=true;const auto filtered=image();
    int cold_pixels=0;
    for(std::size_t i=0;i<cold.size();i+=4)
        cold_pixels+=std::abs(int(cold[i])-int(filtered[i]))+std::abs(int(cold[i+1])-int(filtered[i+1]))+std::abs(int(cold[i+2])-int(filtered[i+2]))>2;
    require(cold_pixels>50,"Cold gas mixture disappears without the explicit hot-only filter");
    // A distinct, optional flame layer, independent of hot mixture colour.
    visual.show_gas=false;visual.show_combustion=true;visual.hot_gas_only=false;
    state.process.gas[0].temperature_k=state.process.gas[1].temperature_k=1000;
    state.process.gas[0].free_volume_m3=.0001;
    const auto hot_without_burning=image();
    visual.show_combustion=false;
    require(hot_without_burning==image(),"Temperature alone draws a flame");
    visual.show_combustion=true;
    state.time_s=1;state.process.gas[0].released_energy_j=10000;
    const auto flame=image();
    if (const char* path=std::getenv("BATTERY_FLAME_PREVIEW_PPM")) {
        // Optional visual inspection artifact; normal tests write no files.
        std::ofstream output(path,std::ios::binary);
        output<<"P6\n"<<width<<' '<<height<<"\n255\n";
        for(int y=height-1;y>=0;--y)for(int x=0;x<width;++x)
            output.write(reinterpret_cast<const char*>(&flame[(y*width+x)*4]),3);
    }
    int fire_pixels=0;
    for(std::size_t i=0;i<flame.size();i+=4)
        fire_pixels+=std::abs(int(flame[i])-int(hot_without_burning[i]))+
                     std::abs(int(flame[i+1])-int(hot_without_burning[i+1]))>3;
    require(fire_pixels>40,"Local R3 flame layer is invisible");
    require(flame==image(),"Flame animation moves on pause");
    state.time_s=1.2;state.process.gas[0].released_energy_j=12000;
    require(flame!=image(),"Flame texture does not animate with physical time");
    visual.show_combustion=false;
    require(hot_without_burning==image(),"Flame visibility switch leaves artifacts");
    visual.show_combustion=true;state.time_s=2;
    require(hot_without_burning==image(),"Flame persists after R3 stops");
}
void pad_regression(battery::SceneRenderer& renderer) {
    using namespace battery;
    Scenario s; CellConfig pad; pad.id="pad"; pad.is_battery=false;
    pad.dimensions_m={.1,.02,.12}; pad.thermal_divisions={1,1,1}; s.cells.push_back(pad);
    const auto mesh=build_thermal_mesh(s);
    VisualGeometry geometry; geometry.solid_materials={VisualMaterialKind::Insulation};
    VisualFrame state; state.cells.resize(1); state.cells[0].temperature_k=298.15;
    state.samples.assign(1,{298.15f,0,0,0});
    OrbitCamera camera; camera.distance=1.4;
    VisualSettings settings; settings.show_enclosure=false; settings.show_ports=false;
    settings.show_interfaces=false; settings.temperature_max_k=900;
    auto image=[&] {
        const auto texture=renderer.render(width,height,s,mesh,state,geometry,camera,settings,-1);
        std::vector<unsigned char> result(width*height*4);
        glBindTexture(GL_TEXTURE_2D,texture);glGetTexImage(GL_TEXTURE_2D,0,GL_RGBA,GL_UNSIGNED_BYTE,result.data());
        require(glGetError()==GL_NO_ERROR,"Pad shader GL error"); return result;
    };
    auto difference=[](const auto& a,const auto& b) {
        int n=0; for(std::size_t i=0;i<a.size();i+=4)
            n+=std::abs(int(a[i])-int(b[i]))+std::abs(int(a[i+1])-int(b[i+1]))+std::abs(int(a[i+2])-int(b[i+2]))>10;
        return n;
    };
    const auto brown=image();
    geometry.solid_materials[0]=VisualMaterialKind::ThermalConductor;
    const auto white=image(); require(difference(brown,white)>100,"Pad materials are not distinguishable");
    settings.show_pads=false; const auto hidden=image();
    require(difference(white,hidden)>100,"Pad visibility does not work independently of enclosure");
    settings.show_pads=true; state.samples[0][0]=850;
    const auto hot=image(); require(difference(white,hot)>100,"Pad thermal mask is missing");
    state.samples[0][1]=.8; const auto damaged=image();
    require(damaged==hot,"Non-shell damage repaints a passive pad");
    settings.show_damage=false; require(image()==hot,"Damage switch changes unrelated thermal field");
    geometry.solid_materials[0]=VisualMaterialKind::Heater;
    settings.show_pads=false; settings.show_enclosure=false;
    settings.show_heaters=true; const auto heater=image();
    settings.show_heaters=false; const auto no_heater=image();
    require(difference(heater,no_heater)>100,"Heater is invisible or tied to enclosure/pad visibility");
}
void picking_regression(battery::SceneRenderer& renderer) {
    using namespace battery;
    Scenario s;
    CellConfig c; c.id="battery";c.dimensions_m={.03,.03,.06};c.position_m.x=-.025;
    s.cells.push_back(c);
    c.id="pad";c.is_battery=false;c.position_m.x=.025;s.cells.push_back(c);
    VisualGeometry geometry;geometry.solid_materials={VisualMaterialKind::Unspecified,VisualMaterialKind::Insulation};
    OrbitCamera camera;camera.distance=1.4;
    VisualSettings settings;
    auto visible=[&](int target,bool alt=false) {
        for(int y=0;y<height;y+=4)for(int x=0;x<width;x+=4)
            if(renderer.pick_cell(x,y,width,height,s,camera,&geometry,&settings,alt)==target)return true;
        return false;
    };
    require(visible(0)&&visible(1),"Cannot click battery and passive pad");
    settings.show_pads=false;require(!visible(1),"Hidden pad intercepts selection");
    geometry.solid_materials[1]=VisualMaterialKind::Enclosure;
    settings.show_enclosure=true;
    require(!visible(1)&&visible(1,true),"Case needs Alt selection without blocking ordinary clicks");
    settings.show_enclosure=false;require(!visible(1,true),"Hidden case intercepts selection");
}
}
void file_gas_regression(battery::SceneRenderer& renderer, const char* path) {
    using namespace battery;
    const auto document=load_scenario_document(path);
    const auto s=scenario_from_json(document);
    auto geometry=load_visual_geometry(path,s);
    require(!geometry.gas.empty()&&geometry.unresolved_gas_regions==0,"File launch has no mapped gas");
    Simulation simulation(s);auto frame=capture_visual_frame(simulation);
    const auto mesh=simulation.thermal_mesh();OrbitCamera camera;
    VisualSettings settings;settings.battery_surface_opacity=.9;settings.gas_opacity=.24;
    settings.temperature_max_k=1273.15;settings.show_flux=false;
    auto image=[&] {
        auto texture=renderer.render(width,height,s,mesh,frame,geometry,camera,settings,-1);
        std::vector<unsigned char> data(width*height*4);
        glBindTexture(GL_TEXTURE_2D,texture);glGetTexImage(GL_TEXTURE_2D,0,GL_RGBA,GL_UNSIGNED_BYTE,data.data());
        require(glGetError()==GL_NO_ERROR,"GL error in module file rendering");return data;
    };
    require(geometry.case_openings.size()==1,"Default case opening missing from visual geometry");
    settings.show_gas=false;settings.enclosure_opacity=.8f;
    const auto cut=image();
    const auto holes=geometry.case_openings;geometry.case_openings.clear();
    const auto intact=image();geometry.case_openings=holes;
    int aperture_pixels=0;
    for(std::size_t i=0;i<cut.size();i+=4)
        aperture_pixels+=std::abs(int(cut[i])-int(intact[i]))+std::abs(int(cut[i+1])-int(intact[i+1]))+
                         std::abs(int(cut[i+2])-int(intact[i+2]))>3;
    require(aperture_pixels>10,"Opening is not visually cut out of the case");
    geometry.case_openings[0].size={.04,.04};
    const auto larger=image();
    geometry.case_openings[0].circle=false;
    geometry.case_openings[0].size={.04,.012};
    const auto rectangle=image();
    int resized_pixels=0,reshaped_pixels=0;
    for(std::size_t i=0;i<cut.size();i+=4) {
        resized_pixels+=std::abs(int(cut[i])-int(larger[i]))+std::abs(int(cut[i+1])-int(larger[i+1]))+
                        std::abs(int(cut[i+2])-int(larger[i+2]))>3;
        reshaped_pixels+=std::abs(int(larger[i])-int(rectangle[i]))+std::abs(int(larger[i+1])-int(rectangle[i+1]))+
                         std::abs(int(larger[i+2])-int(rectangle[i+2]))>3;
    }
    require(resized_pixels>10 && reshaped_pixels>10,"Opening size/shape edits do not change rendered aperture");
    geometry.case_openings=holes;
    bool picked_top=false;
    for(int y=0;y<height && !picked_top;y+=2)for(int x=0;x<width;x+=2) {
        Vec3 point;
        const int wall=renderer.pick_case_surface(x,y,width,height,s,camera,geometry,point);
        if(wall==holes[0].wall && std::abs(point.x)<.009 && std::abs(point.y)<.009) {
            require(std::abs(point.z-holes[0].center.z)<1e-6,"Wall picking returns wrong world coordinates");
            picked_top=true;break;
        }
    }
    require(picked_top,"Cannot place/select a hole on the top wall with the mouse ray");
    settings.enclosure_opacity=.07f;
    settings.show_gas=false;const auto hidden=image();settings.show_gas=true;
    for(double temperature : {298.15,400.}) {
        for(auto& gas:frame.process.gas)gas.temperature_k=temperature;
        const auto visible=image();int changed=0;
        for(std::size_t i=0;i<visible.size();i+=4)
            changed+=std::abs(int(visible[i])-int(hidden[i]))+std::abs(int(visible[i+1])-int(hidden[i+1]))+std::abs(int(visible[i+2])-int(hidden[i+2]))>3;
        std::cout<<"Module gas at "<<temperature<<" K: "<<changed<<" visible pixels, "<<geometry.gas.size()<<" regions\n";
        require(changed>100,"Default module gas is invisible at normal UI opacity");
    }
}
int main(int argc,char** argv){try {
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
    shell_damage_regression(renderer);
    pad_regression(renderer);
    picking_regression(renderer);
    if(argc==2)file_gas_regression(renderer,argv[1]);
    renderer.destroy();glfwDestroyWindow(window);glfwTerminate();
    std::cout<<"Continuous macro field: identical uniform 1/8/27/64-zone render; gradient/cutaway and moving gas verified\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
