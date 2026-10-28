#include "battery_native/renderer.hpp"
#include "battery_native/visual_frame.hpp"
#include <GLFW/glfw3.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {
using Clock=std::chrono::steady_clock;
void style() {
    ImGui::StyleColorsDark();auto& s=ImGui::GetStyle();
    s.WindowPadding={16,14};s.ItemSpacing={8,9};s.FramePadding={10,7};
    s.WindowRounding=0;s.FrameRounding=5;s.GrabRounding=4;s.WindowBorderSize=0;
    s.Colors[ImGuiCol_WindowBg]={.035f,.046f,.060f,1};
    s.Colors[ImGuiCol_ChildBg]={.025f,.033f,.045f,1};
    s.Colors[ImGuiCol_FrameBg]={.08f,.105f,.135f,1};
    s.Colors[ImGuiCol_Button]={.10f,.16f,.21f,1};
    s.Colors[ImGuiCol_ButtonHovered]={.13f,.29f,.36f,1};
    s.Colors[ImGuiCol_ButtonActive]={.12f,.42f,.46f,1};
    s.Colors[ImGuiCol_CheckMark]={.30f,.88f,.80f,1};
    s.Colors[ImGuiCol_SliderGrab]={.30f,.88f,.80f,1};
    s.Colors[ImGuiCol_Text]={.88f,.92f,.95f,1};
    s.Colors[ImGuiCol_TextDisabled]={.46f,.57f,.66f,1};
}
constexpr auto fixed=ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoResize|
    ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoSavedSettings;
ImVec4 color(battery::CellMode mode) {
    if(mode==battery::CellMode::Spent)return {battery::destroyed_rgb[0],battery::destroyed_rgb[1],battery::destroyed_rgb[2],1};
    if(mode==battery::CellMode::Runaway)return {1,.40f,.12f,1};
    return {.29f,.78f,.71f,1};
}
void legend(float width,const battery::VisualSettings& visual) {
    auto* draw=ImGui::GetWindowDrawList();const auto p=ImGui::GetCursorScreenPos();
    for(int i=0;i<160;++i) {
        auto a=battery::field_rgb(visual.field_layer,i/159.f);
        draw->AddRectFilled({p.x+width*i/160,p.y},{p.x+width*(i+1)/160+1,p.y+8},
            ImGui::ColorConvertFloat4ToU32({a[0],a[1],a[2],1}));
    }
    ImGui::Dummy({width,10});
}
void screenshot(GLFWwindow* window,const std::string& path) {
    int w,h;glfwGetFramebufferSize(window,&w,&h);std::vector<unsigned char> pixels(w*h*3);
    glPixelStorei(GL_PACK_ALIGNMENT,1);glReadBuffer(GL_BACK);
    glReadPixels(0,0,w,h,GL_RGB,GL_UNSIGNED_BYTE,pixels.data());
    std::ofstream out(path,std::ios::binary);out<<"P6\n"<<w<<" "<<h<<"\n255\n";
    for(int y=h-1;y>=0;--y)out.write(reinterpret_cast<const char*>(pixels.data()+y*w*3),w*3);
    if(!out)throw std::runtime_error("Cannot write screenshot");
}
}

int main(int argc,char** argv) {
try {
    std::string scene_path="examples/reaction-chain-demo.json",manifest,capture;
    double start_time=0;int frame_limit=0,mesh_scale=1;bool paused=true,hidden=false,isothermal=false;
    for(int i=1;i<argc;++i) {
        const std::string arg=argv[i];
        auto next=[&](){if(i+1>=argc)throw std::runtime_error("Missing argument for "+arg);return std::string(argv[++i]);};
        if(arg=="--replay")manifest=next();else if(arg=="--at")start_time=std::stod(next());
        else if(arg=="--screenshot")capture=next();else if(arg=="--frames")frame_limit=std::stoi(next());
        else if(arg=="--play")paused=false;else if(arg=="--hidden")hidden=true;
        else if(arg=="--mesh-scale")mesh_scale=std::stoi(next());
        else if(arg=="--isothermal")isothermal=true;
        else if(arg.starts_with("--"))throw std::runtime_error("Unknown option: "+arg);
        else scene_path=arg;
    }
    std::unique_ptr<battery::VisualReplay> replay;
    if(!manifest.empty()){replay=std::make_unique<battery::VisualReplay>(manifest);scene_path=replay->scenario_path();}
    auto scenario=battery::load_scenario(scene_path);
    if(replay && (mesh_scale!=1 || isothermal))throw std::runtime_error("Cannot change physics/topology of a recorded trajectory");
    if(mesh_scale<1 || mesh_scale>8)throw std::runtime_error("mesh-scale must be 1..8");
    if(mesh_scale!=1)battery::refine_environment_mesh(scenario,mesh_scale);
    if(isothermal)for(auto& cell:scenario.cells)if(cell.is_battery)cell.isothermal=true;
    auto geometry=battery::load_visual_geometry(scene_path,scenario);
    std::unique_ptr<battery::Simulation> simulation;
    battery::ThermalMesh replay_mesh;
    if(replay)replay_mesh=battery::build_thermal_mesh(scenario);
    else simulation=std::make_unique<battery::Simulation>(scenario);
    const auto& mesh=replay?replay_mesh:simulation->thermal_mesh();
    const double duration=replay?replay->times().back():scenario.settings.duration_s;
    double time=std::clamp(start_time,0.,duration);
    std::vector<int> batteries;
    for(std::size_t i=0;i<scenario.cells.size();++i)if(scenario.cells[i].is_battery)batteries.push_back(i);
    if(batteries.empty())throw std::runtime_error("Scene has no batteries");
    double last_spent_time=-1;
    if(replay) {
        std::vector<std::string> completed;
        for(const auto& e:replay->events())if(e.kind=="runaway_complete") {
            completed.push_back(e.entity_id);last_spent_time=std::max(last_spent_time,e.time_s);
        }
        std::sort(completed.begin(),completed.end());completed.erase(std::unique(completed.begin(),completed.end()),completed.end());
        if(completed.size()!=batteries.size())last_spent_time=-1;
    }
    int selected=batteries.front();
    glfwSetErrorCallback([](int code,const char* msg){std::fprintf(stderr,"GLFW %d: %s\n",code,msg);});
    if(!glfwInit())throw std::runtime_error("GLFW initialization failed");
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
    if(hidden)glfwWindowHint(GLFW_VISIBLE,GLFW_FALSE);
    auto* window=glfwCreateWindow(1600,1000,"Battery Sim | Thermal cascade",nullptr,nullptr);
    if(!window)throw std::runtime_error("Cannot create OpenGL window");
    glfwMakeContextCurrent(window);glfwSwapInterval(1);
    IMGUI_CHECKVERSION();ImGui::CreateContext();ImPlot::CreateContext();style();
    ImGui::GetIO().IniFilename=nullptr;
    const char* font="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf";
    if(std::filesystem::exists(font))ImGui::GetIO().Fonts->AddFontFromFileTTF(font,16);
    ImGui_ImplGlfw_InitForOpenGL(window,true);ImGui_ImplOpenGL3_Init("#version 330");
    battery::SceneRenderer renderer;if(!renderer.init())throw std::runtime_error("Renderer initialization failed");
    battery::VisualSettings visual;visual.temperature_min_k=298.15f;visual.temperature_max_k=1273.15f;
    visual.field_opacity=.82f;visual.shell_opacity=.035f;visual.show_interfaces=false;visual.gas_opacity=.06f;
    visual.battery_surface_opacity=.90f;
    battery::OrbitCamera orbit;orbit.yaw=-.87f;orbit.pitch=.60f;orbit.distance=.98f;
    const auto initial_camera=orbit;
    float rate=5;int auto_pause=0;bool auto_orbit=false,inspector=true,all_spent_paused=false;std::string error;
    auto tick=Clock::now();const auto application_start=tick;int frames=0;double solver_ms=0;
    battery::VisualFrame frame=replay?replay->sample(time,scenario,mesh):battery::capture_visual_frame(*simulation);
    // A paused scene is an unchanged texture. Avoid rebuilding volumes and
    // downloading solver diagnostics at monitor refresh rate while idle.
    unsigned int scene_texture=0;int rendered_w=0,rendered_h=0,rendered_selected=-1;
    double rendered_time=-1;battery::OrbitCamera rendered_camera;
    battery::VisualSettings rendered_settings;
    while(!glfwWindowShouldClose(window)) {
        glfwPollEvents();const auto now=Clock::now();const double wall_dt=std::min(.1,std::chrono::duration<double>(now-tick).count());tick=now;
        const double previous_time=time;
        if(!paused) {
            if(replay) {
                time=std::min(duration,time+wall_dt*rate);
                if(auto_pause==1 || auto_pause==2)for(const auto& event:replay->events()) {
                    const bool stop=auto_pause==1?event.kind=="shell_rupture":event.kind=="runaway_complete";
                    if(stop && event.time_s>previous_time+1e-8 && event.time_s<=time) {time=event.time_s;paused=true;break;}
                }
                if(auto_pause==3 && last_spent_time>previous_time+1e-8 && last_spent_time<=time) {time=last_spent_time;paused=true;}
                if(time>=duration)paused=true;
            } else {
                const auto begin=Clock::now();const auto event_count=simulation->events().size();
                try {
                    // Bound UI work by wall time, not thousands of blocking outer
                    // intervals. Every physical step is still committed unchanged.
                    do {
                        if(!simulation->step()) {paused=true;break;}
                        if(auto_pause)for(std::size_t k=event_count;k<simulation->events().size();++k) {
                            const auto& e=simulation->events()[k];
                            if((auto_pause==1&&e.kind=="shell_rupture") || (auto_pause==2&&e.kind=="runaway_complete"))paused=true;
                        }
                        if(auto_pause==3 && !all_spent_paused && simulation->all_batteries_destroyed()) {paused=true;all_spent_paused=true;}
                    } while(!paused && std::chrono::duration<double,std::milli>(Clock::now()-begin).count()<8);
                } catch(const std::exception& e){error=e.what();paused=true;}
                solver_ms=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();time=simulation->time_s();
            }
        }
        if(frame.time_s!=time)
            frame=replay?replay->sample(time,scenario,mesh):battery::capture_visual_frame(*simulation);
        if(auto_orbit)orbit.yaw+=float(wall_dt*.10);
        ImGui_ImplOpenGL3_NewFrame();ImGui_ImplGlfw_NewFrame();ImGui::NewFrame();
        auto& io=ImGui::GetIO();
        if(!io.WantTextInput && ImGui::IsKeyPressed(ImGuiKey_Space))paused=!paused;
        const float side=inspector?310.f:0.f,top=75.f,bottom=165.f;
        int reached=0,spent=0,holes=0;double tmax=0,gasmax=298.15,area=0,flow=0;
        for(int i:batteries){const auto& c=frame.cells[i];reached+=c.runaway_started_s>=0;spent+=c.mode==battery::CellMode::Spent;tmax=std::max(tmax,c.temperature_k);}
        for(std::size_t i=0;i<frame.process.channels.size();++i)if(scenario.gas_channels[i].rupture) {
            const auto& c=frame.process.channels[i];holes+=c.effective_area_m2>0;area+=c.effective_area_m2;flow+=std::max(0.,c.mass_flow_kg_s);
        }
        for(const auto& g:frame.process.gas)gasmax=std::max(gasmax,g.temperature_k);
        ImGui::SetNextWindowPos({0,0});ImGui::SetNextWindowSize({io.DisplaySize.x,top});ImGui::Begin("header",nullptr,fixed);
        ImGui::TextColored({.38f,.90f,.82f,1},"BATTERY / SIM");ImGui::SameLine(220);
        ImGui::Text("THERMAL CASCADE  /  %zu CELLS",batteries.size());ImGui::SameLine(620);
        ImGui::TextColored({1,.67f,.30f,1},"R2 %d/%zu   |   SPENT %d   |   OPEN PATCHES %d",reached,batteries.size(),spent,holes);
        ImGui::TextDisabled("%s  |  T max %.0f C  |  gas %.0f C",replay?"RECORDED CUDA / time-interpolated":"LIVE CUDA / committed physical steps",tmax-273.15,gasmax-273.15);
        if(replay && ImGui::IsItemHovered())ImGui::SetTooltip("%s",replay->source_label().c_str());
        ImGui::SameLine(io.DisplaySize.x-210);if(ImGui::SmallButton(inspector?"Hide controls [Tab]":"Show controls [Tab]"))inspector=!inspector;
        ImGui::End();
        if(ImGui::IsKeyPressed(ImGuiKey_Tab))inspector=!inspector;
        if(inspector) {
            ImGui::SetNextWindowPos({0,top});ImGui::SetNextWindowSize({side,io.DisplaySize.y-top-bottom});ImGui::Begin("inspector",nullptr,fixed);
            ImGui::TextDisabled("LAYERS");int layer=int(visual.field_layer);
            if(ImGui::RadioButton("Temperature",layer==0))visual.field_layer=battery::FieldLayer::Temperature;
            ImGui::SameLine();if(ImGui::RadioButton("Damage",layer==1))visual.field_layer=battery::FieldLayer::Degradation;
            ImGui::Checkbox("Hot gas volumes",&visual.show_gas);
            ImGui::Checkbox("Shell openings",&visual.show_ruptures);ImGui::SameLine();ImGui::Checkbox("Gas motion",&visual.show_flux);
            ImGui::Checkbox("Damage marks",&visual.show_damage);
            ImGui::Checkbox("Enclosure",&visual.show_enclosure);ImGui::SameLine();ImGui::Checkbox("Orbit",&auto_orbit);
            ImGui::SetNextItemWidth(140);ImGui::SliderFloat("Gas opacity",&visual.gas_opacity,.02f,.8f,"%.2f");
            ImGui::SetNextItemWidth(140);ImGui::SliderFloat("Battery density",&visual.battery_surface_opacity,.1f,1.f,"%.2f");
            ImGui::SetNextItemWidth(140);ImGui::SliderFloat("Cutaway X",&visual.cutaway_fraction,.05f,1,"%.2f");
            if(ImGui::CollapsingHeader("Display range / geometry")) {
                ImGui::Checkbox("Distant chamber gas",&visual.show_far_gas);
                float max_c=visual.temperature_max_k-273.15f;
                if(ImGui::SliderFloat("Max C",&max_c,200,1800,"%.0f"))visual.temperature_max_k=max_c+273.15f;
                ImGui::SliderFloat("Gas threshold K",&visual.gas_threshold_k,1,100,"%.0f");
                ImGui::SliderFloat("Enclosure opacity",&visual.enclosure_opacity,0,.5f,"%.2f");
                ImGui::TextWrapped("%zu mapped gas regions; %zu omitted (no geometry).",geometry.gas.size(),geometry.unresolved_gas_regions);
                ImGui::TextWrapped("Geometry: equivalent-volume prismatic cells; no invented cylindrical contacts.");
            }
            ImGui::Separator();ImGui::TextDisabled("CASCADE MAP / SELECT A CELL");
            for(std::size_t n=0;n<batteries.size();++n) {
                const int i=batteries[n];const auto& c=frame.cells[i];
                auto rgb=battery::field_rgb(visual.field_layer,visual.field_layer==battery::FieldLayer::Temperature?
                    float((c.temperature_k-visual.temperature_min_k)/(visual.temperature_max_k-visual.temperature_min_k)):float(c.damage),c.mode==battery::CellMode::Spent);
                ImGui::PushID(i);ImGui::PushStyleColor(ImGuiCol_Button,ImVec4{rgb[0]*.65f+.04f,rgb[1]*.65f+.04f,rgb[2]*.65f+.04f,1});
                if(ImGui::Button(scenario.cells[i].id.c_str(),{45,28}))selected=i;
                if(ImGui::IsItemHovered())ImGui::SetTooltip("%.1f C | %s",c.temperature_k-273.15,battery::name(c.mode));
                ImGui::PopStyleColor();ImGui::PopID();if(n%5!=4 && n+1<batteries.size())ImGui::SameLine();
            }
            ImGui::Separator();const auto& chosen=frame.cells[selected];
            ImGui::Text("%s",scenario.cells[selected].id.c_str());ImGui::SameLine();ImGui::TextColored(color(chosen.mode),"%s",battery::name(chosen.mode));
            ImGui::SetWindowFontScale(1.6f);ImGui::Text("%.1f C",chosen.temperature_k-273.15);ImGui::SetWindowFontScale(1);
            ImGui::Text("R1 degradation   %.2f %%",chosen.damage*100);
            double shell_damage=0,opened_area=0,shell_area=0;
            for(std::size_t k=0;k<scenario.gas_channels.size();++k) {
                const auto& c=scenario.gas_channels[k];if(c.shell_object<0 || scenario.cells[c.shell_object].shell_contents_id!=scenario.cells[selected].id)continue;
                shell_damage=std::max(shell_damage,frame.process.channels[k].max_thermal_damage);
                opened_area+=frame.process.channels[k].effective_area_m2;shell_area+=battery::gas_channel_area(scenario,c);
            }
            ImGui::Text("Shell damage     %.1f %%",shell_damage*100);
            ImGui::ProgressBar(float(shell_damage),{-1,6},"");
            ImGui::Text("Rupture area     %.1f %%",shell_area>0?100*opened_area/shell_area:0);
            ImGui::Text("R2 power         %.1f W",chosen.chemical_power_w);
            ImGui::Text("Open shell area  %.1f mm2",area*1e6);
            ImGui::Text("Outward flux     %.3f g/s",flow*1e3);
            ImGui::TextDisabled("%0.0f FPS%s",io.Framerate,replay?" / playback":"");
            if(!replay)ImGui::TextDisabled("Solver %.1f ms/frame",solver_ms);
            if(!error.empty())ImGui::TextWrapped("SOLVER PAUSED: %s",error.c_str());
            ImGui::End();
        }
        ImGui::SetNextWindowPos({side,top});ImGui::SetNextWindowSize({io.DisplaySize.x-side,io.DisplaySize.y-top-bottom});ImGui::Begin("scene",nullptr,fixed|ImGuiWindowFlags_NoScrollbar);
        ImGui::TextDisabled("Drag to orbit / wheel to zoom / click a battery / Space to play");
        ImGui::SameLine();if(ImGui::SmallButton("Reset view"))orbit=initial_camera;
        const auto origin=ImGui::GetCursorScreenPos(),available=ImGui::GetContentRegionAvail();
        const int w=std::max(64,int(available.x)),h=std::max(64,int(available.y));
        if(!scene_texture || rendered_w!=w || rendered_h!=h || rendered_time!=time ||
           rendered_selected!=selected || rendered_camera!=orbit || rendered_settings!=visual) {
            scene_texture=renderer.render(w,h,scenario,mesh,frame,geometry,orbit,visual,selected);
            if(hidden && glGetError()!=GL_NO_ERROR)throw std::runtime_error("OpenGL error in hidden renderer verification");
            rendered_w=w;rendered_h=h;rendered_time=time;rendered_selected=selected;
            rendered_camera=orbit;rendered_settings=visual;
        }
        ImGui::Image(ImTextureID(scene_texture),{float(w),float(h)},{0,1},{1,0});
        if(ImGui::IsItemHovered()) {
            if(ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {orbit.yaw-=io.MouseDelta.x*.008f;orbit.pitch=std::clamp(orbit.pitch+io.MouseDelta.y*.008f,-1.4f,1.4f);}
            orbit.distance=std::clamp(orbit.distance*std::exp(-io.MouseWheel*.1f),.55f,6.f);
            if(ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
                const auto delta=ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
                if(delta.x*delta.x+delta.y*delta.y<16) {
                    int hit=renderer.pick_cell(io.MousePos.x-origin.x,io.MousePos.y-origin.y,w,h,scenario,orbit,&geometry);
                    if(hit>=0) {
                        if(!scenario.cells[hit].shell_contents_id.empty())for(int k:batteries)if(scenario.cells[k].id==scenario.cells[hit].shell_contents_id)hit=k;
                        if(scenario.cells[hit].is_battery)selected=hit;
                    }
                }
            }
        }
        auto* overlay=ImGui::GetWindowDrawList();
        overlay->AddText({origin.x+18,origin.y+18},IM_COL32(210,224,235,255),replay?"RECORDED FIELDS  /  time-interpolated":"LIVE FIELDS  /  CUDA");
        overlay->AddText({origin.x+18,origin.y+42},IM_COL32(133,157,178,255),"Moving puffs: accepted mass transfer / spatial temperature interpolation");
        if(!visual.show_far_gas)overlay->AddText({origin.x+18,origin.y+66},IM_COL32(133,157,178,255),"Gas view: module + near field (distant chamber hidden)");
        ImGui::End();
        ImGui::SetNextWindowPos({0,io.DisplaySize.y-bottom});ImGui::SetNextWindowSize({io.DisplaySize.x,bottom});ImGui::Begin("transport",nullptr,fixed);
        if(ImGui::Button(paused?"PLAY  [Space]":"PAUSE [Space]",{145,0}))paused=!paused;
        ImGui::SameLine();if(ImGui::Button("Start")) {if(replay)time=replay->times().front();else simulation->reset();paused=true;all_spent_paused=false;}
        ImGui::SameLine();
        if(replay) {
            if(ImGui::Button("Next rupture"))for(const auto& e:replay->events())if(e.kind=="shell_rupture"&&e.time_s>time+.01){time=e.time_s;paused=true;break;}
            ImGui::SameLine();ImGui::SetNextItemWidth(170);ImGui::SliderFloat("Speed",&rate,.25f,120,"%.1fx",ImGuiSliderFlags_Logarithmic);
        }
        ImGui::SameLine();ImGui::SetNextItemWidth(210);ImGui::Combo("Auto pause",&auto_pause,"Never\0Each rupture\0Each spent cell\0All cells spent\0");
        ImGui::SameLine();ImGui::Text("%8.1f / %.0f s",time,duration);
        float scrub=float(time);ImGui::SetNextItemWidth(io.DisplaySize.x-40);
        ImGui::BeginDisabled(!replay);if(ImGui::SliderFloat("##time",&scrub,0,float(duration),"%.1f s")){time=scrub;paused=true;}ImGui::EndDisabled();
        // Tick marks are exact saved event timestamps, not threshold guesses.
        const auto lo=ImGui::GetItemRectMin(),hi=ImGui::GetItemRectMax();auto* draw=ImGui::GetWindowDrawList();
        const auto& events=replay?replay->events():simulation->events();
        for(const auto& e:events)if(e.kind=="runaway_complete" || e.kind=="shell_rupture") {
            const float x=lo.x+(hi.x-lo.x)*float(e.time_s/duration);
            draw->AddLine({x,hi.y+1},{x,hi.y+5},e.kind=="shell_rupture"?IM_COL32(80,210,240,200):IM_COL32(90,130,255,200));
        }
        legend(240,visual);ImGui::SameLine();
        if(visual.field_layer==battery::FieldLayer::Temperature)
            ImGui::TextDisabled("%.0f - %.0f C / blue - yellow - red",visual.temperature_min_k-273.15,visual.temperature_max_k-273.15);
        else ImGui::TextDisabled("0 - 100%% / R1 damage; gas remains thermal (%.0f - %.0f C)",visual.temperature_min_k-273.15,visual.temperature_max_k-273.15);
        ImGui::SameLine();ImGui::TextDisabled("Hatching = damage / purple = spent");
        const auto known=frame.known_channels.empty()?scenario.gas_channels.size():std::size_t(std::count(frame.known_channels.begin(),frame.known_channels.end(),true));
        ImGui::TextDisabled("Gas flows: %zu / %zu%s. Motion scaled for display; not CFD velocity. Physics unchanged.",known,scenario.gas_channels.size(),replay?" recorded":" live");
        if(replay && replay->source_label().find("PARTIAL")!=std::string::npos) {
            ImGui::SameLine();ImGui::TextColored({1,.65f,.25f,1},"PARTIAL RECORDING");
        }
        ImGui::End();
        ImGui::Render();int fw,fh;glfwGetFramebufferSize(window,&fw,&fh);glViewport(0,0,fw,fh);glClearColor(.02f,.025f,.035f,1);glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        ++frames;if(!capture.empty() && (!frame_limit || frames>=frame_limit)) {screenshot(window,capture);capture.clear();}
        glfwSwapBuffers(window);if(frame_limit && frames>=frame_limit)break;
    }
    std::fprintf(stdout,"Viewer: %d frames in %.3f s (includes vsync); mode=%s; t=%.3f\n",frames,
        std::chrono::duration<double>(Clock::now()-application_start).count(),replay?"recorded":"live",time);
    renderer.destroy();ImGui_ImplOpenGL3_Shutdown();ImGui_ImplGlfw_Shutdown();ImPlot::DestroyContext();ImGui::DestroyContext();glfwDestroyWindow(window);glfwTerminate();
    return 0;
} catch(const std::exception& e){std::fprintf(stderr,"battery viewer: %s\n",e.what());return 1;}
}
