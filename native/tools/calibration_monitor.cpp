#include <GLFW/glfw3.h>
#include <imgui.h>
#include <implot.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

using Json=nlohmann::json;
namespace fs=std::filesystem;
static Json read_json(const fs::path& path) {
    std::ifstream in(path); if(!in) return {};
    try {return Json::parse(in);} catch(const Json::exception&) {return {};}
}
static std::vector<double> values(const Json& obj,const std::string& key) {
    std::vector<double> out;
    if(!obj.is_object() || !obj.contains(key) || !obj[key].is_array())return out;
    for(const auto& x:obj[key])out.push_back(x.is_number()?x.get<double>():std::numeric_limits<double>::quiet_NaN());
    return out;
}
static void plot(const char* label,const std::vector<double>& xs,const Json& collection,const std::string& key) {
    auto ys=values(collection,key);const auto n=std::min(xs.size(),ys.size());
    if(n)ImPlot::PlotLine(label,xs.data(),ys.data(),static_cast<int>(n));
}
int main(int argc,char** argv) {
    if(argc!=2){std::fprintf(stderr,"Usage: %s RUN_DIRECTORY\n",argv[0]);return 2;}
    const fs::path folder=argv[1];
    if(!glfwInit())return 1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window=glfwCreateWindow(1450,900,"BatterySim | Calibration monitor",nullptr,nullptr);
    if(!window){glfwTerminate();return 1;}glfwMakeContextCurrent(window);glfwSwapInterval(1);
    ImGui::CreateContext();ImPlot::CreateContext();ImGui::StyleColorsDark();
    ImGui::GetIO().IniFilename=nullptr;
    ImGui_ImplGlfw_InitForOpenGL(window,true);ImGui_ImplOpenGL3_Init("#version 330");
    Json exp=read_json(folder/"experiment.json"),live=Json::object(),status=Json::object();
    auto last=std::chrono::steady_clock::time_point{};
    std::array<bool,9> sensors{};for(int i=0;i<5;++i)sensors[i]=true;
    std::array<bool,30> cells{};cells[10]=cells[17]=cells[18]=true;
    while(!glfwWindowShouldClose(window)) {
        glfwPollEvents();auto now=std::chrono::steady_clock::now();
        if(now-last>std::chrono::seconds(2)) {
            auto e=read_json(folder/"experiment.json");if(e.is_object())exp=std::move(e);
            auto l=read_json(folder/"live.json");if(l.is_object())live=std::move(l);
            auto s=read_json(folder/"status.json");if(s.is_object())status=std::move(s);last=now;
        }
        ImGui_ImplOpenGL3_NewFrame();ImGui_ImplGlfw_NewFrame();ImGui::NewFrame();
        auto size=ImGui::GetIO().DisplaySize;ImGui::SetNextWindowPos({0,0});ImGui::SetNextWindowSize(size);
        ImGui::Begin("Calibration",nullptr,ImGuiWindowFlags_NoResize|ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoCollapse);
        ImGui::Text("Run: %s",folder.filename().c_str());
        ImGui::Text("Status: %s | Stage: %s",status.value("status",std::string("waiting")).c_str(),status.value("stage",std::string("waiting")).c_str());
        auto p=status.value("progress",Json::object());double t=p.value("time_s",0.0),wall=p.value("wall_s",0.0);
        double start=0.;
        if(live.value("phase",std::string())=="cascade") {
            const auto axis=values(live,"time_s");
            start=(!axis.empty() && axis.front()>1000.)?axis.front():1500.;
        }
        ImGui::Text("Simulation: %.0f s | segment wall: %.0f s | segment speed: %.2fx",t,wall,wall>0?(t-start)/wall:0.);
        if(status.contains("reason") && status["reason"].is_string())ImGui::TextWrapped("Reason: %s",status["reason"].get<std::string>().c_str());
        if(p.contains("secondary_events") && p["secondary_events"].is_array())for(const auto& e:p["secondary_events"])
            ImGui::Text("R2 threshold: %s at %.0f s",e.value("cell",std::string("?")).c_str(),e.value("time_s",0.));
        ImGui::TextUnformatted("Experimental TC positions are not mapped to individual cells; curves are not paired measurements.");
        if(ImGui::CollapsingHeader("Visible curves",ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::TextUnformatted("Experiment:");ImGui::SameLine();
            for(int i=0;i<9;++i){std::string id="T"+std::to_string(i+1);ImGui::Checkbox(id.c_str(),&sensors[i]);if(i<8)ImGui::SameLine();}
            ImGui::TextUnformatted("Model:");ImGui::SameLine();
            for(int i=0;i<30;++i){char id[5];std::snprintf(id,sizeof id,"B%02d",i);ImGui::Checkbox(id,&cells[i]);if(i%10!=9)ImGui::SameLine();}
        }
        const auto xt=values(exp,"time_s"),xm=values(live,"time_s");
        if(ImPlot::BeginPlot("Temperature vs time",ImVec2(-1,-1))) {
            ImPlot::SetupAxes("Time, s","Temperature, C");
            for(int i=0;i<9;++i)if(sensors[i]){auto k="T"+std::to_string(i+1);auto label="Experiment "+k;plot(label.c_str(),xt,exp,k);}
            Json cm=live.value("cells",Json::object());
            for(int i=0;i<30;++i)if(cells[i]){char k[5];std::snprintf(k,sizeof k,"B%02d",i);auto label=std::string("Model ")+k;plot(label.c_str(),xm,cm,k);}
            ImPlot::EndPlot();
        }
        ImGui::End();ImGui::Render();int w,h;glfwGetFramebufferSize(window,&w,&h);glViewport(0,0,w,h);
        glClearColor(.08f,.09f,.11f,1);glClear(GL_COLOR_BUFFER_BIT);ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());glfwSwapBuffers(window);
    }
    ImGui_ImplOpenGL3_Shutdown();ImGui_ImplGlfw_Shutdown();ImPlot::DestroyContext();ImGui::DestroyContext();glfwDestroyWindow(window);glfwTerminate();
}
