#include "editor_internal.hpp"
#include "battery_native/contracts/enclosure_openings.hpp"
#include "battery_native/contracts/experiment_editing.hpp"
#include "imgui.h"
#include "imgui_internal.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include <stdexcept>
#include <cmath>

namespace battery::ui {
struct ExperimentEditorTestAccess {
    static auto& get(ExperimentEditor& editor) { return *editor.impl_; }
};
}
void require(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
int main(int argc, char** argv) {
    try {
        require(argc == 3, "Expected generated and imported contracts");
        require(glfwInit(), "GLFW failed");
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        auto* window = glfwCreateWindow(1400, 1000, "Opening UI test", nullptr, nullptr);
        require(window, "GL context failed"); glfwMakeContextCurrent(window);
        for (int fixture = 1; fixture <= 2; ++fixture) {
            using namespace battery; using J = nlohmann::json;
            ImGui::CreateContext();
            auto& io = ImGui::GetIO();
            io.DisplaySize = {1400, 1000}; io.DeltaTime = 1.f/60; io.IniFilename = nullptr;
            io.ConfigInputTrickleEventQueue = false; io.ConfigWindowsMoveFromTitleBarOnly = true;
            unsigned char* pixels; int fw, fh; io.Fonts->GetTexDataAsRGBA32(&pixels, &fw, &fh);
            {
                ui::ExperimentEditor editor;
                editor.try_load(argv[fixture]);
                auto& p = ui::ExperimentEditorTestAccess::get(editor);
                if (editable_enclosures(p.document.draft).empty()) {
                    J props = {{"inside_dimensions_m", {.12,.12,.08}}, {"position_m", {0.,0.,0.}},
                        {"thickness_m", .002}, {"material_id", "test-wall"}, {"density_kg_m3",1200.},
                        {"specific_heat_j_kg_k",1200.}, {"conductivity_w_per_m_k",{.2,.2,.2}},
                        {"initial_temperature_k",298.15}, {"ambient_temperature_k",298.15},
                        {"gas_contact_h_w_m2_k",10.}, {"ambient_h_w_m2_k",0.},
                        {"damage_onset_k",500.}, {"damage_rate_per_s",0.}, {"damage_scale_k",40.}};
                    p.document.draft = set_enclosure(p.document.draft, "case", "cavity", props);
                    p.enclosure_id = "case";
                }
                p.refresh_preview();
                auto count = [&] { return editable_enclosures(p.document.draft).at(p.enclosure_id).at("openings").size(); };
                ImVec2 add_button, mouse_button, viewport_origin; ImGuiID add_id = 0, mouse_id = 0;
                auto frame = [&] {
                    ImGui::NewFrame();
                    ImGui::SetNextWindowPos({0,0}); ImGui::SetNextWindowSize({1400,1000});
                    ImGui::Begin("Case UI", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
                    ImGui::BeginChild("controls", {500,0}, true);
                    const auto start = ImGui::GetCursorScreenPos();
                    const auto& style = ImGui::GetStyle();
                    const float row_y = start.y + ImGui::GetTextLineHeight() + 2*style.SeparatorTextPadding.y + style.ItemSpacing.y;
                    add_button = {start.x+20, row_y+ImGui::GetFrameHeight()/2};
                    mouse_button = {start.x+ImGui::CalcTextSize("Add opening").x+2*style.FramePadding.x+style.ItemSpacing.x+20,
                                    add_button.y};
                    add_id = ImGui::GetID("Add opening");
                    mouse_id = ImGui::GetID(p.place_case_opening ? "Cancel placement" : "Place with mouse");
                    p.draw_case_openings();
                    ImGui::EndChild(); ImGui::SameLine();
                    ImGui::BeginChild("view", {0,0}, true);
                    p.draw_preview();
                    viewport_origin = ImGui::GetItemRectMin(); // orbit_viewport is the last item
                    ImGui::EndChild(); ImGui::End(); ImGui::Render();
                };
                auto click = [&](ImVec2 pos, ImGuiID expected = 0) {
                    io.AddMousePosEvent(pos.x,pos.y); frame();
                    io.AddMouseButtonEvent(0,true); frame();
                    if (expected) require(ImGui::GetCurrentContext()->ActiveId == expected, "Test did not press the intended UI button");
                    io.AddMouseButtonEvent(0,false); frame();
                };
                frame(); frame();
                const auto original = count();
                click(add_button, add_id);
                require(count() == original+1, "Add opening button failed: " + p.case_opening_error);
                p.refresh_preview(); frame();
                click(mouse_button, mouse_id);
                require(p.place_case_opening && p.case_opening_selection.creating, "Mouse placement button lost its state");
                frame();
                const auto origin = viewport_origin;
                // Locate a geometrically valid mouse hit using the production
                // ray caster, then deliver actual down/up events to the viewport.
                const auto spec = editable_enclosures(p.document.draft).at(p.enclosure_id);
                const auto center = spec.at("position_m").get<std::array<double,3>>();
                ImVec2 target; bool found = false;
                for (int y=25; y<p.preview_height && !found; y+=8) for(int x=25; x<p.preview_width; x+=8) {
                    Vec3 point;
                    int index=p.preview_renderer.pick_case_surface(x,y,p.preview_width,p.preview_height,
                        p.preview_scenario,p.camera,p.preview_geometry,point);
                    if(index<0) continue;
                    auto wid=p.preview_scenario.cells[index].id;
                    int axis=wid.back()-'x'; if(axis<0 || axis>2) continue;
                    const auto face=std::string(wid.find("-plus")!=std::string::npos?"+":"-")+"xyz"[axis];
                    const double coords[]={point.x,point.y,point.z};
                    auto test=spec;
                    test["openings"].push_back({{"id","mouse-test"},{"face",face},{"shape","circle"},
                        {"center_m",{coords[(axis+1)%3]-center[(axis+1)%3],coords[(axis+2)%3]-center[(axis+2)%3]}},
                        {"size_m",{p.case_opening_size_mm[0]/1000,p.case_opening_size_mm[0]/1000}}});
                    try { (void)parse_enclosure_openings(test); }
                    catch(const std::exception&) {continue;}
                    target={origin.x+x,origin.y+y}; found=true; break;
                }
                require(found,"No valid wall hit available");
                click(target);
                require(count()==original+2,"Real viewport click did not create an opening: "+p.message);
                require(!p.place_case_opening,"Successful placement did not exit tool");
                // The viewport must not move when the tool is activated or finishes.
                require(std::abs(viewport_origin.y-origin.y)<.1f,"Tool mode moved the viewport");
                p.refresh_preview(); frame();
                click(mouse_button, mouse_id);
                const auto before=count();
                p.case_opening_size_mm[0]=1000; // oversized: report at tool, keep armed
                click(target);
                require(count()==before && !p.case_opening_error.empty() && p.place_case_opening,
                        "Invalid click was silent or lost placement mode");
                require(p.document.compile().at("gas_network").at("channels").size()>0,"Invalid graph after mouse edit");
                std::cout << "Actual ImGui buttons + mouse down/up + raycast + physical graph: fixture " << fixture << " passed\n";
            }
            ImGui::DestroyContext();
        }
        glfwDestroyWindow(window); glfwTerminate(); return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
