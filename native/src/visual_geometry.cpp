#include "battery_native/visual_frame.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <fstream>
#include <limits>

namespace battery {
VisualGeometry load_visual_geometry(const std::string& path,const Scenario& scenario) {
    std::ifstream input(path);const auto root=nlohmann::json::parse(input);
    const auto meta=root.value("metadata",nlohmann::json::object());
    const auto boxes=meta.value("gas_volume_geometry",nlohmann::json::object());
    const auto reconstruction=meta.value("geometry_reconstruction",nlohmann::json::object());
    const auto polygons=reconstruction.value("polygon_xy_m",nlohmann::json::array());
    VisualGeometry result;
    result.solid_proxies=scenario.cells;
    result.hidden_planar_shells.resize(scenario.cells.size(),false);
    const auto cylinders=meta.value("packed_cylinder_geometry",nlohmann::json::object());
    if(cylinders.contains("diameter_m") && cylinders.contains("height_m")) {
        const double diameter=cylinders.at("diameter_m"),height=cylinders.at("height_m");
        if(!std::isfinite(diameter) || !std::isfinite(height) || diameter<=0 || height<=0)
            throw std::runtime_error("Invalid visual cylinder dimensions");
        for(auto& cell:result.solid_proxies)if(cell.is_battery) {
            cell.shape=ShapeKind::Cylinder;cell.dimensions_m={diameter/2,height,0};
        }
        for(std::size_t i=0;i<scenario.cells.size();++i) {
            const auto& shell=scenario.cells[i];
            result.hidden_planar_shells[i]=!shell.shell_contents_id.empty() &&
                std::any_of(result.solid_proxies.begin(),result.solid_proxies.end(),[&](const auto& c){
                    return c.is_battery && c.shape==ShapeKind::Cylinder && c.id==shell.shell_contents_id;});
        }
    }
    for(std::size_t i=0;i<scenario.gas_compartments.size();++i) {
        const auto& node=scenario.gas_compartments[i];
        // Interior mobile inventory shares a solid numerical cell; do not draw
        // it a second time as an exterior plume.
        if(node.owner_cell>=0 || node.thermal_cell>=0)continue;
        GasVisualRegion region;region.compartment=int(i);
        if(boxes.contains(node.id)) {
            auto lo=boxes.at(node.id).at("low_m").get<std::array<double,3>>();
            auto hi=boxes.at(node.id).at("high_m").get<std::array<double,3>>();
            region.center={(lo[0]+hi[0])/2,(lo[1]+hi[1])/2,(lo[2]+hi[2])/2};
            region.size={hi[0]-lo[0],hi[1]-lo[1],hi[2]-lo[2]};
        } else if(!polygons.empty() && (node.id.starts_with("body-") || node.id.starts_with("top-") || node.id.starts_with("base-"))) {
            const int n=std::stoi(node.id.substr(node.id.find('-')+1));
            if(n<0 || n>=int(polygons.size()))throw std::runtime_error("Invalid gas polygon index");
            const std::string id="B"+(n<10?std::string("0"):std::string{})+std::to_string(n);
            auto cell=std::find_if(scenario.cells.begin(),scenario.cells.end(),[&](auto& c){return c.id==id;});
            if(cell==scenario.cells.end())throw std::runtime_error("Missing gas polygon battery");
            double xmin=1e10,xmax=-1e10,ymin=1e10,ymax=-1e10;
            for(const auto& p:polygons[n]) {
                const double x=p[0],y=p[1];region.polygon.push_back({float(x),float(y)});
                xmin=std::min(xmin,x);xmax=std::max(xmax,x);ymin=std::min(ymin,y);ymax=std::max(ymax,y);
            }
            // Dimensions are read from the reconstructed scene, not another
            // visual geometry fit. Use the actual outer can and ABS inner face.
            auto outer=cell->dimensions_m;double floor=0,roof=0;
            for(const auto& c:scenario.cells) {
                if(c.shell_contents_id==id) {
                    outer[0]=std::max(outer[0],2*std::abs(c.position_m.x-cell->position_m.x)+c.dimensions_m[0]);
                    outer[1]=std::max(outer[1],2*std::abs(c.position_m.y-cell->position_m.y)+c.dimensions_m[1]);
                    outer[2]=std::max(outer[2],2*std::abs(c.position_m.z-cell->position_m.z)+c.dimensions_m[2]);
                }
                if(c.id=="ABS:+z")roof=c.position_m.z-c.dimensions_m[2]/2;
                if(c.id=="ABS:-z")floor=c.position_m.z+c.dimensions_m[2]/2;
            }
            double low=cell->position_m.z-outer[2]/2,high=cell->position_m.z+outer[2]/2;
            if(node.id.starts_with("top-")) {low=high;high=roof;}
            else if(node.id.starts_with("base-")) {high=low;low=floor;}
            else {region.excluded_center=cell->position_m;region.excluded_size={outer[0],outer[1],outer[2]};}
            if(high<=low)throw std::runtime_error("Invalid gas cavity height");
            region.center={(xmin+xmax)/2,(ymin+ymax)/2,(low+high)/2};region.size={xmax-xmin,ymax-ymin,high-low};
        } else {++result.unresolved_gas_regions;continue;}
        for(const auto& c:result.solid_proxies)if(c.is_battery && c.shape==ShapeKind::Cylinder) {
            // This packed fixture is axial. Exclude the real circular cross
            // section, not its solver's equal-volume square, from the plume.
            if(std::abs(c.orientation_xyzw[0])+std::abs(c.orientation_xyzw[1])>1e-8)
                throw std::runtime_error("Tilted visual cylinder gas exclusion unsupported");
            const double r=c.dimensions_m[0],h=c.dimensions_m[1];
            if(std::abs(c.position_m.x-region.center.x)<=region.size.x/2+r &&
               std::abs(c.position_m.y-region.center.y)<=region.size.y/2+r &&
               std::abs(c.position_m.z-region.center.z)<=region.size.z/2+h/2)
                region.excluded_cylinders.push_back({c.position_m,r,h});
        }
        if(!region.excluded_cylinders.empty())region.excluded_size={};
        result.gas.push_back(std::move(region));
    }
    return result;
}
Vec3 project_visual_shell_point(const Scenario& s,const VisualGeometry& g,int shell,Vec3 p) {
    if(shell<0 || g.solid_proxies.empty())return p;
    const auto& panel=s.cells.at(shell);
    const auto parent=std::find_if(g.solid_proxies.begin(),g.solid_proxies.end(),[&](const auto& c){
        return c.id==panel.shell_contents_id && c.is_battery && c.shape==ShapeKind::Cylinder;});
    if(parent==g.solid_proxies.end())return p;
    const double x=p.x-parent->position_m.x,y=p.y-parent->position_m.y;
    const double radius=parent->dimensions_m[0],height=parent->dimensions_m[1];
    const auto& d=panel.dimensions_m;
    if(d[2]<d[0] && d[2]<d[1]) {
        p.z=parent->position_m.z+(panel.position_m.z>parent->position_m.z?1:-1)*height/2;
        const double norm=std::hypot(x,y);
        if(norm>radius){p.x=parent->position_m.x+x*radius/norm;p.y=parent->position_m.y+y*radius/norm;}
    }else {
        const double norm=std::hypot(x,y);
        if(norm>1e-12){p.x=parent->position_m.x+x*radius/norm;p.y=parent->position_m.y+y*radius/norm;}
        p.z=std::clamp(p.z,parent->position_m.z-height/2,parent->position_m.z+height/2);
    }
    return p;
}
}
