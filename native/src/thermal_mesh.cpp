#include "battery_native/thermal_mesh.hpp"
#include "battery_native/reaction.hpp"
#include "battery_native/conversion.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace battery {
double gas_channel_area(const Scenario& s,const GasChannel& ch) {
    if(!ch.opening.enabled)return ch.area_m2;
    const auto p=std::find_if(s.ports.begin(),s.ports.end(),[&](const auto& p){return p.id==ch.port_id;});
    if(p==s.ports.end())throw std::runtime_error("Progressive rupture has no panel port");
    const auto& panel=s.cells.at(ch.shell_object);
    const int count=panel.thermal_divisions[0]*panel.thermal_divisions[1]*panel.thermal_divisions[2];
    if(ch.shell_thermal_volume< -1 || ch.shell_thermal_volume>=count)throw std::runtime_error("Invalid shell thermal volume");
    const double area=p->area_m2/(ch.shell_thermal_volume>=0?count:1);
    if(!std::isfinite(ch.area_m2) || ch.area_m2<0 ||
       (ch.area_m2>0 && std::abs(ch.area_m2-area)>area*1e-9))
        throw std::runtime_error("Progressive rupture area is determined by its whole shell panel; remove the obsolete area_m2 ceiling and recalibrate");
    return area;
}
namespace {
using V = std::array<double, 3>;
V xyz(Vec3 p) { return {p.x, p.y, p.z}; }
V rotate(const CellConfig& c, V v, bool inverse = false) {
    const auto& q = c.orientation_xyzw;
    double norm = 0;
    for (double x : q) norm += x*x;
    if (!std::isfinite(norm) || norm < 1e-20) throw std::runtime_error("Invalid orientation: " + c.id);
    const double s = (inverse ? -1.0 : 1.0) / std::sqrt(norm);
    V u{q[0]*s, q[1]*s, q[2]*s};
    const double w = q[3]/std::sqrt(norm);
    V cross{u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]};
    V twice{u[1]*cross[2]-u[2]*cross[1], u[2]*cross[0]-u[0]*cross[2], u[0]*cross[1]-u[1]*cross[0]};
    for (int i=0;i<3;++i) v[i] += 2*(w*cross[i]+twice[i]);
    return v;
}
V local_point(const CellConfig& c, Vec3 p) {
    auto v = xyz(p), origin = xyz(c.position_m);
    for(int i=0;i<3;++i) v[i] -= origin[i];
    return rotate(c, v, true);
}
void positive(double x, const std::string& what) {
    if (!std::isfinite(x) || x <= 0) throw std::runtime_error("Expected positive " + what);
}
double maximum_g(const ConnectivityLaw& l) {
    return std::max(l.background_conductance_w_per_k+l.active_conductance_w_per_k,
                    l.destroyed_background_conductance_w_per_k+l.destroyed_active_conductance_w_per_k)
           * std::max(1.0, l.damage_conductance_factor_at_one);
}
void add_link(ThermalMesh& mesh, ThermalLink link) {
    link.maximum_g = maximum_g(link.law)*link.weight;
    mesh.links.push_back(link);
}
ThermalLink constant_link(int a, int b, double g) {
    ThermalLink e;
    e.a=a; e.b=b;
    e.law.l0_m=e.law.l_low_m=e.law.l_high_m=1e-4;
    e.law.destroyed_l0_m=e.law.destroyed_l_m=1e-4;
    e.law.background_conductance_w_per_k=g;
    e.law.destroyed_background_conductance_w_per_k=g;
    e.initial.l_m=e.initial.l0_m=1e-4;
    e.initial.conductance_w_per_k=g;
    return e;
}
struct Patch { int volume, axis; double lo[3], hi[3], area, half_resistance; };
// Named planar face. Port area alone has no aspect ratio: full-face ports use
// the actual rectangle, smaller ports use an explicitly documented square.
std::vector<Patch> face_patches(const Scenario& s, const ThermalMesh& m, const PortConfig& p) {
    const auto& c=s.cells[p.cell_index];
    if (p.surface.size()!=2 || (p.surface[0]!='+' && p.surface[0]!='-') ||
        p.surface[1]<'x' || p.surface[1]>'z')
        throw std::runtime_error("CUDA mesh needs a named planar face (+x/-x/+y/-y/+z/-z): " + p.id);
    const int axis=p.surface[1]-'x', u=(axis+1)%3, v=(axis+2)%3;
    const int sign=p.surface[0]=='+' ? 1 : -1;
    const auto center=local_point(c,p.centroid_m);
    const auto normal=rotate(c,xyz(p.normal),true);
    if (std::abs(center[axis]-sign*c.dimensions_m[axis]/2)>1e-6 ||
        std::abs(normal[axis]-sign)>1e-6 || std::abs(normal[u])+std::abs(normal[v])>1e-6)
        throw std::runtime_error("Port geometry does not match its face: " + p.id);
    positive(p.area_m2,"port area");
    const double face_area=c.dimensions_m[u]*c.dimensions_m[v];
    if(p.area_m2>face_area*(1+1e-6)) throw std::runtime_error("Port larger than face: "+p.id);
    const bool full=std::abs(p.area_m2-face_area)<face_area*1e-6;
    const bool rect=p.rectangle_m[0]!=0 || p.rectangle_m[1]!=0;
    if(rect && (p.rectangle_m[0]<=0 || p.rectangle_m[1]<=0 ||
        std::abs(p.rectangle_m[0]*p.rectangle_m[1]-p.area_m2)>p.area_m2*1e-8))
        throw std::runtime_error("Invalid explicit rectangular port: "+p.id);
    const double hu=rect?p.rectangle_m[0]/2:full?c.dimensions_m[u]/2:std::sqrt(p.area_m2)/2;
    const double hv=rect?p.rectangle_m[1]/2:full?c.dimensions_m[v]/2:std::sqrt(p.area_m2)/2;
    const double cu=full&&!rect?0:center[u], cv=full&&!rect?0:center[v];
    if(std::abs(cu)+hu>c.dimensions_m[u]/2+1e-8 || std::abs(cv)+hv>c.dimensions_m[v]/2+1e-8)
        throw std::runtime_error("Square port exceeds face; supply a smaller patch: "+p.id);
    std::vector<Patch> result;
    for(int j=m.object_offsets[p.cell_index]; j<m.object_offsets[p.cell_index+1]; ++j) {
        const auto& volume=m.volumes[j];
        const auto x=xyz(volume.local_center_m), dx=xyz(volume.size_m);
        if(std::abs(x[axis]+sign*dx[axis]/2-sign*c.dimensions_m[axis]/2)>1e-8) continue;
        Patch patch{}; patch.volume=j; patch.axis=axis;
        for(int k=0;k<3;++k) { patch.lo[k]=x[k]-dx[k]/2; patch.hi[k]=x[k]+dx[k]/2; }
        patch.lo[u]=std::max(patch.lo[u],cu-hu); patch.hi[u]=std::min(patch.hi[u],cu+hu);
        patch.lo[v]=std::max(patch.lo[v],cv-hv); patch.hi[v]=std::min(patch.hi[v],cv+hv);
        patch.area=std::max(0.0,patch.hi[u]-patch.lo[u])*std::max(0.0,patch.hi[v]-patch.lo[v]);
        patch.half_resistance=c.isothermal?0:dx[axis]/(2*c.conductivity_w_per_m_k[axis]); // divide by overlap area later
        if(patch.area>0) result.push_back(patch);
    }
    if(result.empty()) throw std::runtime_error("Empty port: "+p.id);
    return result;
}
}

ThermalMesh build_thermal_mesh(const Scenario& s) {
    if(s.cells.empty()) throw std::runtime_error("Empty thermal domain");
    positive(s.settings.max_dt_s,"max_dt_s"); positive(s.settings.duration_s,"duration_s");
    positive(s.settings.output_interval_s,"output_interval_s");
    positive(s.settings.stability_factor,"stability_factor");
    if(!std::isfinite(s.settings.step_interval_s)||s.settings.step_interval_s<0)
        throw std::runtime_error("Invalid outer step interval");
    if(s.settings.stability_factor>0.5) throw std::runtime_error("stability_factor must be <= 0.5");
    ThermalMesh m;
    const auto enclosed=[&](std::size_t i) {
        return std::any_of(s.cells.begin(),s.cells.end(),[&](const auto& c){return c.shell_contents_id==s.cells[i].id;});
    };
    m.object_offsets.push_back(0);
    for(std::size_t object=0;object<s.cells.size();++object) {
        const auto& c=s.cells[object];
        if(c.conversion.enabled) {
            const auto& p=c.conversion;
            if(!c.is_battery || c.isothermal || c.runaway_energy_j!=0 || c.runaway_pulse.profile!=RunawayProfile::ConstantPower ||
                c.heat_capacity_law.factor.count || c.energy_outlet.enabled || c.reaction_structure.enabled || c.bulk_destruction_on_boundary ||
                c.initial_damage!=0 || c.damage_rate_at_onset_per_s!=0 || c.bulk_failure_transient.relaxation_time_s!=0)
                throw std::runtime_error("Conversion has its own composition/capacity/source; legacy chemistry, aging and bulk failure cannot overlap");
            for(double x:{p.mass_kg,p.reactant_cp,p.activated_cp,p.residue_cp,
                p.degradation_rate_per_s,p.reaction_rate_per_s,p.reaction_reference_k,p.max_fraction_per_step,p.max_temperature_step_k})positive(x,"conversion parameter");
            for(double x:{p.net_heat_j_per_kg,p.reaction_activation_j_per_mol,p.gas_yield,p.gas_chemical_fraction,p.overcharge_conversion_per_charge})
                if(!std::isfinite(x)||x<0)throw std::runtime_error("Negative/nonfinite conversion parameter");
            for(double x:p.residue_conductivity)positive(x,"residue conductivity");
            {
                for(double x:{p.degradation_reference_k,p.runaway_reporting_rate_per_s,p.gas_yield_reference_k,p.gas_yield_max})positive(x,"R1/R2 parameter");
                for(double x:{p.degradation_activation_j_per_mol,p.initial_precursor_fraction,p.gas_yield_temperature_per_k,p.gas_yield_density_exponent})
                    if(!std::isfinite(x))throw std::runtime_error("Nonfinite R1/R2 parameter");
                if(p.degradation_activation_j_per_mol<0 ||
                    p.reaction_activation_j_per_mol<=p.degradation_activation_j_per_mol ||
                    p.initial_precursor_fraction<0 || p.initial_precursor_fraction>1 ||
                    p.gas_yield_max>1 || p.gas_yield>p.gas_yield_max)
                    throw std::runtime_error("R1 must be heat-neutral, R2 steeper, precursor/yield fractions bounded");
            }
            if(p.gas_yield>1 || p.gas_chemical_fraction>1 || (p.gas_yield==0 && p.gas_chemical_fraction>0) || p.max_fraction_per_step>.1 ||
                c.heat_capacity_j_per_k<=p.mass_kg*p.reactant_cp || p.gas_compartment<0 || p.gas_compartment>=static_cast<int>(s.gas_compartments.size()) ||
                s.gas_compartments[p.gas_compartment].owner_cell!=static_cast<int>(object))
                throw std::runtime_error("Invalid conversion fractions, inert capacity or gas owner");
        }
        if(c.shape==ShapeKind::Cylinder) throw std::runtime_error("CUDA v1 mesh supports prismatic/pouch volumes; cylinder mesher not implemented: "+c.id);
        if(c.electrical.enabled) {
            const auto& e=c.electrical;
            if(!c.is_battery)throw std::runtime_error("Cell Joule heating requires a battery");
            for(double x:{e.resistance_ohm,e.reference_temperature_k,e.min_resistance_ohm,e.max_resistance_ohm,e.max_temperature_step_k})positive(x,"electrical parameter");
            for(double x:{e.current_a,e.temperature_log_slope_per_k,e.damage_log_slope})
                if(!std::isfinite(x))throw std::runtime_error("Nonfinite electrical law");
            if(e.min_resistance_ohm>e.resistance_ohm || e.resistance_ohm>e.max_resistance_ohm)throw std::runtime_error("Reference resistance outside declared bounds");
            if(!std::isfinite(e.overcharge_fraction) || e.overcharge_fraction<0 || e.overcharge_fraction>1 ||
                !std::isfinite(e.nominal_capacity_coulomb) || e.nominal_capacity_coulomb<0 ||
                (e.overcharge_fraction>0 && (e.nominal_capacity_coulomb<=0 || !c.conversion.enabled)))
                throw std::runtime_error("Overcharge requires a finite nominal capacity and R1/R2 chemistry");
            double last=-1.;for(const auto& q:c.current_profile_a) {
                if(!std::isfinite(q[0]) || q[0]<0 || q[0]<=last || !std::isfinite(q[1]))throw std::runtime_error("Invalid current profile");
                last=q[0];
            }
        }
        if(c.dimensions_m.size()!=3) throw std::runtime_error("Missing 3D geometry: "+c.id);
        positive(c.heat_capacity_j_per_k,"heat capacity"); positive(c.runaway_duration_s,"runaway duration");
        positive(c.damage_temperature_scale_k,"damage temperature scale");
        if(!std::isfinite(c.runaway_pulse.fast_fraction) || c.runaway_pulse.fast_fraction<0 || c.runaway_pulse.fast_fraction>1)
            throw std::runtime_error("Invalid fast energy fraction");
        positive(c.runaway_pulse.tail_time_s,"pulse tail time");
        if(c.runaway_pulse.profile==RunawayProfile::BurstTail && c.runaway_pulse.tail_time_s<c.runaway_duration_s)
            throw std::runtime_error("Pulse tail must not be faster than burst");
        if(c.bulk_destruction_on_boundary) {
            const auto& b=c.bulk_connectivity;
            if(!c.has_bulk_connectivity || b.kind!=TransitionKind::DirectedGeometry || b.endpoint_damage_weight!=1 ||
               b.damage_rate_at_onset_per_s!=0 || b.destruction_damage!=1 || b.destroyed_active_conductance_w_per_k!=0 || c.isothermal)
                throw std::runtime_error("Boundary bulk switch requires a D=1 directed constant destroyed bulk branch");
            positive(b.destroyed_background_conductance_w_per_k,"destroyed bulk factor");
        }
        positive(c.bulk_failure_transient.peak_factor,"bulk transient peak factor");
        if(!std::isfinite(c.bulk_failure_transient.relaxation_time_s) || c.bulk_failure_transient.relaxation_time_s<0 ||
           (c.bulk_failure_transient.relaxation_time_s>0 && (!c.bulk_destruction_on_boundary || !c.is_battery)))
            throw std::runtime_error("Bulk transient requires battery rupture clock and boundary bulk switch");
        if(!std::isfinite(c.runaway_energy_j)||c.runaway_energy_j<0 || c.initial_damage<0 || c.initial_damage>1 ||
           c.aging_fraction<0 || c.aging_fraction>1 || c.damage_rate_at_onset_per_s<0)
            throw std::runtime_error("Invalid material state: "+c.id);
        if(c.runaway_pulse.profile==RunawayProfile::Kinetic) {
            if(!c.is_battery || c.isothermal || c.bulk_destruction_on_boundary || c.bulk_failure_transient.relaxation_time_s>0)
                throw std::runtime_error("Kinetic reaction requires a spatial battery, without legacy timed bulk destruction");
            positive(c.runaway_energy_j,"kinetic energy reserve");
            const auto& p=c.reaction_kinetics;
            positive(p.reference_temperature_k,"reaction reference temperature");positive(p.rate_at_reference_per_s,"reaction reference rate");
            positive(p.activation_damage,"reaction activation damage");positive(p.quench_rate_per_s,"quench reporting rate");
            positive(p.max_fraction_per_step,"reaction fraction step");positive(p.max_temperature_rise_per_step_k,"reaction temperature step");
            if(p.activation_damage>1 || p.max_fraction_per_step>.1 || !std::isfinite(p.activation_energy_j_per_mol) || p.activation_energy_j_per_mol<0 ||
               !std::isfinite(p.damage_sensitivity) || p.damage_sensitivity<0 || !std::isfinite(p.damage_per_conversion) || p.damage_per_conversion<0)
                throw std::runtime_error("Invalid reaction kinetics");
            if(c.reaction_structure.enabled) {
                positive(c.reaction_structure.activity_rate_per_s,"structure activity rate");
                for(int axis=0;axis<3;++axis) {positive(c.reaction_structure.active_factor[axis],"active structure factor");positive(c.reaction_structure.residue_factor[axis],"residue structure factor");}
                if(c.has_bulk_connectivity && (c.bulk_connectivity.endpoint_damage_weight!=0 || c.bulk_connectivity.damage_rate_at_onset_per_s!=0))
                    throw std::runtime_error("Reaction structure cannot be combined with a second bulk destruction law");
            }
            if(c.energy_outlet.enabled) {
                const auto& o=c.energy_outlet;
                for(double x:{o.opening_start_conversion,o.opening_end_conversion,o.closure_start_conversion,o.closure_end_conversion,o.heat_fraction})
                    if(!std::isfinite(x) || x<0 || x>1)throw std::runtime_error("Invalid outlet fraction");
                if(o.opening_start_conversion>=o.opening_end_conversion || o.closure_start_conversion<o.opening_end_conversion ||
                   (o.closure_start_conversion<1 && o.closure_end_conversion<=o.closure_start_conversion) ||
                   !std::isfinite(o.ejected_energy_per_released_energy) || o.ejected_energy_per_released_energy<0)
                    throw std::runtime_error("Invalid outlet conversion window or ejected reserve ratio");
            }
        } else if(c.reaction_structure.enabled || c.energy_outlet.enabled)
            throw std::runtime_error("Reaction structure/outlet requires kinetic reaction");
        auto n=c.thermal_divisions;
        if(!c.is_battery && c.runaway_energy_j!=0)
            throw std::runtime_error("Passive material cannot have a runaway energy source: "+c.id);
        if(c.isothermal && c.has_bulk_connectivity)
            throw std::runtime_error("Isothermal constraint conflicts with finite bulk connectivity: "+c.id);
        if(c.isothermal && !c.initial_temperature_field_k.empty() &&
           std::any_of(c.initial_temperature_field_k.begin(),c.initial_temperature_field_k.end(),
               [&](double t){return t!=c.initial_temperature_field_k.front();}))
            throw std::runtime_error("Isothermal object requires a uniform initial field: "+c.id);
        std::size_t count=1;
        for(int k=0;k<3;++k) {
            if(n[k]<1||n[k]>256) throw std::runtime_error("thermal_divisions must be in [1,256]");
            count*=n[k]; positive(c.dimensions_m[k],"dimension"); positive(c.conductivity_w_per_m_k[k],"conductivity");
        }
        if(count+m.volumes.size()>4000000) throw std::runtime_error("Thermal mesh exceeds 4M-volume safety limit");
        if(!c.initial_temperature_field_k.empty() && c.initial_temperature_field_k.size()!=count)
            throw std::runtime_error("Initial field size does not match thermal_divisions: "+c.id);
        V h{c.dimensions_m[0]/n[0],c.dimensions_m[1]/n[1],c.dimensions_m[2]/n[2]};
        const int base=static_cast<int>(m.volumes.size());
        for(int z=0;z<n[2];++z) for(int y=0;y<n[1];++y) for(int x=0;x<n[0];++x) {
            const int local=x+n[0]*(y+n[1]*z);
            const double t=c.initial_temperature_field_k.empty()?c.initial_temperature_k:c.initial_temperature_field_k[local];
            positive(t,"initial temperature");
            m.volumes.push_back({static_cast<int>(object),
                {(x+0.5)*h[0]-c.dimensions_m[0]/2,(y+0.5)*h[1]-c.dimensions_m[1]/2,(z+0.5)*h[2]-c.dimensions_m[2]/2},
                {h[0],h[1],h[2]},c.heat_capacity_j_per_k/count,t,volume_region(x,y,z,n)});
            int pos[3]{x,y,z}, stride[3]{1,n[0],n[0]*n[1]};
            // An isothermal object uses an energy-conserving algebraic constraint
            // on CUDA, not arbitrarily stiff internal diffusion links.
            for(int k=0;k<3;++k) if(!c.isothermal && pos[k]+1<n[k]) {
                const double area=h[(k+1)%3]*h[(k+2)%3];
                const double base_g=c.conductivity_w_per_m_k[k]*area/h[k];
                auto link=constant_link(base+local,base+local+stride[k],base_g);
                link.geometry_axis=k;
                if(c.has_bulk_connectivity) {
                    link.law=c.bulk_connectivity;
                    link.law.background_conductance_w_per_k*=base_g;
                    link.law.active_conductance_w_per_k*=base_g;
                    link.law.destroyed_background_conductance_w_per_k*=base_g;
                    link.law.destroyed_active_conductance_w_per_k*=base_g;
                    link.initial.l0_m=link.law.l0_m;link.initial.l_m=link.law.l_low_m;
                    link.law.endpoint_a=link.law.endpoint_b=c.has_bulk_response?c.bulk_response:c.surface_connectivity;
                }
                add_link(m,link);
                if(c.bulk_failure_transient.relaxation_time_s>0)
                    m.links.back().maximum_g=std::max(m.links.back().maximum_g,base_g*c.bulk_failure_transient.peak_factor);
                m.links.back().maximum_g*=maximum_reaction_factor(c,k);
                if(c.conversion.enabled)m.links.back().maximum_g*=std::max(1.,c.conversion.residue_conductivity[k]);
            }
        }
        m.object_offsets.push_back(static_cast<int>(m.volumes.size()));
    }
    std::unordered_map<std::string,const PortConfig*> ports;
    for(const auto& p:s.ports) ports.emplace(p.id,&p);
    if(!s.gas_compartments.empty()) {
        const auto& g=s.gas_model;
        positive(g.local_exchange_interval_s,"local gas exchange interval");
        if(g.cuda_batched_transport && !g.cuda_kawasaki_transport)throw std::runtime_error("Batched gas requires the CUDA mixture adapter");
        if(!std::isfinite(g.batch_pressure_fraction) || g.batch_pressure_fraction<=0 || g.batch_pressure_fraction>.25 ||
           !std::isfinite(g.batch_mean_threshold) || g.batch_mean_threshold<0 || g.batch_mean_threshold>32)
            throw std::runtime_error("Invalid gas batch controls");
        if(g.deterministic_sweeps_per_step<1 || g.deterministic_sweeps_per_step>8)
            throw std::runtime_error("Deterministic gas transport allows one to eight sweeps per thermal step");
        if(!std::isfinite(g.deterministic_pressure_fraction) || g.deterministic_pressure_fraction<=0 || g.deterministic_pressure_fraction>1)
            throw std::runtime_error("Invalid deterministic gas pressure budget");
        if(!std::isfinite(g.network_pressure_fraction) || g.network_pressure_fraction<=0 || g.network_pressure_fraction>1)
            throw std::runtime_error("Network gas pressure fraction must be in (0, 1]");
        positive(g.implicit_temperature_tolerance_k,"implicit gas temperature tolerance");
        positive(g.implicit_tracer_relative_tolerance,"implicit gas tracer tolerance");
        if(g.implicit_local_transport && !g.cpu_local_transport)throw std::runtime_error("Implicit groups require CPU local transport");
        if(g.cpu_local_transport && !g.split_transport)throw std::runtime_error("CPU local mass transport requires split LTE");
        if(g.cuda_kawasaki_transport && (!g.split_transport || g.cpu_local_transport || g.implicit_local_transport))
            throw std::runtime_error("CUDA Kawasaki requires split LTE and no CPU gas backend");
        for(double x:{g.reference_temperature_k,g.cv_j_per_kg_k,g.specific_r_j_per_kg_k,g.max_fraction_per_step,g.max_temperature_step_k})positive(x,"gas model");
        if(g.max_fraction_per_step>.1)throw std::runtime_error("Gas step fraction exceeds .1");
        positive(g.pressure_relaxation_fraction,"gas pressure relaxation fraction");
        positive(g.pressure_absolute_tolerance_pa,"gas absolute pressure tolerance");
        if(g.pressure_relaxation_fraction>.5 || !std::isfinite(g.pressure_relative_tolerance) ||
           g.pressure_relative_tolerance<0 || g.pressure_relative_tolerance>1)
            throw std::runtime_error("Invalid gas pressure timestep controls");
        std::unordered_map<std::string,bool> compartment_ids;
        std::unordered_map<int,bool> compartment_owners;
        for(const auto& c:s.gas_compartments) {
            if(c.id.empty() || c.id=="outside_domain" || !compartment_ids.emplace(c.id,true).second)throw std::runtime_error("Invalid/duplicate gas compartment id");
            positive(c.volume_m3,"gas free volume");positive(c.initial_mass_kg,"gas initial mass");positive(c.initial_temperature_k,"gas initial temperature");
            positive(c.oxidizer_heat_j_per_kg,"gas heat per oxidizer mass");
            for(double x:{c.initial_chemical_j,c.initial_oxidizer_mass_fraction,c.reaction_rate_per_s,c.ignition_temperature_k})
                if(!std::isfinite(x)||x<0)throw std::runtime_error("Invalid gas reaction/inventory parameter");
            if(c.initial_oxidizer_mass_fraction>1)throw std::runtime_error("Oxidizer fraction exceeds total gas mass");
            if(c.owner_cell>=static_cast<int>(s.cells.size()) || c.owner_cell< -1 || (c.owner_cell>=0 && !s.cells[c.owner_cell].is_battery))throw std::runtime_error("Invalid gas owner");
            if(c.owner_cell>=0) {
                if(!compartment_owners.emplace(c.owner_cell,true).second)throw std::runtime_error("One internal plenum per battery is supported");
                double envelope=1;for(double d:s.cells[c.owner_cell].dimensions_m)envelope*=d;
                if(c.volume_m3>envelope)throw std::runtime_error("Internal free gas volume exceeds the contents envelope");
            }
        }
        std::unordered_map<std::string,bool> channel_ids;
        for(const auto& ch:s.gas_channels) {
            if(!channel_ids.emplace(ch.id,true).second || ch.id.empty() || ch.from<0 || ch.to< -1 || ch.from==ch.to ||
                ch.from>=static_cast<int>(s.gas_compartments.size()) || ch.to>=static_cast<int>(s.gas_compartments.size()) || (!ch.internal && !ch.passage && !ports.contains(ch.port_id)))
                throw std::runtime_error("Invalid gas channel endpoints/port/id");
            if(ch.shell_thermal_volume< -1 || (ch.shell_thermal_volume>=0 && (!ch.opening.enabled || ch.surface_link_h_w_per_m2_k>0)))
                throw std::runtime_error("Local shell channels require damage opening and no legacy surface shortcut");
            if(ch.passage) {
                if(!g.split_transport || ch.to<0 || ch.internal || ch.rupture || ch.opening.enabled || ch.opening_pressure_pa!=0 || !ch.port_id.empty())
                    throw std::runtime_error("Free-space passage requires two gas volumes and no barrier/port");
                for(int endpoint:{ch.from,ch.to}) {
                    const auto& n=s.gas_compartments[endpoint];
                    if(n.owner_cell>=0 || (n.thermal_cell>=0 && s.cells[n.thermal_cell].is_battery))
                        throw std::runtime_error("Free-space passage cannot bypass a battery casing");
                }
                positive(ch.area_m2,"passage area");positive(ch.discharge_coefficient,"passage discharge coefficient");
                if(ch.discharge_coefficient>1)throw std::runtime_error("Passage discharge coefficient exceeds one");
                continue;
            }
            if(ch.internal) {
                const auto& a=s.gas_compartments[ch.from];
                if(!g.split_transport || ch.to<0 || a.thermal_cell<0 || a.thermal_cell!=s.gas_compartments[ch.to].thermal_cell ||
                    ch.shell_object!=a.thermal_cell || ch.rupture || ch.opening.enabled || ch.opening_pressure_pa!=0 || !ch.port_id.empty())
                    throw std::runtime_error("Internal pore path requires adjacent bound volumes of one material");
                positive(ch.area_m2,"internal pore area");positive(ch.discharge_coefficient,"internal discharge coefficient");
                if(ch.discharge_coefficient>1)throw std::runtime_error("Internal discharge coefficient exceeds one");
                continue; // exact adjacency/face-area check belongs to MassMesh
            }
            const auto& p=*ports.at(ch.port_id);const auto& shell=s.cells[p.cell_index];
            const auto& source=s.gas_compartments[ch.from];
            const int owner=source.owner_cell>=0?source.owner_cell:
                (g.split_transport && source.thermal_cell>=0 && s.cells.at(source.thermal_cell).is_battery?source.thermal_cell:-1);
            if(ch.shell_object!=static_cast<int>(p.cell_index) || shell.isothermal ||
                (owner>=0 && shell.shell_contents_id!=s.cells[owner].id) || (ch.rupture && shell.shell_contents_id.empty()))
                throw std::runtime_error("Battery discharge must cross its own physical shell; rupture cannot refer to contents");
            const double channel_area=gas_channel_area(s,ch);
            positive(channel_area,"channel area");positive(ch.discharge_coefficient,"discharge coefficient");positive(ch.strength_reference_k,"shell strength reference");positive(ch.strength_temperature_scale_k,"shell strength scale");
            if(channel_area>p.area_m2*(1.+1e-9) || ch.discharge_coefficient>1 || !std::isfinite(ch.opening_pressure_pa) || ch.opening_pressure_pa<0 ||
                !std::isfinite(ch.outside_pressure_pa)||ch.outside_pressure_pa<0)throw std::runtime_error("Invalid pressure/area of gas channel");
            face_patches(s,m,p);
            if(ch.opening.enabled) {
                const auto& o=ch.opening;
                if(o.deterministic_aperture) {
                    if(!g.split_transport || ch.shell_thermal_volume<0 || o.binary_pixels || o.stochastic_area || o.stochastic_pixels)
                        throw std::runtime_error("Deterministic aperture requires a local patch and split transport");
                    if(!std::isfinite(o.area_rate_per_s) || o.area_rate_per_s<0 || !std::isfinite(o.pressure_exponent) || o.pressure_exponent<1 ||
                        !std::isfinite(o.valve_radius_m) || o.valve_radius_m<0 || !std::isfinite(o.valve_u_m) || !std::isfinite(o.valve_v_m) ||
                        !std::isfinite(o.valve_pressure_factor) || o.valve_pressure_factor<=0 || o.valve_pressure_factor>1)
                        throw std::runtime_error("Invalid deterministic aperture law");
                    const int axis=p.surface[1]-'x',u=(axis+1)%3,v=(axis+2)%3;
                    const double w=shell.dimensions_m[u]/shell.thermal_divisions[u],h=shell.dimensions_m[v]/shell.thermal_divisions[v];
                    if(std::abs(o.valve_u_m)>=w/2 || std::abs(o.valve_v_m)>=h/2 ||
                        (o.valve_radius_m>0 && (std::abs(o.valve_u_m)+o.valve_radius_m>w/2 || std::abs(o.valve_v_m)+o.valve_radius_m>h/2)))
                        throw std::runtime_error("Valve must fit inside its local shell patch");
                    for(const auto& other:s.gas_channels)if(other.rupture && !other.opening.deterministic_aperture)
                        throw std::runtime_error("Do not mix leading deterministic and legacy trailing rupture operators");
                }
                if(!std::isfinite(o.fixed_area_fraction) || o.fixed_area_fraction<0 || o.fixed_area_fraction>1 ||
                    (o.fixed_area_fraction>0 && !o.stochastic_area))throw std::runtime_error("Invalid fixed area fraction");
                if((o.stochastic_pixels || o.stochastic_area) && ((!o.binary_pixels && !o.stochastic_area) || !std::isfinite(o.pixel_rate_per_s) || o.pixel_rate_per_s<0 ||
                    !std::isfinite(o.pixel_pressure_exponent) || o.pixel_pressure_exponent<1))
                    throw std::runtime_error("Invalid stochastic pixel rupture law");
                if(o.stochastic_area && (o.binary_pixels || o.stochastic_pixels || !std::isfinite(o.fragment_mean_area_m2) || o.fragment_mean_area_m2<=0))
                    throw std::runtime_error("Invalid stochastic area rupture law");
                if(o.binary_pixels && o.initial_mechanical_damage!=0. && o.initial_mechanical_damage!=1.)
                    throw std::runtime_error("Binary shell initial mechanical state must be zero or one");
                if(owner<0)throw std::runtime_error("Progressive shell opening requires its internal gas compartment");
                const auto core=s.cells[owner].position_m;
                if(p.normal.x*(shell.position_m.x-core.x)+p.normal.y*(shell.position_m.y-core.y)+p.normal.z*(shell.position_m.z-core.z)<=0)
                    throw std::runtime_error("Progressive shell opening requires the outward-facing panel port");
                if(!ch.rupture || ch.opening_pressure_pa<=0 || !std::isfinite(o.damage_onset) || o.damage_onset<0 || o.damage_onset>=1 ||
                    !std::isfinite(o.initial_mechanical_damage) || o.initial_mechanical_damage<0 || o.initial_mechanical_damage>1 ||
                    !std::isfinite(o.exponent) || o.exponent<1 || !std::isfinite(o.pressure_damage_rate_per_s) || o.pressure_damage_rate_per_s<0 ||
                    !std::isfinite(o.max_damage_step) || o.max_damage_step<=0 || o.max_damage_step>=1 ||
                    !std::isfinite(o.seam_width_m) || o.seam_width_m<0 ||
                    !std::isfinite(o.fracture_energy_j_per_m2) || o.fracture_energy_j_per_m2<0 ||
                    !std::isfinite(o.seam_strength_factor) || o.seam_strength_factor<=0 || o.seam_strength_factor>1)
                    throw std::runtime_error("Invalid progressive shell opening law");
                const int axis=p.surface[1]-'x';
                const double full=shell.dimensions_m[(axis+1)%3]*shell.dimensions_m[(axis+2)%3];
                if(shell.thermal_divisions[axis]!=1 || o.seam_width_m>=.5*std::min(shell.dimensions_m[(axis+1)%3],shell.dimensions_m[(axis+2)%3]))
                    throw std::runtime_error("Opening needs a thin panel and a seam narrower than half the face");
                if(std::abs(p.area_m2-full)>full*1e-9)
                    throw std::runtime_error("Progressive opening requires a full panel face port");
                // One panel damage field cannot simultaneously obey instant and
                // progressive rupture. Vents do not damage the panel.
                for(const auto& other:s.gas_channels)if(&other!=&ch && other.shell_object==ch.shell_object && other.rupture &&
                    (ch.shell_thermal_volume<0 || other.shell_thermal_volume<0 || ch.shell_thermal_volume==other.shell_thermal_volume))
                    throw std::runtime_error("Rupture channels overlap on a shell volume");
            }
        }
        for(const auto& contact:s.gas_contacts) {
            if(contact.compartment<0 || contact.compartment>=static_cast<int>(s.gas_compartments.size()) || !ports.contains(contact.port_id))throw std::runtime_error("Invalid gas thermal contact");
            positive(contact.h_w_per_m2_k,"gas contact h");const auto& p=*ports.at(contact.port_id);
            if(enclosed(p.cell_index) && s.gas_compartments[contact.compartment].owner_cell!=static_cast<int>(p.cell_index))
                throw std::runtime_error("External gas exposure cannot bypass the physical casing");
            if(s.cells[p.cell_index].isothermal)throw std::runtime_error("Gas exposure requires a spatial receiving solid");
            bool found=false;
            if(contact.thermal_volume< -1)throw std::runtime_error("Invalid gas contact thermal volume");
            for(const auto& patch:face_patches(s,m,p))if(contact.thermal_volume<0 || patch.volume==m.object_offsets[p.cell_index]+contact.thermal_volume) {
                m.gas_patches.push_back({patch.volume,contact.compartment,contact.h_w_per_m2_k*patch.area});found=true;
            }
            if(!found)throw std::runtime_error("Gas contact thermal volume is not on its port");
        }
    } else if(!s.gas_channels.empty() || !s.gas_contacts.empty())throw std::runtime_error("Gas links without compartments");
    append_aperture_surface_links(s,m);
    for(std::size_t i=0;i<s.cells.size();++i)if(s.cells[i].energy_outlet.enabled) {
        const auto found=ports.find(s.cells[i].energy_outlet_port);
        if(found==ports.end() || found->second->cell_index!=i || found->second->kind!="vent")
            throw std::runtime_error("Energy outlet requires its own explicitly declared vent port");
        face_patches(s,m,*found->second); // validate actual outlet geometry, even without gas cells
        for(const auto& e:s.interfaces)if(e.port_a==found->first || e.port_b==found->first)
            throw std::runtime_error("Outside-domain outlet port cannot simultaneously feed an internal contact");
        for(const auto& b:s.boundaries)if(b.port_id==found->first)
            throw std::runtime_error("Energy outlet and thermal boundary need distinct ports to avoid double counting");
    }
    m.interface_offsets.push_back(0);
    for(std::size_t j=0;j<s.interfaces.size();++j) {
        const auto& e=s.interfaces[j];
        if((enclosed(e.cell_a) && s.cells[e.cell_b].shell_contents_id!=s.cells[e.cell_a].id) ||
           (enclosed(e.cell_b) && s.cells[e.cell_a].shell_contents_id!=s.cells[e.cell_b].id))
            throw std::runtime_error("External thermal contact cannot bypass the physical casing: "+e.id);
        if(!ports.contains(e.port_a)||!ports.contains(e.port_b)) throw std::runtime_error("Missing contact port: "+e.id);
        const auto& pa=*ports.at(e.port_a); const auto& pb=*ports.at(e.port_b);
        const auto& ca=s.cells[e.cell_a]; const auto& cb=s.cells[e.cell_b];
        // Parallel aligned material frames, including a shared rigid rotation.
        for(int k=0;k<3;++k) {
            V axis{};axis[k]=1;
            auto ra=rotate(ca,axis),rb=rotate(cb,axis);
            for(int d=0;d<3;++d) if(std::abs(ra[d]-rb[d])>1e-7)
                throw std::runtime_error("Contact mesher requires aligned local frames: "+e.id);
        }
        auto a=face_patches(s,m,pa), b=face_patches(s,m,pb);
        if(a.front().axis!=b.front().axis || pa.surface[0]==pb.surface[0]) throw std::runtime_error("Contact faces must oppose: "+e.id);
        const int axis=a.front().axis,u=(axis+1)%3,v=(axis+2)%3;
        auto shift=local_point(ca,cb.position_m);
        const int first=static_cast<int>(m.links.size());
        double total_area=0;
        for(const auto& aa:a) for(const auto& bb:b) {
            const double du=std::min(aa.hi[u],bb.hi[u]+shift[u])-std::max(aa.lo[u],bb.lo[u]+shift[u]);
            const double dv=std::min(aa.hi[v],bb.hi[v]+shift[v])-std::max(aa.lo[v],bb.lo[v]+shift[v]);
            if(du<=0||dv<=0) continue;
            ThermalLink link; link.a=aa.volume; link.b=bb.volume; link.interface_index=static_cast<int>(j);
            link.law=e.law; link.weight=du*dv;
            link.geometry_axis=axis;
            link.law.endpoint_a=ca.surface_connectivity;link.law.endpoint_b=cb.surface_connectivity;
            link.series_resistance_k_per_w=(aa.half_resistance+bb.half_resistance)/(du*dv);
            link.series_resistance_a_k_per_w=aa.half_resistance/(du*dv);
            link.initial.l0_m=e.initial_phase==EdgePhase::Destroyed?e.law.destroyed_l0_m:e.law.l0_m;
            link.initial.l_m=e.initial_l_m; link.initial.damage=e.initial_damage; link.initial.phase=e.initial_phase;
            total_area+=link.weight; m.links.push_back(link);
        }
        if(total_area<=0) throw std::runtime_error("Contact patches do not overlap: "+e.id);
        for(int k=first;k<static_cast<int>(m.links.size());++k) {
            auto& link=m.links[k]; link.weight/=total_area; link.maximum_g=maximum_g(link.law)*link.weight;
            m.interface_links.push_back(k);
        }
        m.interface_offsets.push_back(static_cast<int>(m.interface_links.size()));
    }
    for(std::size_t boundary_index=0;boundary_index<s.boundaries.size();++boundary_index) {
        const auto& boundary=s.boundaries[boundary_index];
        const auto& profile=boundary.temperature_profile;
        if(!profile.empty()) {
            if(profile.size()<2 || profile.front()[0]!=0 || profile.back()[0]<s.settings.duration_s)
                throw std::runtime_error("Boundary profile must cover [0, duration]: "+boundary.id);
            double previous=-1;
            for(const auto& point:profile) {
                if(!std::isfinite(point[0]) || point[0]<=previous)
                    throw std::runtime_error("Boundary profile time must increase: "+boundary.id);
                positive(point[1],"boundary profile temperature");previous=point[0];
            }
        }
        if(!ports.contains(boundary.port_id)) throw std::runtime_error("Missing boundary port");
        const auto& port=*ports.at(boundary.port_id);
        if(enclosed(port.cell_index))throw std::runtime_error("External boundary cannot bypass the physical casing");
        const auto& c=s.cells[port.cell_index];
        std::vector<Patch> patches;
        if(port.surface=="case") {
            for(int axis=0;axis<3;++axis) for(int sign:{-1,1}) {
                PortConfig face=port;
                face.surface=std::string(sign>0?"+":"-")+"xyz"[axis];
                V pt{},nm{};pt[axis]=sign*c.dimensions_m[axis]/2;nm[axis]=sign;
                pt=rotate(c,pt);nm=rotate(c,nm);
                face.centroid_m={pt[0]+c.position_m.x,pt[1]+c.position_m.y,pt[2]+c.position_m.z};
                face.normal={nm[0],nm[1],nm[2]};
                face.area_m2=c.dimensions_m[(axis+1)%3]*c.dimensions_m[(axis+2)%3];
                auto part=face_patches(s,m,face);patches.insert(patches.end(),part.begin(),part.end());
            }
        } else patches=face_patches(s,m,port);
        double area=0; for(const auto& p:patches) area+=p.area;
        if(!std::isfinite(boundary.conductance_w_per_k)||boundary.conductance_w_per_k<0) throw std::runtime_error("Invalid boundary G");
        positive(boundary.temperature_k,"boundary temperature");
        for(const auto& p:patches) {
            auto link=constant_link(p.volume,-1,boundary.conductance_w_per_k*p.area/area);
            link.geometry_axis=p.axis;
            if(boundary.has_connectivity) {
                link.law=boundary.connectivity;link.weight=p.area/area;
                link.law.endpoint_a=c.surface_connectivity;link.law.endpoint_b=boundary.remote_surface_connectivity;
                link.initial.l0_m=link.law.l0_m;link.initial.l_m=link.law.l_low_m;
            }
            link.series_resistance_k_per_w=p.half_resistance/p.area;
            link.series_resistance_a_k_per_w=link.series_resistance_k_per_w;
            link.boundary_temperature_k=boundary.temperature_k;add_link(m,link);
            m.links.back().boundary_index=static_cast<int>(boundary_index);
        }
    }
    m.adjacency_offsets.assign(m.volumes.size()+1,0);
    for(auto& e:m.links) {
        if(e.b<0)e.region=ThermalLinkRegion::Boundary;
        else if(m.volumes[e.a].object!=m.volumes[e.b].object)e.region=ThermalLinkRegion::MaterialContact;
        else {
            const bool a=m.volumes[e.a].region==ThermalRegion::Core,b=m.volumes[e.b].region==ThermalRegion::Core;
            e.region=a&&b?ThermalLinkRegion::CoreCore:a!=b?ThermalLinkRegion::CoreShell:ThermalLinkRegion::ShellShell;
        }
    }
    for(const auto& e:m.links) {
        const auto& l=e.law;
        if(e.geometry_axis<0 || e.geometry_axis>2)throw std::runtime_error("Invalid geometry direction");
        for(const auto& r:{l.endpoint_a,l.endpoint_b}) {
            positive(r.reference_temperature_k,"structural reference temperature");positive(r.temperature_scale_k,"structural temperature scale");
            for(const auto v:{r.linear,r.quadratic,r.damage,r.temperature_contrast,r.damage_contrast})
                for(double x:{v.x,v.y,v.z})if(!std::isfinite(x))throw std::runtime_error("Non-finite structural response");
        }
        if(!std::isfinite(l.geometry_reference_logit) || !std::isfinite(l.destroyed_geometry_reference_logit))throw std::runtime_error("Non-finite structural reference");
        if((l.kind==TransitionKind::StateGeometry || l.kind==TransitionKind::DirectedGeometry) && (l.relaxation_time_s!=0 || l.damage_l_shift_m!=0 ||
           l.damage_conductance_factor_at_one!=1 || l.destroyed_l_m<l.destroyed_l0_m))
            throw std::runtime_error("State geometry requires instantaneous response and its explicit damage/destroyed branch");
        positive(l.activation_width_m,"activation width");positive(l.transition_width_k,"transition width");
        positive(l.damage_temperature_scale_k,"damage scale");positive(l.damage_conductance_factor_at_one,"damage conductance factor");
        if(!std::isfinite(l.endpoint_damage_weight)||l.endpoint_damage_weight<0||l.endpoint_damage_weight>1)
            throw std::runtime_error("endpoint_damage_weight must be in [0,1]");
        for(double x:{l.background_conductance_w_per_k,l.active_conductance_w_per_k,l.destroyed_background_conductance_w_per_k,
            l.destroyed_active_conductance_w_per_k,l.l0_m,l.l_low_m,l.l_high_m,l.destroyed_l0_m,l.destroyed_l_m,
            l.hysteresis_k,l.relaxation_time_s,l.damage_rate_at_onset_per_s,e.series_resistance_k_per_w,e.initial.l_m})
            if(!std::isfinite(x)||x<0) throw std::runtime_error("Invalid local conductivity law");
        if(!std::isfinite(l.damage_l_shift_m)||!std::isfinite(l.transition_temperature_k)||!std::isfinite(l.damage_onset_k)||
           !std::isfinite(e.initial.damage)||e.initial.damage<0||e.initial.damage>1||
           !std::isfinite(l.destruction_damage)||l.destruction_damage<=0||l.destruction_damage>1)
            throw std::runtime_error("Invalid local damage law");
    }
    for(const auto& e:m.links) { ++m.adjacency_offsets[e.a+1];if(e.b>=0) ++m.adjacency_offsets[e.b+1]; }
    for(std::size_t i=1;i<m.adjacency_offsets.size();++i) m.adjacency_offsets[i]+=m.adjacency_offsets[i-1];
    m.adjacency_links.resize(m.adjacency_offsets.back());
    auto cursor=m.adjacency_offsets;
    for(int j=0;j<static_cast<int>(m.links.size());++j) {
        const auto& e=m.links[j];m.adjacency_links[cursor[e.a]++]=j;
        if(e.b>=0) m.adjacency_links[cursor[e.b]++]=j;
    }
    for(const auto& event:s.primary_failures) {
        if(event.cell_index>=s.cells.size() || event.thermal_volume < -1 ||
           event.thermal_volume>=m.object_offsets[event.cell_index+1]-m.object_offsets[event.cell_index] ||
           !std::isfinite(event.time_s)||event.time_s<0) throw std::runtime_error("Invalid primary failure");
        if(!s.cells[event.cell_index].is_battery)throw std::runtime_error("Cannot ignite a passive material");
        if(s.cells[event.cell_index].conversion.enabled)throw std::runtime_error("Conversion is activated by its material length; use initial heating, not prescribed ignition");
    }
    return m;
}
} // namespace battery
