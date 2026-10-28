#pragma once
// Secant of the EXISTING compressible-orifice law at a pressure iterate.
// Temperature and local polytropic compliance are frozen for this transport
// window; carrier enthalpy is subsequently solved conservatively. This is not
// an isothermal replacement for the energy step.
__device__ inline double pressure_mobility(const kg::Node& left,const kg::Node& right,
    double pl,double pr,const kg::Edge& edge,const kg::Settings& settings) {
    if(!(edge.area>0. && edge.cd>0.))return 0.;
    const bool forward=pl>=pr;
    auto d=forward?left:right,r=forward?right:left;
    double pd=fmax(pl,pr),pb=fmin(pl,pr);
    const double gap=pd-pb,epsilon=fmax(1e-6,1e-6*pd);
    // Same finite equilibrium tangent as before, but re-evaluated when an
    // initially quiet transit volume receives pressure from upstream.
    pd=fmax(pd,pb+epsilon);
    const double td=kg::temperature(d,settings),tr=kg::temperature(r,settings);
    d.mass=pd*d.volume/(settings.R*td);r.mass=pb*r.volume/(settings.R*tr);
    d.energy=(d.solid_capacity+settings.cv*d.mass)*td;
    r.energy=(r.solid_capacity+settings.cv*r.mass)*tr;
    return kg::flow(d,r,edge,settings)/fmax(gap,epsilon);
}

// Relative channel-law defect with a 1 ppm resident-inventory absolute floor.
// 1% flow consistency is stricter than the accepted 5% coarse transport error.
// It does NOT relax global conservation ledgers or the linear pressure solve.
__device__ inline double pressure_flow_defect(double old_cap,double new_cap,double gap,
    double dt,double left_mass,double right_mass) {
    const double expected=fabs(new_cap*gap)*dt;
    const double tolerance=.01*expected+1e-6*fmin(left_mass,right_mass);
    return fabs((old_cap-new_cap)*gap)*dt/fmax(tolerance,1e-300);
}
