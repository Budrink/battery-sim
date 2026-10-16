#pragma once
#include "battery_native/shell_opening.hpp"
#include <cmath>
#ifdef __CUDACC__
#define APERTURE_HD __host__ __device__
#else
#define APERTURE_HD
#endif
namespace battery {
struct OpeningRectangle {double width=0.,height=0.,bands[4]{};};
inline OpeningRectangle opening_rectangle(double u,double v,double w,double h,double panel_w,double panel_h,double band) {
    OpeningRectangle p;p.width=w;p.height=h;
    p.bands[0]=fmax(0.,fmin(w,-panel_w/2+band-(u-w/2)));
    p.bands[1]=fmax(0.,fmin(w,u+w/2-(panel_w/2-band)));
    p.bands[2]=fmax(0.,fmin(h,-panel_h/2+band-(v-h/2)));
    p.bands[3]=fmax(0.,fmin(h,v+h/2-(panel_h/2-band)));
    return p;
}
APERTURE_HD inline double strip_area(const OpeningRectangle& p,double scale,double growth) {
    const double x=(p.bands[0]+p.bands[1])*scale+growth*((p.bands[0]>0)+(p.bands[1]>0));
    const double y=(p.bands[2]+p.bands[3])*scale+growth*((p.bands[2]>0)+(p.bands[3]>0));
    return p.width*p.height-fmax(0.,p.width-x)*fmax(0.,p.height-y);
}
// A partially opened seam is a narrowing perimeter strip. Once the full seam
// is open, rectangular fronts expand from its occupied edges, never neighbours.
inline void opened_strips(const OpeningRectangle& p,double fraction,double* widths) {
    const double target=fmin(1.,fmax(0.,fraction))*p.width*p.height;
    const bool seam_only=target<=strip_area(p,1.,0.);
    double lo=0.,hi=seam_only?1.:fmax(p.width,p.height);
    for(int k=0;k<48;++k) {const double mid=(lo+hi)/2;
        if(strip_area(p,seam_only?mid:1.,seam_only?0.:mid)<target)lo=mid;else hi=mid;}
    const double x=(lo+hi)/2;
    for(int k=0;k<4;++k)widths[k]=p.bands[k]>0?fmin(k<2?p.width:p.height,p.bands[k]*(seam_only?x:1.)+(seam_only?0.:x)):0.;
}
// Integral over a positive rectangle [0,x] x [0,y] intersected with a disk.
APERTURE_HD inline double disk_quadrant(double x,double y,double r) {
    if(r<=0 || x<=0 || y<=0)return 0.;
    x=fmin(x,r);y=fmin(y,r);
    const double t=sqrt(fmax(0.,r*r-y*y));
    if(x<=t)return x*y;
    const double jx=.5*(x*sqrt(fmax(0.,r*r-x*x))+r*r*asin(fmin(1.,x/r)));
    const double jt=.5*(t*y+r*r*asin(fmin(1.,t/r)));
    return t*y+jx-jt;
}
APERTURE_HD inline double signed_disk_quadrant(double x,double y,double r) {
    return (x<0?-1.:1.)*(y<0?-1.:1.)*disk_quadrant(fabs(x),fabs(y),r);
}
APERTURE_HD inline double clipped_disk_area(double width,double height,double u,double v,double r) {
    const double x0=-width/2-u,x1=width/2-u,y0=-height/2-v,y1=height/2-v;
    return fmin(width*height,fmax(0.,signed_disk_quadrant(x1,y1,r)-signed_disk_quadrant(x0,y1,r)
        -signed_disk_quadrant(x1,y0,r)+signed_disk_quadrant(x0,y0,r)));
}
// Radius is diagnostic geometry, never a second aperture state.
inline double clipped_disk_radius(double width,double height,double u,double v,double fraction) {
    double lo=0.,hi=std::hypot(width/2+std::abs(u),height/2+std::abs(v));
    if(fraction<=0)return 0.;if(fraction>=1)return hi;
    for(int k=0;k<48;++k) {const double mid=(lo+hi)/2;
        if(clipped_disk_area(width,height,u,v,mid)<fraction*width*height)lo=mid;else hi=mid;}
    return (lo+hi)/2;
}
APERTURE_HD inline double aperture_rate(const ShellOpeningLaw& law,double dp,double strength) {
    return law.area_rate_per_s*pow(fmax(0.,dp/strength-1.),law.pressure_exponent);
}
// Exact frozen-pressure integration. A seam/valve opens before the stronger
// matrix; unused time after reaching its edge belongs to matrix growth.
APERTURE_HD inline double deterministic_fraction(const ShellOpeningLaw& law,double a,double dp,double strength,
    double weak_fraction,double weak_factor,double dt,double speed=1.) {
    if(dt<=0 || a>=1.)return a;
    if(a<weak_fraction) {
        const double rate=speed*aperture_rate(law,dp,strength*weak_factor);
        if(rate<=0)return a;
        const double crossing=log((1.-a)/(1.-weak_fraction))/rate;
        if(dt<crossing)return fmin(weak_fraction,advance_shell_damage(a,rate,dt));
        a=weak_fraction;dt-=crossing;
    }
    return advance_shell_damage(a,speed*aperture_rate(law,dp,strength),dt);
}
}
#undef APERTURE_HD
