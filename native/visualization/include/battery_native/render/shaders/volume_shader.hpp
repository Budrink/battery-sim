#pragma once

namespace battery {
// A macro-object is one optical volume, independently of its solver grid.
inline constexpr const char* volume_vertex_shader = R"glsl(
#version 330 core
layout(location=0) in vec3 aPosition;
uniform mat4 uMVP;
uniform float uCut;
out vec3 vLocal;
void main() {
    vLocal=aPosition;
    vLocal.x=(aPosition.x+0.5)*uCut-0.5;
    gl_Position=uMVP*vec4(vLocal,1.0);
}
)glsl";
inline constexpr const char* volume_fragment_shader = R"glsl(
#version 330 core
in vec3 vLocal;
out vec4 fragColor;
uniform samplerBuffer uSamples;
uniform sampler1D uPalette;
uniform ivec3 uGrid;
uniform vec3 uFirstCenter, uSecondCenter, uLastCenter, uPenultimateCenter;
uniform vec3 uFirstCut, uGridSpacing;
uniform int uOffset;
uniform int uLayer;
uniform bool uCylinder;
uniform vec3 uLocalEye;
uniform vec3 uDestroyed;
uniform vec2 uRange;
uniform float uOpacity;
uniform vec4 uMaterialTint; // alpha enables material base + thermal/damage overlay
uniform bool uMaterialDamage;
uniform int uPlateAxis;
uniform float uCut;
uniform int uHoleCount;
uniform int uHoleAxis;
uniform vec3 uDimensions;
uniform vec4 uHoles[64]; // local centre (m), radius (m)
uniform vec2 uHoleBounds[64];
bool opening(vec3 p) {
    int u=(uHoleAxis+1)%3,v=(uHoleAxis+2)%3;
    vec3 physical=p*uDimensions;
    for(int k=0;k<uHoleCount;++k) {
        vec2 d=vec2(physical[u]-uHoles[k][u],physical[v]-uHoles[k][v]);
        if(all(lessThanEqual(abs(d),uHoleBounds[k])) && (uHoles[k].w<0.0 || length(d)<=uHoles[k].w))return true;
    }
    return false;
}
vec4 sampleAt(ivec3 p) {
    p=clamp(p,ivec3(0),uGrid-1);
    return texelFetch(uSamples,uOffset+p.x+uGrid.x*(p.y+uGrid.y*p.z));
}
vec4 field(vec3 p) {
    // Casing edge tiles include corner strips. Interpolate at actual centres,
    // and classify destruction at actual cuts, not a stretched uniform grid.
    vec3 physical=p*uDimensions;
    vec3 q=vec3(0);
    for(int d=0;d<3;++d) if(uGrid[d]>1) {
        q[d]=1.0+(physical[d]-uSecondCenter[d])/uGridSpacing[d];
        if(physical[d]<uSecondCenter[d])
            q[d]=(physical[d]-uFirstCenter[d])/(uSecondCenter[d]-uFirstCenter[d]);
        if(physical[d]>uPenultimateCenter[d])
            q[d]=float(uGrid[d]-2)+(physical[d]-uPenultimateCenter[d])/(uLastCenter[d]-uPenultimateCenter[d]);
    }
    q=clamp(q,vec3(0),vec3(uGrid-1));
    ivec3 a=ivec3(floor(q));vec3 f=fract(q);
    vec4 c=mix(mix(mix(sampleAt(a),sampleAt(a+ivec3(1,0,0)),f.x),
                   mix(sampleAt(a+ivec3(0,1,0)),sampleAt(a+ivec3(1,1,0)),f.x),f.y),
               mix(mix(sampleAt(a+ivec3(0,0,1)),sampleAt(a+ivec3(1,0,1)),f.x),
                   mix(sampleAt(a+ivec3(0,1,1)),sampleAt(a+ivec3(1,1,1)),f.x),f.y),f.z);
    // Destruction is categorical, not another point on either scalar palette.
    c.z=sampleAt(ivec3(floor(vec3(1)+(physical-uFirstCut)/uGridSpacing))).z;
    return c;
}
void main() {
    vec3 d=normalize(vLocal-uLocalEye);
    vec3 safeD=mix(vec3(1e-8),d,greaterThan(abs(d),vec3(1e-8)));
    vec3 a=(vec3(-0.5)-uLocalEye)/safeD;
    vec3 b=(vec3(uCut-0.5,0.5,0.5)-uLocalEye)/safeD;
    vec3 lo=min(a,b),hi=max(a,b);
    float entry=max(0.0,max(lo.x,max(lo.y,lo.z)));
    float exit=min(hi.x,min(hi.y,hi.z));
    if(exit<=entry)discard;
    const int steps=24;
    float ds=(exit-entry)/float(steps);
    float alpha=1.0-exp(log(max(1.0-uOpacity,1e-5))*ds);
    vec4 sum=vec4(0);
    for(int i=0;i<steps;++i) {
        vec3 point=uLocalEye+d*(entry+(float(i)+0.5)*ds);
        if(uCylinder && dot(point.xy,point.xy)>.25)continue;
        if(opening(point))continue;
        vec4 s=field(point);
        float t=uLayer==0?(s.x-uRange.x)/max(uRange.y-uRange.x,1.0):s.y;
        vec3 rgb=texture(uPalette,(clamp(t,0.0,1.0)*255.0+0.5)/256.0).rgb;
        if(uMaterialTint.a>0.5) {
            // Cold pads retain their material identity; heating overlays the scalar palette.
            float mask=uLayer==0?smoothstep(0.0,0.25,max(t,0.0)):clamp(s.y,0.0,1.0);
            rgb=mix(uMaterialTint.rgb,rgb,mask);
        }
        if(uMaterialDamage) {
            float coordinate=point.x+point.y+point.z;
            if(uPlateAxis>=0)coordinate=point[(uPlateAxis+1)%3]+point[(uPlateAxis+2)%3];
            float stripes=step(.68,fract(coordinate*9.0));
            rgb=mix(rgb,uDestroyed,stripes*clamp(s.y,0.0,1.0)*.9);
            if(s.z>.5 || s.y>.995)rgb=uDestroyed;
        }
        if(uLayer!=0 && s.z>0.5)rgb=uDestroyed;
        sum.rgb+=(1.0-sum.a)*alpha*rgb;
        sum.a+=(1.0-sum.a)*alpha;
    }
    fragColor=vec4(sum.rgb/max(sum.a,1e-8),sum.a);
}
)glsl";
} // namespace battery
