#pragma once
namespace battery {
// Bounded spatial interpolation of recorded/live gas values. The optical
// texture is advected by the accepted graph transfer, never by wall time.
inline constexpr const char* gas_fragment_shader=R"glsl(
#version 330 core
in vec3 vLocal;
out vec4 fragColor;
uniform vec3 uLocalEye,uCenter,uSize,uExcludeCenter,uExcludeSize;
uniform int uPolygonCount;
uniform int uCylinderCount;
uniform vec4 uCylinders[64]; // world centre.xyz, radius
uniform float uCylinderHeights[64];
uniform vec2 uPolygon[12];
uniform vec3 uGradient,uDisplacement;
uniform vec2 uBounds,uRange,uViewport;
uniform float uTemperature,uDensity,uThreshold;
uniform sampler1D uPalette;
uniform sampler2D uSceneDepth;
uniform mat4 uInverseVP;
bool occupied(vec3 p) {
    for(int k=0;k<uCylinderCount;++k) {
        vec3 d=p-uCylinders[k].xyz;
        if(dot(d.xy,d.xy)<uCylinders[k].w*uCylinders[k].w && abs(d.z)<uCylinderHeights[k]*.5)return false;
    }
    if(uExcludeSize.x>0.0 && all(lessThan(abs(p-uExcludeCenter),uExcludeSize*0.5)))return false;
    if(uPolygonCount>0) {
        bool positive=false,negative=false;
        for(int k=0;k<uPolygonCount;++k) {
            vec2 a=uPolygon[k],b=uPolygon[(k+1)%uPolygonCount];
            float d=(b.x-a.x)*(p.y-a.y)-(b.y-a.y)*(p.x-a.x);
            positive=positive||d>1e-8;negative=negative||d< -1e-8;
        }
        if(positive&&negative)return false;
    }
    return true;
}
float hash(vec3 p) {p=fract(p*.1031);p+=dot(p,p.yzx+33.33);return fract((p.x+p.y)*p.z);}
float noise(vec3 p) {
    vec3 a=floor(p),f=fract(p);f=f*f*(3.-2.*f);
    return mix(mix(mix(hash(a),hash(a+vec3(1,0,0)),f.x),mix(hash(a+vec3(0,1,0)),hash(a+vec3(1,1,0)),f.x),f.y),
               mix(mix(hash(a+vec3(0,0,1)),hash(a+vec3(1,0,1)),f.x),mix(hash(a+vec3(0,1,1)),hash(a+vec3(1,1,1)),f.x),f.y),f.z);
}
void main() {
    vec3 d=normalize(vLocal-uLocalEye);
    vec3 safeD=mix(vec3(1e-8),d,greaterThan(abs(d),vec3(1e-8)));
    vec3 a=(-vec3(.5)-uLocalEye)/safeD,b=(vec3(.5)-uLocalEye)/safeD;
    vec3 lo=min(a,b),hi=max(a,b);
    float start=max(0.,max(lo.x,max(lo.y,lo.z))),end=min(hi.x,min(hi.y,hi.z));
    vec2 uv=gl_FragCoord.xy/uViewport;
    float depth=texture(uSceneDepth,uv).r;
    if(depth<.999999) {
        vec4 world=uInverseVP*vec4(uv*2.-1.,depth*2.-1.,1.);
        vec3 local=(world.xyz/world.w-uCenter)/uSize;
        end=min(end,dot(local-uLocalEye,d));
    }
    if(end<=start)discard;
    float ds=(end-start)/24.,worldDs=ds*length(d*uSize);
    vec4 sum=vec4(0);
    for(int k=0;k<24;++k) {
        vec3 p=uCenter+(uLocalEye+d*(start+(float(k)+.5)*ds))*uSize;
        if(!occupied(p))continue;
        float t=clamp(uTemperature+dot(uGradient,p-uCenter),uBounds.x,uBounds.y);
        float hot=smoothstep(uThreshold,uThreshold+100.,t);
        vec3 q=(p-uDisplacement)/.007;
        // A mottled optical density, not a simulated turbulent velocity field.
        // Low-density gaps keep the batteries readable through the plume.
        float cloud=.04+2.8*smoothstep(.30,.78,.72*noise(q)+.28*noise(q*2.1+7.));
        float alpha=1.-exp(-uDensity*worldDs*hot*cloud);
        vec3 rgb=texture(uPalette,(clamp((t-uRange.x)/(uRange.y-uRange.x),0.,1.)*255.+.5)/256.).rgb;
        sum.rgb+=(1.-sum.a)*alpha*rgb;
        sum.a+=(1.-sum.a)*alpha;
    }
    if(sum.a<.002)discard;
    fragColor=vec4(sum.rgb/max(sum.a,1e-6),sum.a);
}
)glsl";
inline constexpr const char* gas_puff_vertex_shader=R"glsl(
#version 330 core
layout(location=0) in vec4 aPositionRadius;
layout(location=1) in vec4 aColor;
uniform mat4 uVP;
uniform float uPixelScale;
out vec4 vColor;
void main() {
    gl_Position=uVP*vec4(aPositionRadius.xyz,1);
    gl_PointSize=clamp(aPositionRadius.w*uPixelScale/max(gl_Position.w,.001),1.,64.);
    vColor=aColor;
}
)glsl";
inline constexpr const char* gas_puff_fragment_shader=R"glsl(
#version 330 core
in vec4 vColor;
out vec4 fragColor;
void main() {
    vec2 q=gl_PointCoord*2.-1.;float r=dot(q,q);if(r>=1.)discard;
    float soft=exp(-3.5*r)*(1.-smoothstep(.5,1.,r));
    fragColor=vec4(vColor.rgb,vColor.a*soft);
}
)glsl";
}
