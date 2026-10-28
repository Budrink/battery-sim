// Actual production pressure operator against a dense long-double reference.
// CPU arithmetic here is a test oracle, never a simulation backend.
#include "../src/batched_gas.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace kg=battery::kawasaki;
namespace bg=battery::batched_gas;
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>struct Buffer {
    T* p=nullptr;std::size_t n;
    explicit Buffer(std::size_t count):n(count){if(n)check(cudaMalloc(&p,n*sizeof(T)));}
    ~Buffer(){if(p)cudaFree(p);}
    void put(const std::vector<T>& v){require(v.size()==n,"Buffer size");if(n)check(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> get(){std::vector<T> v(n);if(n)check(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
__global__ void residual_scale_case(bg::AverageState* state,bg::Control* control,double* result) {
    __shared__ double shared[128];
    const auto grid=cooperative_groups::this_grid();const int i=threadIdx.x;
    const double value=bg::average_pressure_residual(grid,i,blockDim.x,2,state,shared,control);
    if(i==0)*result=value;
}
void near_vacuum_residual_scale() {
    Buffer<bg::AverageState> states(2);Buffer<bg::Control> control(1);Buffer<double> result(1);
    std::vector<bg::AverageState> a(2);
    a[0].pressure=.0094;a[0].correction=248124.;a[0].compliance=5.788e-12;a[0].residual=-7.835e-18;
    a[1].pressure=100000.;a[1].correction=-99999.;a[1].compliance=1e-11;a[1].residual=1e-13;
    states.put(a);control.put({{}});auto* s=states.p;auto* c=control.p;auto* r=result.p;
    void* args[]={&s,&c,&r};
    check(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(residual_scale_case),1,128,args));
    check(cudaDeviceSynchronize());
    const double measured=result.get()[0];
    require(std::abs(measured-1e-7)<1e-16,"Incoming and draining local inventory scales");
}
std::vector<long double> reference(std::vector<long double> a,std::vector<long double> b) {
    const int n=b.size();
    for(int i=0;i<n;++i)for(int j=0;j<=i;++j) {
        long double x=a[i*n+j];for(int k=0;k<j;++k)x-=a[i*n+k]*a[j*n+k];
        if(i==j){require(x>0.,"Pressure reference is not SPD");a[i*n+j]=sqrtl(x);}
        else a[i*n+j]=x/a[j*n+j];
    }
    for(int i=0;i<n;++i){for(int j=0;j<i;++j)b[i]-=a[i*n+j]*b[j];b[i]/=a[i*n+i];}
    for(int i=n-1;i>=0;--i){for(int j=i+1;j<n;++j)b[i]-=a[j*n+i]*b[j];b[i]/=a[i*n+i];}
    return b;
}
// Recycle a frozen production window (including a deliberately bad forecast).
// The learnt pattern must solve the same
// operator without CG work; a useless constant pattern must be rejected and
// fall back to the original solve. No alternate test pressure implementation.
__global__ void pattern_repeat(int count,int edge_count,const int* offsets,const int* ids,const kg::Edge* edges,
    const kg::Node* input,kg::Node* output,bg::Proposal* proposals,battery::GasChannelState* channels,
    kg::Settings settings,battery::GasModel model,double dt,bg::Control* control,bg::AverageState* a,bg::AverageState* b,
    int* error,unsigned* used) {
    const auto grid=cooperative_groups::this_grid();const int tid=blockIdx.x*blockDim.x+threadIdx.x,stride=gridDim.x*blockDim.x;
    __shared__ double shared[256];
    if(tid==0){*control={};*error=0;}grid.sync();
    for(int trial=0;trial<3;++trial) {
        if(trial==2){for(int i=tid;i<count;i+=stride)a[i].pressure_pattern=1.;
            if(tid==0)control->averaged_pressure_used=512;grid.sync();}
        if(!bg::average_transport(grid,tid,stride,count,edge_count,offsets,ids,edges,input,output,proposals,channels,
            settings,model,dt,dt,control,a,b,shared,error,{}))return;
        if(tid==0)used[trial]=control->averaged_first_linear_used;grid.sync();
    }
}
void fixture(bool graded,int blocks,bool chain=false,bool pattern=false) {
    const int count=chain?513:27;const double dt=.025;kg::Settings settings;
    std::vector<kg::Node> input;std::vector<kg::Edge> topology;
    std::vector<std::vector<int>> adjacency(count);
    for(int i=0;i<count;++i) {
        const double volume=chain?(i==0?.064:1e-8):(graded?(i==0?.064:1e-8*std::pow(10.,i%4)):1e-5);
        const double temperature=500.,pressure=101325.+(chain?(i==0?100.:0.):(i%7)*100.);
        const double mass=pressure*volume/(settings.R*temperature);
        input.push_back({mass,mass*settings.cv*temperature,volume});
        for(int axis=0,step=1;axis<(chain?1:3);++axis,step*=3)if(chain?i+1<count:(i/step)%3<2) {
            const int j=i+step,e=topology.size();kg::Edge edge{i,j,1e-4,.7,.02};
            topology.push_back(edge);adjacency[i].push_back(e);adjacency[j].push_back(e);
        }
    }
    std::vector<int> offset{0},incidence;
    for(const auto& row:adjacency){incidence.insert(incidence.end(),row.begin(),row.end());offset.push_back(incidence.size());}
    const int edge_count=topology.size();Buffer<kg::Node> nodes(count),scratch(count);nodes.put(input);
    Buffer<kg::Edge> edges(edge_count);edges.put(topology);Buffer<kg::Clock> clocks(edge_count);clocks.put(std::vector<kg::Clock>(edge_count));
    Buffer<bg::Proposal> proposals(edge_count);Buffer<bg::Budget> budgets(count);
    Buffer<battery::GasChannelState> channels(edge_count);std::vector<battery::GasChannelState> initial(edge_count);
    for(int e=0;e<edge_count;++e){initial[e].open=true;initial[e].effective_area_m2=topology[e].area;}channels.put(initial);
    Buffer<int> offsets(offset.size()),ids(incidence.size()),error(1);offsets.put(offset);ids.put(incidence);error.put({0});
    Buffer<bg::Control> control(1);control.put({{}});Buffer<bg::AverageState> a(count),b(count);
    battery::GasModel model;model.cuda_network_transport=model.cuda_batched_transport=model.averaged_network_transport=true;
    model.minimum_step_s=.005;model.averaged_pressure_iterations=2048;model.averaged_payload_iterations=128;
    model.mixing_diffusivity_m2_s=model.mixing_flow_fraction=0.;
    Buffer<unsigned> used(3);
    if(pattern){model.pressure_pattern_predictor=true;
        auto* in=nodes.p;auto* out=scratch.p;auto* e=edges.p;auto* o=offsets.p;auto* id=ids.p;
        auto* p=proposals.p;auto* c=channels.p;auto* ctl=control.p;auto* aa=a.p;auto* bb=b.p;auto* err=error.p;auto* u=used.p;
        void* args[]={const_cast<int*>(&count),const_cast<int*>(&edge_count),&o,&id,&e,&in,&out,&p,&c,&settings,&model,
            const_cast<double*>(&dt),&ctl,&aa,&bb,&err,&u};
        check(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(pattern_repeat),blocks,128,args));
    }else check(bg::launch(count,edge_count,offsets.p,ids.p,edges.p,nodes.p,scratch.p,proposals.p,budgets.p,clocks.p,
        channels.p,settings,model,dt,error.p,control.p,128,blocks,{},a.p,b.p));
    check(cudaDeviceSynchronize());const auto c=control.get()[0];
    require(error.get()[0]==0,"Production pressure window rejected");
    require(c.averaged_pressure_residual<=1e-6,"True local residual failed");
    if(pattern){const auto work=used.get();
        require(work[0]>128 && work[1]==0 && work[2]>128,"Pressure pattern did not skip the repeated solve / fall back");
        require(c.pressure_pattern_attempts==2 && c.pressure_pattern_accepted==1,"Pattern acceptance accounting");
        std::cout<<"Pattern CG iterations "<<work[0]<<" -> "<<work[1]<<"; rejected constant -> "<<work[2]<<'\n';}
    const auto state=a.get();const auto flux=proposals.get();std::vector<long double> matrix(count*count),rhs(count);
    for(int i=0;i<count;++i)matrix[i*count+i]=state[i].compliance;
    for(int e=0;e<edge_count;++e) {
        const int i=topology[e].a,j=topology[e].b;const long double k=dt*flux[e].cap;
        matrix[i*count+i]+=k;matrix[j*count+j]+=k;matrix[i*count+j]-=k;matrix[j*count+i]-=k;
        const long double source=k*(kg::pressure(input[i],settings)-kg::pressure(input[j],settings));rhs[i]-=source;rhs[j]+=source;
    }
    const auto expected=reference(matrix,rhs);double worst=0.;
    for(int i=0;i<count;++i)worst=std::max(worst,double(fabsl(expected[i]-state[i].correction)/kg::pressure(input[i],settings)));
    require(worst<1e-6,"Production pressure differs from independent dense reference");
    const auto output=pattern?scratch.get():nodes.get();double mass=0.,energy=0.;
    for(int i=0;i<count;++i){mass+=output[i].mass-input[i].mass;energy+=output[i].energy-input[i].energy;}
    require(std::abs(mass)<1e-12 && std::abs(energy)<1e-7,"Pressure window loses mass or heat");
    if(chain)require(c.averaged_pressure_used>128,"Long-path regression no longer exercises CG history beyond 128 iterations");
    std::cout<<"Pressure dense reference: nodes="<<count<<" graded="<<graded<<" blocks="<<blocks<<" relative_error="<<worst
        <<" true_residual="<<c.averaged_pressure_residual<<" iterations="<<c.averaged_pressure_used<<'\n';
}
// Independent host Newton oracle for the NONLINEAR three-volume equations.
// Uses finite-difference Jacobians + pivoted elimination, not production PCG
// or its mobility update. It is test arithmetic only, never a CPU gas backend.
std::vector<double> nonlinear_reference(const std::vector<kg::Node>& input,
    const std::vector<kg::Edge>& edges,const kg::Settings& settings,double dt) {
    const int n=input.size();std::vector<double> p{1.5e6,1e6,103000.},compliance(n),initial(n);
    for(int i=0;i<n;++i){const auto& v=input[i];initial[i]=kg::pressure(v,settings);
        compliance[i]=v.volume/(settings.R*kg::temperature(v,settings)*(1+v.mass*settings.R/(v.solid_capacity+settings.cv*v.mass)));}
    auto residual=[&](const std::vector<double>& pressure){std::vector<double> r(n);auto state=input;
        for(int i=0;i<n;++i){const double t=kg::temperature(input[i],settings);
            state[i].mass=pressure[i]*state[i].volume/(settings.R*t);
            state[i].energy=(state[i].solid_capacity+settings.cv*state[i].mass)*t;
            r[i]=compliance[i]*(pressure[i]-initial[i]);}
        for(const auto& e:edges){const double q=dt*(pressure[e.a]>=pressure[e.b]?
            kg::flow(state[e.a],state[e.b],e,settings):-kg::flow(state[e.b],state[e.a],e,settings));r[e.a]+=q;r[e.b]-=q;}
        for(int i=0;i<n;++i)r[i]/=input[i].mass;
        return r;};
    auto norm=[](const auto& r){double result=0.;for(auto v:r)result=std::max(result,std::abs(v));return result;};
    for(int iteration=0;iteration<80;++iteration){auto r=residual(p);const double before=norm(r);
        if(before<1e-10)return p;
        std::vector<double> jac(n*n),rhs(n);for(int i=0;i<n;++i)rhs[i]=-r[i];
        for(int j=0;j<n;++j){auto moved=p;const double h=1e-6*p[j];moved[j]+=h;auto rr=residual(moved);
            for(int i=0;i<n;++i)jac[i*n+j]=(rr[i]-r[i])/h;}
        for(int j=0;j<n;++j){int pivot=j;for(int i=j+1;i<n;++i)if(std::abs(jac[i*n+j])>std::abs(jac[pivot*n+j]))pivot=i;
            for(int k=0;k<n;++k)std::swap(jac[j*n+k],jac[pivot*n+k]);std::swap(rhs[j],rhs[pivot]);
            require(std::abs(jac[j*n+j])>1e-30,"Singular nonlinear reference Jacobian");
            for(int i=j+1;i<n;++i){const double f=jac[i*n+j]/jac[j*n+j];
                for(int k=j;k<n;++k)jac[i*n+k]-=f*jac[j*n+k];rhs[i]-=f*rhs[j];}}
        for(int i=n-1;i>=0;--i){for(int j=i+1;j<n;++j)rhs[i]-=jac[i*n+j]*rhs[j];rhs[i]/=jac[i*n+i];}
        bool accepted=false;for(double alpha=1.;alpha>1e-10;alpha*=.5){auto candidate=p;bool positive=true;
            for(int i=0;i<n;++i){candidate[i]+=alpha*rhs[i];positive=positive && candidate[i]>0.;}
            if(positive && norm(residual(candidate))<before){p=candidate;accepted=true;break;}}
        require(accepted,"Nonlinear reference line search stalled");
    }
    throw std::runtime_error("Nonlinear reference did not converge");
}
void transit_restriction(double area_scale,double dt,int blocks) {
    const int count=3,edge_count=2;kg::Settings settings;std::vector<kg::Node> input;
    for(int i=0;i<count;++i){const double volume=i==2?.01:1e-5,pressure=i==0?3e6:101325.,temperature=500.;
        const double mass=pressure*volume/(settings.R*temperature);input.push_back({mass,mass*settings.cv*temperature,volume});}
    const std::vector<kg::Edge> topology{{0,1,1e-5,.7,.02},{1,2,1.55e-6*area_scale,.7,.02}};
    Buffer<kg::Node> nodes(count),scratch(count);nodes.put(input);Buffer<kg::Edge> edges(edge_count);edges.put(topology);
    Buffer<kg::Clock> clocks(edge_count);clocks.put(std::vector<kg::Clock>(edge_count));
    Buffer<bg::Proposal> proposals(edge_count);Buffer<bg::Budget> budgets(count);
    Buffer<battery::GasChannelState> channels(edge_count);std::vector<battery::GasChannelState> initial(edge_count);
    for(int e=0;e<edge_count;++e){initial[e].open=true;initial[e].effective_area_m2=topology[e].area;}channels.put(initial);
    Buffer<int> offsets(4),ids(4),error(1);offsets.put({0,1,3,4});ids.put({0,0,1,1});error.put({0});
    Buffer<bg::Control> control(1);control.put({{}});Buffer<bg::AverageState> a(count),b(count);
    battery::GasModel model;model.cuda_network_transport=model.cuda_batched_transport=model.averaged_network_transport=true;
    model.minimum_step_s=.005;model.averaged_pressure_iterations=2048;model.averaged_payload_iterations=128;
    model.mixing_diffusivity_m2_s=model.mixing_flow_fraction=0.;
    check(bg::launch(count,edge_count,offsets.p,ids.p,edges.p,nodes.p,scratch.p,proposals.p,budgets.p,clocks.p,
        channels.p,settings,model,dt,error.p,control.p,128,blocks,{},a.p,b.p));check(cudaDeviceSynchronize());
    require(error.get()[0]==0,"Restricted nonlinear CUDA window rejected");
    const auto state=a.get();const auto expected=nonlinear_reference(input,topology,settings,dt);
    const auto flux=proposals.get();const auto output=nodes.get();const auto ctl=control.get()[0];
    require(ctl.averaged_scale==1.,"Regression concealed a bad flow by entropy scaling");
    double mass=0.,energy=0.;
    for(int i=0;i<count;++i){require(std::abs(state[i].pressure/expected[i]-1.)<.01,"Pressure disagrees with nonlinear reference");
        mass+=output[i].mass-input[i].mass;energy+=output[i].energy-input[i].energy;}
    require(std::abs(mass)<1e-12 && std::abs(energy)<1e-7,"Nonlinear solve loses mass or heat");
    const double expected_out=state[2].compliance*(expected[2]-kg::pressure(input[2],settings));
    require(std::abs(flux[1].mass/expected_out-1.)<.015,"Outlet exceeds nonlinear orifice allowance");
    require(kg::pressure(output[1],settings)>kg::pressure(output[2],settings),"Transit emptied below the receiving pressure");
    std::cout<<"Nonlinear restriction: area="<<area_scale<<" dt="<<dt<<" blocks="<<blocks
        <<" outlet_kg_s="<<flux[1].mass/dt<<" reference_kg_s="<<expected_out/dt
        <<" outer_iterations="<<ctl.averaged_nonlinear_used<<" law_defect="<<ctl.averaged_flow_defect<<'\n';
}
int main(){try{
    near_vacuum_residual_scale();
    for(int blocks:{1,2})for(double area:{1.,.1})for(double dt:{.005,.01,.025})transit_restriction(area,dt,blocks);
    for(bool graded:{false,true})for(int blocks:{1,2})fixture(graded,blocks);
    for(int blocks:{1,2}){fixture(true,blocks,true);fixture(true,blocks,true,true);}}
catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
