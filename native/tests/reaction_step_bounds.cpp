#include "battery_native/reaction_step_bounds.hpp"
#include "battery_native/time_step_policy.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
void require(bool x,const char* text){if(!x)throw std::runtime_error(text);}
int main(){try {
    using namespace battery;
    require(bounded_time_step(1e-8,.079,.2,.005)==.005,"accuracy estimate ignored timestep floor");
    require(bounded_time_step(1e-8,.001,.2,.005)<0.,"timestep floor bypassed hard conduction stability");
    require(bounded_time_step(1e-8,.079,.001,.005)==.001,"exact external alignment was lost");
    require(retry_time_cap(.02,.005,.005)==.005,"retry safety margin bypassed immutable floor");
    require(retry_time_cap(.001,.0001,.001)==.001,"retry subdivided an already short external alignment");
    require(retry_time_cap(.02,.005,0.)==.95*.005,"reference retry policy changed");
    double retry_cap=.02;
    for(double proposed:{.01,.0051,.005,1e-9}) {
        retry_cap=retry_time_cap(retry_cap,proposed,.005);
        require(retry_cap>=.005,"repeated retries eroded immutable lower bound");
    }
    for(double duration:{.005,.009,.01,.0792843,.2}) {
        double elapsed=0.;int intervals=0;
        while(elapsed<duration) {
            const double h=balanced_transport_step(duration-elapsed,1e-9,.005);
            require(h>=.005*(1-1e-10),"gas graph reintroduced adaptive microsteps or a tiny tail");
            elapsed+=h;require(++intervals<=41,"gas graph step count is unbounded");
        }
    }
    require(completion_rate_bound(0.,0.,1.,.2)==0.,"neutral precursor feed fabricated a heat rate");
    const double bounded=completion_rate_bound(.01,0.,1.,.2);
    require(bounded>0. && bounded<.002,"slow R2 treated all fresh R1 product as instantaneous heat");
    for(double rate:{.001,.01,1.,100.,10000.})for(double horizon:{.001,.02,.2,1.}) {
        const double bound=completion_rate_bound(rate,.1,2.,horizon);
        for(double ratio:{.001,.1,.5,1.}) {
            const double h=horizon*ratio;
            const double old=.1*(-std::expm1(-rate*h));
            const double fresh=2.*(h+std::expm1(-rate*h)/rate);
            require(old+fresh<=bound*h+1e-12,"fresh completion heat bound is not conservative");
        }
    }
    std::cout<<"reaction feed bound: neutral R1, finite R2 time, conservative fresh-stock/charge heat pass\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
