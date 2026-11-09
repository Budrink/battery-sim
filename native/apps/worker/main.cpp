#include "battery_job/job.hpp"
#include <csignal>
#include <iostream>

namespace {
volatile std::sig_atomic_t cancelled = 0;
void request_cancel(int) {
    cancelled = 1;
}
} // namespace
int main(int argc, char** argv) {
    if (argc != 4 || std::string(argv[2]) != "--output") {
        std::cerr << "usage: battery_sim_worker JOB.json --output NEW_ATTEMPT_DIRECTORY\n";
        return 2;
    }
    std::signal(SIGINT, request_cancel);
    std::signal(SIGTERM, request_cancel);
    // A disconnected supervisor must not terminate the worker inside a CUDA
    // step. The checked stream will report an I/O failure at the next emission.
    std::signal(SIGPIPE, SIG_IGN);
    return battery::job::execute(argv[1], argv[3], std::cout, [] { return cancelled != 0; });
}
