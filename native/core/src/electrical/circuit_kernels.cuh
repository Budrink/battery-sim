// Included in the solver's device namespace after Material/ElectricalState.
// One deterministic MNA solve for a small module; no inter-block partial reads.
__global__ void circuit_kernel(int nodes, const CircuitBranch* branches, int count, double* resistance,
                               double* matrix, double* rhs, double* current, double* voltage,
                               CurrentPoint* terminal_voltage, Material* materials, ElectricalState* state) {
    if (blockIdx.x || threadIdx.x)
        return;
    bool valid = true;
    for (int k = 0; k < count; ++k) {
        const auto& b = branches[k];
        resistance[k] = b.kind == CircuitKind::Battery ? state[b.object].resistance_ohm : 0;
        if (b.kind == CircuitKind::Battery && state[b.object].error)
            valid = false;
    }
    valid = valid && solve_circuit(nodes, branches, count, resistance, matrix, rhs, current, voltage);
    for (int k = 0; k < count; ++k) {
        const auto& b = branches[k];
        if (b.kind != CircuitKind::Battery)
            continue;
        auto& e = state[b.object];
        auto& m = materials[b.object];
        if (!valid || current[k] < -1e-10 || voltage[k] < -1e-10) {
            e.error = 4; // reject candidate: singular circuit, KCL/work error or unsupported discharge
            m.electrical.current_a = 0;
            continue;
        }
        const double i = fmax(0., current[k]);
        terminal_voltage[b.object] = {0., fmax(0., voltage[k])};
        m.electrical.current_a = i;
        m.current = {nullptr, 0};
        m.voltage = {terminal_voltage + b.object, 1};
        e.current_a = e.outlet_current_a = i;
        e.ohmic_drop_v = i * e.resistance_ohm;
        e.power_w = i * i * e.resistance_ohm;
    }
}
