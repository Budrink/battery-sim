#pragma once
#include "battery_native/electrical/circuit_model.hpp"
#include <cmath>
#ifdef __CUDACC__
#define CIRCUIT_HD __host__ __device__
#else
#define CIRCUIT_HD
#endif

namespace battery {
CIRCUIT_HD inline bool circuit_finite(double x) {
#ifdef __CUDA_ARCH__
    return isfinite(x);
#else
    return std::isfinite(x);
#endif
}
// Dense modified nodal analysis for small module circuits. Workspace is caller-owned;
// no allocations, CUDA atomics, host synchronization or hidden conductance floors.
CIRCUIT_HD inline int circuit_unknowns(int nodes, const CircuitBranch* b, int count) {
    int n = nodes - 1;
    for (int k = 0; k < count; ++k)
        n += b[k].kind == CircuitKind::VoltageSource;
    return n;
}
CIRCUIT_HD inline bool solve_circuit(int nodes, const CircuitBranch* branches, int count,
                                     const double* resistance, double* matrix, double* rhs, double* currents,
                                     double* voltages) {
    const int n = circuit_unknowns(nodes, branches, count);
    if (n < 1 || n > 128)
        return false;
    for (int i = 0; i < n * n; ++i)
        matrix[i] = 0;
    for (int i = 0; i < n; ++i)
        rhs[i] = 0;
    int source = nodes - 1;
    for (int k = 0; k < count; ++k) {
        const auto& b = branches[k];
        const int a = b.from - 1, z = b.to - 1;
        if (b.kind == CircuitKind::Battery) {
            const double r = resistance[k];
            if (!(r > 0) || !circuit_finite(r))
                return false;
            const double g = 1 / r;
            if (a >= 0) {
                matrix[a * n + a] += g;
                rhs[a] += g * b.value;
            }
            if (z >= 0) {
                matrix[z * n + z] += g;
                rhs[z] -= g * b.value;
            }
            if (a >= 0 && z >= 0) {
                matrix[a * n + z] -= g;
                matrix[z * n + a] -= g;
            }
        } else if (b.kind == CircuitKind::CurrentSource) {
            if (a >= 0)
                rhs[a] -= b.value;
            if (z >= 0)
                rhs[z] += b.value;
        } else {
            if (a >= 0)
                matrix[a * n + source] = matrix[source * n + a] = 1;
            if (z >= 0)
                matrix[z * n + source] = matrix[source * n + z] = -1;
            rhs[source++] = b.value;
        }
    }
    for (int col = 0; col < n; ++col) {
        int pivot = col;
        for (int row = col + 1; row < n; ++row)
            if (::fabs(matrix[row * n + col]) > ::fabs(matrix[pivot * n + col]))
                pivot = row;
        if (!(::fabs(matrix[pivot * n + col]) > 1e-20))
            return false;
        if (pivot != col) {
            for (int j = col; j < n; ++j) {
                const double tmp = matrix[col * n + j];
                matrix[col * n + j] = matrix[pivot * n + j];
                matrix[pivot * n + j] = tmp;
            }
            const double tmp = rhs[col];
            rhs[col] = rhs[pivot];
            rhs[pivot] = tmp;
        }
        for (int row = col + 1; row < n; ++row) {
            const double f = matrix[row * n + col] / matrix[col * n + col];
            for (int j = col + 1; j < n; ++j)
                matrix[row * n + j] -= f * matrix[col * n + j];
            rhs[row] -= f * rhs[col];
        }
    }
    for (int row = n - 1; row >= 0; --row) {
        for (int j = row + 1; j < n; ++j)
            rhs[row] -= matrix[row * n + j] * rhs[j];
        rhs[row] /= matrix[row * n + row];
        if (!circuit_finite(rhs[row]))
            return false;
    }
    source = nodes - 1;
    double power = 0, scale = 0;
    for (int k = 0; k < count; ++k) {
        const auto& b = branches[k];
        voltages[k] = (b.from ? rhs[b.from - 1] : 0) - (b.to ? rhs[b.to - 1] : 0);
        currents[k] = b.kind == CircuitKind::Battery         ? (voltages[k] - b.value) / resistance[k]
                      : b.kind == CircuitKind::CurrentSource ? b.value
                                                             : rhs[source++];
        if (!circuit_finite(currents[k]) || !circuit_finite(voltages[k]))
            return false;
        power += voltages[k] * currents[k];
        scale += ::fabs(voltages[k] * currents[k]);
    }
    if (::fabs(power) > 1e-8 * ::fmax(1., scale))
        return false;
    for (int node = 0; node < nodes; ++node) {
        double residual = 0, magnitude = 0;
        for (int k = 0; k < count; ++k) {
            if (branches[k].from == node) {
                residual += currents[k];
                magnitude += ::fabs(currents[k]);
            }
            if (branches[k].to == node) {
                residual -= currents[k];
                magnitude += ::fabs(currents[k]);
            }
        }
        if (::fabs(residual) > 1e-9 * ::fmax(1., magnitude))
            return false;
    }
    return true;
}
} // namespace battery
#undef CIRCUIT_HD
