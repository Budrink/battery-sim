# Temperature-dependent gas exchange

Pressure transport and counterexchange are distinct contributions. The existing
compressible orifice law and its pressure/inventory budgets are unchanged.
Temperature does not create a synthetic one-way pressure flux.

The counterexchange rate (kg/s) is now

`g = (rho_harmonic * D * Aopen / L) * F_S(Ti,Tj,axis) + alpha * abs(mdot_pressure)`.

`F_S` is the shared `evaluate_directed_geometry` response normalized to a unit
reference conductance. It selects the hotter source's current tensor, without
averaging forward/reverse responses. The material response comes from
`gas_network.surface_connectivity`; the relative geometry law is
`gas_network.mixing_connectivity`. `mixing_axis` on each channel selects its
principal direction. Density, open area and actual path length remain physical
geometric inputs, not a stored connectivity matrix.

The pressure-driven entrainment coefficient is NOT multiplied by S a second
time. At equal pressure, the thermal contribution remains active. At equal
temperature, composition gradients can still exchange fuel and oxygen.

For one frozen pair, `dm = m_reduced * (1 - exp(-g*dt/m_reduced))` is the exact
exchange integral. Each endpoint exchanges the SAME mass. One packet contains
`dm*cv*(Ti-Tj)` of sensible energy plus concentration-difference payloads for
fuel, oxygen, vapor and latent energy. No independent duplicate heat flux is
added. Global mass stays fixed; local oxygen need not burn before travelling.
The graph's shared outgoing-inventory budget preserves positivity at junctions.
Every stage uses one common snapshot, fixed CSR gathers and no floating atomics.

A closed barrier supplies zero open area and blocks BOTH contributions.
Material damage changes the aperture through the existing shell mechanics;
gas S does not open another hole or evolve a second barrier-damage history.

The default neutral unit law reproduces the previous exchange rate. The UPV
trial constructor now supplies the SAME already-declared monotone gas profile
to gas/gas exchange and gas/surface contacts, with separate physical reference
scales. This is an explicit uncalibrated constitutive hypothesis, not a claim
that gas/gas and gas/solid conductances are identical. No new gain or fuel stock
is fitted by this correction. Buoyancy, jet momentum and CFD remain unresolved.

`check_upv_thermal_mixing.py` replays the last open-edge monotone scenario from
ambient twice using the same new binary: neutral counterexchange control and
temperature-dependent S. It verifies the neutral trajectory against the old
archive and writes actual accepted gas/surface heat budgets and temperatures.
The regular timestep floor remains 0.005 s; speed guards remain 0.5x average
and 0.25x over a 20 s window. No extra host synchronization or adaptive internal
transport loop was added.
