# Continuous gas reaction and pressure transport

The CUDA mixture adapter integrates gas combustion inside the pressure-transfer
interval, rather than reacting a snapshot once before or after all advection.
Solid heat exchange, material conversion, casing failure and calibrated thermal
contact laws retain their existing operators.

`gas_combustion.hpp` owns the local effective law. In each volume, the available
heat stock is `min(fuel_j, oxygen_kg * oxidizer_heat_j_per_kg)`. Its independent
normalized Arrhenius law is

`k3(T) = reaction_rate_per_s * exp[E3/Rgas * (1/reaction_reference_k - 1/T)]`.

`T` is the LTE mixture temperature recovered from **sensible** energy and current
gas/solid heat capacities. Chemical reserve is not sensible energy: cold fuel
must not ignite merely because it carries a large potential heat stock.
At the local temperature it converts the fraction `1 - exp(-k3(T) * dt)` into sensible
energy. Total gas mass stays constant: consumed oxygen becomes products within
the same effective gas inventory. This is not a detailed flame chemistry model.
`ignition_temperature_k` is an optional extra gate (zero in the UPV model), not a
replacement for Arrhenius. JSON with an enabled R3 must explicitly provide both
reference temperature and activation energy. Ea=0 remains a mathematical
constant-rate limit for analytic numerical tests, not an implicit old-scenario mode.

The UPV starting prior is **20 s^-1 at 600 K, E3=80 kJ/mol**, not fitted gas
chemistry. It gives about 2.2e-6 s^-1 at 300 K, 0.096 s^-1 at 450 K and 20 s^-1
at 600 K. All volumes have the same law; internal pores start with zero oxygen.
Oxygen arriving through an opened barrier can enable internal oxidation.
Old constant-rate warmup parameters, including phi, are not a calibration of
this new model. New fitting output is isolated in `upv-r123-arrhenius-v3`.

Each gas interval uses symmetric integration: half of the local reaction,
conservative pressure-driven transport, then the remaining half reaction. New
arrivals therefore burn within the current interval. Fuel, oxygen and enthalpy
flow through the same accepted mass packets. Subsequent intervals re-evaluate
pressure and flow using the newly heated mixture. Reaction heat history remains
at the volume where it was released; it is not transported as another stock.

When a volume can mix fuel and oxygen at ignition temperature, including
upwind arrivals through an open channel, the existing numerical
`max_fraction_per_step` also bounds the fraction reacting per coupled interval:
`h <= -log(1 - max_fraction_per_step) / max(k3(T))`.
The interval is recomputed on every current gas snapshot, including possible
hot upwind arrivals; it is not bounded by the rate at the reference temperature.
For nonlinear R3, a thermal-rise bound also resolves its self-heating. Each local
reaction half-step further re-evaluates k(T) while consuming finite stocks.
The analytic network mode takes these adaptive intervals inside
its resident cooperative CUDA kernel. Cold or isolated oxygen-free fuel stocks
do not force extra kinetic intervals. This does not add CPU synchronization or
require resolving the fastest orifice relaxation time. Roundoff-sized fuel
stocks skip extra transport subcycling, but still undergo the exponential local
reaction. Nonreactive networks retain the original one-pass network update.

This is a time-discretized continuous reactor/transport model. The reaction
update is exact with frozen rate, while the coupled flow, mixing and ignition
must converge with time resolution. It assumes a well-mixed volume in local
thermal equilibrium, not a spatially resolved flame within an edge. Changing
operator integration alone cannot guarantee more heating near a witness: that
also depends on physical residence time, oxygen availability and reaction rate.

`cuda_gas_combustion` checks closed-reactor kinetics, oxygen exhaustion, ignition,
same-interval burning of incoming fuel, downstream burning, local packet ledgers,
global mass/thermal-plus-chemical conservation, time refinement and CUDA tile /
edge-order invariance. No physical parameter is fitted by these checks.
