# Gas mixing and finite volatile inventory — 2026-10-19

Status: implemented diagnostic physics, **not calibrated and not validated**.
The adaptive destruction search was stopped on request; its results and the
previously accepted profile were not replaced. Native solver remains CUDA.

## Evidence motivating the change

In `adaptive-destruction-20261018/trial-0005`, at 925 s:

- Net generated gas: 28.4501 g (all gas inventory minus its initial value).
- R3 heat: 119.122 kJ, including 113.051 kJ in the chamber, 6.065 kJ in
  peripheral cavities, and only 4.660 J in all interlayer gaps combined.
- Oxygen: 24.261 g remains in the chamber; almost none remains in the gaps.
- C2 R2 thermal power drops from 9616.84 W at 887.110 s to 33.03 W at
  890.486 s. Remaining precursor at that later sample: 26.617 g.

These facts do not identify gas yield uniquely. They require separating
source exhaustion/quenching, local reagent access, and the route of heat.
`casing_probe` now optionally records each zone's T, precursor inventory,
R2 rate/power, cumulative reacted mass and gas production. Enable
`metadata.zone_diagnostics`; it does not drive the model.

## Open-channel counterexchange

This is an additional constitutive term on the **existing gas graph**, not a
second mesh or an artificial heat link through an intact solid.

For an open channel of area A and path length L, let

```
K = D_eff A rho_harmonic / L + entrainment_fraction |mass_flow|
m_reduced = m_a m_b / (m_a + m_b)
q = m_reduced [1 - exp(-K dt / m_reduced)]
```

Exchange q in each direction. Net mass is zero. Fuel, oxygen, condensable
vapor and phase-reference energy follow their mass fractions. Sensible heat
is `q cv (T_a-T_b)`: the counterexchange does not double-count the net orifice
flow's pressure work. Equal pressures do not suppress composition exchange.

All edge proposals use one snapshot. CSR node gathers cap aggregate each-way
exchange at half the resident mass; the same scaling and signed payload apply
to both endpoints. There are no floating-point scatter atomics. The physical
coefficients are in SI units, not percentages per algorithmic step. Pairwise
frozen-state composition relaxation is exactly time-equivalent; whole-graph
advection/reaction/thermal splitting still requires refinement checks.

Closed casing patches have zero effective area and cannot mix. The flow-driven
term vanishes with flow; molecular/effective dispersion may persist. There is
no persistent turbulence reservoir or CFD momentum field. A high-degree node's
budget and unresolved geometry can affect effective network mixing; coefficients
are not yet identified from measurements. Reaction arrival bounds include
counterexchange, so newly arriving oxygen cannot bypass the thermal accuracy
budget.

## Finite liquid/vapor phase

`conversion.volatile_phase` declares a separate initial liquid inventory,
density, liquid cp, latent energy, reference boiling T/pressure, relaxation rate,
and optional fuel energy per kg. The liquid is allocated inside the reference
inert capacity: **measured initial battery heat capacity does not increase**.
Minimum solid capacity remains positive. Liquid occupies volume; evaporation
increases the free gas volume and condensation decreases it.

Use a constant-latent, ideal-pseudo-fluid Clausius–Clapeyron saturation curve.
Vapor pressure/latent-heat relations are the standard equilibrium approximation,
not a claim about unidentified electrolyte composition; see
[NIST's discussion of vaporization enthalpy and Clapeyron](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=920759).
The solver retains one common R and cv for the gas; the volatile pseudo-species
must use that same R. Multicomponent EOS and detailed electrolyte chemistry
are explicitly not implemented.

A bounded implicit local relaxation solves for the mass transferred towards
saturation **together with its temperature**, conserving

```
solid sensible energy + gas sensible energy + mobile phase-reference energy
```

Latent energy is never discarded as an external loss. Condensation returns it.
Both liquid and mobile gas heat capacities change with transferred mass.
Condensate can return only to available original local liquid capacity, not to
an unlimited invented pool. The implicit phase solve avoids making the whole
graph take microscopic steps just to resolve a local stiff phase relaxation.

If the vapor is combustible, its fuel reserve belongs initially to the liquid
and moves to/from gas upon phase change. R3 needs oxygen and consumes fuels in
proportion to chemical reserves; oxidized vapor is no longer condensable but
its mass remains gas products. Mobile phase-reference energy remains accounted
for after combustion. This is a lumped closure, not identified reaction species.

Only CUDA network + split LTE supports these extensions. Other backends reject
an enabled phase/mixing model rather than silently dropping its inventories.

## Execution order and files

1. Leading casing damage/fracture, including irreversible fracture work.
2. Existing thermal exchange, Joule heating and continuous R1/R2 chemistry.
3. Local LTE and implicit liquid/vapor exchange.
4. CUDA gas stage: half R3, conservative pressure advection, open-channel
   counterexchange, half R3. Repeated intervals re-evaluate state and arrivals.
5. LTE, diagnostics, candidate validation and single transactional commit.

Laws: `volatile_phase.hpp`, `gas_mixing.hpp`; CUDA mixing gather:
`gas_mixing.cuh`; integration remains in `mass_step.cuh`/`batched_gas.cuh`.

`scripts/check_upv_phase_mixing.py` runs a small frozen-parameter ablation:
control / mixing / phase / combined. Its explicit liquid and mixing priors are
**not datasheet measurements**. Existing solid properties, contact laws,
electrical parameters and R1/R2 rates are unchanged. A new fit must not count
old effective evaporation yield and the new volatile source twice: R2 gas
yield now denotes decomposition products, not the explicit liquid reservoir.
No calibration is implied by a diagnostic pressure/temperature improvement.

The initial diagnostic uses **zero additional volatile fuel energy**. This
isolates gas mass, phase cooling and mixing without improving a fit simply by
adding unmeasured chemical energy. The implemented combustible-vapor option
requires an independently identified or explicitly repartitioned whole-cell
chemical budget before it is used in calibration. An initial control-only
attempt `phase-mixing-20261019-r1` was stopped before its combined case to make
this choice explicit; its partial results were retained.

Tests cover finite inventory, mass/sensible/latent/chemical ledgers, reversible
phase heat, cold/hot behavior, CUDA agreement, time refinement, equal-pressure
composition exchange, closed barriers, analytical pair relaxation, immediate
reaction of arrivals, and CUDA block/retry invariance.
