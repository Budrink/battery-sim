# Shared thermal connectivity

All finite heat contacts in the UPV model use `evaluate_directed_geometry`:
bulk battery and passive-material links, material interfaces, external thermal
boundaries, gas/casing, gas/pad, gas/chamber wall and exposed aperture surfaces.
The full assembled UPV mesh is checked by `battery_sim_gas_film_audit` for any
link that bypasses this law. Explicit old transition scenarios are not the UPV
model; no new scalar gas-flux implementation is retained for those inputs.

The hotter endpoint supplies the ordered source response. Its metric depends
on temperature, damage, principal direction and contrasts to the receiver.
Forward/reverse conductances are **not averaged**. Gas has no defect state;
its temperature response is `gas_surface_connectivity` on a contact (or the
network-wide `surface_connectivity`). The receiver uses its material profile.
Unidentified gas profiles default to a neutral metric, not an invented fit.

`h*A` is only a **reference scale**, normalized to the same directed law as
solid contacts; it is never the runtime gas heat-flux law. New UPV scenarios
write explicit `connectivity` on every gas contact. Archived reference-h inputs
are normalized during meshing. Material heat capacity is not modified.
Implicit bulk links similarly use measured `k*A/dx` as their reference scale,
then evaluate the node metric; boundaries no longer silently bypass it either.

Optional exterior radiation is another nonlinear edge in this same heat pass,
with a grey-body constitutive law and the existing material half-cell resistance.
It does not replace or rescale gas convection; see [thermal radiation](thermal-radiation.md).

## CUDA coupling

`gas_thermal_contact.hpp` selects the shared directed law. The small
`gas_surface_exchange.hpp` stage evaluates every current contact before local
reaction/cooling integration. After mass transport, it evaluates again with
the arriving mixture, allowing a different donor and conductance.

The local frozen-coefficient reactor debits gas energy. The SAME conductances
allocate that accepted heat to patch records, with roundoff closed on the last
nonzero patch. A fixed-order gather credits each solid once. No gas endpoint or
solid endpoint receives concurrent floating-point writes; no extra host round
trip or hidden microstep is introduced. Rejected trials reset scratch records.
Parallel reduction over several surfaces is not averaging `Sij` with `Sji`.

Mass-carried enthalpy remains in the separate conservative mass-transfer
operator. Ideal gas/condensed local equilibrium inside a shared volume remains
an algebraic constraint, not a separate finite film-transfer law.

## Verification

`cuda_gas_local_reactor` checks actual CUDA sensitivity to donor temperature,
receiver/source reversal and damage, plus anisotropy, energy closure, retry,
tile independence, refinement and the 0.005 s physical-step floor.
`battery_sim_gas_film_audit <full-pad-scenario>` checks all compiled UPV thermal
links and nine finite-capacity analytical cases. Accepted-step traces report
both reference and instantaneous conductance, not a scalar reference posing as
the actual transfer coefficient.

Replay diagnostics start from the archived 750 s closed-shell checkpoint.
They isolate the effect after that state; they are NOT a new full warm-up,
calibration or independent validation. Chemistry, Cp, k and fitted parameter
values are held fixed. A chosen corrected model still needs a fresh full run.
