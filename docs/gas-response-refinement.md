# Gas response on the latest effective geometry (2026-10-22)

Explicit reference: `gas-access-refinement-20261022-r4-fresh` under
`data/calibration/upv-separated-chemistry-v6`. No reconstruction from an older
parameter file. Native fingerprint must match its completed full fresh run.

The user confirmed the existing C1/C3 numbering convention; C2 ISS/TIM mapping
is not changed. The objective is acceptable entire cooling curves, not only a
single pulse. Source inventories, R1/R2/R3, UI/Joule heating, measured Cp/k,
all solid interfaces, geometry, shell physics, wall cooling and numerical
tolerances stay frozen during this stage. Regular floor remains 0.02 s.

Two relative calibration axes use the existing directed state-dependent laws:

- `mixing_scale` (0.03..3): scale free-gas response at the existing hot knots,
  retaining its exact room-temperature reference. A new logit/span represents
  the prescribed knot gains without saturation masquerading as a scale change.
  No change to molecular/internal edges, channel area, Cd, pressure-flow law,
  gas-to-solid endpoint laws, or chamber wall. Between knots the original
  log-response interpolation remains; do not claim a uniform scale there.
- `damage_delta` (-6..6): one common increment in the reverse **casing donor**
  damage metric. It changes hot casing -> gas response as shell damage evolves,
  not forward gas heating or solid/solid contact properties. Added to the
  explicit reference, not repeatedly compounded across trials. The reference
  increment already equals 4; a delta of -4 removes it, not the damage model.

The 835 s sealed state is reused through the existing native gas-response
contract, never by bypassing the binary/source identity. Such continuations
remain conditional until a newly selected law is rerun from ambient.

Each suffix extends to at least 1100 s, with the same <=950 s observation
schedule as the reference. A direct audit confirms exact equality of that
schedule before the first trial. End=950 s is not used as a cooling fit.
Scoring reuses `upv_cooling_fit`: all six temperature traces, time-balanced
cooling error, pressure and a separate pressure/C3 ISS sharp-front gate.
Temperature peak amplitude is soft but not discarded. Complete raw phase
errors and the previous curve+front shape metric are retained separately.
No fitted time shift, per-TC gain, extrapolated endpoint or gap filling.

An adaptive bounded queue starts with an extended physical control, free-gas
ratios 0.3/0.1, and common reverse-damage contrasts. It follows improving axes
and brackets locally. A local score is not a proof of full curve agreement;
saved raw shape diagnostics must be reviewed before final acceptance.
Average speed floor 0.5x, short-window 0.25x/20 s and strict mass/energy
ledgers stay enabled. Unacceptable runtime never causes a forced larger dt.
Physical/temperature divergence guards save partial evidence before rejection.

Artifacts: `gas-response-refinement-20261022-r1`. Cooling tolerance convergence
is explicitly pending a fresh long verification and is not experimental
validation. Held-out active cooling is not used to select any parameter.

## Pulse-preserving selection correction

The extended neutral control repeats all 12 old <=950 s temperatures bitwise.
Initial mixing reduction improved the cooling scalar but eventually erased
C1/ISS's observed pulse. Under `cooling-with-witness-pulse-v2`, every old
completed trial is reassessed identically: C1/ISS maximum 1.5 s rise must be at
least half the measured jump. This is an event-presence gate, not exact TC peak
fitting. Pressure peak must be within 30% for exploration and 5% for final
cooling acceptance, in addition to its existing sharpness/timing/late loss.
The gate is checked early once the burst window is complete; an invalid trial
is not integrated to the end just because its late scalar might be low.

`--resume` preserves the prior full assessment in a numbered audit JSON and
leaves original per-trial assessments untouched. Completed runs remain
completed even when no longer fit-eligible. Interrupted evidence is recorded
as such, never scored as a complete trajectory. Source/core/end contracts
must match; a still-running controller cannot be resumed over. New trials
use unused folder indices. The restored best is recalculated from all eligible
completed candidates, not inherited blindly from the old criterion.
