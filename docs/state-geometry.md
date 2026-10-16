# State-dependent contact geometry (CUDA)

**Legacy v5 symmetric model.** Retained for reproducibility, not the current
directed operator. Its endpoint averaging does not implement the requested
source/receiver distinction. See [directed geometry v6](directed-geometry.md).

`state_geometry` is an opt-in native law. The UPV v5 calibration enables it on
every solid–solid interface **and on the measured C2 source contacts**. Existing
synthetic scenarios retain their old laws and regression results. The Python
reference integrator is not the implementation of this model.

## State → directional geometry → L → heat flux

Each material object supplies `thermal.surface_connectivity`. For its principal
direction `k` and the current local volume state:

```
theta_i = (T_i - T_ref_i) / T_scale_i
log(S_i,k) = a_i,k theta_i + b_i,k theta_i² + d_i,k D_i
z_ij = z_ref,ij + 0.5 [log(S_i,k) + log(S_j,k)]
L_ij = l0_ij + activation_width exp(z_ij)
f_ij = exp(z_ij) / [1 + exp(z_ij)]
G_contact,patch = overlap_weight [G_floor + G_span f_ij]
G_patch = 1 / [1/G_contact,patch + R_bulk,i + R_bulk,j]
Q_ij = dt G_patch (T_j - T_i)
```

The metric is positive, diagonal in the material's local frame. The aligned-face
mesher selects `nᵀ S_i n` by its local principal axis. A common rigid rotation of
the assembly leaves the result unchanged. Arbitrarily rotated, nonaligned
contact faces remain unsupported by this mesher. No dense tensor of pairwise
links is stored. Existing overlap/port geometry defines possible contacts;
state determines each contact's instantaneous availability. This is not a
mechanical deformation solver and does not create new geometric neighbors.

The quadratic is evaluated **at each endpoint before averaging**, so a hot/cold
pair need not behave like two volumes at their mean temperature. Coefficients
can be signed, direction- and material-dependent. The response can rise, fall,
or be nonmonotonic. Log arguments are clamped to ±40 for numerical safety.
`L` is an effective connectivity coordinate, not a computed physical gap or
measured expansion. Its absolute scale and `l0` cannot be identified from the
available thermocouple data; their length conventions are explicitly fixed.

Only one reciprocal, nonnegative conductance is evaluated per patch. The CUDA
edge kernel calculates one flux from the old temperature field; volumes gather
opposite signs. Bulk half-cell resistances, measured conductivity tensors and
heat capacities are not replaced or rescaled by this contact law. The time-step
bound includes the maximum of both intact and destroyed conductance branches.
An outer step continues to consist of multiple stable CUDA substeps.

## Damage and destruction

The local node damage `D_i` is accumulated by the existing volume model.
The interface also retains its own damage/history. Its damage is at least
`endpoint_damage_weight * max(D_i, D_j)`, not the average: destruction of one
endpoint must not be hidden by a healthy neighbor. Optional local thermal damage
adds to this history. Reaching `destruction_damage` irreversibly selects
`destroyed_l0_m`, `destroyed_l_m`, and destroyed floor/span conductance.
There is no relaxation ramp at this switch. A destroyed contact can be much
less **or** much more conductive. Cooling does not restore the intact state.

For the destroyed branch, `destroyed_l_m - destroyed_l0_m` defines the reference
opening before applying the local metric. A zero opening closes the active
part, leaving only the destroyed background conductance. The state-geometry
law rejects legacy relaxation, length-damage-shift and damage-multiplier knobs
that would otherwise be silently ignored. Existing first/second-order laws are
still available separately.

The pre-event UPV experiments cannot calibrate damage coefficients, damage
kinetics or the destruction jump. Their settings here are explicit placeholders,
not experimentally established properties. Synthetic GPU tests verify the
mechanism, not the predictive accuracy of a cascade or rupture model.

## UPV fitting protocol

`--parameter-set state_geometry` fits five material-pair **reference** contact
coefficients at 25 °C, one effective environmental heat-transfer coefficient,
and two shared log-metric temperature coefficients `a,b` with `T_scale=100 K`.
The engine supports distinct profiles, but this first fit deliberately uses a
shared closure; it does not identify five different material response curves.

Fixed values:

- Battery: k = (0.9, 28.3, 28.3) W/m/K; cp = 1007 J/kg/K; mass 0.410 kg.
- ISS: k = 0.145 W/m/K; cp = 770 J/kg/K; thickness 2 mm; density 200 kg/m³.
- TIM: k = 2 W/m/K; cp = 1175 J/kg/K; thickness 2 mm; density 3300 kg/m³.
- Aluminium/steel properties, dimensions, source geometry and cooling-tip prior.

The pad cp values are **working composition-dependent assumptions**, not UPV
measurements: the actual low-density ISS pad need not have pure diopside cp;
the silicone TIM value is the midpoint of the proposed 1050–1300 range. No cp
or bulk k is optimized. Both native and Python contracts reject attempts to
fit battery properties, and v5 also rejects pad-cp fitting.

Bounds: contact h(T,D=0) 1–10000 W/m²/K; reference h at 25 °C 2–9900;
environment h 0.1–30; a,b in [-12,12]. These are exploratory engineering bounds,
not measured material uncertainty. Boundary hits and ill-conditioned parameter
sensitivities mean a low residual is **not** unique material identification.

Only no-cooling data are fitted (60–884 s). Parameters and source/binary hashes
are frozen before active-cooling evaluation (60–1412 s). The latter was already
inspected in previous debugging, so it is not a pristine blind holdout. The
unconfirmed C1/C3 acquisition-numbering hypothesis remains unchanged. Both cases
are conditional on measured C2 surface forcing; neither validates an autonomous
runaway source.

Surface thermocouple predictions use instantaneous, zero-time-advance evaluation
of the **same host/CUDA law** on current local temperatures and stored edge
history. The face reconstruction includes the resulting local contact flux;
it does not reuse a stale constant conductance or accumulate damage on readout.
Reports save actual native material properties and time series of interface
conductance and L. Calibration snapshots remain parameterized baselines; use
`export_upv_state_geometry.py` to produce directly runnable native scenes.

Tests: local/directional response, endpoint reversal, nonlinear endpoint mixing,
irreversible opening/closing on failure, CPU/CUDA formula agreement, shared
rotation, conservative heat exchange, maximum principle, driven-boundary energy,
step convergence, source-contact parity and fixed-property guards.

## Recorded v5 result

Artifacts: `data/calibration/upv-state-geometry-*`; runnable scenarios:
`examples/upv-state-geometry-calibrated.json` and
`examples/upv-state-geometry-validation.json`.

| Case | RMSE, K | Maximum absolute error, K |
| --- | ---: | ---: |
| No-cooling, fitted nonlinear contacts | 2.183601 | 8.205063 |
| Active-cooling, frozen parameters | 2.230057 | 9.684975 |
| No-cooling, refitted static contacts with identical materials | 2.192746 | 8.215634 |

The two nonlinear starts reached RMSE 2.1836005 and 2.1836014 K (1168 total
forward evaluations including finite-difference Jacobians). Both stopped on the
objective-change tolerance, not a proof of a global optimum. The fitted shared
coefficients are a = -3.13736, b = 11.99725; contact reference h is about 9900
W/m²/K for four classes and 1491.37 for ISS–aluminium. Environmental h is 30.
Six of eight parameters are near a search boundary; local Jacobian condition
number is about 524.5. These are bounded effective fits, **not identified
material constants**. The tiny 0.00915 K training improvement over the fair
static comparison does not establish that these experiments identify the
temperature-response curve. Changes from the older 2.261 K result also include
the corrected, now-fixed pad heat-capacity assumptions.

The law is genuinely dynamic: for example, total Al2–ISS3 conductance in the
training run varies from 2.1144 to 2.3210 W/K, while colder Al3–ISS4 follows a
different history despite sharing the same material-pair law. Most other
contacts are near their high-conductance limit, so their total transfer is
primarily limited by unchanged bulk thermal resistances. The remaining error,
especially the C1 TIM-face overshoot, has not been explained away by the fit.

| Numerical check, no parameter refit | No-cooling maximum change, K | Active-cooling maximum change, K |
| --- | ---: | ---: |
| Halve maximum timestep (0.5 → 0.25 s) | 0.00360 | 0.00294 |
| Double passive divisions (112 → 784 total volumes; batteries unchanged) | 0.20014 | 0.50001 |

The refined-grid RMSE is 2.18235 / 2.25634 K respectively. Grid sensitivity is
larger than the small gain from introducing the nonlinear law; do not interpret
that gain as statistically or numerically established model superiority. One
refinement is a sensitivity check, not proof of mesh independence. Absolute
energy-balance residual stayed below 5.2e-9 J in these runs. 38 Python/contract
tests (including real GPU probes) and all 10 native CTest entries passed; the
new destruction test also checks a scheduled in-run failure, not just an
initially destroyed contact. The exported first scenario ran directly in the
native CUDA solver and conserved energy.
