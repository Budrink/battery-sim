# Справочник параметров редактора

Единый каталог с подсказками UI. Сначала прочтите [руководство по разделам](experiment-editor.md).
Технический ключ указан для поиска в контракте; единицы непосредственно у поля учитывают перевод в °C, мм, г, кПа и другие удобные единицы.
Не все поля присутствуют в каждом импортированном законе: отсутствующее поле не создаётся простым просмотром.

### Total heat capacity [J/K]

Ключ: `heat_capacity_j_per_k`.

Energy required to raise this entire object by one kelvin. Not specific heat per kg. For battery contents, reactive and liquid contributions are INCLUDED; the metal shell is additional. Must exceed the sum of component capacities.

### Reactive mass [kg]

Ключ: `mass_kg`.

Mass available to the battery reaction chain. Together with specific heat it contributes to heat storage; together with reaction energy it limits the chemical reserve. UI may display grams.

### Original material specific heat [J/(kg K)]

Ключ: `reactant_cp`.

Sensible heat stored per kilogram per kelvin in unreacted material. This is not the energy released by R2.

### R2 precursor specific heat [J/(kg K)]

Ключ: `activated_cp`.

Heat storage of material converted into precursor, before R2 consumes it.

### Residue specific heat [J/(kg K)]

Ключ: `residue_cp`.

Heat storage of the solid reaction product; residue is not removed from the thermal balance.

### Initial inventory mass [kg]

Ключ: `initial_mass_kg`.

Mass initially present in this liquid or gas inventory. The displayed section identifies the substance; liquid and initial cavity gas are separate inventories. UI may display g or mg.

### Liquid specific heat [J/(kg K)]

Ключ: `liquid_cp`.

Sensible heat storage of unvaporized liquid, distinct from latent heat of evaporation.

### Thermal conductivity along local X/Y/Z [W/(m K)]

Ключ: `conductivity_w_per_m_k`.

Conductivity of the material, not a whole-contact conductance. Geometry, contact area and state-dependent connectivity determine the final heat transfer.

### Residue conductivity ratios

Ключ: `residue_conductivity`.

Directional conductivity multipliers relative to original material. Values below one conduct less; they do not change heat capacity or chemical energy.

### Background conduction multiplier

Ключ: `background_factor`.

Scales the background part of internal thermal connectivity.

### State-dependent conduction multiplier

Ключ: `active_factor`.

Scales the additional internal conduction controlled by the connectivity state.

### Destroyed background multiplier

Ключ: `destroyed_background_factor`.

Background internal conduction after destruction; no restoration of intact material is implied.

### Destroyed active multiplier

Ключ: `destroyed_active_factor`.

State-dependent internal conduction amplitude after destruction.

### R1 reference rate [1/s]

Ключ: `degradation_rate_per_s`.

Heat-neutral degradation rate at the R1 reference temperature. Damage produces some R2 precursor and changes electrical/material properties.

### R1 reference temperature

Ключ: `degradation_reference_k`.

Temperature at which the specified R1 rate is defined. UI uses Celsius; contract uses kelvin.

### R1 activation energy

Ключ: `degradation_activation_j_per_mol`.

Controls temperature sensitivity of degradation through Arrhenius kinetics. Raising it does not mean a universally higher rate: compare temperatures to the reference.

### R1 precursor yield [0..1]

Ключ: `precursor_yield`.

Fraction of degraded reactive material that becomes R2 precursor rather than inert degradation product.

### Initial degraded precursor fraction [0..1]

Ключ: `initial_precursor_fraction`.

R2 precursor already present before the run; distinct from material charged electrically.

### Initial charged material fraction [0..1]

Ключ: `initial_charged_fraction`.

Initial chemical reserve associated with charge. Fractions share a finite material inventory; they are not extra mass.

### R2 / R3 reference reaction rate [1/s]

Ключ: `reaction_rate_per_s`.

Rate at the reference temperature of the reaction in this section. R2 consumes battery precursor; R3 consumes gas fuel with oxygen. They have independent rates.

### Reaction reference temperature

Ключ: `reaction_reference_k`.

Temperature defining the reference rate, not a trigger temperature. UI uses Celsius; the solver uses kelvin.

### Reaction activation energy

Ключ: `reaction_activation_j_per_mol`.

Arrhenius temperature sensitivity. Larger activation energy steepens acceleration ABOVE the reference and suppresses the rate BELOW it; it does not directly increase available energy.

### R2 specific chemical energy

Ключ: `net_heat_j_per_kg`.

Chemical energy released per reacted mass, limited by precursor inventory. Independent of sensible heat capacity. UI displays kJ/kg.

### R2 gas mass yield [kg/kg]

Ключ: `gas_yield`.

Base fraction of reacted mass converted into gas. Changes gas inventory, pressure and mass-carried energy, not just visual smoke.

### Maximum R2 gas mass yield [kg/kg]

Ключ: `gas_yield_max`.

Upper yield used by the state-dependent gas production law. Must remain consistent with the base yield and mass conservation.

### Chemical energy retained in gas [0..1]

Ключ: `gas_chemical_fraction`.

Share of released chemical energy exported as combustible gas reserve rather than immediately deposited as heat inside the battery.

### Material density [kg/m3]

Ключ: `density_kg_m3`.

Mass per material volume. For plates, density times volume sets mass; density times specific heat times volume sets heat capacity.

### Latent heat of evaporation

Ключ: `latent_j_per_kg`.

Energy absorbed per kilogram of liquid vaporized and returned on condensation. Distinct from R2/R3 chemical energy.

### Reference boiling temperature

Ключ: `boiling_temperature_k`.

Boiling temperature at the reference saturation pressure; local phase equilibrium also depends on pressure.

### Reference saturation pressure

Ключ: `saturation_reference_pa`.

Pressure paired with the reference boiling temperature in the liquid/vapour equilibrium law. UI displays kPa.

### Phase relaxation rate [1/s]

Ключ: `relaxation_per_s`.

How rapidly liquid/vapour exchange approaches the model's local equilibrium. Energy and remaining liquid inventory constrain the exchange.

### Electrical resistance

Ключ: `resistance_ohm`.

Reference resistance of this battery or circuit branch. Joule heating is I squared R and remains subject to the electrical energy balance. Battery UI uses milliohm; branch UI uses ohm.

### Nominal electrical capacity

Ключ: `nominal_capacity_coulomb`.

Charge capacity, displayed in Ah (3600 coulomb per Ah). Not heat capacity. Used by charging/degradation laws.

### Charge acceptance [0..1]

Ключ: `charge_acceptance_fraction`.

Fraction governing conversion of supplied charge into chemical reserve, subject to available capacity and energy.

### Wall or shell thickness

Ключ: `thickness_m`.

Physical plate thickness. With area, density and specific heat it sets thermal mass. UI uses mm.

### Material specific heat [J/(kg K)]

Ключ: `specific_heat_j_kg_k`.

Heat storage per kilogram per kelvin. Multiply by physical mass to obtain the whole object's J/K.

### Contents-to-shell heat transfer [W/(m2 K)]

Ключ: `contents_contact_h`.

Simple contact coefficient per area, used when no explicit state-dependent contact law overrides it.

### Between-plate heat transfer [W/(m2 K)]

Ключ: `seam_contact_h`.

Thermal contact between neighbouring shell plates, not valve mass permeability.

### Thermal damage onset temperature

Ключ: `damage_onset_k`.

Temperature scale where thermal damage starts accumulating. Does not by itself open a hole; pressure and fracture mechanics still apply.

### Thermal damage rate [1/s]

Ключ: `damage_rate_per_s`.

Rate of irreversible plate damage at its onset. Zero disables this thermal damage term; it does not disable an independently configured pressure vent.

### Damage temperature scale [K]

Ключ: `damage_scale_k`.

Temperature rise for an e-fold increase of the damage rate. Smaller values give steeper temperature sensitivity.

### Destroyed conduction multiplier

Ключ: `destroyed_conductance_factor`.

Remaining conduction relative to intact material after destruction. Does not heal when the temperature drops.

### Contact conductance [W/K]

Ключ: `conductance_w_per_k`.

Whole-contact heat flow per kelvin difference. Unlike conductivity, geometry/area are already included.

### Background contact conductance

Ключ: `background_conductance_w_per_k`.

Baseline term in the connectivity law. Units are W/K for a whole contact, W/(m2 K) in a per-area shell law; follow the field label.

### State-dependent contact amplitude

Ключ: `active_conductance_w_per_k`.

Additional conductance modulated by current connectivity. Units depend on whether this is a whole-contact or per-area law.

### Destroyed background contact conductance

Ключ: `destroyed_background_conductance_w_per_k`.

Baseline contact term after destruction. Whole-contact or per-area units are shown in the field label.

### Destroyed active contact amplitude

Ключ: `destroyed_active_conductance_w_per_k`.

State-dependent contact term after destruction; separate from the intact term.

### Reference connectivity bias

Ключ: `geometry_reference_logit`.

Dimensionless log-odds bias of the connectivity activation. It is not conductivity or contact area.

### Damage influence on connectivity

Ключ: `endpoint_damage_weight`.

Weights endpoint damage in the existing connectivity law; does not add a separate heat source.

### R3 energy per consumed oxygen mass

Ключ: `oxidizer_heat_j_per_kg`.

Heat budget per kilogram of oxygen consumed, displayed in MJ/kg O2. NOT a fuel heating value. R3 is limited by both oxygen and available fuel energy.

### Vapour fuel chemical energy

Ключ: `vapor_fuel_heat_j_kg`.

Chemical reserve carried per kilogram of evaporated liquid, displayed in MJ/kg vapour. Changes the liquid chemical reserve globally, not its heat capacity or latent heat.

### Optional R3 cutoff temperature

Ключ: `ignition_temperature_k`.

Explicit lower gas-temperature cutoff for R3. With cutoff disabled, the Arrhenius rate still controls burning. Oxygen is always required.

### Initial temperature

Ключ: `initial_temperature_k`.

Temperature at time zero; UI uses Celsius and the contract kelvin. In a heater controller this is the initial TARGET, while apparatus temperature and sensor temperature are separate fields.

### Controller sensor

Ключ: `sensor_id`.

ID of the battery/shell element monitored by this heater. It controls the target feedback and cutoff, not where heat is applied.

### Target heating rate

Ключ: `ramp_k_per_s`.

Rate of the controller's target ramp, shown in C/min. It is not guaranteed actual heating: power limits, losses and reactions still determine temperature.

### Controller gain [W/K]

Ключ: `proportional_w_per_k`.

Additional heater command per kelvin of target-minus-sensor error. The command is clipped to nonnegative power and maximum power.

### Maximum heater power [W]

Ключ: `maximum_power_w`.

Upper bound on controller output. Must be positive and no smaller than base power.

### Base heater power [W]

Ключ: `feedforward_w`.

Feedforward contribution to the controller command. Gain zero gives constant base power until cutoff.

### Cutoff sensor temperature

Ключ: `cutoff_temperature_k`.

Temperature threshold for latching this heater off. If a positive rise threshold is set, BOTH conditions are required. No automatic restart after cooling.

### Cutoff minimum rise [C/s]

Ключ: `cutoff_rate_k_per_s`.

Required sensor heating rate together with the cutoff temperature. Zero means temperature-only cutoff.

### Heater contact [W/(m2 K)]

Ключ: `heater_contact_h`.

Reference heater-to-battery heat transfer per area. Editing scales all contact sectors while retaining the existing temperature/damage response.

### Uncontrolled heater power [W]

Ключ: `heater_power_w`.

Constant electrical heater power without a controller. With controller enabled, use base power, gain and maximum instead.

### Plate rupture pressure difference

Ключ: `opening_pressure_pa`.

Pressure DIFFERENCE across the shell plate, displayed in kPa. Damage, temperature and a valve/seam strength factor modify the effective local threshold.

### Opening growth rate [1/s]

Ключ: `area_rate_per_s`.

Controls how rapidly the fracture aperture grows above the effective pressure threshold. Not a gas velocity or reaction rate.

### Fracture energy [J/m2]

Ключ: `fracture_energy_j_per_m2`.

Energy cost per newly opened area, charged to the thermodynamic balance during rupture.

### Valve/seam relative strength [0..1]

Ключ: `pressure_factor`.

Weak-region pressure threshold multiplier relative to the surrounding plate. Smaller values open the weak region more easily.

### Valve radius

Ключ: `radius_m`.

Radius of the circular weak region in the selected shell face. It must fit within that face. UI uses mm.

### Perimeter seam width

Ключ: `seam_width_m`.

Width of the weakened strip around the shell face boundary, displayed in mm.

### Valve center U/V

Ключ: `valve_center`.

Local coordinates on the chosen face, relative to its center. U and V follow the two cyclic axes perpendicular to the face normal.

### Vent destination

Ключ: `gas_destination`.

Existing gas region reached after shell opening; outside_domain means exterior. Selecting a route is not forced ventilation.

### Volumetric heat capacity [J/(m3 K)]

Ключ: `volumetric_heat_capacity_j_m3_k`.

Density times specific heat. Multiplying by wall volume gives its total thermal capacity.

### Case internal dimensions

Ключ: `inside_dimensions_m`.

Free cavity dimensions before subtracting batteries and pads. UI uses mm. Locked for imported explicit gas meshes to avoid silently invalidating their topology.

### Object center

Ключ: `position_m`.

Position in assembly coordinates, displayed in mm. Moving an object does not automatically rewire solid contacts or the electrical network.

### Contents dimensions

Ключ: `dimensions_m`.

Prism width/length/height or cylinder radius/height in the contract. UI displays cylinder DIAMETER. Shell is additional; mass and total heat capacity are not automatically rescaled.

### Orientation quaternion X/Y/Z/W

Ключ: `orientation_xyzw`.

Rotation in assembly coordinates; identity is 0,0,0,1. Not Euler angles. Use a normalized quaternion.

### Initial damage [0..1]

Ключ: `damage`.

Initial accumulated degradation of the selected element, not a requested rupture area.

### Initial state of charge [0..1]

Ключ: `charged_fraction`.

Initial charged fraction override for the selected battery, distinct from thermal damage.

### Ambient temperature

Ключ: `ambient_temperature_k`.

External environment temperature used by case boundary generation. UI uses Celsius.

### Gas-to-case heat transfer [W/(m2 K)]

Ключ: `gas_contact_h_w_m2_k`.

Reference per-area heat transfer from cavity gas to the case wall through the common connectivity model.

### Gas-to-solid heat transfer [W/(m2 K)]

Ключ: `solid_contact_h_w_m2_k`.

Reference per-area gas contact with solid objects for the generated cavity mesh. Not a direct solid-to-solid contact.

### Case-to-environment heat transfer [W/(m2 K)]

Ключ: `ambient_h_w_m2_k`.

External case heat loss coefficient. Zero disables this cooling channel; the case still stores and exchanges heat internally.

### Maximum gas cell size

Ключ: `gas_cell_size_m`.

Requested cavity gas resolution, displayed in mm. The compiler fits cells to geometry; smaller values increase work and require convergence checks.

### Gas passage discharge coefficient

Ключ: `gas_discharge_coefficient`.

Dimensionless multiplier of pressure-driven flow through generated passages. Not gas diffusivity or wall conductivity.

### Initial oxygen mass fraction

Ключ: `initial_oxidizer_mass_fraction`.

Oxygen inventory in the selected cavity, displayed as percent. R3 can exhaust it; it is not an unlimited combustion source.

### Case opening diameter / width / height

Ключ: `opening_size`.

Circle uses diameter; rectangle uses width U and height V. Dimensions in mm. A hole must fit its wall without overlapping another hole.

### Case opening center U/V

Ключ: `opening_center`.

Local coordinates in mm relative to wall center. Top/bottom use X,Y; X walls use Y,Z; Y walls use Z,X.

### First contact surface

Ключ: `port_a`.

ID of the first endpoint port. Must refer to an existing declared surface.

### Second contact surface

Ключ: `port_b`.

ID of the second endpoint port. Changing an ID redirects the heat path; it does not move the surface.

### Radiating exchange area

Ключ: `exchange_area_m2`.

Area contributing to radiative exchange, shown in mm2. Separate from conductive contact area.

### Effective emissivity [0..1]

Ключ: `effective_emissivity`.

Multiplier for radiative exchange. Does not alter conductive or mass-transfer laws.

### Contact area

Ключ: `area_m2`.

Geometric area of the selected surface, displayed in mm2. Keep consistent with its dimensions and thermal discretization.

### Contact rectangle dimensions

Ключ: `rectangle_m`.

Width and height of a rectangular contact patch, displayed in mm.

### Contact center

Ключ: `centroid_m`.

Center of the declared contact patch in contract coordinates, displayed in mm.

### Outward normal

Ключ: `normal`.

Unit direction vector used by directional contact geometry; not a flow velocity.

### Destination node or gas region

Ключ: `to`.

Endpoint identifier: circuit node in Circuit, receiving gas region in vent connections. outside_domain denotes exterior gas.

### Gas mixing diffusivity [m2/s]

Ключ: `mixing_diffusivity_m2_s`.

Diffusive redistribution between connected gas volumes, in addition to pressure-driven directed transport.

### Flow-induced mixing multiplier

Ключ: `mixing_flow_fraction`.

Strength of extra mixing associated with the directed mass flow. Does not create gas or energy.

### Buoyancy mixing multiplier

Ключ: `thermal_stirring_coefficient`.

Strength of temperature-driven stirring in the reduced gas model; not a resolved CFD velocity.

### Buoyancy mixing length

Ключ: `thermal_stirring_length_m`.

Characteristic length in the stirring model, displayed in mm.

### Mixing length limit

Ключ: `mixing_length_cap_m`.

Maximum permitted mixing length, displayed in mm. Used to limit the reduced mixing model's spatial scale.

### Gas dynamic viscosity [Pa s]

Ключ: `dynamic_viscosity_pa_s`.

Viscous resistance parameter in the gas model. It is not a thermal conduction coefficient.

### Gas specific heat Cv [J/(kg K)]

Ключ: `cv_j_per_kg_k`.

Sensible internal energy storage per gas mass per kelvin; composition may contribute additional inventory-dependent properties.

### Gas specific constant [J/(kg K)]

Ключ: `specific_r_j_per_kg_k`.

Specific gas constant used in the equation of state linking mass, free volume, temperature and pressure.

### Cooling surface

Ключ: `port_id`.

Declared port connected to the environmental or coolant thermal boundary.

### Coolant / boundary temperature

Ключ: `temperature_k`.

Constant imposed boundary temperature, displayed in Celsius. An attached measured temperature profile takes precedence.

### Radiating area fraction [0..1]

Ключ: `area_fraction`.

Fraction of the boundary surface exposed to radiation.

### Electrical reference node

Ключ: `reference_node`.

Ground/reference node for the DC circuit potential solve. It is not necessarily the physical enclosure.

### Source electrical node

Ключ: `from`.

Positive/origin node of the oriented branch; combined with To defines circuit topology.

### Battery / element reference

Ключ: `cell_id`.

Identifier of the object assigned to this electrical branch or measurement. Does not guarantee an experimental thermocouple's physical location.

### Battery EMF [V]

Ключ: `emf_v`.

Electromotive force assigned to a battery branch; separate from its internal resistance and the external supply voltage.

### Applied current [A]

Ключ: `current_a`.

Current drive of this source/battery. Positive battery current denotes charging in the supported model.

### Applied voltage [V]

Ключ: `voltage_v`.

Supply voltage. For a series wiring wizard this is the TOTAL voltage, not a voltage per battery.

### Power-on time [s]

Ключ: `start_s`.

Simulation time when this drive becomes active.

### Power-off time [s]

Ключ: `stop_s`.

Simulation time when this drive stops; existing measured profiles are preserved.

### Simulation duration [s]

Ключ: `duration_s`.

End time of the next run, not a fixed 3000-second application limit.

### Maximum physical timestep [s]

Ключ: `max_dt_s`.

Upper bound on the physical integrator step. Not frame rate or replay speed; the solver may need smaller accepted steps.

### Minimum permitted timestep [s]

Ключ: `minimum_dt_s`.

Numerical lower bound required by the configured run. Do not relax it to conceal balance or convergence failures.

### Recording interval [s]

Ключ: `output_interval_s`.

Spacing of emitted observations/history. Separate from internal integration accuracy.

### Integration output interval [s]

Ключ: `step_interval_s`.

Outer integration/observation interval used by this scenario; distinct from internal adaptive physical substeps.

### Explicit thermal safety factor [0..1]

Ключ: `stability_factor`.

Margin used by explicit thermal stability limits. Smaller values are more conservative, usually slower.

### Thermal grid divisions X/Y/Z

Ключ: `divisions`.

Number of thermal zones along each local axis. Shell subdivisions must align with the contents/contact geometry. More zones require a convergence check.

### Gas/thermal coupling interval [s]

Ключ: `coupled_transport_interval_s`.

Physical interval used to couple mass transport and thermal evolution. Not a GUI refresh interval.

### Pressure iteration limit

Ключ: `averaged_pressure_iterations`.

Maximum iterations of the configured averaged pressure solve. Increasing it is not a substitute for checking convergence.

### Transport iteration limit

Ключ: `averaged_payload_iterations`.

Maximum iterations for coupled transported quantities in the averaged network solver.

### Measurement location notes

Ключ: `notes`.

Human-readable provenance and uncertainty of the sensor mapping; not a fitted physical parameter.

### Experimental data artifact

Ключ: `artifact_id`.

Reference to the source dataset/artifact used by this measurement.

### Experimental time column

Ключ: `time_column`.

Name of the time column in the referenced data.

### Experimental value column

Ключ: `value_column`.

Name of the measured quantity column; retain its documented units.

### Experiment time offset [s]

Ключ: `time_offset_s`.

Time alignment of the recorded measurement; does not shift the physical solver's reactions.
